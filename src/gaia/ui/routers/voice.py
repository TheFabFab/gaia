# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Voice capability detection, status endpoint, and WebSocket ASR/TTS streaming.

Provides:
- ``GET /api/voice/status`` — queries Lemonade for available ASR/TTS models.
- ``WS /api/voice/stream`` — WebSocket endpoint that accepts binary PCM audio
  from the browser, forwards it to Lemonade's realtime ASR WebSocket,
  returns transcript JSON, calls LLM for a response, and streams TTS
  audio back to the client.
"""

import asyncio
import base64
import json
import logging
import os
import re
from typing import Any

import httpx
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

# ── Lemonade recipe identifiers ─────────────────────────────────────────
_RECIPE_ASR = "whispercpp"
_RECIPE_TTS = "kokoro"


# ── Pydantic response models ────────────────────────────────────────────


class VoiceModelInfo(BaseModel):
    """Minimal model descriptor returned in voice status."""

    name: str
    recipe: str


class AsrStatus(BaseModel):
    """ASR capability summary."""

    available: bool = False
    models: list[VoiceModelInfo] = []
    defaultModel: str | None = None
    websocketPort: int | None = None


class TtsStatus(BaseModel):
    """TTS capability summary."""

    available: bool = False
    models: list[VoiceModelInfo] = []
    defaultModel: str | None = None


class VoiceStatusResponse(BaseModel):
    """Top-level response for ``GET /api/voice/status``."""

    asr: AsrStatus = AsrStatus()
    tts: TtsStatus = TtsStatus()


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_lemonade_voice_base_url() -> str:
    """Return the Lemonade voice server base URL.

    Uses ``LEMONADE_VOICE_BASE_URL`` if set, otherwise falls back to
    ``LEMONADE_BASE_URL`` (stripping the ``/api/v1`` suffix if present
    so we can build both ``/api/v1/health`` and ``/v1/models`` paths).
    """
    voice_url = os.environ.get("LEMONADE_VOICE_BASE_URL")
    if voice_url:
        return voice_url.rstrip("/")

    base = os.environ.get("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
    # Strip trailing /api/v1 so we have the root origin
    for suffix in ("/api/v1", "/api/v1/"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/")


async def _lemonade_get(path: str, *, timeout: float = 5.0) -> httpx.Response:
    """Send an async GET request to the Lemonade voice server.

    Args:
        path: URL path (will be appended to the base URL).
        timeout: Request timeout in seconds.

    Returns:
        The ``httpx.Response`` object.

    Raises:
        httpx.ConnectError: When the server is unreachable.
        httpx.TimeoutException: When the request times out.
    """
    base_url = _get_lemonade_voice_base_url()
    url = f"{base_url}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(url)


async def _lemonade_post(
    path: str, *, json_data: dict[str, Any] | None = None, timeout: float = 30.0
) -> httpx.Response:
    """Send an async POST request to the Lemonade voice server.

    Args:
        path: URL path (will be appended to the base URL).
        json_data: JSON body to include in the request.
        timeout: Request timeout in seconds.

    Returns:
        The ``httpx.Response`` object.
    """
    base_url = _get_lemonade_voice_base_url()
    url = f"{base_url}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(url, json=json_data)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at common boundary punctuation.

    Args:
        text: Input text to split.

    Returns:
        List of non-empty sentence strings.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


# ── Endpoint ─────────────────────────────────────────────────────────────


@router.get("/api/voice/status", response_model=VoiceStatusResponse)
async def voice_status() -> VoiceStatusResponse:
    """Check Lemonade for available ASR and TTS models.

    Queries the Lemonade health and models endpoints, filters by recipe
    to identify ASR (``whispercpp``) and TTS (``kokoro``) models, and
    returns a structured capability summary.

    Returns:
        VoiceStatusResponse with ASR and TTS availability info.
    """
    return await _fetch_voice_status()


# ── Internal helpers for voice status (shared with WebSocket) ────────────


async def _fetch_voice_status() -> VoiceStatusResponse:
    """Fetch current ASR/TTS capability status from Lemonade.

    Shared between the REST endpoint and WebSocket on-connect status
    message so both return identical data.

    Returns:
        VoiceStatusResponse with current availability info.
    """
    try:
        health_resp = await _lemonade_get("/api/v1/health")
        models_resp = await _lemonade_get("/v1/models")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning("Lemonade voice server unreachable: %s", exc)
        return VoiceStatusResponse()

    if health_resp.status_code != 200 or models_resp.status_code != 200:
        return VoiceStatusResponse()

    health_data: dict[str, Any] = health_resp.json()
    models_data: dict[str, Any] = models_resp.json()

    websocket_port: int | None = health_data.get("websocket_port")

    asr_models: list[VoiceModelInfo] = []
    tts_models: list[VoiceModelInfo] = []

    for model in models_data.get("data", []):
        recipe = model.get("recipe", "")
        model_id = model.get("id", "")
        if recipe == _RECIPE_ASR:
            asr_models.append(VoiceModelInfo(name=model_id, recipe=recipe))
        elif recipe == _RECIPE_TTS:
            tts_models.append(VoiceModelInfo(name=model_id, recipe=recipe))

    asr = AsrStatus(
        available=len(asr_models) > 0,
        models=asr_models,
        defaultModel=asr_models[0].name if asr_models else None,
        websocketPort=websocket_port if asr_models else None,
    )
    tts = TtsStatus(
        available=len(tts_models) > 0,
        models=tts_models,
        defaultModel=tts_models[0].name if tts_models else None,
    )
    return VoiceStatusResponse(asr=asr, tts=tts)


# ── LLM + TTS processing ────────────────────────────────────────────────


async def _process_llm_and_tts(
    transcript: str,
    conversation_history: list[dict[str, str]],
    client_ws: WebSocket,
    cancel_event: asyncio.Event,
    db: Any | None,
    session_state: dict[str, Any],
) -> None:
    """Call LLM chat completions with transcript, then stream TTS audio.

    Sends ``response`` JSON with LLM text, followed by ``tts_start``,
    binary PCM audio frames (one per sentence), and ``tts_end``.

    Uses sentence-splitting for pseudo-streaming TTS (DR-003).

    Args:
        transcript: The user's spoken text from ASR.
        conversation_history: Mutable list of chat messages for context.
        client_ws: The browser-facing WebSocket to send messages on.
        cancel_event: Set by interrupt handler to abort TTS streaming.
        db: ChatDatabase instance for persisting messages (may be None).
        session_state: Mutable dict holding ``voice_session_id``.
    """
    conversation_history.append({"role": "user", "content": transcript})

    # Persist user message
    if db:
        try:
            if not session_state.get("voice_session_id"):
                session = db.create_session(
                    title="Voice Conversation", agent_type="voice"
                )
                session_state["voice_session_id"] = session["id"]
            db.add_message(
                session_state["voice_session_id"], "user", transcript
            )
        except Exception as exc:
            logger.warning("Failed to persist voice message: %s", exc)

    # Call LLM
    try:
        response = await _lemonade_post(
            "/v1/chat/completions",
            json_data={"messages": conversation_history},
        )
        if response.status_code != 200:
            try:
                await client_ws.send_json({
                    "type": "error",
                    "message": "LLM request failed",
                })
            except Exception:
                pass
            return
        llm_data = response.json()
        llm_text = llm_data["choices"][0]["message"]["content"]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        try:
            await client_ws.send_json({
                "type": "error",
                "message": f"LLM error: {exc}",
            })
        except Exception:
            pass
        return

    conversation_history.append({"role": "assistant", "content": llm_text})
    await client_ws.send_json({"type": "response", "text": llm_text})

    # Persist assistant message
    if db:
        try:
            db.add_message(
                session_state["voice_session_id"], "assistant", llm_text
            )
        except Exception as exc:
            logger.warning("Failed to persist voice response: %s", exc)

    if cancel_event.is_set():
        return

    # TTS streaming (sentence-by-sentence per DR-003)
    sentences = _split_sentences(llm_text)
    tts_started = False

    try:
        await client_ws.send_json({"type": "tts_start"})
        tts_started = True

        for sentence in sentences:
            if cancel_event.is_set():
                break

            try:
                tts_response = await _lemonade_post(
                    "/api/v1/audio/speech",
                    json_data={
                        "model": "kokoro-v1",
                        "input": sentence.strip(),
                        "voice": "af_heart",
                        "response_format": "pcm",
                        "speed": 1.0,
                    },
                    timeout=60.0,
                )
                if cancel_event.is_set():
                    break
                if tts_response.status_code == 200 and tts_response.content:
                    await client_ws.send_bytes(tts_response.content)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TTS error for sentence: %s", exc)
                try:
                    await client_ws.send_json({
                        "type": "error",
                        "message": f"TTS error: {exc}",
                    })
                except Exception:
                    pass
                break
    except asyncio.CancelledError:
        pass  # Interrupted — finally block handles tts_end
    finally:
        if tts_started:
            try:
                await client_ws.send_json({"type": "tts_end"})
            except Exception:
                pass  # Client may have disconnected


# ── WebSocket ASR streaming endpoint ────────────────────────────────────


async def _listen_lemonade(
    lemonade_ws: Any,
    client_ws: WebSocket,
    stop_event: asyncio.Event,
    transcript_queue: asyncio.Queue[str] | None = None,
) -> None:
    """Forward transcript events from Lemonade ASR to the browser client.

    Runs as a background task, listening for messages from the Lemonade
    realtime WebSocket and forwarding relevant transcript events to the
    browser WebSocket.

    Args:
        lemonade_ws: The open WebSocket connection to Lemonade ASR.
        client_ws: The browser-facing FastAPI WebSocket.
        stop_event: Signals this listener to stop.
        transcript_queue: Optional queue to provide transcripts to the
            stop handler for LLM+TTS processing.
    """
    try:
        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(lemonade_ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "session.created":
                # Send session.update with German language config
                update_msg = {
                    "type": "session.update",
                    "session": {
                        "input_audio_transcription": {
                            "language": "de",
                        }
                    },
                }
                await lemonade_ws.send(json.dumps(update_msg))

            elif msg_type == "conversation.item.input_audio_transcription.completed":
                transcript = msg.get("transcript", "")
                await client_ws.send_json({
                    "type": "transcript",
                    "text": transcript,
                })
                if transcript_queue is not None:
                    await transcript_queue.put(transcript)
    except Exception as exc:
        logger.debug("Lemonade listener stopped: %s", exc)


@router.websocket("/api/voice/stream")
async def voice_stream(ws: WebSocket) -> None:
    """WebSocket endpoint for realtime voice interaction.

    Protocol:
    - On connect: sends a ``status`` JSON message with ASR/TTS availability.
    - Client sends ``{"type": "start", "model": "..."}`` to begin ASR.
    - Client sends binary PCM frames (16-bit, 16 kHz, mono).
    - Server forwards audio to Lemonade ASR WebSocket (base64-encoded).
    - Server sends ``{"type": "transcript", "text": "..."}`` on transcription.
    - Client sends ``{"type": "stop"}`` to end ASR session and trigger LLM+TTS.
    - Server sends ``{"type": "response", "text": "..."}`` with LLM response.
    - Server sends ``{"type": "tts_start"}`` before TTS audio.
    - Server sends binary PCM frames (24 kHz, 16-bit mono) for TTS audio.
    - Server sends ``{"type": "tts_end"}`` after TTS audio completes.
    - Client sends ``{"type": "interrupt"}`` to cancel active TTS streaming.
    - On disconnect: cleans up all resources.

    Args:
        ws: The incoming FastAPI WebSocket connection.
    """
    await ws.accept()

    # Send initial status to client
    status = await _fetch_voice_status()
    await ws.send_json({
        "type": "status",
        "asr": status.asr.model_dump(),
        "tts": status.tts.model_dump(),
    })

    lemonade_ws: Any | None = None
    listener_task: asyncio.Task[None] | None = None
    llm_tts_task: asyncio.Task[None] | None = None
    stop_event = asyncio.Event()
    tts_cancel_event = asyncio.Event()
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    conversation_history: list[dict[str, str]] = []
    session_state: dict[str, Any] = {"voice_session_id": None}

    # Access database for chat history persistence
    db: Any | None = getattr(getattr(ws, "app", None), "state", None)
    if db is not None:
        db = getattr(db, "db", None)

    async def _cleanup() -> None:
        """Close Lemonade WebSocket and cancel background tasks."""
        nonlocal lemonade_ws, listener_task, llm_tts_task
        stop_event.set()
        tts_cancel_event.set()
        if llm_tts_task and not llm_tts_task.done():
            llm_tts_task.cancel()
            try:
                await llm_tts_task
            except (asyncio.CancelledError, Exception):
                pass
            llm_tts_task = None
        if listener_task and not listener_task.done():
            listener_task.cancel()
            try:
                await listener_task
            except (asyncio.CancelledError, Exception):
                pass
            listener_task = None
        if lemonade_ws:
            try:
                await lemonade_ws.close()
            except Exception:
                pass
            lemonade_ws = None

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Binary PCM frame from browser
            if "bytes" in message and message["bytes"] is not None:
                if lemonade_ws:
                    pcm_data = message["bytes"]
                    audio_b64 = base64.b64encode(pcm_data).decode("ascii")
                    append_msg = json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    })
                    try:
                        await lemonade_ws.send(append_msg)
                    except Exception as exc:
                        logger.warning("Failed to forward audio to Lemonade: %s", exc)
                continue

            # JSON text message from browser
            if "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "start":
                    # Open Lemonade ASR WebSocket
                    model = data.get("model", "")
                    ws_port = status.asr.websocketPort or 9000
                    lemonade_url = (
                        f"ws://127.0.0.1:{ws_port}/realtime?model={model}"
                    )

                    try:
                        lemonade_ws = await websockets.connect(lemonade_url)
                        stop_event.clear()
                        tts_cancel_event.clear()
                        listener_task = asyncio.create_task(
                            _listen_lemonade(
                                lemonade_ws, ws, stop_event, transcript_queue
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to connect to Lemonade ASR: %s", exc
                        )
                        await ws.send_json({
                            "type": "error",
                            "message": f"ASR connection failed: {exc}",
                        })

                elif msg_type == "stop":
                    # Commit audio buffer and wait for transcript
                    if lemonade_ws:
                        try:
                            await lemonade_ws.send(
                                json.dumps({"type": "input_audio_buffer.commit"})
                            )
                            # Give listener time to receive the transcript
                            await asyncio.sleep(0.5)
                        except Exception as exc:
                            logger.warning(
                                "Failed to commit audio buffer: %s", exc
                            )
                    await _cleanup()

                    # Start LLM+TTS if a transcript was received
                    try:
                        last_transcript = transcript_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        last_transcript = None

                    if last_transcript:
                        tts_cancel_event.clear()
                        llm_tts_task = asyncio.create_task(
                            _process_llm_and_tts(
                                last_transcript,
                                conversation_history,
                                ws,
                                tts_cancel_event,
                                db,
                                session_state,
                            )
                        )

                elif msg_type == "interrupt":
                    # Cancel active TTS streaming
                    tts_cancel_event.set()
                    if llm_tts_task and not llm_tts_task.done():
                        llm_tts_task.cancel()
                        try:
                            await llm_tts_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        llm_tts_task = None
                    tts_cancel_event.clear()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Voice WebSocket error: %s", exc)
    finally:
        await _cleanup()
