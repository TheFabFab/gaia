# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Voice capability detection, status endpoint, and WebSocket ASR streaming.

Provides:
- ``GET /api/voice/status`` — queries Lemonade for available ASR/TTS models.
- ``WS /api/voice/stream`` — WebSocket endpoint that accepts binary PCM audio
  from the browser, forwards it to Lemonade's realtime ASR WebSocket, and
  returns transcript JSON back to the client.
"""

import asyncio
import base64
import json
import logging
import os
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


# ── WebSocket ASR streaming endpoint ────────────────────────────────────


async def _listen_lemonade(
    lemonade_ws: Any,
    client_ws: WebSocket,
    stop_event: asyncio.Event,
) -> None:
    """Forward transcript events from Lemonade ASR to the browser client.

    Runs as a background task, listening for messages from the Lemonade
    realtime WebSocket and forwarding relevant transcript events to the
    browser WebSocket.

    Args:
        lemonade_ws: The open WebSocket connection to Lemonade ASR.
        client_ws: The browser-facing FastAPI WebSocket.
        stop_event: Signals this listener to stop.
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
    except Exception as exc:
        logger.debug("Lemonade listener stopped: %s", exc)


@router.websocket("/api/voice/stream")
async def voice_stream(ws: WebSocket) -> None:
    """WebSocket endpoint for realtime voice ASR streaming.

    Protocol:
    - On connect: sends a ``status`` JSON message with ASR/TTS availability.
    - Client sends ``{"type": "start", "model": "..."}`` to begin ASR.
    - Client sends binary PCM frames (16-bit, 16 kHz, mono).
    - Server forwards audio to Lemonade ASR WebSocket (base64-encoded).
    - Server sends ``{"type": "transcript", "text": "..."}`` on transcription.
    - Client sends ``{"type": "stop"}`` to end ASR session.
    - On disconnect: cleans up all Lemonade WebSocket resources.

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
    stop_event = asyncio.Event()

    async def _cleanup() -> None:
        """Close Lemonade WebSocket and cancel listener task."""
        nonlocal lemonade_ws, listener_task
        stop_event.set()
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
                        listener_task = asyncio.create_task(
                            _listen_lemonade(lemonade_ws, ws, stop_event)
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
                    # Commit audio buffer and wait for transcript before closing
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

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Voice WebSocket error: %s", exc)
    finally:
        await _cleanup()
