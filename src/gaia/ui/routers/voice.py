# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Voice capability detection and status endpoint.

Provides ``GET /api/voice/status`` which queries the Lemonade server for
available ASR (speech-to-text) and TTS (text-to-speech) models and returns
a structured capability summary for the frontend.
"""

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter
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
    try:
        health_resp = await _lemonade_get("/api/v1/health")
        models_resp = await _lemonade_get("/v1/models")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning("Lemonade voice server unreachable: %s", exc)
        return VoiceStatusResponse()

    if health_resp.status_code != 200 or models_resp.status_code != 200:
        logger.warning(
            "Lemonade voice server returned non-200: health=%d models=%d",
            health_resp.status_code,
            models_resp.status_code,
        )
        return VoiceStatusResponse()

    health_data: dict[str, Any] = health_resp.json()
    models_data: dict[str, Any] = models_resp.json()

    websocket_port: int | None = health_data.get("websocket_port")

    # Build model lists filtered by recipe
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
