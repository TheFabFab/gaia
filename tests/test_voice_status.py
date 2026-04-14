# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Component tests for GET /api/voice/status endpoint.

PRD-116 Task 1 — Verifies Lemonade capability detection for ASR and TTS.

Tests mock external HTTP calls to the Lemonade server so they run without
a live Lemonade instance.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from gaia.ui.server import create_app


@pytest.fixture()
def app():
    """Create a minimal GAIA app for testing (in-memory DB, no frontend)."""
    return create_app(db_path=":memory:", webui_dist="/nonexistent")


@pytest.fixture()
def client(app):
    """Provide a synchronous test client for the GAIA app."""
    return TestClient(app, raise_server_exceptions=False)


# ── Lemonade response fixtures ──────────────────────────────────────────


LEMONADE_HEALTH_RESPONSE = {
    "status": "ok",
    "websocket_port": 9000,
    "models": [
        {"name": "Whisper-Tiny-en", "recipe": "whispercpp"},
        {"name": "Qwen3.5-35B-A3.6B", "recipe": "llamacpp"},
        {"name": "kokoro-v1", "recipe": "kokoro"},
    ],
}

LEMONADE_MODELS_RESPONSE = {
    "data": [
        {
            "id": "Whisper-Tiny-en",
            "recipe": "whispercpp",
            "labels": ["audio"],
        },
        {
            "id": "Qwen3.5-35B-A3.6B",
            "recipe": "llamacpp",
            "labels": ["llm"],
        },
        {
            "id": "kokoro-v1",
            "recipe": "kokoro",
            "labels": ["tts"],
        },
    ]
}


def _mock_httpx_response(status_code: int, json_data: dict) -> httpx.Response:
    """Build a fake httpx.Response with the given status and JSON body."""
    resp = httpx.Response(status_code=status_code, json=json_data)
    return resp


# ── Tests ────────────────────────────────────────────────────────────────


class TestVoiceStatusEndpoint:
    """Component tests for GET /api/voice/status."""

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_returns_available_asr_models(self, mock_get, client):
        """Returns available ASR models from Lemonade model list."""

        async def fake_get(path: str, **kwargs):
            if "health" in path:
                return _mock_httpx_response(200, LEMONADE_HEALTH_RESPONSE)
            if "models" in path:
                return _mock_httpx_response(200, LEMONADE_MODELS_RESPONSE)
            raise ValueError(f"Unexpected path: {path}")

        mock_get.side_effect = fake_get

        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["asr"]["available"] is True
        assert len(data["asr"]["models"]) == 1
        assert data["asr"]["models"][0]["name"] == "Whisper-Tiny-en"
        assert data["asr"]["models"][0]["recipe"] == "whispercpp"
        assert data["asr"]["websocketPort"] == 9000

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_returns_asr_unavailable_when_lemonade_unreachable(
        self, mock_get, client
    ):
        """Returns asr.available: false when Lemonade is unreachable."""

        async def fake_get(path: str, **kwargs):
            raise httpx.ConnectError("Connection refused")

        mock_get.side_effect = fake_get

        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["asr"]["available"] is False
        assert data["asr"]["models"] == []
        assert data["tts"]["available"] is False
        assert data["tts"]["models"] == []

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_returns_tts_unavailable_when_no_tts_model(self, mock_get, client):
        """Returns tts.available: false when Lemonade has no TTS model."""
        health_no_tts = {
            "status": "ok",
            "websocket_port": 9000,
            "models": [
                {"name": "Whisper-Tiny-en", "recipe": "whispercpp"},
                {"name": "Qwen3.5-35B-A3.6B", "recipe": "llamacpp"},
            ],
        }
        models_no_tts = {
            "data": [
                {"id": "Whisper-Tiny-en", "recipe": "whispercpp", "labels": ["audio"]},
                {"id": "Qwen3.5-35B-A3.6B", "recipe": "llamacpp", "labels": ["llm"]},
            ]
        }

        async def fake_get(path: str, **kwargs):
            if "health" in path:
                return _mock_httpx_response(200, health_no_tts)
            if "models" in path:
                return _mock_httpx_response(200, models_no_tts)
            raise ValueError(f"Unexpected path: {path}")

        mock_get.side_effect = fake_get

        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["asr"]["available"] is True
        assert data["tts"]["available"] is False
        assert data["tts"]["models"] == []

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_returns_default_whisper_model(self, mock_get, client):
        """Returns default Whisper model selection."""

        async def fake_get(path: str, **kwargs):
            if "health" in path:
                return _mock_httpx_response(200, LEMONADE_HEALTH_RESPONSE)
            if "models" in path:
                return _mock_httpx_response(200, LEMONADE_MODELS_RESPONSE)
            raise ValueError(f"Unexpected path: {path}")

        mock_get.side_effect = fake_get

        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["asr"]["defaultModel"] == "Whisper-Tiny-en"

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_returns_default_tts_model(self, mock_get, client):
        """Returns default TTS model selection."""

        async def fake_get(path: str, **kwargs):
            if "health" in path:
                return _mock_httpx_response(200, LEMONADE_HEALTH_RESPONSE)
            if "models" in path:
                return _mock_httpx_response(200, LEMONADE_MODELS_RESPONSE)
            raise ValueError(f"Unexpected path: {path}")

        mock_get.side_effect = fake_get

        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.json()

        assert data["tts"]["available"] is True
        assert data["tts"]["defaultModel"] == "kokoro-v1"
        assert len(data["tts"]["models"]) == 1
        assert data["tts"]["models"][0]["name"] == "kokoro-v1"
