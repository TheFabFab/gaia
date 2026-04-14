# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Component tests for WS /api/voice/stream endpoint.

PRD-116 Tasks 2 & 3 — Verifies WebSocket ASR integration with Lemonade
realtime, LLM chat completions, and TTS audio streaming.

Tests mock the Lemonade WebSocket and HTTP calls so they run without
a live Lemonade instance.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

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
        {"name": "kokoro-v1", "recipe": "kokoro"},
    ],
}

LEMONADE_MODELS_RESPONSE = {
    "data": [
        {"id": "Whisper-Tiny-en", "recipe": "whispercpp", "labels": ["audio"]},
        {"id": "kokoro-v1", "recipe": "kokoro", "labels": ["tts"]},
    ]
}


def _mock_httpx_response(status_code: int, json_data: dict) -> httpx.Response:
    """Build a fake httpx.Response with the given status and JSON body."""
    return httpx.Response(status_code=status_code, json=json_data)


def _make_lemonade_get_mock(health=None, models=None):
    """Create a mock for _lemonade_get returning configured responses."""
    _health = health or LEMONADE_HEALTH_RESPONSE
    _models = models or LEMONADE_MODELS_RESPONSE

    async def fake_get(path: str, **kwargs):
        if "health" in path:
            return _mock_httpx_response(200, _health)
        if "models" in path:
            return _mock_httpx_response(200, _models)
        raise ValueError(f"Unexpected path: {path}")

    return fake_get


class FakeLemonadeWs:
    """Simulates a Lemonade ASR WebSocket for testing.

    Tracks sent messages and optionally queues scripted responses.
    """

    def __init__(self, responses=None):
        self.sent: list[str] = []
        self._responses: list[str] = list(responses or [])
        self._recv_index = 0
        self.closed = False

    async def send(self, data: str) -> None:
        """Record a sent message."""
        self.sent.append(data)

    async def recv(self) -> str:
        """Return the next scripted response, or block forever."""
        if self._recv_index < len(self._responses):
            msg = self._responses[self._recv_index]
            self._recv_index += 1
            return msg
        # Block until closed to simulate idle WebSocket
        while not self.closed:
            await asyncio.sleep(0.05)
        raise Exception("connection closed")

    async def close(self) -> None:
        """Mark WebSocket as closed."""
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# ── Tests ────────────────────────────────────────────────────────────────


class TestVoiceWebSocketConnect:
    """Tests for WebSocket connection and status message."""

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_ws_connect_returns_status_message(self, mock_get, client):
        """WebSocket connection returns status message with ASR/TTS availability."""
        mock_get.side_effect = _make_lemonade_get_mock()

        with client.websocket_connect("/api/voice/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "status"
            assert msg["asr"]["available"] is True
            assert msg["asr"]["websocketPort"] == 9000
            assert msg["tts"]["available"] is True

    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_ws_connect_returns_unavailable_when_lemonade_down(
        self, mock_get, client
    ):
        """WebSocket status shows unavailable when Lemonade is unreachable."""

        async def fail_get(path: str, **kwargs):
            raise httpx.ConnectError("Connection refused")

        mock_get.side_effect = fail_get

        with client.websocket_connect("/api/voice/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "status"
            assert msg["asr"]["available"] is False
            assert msg["tts"]["available"] is False


class TestVoiceWebSocketAsr:
    """Tests for ASR binary frame forwarding and transcript reception."""

    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_binary_pcm_frames_trigger_asr(self, mock_get, mock_ws_lib, client):
        """Binary PCM frames are base64-encoded and forwarded to Lemonade ASR."""
        mock_get.side_effect = _make_lemonade_get_mock()

        # Prepare a Lemonade WebSocket that sends session.created then
        # waits until closed.
        fake_ws = FakeLemonadeWs(responses=[
            json.dumps({"type": "session.created", "session": {"id": "s1"}}),
        ])

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)

        pcm_data = b"\x00\x01" * 160  # 320 bytes = 10ms of 16kHz 16-bit mono

        with client.websocket_connect("/api/voice/stream") as ws:
            # Receive initial status
            ws.receive_json()

            # Send start message
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})

            # Give server time to connect to Lemonade
            import time
            time.sleep(0.2)

            # Send binary PCM frame
            ws.send_bytes(pcm_data)

            # Give server time to forward
            time.sleep(0.2)

        # Verify the data was forwarded as base64-encoded JSON
        append_msgs = [
            json.loads(m) for m in fake_ws.sent
            if "input_audio_buffer.append" in m
        ]
        assert len(append_msgs) >= 1
        # Verify the audio was base64-encoded correctly
        decoded = base64.b64decode(append_msgs[0]["audio"])
        assert decoded == pcm_data

    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_transcript_sent_on_lemonade_commit(
        self, mock_get, mock_ws_lib, client
    ):
        """Transcript message sent when Lemonade returns transcription event."""
        mock_get.side_effect = _make_lemonade_get_mock()

        transcript_text = "Hallo, wie geht es Ihnen?"
        fake_ws = FakeLemonadeWs(responses=[
            json.dumps({"type": "session.created", "session": {"id": "s1"}}),
            json.dumps({"type": "input_audio_buffer.committed"}),
            json.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": transcript_text,
            }),
        ])

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)

        with client.websocket_connect("/api/voice/stream") as ws:
            # Receive status
            ws.receive_json()

            # Start ASR
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)

            # Send some audio and then stop
            ws.send_bytes(b"\x00\x01" * 160)
            time.sleep(0.1)

            ws.send_json({"type": "stop"})

            # Should receive transcript
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "transcript"
            assert msg["text"] == transcript_text

    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_stop_closes_asr_websocket(self, mock_get, mock_ws_lib, client):
        """Stop message from client closes the Lemonade ASR WebSocket cleanly."""
        mock_get.side_effect = _make_lemonade_get_mock()

        fake_ws = FakeLemonadeWs(responses=[
            json.dumps({"type": "session.created", "session": {"id": "s1"}}),
            json.dumps({"type": "input_audio_buffer.committed"}),
            json.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "test",
            }),
        ])

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)

            ws.send_json({"type": "stop"})
            time.sleep(0.2)

            # Receive transcript (may come before close completes)
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "transcript"

        # After client disconnects, Lemonade WS should be closed
        assert fake_ws.closed is True

        # Verify commit was sent
        commit_msgs = [m for m in fake_ws.sent if "commit" in m]
        assert len(commit_msgs) >= 1

    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_error_when_lemonade_asr_unavailable(
        self, mock_get, mock_ws_lib, client
    ):
        """Error message sent when Lemonade ASR WebSocket connection fails."""
        mock_get.side_effect = _make_lemonade_get_mock()

        async def fail_connect(url, **kwargs):
            raise ConnectionRefusedError("Connection refused")

        mock_ws_lib.connect = AsyncMock(side_effect=fail_connect)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})

            # Should receive an error message
            import time
            time.sleep(0.3)
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "error"
            assert "asr" in msg.get("message", "").lower() or "connect" in msg.get("message", "").lower()

    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_client_disconnect_cleans_up_resources(
        self, mock_get, mock_ws_lib, client
    ):
        """Client disconnect during ASR does not leak resources."""
        mock_get.side_effect = _make_lemonade_get_mock()

        fake_ws = FakeLemonadeWs(responses=[
            json.dumps({"type": "session.created", "session": {"id": "s1"}}),
        ])

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            # Client disconnects without sending stop

        # Give cleanup time to run
        import time
        time.sleep(0.3)

        # Lemonade WS should be cleaned up
        assert fake_ws.closed is True


# ── Task 3: LLM + TTS streaming tests ───────────────────────────────────


def _make_lemonade_post_mock(
    llm_text: str = "Das ist eine Antwort.",
    tts_audio: bytes = b"\x00\x01" * 100,
    llm_fail: bool = False,
    tts_fail: bool = False,
):
    """Create a mock for _lemonade_post returning configured LLM/TTS responses."""

    async def fake_post(path: str, **kwargs):
        if "chat/completions" in path:
            if llm_fail:
                raise httpx.ConnectError("LLM unreachable")
            return _mock_httpx_response(200, {
                "choices": [{"message": {"content": llm_text}}],
            })
        if "audio/speech" in path:
            if tts_fail:
                raise httpx.ConnectError("TTS unreachable")
            return httpx.Response(200, content=tts_audio)
        raise ValueError(f"Unexpected POST path: {path}")

    return fake_post


def _make_asr_transcript_ws(transcript: str = "Hallo Welt"):
    """Create a FakeLemonadeWs that returns session.created then transcript."""
    return FakeLemonadeWs(responses=[
        json.dumps({"type": "session.created", "session": {"id": "s1"}}),
        json.dumps({"type": "input_audio_buffer.committed"}),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": transcript,
        }),
    ])


class TestVoiceWebSocketLlmTts:
    """Tests for LLM response and TTS audio streaming after transcript (Task 3)."""

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_llm_response_sent_after_transcript(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """LLM response is sent after transcript is complete."""
        mock_get.side_effect = _make_lemonade_get_mock()

        transcript = "Was ist KPG?"
        llm_text = "KPG steht für Kooperative Prozessgestaltung."
        fake_ws = _make_asr_transcript_ws(transcript)

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)
        mock_post.side_effect = _make_lemonade_post_mock(llm_text=llm_text)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            time.sleep(0.1)
            ws.send_json({"type": "stop"})

            # Should receive transcript
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "transcript"
            assert msg["text"] == transcript

            # Should receive LLM response
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "response"
            assert msg["text"] == llm_text

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_tts_audio_frames_sent_after_llm_response(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """TTS audio binary frames are sent after LLM response."""
        mock_get.side_effect = _make_lemonade_get_mock()

        tts_audio = b"\xAB\xCD" * 200
        fake_ws = _make_asr_transcript_ws("Test")

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)
        mock_post.side_effect = _make_lemonade_post_mock(
            llm_text="Kurze Antwort.", tts_audio=tts_audio
        )

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            ws.send_json({"type": "stop"})

            ws.receive_json(mode="text")  # transcript
            ws.receive_json(mode="text")  # response

            # Should receive tts_start, binary audio, tts_end
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "tts_start"

            audio_data = ws.receive_bytes()
            assert len(audio_data) > 0

            msg = ws.receive_json(mode="text")
            assert msg["type"] == "tts_end"

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_tts_start_and_end_bracket_audio(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """tts_start and tts_end control messages bracket TTS audio."""
        mock_get.side_effect = _make_lemonade_get_mock()

        tts_audio = b"\x01\x02" * 50
        fake_ws = _make_asr_transcript_ws("Hallo")

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)
        mock_post.side_effect = _make_lemonade_post_mock(
            llm_text="Antwort.", tts_audio=tts_audio
        )

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            ws.send_json({"type": "stop"})

            ws.receive_json(mode="text")  # transcript
            ws.receive_json(mode="text")  # response

            # First message must be tts_start
            msg = ws.receive_json(mode="text")
            assert msg == {"type": "tts_start"}

            # Then at least one binary audio frame
            audio = ws.receive_bytes()
            assert len(audio) > 0

            # Then tts_end
            msg = ws.receive_json(mode="text")
            assert msg == {"type": "tts_end"}

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_interrupt_stops_tts_streaming(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """Interrupt message stops TTS streaming immediately."""
        mock_get.side_effect = _make_lemonade_get_mock()

        # Use multi-sentence text and slow TTS to give time for interrupt
        llm_text = "Erste Antwort. Zweite Antwort. Dritte Antwort. Vierte Antwort."
        tts_audio = b"\x00\x01" * 500

        # Make TTS slightly slow so interrupt can fire between sentences
        call_count = 0

        async def slow_post(path: str, **kwargs):
            nonlocal call_count
            if "chat/completions" in path:
                return _mock_httpx_response(200, {
                    "choices": [{"message": {"content": llm_text}}],
                })
            if "audio/speech" in path:
                call_count += 1
                await asyncio.sleep(0.1)  # Simulate TTS latency
                return httpx.Response(200, content=tts_audio)
            raise ValueError(f"Unexpected POST path: {path}")

        mock_post.side_effect = slow_post

        fake_ws = _make_asr_transcript_ws("Test Interrupt")

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            ws.send_json({"type": "stop"})

            ws.receive_json(mode="text")  # transcript
            ws.receive_json(mode="text")  # response
            ws.receive_json(mode="text")  # tts_start

            # Receive first audio chunk then send interrupt
            ws.receive_bytes()
            ws.send_json({"type": "interrupt"})

            # Should receive tts_end (interrupt acknowledged)
            time.sleep(0.3)
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "tts_end"

        # Should NOT have processed all 4 sentences
        assert call_count < 4

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_error_on_tts_failure(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """Error message sent when Lemonade TTS request fails."""
        mock_get.side_effect = _make_lemonade_get_mock()

        fake_ws = _make_asr_transcript_ws("Test TTS Fehler")

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)
        mock_post.side_effect = _make_lemonade_post_mock(
            llm_text="Antwort.", tts_fail=True
        )

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            ws.send_json({"type": "stop"})

            ws.receive_json(mode="text")  # transcript
            ws.receive_json(mode="text")  # response
            ws.receive_json(mode="text")  # tts_start

            # Should receive error about TTS failure
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "error"
            assert "tts" in msg.get("message", "").lower()

            # Should still get tts_end
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "tts_end"

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_error_on_llm_failure(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """Error message sent when Lemonade LLM request fails."""
        mock_get.side_effect = _make_lemonade_get_mock()

        fake_ws = _make_asr_transcript_ws("Test LLM Fehler")

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)
        mock_post.side_effect = _make_lemonade_post_mock(llm_fail=True)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            ws.send_json({"type": "stop"})

            ws.receive_json(mode="text")  # transcript

            # Should receive error about LLM failure
            msg = ws.receive_json(mode="text")
            assert msg["type"] == "error"
            assert "llm" in msg.get("message", "").lower()

    @patch("gaia.ui.routers.voice._lemonade_post")
    @patch("gaia.ui.routers.voice.websockets")
    @patch("gaia.ui.routers.voice._lemonade_get")
    def test_client_disconnect_during_tts_cleans_up(
        self, mock_get, mock_ws_lib, mock_post, client
    ):
        """Client disconnect during TTS does not leak resources."""
        mock_get.side_effect = _make_lemonade_get_mock()

        llm_text = "Erste. Zweite. Dritte. Vierte. Fünfte."
        tts_audio = b"\x00\x01" * 500

        async def slow_post(path: str, **kwargs):
            if "chat/completions" in path:
                return _mock_httpx_response(200, {
                    "choices": [{"message": {"content": llm_text}}],
                })
            if "audio/speech" in path:
                await asyncio.sleep(0.2)  # Slow TTS
                return httpx.Response(200, content=tts_audio)
            raise ValueError(f"Unexpected POST path: {path}")

        mock_post.side_effect = slow_post

        fake_ws = _make_asr_transcript_ws("Test Disconnect")

        async def fake_connect(url, **kwargs):
            return fake_ws

        mock_ws_lib.connect = AsyncMock(side_effect=fake_connect)

        with client.websocket_connect("/api/voice/stream") as ws:
            ws.receive_json()  # status
            ws.send_json({"type": "start", "model": "Whisper-Tiny-en"})
            import time
            time.sleep(0.2)
            ws.send_bytes(b"\x00\x01" * 160)
            ws.send_json({"type": "stop"})

            ws.receive_json(mode="text")  # transcript
            ws.receive_json(mode="text")  # response
            # Client disconnects during TTS (before audio completes)

        # Give async cleanup time
        import time
        time.sleep(0.5)

        # Resources should be cleaned up
        assert fake_ws.closed is True
