"""
Investigation script: Lemonade ASR WebSocket realtime API contract discovery.

PRD-116 Task 0 — Documents the exact ASR WebSocket protocol for browser integration.

FINDINGS SUMMARY (consolidated from all probe scripts):
- Protocol: OpenAI Realtime API compatible (JSON messages only, NOT binary frames)
- WebSocket host: MUST use 127.0.0.1 (rejects connections on 'localhost')
- URL format: ws://127.0.0.1:{ws_port}/realtime?model={model_name}
- Audio encoding: PCM 16-bit 16kHz mono, base64-encoded in JSON messages
- Session management: session.created → session.update → input_audio_buffer.append/commit
- Barge-in: Fully feasible (~2-4ms reconnect, ~150-250ms full ASR cycle)

Usage:
    cd host/gaia/src
    python tests/test_lemonade_asr_api.py
"""

import asyncio
import base64
import json
import math
import struct
import sys
import time

import httpx
import websockets

LEMONADE_BASE = "http://localhost:8060"
WS_HOST = "127.0.0.1"  # Must use 127.0.0.1, not localhost (Lemonade rejects WS on 'localhost')
WS_PORT = 9000
ASR_MODEL = "Whisper-Tiny"


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def generate_pcm_tone(
    duration_s: float = 2.0, sample_rate: int = 16000, frequency: float = 440.0
) -> bytes:
    """Generate PCM 16-bit mono audio with a sine wave tone."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def generate_pcm_silence(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM 16-bit mono audio."""
    num_samples = int(sample_rate * duration_s)
    return b"\x00\x00" * num_samples


async def investigate_websocket_connection():
    """Test basic WebSocket connection to the Lemonade realtime ASR."""
    separator("1. WebSocket Connection Test")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"
    print(f"Connecting to: {ws_url}")

    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            print("Connected!")

            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            parsed = json.loads(msg)
            print(f"Initial message type: {parsed['type']}")
            print(f"Session ID: {parsed['session']['id']}")
            print(f"Full message: {json.dumps(parsed, indent=2)}")

            await ws.close()
            print("Connection closed cleanly")
            return True

    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")
        return False


async def investigate_host_binding():
    """Test which hosts the WebSocket server accepts."""
    separator("2. Host Binding Investigation")

    hosts_to_test = [
        ("127.0.0.1", "loopback IP"),
        ("localhost", "localhost hostname"),
    ]

    for host, desc in hosts_to_test:
        url = f"ws://{host}:{WS_PORT}/realtime?model={ASR_MODEL}"
        try:
            async with websockets.connect(url, open_timeout=5) as ws:
                await asyncio.wait_for(ws.recv(), timeout=2)
                print(f"  {host} ({desc}): CONNECTED")
                await ws.close()
        except Exception as e:
            print(f"  {host} ({desc}): REJECTED ({type(e).__name__})")


async def investigate_url_variations():
    """Test different WebSocket URL formats."""
    separator("3. WebSocket URL Variations")

    urls = [
        (f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}", "with model"),
        (f"ws://{WS_HOST}:{WS_PORT}/realtime", "without model"),
        (f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}&language=en", "with language"),
        (f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}&prompt=test", "with prompt"),
        (f"ws://{WS_HOST}:{WS_PORT}/v1/realtime?model={ASR_MODEL}", "/v1 prefix"),
        (f"ws://{WS_HOST}:{WS_PORT}/", "root path"),
    ]

    for url, desc in urls:
        try:
            async with websockets.connect(url, open_timeout=5) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
                print(f"  [{desc}]: CONNECTED - {msg[:100]}")
                await ws.close()
        except Exception as e:
            print(f"  [{desc}]: FAILED - {type(e).__name__}")


async def investigate_protocol():
    """Test the full protocol lifecycle using OpenAI Realtime API format."""
    separator("4. Full Protocol Lifecycle")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"
    pcm_data = generate_pcm_tone(duration_s=2.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(ws_url, open_timeout=10) as ws:
        # Step 1: Receive session.created
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        parsed = json.loads(msg)
        print(f"Step 1 - {parsed['type']}:")
        print(f"  {json.dumps(parsed, indent=2)}")

        # Step 2: Send session.update (optional config)
        update = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "language": "en",
                }
            }
        }
        print(f"\nStep 2 - Sending session.update...")
        await ws.send(json.dumps(update))
        resp = await asyncio.wait_for(ws.recv(), timeout=5)
        parsed = json.loads(resp)
        print(f"  Response type: {parsed['type']}")
        print(f"  {json.dumps(parsed, indent=2)}")

        # Step 3: Send audio via input_audio_buffer.append
        append_msg = {"type": "input_audio_buffer.append", "audio": audio_b64}
        print(f"\nStep 3 - Sending input_audio_buffer.append "
              f"({len(audio_b64)} base64 chars, {len(pcm_data)} bytes PCM)...")
        await ws.send(json.dumps(append_msg))
        # This is accepted silently (no response)
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=1)
            print(f"  Unexpected response: {resp[:200]}")
        except asyncio.TimeoutError:
            print("  Accepted silently (no response)")

        # Step 4: Trigger transcription via input_audio_buffer.commit
        print(f"\nStep 4 - Sending input_audio_buffer.commit...")
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Collect all responses
        responses = []
        for _ in range(20):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=8)
                parsed = json.loads(resp)
                responses.append(parsed)
                print(f"  [{parsed['type']}]: {json.dumps(parsed, indent=2)}")
            except asyncio.TimeoutError:
                break

        print(f"\nProtocol summary:")
        print(f"  Message types received: {[r['type'] for r in responses]}")

        await ws.close()


async def investigate_session_update_fields():
    """Test which session.update fields are accepted."""
    separator("5. Session Update Field Support")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"

    configs = [
        ("language only", {"session": {"input_audio_transcription": {"language": "de"}}}),
        ("prompt/vocabulary", {"session": {"input_audio_transcription": {"prompt": "KPG Kooperative"}}}),
        ("model change", {"session": {"input_audio_transcription": {"model": "Whisper-Large-v3-Turbo"}}}),
        ("all fields", {"session": {"input_audio_transcription": {"model": "Whisper-Tiny", "language": "en", "prompt": "test"}}}),
        ("task field", {"session": {"input_audio_transcription": {"task": "transcribe"}}}),
        ("vocab array", {"session": {"input_audio_transcription": {"vocabulary": ["KPG", "test"]}}}),
    ]

    for desc, session_cfg in configs:
        update_msg = {"type": "session.update", **session_cfg}
        try:
            async with websockets.connect(ws_url, open_timeout=5) as ws:
                await asyncio.wait_for(ws.recv(), timeout=3)
                await ws.send(json.dumps(update_msg))
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                parsed = json.loads(resp)
                if parsed.get("type") == "session.updated":
                    print(f"  [{desc}]: ACCEPTED")
                elif parsed.get("type") == "error":
                    print(f"  [{desc}]: REJECTED ({parsed['error']['message']})")
                await ws.close()
        except Exception as e:
            print(f"  [{desc}]: ERROR - {e}")


async def investigate_multi_turn():
    """Test multiple sequential audio buffers in one session."""
    separator("6. Multi-turn (Sequential Commits)")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"
    pcm_data = generate_pcm_tone(duration_s=1.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(ws_url, open_timeout=10) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created

        for turn in range(3):
            start = time.perf_counter()
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            transcript = ""
            for _ in range(10):
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=8)
                    parsed = json.loads(resp)
                    if parsed["type"] == "conversation.item.input_audio_transcription.completed":
                        transcript = parsed.get("transcript", "").strip()
                        break
                except asyncio.TimeoutError:
                    break

            elapsed = time.perf_counter() - start
            print(f"  Turn {turn+1}: '{transcript}' ({elapsed:.3f}s)")

        await ws.close()


async def investigate_buffer_clear():
    """Test input_audio_buffer.clear message."""
    separator("7. Buffer Clear")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"
    pcm_data = generate_pcm_tone(duration_s=1.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(ws_url, open_timeout=10) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created

        # Append audio then clear
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=3)
            parsed = json.loads(resp)
            print(f"  Clear response: {parsed['type']}")
        except asyncio.TimeoutError:
            print("  No response (silent)")

        # Commit on empty buffer
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=3)
            parsed = json.loads(resp)
            print(f"  Commit on empty: {parsed.get('type', 'unknown')}")
        except asyncio.TimeoutError:
            print("  Commit on empty: no response (nothing to transcribe)")

        await ws.close()


async def investigate_reconnection():
    """Test rapid disconnect/reconnect (barge-in feasibility)."""
    separator("8. Rapid Reconnection (Barge-in)")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"
    pcm_data = generate_pcm_tone(duration_s=0.5)
    audio_b64 = base64.b64encode(pcm_data).decode()

    times = []
    for i in range(5):
        start = time.perf_counter()
        async with websockets.connect(ws_url, open_timeout=5) as ws:
            await asyncio.wait_for(ws.recv(), timeout=3)
            connect_time = time.perf_counter() - start

            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            for _ in range(5):
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=5)
                    parsed = json.loads(resp)
                    if parsed["type"] == "conversation.item.input_audio_transcription.completed":
                        break
                except asyncio.TimeoutError:
                    break

            total = time.perf_counter() - start
            times.append(total)
            print(f"  Cycle {i+1}: connect={connect_time:.3f}s, total={total:.3f}s")
            await ws.close()

        await asyncio.sleep(0.05)

    if times:
        print(f"\n  Average: {sum(times)/len(times):.3f}s, "
              f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")
        print("  Rapid reconnection works cleanly for barge-in")


async def investigate_unsupported_messages():
    """Test which message types are NOT supported."""
    separator("9. Unsupported Message Types")

    ws_url = f"ws://{WS_HOST}:{WS_PORT}/realtime?model={ASR_MODEL}"

    unsupported = [
        ("response.create", {"type": "response.create"}),
        ("response.cancel", {"type": "response.cancel"}),
        ("conversation.item.create", {"type": "conversation.item.create", "item": {"type": "message"}}),
    ]

    for desc, msg in unsupported:
        try:
            async with websockets.connect(ws_url, open_timeout=5) as ws:
                await asyncio.wait_for(ws.recv(), timeout=3)
                await ws.send(json.dumps(msg))
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                parsed = json.loads(resp)
                if parsed.get("type") == "error":
                    print(f"  [{desc}]: NOT SUPPORTED ({parsed['error']['message']})")
                else:
                    print(f"  [{desc}]: Responded with {parsed['type']}")
                await ws.close()
        except Exception as e:
            print(f"  [{desc}]: ERROR - {e}")


async def main():
    print("=" * 60)
    print("  Lemonade ASR WebSocket API Investigation")
    print(f"  WebSocket: ws://{WS_HOST}:{WS_PORT}/realtime")
    print(f"  Model: {ASR_MODEL}")
    print("=" * 60)

    # Check server health
    try:
        health = httpx.get(f"{LEMONADE_BASE}/api/v1/health", timeout=5)
        health_data = health.json()
        print(f"\nServer status: {health_data['status']}, version: {health_data['version']}")
        print(f"WebSocket port: {health_data.get('websocket_port')}")
        asr_models = [
            m for m in health_data.get("all_models_loaded", []) if m["type"] == "audio"
        ]
        if not asr_models:
            print("ERROR: No ASR model loaded!")
            sys.exit(1)
        print(f"ASR model: {asr_models[0]['model_name']} "
              f"(recipe: {asr_models[0]['recipe']})")
    except Exception as e:
        print(f"ERROR: Lemonade server not reachable: {e}")
        sys.exit(1)

    await investigate_websocket_connection()
    await investigate_host_binding()
    await investigate_url_variations()
    await investigate_protocol()
    await investigate_session_update_fields()
    await investigate_multi_turn()
    await investigate_buffer_clear()
    await investigate_reconnection()
    await investigate_unsupported_messages()

    separator("INVESTIGATION COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
