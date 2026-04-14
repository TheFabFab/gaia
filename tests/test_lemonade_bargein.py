"""Probe: Lemonade ASR barge-in and reconnection feasibility."""

import asyncio
import base64
import json
import math
import struct
import time

import httpx
import websockets

WS_URL = "ws://127.0.0.1:9000/realtime?model=Whisper-Tiny"
TTS_ENDPOINT = "http://localhost:8060/api/v1/audio/speech"
LEMONADE_HEALTH = "http://localhost:8060/api/v1/health"


def generate_pcm_tone(duration_s=1.0, sample_rate=16000, frequency=440.0):
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


async def test_rapid_reconnection():
    """Test rapid disconnect/reconnect cycles (barge-in simulation)."""
    print("=" * 60)
    print("  Test 1: Rapid reconnection (barge-in)")
    print("=" * 60)

    pcm_data = generate_pcm_tone(duration_s=0.5)
    audio_b64 = base64.b64encode(pcm_data).decode()

    times = []
    for i in range(5):
        start = time.perf_counter()
        try:
            async with websockets.connect(WS_URL, open_timeout=5) as ws:
                created = await asyncio.wait_for(ws.recv(), timeout=3)
                connect_time = time.perf_counter() - start

                # Send audio and commit
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                # Wait for transcript
                transcript = ""
                for _ in range(5):
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=5)
                        parsed = json.loads(resp)
                        if parsed.get("type") == "conversation.item.input_audio_transcription.completed":
                            transcript = parsed.get("transcript", "").strip()
                            break
                    except asyncio.TimeoutError:
                        break

                total_time = time.perf_counter() - start
                times.append(total_time)
                print(f"  Cycle {i+1}: connect={connect_time:.3f}s, total={total_time:.3f}s, transcript='{transcript}'")

                # Abrupt close (simulating barge-in)
                await ws.close()

        except Exception as e:
            print(f"  Cycle {i+1}: ERROR - {e}")

        # Minimal delay between reconnects
        await asyncio.sleep(0.05)

    if times:
        print(f"\n  Average cycle time: {sum(times)/len(times):.3f}s")
        print(f"  Min/Max: {min(times):.3f}s / {max(times):.3f}s")


async def test_concurrent_tts_and_asr():
    """Test if ASR WebSocket works while TTS HTTP is streaming."""
    print("\n" + "=" * 60)
    print("  Test 2: Concurrent TTS + ASR (barge-in scenario)")
    print("=" * 60)

    pcm_data = generate_pcm_tone(duration_s=1.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async def run_tts():
        """Start a long TTS request."""
        long_text = (
            "This is a long sentence to test concurrent TTS and ASR processing. "
            "We want to verify that the ASR WebSocket can accept and process audio "
            "while TTS is simultaneously generating speech output."
        )
        payload = {"model": "kokoro-v1", "input": long_text, "voice": "af_heart"}

        start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", TTS_ENDPOINT, json=payload, timeout=60) as resp:
                total_bytes = 0
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    total_bytes += len(chunk)
                elapsed = time.perf_counter() - start
                print(f"  TTS completed: {total_bytes} bytes in {elapsed:.3f}s")

    async def run_asr():
        """Run ASR while TTS is streaming."""
        await asyncio.sleep(0.5)  # Let TTS start first

        start = time.perf_counter()
        async with websockets.connect(WS_URL, open_timeout=10) as ws:
            # session.created
            await asyncio.wait_for(ws.recv(), timeout=5)
            connect_time = time.perf_counter() - start

            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            transcript = ""
            for _ in range(10):
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=8)
                    parsed = json.loads(resp)
                    if parsed.get("type") == "conversation.item.input_audio_transcription.completed":
                        transcript = parsed.get("transcript", "").strip()
                        break
                except asyncio.TimeoutError:
                    break

            total_time = time.perf_counter() - start
            print(f"  ASR completed: connect={connect_time:.3f}s, total={total_time:.3f}s, transcript='{transcript}'")
            await ws.close()

    print("  Starting concurrent TTS + ASR...")
    await asyncio.gather(run_tts(), run_asr())
    print("  Both completed successfully (barge-in is feasible)")


async def test_tts_abort_during_stream():
    """Test aborting TTS mid-stream and immediately starting ASR."""
    print("\n" + "=" * 60)
    print("  Test 3: TTS abort + immediate ASR start")
    print("=" * 60)

    pcm_data = generate_pcm_tone(duration_s=1.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    long_text = (
        "This is a very long text for TTS generation that we will abort midway. "
        * 5
    )

    # Start TTS
    print("  Starting TTS...")
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", TTS_ENDPOINT,
            json={"model": "kokoro-v1", "input": long_text, "voice": "af_heart"},
            timeout=60,
        ) as resp:
            partial = 0
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                partial += len(chunk)
                if partial > 8192:
                    break  # Abort TTS
    abort_time = time.perf_counter() - start
    print(f"  TTS aborted after {partial} bytes in {abort_time:.3f}s")

    # Immediately start ASR
    print("  Starting ASR immediately after TTS abort...")
    asr_start = time.perf_counter()
    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)
        connect_time = time.perf_counter() - asr_start

        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        transcript = ""
        for _ in range(10):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=8)
                parsed = json.loads(resp)
                if parsed.get("type") == "conversation.item.input_audio_transcription.completed":
                    transcript = parsed.get("transcript", "").strip()
                    break
            except asyncio.TimeoutError:
                break

        total = time.perf_counter() - asr_start
        print(f"  ASR after abort: connect={connect_time:.3f}s, total={total:.3f}s, transcript='{transcript}'")
        await ws.close()

    # Verify server health
    health = httpx.get(LEMONADE_HEALTH, timeout=5)
    print(f"  Server health after all tests: {health.json()['status']}")


async def main():
    await test_rapid_reconnection()
    await test_concurrent_tts_and_asr()
    await test_tts_abort_during_stream()


if __name__ == "__main__":
    asyncio.run(main())
