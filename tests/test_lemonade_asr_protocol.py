"""Quick probe: OpenAI Realtime API format for Lemonade ASR WebSocket."""

import asyncio
import base64
import json
import math
import struct
import time

import websockets

WS_URL = "ws://127.0.0.1:9000/realtime?model=Whisper-Tiny"


def generate_pcm_tone(duration_s=2.0, sample_rate=16000, frequency=440.0):
    """Generate PCM 16-bit mono audio."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


async def test_openai_realtime_format():
    pcm_data = generate_pcm_tone(duration_s=2.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        # Read session.created
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        parsed = json.loads(msg)
        print(f"1. Session created: type={parsed['type']}, session_id={parsed['session']['id']}")

        # Try input_audio_buffer.append
        append_msg = {"type": "input_audio_buffer.append", "audio": audio_b64}
        print(f"\n2. Sending input_audio_buffer.append ({len(audio_b64)} chars base64)...")
        await ws.send(json.dumps(append_msg))

        # Wait briefly
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=2)
            print(f"   Response after append: {resp[:200]}")
        except asyncio.TimeoutError:
            print("   No response after append (buffer accepted silently)")

        # Send commit
        commit_msg = {"type": "input_audio_buffer.commit"}
        print("\n3. Sending input_audio_buffer.commit...")
        await ws.send(json.dumps(commit_msg))

        # Collect all responses
        print("\n4. Waiting for transcript responses...")
        for i in range(30):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=8)
                parsed = json.loads(resp)
                msg_type = parsed.get("type", "unknown")
                print(f"\n   [{msg_type}]")
                print(f"   {json.dumps(parsed, indent=2)}")

                if msg_type in ("response.done", "conversation.item.input_audio_transcription.completed"):
                    pass  # keep reading
                if msg_type == "error":
                    break
            except asyncio.TimeoutError:
                print("   No more responses (timeout)")
                break

        await ws.close()
        print("\n5. Connection closed cleanly")


async def test_session_update():
    """Test session.update for vocabulary/language config."""
    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"\nSession created: {msg}")

        # Try session.update
        update_msg = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "model": "Whisper-Tiny",
                    "language": "en",
                    "prompt": "KPG Kooperative Prozessgestaltung",
                },
            },
        }
        print(f"\nSending session.update: {json.dumps(update_msg)}")
        await ws.send(json.dumps(update_msg))

        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=3)
            print(f"Response: {resp}")
        except asyncio.TimeoutError:
            print("No response (accepted silently)")

        await ws.close()


async def test_response_create():
    """Test if response.create triggers ASR processing."""
    pcm_data = generate_pcm_tone(duration_s=1.5)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"\nSession: {msg}")

        # Append audio
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

        # Try response.create instead of commit
        print("\nTrying response.create...")
        await ws.send(json.dumps({"type": "response.create"}))

        for _ in range(20):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5)
                parsed = json.loads(resp)
                print(f"  [{parsed.get('type')}] {json.dumps(parsed)[:300]}")
            except asyncio.TimeoutError:
                print("  No more responses")
                break

        await ws.close()


async def main():
    print("=" * 60)
    print("  Lemonade ASR - OpenAI Realtime API Protocol Test")
    print("=" * 60)

    print("\n--- Test 1: input_audio_buffer.append + commit ---")
    await test_openai_realtime_format()

    print("\n--- Test 2: session.update for config ---")
    await test_session_update()

    print("\n--- Test 3: response.create ---")
    await test_response_create()


if __name__ == "__main__":
    asyncio.run(main())
