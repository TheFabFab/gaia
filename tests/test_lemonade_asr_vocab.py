"""Probe: Lemonade ASR session.update fields and vocabulary support."""

import asyncio
import base64
import json
import math
import struct

import websockets

WS_URL = "ws://127.0.0.1:9000/realtime?model=Whisper-Tiny"


def generate_pcm_tone(duration_s=1.0, sample_rate=16000, frequency=440.0):
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


async def test_session_update_fields():
    """Probe which session.update fields are accepted."""
    print("=" * 60)
    print("  Testing session.update field acceptance")
    print("=" * 60)

    configs = [
        # Test language
        {
            "session": {
                "input_audio_transcription": {
                    "language": "de",
                }
            }
        },
        # Test prompt (vocabulary boosting)
        {
            "session": {
                "input_audio_transcription": {
                    "prompt": "Kooperative Prozessgestaltung Sozialpädagogik",
                }
            }
        },
        # Test model switch
        {
            "session": {
                "input_audio_transcription": {
                    "model": "Whisper-Large-v3-Turbo",
                }
            }
        },
        # Test all together
        {
            "session": {
                "input_audio_transcription": {
                    "model": "Whisper-Tiny",
                    "language": "en",
                    "prompt": "Hello world test",
                }
            }
        },
        # Test with translate vs transcribe
        {
            "session": {
                "input_audio_transcription": {
                    "task": "transcribe",
                }
            }
        },
        # Test unknown fields
        {
            "session": {
                "input_audio_transcription": {
                    "vocabulary": ["KPG", "Kooperativ"],
                }
            }
        },
    ]

    for i, session_cfg in enumerate(configs):
        update_msg = {"type": "session.update", **session_cfg}
        try:
            async with websockets.connect(WS_URL, open_timeout=5) as ws:
                await asyncio.wait_for(ws.recv(), timeout=3)  # session.created

                await ws.send(json.dumps(update_msg))

                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=3)
                    parsed = json.loads(resp)
                    rtype = parsed.get("type")
                    if rtype == "session.updated":
                        print(f"\nConfig {i+1}: ACCEPTED")
                        print(f"  Sent: {json.dumps(session_cfg)}")
                        print(f"  Response session: {json.dumps(parsed.get('session', {}))}")
                    elif rtype == "error":
                        print(f"\nConfig {i+1}: REJECTED")
                        print(f"  Sent: {json.dumps(session_cfg)}")
                        print(f"  Error: {parsed['error']['message']}")
                    else:
                        print(f"\nConfig {i+1}: UNEXPECTED - {rtype}")
                        print(f"  {json.dumps(parsed)}")
                except asyncio.TimeoutError:
                    print(f"\nConfig {i+1}: NO RESPONSE (accepted silently?)")
                    print(f"  Sent: {json.dumps(session_cfg)}")

                await ws.close()
        except Exception as e:
            print(f"\nConfig {i+1}: ERROR - {e}")


async def test_prompt_boosting_effect():
    """Test if prompt actually affects transcription output."""
    print("\n" + "=" * 60)
    print("  Testing prompt boosting effect on transcription")
    print("=" * 60)

    pcm_data = generate_pcm_tone(duration_s=1.5)
    audio_b64 = base64.b64encode(pcm_data).decode()

    # Run without prompt
    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        transcript_no_prompt = ""
        for _ in range(10):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=8)
                parsed = json.loads(resp)
                if parsed.get("type") == "conversation.item.input_audio_transcription.completed":
                    transcript_no_prompt = parsed.get("transcript", "")
                    break
            except asyncio.TimeoutError:
                break
        await ws.close()

    # Run with prompt
    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created

        # Set prompt
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "prompt": "Kooperative Prozessgestaltung KPG Sozialpädagogik",
                    "language": "de",
                }
            }
        }))
        resp = await asyncio.wait_for(ws.recv(), timeout=3)
        print(f"  session.update response: {resp}")

        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        transcript_with_prompt = ""
        for _ in range(10):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=8)
                parsed = json.loads(resp)
                if parsed.get("type") == "conversation.item.input_audio_transcription.completed":
                    transcript_with_prompt = parsed.get("transcript", "")
                    break
            except asyncio.TimeoutError:
                break
        await ws.close()

    print(f"\n  Without prompt: '{transcript_no_prompt.strip()}'")
    print(f"  With prompt:    '{transcript_with_prompt.strip()}'")


async def test_multiple_commits():
    """Test sending multiple audio buffers in sequence (multi-turn)."""
    print("\n" + "=" * 60)
    print("  Testing multiple sequential commits in one session")
    print("=" * 60)

    pcm_data = generate_pcm_tone(duration_s=1.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        created = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"  Session: {created}")

        for turn in range(3):
            print(f"\n  Turn {turn+1}: append + commit...")
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            for _ in range(10):
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=8)
                    parsed = json.loads(resp)
                    rtype = parsed.get("type")
                    if rtype == "conversation.item.input_audio_transcription.completed":
                        print(f"    Transcript: '{parsed.get('transcript', '').strip()}'")
                        break
                    elif rtype == "input_audio_buffer.committed":
                        print(f"    Buffer committed")
                    else:
                        print(f"    [{rtype}] {json.dumps(parsed)[:200]}")
                except asyncio.TimeoutError:
                    print(f"    Timeout")
                    break

        await ws.close()
        print("\n  Multi-turn test completed")


async def test_clear_buffer():
    """Test input_audio_buffer.clear message."""
    print("\n" + "=" * 60)
    print("  Testing input_audio_buffer.clear")
    print("=" * 60)

    pcm_data = generate_pcm_tone(duration_s=1.0)
    audio_b64 = base64.b64encode(pcm_data).decode()

    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # session.created

        # Append audio
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        print("  Appended audio, now clearing...")

        # Clear buffer
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=3)
            parsed = json.loads(resp)
            print(f"  Clear response: [{parsed.get('type')}] {json.dumps(parsed)}")
        except asyncio.TimeoutError:
            print("  No response (cleared silently)")

        # Now commit (should have nothing to transcribe)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        for _ in range(5):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                parsed = json.loads(resp)
                print(f"  After commit on cleared buffer: [{parsed.get('type')}] {json.dumps(parsed)[:200]}")
            except asyncio.TimeoutError:
                print("  No response after commit (buffer was empty)")
                break

        await ws.close()


async def main():
    await test_session_update_fields()
    await test_prompt_boosting_effect()
    await test_multiple_commits()
    await test_clear_buffer()


if __name__ == "__main__":
    asyncio.run(main())
