"""
Investigation script: Lemonade TTS REST API contract discovery.

PRD-116 Task 0 — Documents the exact TTS API contract for browser integration.

Usage:
    cd host/gaia/src
    python tests/test_lemonade_tts_api.py
"""

import io
import struct
import sys
import time

import httpx

LEMONADE_BASE = "http://localhost:8060"
TTS_ENDPOINT = f"{LEMONADE_BASE}/api/v1/audio/speech"
MODELS_ENDPOINT = f"{LEMONADE_BASE}/v1/models"

TEST_TEXT = "Hello, this is a test of the text to speech system."
SHORT_TEXT = "Hi there."
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a longer sentence to test streaming behavior and chunked transfer encoding. "
    "We want to see how the Lemonade server handles multi-sentence input and whether "
    "it streams the audio back in chunks or sends it all at once."
)


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def investigate_available_voices():
    """Check what TTS voices are available from the models endpoint."""
    separator("1. Available TTS Models & Voices")

    resp = httpx.get(MODELS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    models = resp.json()

    tts_models = [m for m in models["data"] if "tts" in m.get("labels", [])]
    print(f"TTS models found: {len(tts_models)}")
    for m in tts_models:
        print(f"  - id: {m['id']}")
        print(f"    recipe: {m['recipe']}")
        print(f"    checkpoint: {m['checkpoint']}")
        print(f"    labels: {m['labels']}")
        print(f"    recipe_options: {m.get('recipe_options', {})}")
    return tts_models


def investigate_basic_tts_request():
    """Test the basic TTS request format (OpenAI-compatible)."""
    separator("2. Basic TTS Request (OpenAI-compatible format)")

    # Try OpenAI-compatible format
    payload = {
        "model": "kokoro-v1",
        "input": TEST_TEXT,
        "voice": "af_heart",
    }
    print(f"Request payload: {payload}")

    start = time.perf_counter()
    resp = httpx.post(TTS_ENDPOINT, json=payload, timeout=60)
    elapsed = time.perf_counter() - start

    print(f"Status code: {resp.status_code}")
    print(f"Response headers:")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")
    print(f"Content-Type: {resp.headers.get('content-type')}")
    print(f"Content-Length: {resp.headers.get('content-length', 'not set')}")
    print(f"Transfer-Encoding: {resp.headers.get('transfer-encoding', 'not set')}")
    print(f"Response body size: {len(resp.content)} bytes")
    print(f"Time to first byte (approx): {elapsed:.3f}s")

    return resp


def investigate_audio_format(resp: httpx.Response):
    """Analyze the audio format of the TTS response."""
    separator("3. Audio Format Analysis")

    content_type = resp.headers.get("content-type", "unknown")
    data = resp.content
    print(f"Content-Type: {content_type}")
    print(f"Total bytes: {len(data)}")

    # Check for WAV header (RIFF)
    if data[:4] == b"RIFF":
        print("Format: WAV (RIFF header detected)")
        # Parse WAV header
        riff_size = struct.unpack_from("<I", data, 4)[0]
        wave_id = data[8:12]
        print(f"  RIFF size: {riff_size}")
        print(f"  WAVE ID: {wave_id}")

        # Parse fmt chunk
        if data[12:16] == b"fmt ":
            fmt_size = struct.unpack_from("<I", data, 16)[0]
            audio_format = struct.unpack_from("<H", data, 20)[0]
            num_channels = struct.unpack_from("<H", data, 22)[0]
            sample_rate = struct.unpack_from("<I", data, 24)[0]
            byte_rate = struct.unpack_from("<I", data, 28)[0]
            block_align = struct.unpack_from("<H", data, 32)[0]
            bits_per_sample = struct.unpack_from("<H", data, 34)[0]

            format_names = {1: "PCM", 3: "IEEE Float", 6: "A-law", 7: "mu-law"}
            print(f"  Audio format: {format_names.get(audio_format, audio_format)}")
            print(f"  Channels: {num_channels}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Byte rate: {byte_rate}")
            print(f"  Block align: {block_align}")
            print(f"  Bits per sample: {bits_per_sample}")

            # Calculate duration
            # Find data chunk
            offset = 20 + fmt_size
            while offset < len(data) - 8:
                chunk_id = data[offset : offset + 4]
                chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
                if chunk_id == b"data":
                    duration = chunk_size / byte_rate
                    print(f"  Data chunk size: {chunk_size} bytes")
                    print(f"  Duration: {duration:.2f}s")
                    break
                offset += 8 + chunk_size
        return "wav"

    # Check for MP3 header (0xFF 0xFB or ID3)
    elif data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        print("Format: MP3")
        return "mp3"

    # Check for OGG/Opus header
    elif data[:4] == b"OggS":
        print("Format: OGG (possibly Opus)")
        return "ogg"

    else:
        print(f"Format: Unknown (first 16 bytes: {data[:16].hex()})")
        # Might be raw PCM
        print("  Could be raw PCM - checking if data length is consistent")
        # Common: 16-bit, 24kHz, mono => 48000 bytes/sec
        for sr in [16000, 22050, 24000, 44100, 48000]:
            for bits in [16, 32]:
                bps = sr * (bits // 8)
                duration = len(data) / bps
                if 0.5 < duration < 30:
                    print(f"  If PCM {bits}-bit {sr}Hz mono: {duration:.2f}s")
        return "unknown"


def investigate_streaming_behavior():
    """Test if TTS supports chunked/streaming responses."""
    separator("4. Streaming Behavior")

    payload = {
        "model": "kokoro-v1",
        "input": LONG_TEXT,
        "voice": "af_heart",
    }
    print(f"Testing with long text ({len(LONG_TEXT)} chars)...")

    start = time.perf_counter()
    chunks = []
    chunk_times = []

    with httpx.stream("POST", TTS_ENDPOINT, json=payload, timeout=120) as resp:
        print(f"Status: {resp.status_code}")
        print(f"Transfer-Encoding: {resp.headers.get('transfer-encoding', 'not set')}")
        print(f"Content-Length: {resp.headers.get('content-length', 'not set')}")

        for chunk in resp.iter_bytes(chunk_size=4096):
            t = time.perf_counter() - start
            chunks.append(chunk)
            chunk_times.append(t)

    total_size = sum(len(c) for c in chunks)
    total_time = time.perf_counter() - start

    print(f"\nReceived {len(chunks)} chunks, total {total_size} bytes in {total_time:.3f}s")
    if chunk_times:
        print(f"Time to first chunk: {chunk_times[0]:.3f}s")
        if len(chunk_times) > 1:
            print(f"Time to last chunk: {chunk_times[-1]:.3f}s")
            # Show chunk size distribution
            chunk_sizes = [len(c) for c in chunks]
            print(f"Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.0f}")
            # Show timing between chunks
            if len(chunk_times) > 2:
                deltas = [chunk_times[i+1] - chunk_times[i] for i in range(len(chunk_times)-1)]
                print(f"Inter-chunk delays: min={min(deltas)*1000:.1f}ms, max={max(deltas)*1000:.1f}ms, avg={sum(deltas)/len(deltas)*1000:.1f}ms")


def investigate_voice_options():
    """Test different voice parameters."""
    separator("5. Voice Selection")

    # Try different voices
    voices_to_test = ["af_heart", "af_bella", "am_adam", "bf_emma", "alloy", "nova", "shimmer"]

    for voice in voices_to_test:
        payload = {
            "model": "kokoro-v1",
            "input": SHORT_TEXT,
            "voice": voice,
        }
        try:
            resp = httpx.post(TTS_ENDPOINT, json=payload, timeout=30)
            if resp.status_code == 200:
                print(f"  Voice '{voice}': OK ({len(resp.content)} bytes)")
            else:
                print(f"  Voice '{voice}': {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  Voice '{voice}': ERROR - {e}")


def investigate_request_formats():
    """Test different request format variations."""
    separator("6. Request Format Variations")

    # Test with response_format parameter
    formats_to_test = ["wav", "mp3", "opus", "pcm", "aac", "flac"]
    for fmt in formats_to_test:
        payload = {
            "model": "kokoro-v1",
            "input": SHORT_TEXT,
            "voice": "af_heart",
            "response_format": fmt,
        }
        try:
            resp = httpx.post(TTS_ENDPOINT, json=payload, timeout=30)
            ct = resp.headers.get("content-type", "unknown")
            if resp.status_code == 200:
                print(f"  response_format='{fmt}': OK, content-type={ct}, {len(resp.content)} bytes")
                # Check first bytes
                first_bytes = resp.content[:4]
                print(f"    First 4 bytes: {first_bytes.hex()} ({first_bytes})")
            else:
                print(f"  response_format='{fmt}': {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  response_format='{fmt}': ERROR - {e}")

    # Test with speed parameter
    separator("7. Speed Parameter")
    for speed in [0.5, 1.0, 1.5, 2.0]:
        payload = {
            "model": "kokoro-v1",
            "input": SHORT_TEXT,
            "voice": "af_heart",
            "speed": speed,
        }
        try:
            resp = httpx.post(TTS_ENDPOINT, json=payload, timeout=30)
            if resp.status_code == 200:
                print(f"  speed={speed}: OK ({len(resp.content)} bytes)")
            else:
                print(f"  speed={speed}: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  speed={speed}: ERROR - {e}")


def investigate_abort_behavior():
    """Test if TTS stream can be aborted mid-response."""
    separator("8. Abort (Barge-in) Behavior")

    payload = {
        "model": "kokoro-v1",
        "input": LONG_TEXT,
        "voice": "af_heart",
    }
    print("Starting long TTS request and aborting after first chunk...")

    start = time.perf_counter()
    partial_bytes = 0
    try:
        with httpx.stream("POST", TTS_ENDPOINT, json=payload, timeout=120) as resp:
            for chunk in resp.iter_bytes(chunk_size=4096):
                partial_bytes += len(chunk)
                if partial_bytes > 4096:
                    elapsed = time.perf_counter() - start
                    print(f"  Received {partial_bytes} bytes in {elapsed:.3f}s, aborting...")
                    break
        abort_time = time.perf_counter() - start
        print(f"  Connection closed after {abort_time:.3f}s")
        print(f"  Abort appears to work (client-side disconnect)")
    except Exception as e:
        print(f"  Abort error: {e}")

    # Verify server is still healthy after abort
    try:
        health = httpx.get(f"{LEMONADE_BASE}/api/v1/health", timeout=5)
        print(f"  Server health after abort: {health.status_code}")
    except Exception as e:
        print(f"  Server health check failed: {e}")


def main():
    print("=" * 60)
    print("  Lemonade TTS REST API Investigation")
    print(f"  Endpoint: {TTS_ENDPOINT}")
    print("=" * 60)

    # Check server health first
    try:
        health = httpx.get(f"{LEMONADE_BASE}/api/v1/health", timeout=5)
        health_data = health.json()
        print(f"\nServer status: {health_data['status']}")
        print(f"Version: {health_data['version']}")
        tts_models = [m for m in health_data.get("all_models_loaded", []) if m["type"] == "tts"]
        if not tts_models:
            print("ERROR: No TTS model loaded!")
            sys.exit(1)
        print(f"TTS model: {tts_models[0]['model_name']} (recipe: {tts_models[0]['recipe']})")
    except Exception as e:
        print(f"ERROR: Lemonade server not reachable: {e}")
        sys.exit(1)

    investigate_available_voices()
    resp = investigate_basic_tts_request()
    investigate_audio_format(resp)
    investigate_streaming_behavior()
    investigate_voice_options()
    investigate_request_formats()
    investigate_abort_behavior()

    separator("INVESTIGATION COMPLETE")
    print("See output above for TTS API contract details.")


if __name__ == "__main__":
    main()
