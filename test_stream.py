"""Test client for Chatterbox Turbo streaming endpoint with real-time playback."""

import argparse
import queue
import threading

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

BASE_URL = "http://localhost:8002"
SAMPLE_RATE = 24000


def register_voice(filepath: str, voice_id: str = None) -> str:
    """Register a reference audio file (precomputes conditionals). Returns voice_id."""
    with open(filepath, "rb") as f:
        files = {"file": (filepath, f, "audio/wav")}
        data = {"voice_id": voice_id} if voice_id else {}
        response = httpx.post(f"{BASE_URL}/voice/register", files=files, data=data, timeout=60.0)
        response.raise_for_status()
        result = response.json()
        print(f"Registered voice: {result}")
        return result["voice_id"]


def list_voices():
    """List all registered voices."""
    response = httpx.get(f"{BASE_URL}/voices", timeout=10.0)
    response.raise_for_status()
    result = response.json()
    print(f"Registered voices ({result['total']}):")
    for vid, info in result["voices"].items():
        status = "in-memory" if info["cached_in_memory"] else "on-disk"
        print(f"  {vid} ({status})")
    return result


def test_full(text: str, voice_id: str = None, **kwargs):
    """Test the /tts endpoint (full WAV response)."""
    payload = {"text": text, **kwargs}
    if voice_id:
        payload["voice_id"] = voice_id

    print(f"Requesting full synthesis...")
    response = httpx.post(f"{BASE_URL}/tts", json=payload, timeout=120.0)
    response.raise_for_status()

    sf.write("test_full_output.wav", np.frombuffer(response.content, dtype=np.int16), SAMPLE_RATE)
    # Actually let's just save the raw wav bytes
    with open("test_full_output.wav", "wb") as f:
        f.write(response.content)

    print(f"Saved to test_full_output.wav ({len(response.content)} bytes)")


def test_stream(text: str, voice_id: str = None, **kwargs):
    """Test the /tts/stream endpoint with real-time chunk playback."""
    audio_queue = queue.Queue()
    all_chunks = []

    def audio_callback(outdata, frames, time, status):
        try:
            data = audio_queue.get_nowait()
            if len(data) < len(outdata):
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:len(outdata)].reshape(-1, 1)
        except queue.Empty:
            outdata[:] = 0

    payload = {"text": text, **kwargs}
    if voice_id:
        payload["voice_id"] = voice_id

    print(f"Text: {text}")
    print(f"Voice: {voice_id or 'default'}")
    print("Streaming and playing in real-time...\n")

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=1024,
    )
    stream.start()

    total_bytes = 0

    try:
        with httpx.stream("POST", f"{BASE_URL}/tts/stream", json=payload, timeout=120.0) as response:
            print(f"Status: {response.status_code}")

            for i, chunk in enumerate(response.iter_bytes(chunk_size=4096)):
                total_bytes += len(chunk)
                print(f"Chunk {i + 1}: {len(chunk)} bytes")

                audio = np.frombuffer(chunk, dtype=np.float32)
                all_chunks.append(audio)
                audio_queue.put(audio)

    except Exception as e:
        print(f"Error: {e}")

    while not audio_queue.empty():
        sd.sleep(100)
    sd.sleep(500)

    stream.stop()
    stream.close()

    print(f"\nTotal bytes: {total_bytes}")

    if all_chunks:
        full_audio = np.concatenate(all_chunks)
        sf.write("test_stream_output.wav", full_audio, SAMPLE_RATE)
        print("Saved to test_stream_output.wav")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Chatterbox Turbo streaming endpoint")
    parser.add_argument(
        "text",
        nargs="?",
        default=(
            "Hello there! This is a test of the Chatterbox Turbo text to speech service. "
            "We are testing how well the system handles multiple sentences. "
            "Can it handle questions? What about exclamations! "
            "Finally, we conclude this test with a simple goodbye."
        ),
        help="Text to synthesize",
    )
    parser.add_argument("--voice-id", default=None, help="Voice ID from /voice/register")
    parser.add_argument("--register", default=None, help="Path to reference audio to register first")
    parser.add_argument("--list-voices", action="store_true", help="List registered voices and exit")
    parser.add_argument("--mode", default="stream", choices=["stream", "full"], help="Test mode")
    parser.add_argument("--url", default=BASE_URL, help="Server URL")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    BASE_URL = args.url

    if args.list_voices:
        list_voices()
        exit(0)

    voice_id = args.voice_id
    if args.register:
        voice_id = register_voice(args.register, voice_id)

    extra = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}

    if args.mode == "stream":
        test_stream(args.text, voice_id, **extra)
    else:
        test_full(args.text, voice_id, **extra)
