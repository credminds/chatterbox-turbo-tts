"""
FastAPI server for Chatterbox Turbo TTS service.

Key design for voice agent use:
- Voice conditionals are precomputed on upload and cached to disk as .pt files
- On each /tts request, we just load the cached tensor — no re-processing the wav
- Model weights load once at startup and stay in GPU memory
- Pseudo-streaming by splitting text into sentences

Endpoints:
- POST /voice/register - Upload ref audio, precompute + cache conditionals
- POST /tts - Full WAV synthesis
- POST /tts/stream - Raw PCM streaming (sentence-level chunks)
- POST /tts/stream/wav - Buffered WAV streaming
- GET  /health, /voices
"""

import asyncio
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None  # ChatterboxTurboTTS instance (loaded once)

SAMPLE_RATE = 24000  # S3GEN_SR

VOICES_DIR = Path(os.getenv("VOICES_DIR", "./voices"))
# voices/<voice_id>/ref.wav        — original reference audio
# voices/<voice_id>/conds.pt       — precomputed Conditionals tensor
# voices/<voice_id>/meta.json      — {"name": "Sarah's Voice", "created_at": "..."}

# In-memory LRU of loaded Conditionals (voice_id -> Conditionals on GPU)
_conds_cache: dict = {}
MAX_CACHE_SIZE = int(os.getenv("MAX_VOICE_CACHE", "50"))

DEFAULT_VOICE_ID = "default"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for pseudo-streaming."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Conditionals cache
# ---------------------------------------------------------------------------

def _read_voice_meta(voice_id: str) -> dict:
    """Read meta.json for a voice, or return defaults."""
    meta_path = VOICES_DIR / voice_id / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"name": voice_id, "voice_id": voice_id}


def _write_voice_meta(voice_id: str, meta: dict):
    """Write meta.json for a voice."""
    meta_path = VOICES_DIR / voice_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


def _precompute_and_save(voice_id: str, wav_path: str) -> float:
    """
    Run prepare_conditionals() once, save the resulting Conditionals to disk.
    Returns the time taken in seconds.
    """
    t0 = time.time()
    model.prepare_conditionals(wav_path, norm_loudness=True)
    conds = model.conds  # just computed

    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    conds.save(voice_dir / "conds.pt")

    # Also cache in memory
    _conds_cache[voice_id] = conds
    _evict_cache()

    elapsed = time.time() - t0
    logger.info(f"Precomputed conditionals for '{voice_id}' in {elapsed:.2f}s")
    return elapsed


def _load_conds(voice_id: str):
    """
    Load Conditionals for a voice_id.
    1. Check in-memory cache (instant)
    2. Load from disk .pt file (~5-20ms)
    Raises HTTPException if not found.
    """
    # 1. Memory cache
    if voice_id in _conds_cache:
        return _conds_cache[voice_id]

    # 2. Disk cache
    conds_path = VOICES_DIR / voice_id / "conds.pt"
    if not conds_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_id}' not found. Register one via POST /voice/register"
        )

    from chatterbox.tts_turbo import Conditionals
    conds = Conditionals.load(conds_path).to(model.device)
    _conds_cache[voice_id] = conds
    _evict_cache()
    return conds


def _evict_cache():
    """Simple FIFO eviction when cache exceeds MAX_CACHE_SIZE."""
    while len(_conds_cache) > MAX_CACHE_SIZE:
        oldest_key = next(iter(_conds_cache))
        del _conds_cache[oldest_key]


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    try:
        load_model()
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Chatterbox Turbo TTS server")


app = FastAPI(
    title="Chatterbox Turbo TTS Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Audio-Format", "X-Audio-Encoding", "X-Audio-Sample-Rate", "X-Audio-Channels"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None  # Registered voice ID. None = built-in default.
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True


# ---------------------------------------------------------------------------
# Model loading (once)
# ---------------------------------------------------------------------------

def load_model():
    global model

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"CUDA available: {torch.cuda.is_available()}, device: {device}")

    logger.info("Loading ChatterboxTurboTTS from pretrained...")
    model = ChatterboxTurboTTS.from_pretrained(device)
    logger.info(f"Model loaded on {device}, sample_rate={model.sr}")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-warm: load any existing cached conditionals into memory
    count = 0
    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir() and (voice_dir / "conds.pt").exists():
            try:
                _load_conds(voice_dir.name)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to pre-load voice '{voice_dir.name}': {e}")
    if count:
        logger.info(f"Pre-loaded {count} cached voice(s) into memory")


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_audio(text: str, conds, request: TTSRequest) -> np.ndarray:
    """
    Generate audio using pre-loaded conditionals (no wav re-processing).
    Returns float32 numpy array.
    """
    # Swap the model's active conditionals to this voice
    model.conds = conds

    wav_tensor = model.generate(
        text,
        audio_prompt_path=None,  # conditionals already set, skip re-processing
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        norm_loudness=request.norm_loudness,
    )
    wav = wav_tensor.squeeze(0).cpu().numpy()
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return wav


def resolve_conds(request: TTSRequest):
    """Resolve the Conditionals object for a request."""
    voice_id = request.voice_id

    if not voice_id or voice_id == DEFAULT_VOICE_ID:
        # Use built-in default from the model weights
        if model.conds is None:
            raise HTTPException(
                status_code=400,
                detail="No default voice available. Register a voice via POST /voice/register"
            )
        return model.conds

    return _load_conds(voice_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": "ChatterboxTurboTTS",
        "sample_rate": SAMPLE_RATE,
        "cached_voices": len(_conds_cache),
        "streaming_supported": True,
    }


@app.get("/voices")
async def list_voices():
    """List all registered voices with their name and status."""
    voices = {}
    if VOICES_DIR.exists():
        for voice_dir in VOICES_DIR.iterdir():
            if voice_dir.is_dir():
                vid = voice_dir.name
                meta = _read_voice_meta(vid)
                voices[vid] = {
                    "name": meta.get("name", vid),
                    "created_at": meta.get("created_at"),
                    "cached_in_memory": vid in _conds_cache,
                    "has_conds": (voice_dir / "conds.pt").exists(),
                    "has_ref_audio": (voice_dir / "ref.wav").exists(),
                }
    return {
        "voices": voices,
        "default": DEFAULT_VOICE_ID,
        "total": len(voices),
    }


@app.post("/voice/register")
async def register_voice(
    file: UploadFile = File(...),
    voice_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
):
    """
    Upload a reference audio and precompute voice conditionals.

    This is a ONE-TIME operation per voice. After this:
    - Conditionals are saved to disk (survive server restarts)
    - Conditionals are cached in GPU memory (instant lookup)
    - Every /tts call with this voice_id skips all audio processing

    Args:
        file: Reference audio file (WAV/MP3/etc), must be >5 seconds.
        voice_id: Optional custom ID (auto-generated if omitted).
        name: Human-readable name, e.g. "Sarah - Customer Support".

    Requirements: reference audio must be >5 seconds.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vid = voice_id or str(uuid.uuid4())[:8]
    vid = re.sub(r'[^a-zA-Z0-9_-]', '_', vid)

    voice_dir = VOICES_DIR / vid
    voice_dir.mkdir(parents=True, exist_ok=True)
    wav_dest = voice_dir / "ref.wav"

    # Save the uploaded file
    with open(wav_dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Validate duration
    try:
        import librosa
        audio, sr = librosa.load(str(wav_dest), sr=SAMPLE_RATE)
        duration = len(audio) / sr
        if duration < 5.0:
            shutil.rmtree(voice_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Audio must be longer than 5 seconds (got {duration:.1f}s)"
            )
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Precompute conditionals (the expensive step — only happens once)
    try:
        elapsed = _precompute_and_save(vid, str(wav_dest))
    except Exception as e:
        shutil.rmtree(voice_dir, ignore_errors=True)
        logger.error(f"Failed to precompute conditionals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process voice: {e}")

    # Save metadata
    voice_name = name or vid
    from datetime import datetime, timezone
    _write_voice_meta(vid, {
        "voice_id": vid,
        "name": voice_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 1),
    })

    return {
        "voice_id": vid,
        "name": voice_name,
        "duration_seconds": round(duration, 1),
        "precompute_time_seconds": round(elapsed, 2),
        "message": f"Voice '{voice_name}' ({vid}) registered and cached. Use voice_id=\"{vid}\" in /tts requests.",
    }


@app.get("/voice/{voice_id}")
async def get_voice(voice_id: str):
    """Get details for a specific registered voice."""
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    meta = _read_voice_meta(voice_id)
    return {
        **meta,
        "cached_in_memory": voice_id in _conds_cache,
        "has_conds": (voice_dir / "conds.pt").exists(),
        "has_ref_audio": (voice_dir / "ref.wav").exists(),
    }


class VoiceUpdateRequest(BaseModel):
    name: str


@app.patch("/voice/{voice_id}")
async def update_voice(voice_id: str, request: VoiceUpdateRequest):
    """Update voice metadata (e.g. rename)."""
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    meta = _read_voice_meta(voice_id)
    meta["name"] = request.name
    _write_voice_meta(voice_id, meta)
    return {"voice_id": voice_id, "name": request.name, "message": "Voice updated"}


@app.delete("/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a registered voice and its cached conditionals."""
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    meta = _read_voice_meta(voice_id)
    _conds_cache.pop(voice_id, None)
    shutil.rmtree(voice_dir, ignore_errors=True)
    return {"message": f"Voice '{meta.get('name', voice_id)}' ({voice_id}) deleted"}


@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text. Returns a complete WAV file."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    conds = resolve_conds(request)

    try:
        logger.info(f"Synthesizing: '{request.text[:80]}...' voice={request.voice_id or 'default'}")

        t0 = time.time()
        with torch.inference_mode():
            wav = generate_audio(request.text, conds, request)

        max_val = np.abs(wav).max()
        if max_val > 0:
            wav = wav * (0.95 / max_val)
        wav = np.clip(wav, -1.0, 1.0)

        elapsed = time.time() - t0
        logger.info(f"Generated in {elapsed:.2f}s, {len(wav)} samples")

        buffer = io.BytesIO()
        sf.write(buffer, wav, SAMPLE_RATE, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="tts_output.wav"',
                "X-Generation-Time": f"{elapsed:.3f}",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@app.post("/tts/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    Pseudo-streaming TTS: splits text into sentences, streams audio
    chunks as each sentence finishes generating.

    Audio format: Raw PCM float32, mono, 24kHz
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    conds = resolve_conds(request)

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        try:
            sentences = split_sentences(request.text)
            if not sentences:
                sentences = [request.text]

            logger.info(f"Stream: {len(sentences)} sentence(s), voice={request.voice_id or 'default'}")

            with torch.inference_mode():
                for sentence in sentences:
                    t0 = time.time()
                    wav = generate_audio(sentence, conds, request)
                    wav = np.clip(wav, -1.0, 1.0)
                    elapsed = time.time() - t0
                    logger.info(f"  Sentence ({elapsed:.2f}s): '{sentence[:50]}...'")

                    yield wav.tobytes()
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            raise

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Audio-Format": "pcm",
            "X-Audio-Encoding": "float32",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": "1",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/tts/stream/wav")
async def synthesize_speech_stream_wav(request: TTSRequest):
    """
    Sentence-level generation, returned as a single WAV.
    For lower latency use /tts/stream with raw PCM.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    conds = resolve_conds(request)

    async def wav_generator() -> AsyncGenerator[bytes, None]:
        try:
            sentences = split_sentences(request.text)
            if not sentences:
                sentences = [request.text]

            all_chunks = []
            with torch.inference_mode():
                for sentence in sentences:
                    wav = generate_audio(sentence, conds, request)
                    all_chunks.append(wav)
                    await asyncio.sleep(0)

            if not all_chunks:
                return

            full_audio = np.concatenate(all_chunks)
            max_val = np.abs(full_audio).max()
            if max_val > 0:
                full_audio = full_audio * (0.95 / max_val)
            full_audio = np.clip(full_audio, -1.0, 1.0)

            buffer = io.BytesIO()
            sf.write(buffer, full_audio, SAMPLE_RATE, format='WAV')
            buffer.seek(0)

            while True:
                data = buffer.read(8192)
                if not data:
                    break
                yield data

        except Exception as e:
            logger.error(f"WAV streaming error: {e}", exc_info=True)
            raise

    return StreamingResponse(
        wav_generator(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="tts_output.wav"',
            "Transfer-Encoding": "chunked",
        },
    )


@app.get("/")
async def root():
    return {
        "service": "Chatterbox Turbo TTS Service",
        "version": "1.0.0",
        "model": "ChatterboxTurboTTS (ResembleAI)",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cached_voices": len(_conds_cache),
        "endpoints": {
            "health": "GET /health",
            "voices": "GET /voices - List registered voices",
            "register_voice": "POST /voice/register - Upload ref audio, precompute conditionals (one-time)",
            "delete_voice": "DELETE /voice/{voice_id}",
            "tts": "POST /tts - Full WAV synthesis",
            "tts_stream": "POST /tts/stream - Sentence-level raw PCM streaming",
            "tts_stream_wav": "POST /tts/stream/wav - Buffered WAV streaming",
        },
        "audio": {
            "format": "PCM float32 / WAV",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        },
    }


def main():
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8009"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
