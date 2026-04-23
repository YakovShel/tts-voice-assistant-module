from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# Faster defaults for local CPU usage.
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "base")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
ASR_CPU_THREADS = int(os.getenv("ASR_CPU_THREADS", "4"))
ASR_NUM_WORKERS = int(os.getenv("ASR_NUM_WORKERS", "1"))
ASR_VAD_FILTER = os.getenv("ASR_VAD_FILTER", "true").strip().lower() in {"1", "true", "yes", "on"}
ASR_MAX_UPLOAD_MB = int(os.getenv("ASR_MAX_UPLOAD_MB", "25"))

_whisper_model = None
_model_lock = threading.Lock()
_model_error: str | None = None
_model_loading = False


def _package_available() -> bool:
    return WhisperModel is not None


def warmup_model() -> None:
    try:
        _get_model()
    except Exception:
        pass


def _normalize_language(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    if not value:
        return None
    if len(value) > 12:
        raise RuntimeError("Language code is too long")
    return value


def _get_model():
    global _whisper_model, _model_error, _model_loading
    if _whisper_model is not None:
        return _whisper_model
    if WhisperModel is None:
        _model_error = "faster-whisper is not installed"
        return None

    with _model_lock:
        if _whisper_model is not None:
            return _whisper_model
        _model_loading = True
        start = perf_counter()
        try:
            _whisper_model = WhisperModel(
                ASR_MODEL_SIZE,
                device=ASR_DEVICE,
                compute_type=ASR_COMPUTE_TYPE,
                cpu_threads=ASR_CPU_THREADS,
                num_workers=ASR_NUM_WORKERS,
            )
            elapsed = perf_counter() - start
            logger.info(
                "ASR model initialized in %.2fs (size=%s, device=%s, compute_type=%s, vad=%s)",
                elapsed,
                ASR_MODEL_SIZE,
                ASR_DEVICE,
                ASR_COMPUTE_TYPE,
                ASR_VAD_FILTER,
            )
            _model_error = None
            return _whisper_model
        except Exception as exc:
            _model_error = str(exc)
            logger.exception("Failed to initialize faster-whisper model: %s", exc)
            return None
        finally:
            _model_loading = False


def get_asr_capabilities() -> dict[str, Any]:
    package_installed = _package_available()
    model_ready = _whisper_model is not None
    available = package_installed
    reason = None
    if not package_installed:
        reason = "faster-whisper is unavailable; install dependencies from requirements.txt"
    elif _model_error and not model_ready:
        reason = f"model is not ready yet: {_model_error}"

    return {
        "available": available,
        "ready": model_ready,
        "loading": _model_loading,
        "engine": "faster-whisper",
        "model_size": ASR_MODEL_SIZE,
        "device": ASR_DEVICE,
        "compute_type": ASR_COMPUTE_TYPE,
        "cpu_threads": ASR_CPU_THREADS,
        "vad_filter": ASR_VAD_FILTER,
        "max_upload_mb": ASR_MAX_UPLOAD_MB,
        "reason": reason,
    }


def _suffix_from_name(name: str | None, content_type: str | None) -> str:
    if name:
        suffix = Path(name).suffix.strip()
        if suffix:
            return suffix
    mapping = {
        "audio/webm": ".webm",
        "video/webm": ".webm",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
    }
    return mapping.get((content_type or "").lower(), ".bin")


def save_upload_to_temp(upload_file) -> Path:
    suffix = _suffix_from_name(getattr(upload_file, "filename", None), getattr(upload_file, "content_type", None))
    written = 0
    limit_bytes = ASR_MAX_UPLOAD_MB * 1024 * 1024
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        try:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    raise RuntimeError(f"Audio file is too large. Limit: {ASR_MAX_UPLOAD_MB} MB")
                temp_file.write(chunk)
            return Path(temp_file.name)
        except Exception:
            Path(temp_file.name).unlink(missing_ok=True)
            raise


def transcribe_audio_file(file_path: Path, language: str | None = None) -> dict[str, Any]:
    model = _get_model()
    if model is None:
        raise RuntimeError("Speech recognition is unavailable: install faster-whisper and restart the server")

    normalized_language = _normalize_language(language)

    try:
        start = perf_counter()
        segments, info = model.transcribe(
            str(file_path),
            language=normalized_language,
            task="transcribe",
            vad_filter=ASR_VAD_FILTER,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        items = []
        text_parts: list[str] = []
        for segment in segments:
            segment_text = (segment.text or "").strip()
            if not segment_text:
                continue
            text_parts.append(segment_text)
            items.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment_text,
            })
        elapsed = perf_counter() - start
        text = " ".join(text_parts).strip()
        detected_language = getattr(info, "language", None)
        logger.info(
            "ASR transcription completed in %.2fs for %s (requested_language=%s, detected_language=%s)",
            elapsed,
            file_path.name,
            normalized_language,
            detected_language,
        )
        return {
            "text": text,
            "language": detected_language,
            "requested_language": normalized_language,
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
            "duration_sec": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
            "segments": items,
            "processing_sec": round(elapsed, 2),
        }
    except Exception as exc:
        logger.exception("Audio transcription failed: %s", exc)
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc
