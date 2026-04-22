from __future__ import annotations

import logging
import platform
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Generator, Iterable, Literal

import numpy as np
import soundfile as sf

from app.audio_utils import save_wav_pcm16

logger = logging.getLogger(__name__)

try:
    from kokoro import KPipeline
except Exception:
    KPipeline = None

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'generated_audio'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_RATE = 24000

KOKORO_VOICES = [
    'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'am_adam',
    'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
]

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if KPipeline is None:
        return None
    try:
        _pipeline = KPipeline(lang_code='a')
        return _pipeline
    except Exception as exc:
        logger.exception('Failed to initialize Kokoro pipeline: %s', exc)
        return None


@lru_cache(maxsize=1)
def get_system_voices() -> list[str]:
    if platform.system() != 'Darwin':
        return []
    try:
        result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True, check=True)
        voices: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            voices.append(line.split()[0])
        return voices
    except Exception as exc:
        logger.warning('Could not read system voices: %s', exc)
        return []


@lru_cache(maxsize=1)
def get_neural_voices() -> list[str]:
    return KOKORO_VOICES if _get_pipeline() is not None else []


def get_engine_capabilities() -> dict:
    system_voices = get_system_voices()
    neural_voices = get_neural_voices()
    system_available = platform.system() == 'Darwin' and len(system_voices) > 0
    neural_available = len(neural_voices) > 0
    return {
        'system': {
            'available': system_available,
            'voices': system_voices,
            'streaming': False,
            'reason': None if system_available else 'available only on macOS with say voices',
        },
        'neural': {
            'available': neural_available,
            'voices': neural_voices,
            'streaming': neural_available,
            'reason': None if neural_available else 'kokoro pipeline is unavailable',
        },
        'auto': {
            'available': system_available or neural_available,
            'voices': [],
            'streaming': neural_available,
            'reason': None if (system_available or neural_available) else 'no engine available',
        },
    }


def normalize_system_voice(voice: str) -> str | None:
    if not voice or voice == 'default':
        return None
    return voice


def normalize_neural_voice(voice: str) -> str:
    if not voice or voice == 'default':
        return 'af_heart'
    return voice


def _new_output_path(suffix: str) -> Path:
    return OUTPUT_DIR / f'{uuid.uuid4()}{suffix}'


def _atomic_save_wav(target_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + f'.{uuid.uuid4().hex}.tmp')
    try:
        save_wav_pcm16(temp_path, audio, sample_rate)
        temp_path.replace(target_path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def apply_neural_rate(audio: np.ndarray, rate: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    if abs(rate - 1.0) < 1e-6:
        return audio
    if rate <= 0:
        raise ValueError('rate must be greater than 0')
    new_length = max(1, int(round(len(audio) / rate)))
    old_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=True)
    return np.asarray(np.interp(new_positions, old_positions, audio), dtype=np.float32)


def _convert_audio_to_wav(input_path: Path, output_path: Path) -> None:
    data, sample_rate = sf.read(str(input_path), dtype='float32')
    if getattr(data, 'ndim', 1) > 1:
        data = data[:, 0]
    save_wav_pcm16(output_path, np.asarray(data, dtype=np.float32), sample_rate)


def synthesize_with_system(text: str, voice: str, rate: float) -> Path:
    if platform.system() != 'Darwin':
        raise RuntimeError("System TTS via 'say' is available only on macOS")
    temp_aiff_path = _new_output_path('.aiff')
    output_path = _new_output_path('.wav')
    cmd = ['say']
    normalized_voice = normalize_system_voice(voice)
    if normalized_voice:
        cmd += ['-v', normalized_voice]
    speed = max(80, int(200 * rate))
    cmd += ['-r', str(speed), '-o', str(temp_aiff_path), text]
    try:
        subprocess.run(cmd, check=True)
        _convert_audio_to_wav(temp_aiff_path, output_path)
        return output_path
    except subprocess.CalledProcessError as exc:
        logger.exception('System synthesis failed: %s', exc)
        raise RuntimeError('System TTS synthesis failed') from exc
    finally:
        try:
            if temp_aiff_path.exists():
                temp_aiff_path.unlink()
        except Exception:
            pass


def _ensure_neural_available() -> None:
    if _get_pipeline() is None:
        raise RuntimeError('Neural TTS is unavailable: install kokoro and its dependencies')


def _generate_kokoro_chunks(text: str, voice: str) -> Iterable[np.ndarray]:
    _ensure_neural_available()
    kokoro_voice = normalize_neural_voice(voice)
    pipeline = _get_pipeline()
    generator = pipeline(text, voice=kokoro_voice)
    produced = False
    for _, _, audio in generator:
        produced = True
        yield np.asarray(audio, dtype=np.float32)
    if not produced:
        raise RuntimeError('Kokoro did not generate audio')


def synthesize_with_kokoro(text: str, voice: str, rate: float) -> Path:
    output_path = _new_output_path('.wav')
    audio_chunks = []
    for chunk in _generate_kokoro_chunks(text, voice):
        audio_chunks.append(apply_neural_rate(chunk, rate))
    if not audio_chunks:
        raise RuntimeError('Kokoro did not generate audio')
    full_audio = np.concatenate(audio_chunks)
    save_wav_pcm16(output_path, full_audio, DEFAULT_SAMPLE_RATE)
    return output_path


def stream_kokoro_pcm(text: str, voice: str, rate: float, cache_wav_path: Path | None = None) -> Generator[bytes, None, None]:
    _ensure_neural_available()
    collected_chunks: list[np.ndarray] = []
    for audio in _generate_kokoro_chunks(text, voice):
        adjusted_chunk = apply_neural_rate(audio, rate)
        mono_chunk = np.asarray(adjusted_chunk, dtype=np.float32)
        if mono_chunk.ndim > 1:
            mono_chunk = mono_chunk[:, 0]
        collected_chunks.append(mono_chunk)
        yield mono_chunk.tobytes()
    if cache_wav_path is not None and collected_chunks:
        full_audio = np.concatenate(collected_chunks)
        _atomic_save_wav(cache_wav_path, full_audio, DEFAULT_SAMPLE_RATE)


def stream_wav_file_as_pcm(wav_path: Path, block_size: int = 4096) -> Generator[bytes, None, None]:
    with sf.SoundFile(str(wav_path), mode='r') as f:
        while True:
            data = f.read(block_size, dtype='float32')
            if len(data) == 0:
                break
            if getattr(data, 'ndim', 1) > 1:
                data = data[:, 0]
            yield np.asarray(data, dtype=np.float32).tobytes()


def synthesize_to_file(text: str, engine: Literal['system', 'neural'], voice: str, rate: float) -> Path:
    if engine == 'system':
        return synthesize_with_system(text, voice, rate)
    if engine == 'neural':
        return synthesize_with_kokoro(text, voice, rate)
    raise ValueError(f'Unknown engine: {engine}')
