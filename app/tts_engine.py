from __future__ import annotations

import logging
import platform
import re
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
MIXED_SEGMENT_PAUSE_SEC = 0.06

KOKORO_VOICES = [
    'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'am_adam',
    'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
]

KNOWN_RUSSIAN_SYSTEM_VOICES = (
    'Milena', 'Yuri', 'Katya', 'Anna', 'Alena', 'Alyona', 'Daria', 'Irina', 'Pavel',
)
KNOWN_ENGLISH_SYSTEM_VOICES = (
    'Samantha', 'Alex', 'Victoria', 'Daniel', 'Karen', 'Moira', 'Tessa', 'Rishi',
    'Fiona', 'Ava', 'Allison', 'Fred', 'Serena', 'Tom', 'Zoe', 'Bruce', 'Junior',
)
CYRILLIC_RE = re.compile(r'[А-Яа-яЁё]')
LATIN_RE = re.compile(r'[A-Za-z]')
LANG_TOKEN_RE = re.compile(r'[A-Za-z]+(?:[\-\'][A-Za-z]+)*|[А-Яа-яЁё]+(?:[\-\'][А-Яа-яЁё]+)*|\s+|[^\w\s]+|\d+')

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
    supports_mixed = system_available or neural_available
    return {
        'system': {
            'available': system_available,
            'voices': system_voices,
            'streaming': False,
            'mixed_language': system_available,
            'reason': None if system_available else 'available only on macOS with say voices',
        },
        'neural': {
            'available': neural_available,
            'voices': neural_voices,
            'streaming': neural_available,
            'mixed_language': supports_mixed,
            'reason': None if neural_available else 'kokoro pipeline is unavailable',
        },
        'auto': {
            'available': system_available or neural_available,
            'voices': [],
            'streaming': neural_available,
            'mixed_language': supports_mixed,
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


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.size == 0 or source_rate == target_rate:
        return mono
    new_length = max(1, int(round(len(mono) * float(target_rate) / float(source_rate))))
    old_positions = np.linspace(0.0, 1.0, num=len(mono), endpoint=True)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=True)
    return np.asarray(np.interp(new_positions, old_positions, mono), dtype=np.float32)


def _load_audio_mono(input_path: Path, target_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    data, sample_rate = sf.read(str(input_path), dtype='float32')
    if getattr(data, 'ndim', 1) > 1:
        data = data[:, 0]
    return _resample_audio(np.asarray(data, dtype=np.float32), sample_rate, target_rate)


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


def _contains_ru_and_en(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text)) and bool(LATIN_RE.search(text))


def _classify_token_language(token: str) -> str | None:
    if CYRILLIC_RE.search(token):
        return 'ru'
    if LATIN_RE.search(token):
        return 'en'
    return None


def split_text_by_language(text: str) -> list[dict[str, str]]:
    stripped = (text or '').strip()
    if not stripped:
        return []
    tokens = LANG_TOKEN_RE.findall(stripped)
    if not tokens:
        return [{'lang': 'ru' if CYRILLIC_RE.search(stripped) else 'en', 'text': stripped}]

    segments: list[dict[str, str | None]] = []
    pending_neutral = ''
    for token in tokens:
        lang = _classify_token_language(token)
        if lang is None:
            if segments:
                segments[-1]['text'] += token
            else:
                pending_neutral += token
            continue
        if segments and segments[-1]['lang'] == lang:
            segments[-1]['text'] += pending_neutral + token
        else:
            segments.append({'lang': lang, 'text': pending_neutral + token})
        pending_neutral = ''

    if pending_neutral:
        if segments:
            segments[-1]['text'] += pending_neutral
        else:
            fallback_lang = 'ru' if CYRILLIC_RE.search(pending_neutral) else 'en'
            segments.append({'lang': fallback_lang, 'text': pending_neutral})

    merged: list[dict[str, str]] = []
    for segment in segments:
        seg_text = str(segment['text']).strip()
        seg_lang = str(segment['lang'])
        if not seg_text:
            continue
        if merged and merged[-1]['lang'] == seg_lang:
            merged[-1]['text'] = f"{merged[-1]['text']} {seg_text}".strip()
        else:
            merged.append({'lang': seg_lang, 'text': seg_text})
    return merged


def _pick_first_matching_voice(available: list[str], preferred_names: tuple[str, ...]) -> str | None:
    lookup = {voice.lower(): voice for voice in available}
    for name in preferred_names:
        voice = lookup.get(name.lower())
        if voice:
            return voice
    for voice in available:
        lower = voice.lower()
        if any(name.lower() in lower for name in preferred_names):
            return voice
    return None


def _guess_system_voice_for_language(lang: str) -> str | None:
    available = get_system_voices()
    if not available:
        return None
    if lang == 'ru':
        return _pick_first_matching_voice(available, KNOWN_RUSSIAN_SYSTEM_VOICES)
    return _pick_first_matching_voice(available, KNOWN_ENGLISH_SYSTEM_VOICES) or available[0]


def _select_system_voice_for_segment(lang: str, requested_voice: str) -> str:
    available = get_system_voices()
    normalized = normalize_system_voice(requested_voice)
    if normalized and normalized in available:
        if lang == 'en':
            return normalized
        lower_voice = normalized.lower()
        if any(name.lower() in lower_voice for name in KNOWN_RUSSIAN_SYSTEM_VOICES):
            return normalized
    guessed = _guess_system_voice_for_language(lang)
    if guessed:
        return guessed
    return normalized or 'default'


def _segment_engine_and_voice(base_engine: Literal['system', 'neural'], requested_voice: str, lang: str) -> tuple[Literal['system', 'neural'], str]:
    if lang == 'en':
        if base_engine == 'neural' and get_neural_voices():
            return 'neural', normalize_neural_voice(requested_voice)
        return 'system', _select_system_voice_for_segment('en', requested_voice)

    if get_system_voices():
        return 'system', _select_system_voice_for_segment('ru', requested_voice)

    if base_engine == 'neural' and get_neural_voices():
        logger.warning('No Russian system voice available; falling back to neural voice for Cyrillic text')
        return 'neural', normalize_neural_voice(requested_voice)

    return 'system', _select_system_voice_for_segment('ru', requested_voice)


def _synthesize_single_engine_to_file(text: str, engine: Literal['system', 'neural'], voice: str, rate: float) -> Path:
    if engine == 'system':
        return synthesize_with_system(text, voice, rate)
    if engine == 'neural':
        return synthesize_with_kokoro(text, voice, rate)
    raise ValueError(f'Unknown engine: {engine}')


def synthesize_mixed_language_to_file(text: str, engine: Literal['system', 'neural'], voice: str, rate: float) -> Path:
    output_path = _new_output_path('.wav')
    segment_paths: list[Path] = []
    try:
        segments = split_text_by_language(text)
        if not segments:
            raise RuntimeError('No text segments available for synthesis')
        rendered_chunks: list[np.ndarray] = []
        pause_samples = max(1, int(DEFAULT_SAMPLE_RATE * MIXED_SEGMENT_PAUSE_SEC))
        pause_chunk = np.zeros(pause_samples, dtype=np.float32)

        for index, segment in enumerate(segments):
            segment_engine, segment_voice = _segment_engine_and_voice(engine, voice, segment['lang'])
            path = _synthesize_single_engine_to_file(segment['text'], segment_engine, segment_voice, rate)
            segment_paths.append(path)
            rendered_chunks.append(_load_audio_mono(path, target_rate=DEFAULT_SAMPLE_RATE))
            if index < len(segments) - 1:
                rendered_chunks.append(pause_chunk)

        full_audio = np.concatenate([chunk for chunk in rendered_chunks if chunk.size > 0])
        if full_audio.size == 0:
            raise RuntimeError('Mixed-language synthesis produced empty audio')
        save_wav_pcm16(output_path, full_audio, DEFAULT_SAMPLE_RATE)
        return output_path
    finally:
        for path in segment_paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass


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
    cleaned = (text or '').strip()
    if _contains_ru_and_en(cleaned):
        return synthesize_mixed_language_to_file(cleaned, engine, voice, rate)
    return _synthesize_single_engine_to_file(cleaned, engine, voice, rate)
