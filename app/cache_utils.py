from pathlib import Path
import hashlib

from app.config import CACHE_DIR


def build_cache_key(text: str, engine: str, voice: str, rate: float) -> str:
    normalized_rate = f'{float(rate):.4f}'
    raw = f'{text}|{engine}|{voice}|{normalized_rate}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def get_cached_file_path(cache_key: str, suffix: str) -> Path:
    return CACHE_DIR / f'{cache_key}{suffix}'


def has_cached_file(cache_key: str, suffix: str) -> bool:
    return get_cached_file_path(cache_key, suffix).exists()
