from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def save_wav_pcm16(file_path: str | Path, audio_data: np.ndarray, sample_rate: int) -> None:
    """Save audio to a browser-friendly WAV PCM_16 file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(audio_data)
    if data.size == 0:
        raise ValueError("audio_data is empty")

    if data.ndim > 1:
        data = data[:, 0]

    if data.dtype.kind == "f":
        data = np.clip(data, -1.0, 1.0).astype(np.float32)

    sf.write(str(path), data, sample_rate, format="WAV", subtype="PCM_16")
