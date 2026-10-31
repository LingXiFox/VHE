"""Mono PCM I/O helpers shared by synthesis, data adapters and evaluation.

Only 16-bit PCM WAV is supported on read: that is what the mechanical baseline writes
and what AISHELL-3 ships. Anything else fails loudly instead of being coerced.
"""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit PCM WAV. Refuses overwrite to protect experimental records."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    # Quantise in float32 exactly as the v0.0.1 baseline did: converting through
    # float64 first shifts 13/28800 samples by 1 LSB and breaks historical WAV hashes.
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("write_wav expects mono audio with shape [N]")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV as float32 in [-1, 1].

    Returns (audio[N], sample_rate). Channels/sample-width mismatches raise instead of
    being silently averaged or rescaled: corpus statistics must stay comparable.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1:
        raise ValueError(f"{path}: expected mono, found {channels} channels")
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, found {width * 8}-bit")
    pcm = np.frombuffer(frames, dtype="<i2")
    if pcm.size == 0:
        raise ValueError(f"{path}: empty PCM payload")
    return (pcm.astype(np.float32) / 32768.0), int(rate)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    """Scale to target peak; raises on silence instead of returning zeros."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-10 or not np.isfinite(peak):
        raise RuntimeError("Invalid/silent audio: cannot peak-normalize")
    return (target_peak * audio / peak).astype(np.float32)
