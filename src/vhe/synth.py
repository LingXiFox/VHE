"""Deterministic SOURCE-FILTER / FORMANT proxy baseline (NOT a full articulatory model).

This baseline deliberately remains mechanical; it supplies a reproducible input for
future VHE experiments. Do not present these generated sounds as neural humanization.
"""

from __future__ import annotations

from dataclasses import dataclass
import wave
from pathlib import Path

import numpy as np

# (F1, F2, F3) targets in Hz, chosen for *demo only*; not speaker-independent phonetics.
VOWELS = {
    "a": (780.0, 1250.0, 2700.0),
    "i": (300.0, 2250.0, 3050.0),
    "u": (330.0, 870.0, 2300.0),
}


@dataclass(frozen=True)
class VoiceTemplate:
    formant_scale: float = 1.0
    spectral_tilt: float = 1.30
    breathiness: float = 0.008
    vibrato_cents: float = 7.0
    vibrato_hz: float = 5.0

    def validate(self) -> None:
        if not 0.65 <= self.formant_scale <= 1.40:
            raise ValueError("formant_scale must be in [0.65,1.40]")
        if not 0.6 <= self.spectral_tilt <= 3.0:
            raise ValueError("spectral_tilt must be in [0.6,3.0]")
        if not 0.0 <= self.breathiness <= 0.1:
            raise ValueError("breathiness must be in [0,0.1]")
        if not 0.0 <= self.vibrato_cents <= 100.0:
            raise ValueError("vibrato_cents must be in [0,100]")
        if not 0.0 <= self.vibrato_hz <= 10.0:
            raise ValueError("vibrato_hz must be in [0,10]")


PRESETS = {
    "yang": VoiceTemplate(formant_scale=1.08, spectral_tilt=1.20, breathiness=0.012),
    "yin": VoiceTemplate(formant_scale=0.87, spectral_tilt=1.50, breathiness=0.006),
}


def synthesize(
    phonemes: list[str] | tuple[str, ...],
    *,
    f0_hz: float = 220.0,
    duration_s: float = 1.2,
    sample_rate: int = 24000,
    template: VoiceTemplate | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Synthesize vowel-only float32 PCM, shape [N], deterministic with seed.

    Initial proof-of-wiring only. No Mandarin consonants, tones, or true MIDI mapping.
    """
    template = template or PRESETS["yang"]
    template.validate()
    if not phonemes or any(p not in VOWELS for p in phonemes):
        raise ValueError("MVP accepts only vowel phonemes a,i,u. Do not silently skip others.")
    if not 65 <= f0_hz <= 900:
        raise ValueError("f0_hz must be 65..900")
    if not 0.10 <= duration_s <= 10:
        raise ValueError("duration_s must be 0.1..10")
    if sample_rate not in (16000, 24000, 48000):
        raise ValueError("sample_rate must be 16000, 24000 or 48000")

    n = round(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    # Interpolate between vowel centers continuously rather than switching abruptly.
    midpoints = (np.arange(len(phonemes)) + 0.5) * duration_s / len(phonemes)
    target = np.asarray([VOWELS[p] for p in phonemes], dtype=np.float64)
    formants = [
        np.interp(t, midpoints, target[:, j]) * template.formant_scale
        for j in range(3)
    ]

    vib = template.vibrato_cents * np.sin(2 * np.pi * template.vibrato_hz * t)
    f0 = f0_hz * np.power(2.0, vib / 1200.0)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    signal = np.zeros(n, dtype=np.float64)
    max_harmonic = int((sample_rate * 0.43) // (f0_hz * 1.06))
    bandwidths = (110., 150., 210.)
    weights = (1.0, 0.72, 0.38)

    # Constant baseline harmonic oscillator, frequency-domain formant emphasis.
    for harmonic in range(1, max_harmonic + 1):
        freq = harmonic * f0
        formant_gain = 0.008 + sum(
            w * np.exp(-0.5 * ((freq - fj) / bw) ** 2)
            for fj, bw, w in zip(formants, bandwidths, weights)
        )
        amplitude = formant_gain * harmonic ** (-template.spectral_tilt)
        signal += amplitude * np.sin(harmonic * phase)

    rng = np.random.default_rng(seed)
    signal += template.breathiness * rng.standard_normal(n)
    fade = max(1, int(0.03 * sample_rate))
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    signal *= envelope
    peak = np.max(np.abs(signal))
    if peak <= 1e-10 or not np.isfinite(peak):
        raise RuntimeError("Invalid/silent synthesis output")
    return (0.85 * signal / peak).astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write 16-bit PCM WAV. Refuses overwrite to protect experimental records."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
