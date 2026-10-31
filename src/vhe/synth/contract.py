"""Synthesizer contract: the only legitimate producer of VHE input audio.

Data flow (see docs/SYNTH_CONTRACT.md):

    VoiceTemplate + PhonemeSequence + PitchCurve + Timing
        -> SynthBackend.render() -> SynthesisResult
        -> x_mechanical: float32 [N] mono, [-1, 1], sample_rate Hz

Units are explicit and uniform:
  * duration / silence / pitch times: seconds (float)
  * f0: Hz (never MIDI note numbers, never semitones)
  * sample_rate: Hz, one of SUPPORTED_SAMPLE_RATES
  * audio: float32 in [-1, 1], shape [N]; VHE receives [B, 1, N]

The synthesizer may consume a voice template (timbre parameters), phonemes and
pitch/timing derived from an authorised recording. VHE itself receives only the
rendered waveform: no text, no phoneme labels, no speaker identity, no reference audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Protocol, runtime_checkable

import numpy as np

SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (16000, 24000, 48000)
F0_MIN_HZ: float = 65.0
F0_MAX_HZ: float = 900.0
MIN_VOICED_SAMPLES: int = 64
MAX_DURATION_S: float = 60.0
MAX_SILENCE_S: float = 5.0


@dataclass(frozen=True)
class VowelTarget:
    """Formant targets in Hz. Demo-grade placeholders, NOT speaker-independent phonetics."""

    f1_hz: float
    f2_hz: float
    f3_hz: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.f1_hz, self.f2_hz, self.f3_hz)


# MVP inventory: three vowels only. Adding phonemes requires new tests and a new
# backend version; silently substituting a different vowel changes speech content.
PHONEME_INVENTORY: dict[str, VowelTarget] = {
    "a": VowelTarget(780.0, 1250.0, 2700.0),
    "i": VowelTarget(300.0, 2250.0, 3050.0),
    "u": VowelTarget(330.0, 870.0, 2300.0),
}


def supported_phonemes() -> tuple[str, ...]:
    return tuple(PHONEME_INVENTORY)


def inventory_error(phoneme: str | None = None) -> str:
    names = ",".join(PHONEME_INVENTORY)
    prefix = "No phonemes supplied. " if phoneme is None else f"Unsupported phoneme {phoneme!r}. "
    return prefix + f"MVP accepts only vowel phonemes {names}. Do not silently skip others."


@dataclass(frozen=True)
class VoiceTemplate:
    """Timbre controls of the upstream synthesizer (not VHE input).

    formant_scale : unitless multiplier on formant frequencies, 0.65 .. 1.40
    spectral_tilt: harmonic amplitude exponent, 0.6 .. 3.0 (larger = darker)
    breathiness  : white-noise gain, 0 .. 0.1
    vibrato_cents: peak pitch deviation in cents, 0 .. 100
    vibrato_hz   : vibrato rate in Hz, 0 .. 10
    """

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


PRESETS: dict[str, VoiceTemplate] = {
    "yang": VoiceTemplate(formant_scale=1.08, spectral_tilt=1.20, breathiness=0.012),
    "yin": VoiceTemplate(formant_scale=0.87, spectral_tilt=1.50, breathiness=0.006),
}


@dataclass(frozen=True)
class PhonemeSegment:
    """One phoneme held for duration_s seconds. duration_s is in seconds, > 0."""

    phoneme: str
    duration_s: float

    def validate(self) -> None:
        if self.phoneme not in PHONEME_INVENTORY:
            raise ValueError(inventory_error(self.phoneme))
        if not 0.01 <= self.duration_s <= MAX_DURATION_S:
            raise ValueError(f"segment duration_s must be 0.01..{MAX_DURATION_S}")


@dataclass(frozen=True)
class PhonemeSequence:
    """Ordered phoneme timeline; total voiced duration = sum of segment durations."""

    segments: tuple[PhonemeSegment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def duration_s(self) -> float:
        return float(sum(segment.duration_s for segment in self.segments))

    def validate(self) -> None:
        if not self.segments:
            raise ValueError(inventory_error())
        for segment in self.segments:
            segment.validate()
        if self.duration_s > MAX_DURATION_S:
            raise ValueError(f"total phoneme duration exceeds {MAX_DURATION_S} s")


@dataclass(frozen=True)
class PitchCurve:
    """Piecewise-linear f0 contour: times_s (seconds, strictly increasing) -> f0_hz (Hz).

    Values outside the control-point range are clamped, never extrapolated.
    Vibrato is applied downstream by the backend, so this curve is the base contour.
    """

    times_s: tuple[float, ...]
    f0_hz: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "times_s", tuple(float(v) for v in self.times_s))
        object.__setattr__(self, "f0_hz", tuple(float(v) for v in self.f0_hz))

    def validate(self) -> None:
        if not self.times_s or len(self.times_s) != len(self.f0_hz):
            raise ValueError("PitchCurve needs equal, non-empty times_s and f0_hz")
        if self.times_s[0] < 0.0:
            raise ValueError("PitchCurve times_s must be >= 0")
        for previous, current in zip(self.times_s, self.times_s[1:]):
            if current <= previous:
                raise ValueError("PitchCurve times_s must be strictly increasing")
        for value in self.f0_hz:
            if not F0_MIN_HZ <= value <= F0_MAX_HZ:
                raise ValueError(f"f0_hz must be in [{F0_MIN_HZ},{F0_MAX_HZ}] Hz")

    def sample(self, n_samples: int, sample_rate: int) -> np.ndarray:
        """Return f0 in Hz for n_samples at sample_rate (log-domain interpolation, clamped).

        Interpolating in Hz makes a rising glide spend most of its time in the top of the
        range: 100 -> 400 Hz reaches 250 Hz at the midpoint, 386 cents above the geometric
        mean a listener hears. Interpolating log2(f0) keeps the glide linear in pitch, and a
        constant curve is bit-identical either way (so the byte baseline is unaffected).
        """
        self.validate()
        times = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
        octaves = np.log2(np.asarray(self.f0_hz, dtype=np.float64))
        return np.power(2.0, np.interp(times, np.asarray(self.times_s), octaves))

    @classmethod
    def constant(cls, f0_hz: float, duration_s: float) -> "PitchCurve":
        return cls(times_s=(0.0, float(duration_s)), f0_hz=(float(f0_hz), float(f0_hz)))


@dataclass(frozen=True)
class Timing:
    """Silence around the voiced region, in seconds. Silence is not phoneme content."""

    lead_silence_s: float = 0.0
    tail_silence_s: float = 0.0

    def validate(self) -> None:
        for name, value in (("lead_silence_s", self.lead_silence_s),
                            ("tail_silence_s", self.tail_silence_s)):
            if not 0.0 <= value <= MAX_SILENCE_S:
                raise ValueError(f"{name} must be in [0,{MAX_SILENCE_S}] s")


@dataclass(frozen=True)
class SynthesisRequest:
    """Everything the upstream synthesizer may use. Note the absence of text,
    speaker identity, MIDI and reference audio: those never reach VHE."""

    template: VoiceTemplate
    phonemes: PhonemeSequence
    pitch: PitchCurve
    timing: Timing = field(default_factory=Timing)
    sample_rate: int = 24000
    seed: int = 42

    def validate(self) -> None:
        self.template.validate()
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}")
        self.phonemes.validate()
        self.timing.validate()
        self.pitch.validate()
        if self.pitch.times_s[-1] < self.voiced_duration_s:
            raise ValueError("PitchCurve does not cover the voiced region")
        if round(self.voiced_duration_s * self.sample_rate) < MIN_VOICED_SAMPLES:
            raise ValueError(f"voiced region must have at least {MIN_VOICED_SAMPLES} samples")

    @property
    def voiced_duration_s(self) -> float:
        return self.phonemes.duration_s

    @property
    def total_duration_s(self) -> float:
        return self.voiced_duration_s + self.timing.lead_silence_s + self.timing.tail_silence_s

    def fingerprint(self) -> str:
        """Stable sha256 of the request parameters (for manifests, not audio)."""
        payload: dict[str, Any] = {
            "template": {
                "formant_scale": self.template.formant_scale,
                "spectral_tilt": self.template.spectral_tilt,
                "breathiness": self.template.breathiness,
                "vibrato_cents": self.template.vibrato_cents,
                "vibrato_hz": self.template.vibrato_hz,
            },
            "phonemes": [[s.phoneme, s.duration_s] for s in self.phonemes.segments],
            "pitch": {"times_s": list(self.pitch.times_s), "f0_hz": list(self.pitch.f0_hz)},
            "timing": {
                "lead_silence_s": self.timing.lead_silence_s,
                "tail_silence_s": self.timing.tail_silence_s,
            },
            "sample_rate": self.sample_rate,
            "seed": self.seed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SynthesisResult:
    """Rendered mechanical audio plus provenance. audio is float32 [N] in [-1, 1]."""

    audio: np.ndarray
    sample_rate: int
    backend_name: str
    backend_version: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio)
        if audio.dtype != np.float32:
            raise TypeError("SynthesisResult.audio must be float32")
        if audio.ndim != 1:
            raise ValueError("SynthesisResult.audio must be mono [N]")
        if not np.isfinite(audio).all():
            raise ValueError("SynthesisResult.audio contains non-finite samples")
        object.__setattr__(self, "audio", audio)

    def as_batch(self) -> np.ndarray:
        """VHE input layout: [1, 1, N] float32 (no normalisation, no metadata)."""
        return self.audio.reshape(1, 1, -1)


@runtime_checkable
class SynthBackend(Protocol):
    """Pluggable upstream synthesizer.

    Implementations must be deterministic for a fixed request (same bytes on the
    same platform and version), must raise on unsupported input, and must never
    read reference recordings or external state.
    """

    name: str
    version: str
    supported_phonemes: tuple[str, ...]

    def render(self, request: SynthesisRequest) -> SynthesisResult:
        ...
