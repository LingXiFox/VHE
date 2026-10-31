"""Mechanical synthesizer package (upstream of VHE).

Public API
----------
Contract types : VoiceTemplate, PhonemeSegment, PhonemeSequence, PitchCurve, Timing,
                 SynthesisRequest, SynthesisResult, SynthBackend, PHONEME_INVENTORY
Backends       : get_backend(), list_backends(), FormantProxySynth, MandarinFormantSynth
Phonology      : MANDARIN_PHONEMES, tone_curve_points(), INVENTORIES
Legacy wrapper : synthesize(...) renders a vowel sequence with equal segment lengths
I/O            : write_wav()
"""

from __future__ import annotations

import numpy as np

from ..audio import write_wav
from .contract import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    INVENTORIES,
    PHONEME_INVENTORY,
    PRESETS,
    PitchCurve,
    PhonemeSegment,
    PhonemeSequence,
    SynthesisRequest,
    SynthesisResult,
    SynthBackend,
    Timing,
    VoiceTemplate,
    inventory_error,
    supported_phonemes,
)
from .formant_proxy import FormantProxySynth
from .mandarin import MandarinFormantSynth
from .phonology import (MANDARIN_PHONEMES, MANDARIN_PHONEME_IDS, TONES, PhonemeSpec,
                        tone_curve_points, tone_offsets)
from .registry import DEFAULT_BACKEND, get_backend, list_backends

__all__ = [
    "DEFAULT_BACKEND", "F0_MAX_HZ", "F0_MIN_HZ", "INVENTORIES", "MANDARIN_PHONEMES",
    "MANDARIN_PHONEME_IDS", "PHONEME_INVENTORY", "PRESETS", "TONES",
    "FormantProxySynth", "MandarinFormantSynth", "PhonemeSpec", "PitchCurve",
    "PhonemeSegment", "PhonemeSequence", "SynthesisRequest", "SynthesisResult",
    "SynthBackend", "Timing", "VoiceTemplate", "get_backend", "inventory_error",
    "list_backends", "supported_phonemes", "synthesize", "tone_curve_points",
    "tone_offsets", "write_wav",
]


def synthesize(
    phonemes: list[str] | tuple[str, ...],
    *,
    f0_hz: float = 220.0,
    duration_s: float = 1.2,
    sample_rate: int = 24000,
    template: VoiceTemplate | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Render equal-length vowels to a float32 [N] waveform (legacy convenience API).

    Kept for the existing CLI/tests. New code should build a SynthesisRequest so that
    per-segment durations, the f0 contour and silence are explicit in the record.
    """
    if not phonemes:
        raise ValueError(inventory_error())
    if not 0.10 <= duration_s <= 10:
        raise ValueError("duration_s must be 0.1..10")
    segment_duration = duration_s / len(phonemes)
    request = SynthesisRequest(
        template=template or PRESETS["yang"],
        phonemes=PhonemeSequence(
            tuple(PhonemeSegment(str(p), segment_duration) for p in phonemes)
        ),
        pitch=PitchCurve.constant(f0_hz, duration_s),
        timing=Timing(),
        sample_rate=sample_rate,
        seed=seed,
    )
    result = get_backend().render(request)
    return result.audio
