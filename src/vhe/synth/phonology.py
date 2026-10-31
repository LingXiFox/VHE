"""Mandarin phonology data for the articulatory backend.

Pure data + small helpers, no audio code, no dependency on the synth contract (the
contract imports this module to know which phoneme inventories exist).

Scope: a deliberately LIMITED but representative inventory. Everything here is an
idealised articulatory target, not a measurement: formant values are adult-male
placeholders in the ballpark of published Mandarin averages, and the noise bands are
place-of-articulation heuristics. Absolute values do not matter much for VHE - what
matters is that phonemes differ in a physically motivated way and that the differences
are stable, so the model sees real phonetic structure instead of a vowel-only world.

Naming: ASCII pinyin initials/finals ("b", "p", "m", "a", "ai", "an", "ang"). Tones are
NOT part of the phoneme id: a tone becomes an F0 contour (PitchCurve) upstream, which is
why VHE never has to see tone labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["vowel", "diphthong", "nasal", "lateral", "stop", "fricative", "affricate"]

# Formant triple in Hz: (F1, F2, F3).
Formants = tuple[float, float, float]


@dataclass(frozen=True)
class PhonemeSpec:
    """One segment type: where the articulators are and what the source does."""

    phoneme: str
    kind: Kind
    voiced: bool
    # Formant targets over the segment, (fraction_of_segment, F1, F2, F3), sorted by fraction.
    targets: tuple[tuple[float, float, float, float], ...]
    # Frication/aspiration band (low_hz, high_hz) and its relative gain vs the vowel body.
    noise_band: tuple[float, float] | None = None
    noise_gain: float = 0.0
    # Stop/affricate release burst: (centre_hz, bandwidth_hz, duration_s, gain).
    burst: tuple[float, float, float, float] | None = None
    closure_s: float = 0.0        # silent closure before the release
    vot_s: float = 0.0            # aspiration after the release (voice onset time)
    nasal_from: float | None = None  # nasal resonance ramps in from this fraction

    def targets_at(self, fraction: float) -> Formants:
        """Formant triple at a normalised position inside the segment."""
        fraction = min(max(fraction, 0.0), 1.0)
        points = self.targets
        if fraction <= points[0][0]:
            return (points[0][1], points[0][2], points[0][3])
        for (t0, *f0), (t1, *f1) in zip(points, points[1:]):
            if fraction <= t1:
                span = t1 - t0
                weight = 0.0 if span <= 0 else (fraction - t0) / span
                return (
                    f0[0] + (f1[0] - f0[0]) * weight,
                    f0[1] + (f1[1] - f0[1]) * weight,
                    f0[2] + (f1[2] - f0[2]) * weight,
                )
        return (points[-1][1], points[-1][2], points[-1][3])


def _vowel(phoneme: str, f1: float, f2: float, f3: float) -> PhonemeSpec:
    return PhonemeSpec(phoneme, "vowel", True, ((0.0, f1, f2, f3),))


MANDARIN_PHONEMES: dict[str, PhonemeSpec] = {
    # --- finals (vowels / diphthongs / nasal codas) ---------------------------------
    "a": _vowel("a", 800.0, 1250.0, 2600.0),
    "i": _vowel("i", 300.0, 2250.0, 3050.0),
    "u": _vowel("u", 330.0, 870.0, 2300.0),
    "ai": PhonemeSpec("ai", "diphthong", True, (
        (0.0, 800.0, 1250.0, 2600.0),
        (0.5, 600.0, 1700.0, 2750.0),
        (1.0, 350.0, 2150.0, 2950.0),
    )),
    "ei": PhonemeSpec("ei", "diphthong", True, (
        (0.0, 550.0, 1900.0, 2600.0),
        (0.5, 420.0, 2050.0, 2800.0),
        (1.0, 330.0, 2200.0, 3000.0),
    )),
    "au": PhonemeSpec("au", "diphthong", True, (
        (0.0, 800.0, 1250.0, 2600.0),
        (0.5, 600.0, 950.0, 2500.0),
        (1.0, 350.0, 850.0, 2300.0),
    )),
    "ou": PhonemeSpec("ou", "diphthong", True, (
        (0.0, 550.0, 1000.0, 2500.0),
        (0.5, 450.0, 900.0, 2400.0),
        (1.0, 330.0, 870.0, 2300.0),
    )),
    "an": PhonemeSpec("an", "nasal", True, (
        (0.0, 800.0, 1250.0, 2600.0),
        (0.6, 750.0, 1400.0, 2600.0),
        (1.0, 500.0, 1300.0, 2500.0),
    ), nasal_from=0.62),
    "ang": PhonemeSpec("ang", "nasal", True, (
        (0.0, 800.0, 1250.0, 2600.0),
        (0.6, 750.0, 1000.0, 2500.0),
        (1.0, 600.0, 900.0, 2400.0),
    ), nasal_from=0.55),
    # --- sonorant initials ----------------------------------------------------------
    "m": PhonemeSpec("m", "nasal", True, ((0.0, 250.0, 1000.0, 2200.0),), nasal_from=0.0),
    "n": PhonemeSpec("n", "nasal", True, ((0.0, 250.0, 1600.0, 2600.0),), nasal_from=0.0),
    "l": PhonemeSpec("l", "lateral", True, ((0.0, 350.0, 1100.0, 2600.0),)),
    # --- stops: closure + release burst (+ aspiration for the aspirated series) ------
    "b": PhonemeSpec("b", "stop", False, ((0.0, 300.0, 900.0, 2200.0),),
                     burst=(700.0, 600.0, 0.006, 1.4), closure_s=0.030, vot_s=0.012),
    "p": PhonemeSpec("p", "stop", False, ((0.0, 300.0, 900.0, 2200.0),),
                     noise_band=(600.0, 4000.0), noise_gain=0.22,
                     burst=(700.0, 600.0, 0.006, 1.4), closure_s=0.030, vot_s=0.060),
    "d": PhonemeSpec("d", "stop", False, ((0.0, 300.0, 1700.0, 2600.0),),
                     burst=(3000.0, 1500.0, 0.006, 1.3), closure_s=0.030, vot_s=0.012),
    "t": PhonemeSpec("t", "stop", False, ((0.0, 300.0, 1700.0, 2600.0),),
                     noise_band=(2500.0, 8000.0), noise_gain=0.22,
                     burst=(3000.0, 1500.0, 0.006, 1.2), closure_s=0.030, vot_s=0.060),
    "g": PhonemeSpec("g", "stop", False, ((0.0, 250.0, 2000.0, 2600.0),),
                     burst=(2000.0, 1200.0, 0.006, 1.2), closure_s=0.030, vot_s=0.014),
    "k": PhonemeSpec("k", "stop", False, ((0.0, 250.0, 2000.0, 2600.0),),
                     noise_band=(1500.0, 6000.0), noise_gain=0.22,
                     burst=(2000.0, 1200.0, 0.006, 1.1), closure_s=0.030, vot_s=0.065),
    # --- fricatives: noise only -----------------------------------------------------
    "f": PhonemeSpec("f", "fricative", False, ((0.0, 400.0, 1200.0, 2400.0),),
                     noise_band=(1500.0, 8000.0), noise_gain=0.14),
    "s": PhonemeSpec("s", "fricative", False, ((0.0, 320.0, 1800.0, 2600.0),),
                     noise_band=(4000.0, 10000.0), noise_gain=0.34),
    "sh": PhonemeSpec("sh", "fricative", False, ((0.0, 320.0, 1600.0, 2500.0),),
                      noise_band=(2000.0, 6000.0), noise_gain=0.30),
    "x": PhonemeSpec("x", "fricative", False, ((0.0, 320.0, 1700.0, 2500.0),),
                     noise_band=(3000.0, 8000.0), noise_gain=0.28),
    # --- affricates: closure + burst + frication (+ aspiration) ---------------------
    "z": PhonemeSpec("z", "affricate", False, ((0.0, 320.0, 1800.0, 2600.0),),
                     noise_band=(4000.0, 10000.0), noise_gain=0.34,
                     burst=(5000.0, 2500.0, 0.004, 0.9), closure_s=0.035, vot_s=0.030),
    "c": PhonemeSpec("c", "affricate", False, ((0.0, 320.0, 1800.0, 2600.0),),
                     noise_band=(4000.0, 10000.0), noise_gain=0.34,
                     burst=(5000.0, 2500.0, 0.004, 0.9), closure_s=0.035, vot_s=0.070),
    "j": PhonemeSpec("j", "affricate", False, ((0.0, 300.0, 2000.0, 2700.0),),
                     noise_band=(3000.0, 8000.0), noise_gain=0.28,
                     burst=(4000.0, 2000.0, 0.004, 0.8), closure_s=0.030, vot_s=0.030),
    "q": PhonemeSpec("q", "affricate", False, ((0.0, 300.0, 2000.0, 2700.0),),
                     noise_band=(3000.0, 8000.0), noise_gain=0.28,
                     burst=(4000.0, 2000.0, 0.004, 0.8), closure_s=0.030, vot_s=0.070),
}

MANDARIN_PHONEME_IDS: tuple[str, ...] = tuple(MANDARIN_PHONEMES)

TONES = ("1", "2", "3", "4", "5")  # 5 = neutral tone (short, mid level)

# Chao-scale (1..5) contours as (fraction, level) control points. Tone 3 is the full
# dipping 214, not the half-third-tone sandhi variant: sandhi is a prosody rule above
# the segment layer and is out of scope here.
TONE_CONTOURS: dict[str, tuple[tuple[float, float], ...]] = {
    "1": ((0.0, 5.0), (1.0, 5.0)),
    "2": ((0.0, 3.0), (0.5, 3.6), (1.0, 5.0)),
    "3": ((0.0, 2.0), (0.35, 1.0), (0.75, 2.6), (1.0, 4.0)),
    "4": ((0.0, 5.0), (0.7, 1.6), (1.0, 1.0)),
    "5": ((0.0, 3.0), (1.0, 3.0)),
}

SEMITONES_PER_LEVEL = 1.5


def tone_offsets(tone: str) -> tuple[tuple[float, float], ...]:
    """(fraction, semitone offset from the syllable mid level) for one tone."""
    try:
        contour = TONE_CONTOURS[tone]
    except KeyError:
        raise ValueError(f"unknown tone {tone!r}; expected one of {TONES}") from None
    pivot = 3.0
    return tuple((fraction, (level - pivot) * SEMITONES_PER_LEVEL) for fraction, level in contour)


def tone_curve_points(tones: tuple[str, ...], durations_s: tuple[float, ...]) -> tuple[
        tuple[float, ...], tuple[float, ...]]:
    """Control points (times_s, semitone offsets) for a syllable sequence.

    The curve is continuous across syllables: each syllable contributes its own contour
    stretched over its duration, and the junction value is the mean of the two neighbours
    so that the pitch track does not jump between syllables.
    """
    if len(tones) != len(durations_s):
        raise ValueError("tones and durations_s must have the same length")
    if not tones:
        raise ValueError("at least one syllable is required")
    times: list[float] = []
    values: list[float] = []
    start = 0.0
    previous_end: float | None = None
    for tone, duration in zip(tones, durations_s):
        if duration <= 0.0:
            raise ValueError("syllable duration must be positive")
        points = tone_offsets(tone)
        for index, (fraction, offset) in enumerate(points):
            moment = start + fraction * duration
            if index == 0 and previous_end is not None:
                # Blend the junction instead of stepping: half-way between the previous
                # syllable's final offset and this syllable's initial offset.
                values.append(0.5 * (previous_end + offset))
            else:
                values.append(offset)
            times.append(moment)
        previous_end = points[-1][1]
        start += duration
    return tuple(times), tuple(values)
