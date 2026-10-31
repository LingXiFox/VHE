"""Pinyin syllable -> phoneme plan for the Mandarin physical synthesizer (P1-C).

This module is the *only* place where a pinyin spelling is turned into phoneme ids. It is
deliberately strict: a syllable is either rendered with every one of its phonemes or it is
blocked with a reason. Nothing is replaced by a vowel, by silence, or by a "close enough"
neighbour, because a plausible-sounding substitute silently teaches the model that the
missing consonant is a vowel.

Scope: the 26-phoneme subset in :mod:`vhe.synth.phonology`. Roughly half of the Mandarin
finals (ia, ie, ian, uo, uan, üe, ...) and six onsets (zh, ch, h, r, plus the y/w glide
spellings other than yi/wu) are outside it, so many real syllables are blocked. That is the
intended behaviour until the inventory grows.

Timing: the backend renders phonemes back-to-back with no silence slots, so the syllable
duration measured from the recording has to be split between the onset and the nucleus.
The split below is a fixed heuristic, not a forced alignment: onsets get their nominal
physical duration (closure + burst + VOT for stops, frication length for fricatives) and the
nucleus absorbs the remainder. Every allocation is returned so it can be audited and later
replaced by real forced-alignment boundaries.
"""

from __future__ import annotations

from typing import Optional

from ..synth.contract import INVENTORIES
from ..synth.phonology import MANDARIN_PHONEMES, PhonemeSpec

#: pinyin final -> phoneme id, only for finals that are one phoneme of the inventory.
PINYIN_FINAL_TO_PHONEME: dict[str, str] = {
    "a": "a", "i": "i", "u": "u",
    "ai": "ai", "ei": "ei", "ao": "au", "ou": "ou",
    "an": "an", "ang": "ang",
}

#: Onsets written with two letters that the inventory does have.
PINYIN_ONSET_TO_PHONEME: dict[str, str] = {
    "b": "b", "p": "p", "m": "m", "f": "f", "d": "d", "t": "t", "n": "n", "l": "l",
    "g": "g", "k": "k", "sh": "sh", "s": "s", "x": "x", "z": "z", "c": "c", "j": "j",
    "q": "q",
}

#: Zero-initial spellings that are a single vowel: pinyin writes the glide, there is none.
ZERO_INITIAL_SPELLINGS: dict[tuple[str, str], str] = {("y", "i"): "i", ("w", "u"): "u"}

#: Onsets whose "i" is the apical vowel [ɿ]/[ʅ], not /i/. Substituting /i/ would be a lie.
APICAL_ONSETS = ("z", "c", "s", "zh", "ch", "sh", "r")

#: Onsets after which a written "u" is really "ü" [y].
FRONT_ROUNDED_ONSETS = ("j", "q", "x")

#: A nucleus shorter than this is not a syllable any more.
NUCLEUS_MIN_S = 0.060
#: An onset shorter than this loses its closure/burst structure entirely.
ONSET_MIN_S = 0.020


def nominal_onset_s(phoneme: str) -> float:
    """Physical length of one onset consonant (closure + burst + VOT, or frication)."""
    spec: PhonemeSpec = MANDARIN_PHONEMES[phoneme]
    if spec.kind in ("stop", "affricate"):
        burst_s = spec.burst[2] if spec.burst is not None else 0.0
        return spec.closure_s + burst_s + spec.vot_s + 0.008
    if spec.kind == "fricative":
        return 0.080
    if spec.kind in ("nasal", "lateral"):
        return 0.060
    return 0.0


def syllable_phonemes(initial: str, final: str, inventory: str = "mandarin"
                      ) -> tuple[tuple[str, ...], Optional[str]]:
    """Phoneme ids for one pinyin syllable, or an empty tuple plus a blocking reason."""
    if inventory not in INVENTORIES:  # pragma: no cover - guarded by PhonemeSequence
        raise ValueError(f"unknown inventory {inventory!r}")
    if inventory == "vowels":
        if initial:
            return (), f"onset {initial!r} is not in the {inventory!r} inventory"
        if final not in INVENTORIES["vowels"]:
            return (), f"final {final!r} is not in the {inventory!r} inventory"
        return (final,), None

    if (initial, final) in ZERO_INITIAL_SPELLINGS:
        return (ZERO_INITIAL_SPELLINGS[(initial, final)],), None
    if initial and initial not in PINYIN_ONSET_TO_PHONEME:
        return (), f"onset {initial!r} is not in the {inventory!r} inventory"
    if final == "i" and initial in APICAL_ONSETS:
        return (), (f"pinyin {initial + final!r} spells the apical vowel, which is not in "
                    f"the {inventory!r} inventory")
    if initial in FRONT_ROUNDED_ONSETS and final.startswith("u"):
        return (), (f"pinyin {initial + final!r} spells ü, which is not in the "
                    f"{inventory!r} inventory")
    if final not in PINYIN_FINAL_TO_PHONEME:
        return (), f"final {final!r} is not in the {inventory!r} inventory"
    phonemes = (PINYIN_ONSET_TO_PHONEME[initial],) if initial else ()
    return phonemes + (PINYIN_FINAL_TO_PHONEME[final],), None


def syllable_plan(initial: str, final: str, duration_s: float, inventory: str = "mandarin"
                  ) -> tuple[tuple[tuple[str, float], ...], Optional[str]]:
    """((phoneme, seconds), ...) fitting one measured syllable duration, or a reason.

    The allocation is a heuristic: the onset takes its nominal physical length and the
    nucleus takes the rest. A syllable too short to hold both is blocked instead of being
    squeezed into a click-length consonant.
    """
    phonemes, reason = syllable_phonemes(initial, final, inventory)
    if reason is not None:
        return (), reason
    if len(phonemes) == 1:
        return ((phonemes[0], float(duration_s)),), None
    onset, nucleus = phonemes
    onset_s = nominal_onset_s(onset)
    available = float(duration_s)
    if available < ONSET_MIN_S + NUCLEUS_MIN_S:
        return (), (f"syllable of {available * 1000.0:.0f} ms cannot hold onset {onset!r} "
                    f"+ nucleus {nucleus!r}")
    onset_s = min(onset_s, available - NUCLEUS_MIN_S)
    onset_s = max(onset_s, ONSET_MIN_S)
    return ((onset, onset_s), (nucleus, available - onset_s)), None
