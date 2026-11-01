"""Pinyin syllable -> phoneme plan for the Mandarin physical synthesizer (P1-C/P1.3).

This module is the only place where a pinyin spelling is turned into phoneme ids. It is
deliberately strict: a syllable is either rendered with every one of its phonemes or it is
blocked with a reason. Nothing is replaced by a vowel, by silence, or by a 'close enough'
neighbour, because a plausible-sounding substitute silently teaches the model that the
missing consonant is a vowel.

Two inputs exist and they must not be confused:

* The orthographic (initial, final) pair as written in the annotation ('y' + 'i', 'j' +
  'u', 'zh' + 'i', 'l' + 'iu'). Resolution to the underlying final lives in
  vhe.data.mandarin and is shared with the corpus scanner, so coverage statistics and the
  synthesizer can never disagree about what a token means.
* The synthesis inventory: the 26 phonemes in vhe.synth.phonology. Finals are units
  ('an', 'ai', ...) with their own formant path; onsets are single consonants.

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
from .mandarin import ONSETS, PinyinError, Token, resolve_final, split_token

#: Underlying final id -> synthesizer phoneme id. Only finals that are one unit there.
SYNTH_FINAL_TO_PHONEME: dict[str, str] = {
    "a": "a", "i": "i", "u": "u", "e": "e",
    "ai": "ai", "ei": "ei", "ao": "au", "ou": "ou",
    "an": "an", "ang": "ang",
    "ian": "ian", "ing": "ing", "iou": "iou",
}

#: Onsets the synthesizer can render, derived from the inventory itself (no second list).
SUPPORTED_ONSETS: frozenset[str] = frozenset(onset for onset in ONSETS
                                             if onset in MANDARIN_PHONEMES)
SUPPORTED_FINALS: frozenset[str] = frozenset(SYNTH_FINAL_TO_PHONEME)

#: Onsets whose 'i' is the apical vowel [ɿ]/[ʅ], not /i/. Substituting /i/ would be a lie.
APICAL_ONSETS = ("z", "c", "s", "zh", "ch", "sh", "r")

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


def resolve(initial: str, final: str) -> tuple[Optional[Token], Optional[str]]:
    """(token, None) or (None, blocking reason) for one orthographic syllable."""
    try:
        token = split_token(f"{initial}{final}1")
    except PinyinError as error:
        return None, (f"pinyin {initial + final!r} is not a Mandarin syllable: "
                      f"{error.detail}")
    return token, None


def syllable_phonemes(initial: str, final: str, inventory: str = "mandarin", *,
                      erhua: bool = False) -> tuple[tuple[str, ...], Optional[str]]:
    """Phoneme ids for one pinyin syllable, or an empty tuple plus a blocking reason."""
    if inventory not in INVENTORIES:  # pragma: no cover - guarded by PhonemeSequence
        raise ValueError(f"unknown inventory {inventory!r}")
    if inventory == "vowels":
        if initial:
            return (), f"onset {initial!r} is not in the {inventory!r} inventory"
        if final not in INVENTORIES["vowels"]:
            return (), f"final {final!r} is not in the {inventory!r} inventory"
        return (final,), None

    token, reason = resolve(initial, final)
    if token is None:
        return (), reason
    token = Token(token.pinyin, token.initial, token.final, token.tone, erhua)
    if erhua:
        return (), (f"pinyin {initial + final!r} is erhua (儿化): the final is modified, "
                    f"not followed by /r/, and the {inventory!r} inventory has no erhua unit")
    # y/w are orthographic glides, never onsets: 'yi' is /i/, 'wu' is /u/.
    if initial and initial not in ("y", "w") and initial not in SUPPORTED_ONSETS:
        return (), f"onset {initial!r} is not in the {inventory!r} inventory"
    if token.final.startswith("apical_"):
        symbol = "[ɿ]" if token.final == "apical_dental" else "[ʅ]"
        return (), (f"pinyin {initial + final!r} spells the apical vowel {symbol}, which is "
                    f"not in the {inventory!r} inventory")
    if token.final.startswith("v"):
        return (), (f"pinyin {initial + final!r} spells ü, which is not in the "
                    f"{inventory!r} inventory")
    if token.final not in SYNTH_FINAL_TO_PHONEME:
        return (), (f"pinyin {initial + final!r} needs the final {token.final!r}, which is "
                    f"not in the {inventory!r} inventory")
    phonemes = (initial,) if initial and initial not in ("y", "w") else ()
    return phonemes + (SYNTH_FINAL_TO_PHONEME[token.final],), None


def syllable_plan(initial: str, final: str, duration_s: float, inventory: str = "mandarin",
                  *, erhua: bool = False) -> tuple[tuple[tuple[str, float], ...], Optional[str]]:
    """((phoneme, seconds), ...) fitting one measured syllable duration, or a reason.

    The allocation is a heuristic: the onset takes its nominal physical length and the
    nucleus takes the rest. A syllable too short to hold both is blocked instead of being
    squeezed into a click-length consonant.
    """
    phonemes, reason = syllable_phonemes(initial, final, inventory, erhua=erhua)
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
