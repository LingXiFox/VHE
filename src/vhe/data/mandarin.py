"""Mandarin phone inventory and pinyin orthography resolution (P1.3).

Two layers are kept explicitly separate, because conflating them is exactly how a
pinyin-to-phoneme table starts lying:

* Synthesis units - what the physical synthesizer can render: one onset phoneme plus one
  *final unit* (a monophthong, diphthong or nasal final with its own formant path).
  vhe.data.g2p maps onto that limited inventory and blocks what is missing.
* Phonemic segments - the underlying phones of a final (medial + nucleus + coda), used for
  corpus statistics and for the next inventory expansion. They are phonemic (/ian/ = j + a
  + n), not narrow phonetic values; the synthesizer's final units are the allophonic
  realisations.

The orthography layer resolves what pinyin actually writes:

* y/w are not phonemes. 'yi' is /i/, 'wu' is /u/, 'yu' is /y/, 'ya' is /ia/, 'wei' is
  /uei/, 'wen' is /uen/.
* j q x + u spells the high front rounded vowel (ju = jü, xue = xüe, juan = jüan, jun =
  jün); n l write the umlaut as v in AISHELL-3 (nv, lve).
* iou/uei/uen are abbreviated to iu/ui/un after a real onset.
* i after z c s is the dental apical vowel [ɿ], after zh ch sh r the retroflex apical
  [ʅ]; neither is /i/.
* bo po mo fo are /puo pʰuo muo fuo/ in standard Mandarin; the written o after a labial is
  /uo/.
* Erhua is written as a trailing r (bar, wanr). It is a change of the final, not a coda
  /r/, so it is flagged and never silently rendered as base + /r/.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The 21 Mandarin onsets, in pinyin spelling. y/w are orthographic, not onsets.
ONSETS: tuple[str, ...] = (
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
)

#: Underlying finals. v is ü (as AISHELL-3 writes it), ve/van/vn are üe/üan/ün.
FINALS: tuple[str, ...] = (
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iou", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "uei", "uan", "uen", "uang", "ueng",
    "ve", "van", "vn",
    "er", "apical_dental", "apical_retroflex", "io",
)

#: Phonemic segments of every final (medial + nucleus + coda). ASCII ids on purpose.
FINAL_SEGMENTS: dict[str, tuple[str, ...]] = {
    "a": ("a",), "o": ("o",), "e": ("e",), "i": ("i",), "u": ("u",), "v": ("y",),
    "ai": ("a", "i"), "ei": ("e", "i"), "ao": ("a", "u"), "ou": ("o", "u"),
    "an": ("a", "n"), "en": ("e", "n"), "ang": ("a", "ng"), "eng": ("e", "ng"),
    "ong": ("u", "ng"),
    "ia": ("j", "a"), "ie": ("j", "e"), "iao": ("j", "a", "u"), "iou": ("j", "o", "u"),
    "ian": ("j", "a", "n"), "in": ("i", "n"), "iang": ("j", "a", "ng"),
    "ing": ("i", "ng"), "iong": ("j", "u", "ng"),
    "ua": ("w", "a"), "uo": ("w", "o"), "uai": ("w", "a", "i"), "uei": ("w", "e", "i"),
    "uan": ("w", "a", "n"), "uen": ("w", "e", "n"), "uang": ("w", "a", "ng"),
    "ueng": ("w", "e", "ng"),
    "ve": ("yu", "e"), "van": ("yu", "a", "n"), "vn": ("y", "n"),
    "er": ("er",),
    "apical_dental": ("apical_dental",), "apical_retroflex": ("apical_retroflex",),
    "io": ("j", "o"),
}

#: Segment inventory implied by FINAL_SEGMENTS plus the onsets.
SEGMENTS: tuple[str, ...] = (
    "a", "o", "e", "i", "u", "y", "er", "apical_dental", "apical_retroflex",
    "j", "w", "yu", "n", "ng",
)

#: The only finals that can stand alone without an onset (a syllable written as is).
ZERO_INITIAL_FINALS = ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er")

#: y + rest; the rest is not the underlying final (ya = /ia/, yu = /y/).
_Y_SPELLINGS: dict[str, str] = {
    "i": "i", "in": "in", "ing": "ing",
    "a": "ia", "e": "ie", "ao": "iao", "ou": "iou", "an": "ian", "ang": "iang",
    "ong": "iong", "o": "io",
    "u": "v", "ue": "ve", "uan": "van", "un": "vn",
}
#: w + rest (wu = /u/, wei = /uei/, wen = /uen/).
_W_SPELLINGS: dict[str, str] = {
    "u": "u", "a": "ua", "o": "uo", "ai": "uai", "ei": "uei",
    "an": "uan", "en": "uen", "ang": "uang", "eng": "ueng",
}
#: Finals written identically after a real onset.
_PLAIN_FINALS = set(FINALS) - {"io"}

#: Onsets after which written u/ue/uan/un are ü/üe/üan/ün.
FRONT_ROUNDED_ONSETS = ("j", "q", "x")
#: Onsets whose written i is an apical vowel.
DENTAL_APICAL_ONSETS = ("z", "c", "s")
RETROFLEX_APICAL_ONSETS = ("zh", "ch", "sh", "r")
#: Onsets that take o for /uo/ (bo po mo fo).
LABIAL_ONSETS = ("b", "p", "m", "f")

TONE_DIGITS = "12345"


class PinyinError(ValueError):
    """A pinyin token that cannot be resolved. code is stable for counting."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Token:
    """One resolved pinyin syllable."""

    pinyin: str          # as written, with the tone digit
    initial: str         # '' | one of ONSETS | 'y' | 'w' (orthographic, as written)
    final: str           # underlying final id, e.g. 'ian', 've', 'apical_retroflex'
    tone: int            # 1..5, 5 = neutral
    erhua: bool = False  # trailing erhua: the final is modified, not followed by /r/
    #: the final as written after the onset ('ue' in yue, 'i' in zhi), '' if unknown
    written: str = ""

    @property
    def segments(self) -> tuple[str, ...]:
        onset = (self.initial,) if self.initial in ONSETS else ()
        return onset + FINAL_SEGMENTS[self.final]

    def as_dict(self) -> dict:
        return {"pinyin": self.pinyin, "initial": self.initial, "final": self.final,
                "written": self.written, "tone": self.tone, "erhua": self.erhua,
                "segments": list(self.segments)}


def _split_onset(body: str) -> tuple[str, str]:
    """Longest written-onset match; '' when the body starts with a vowel or y/w."""
    if body[:1] in ("y", "w"):
        return body[:1], body[1:]
    for onset in ONSETS:
        if body.startswith(onset):
            return onset, body[len(onset):]
    return "", body


def resolve_final(initial: str, written: str) -> str:
    """Underlying final id for an orthographic (initial, written final) pair."""
    if initial == "y":
        try:
            return _Y_SPELLINGS[written]
        except KeyError:
            raise PinyinError("unknown_spelling",
                              f"y + {written!r} is not a Mandarin spelling") from None
    if initial == "w":
        try:
            return _W_SPELLINGS[written]
        except KeyError:
            raise PinyinError("unknown_spelling",
                              f"w + {written!r} is not a Mandarin spelling") from None
    if initial == "":
        # Written alone, the final is its own underlying form. Orthodox standalone finals
        # are in ZERO_INITIAL_FINALS ('yi' is the orthodox spelling of /i/, so a bare 'i'
        # is a shorthand), but any known final id is accepted as itself instead of being
        # guessed: 'ian' means /ian/ and is then blocked by the inventory, never repaired.
        if written in FINALS:
            return written
        raise PinyinError("unknown_final",
                          f"final {written!r} cannot stand alone as a syllable")
    if initial in FRONT_ROUNDED_ONSETS:
        # j/q/x take ü where other onsets take u, so 'u' spellings are checked first. 'iu'
        # is the contraction of 'iou' (jiu/qiu/xiu) and is not a ü spelling: it is listed
        # here because this branch returns before the general contraction rules below.
        front = {"u": "v", "ue": "ve", "uan": "van", "un": "vn",
                 "iu": "iou", "v": "v", "ve": "ve", "van": "van", "vn": "vn"}
        if written in front:
            return front[written]
        if written in _PLAIN_FINALS:
            return written
        raise PinyinError("unknown_final", f"{initial} + {written!r} is not a Mandarin final")
    if initial in LABIAL_ONSETS and written == "o":
        return "uo"  # bo/po/mo/fo are /puo pʰuo muo fuo/
    if initial in DENTAL_APICAL_ONSETS and written == "i":
        return "apical_dental"
    if initial in RETROFLEX_APICAL_ONSETS and written == "i":
        return "apical_retroflex"
    if written == "iu":
        return "iou"
    if written == "ui":
        return "uei"
    if written == "un":
        return "uen"
    if written in _PLAIN_FINALS:
        return written
    raise PinyinError("unknown_final", f"{initial} + {written!r} is not a Mandarin final")


def _resolve_body(body: str) -> tuple[str, str]:
    initial, written = _split_onset(body)
    return initial, resolve_final(initial, written)


def split_token(token: str) -> Token:
    """Resolve 'guang3' / 'nv3' / 'bar4' into (initial, underlying final, tone, erhua)."""
    text = token.strip().lower()
    if not text:
        raise PinyinError("empty", "empty pinyin token")
    if text[-1] not in TONE_DIGITS:
        raise PinyinError("no_tone", f"pinyin token must end with a tone digit 1-5: {token!r}")
    tone = int(text[-1])
    body = text[:-1].replace("ü", "v")
    if not body:
        raise PinyinError("empty_body", f"pinyin token has no body: {token!r}")
    if not body.isalpha() or not body.isascii():
        raise PinyinError("non_alpha", f"pinyin body must be ASCII letters: {token!r}")
    if body != "er" and body.endswith("r"):
        try:
            initial, final = _resolve_body(body[:-1])
        except PinyinError:
            pass
        else:
            return Token(text, initial, final, tone, True, body[:-1][len(initial):])
    initial, final = _resolve_body(body)
    return Token(text, initial, final, tone, False, body[len(initial):])


def parse_token(token: str) -> Token:
    """Alias kept for readability at call sites that parse annotation files."""
    return split_token(token)


def renderable_units(initial: str, final: str, onsets: set[str], finals: set[str],
                     *, erhua: bool = False) -> bool:
    """Whether an inventory holding onsets/finals can render this syllable."""
    if erhua:
        return False
    if initial and initial not in ("y", "w") and initial not in onsets:
        return False
    return final in finals


def missing_units(initial: str, final: str, onsets: set[str], finals: set[str],
                  *, erhua: bool = False) -> tuple[str, ...]:
    """The units whose absence blocks this syllable (empty when renderable)."""
    missing: list[str] = []
    # y/w are orthographic glides, not onsets: 'ya' is /ia/, rendered from the final alone,
    # so the synthesizer never needs a unit for them. Counting them would inflate the
    # missing-onset mass with 86k syllables the inventory can already cover.
    if initial and initial not in ("y", "w") and initial not in onsets:
        missing.append(f"onset:{initial}")
    if final not in finals:
        missing.append(f"final:{final}")
    if erhua:
        missing.append("erhua")
    return tuple(missing)
