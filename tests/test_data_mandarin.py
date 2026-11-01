"""Pinyin orthography tests: written form -> underlying phonemes.

Every case here was found in, or is required by, the AISHELL-3 annotation (1,920 distinct
tokens). The point of the table is that no spelling is ever "close enough": a form that is
not a Mandarin syllable must raise, and a contracted form must resolve to the full final.
"""

from __future__ import annotations

import pytest

from vhe.data.g2p import SUPPORTED_FINALS, SUPPORTED_ONSETS
from vhe.data.mandarin import (
    PinyinError,
    missing_units,
    renderable_units,
    split_token,
)


def final_of(token: str) -> str:
    return split_token(token).final


@pytest.mark.parametrize(("token", "initial", "final"), [
    # iou is written iu after every onset except y (you), which keeps the full spelling.
    ("jiu3", "j", "iou"), ("qiu2", "q", "iou"), ("xiu1", "x", "iou"),
    ("liu2", "l", "iou"), ("diu1", "d", "iou"), ("niu2", "n", "iou"),
    # uei -> ui, uen -> un.
    ("gui4", "g", "uei"), ("dui4", "d", "uei"), ("zhui1", "zh", "uei"),
    ("chun1", "ch", "uen"), ("lun2", "l", "uen"), ("zun1", "z", "uen"),
    # j/q/x take ü where other onsets take u: 'un' is ün, 'iu' is still iou.
    ("jun1", "j", "vn"), ("xun4", "x", "vn"), ("yun2", "y", "vn"),
    ("xue2", "x", "ve"), ("jue2", "j", "ve"), ("yue4", "y", "ve"),
    ("juan4", "j", "van"), ("yuan2", "y", "van"), ("yu2", "y", "v"),
    ("nv3", "n", "v"), ("lv4", "l", "v"),
    # y/w are orthographic: 'ya' is /ia/, 'wo' is /uo/, 'you' is /iou/.
    ("ya1", "y", "ia"), ("wo3", "w", "uo"), ("you3", "y", "iou"),
    ("wen2", "w", "uen"), ("weng1", "w", "ueng"), ("yong3", "y", "iong"),
    ("yi1", "y", "i"), ("wu3", "w", "u"),
    # Apical vowels: the written 'i' after z/c/s differs from the one after zh/ch/sh/r.
    ("zhi4", "zh", "apical_retroflex"), ("ri4", "r", "apical_retroflex"),
    ("shi2", "sh", "apical_retroflex"),
    ("zi4", "z", "apical_dental"), ("ci2", "c", "apical_dental"),
    ("si1", "s", "apical_dental"),
    # Labial + o is /uo/.
    ("bo1", "b", "uo"), ("po4", "p", "uo"), ("mo2", "m", "uo"), ("fo2", "f", "uo"),
])
def test_spelling_resolves_to_the_underlying_final(token, initial, final) -> None:
    parsed = split_token(token)
    assert (parsed.initial, parsed.final, parsed.tone) == (initial, final, int(token[-1]))


def test_written_final_is_kept_for_the_record() -> None:
    assert split_token("yue4").written == "ue"
    assert split_token("zhi4").written == "i"
    assert split_token("jiu3").written == "iu"


@pytest.mark.parametrize(("token", "base_initial", "base_final"), [
    ("bar4", "b", "a"), ("dianr4", "d", "ian"), ("jinr1", "j", "in"),
    ("nv3", "n", "v"),
])
def test_erhua_is_flagged_not_split_into_a_rhotic(token, base_initial, base_final) -> None:
    parsed = split_token(token)
    assert parsed.erhua is token[:-1].endswith("r")  # tone digit is the last character
    assert (parsed.initial, parsed.final) == (base_initial, base_final)


def test_bare_er_is_a_syllable_not_erhua() -> None:
    parsed = split_token("er2")
    assert (parsed.initial, parsed.final, parsed.erhua) == ("", "er", False)


@pytest.mark.parametrize("token", ["dax4", "a", "shi6", "b", "yu0", "xyz1", ""])
def test_unparsable_tokens_raise_instead_of_being_approximated(token) -> None:
    with pytest.raises(PinyinError):
        split_token(token)


def test_y_and_w_are_not_missing_onsets() -> None:
    """The synthesizer renders 'ya' from /ia/: counting a unit for y would be a false gap."""
    assert missing_units("y", "ia", SUPPORTED_ONSETS, SUPPORTED_FINALS) == ("final:ia",)
    # 'uo' became renderable in inventory stage 2 (EXP-0016); 'ua' is still missing, so the
    # y/w property is asserted with a final the inventory does not cover yet.
    assert missing_units("w", "ua", SUPPORTED_ONSETS, SUPPORTED_FINALS) == ("final:ua",)
    assert renderable_units("y", "a", SUPPORTED_ONSETS, SUPPORTED_FINALS)
    # 'zh' became renderable in inventory stage 3 (EXP-0016); 'ch' is still an onset gap.
    assert missing_units("ch", "a", SUPPORTED_ONSETS, SUPPORTED_FINALS) == ("onset:ch",)


def test_erhua_is_reported_as_its_own_missing_unit() -> None:
    missing = missing_units("b", "a", SUPPORTED_ONSETS, SUPPORTED_FINALS, erhua=True)
    assert missing == ("erhua",)
