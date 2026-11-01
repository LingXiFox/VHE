"""Tests for the pinyin -> phoneme planner (P1-C).

The point of these tests is the refusals: a syllable the backend cannot pronounce must be
blocked with a reason, never rendered as a vowel or as silence.
"""

from __future__ import annotations

import pytest

from vhe.data.g2p import (NUCLEUS_MIN_S, ONSET_MIN_S, SUPPORTED_FINALS, SUPPORTED_ONSETS,
                          nominal_onset_s, syllable_phonemes, syllable_plan)
from vhe.data.pairing import parse_syllable
from vhe.synth import MANDARIN_PHONEMES


@pytest.mark.parametrize("initial,final,expected", [
    ("", "a", ("a",)),
    ("m", "a", ("m", "a")),
    ("sh", "ang", ("sh", "ang")),
    ("b", "ai", ("b", "ai")),
    ("l", "ei", ("l", "ei")),
    ("n", "ou", ("n", "ou")),
    # inventory stage 1 (EXP-0016): a final unit carries its own medial glide, so the
    # phoneme list stays onset + one unit and never becomes a segment concatenation.
    ("", "e", ("e",)),
    ("g", "e", ("g", "e")),
    ("b", "ian", ("b", "ian")),
    ("y", "an", ("ian",)),      # yan = /ian/, the glide belongs to the unit
    ("p", "ing", ("p", "ing")),
    ("l", "iu", ("l", "iou")),  # liu = /liou/
    ("y", "ou", ("iou",)),      # you = /iou/
    # inventory stage 2 (EXP-0016): the apical vowel is its own unit, never /i/.
    ("sh", "i", ("sh", "apical_retroflex")),
    ("d", "uo", ("d", "uo")),
    ("d", "ong", ("d", "ong")),
    ("b", "en", ("b", "en")),
    # inventory stage 3 (EXP-0016): initials zh and h.
    ("zh", "i", ("zh", "apical_retroflex")),
    ("zh", "an", ("zh", "an")),
    ("h", "e", ("h", "e")),
    ("h", "ong", ("h", "ong")),
])
def test_renderable_syllables_map_to_onset_plus_nucleus(initial, final, expected):
    phonemes, reason = syllable_phonemes(initial, final)
    assert reason is None
    assert phonemes == expected
    for phoneme in phonemes:
        assert phoneme in MANDARIN_PHONEMES


@pytest.mark.parametrize("initial,final,phoneme", [("y", "i", "i"), ("w", "u", "u")])
def test_zero_initial_glide_spellings_are_single_vowels(initial, final, phoneme):
    assert syllable_phonemes(initial, final) == ((phoneme,), None)


@pytest.mark.parametrize("initial,final,needle", [
    ("z", "i", "apical"),          # zi  -> [tsɿ], not /tsi/ (dental not yet a unit)
    ("s", "i", "apical"),          # si  -> [sɿ]
    ("j", "u", "ü"),               # ju  -> jü
    ("x", "uan", "inventory"),     # xuan: final not in the subset
    ("ch", "a", "onset"),          # ch is still not an inventory unit (zh was stage 3)
    ("ch", "e", "onset"),
    ("r", "en", "onset"),
    ("l", "ie", "final"),          # ie is still not a unit (stage 1 did not add it)
    ("b", "ie", "final"),
    ("d", "uan", "final"),
    ("", "er", "final"),
])
def test_unrenderable_syllables_are_blocked_with_a_reason(initial, final, needle):
    phonemes, reason = syllable_phonemes(initial, final)
    assert phonemes == ()
    assert reason is not None and needle in reason, reason


def test_vowels_inventory_keeps_the_old_behaviour():
    assert syllable_phonemes("", "a", "vowels") == (("a",), None)
    phonemes, reason = syllable_phonemes("m", "a", "vowels")
    assert phonemes == () and "onset" in reason


def test_unknown_inventory_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="unknown inventory"):
        syllable_phonemes("m", "a", "klingon")


def test_plan_splits_measured_duration_between_onset_and_nucleus():
    plan, reason = syllable_plan("m", "a", 0.30)
    assert reason is None
    assert [phoneme for phoneme, _seconds in plan] == ["m", "a"]
    assert plan[0][1] == pytest.approx(nominal_onset_s("m"))
    assert sum(seconds for _phoneme, seconds in plan) == pytest.approx(0.30)


def test_plan_lets_the_nucleus_absorb_a_long_syllable():
    plan, _reason = syllable_plan("b", "ai", 0.60)
    assert plan[0][1] == pytest.approx(nominal_onset_s("b"))
    assert plan[1][1] > 0.5


def test_plan_shrinks_the_onset_when_the_syllable_is_short():
    plan, reason = syllable_plan("sh", "a", 0.130)
    assert reason is None
    onset, nucleus = plan
    # The fricative nominal length is 80 ms; 130 ms leaves the nucleus its 60 ms minimum.
    assert onset[1] == pytest.approx(0.070)
    assert nucleus[1] == pytest.approx(NUCLEUS_MIN_S)


def test_plan_blocks_a_syllable_too_short_for_both_phonemes():
    plan, reason = syllable_plan("g", "a", 0.070)
    assert plan == ()
    assert reason is not None and "cannot hold onset" in reason
    assert 0.070 < ONSET_MIN_S + NUCLEUS_MIN_S


def test_plan_of_a_single_phoneme_syllable_takes_the_whole_duration():
    plan, reason = syllable_plan("", "a", 0.24)
    assert reason is None
    assert plan == (("a", 0.24),)


def test_parse_syllable_reports_the_planning_result():
    ok = parse_syllable("ma1", "mandarin")
    assert (ok.initial, ok.final, ok.tone) == ("m", "a", 1)
    assert ok.phonemes == ("m", "a") and ok.reason is None
    assert parse_syllable("ma1").phonemes == ()  # vowels inventory: "m" is missing

    blocked = parse_syllable("guang3", "mandarin")
    assert blocked.phonemes == ()
    assert "final" in blocked.reason


@pytest.mark.parametrize("token", ["", "ma", "ma6", "m a1", "1"])
def test_bad_pinyin_tokens_are_rejected(token):
    with pytest.raises(ValueError):
        parse_syllable(token)


@pytest.mark.parametrize("token", [
    "a1", "yi1", "wu1", "yu2", "he1", "ai4", "bei4", "ao4", "gou4", "ban1", "bang1",
    "bian1", "bing1", "liu2", "duo1", "dong1", "gen1", "zhi1", "guang3", "xiang1",
    "yue4", "nv3", "zher4", "chun1", "er2", "shua1", "jiong3", "cuei4",
])
def test_the_coverage_predicate_and_the_synthesiser_agree_on_every_token(token):
    """The manifest's 'renderable' flag and build_pair must never disagree.

    Coverage asks missing_units() about the *underlying* final; synthesis goes through the
    written spelling. They once disagreed for 59% of the "renderable" corpus, so this
    invariant is checked on both sides for the same token.
    """
    from vhe.data.mandarin import missing_units, split_token

    token_parts = split_token(token)
    missing = missing_units(token_parts.initial, token_parts.final, SUPPORTED_ONSETS,
                            SUPPORTED_FINALS, erhua=token_parts.erhua)
    syllable = parse_syllable(token, "mandarin")
    assert (len(missing) == 0) == bool(syllable.phonemes), (missing, syllable.reason)
