"""Tests for the energy-valley syllable aligner (P1.5).

The fixtures are synthetic harmonic bursts, so these tests pin the search behaviour (global
duration constraints, valley depth reporting, refusal when the region is too short) and not
whether a boundary lands on a real phoneme. That question is answered only by the manual
listening/acceptance step.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhe.data.align import align_syllables
from vhe.data.pitch import track_pitch

SR = 24000


def burst(duration=0.18, f0_hz=200.0, sample_rate=SR):
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = sum(np.sin(2 * np.pi * f0_hz * k * t) / k for k in range(1, 6))
    envelope = np.minimum(1.0, np.minimum(t / 0.02, (duration - t) / 0.02))
    return (signal * envelope).astype(np.float32)


def silence(duration, sample_rate=SR):
    return np.zeros(int(round(duration * sample_rate)), dtype=np.float32)


def test_three_bursts_are_located_with_high_confidence():
    audio = np.concatenate([burst(), silence(0.10), burst(), silence(0.10), burst()])
    alignment = align_syllables(audio, SR, 3)
    assert len(alignment.boundaries) == 3
    assert alignment.stats()["mean_confidence"] == 1.0
    assert alignment.stats()["low_confidence_boundaries"] == 0
    first, middle, last = alignment.boundaries
    assert first.start_s <= 0.02
    # The gap between burst 1 and 2 is 0.18-0.28 s; the boundary must sit inside it.
    assert 0.175 <= middle.start_s <= 0.285
    assert 0.455 <= last.start_s <= 0.565
    assert last.end_s >= 0.70
    assert middle.dip_db is not None and middle.dip_db > 20.0


def test_continuous_signal_boundaries_carry_no_confidence():
    alignment = align_syllables(burst(0.45), SR, 3)
    stats = alignment.stats()
    assert stats["syllables"] == 3
    # No energy valley exists, so the duration constraint decides and every boundary is
    # reported as untrustworthy instead of being silently accepted.
    assert stats["min_confidence"] is not None and stats["min_confidence"] < 0.01
    assert stats["low_confidence_boundaries"] >= 2
    assert alignment.issues == []


def test_region_too_short_is_an_error_not_a_best_effort_split():
    alignment = align_syllables(burst(0.45), SR, 10)
    assert alignment.boundaries == []
    codes = [issue["code"] for issue in alignment.issues]
    assert codes == ["alignment_region_too_short"]
    assert alignment.issues[0]["severity"] == "error"


def test_single_syllable_spans_the_speech_region():
    alignment = align_syllables(burst(0.45), SR, 1)
    assert len(alignment.boundaries) == 1
    boundary = alignment.boundaries[0]
    assert boundary.start_s <= 0.05
    assert boundary.end_s >= 0.40
    assert boundary.confidence is None


def test_silence_and_short_signals_report_explicit_failures():
    silent = align_syllables(silence(0.5), SR, 2)
    assert [issue["code"] for issue in silent.issues] == ["alignment_no_speech"]
    tiny = align_syllables(burst(0.01), SR, 2)
    assert tiny.issues and tiny.issues[0]["severity"] == "error"


def test_alignment_is_deterministic():
    audio = np.concatenate([burst(), silence(0.08), burst()])
    first = align_syllables(audio, SR, 2)
    second = align_syllables(audio, SR, 2)
    assert [b.as_dict() for b in first.boundaries] == [b.as_dict() for b in second.boundaries]


def test_voiced_ratio_is_attached_when_an_f0_track_is_given():
    audio = np.concatenate([burst(), silence(0.10), burst()])
    track = track_pitch(audio, SR)
    alignment = align_syllables(audio, SR, 2, f0_track=track)
    assert all(boundary.voiced_ratio is not None for boundary in alignment.boundaries)
    assert alignment.boundaries[0].voiced_ratio > 0.5


def test_count_and_duration_arguments_are_validated():
    with pytest.raises(ValueError):
        align_syllables(burst(0.2), SR, 0)
    with pytest.raises(ValueError):
        align_syllables(burst(0.2), SR, 2, min_duration_s=0.0)
    with pytest.raises(ValueError):
        align_syllables(np.zeros(0, dtype=np.float32), SR, 1)
