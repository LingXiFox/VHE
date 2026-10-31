"""Tests for the P1-B signal front-end: resampling, F0 tracking, segmentation.

These run on synthetic signals only. Real AISHELL-3 audio cannot be exercised here
(no corpus on disk, see docs/DATA_POLICY.md); the tests fix the numerical contract so a
later real-data run can be judged against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhe.data.f0 import estimate_f0
from vhe.data.resample import resample
from vhe.data.segment import check_syllable_count, segment_by_energy

SR = 24000


def tone(freq_hz: float, seconds: float = 1.0, sample_rate: int = SR,
         amplitude: float = 0.6) -> np.ndarray:
    times = np.arange(int(round(sample_rate * seconds))) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * times)).astype(np.float32)


def bursts(freqs: list[float], *, burst_s: float = 0.12, gap_s: float = 0.10,
           lead_s: float = 0.05, sample_rate: int = SR,
           amplitude: float = 0.6) -> np.ndarray:
    parts = [np.zeros(int(round(lead_s * sample_rate)), dtype=np.float32)]
    for freq in freqs:
        parts.append(tone(freq, burst_s, sample_rate, amplitude))
        parts.append(np.zeros(int(round(gap_s * sample_rate)), dtype=np.float32))
    return np.concatenate(parts)


# --- resampling -------------------------------------------------------------------


def test_resample_length_is_exact_rational():
    audio = tone(440.0, 1.0, 44100)
    out = resample(audio, 44100, 24000)
    assert out.size == 24000
    assert out.dtype == np.float32
    short = tone(440.0, 0.5, 16000)
    assert resample(short, 16000, 24000).size == round(short.size * 24000 / 16000)


@pytest.mark.parametrize("freq", [500.0, 1000.0, 2000.0, 4000.0])
def test_resample_preserves_passband_gain(freq: float):
    # RMS over whole periods, not peak: at 4 kHz/24 kHz the peak estimate itself is
    # phase dependent and reads ~1.25 dB low even for a perfectly flat filter.
    audio = tone(freq, 0.5, 44100)
    out = resample(audio, 44100, 24000)[1000:-1000]
    period = int(round(24000 / freq))
    whole = (out.size // period) * period
    gain_db = 20.0 * np.log10(float(np.sqrt(np.mean(out[:whole] ** 2))) / (0.6 / np.sqrt(2.0)))
    assert abs(gain_db) <= 0.5, gain_db


def test_resample_rejects_aliasing_above_new_nyquist():
    audio = tone(15000.0, 0.5, 44100)
    out = resample(audio, 44100, 24000)[1000:-1000]
    residual_db = 20.0 * np.log10(max(np.max(np.abs(out)), 1e-12) / 0.6)
    assert residual_db <= -40.0, residual_db


def test_resample_identity_and_determinism():
    audio = tone(330.0, 0.2, 24000)
    same = resample(audio, 24000, 24000)
    assert np.array_equal(same, audio)
    assert not np.shares_memory(same, audio)
    first = resample(audio, 44100, 24000)
    second = resample(audio, 44100, 24000)
    assert np.array_equal(first, second)


def test_resample_rejects_bad_input():
    with pytest.raises(ValueError):
        resample(np.zeros((2, 100), dtype=np.float32), 44100, 24000)
    with pytest.raises(ValueError):
        resample(np.zeros(0, dtype=np.float32), 44100, 24000)


# --- F0 ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq", [120.0, 220.0, 300.0, 400.0])
def test_f0_steady_tone_within_three_percent(freq: float):
    track = estimate_f0(tone(freq), SR)
    voiced = track.voiced_f0
    assert track.voiced_ratio > 0.9
    median = float(np.median(voiced))
    assert abs(median - freq) / freq <= 0.03, median


def test_f0_tracks_rising_glide():
    times = np.arange(SR) / SR
    contour = np.linspace(100.0, 400.0, times.size)
    phase = 2.0 * np.pi * np.cumsum(contour) / SR
    track = estimate_f0((0.6 * np.sin(phase)).astype(np.float32), SR)
    frame_times, values = track.contour()
    voiced = values > 0
    assert voiced.sum() > 0.9 * values.size
    expected = np.interp(frame_times[voiced], times, contour)
    rel_error = np.abs(values[voiced] - expected) / expected
    assert float(np.mean(rel_error)) <= 0.03, float(np.mean(rel_error))
    assert values[voiced][-1] > values[voiced][0]


def test_f0_marks_silence_unvoiced():
    track = estimate_f0(np.zeros(SR // 2, dtype=np.float32), SR)
    assert track.voiced_ratio == 0.0
    assert track.voiced_f0.size == 0
    assert track.stats()["median_f0_hz"] is None


def test_f0_rejects_invalid_range():
    with pytest.raises(ValueError):
        estimate_f0(tone(220.0), SR, fmin_hz=900.0, fmax_hz=65.0)


# --- segmentation -----------------------------------------------------------------


def test_segmentation_finds_four_syllables():
    audio = bursts([220.0, 260.0, 300.0, 240.0])
    result = segment_by_energy(audio, SR)
    assert len(result.segments) == 4, result.stats()
    assert [seg.start_s for seg in result.segments] == sorted(seg.start_s for seg in result.segments)
    for seg in result.segments:
        assert 0.10 <= seg.duration_s <= 0.20, seg.duration_s
    assert not [i for i in result.issues if i["severity"] == "error"]


def test_segmentation_merges_short_gaps():
    # Two bursts 20 ms apart are one syllable in read speech; a 100 ms gap must not merge.
    joined = np.concatenate([np.zeros(SR // 20, dtype=np.float32), tone(220.0, 0.10),
                             np.zeros(SR // 50, dtype=np.float32), tone(260.0, 0.10),
                             np.zeros(SR // 20, dtype=np.float32)])
    result = segment_by_energy(joined, SR)
    assert len(result.segments) == 1, [seg.duration_s for seg in result.segments]


def test_segmentation_reports_dropped_short_burst():
    # A lone 10 ms click must not be promoted to a syllable, and dropping it must be visible.
    audio = np.concatenate([np.zeros(SR // 10, dtype=np.float32), tone(220.0, 0.15),
                            np.zeros(SR // 10, dtype=np.float32), tone(260.0, 0.01),
                            np.zeros(SR // 10, dtype=np.float32), tone(300.0, 0.15),
                            np.zeros(SR // 10, dtype=np.float32)])
    result = segment_by_energy(audio, SR)
    codes = {issue["code"] for issue in result.issues}
    assert "segment_too_short" in codes
    assert len(result.segments) == 2


def test_segmentation_flags_clipping():
    audio = np.clip(bursts([220.0, 260.0], amplitude=2.0), -1.0, 1.0)
    result = segment_by_energy(audio, SR)
    assert "clipping" in {issue["code"] for issue in result.issues}


def test_segmentation_silence_is_an_error():
    result = segment_by_energy(np.zeros(SR, dtype=np.float32), SR)
    assert result.segments == []
    assert any(issue["code"] == "no_speech" and issue["severity"] == "error"
               for issue in result.issues)


def test_segmentation_voiced_flag_uses_f0_track():
    audio = bursts([220.0, 260.0, 300.0, 240.0])
    track = estimate_f0(audio, SR)
    result = segment_by_energy(audio, SR, f0_track=track)
    assert all(seg.voiced is True for seg in result.segments)


def test_syllable_count_mismatch_is_recorded():
    audio = bursts([220.0, 260.0, 300.0, 240.0])
    result = segment_by_energy(audio, SR)
    assert check_syllable_count(result, 4) is True
    assert check_syllable_count(result, 3) is False
    assert "syllable_count_mismatch" in {issue["code"] for issue in result.issues}
