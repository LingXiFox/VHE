"""Tests for the multi-candidate pitch tracker (EXP-0015)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vhe.data.pitch import (
    PeriodCandidate,
    REASONS,
    frame_candidates,
    octave_error_kind,
    prefer_shortest_period,
    track_pitch,
)
from vhe.synth import (
    PRESETS,
    PhonemeSegment,
    PhonemeSequence,
    PitchCurve,
    SynthesisRequest,
    Timing,
)
from vhe.synth.mandarin import MandarinFormantSynth

SR = 24000
SYNTH = MandarinFormantSynth()


def render_a(curve: PitchCurve, duration_s: float) -> np.ndarray:
    request = SynthesisRequest(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("a", duration_s),), inventory="mandarin"),
        pitch=curve, timing=Timing(0.0, 0.0), sample_rate=SR)
    return SYNTH.render(request).audio


def cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(a / b)


@pytest.mark.parametrize("f0", [210.0, 300.0])
def test_synth_vowel_is_read_correctly_without_narrowing_fmax(f0: float) -> None:
    """EXP-0014 regression: these two read 2x-4x too high on the single-dip estimator."""
    track = track_pitch(render_a(PitchCurve.constant(f0, 0.6), 0.6), SR)
    assert track.stats()["voiced_ratio"] > 0.9
    assert abs(cents(float(np.median(track.voiced_f0)), f0)) < 50.0


def test_glide_and_vibrato_stay_within_50_cents() -> None:
    times = np.linspace(0.0, 0.6, 61)
    glide = PitchCurve(times_s=tuple(times), f0_hz=tuple(np.linspace(110.0, 220.0, 61)))
    track = track_pitch(render_a(glide, 0.6), SR)
    errors = [abs(cents(frame.f0_hz, float(np.interp(frame.time_s, times,
                                                          np.linspace(110.0, 220.0, 61)))))
              for frame in track.frames if frame.f0_hz > 0.0]
    assert errors and max(errors) < 50.0

    vibrato_t = np.linspace(0.0, 0.8, 81)
    vibrato = 200.0 + 12.0 * np.sin(2.0 * math.pi * 5.0 * vibrato_t)
    vibrato_track = track_pitch(render_a(PitchCurve(times_s=tuple(vibrato_t),
                                                    f0_hz=tuple(vibrato)), 0.8), SR)
    errors = [abs(cents(frame.f0_hz, float(np.interp(frame.time_s, vibrato_t, vibrato))))
              for frame in vibrato_track.frames if frame.f0_hz > 0.0]
    assert errors and max(errors) < 50.0


def test_frame_candidates_keep_the_fundamental_and_drop_the_2x_relative() -> None:
    t = np.arange(int(0.06 * SR)) / SR
    tone = (0.5 * np.sin(2.0 * math.pi * 200.0 * t)).astype(np.float32)
    values = [c.f0_hz for c in frame_candidates(tone, SR)]
    assert any(abs(cents(v, 200.0)) < 20.0 for v in values)
    # A pure tone is equally periodic at 2T, so 100 Hz is generated as a candidate and then
    # removed by the shortest-period pass; leaving it in would make the Viterbi free to drift
    # an octave down.
    assert not any(abs(cents(v, 100.0)) < 20.0 for v in values)


def test_prefer_shortest_period_keeps_the_shortest_of_an_equal_family() -> None:
    candidates = [PeriodCandidate(f0_hz=f, lag=SR / f, aperiodicity=0.01, harmonic_support=0.99,
                                 score=0.95) for f in (100.0, 200.0, 400.0)]
    kept = prefer_shortest_period(candidates)
    assert [round(c.f0_hz) for c in kept] == [400]


def test_prefer_shortest_period_keeps_a_better_longer_period() -> None:
    short = PeriodCandidate(f0_hz=400.0, lag=SR / 400.0, aperiodicity=0.2,
                            harmonic_support=0.6, score=0.55)
    long = PeriodCandidate(f0_hz=200.0, lag=SR / 200.0, aperiodicity=0.01,
                           harmonic_support=0.99, score=0.95)
    kept = prefer_shortest_period([short, long])
    assert [round(c.f0_hz) for c in kept] == [200, 400]


def test_white_noise_is_rejected_instead_of_guessed() -> None:
    noise = np.random.default_rng(3).standard_normal(SR // 4).astype(np.float32) * 0.2
    track = track_pitch(noise, SR)
    assert all(frame.f0_hz == 0.0 for frame in track.frames)
    assert track.stats()["voiced_ratio"] == 0.0
    assert set(track.reasons()) <= set(REASONS)


def test_silence_is_reported_as_silence_not_as_low_confidence() -> None:
    track = track_pitch(np.zeros(SR // 4, dtype=np.float32), SR)
    assert track.reasons()["silence"] == len(track.frames)


def test_reasons_partition_every_frame() -> None:
    track = track_pitch(render_a(PitchCurve.constant(220.0, 0.4), 0.4), SR)
    counts = track.reasons()
    assert sum(counts.values()) == len(track.frames)
    assert counts["ok"] > 0


def test_rejections_are_not_interpolated() -> None:
    """A rejected frame keeps f0 = 0 even when both neighbours are voiced."""
    track = track_pitch(render_a(PitchCurve.constant(220.0, 0.4), 0.4), SR)
    rejected = [i for i, frame in enumerate(track.frames) if frame.f0_hz == 0.0]
    for index in rejected:
        assert track.frames[index].reason != "ok"
        assert track.frames[index].confidence >= 0.0


def test_to_f0_track_matches_frame_count_and_times() -> None:
    track = track_pitch(render_a(PitchCurve.constant(180.0, 0.3), 0.3), SR)
    legacy = track.to_f0_track()
    assert len(legacy.frames) == len(track.frames)
    assert [round(f.time_s, 6) for f in legacy.frames] == [round(f.time_s, 6)
                                                            for f in track.frames]
    assert legacy.sample_rate == SR


@pytest.mark.parametrize("estimate,reference,expected", [
    (200.0, 200.0, None), (400.0, 200.0, "2x"), (600.0, 200.0, "3x"),
    (100.0, 200.0, "1/2x"), (50.0, 200.0, "1/4x"), (260.0, 200.0, None),
])
def test_octave_error_kind(estimate: float, reference: float, expected: str | None) -> None:
    assert octave_error_kind(estimate, reference) == expected
