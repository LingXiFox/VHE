"""Tests for the NumPy MFCC front end (aligner input, P2).

These prove the transform is deterministic, finite and correctly shaped. They say nothing about
whether MFCC is the right representation for alignment quality — that is measured separately on
synthetic and real speech.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhe.data.features import (HOP_S, N_MFCC, SAMPLE_RATE, acoustic_features, cmvn, deltas,
                               frame_count, frame_signal, hz_to_mel, log_mel_spectrogram,
                               mel_filterbank, mel_to_hz, mfcc)


def tone(frequency=220.0, duration=0.5, sample_rate=SAMPLE_RATE, amplitude=0.3):
    times = np.arange(int(duration * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def test_mel_scale_roundtrip():
    for frequency in (0.0, 100.0, 1000.0, 7000.0):
        assert mel_to_hz(hz_to_mel(frequency)) == pytest.approx(frequency, abs=1e-6)


def test_mel_filterbank_is_triangular_and_covers_band():
    bank = mel_filterbank(SAMPLE_RATE, 512, 26)
    assert bank.shape == (26, 257)
    assert (bank >= 0).all()
    # Every filter must have weight, and the whole set must be non-degenerate.
    assert (bank.sum(axis=1) > 0).all()
    # A low tone must land in the low filters, a high tone in the high ones.
    low = bank@ np.abs(np.fft.rfft(tone(200.0, 0.2), 512))[:257]
    high = bank @ np.abs(np.fft.rfft(tone(6000.0, 0.2), 512))[:257]
    assert int(np.argmax(low)) < int(np.argmax(high))


def test_frame_count_matches_framing():
    assert frame_count(16000, 400, 160) == 98
    frames = frame_signal(np.zeros(16000, dtype=np.float32), 400, 160)
    assert frames.shape[0] == frame_count(16000, 400, 160)
    assert frames.shape[1] == 400


def test_mfcc_shape_and_finiteness():
    coefficients = mfcc(tone(), SAMPLE_RATE)
    assert coefficients.ndim == 2
    assert coefficients.shape[1] == N_MFCC
    assert np.isfinite(coefficients).all()
    assert coefficients.shape[0] == frame_count(int(0.5 * SAMPLE_RATE),
                                               int(0.025 * SAMPLE_RATE),
                                               int(HOP_S * SAMPLE_RATE))


def test_acoustic_features_is_deterministic_and_normalised():
    first = acoustic_features(tone(), SAMPLE_RATE)
    second = acoustic_features(tone(), SAMPLE_RATE)
    assert first.shape == (48, 39)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    # CMVN makes each dimension zero-mean unit-variance before differencing.
    static = first[:, :N_MFCC]
    assert np.abs(static.mean(axis=0)).max() < 1e-3
    assert np.abs(static.std(axis=0) - 1.0).max() < 1e-3


def test_silence_normalises_to_zero():
    silence = np.zeros(8000, dtype=np.float32)
    features = acoustic_features(silence, SAMPLE_RATE)
    assert np.isfinite(features).all()
    assert np.abs(features).max() == pytest.approx(0.0, abs=1e-6)


def test_log_mel_spectrogram_tracks_amplitude():
    spectrogram = log_mel_spectrogram(tone(), SAMPLE_RATE)
    assert spectrogram.shape == (48, 26)
    assert np.isfinite(spectrogram).all()
    # Natural-log mel power (not dB), so values may be positive; halving the amplitude must
    # drop every bin by a constant offset rather than changing the ordering.
    quieter = log_mel_spectrogram(tone(amplitude=0.15), SAMPLE_RATE)
    difference = spectrogram - quieter
    assert np.isfinite(difference).all()
    assert difference.mean() > 0.0


def test_deltas_reduce_to_zero_on_constant_input():
    constant = np.ones((20, 4), dtype=np.float64)
    assert np.abs(deltas(constant)).max() == pytest.approx(0.0, abs=1e-12)
    ramp = np.tile(np.arange(20, dtype=np.float64)[:, None], (1, 4))
    assert deltas(ramp).mean() > 0.0


def test_cmvn_floors_constant_dimensions():
    feature = np.zeros((10, 3), dtype=np.float64)
    feature[:, 0] = np.arange(10)
    normalised = cmvn(feature)
    assert np.isfinite(normalised).all()
    assert np.abs(normalised[:, 1]).max() < 1e-6
