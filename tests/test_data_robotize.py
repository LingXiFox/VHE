"""Tests for the analysis-resynthesis mechanical source.

What these tests prove:
  * robotize keeps length, sample rate, dtype and loudness, is deterministic for a
    seed, returns silence for silence, and preserves the fundamental within 5 %;
  * the overlap-add edges are not amplified (regression for the padding bug that
    produced a peak of 87 for an input peaking at 4.6).

What they do not prove: that the result is intelligible or that a listener would
call it mechanical. Both are listening questions, recorded in EXP-0029.
"""
from __future__ import annotations

import numpy as np
import pytest

from vhe.data.robotize import robotize

SAMPLE_RATE = 24000


def _vowel(f0: float = 220.0, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    signal = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * time) for h in range(1, 25))
    return (0.5 * signal / np.abs(signal).max()).astype(np.float64)


def _first_strong_lag_f0(signal: np.ndarray, sample_rate: int, fmin: float = 65.0,
                         fmax: float = 500.0) -> float:
    """Fundamental from the first autocorrelation lag above 90 % of the maximum.

    Taking the plain argmax is ambiguous for a periodic signal: twice the period
    peaks just as high, and integer lag quantisation decides which wins.
    """
    centred = signal - signal.mean()
    lags = np.arange(int(sample_rate / fmax), int(sample_rate / fmin) + 1)
    correlation = np.array([np.dot(centred[:len(centred) - lag], centred[lag:])
                            for lag in lags])
    correlation = correlation / np.linalg.norm(centred) ** 2 * len(centred) / (len(centred) - lags)
    strong = np.nonzero(correlation >= 0.9 * correlation.max())[0]
    return float(sample_rate / lags[strong[0]])


def test_robotize_keeps_length_rate_and_dtype() -> None:
    signal = _vowel(seconds=0.5)
    output = robotize(signal, SAMPLE_RATE)
    assert output.shape == signal.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert float(np.abs(output).max()) < 1.0


def test_robotize_matches_loudness() -> None:
    signal = _vowel()
    output = robotize(signal, SAMPLE_RATE).astype(np.float64)
    ratio = np.sqrt((output ** 2).mean()) / np.sqrt((signal ** 2).mean())
    # The peak guard can pull the level down after the RMS match, so the band is
    # "within 3 dB, never louder" rather than a tight equality.
    assert 0.7 < ratio < 1.1


def test_robotize_is_deterministic_for_a_seed() -> None:
    signal = _vowel(seconds=0.5)
    first = robotize(signal, SAMPLE_RATE, seed=7)
    second = robotize(signal, SAMPLE_RATE, seed=7)
    assert np.array_equal(first, second)


def test_robotize_returns_silence_for_silence() -> None:
    output = robotize(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert output.shape == (SAMPLE_RATE,)
    assert float(np.abs(output).max()) == pytest.approx(0.0, abs=1e-7)


def test_robotize_preserves_the_fundamental() -> None:
    signal = _vowel(f0=220.0)
    output = robotize(signal, SAMPLE_RATE).astype(np.float64)
    assert _first_strong_lag_f0(output, SAMPLE_RATE) == pytest.approx(220.0, rel=0.05)


def test_robotize_does_not_amplify_the_edges() -> None:
    signal = _vowel()
    output = robotize(signal, SAMPLE_RATE).astype(np.float64)
    rms = float(np.sqrt((output ** 2).mean()))
    assert float(np.abs(output[:64]).max()) < 10 * rms
    assert float(np.abs(output[-64:]).max()) < 10 * rms


def test_robotize_rejects_a_two_dimensional_signal() -> None:
    with pytest.raises(ValueError, match="1-D"):
        robotize(np.zeros((2, 128)), SAMPLE_RATE)


def test_robotize_keeps_the_spectral_envelope() -> None:
    """A band-limited burst keeps its energy where it started."""
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * 1500 * time) * np.sin(2 * np.pi * 3 * time) ** 2
    output = robotize(carrier, SAMPLE_RATE).astype(np.float64)
    spectrum_in = np.abs(np.fft.rfft(carrier * np.hanning(len(carrier))))
    spectrum_out = np.abs(np.fft.rfft(output * np.hanning(len(output))))
    freqs = np.fft.rfftfreq(len(carrier), 1 / SAMPLE_RATE)
    peak_in = freqs[int(np.argmax(spectrum_in))]
    peak_out = freqs[int(np.argmax(spectrum_out))]
    assert peak_in == pytest.approx(1500.0, abs=20.0)
    assert peak_out == pytest.approx(peak_in, abs=200.0)
