"""Numerical properties of the v0.1 loss suite (see docs/LOSS_DESIGN.md).

These tests pin down *behaviour*, not quality: they check that a term reacts to the
distortion it is supposed to see and stays quiet when it should. They say nothing about
whether minimising the terms produces natural speech.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from vhe.losses import (DEFAULT_WEIGHTS, energy_envelope_loss, humanization_loss,
                        multi_resolution_stft_loss, periodicity_loss, residual_penalty)


def tone(f0=220.0, seconds=1.0, sample_rate=24000, amplitude=0.3, harmonics=4):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = sum(amplitude / (k + 1) * np.sin(2 * math.pi * f0 * (k + 1) * t)
                 for k in range(harmonics))
    return torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)


def test_stft_loss_is_zero_for_identical_signals():
    x = tone()
    terms = multi_resolution_stft_loss(x, x)
    assert float(terms["spectral_convergence"]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["log_magnitude"]) == pytest.approx(0.0, abs=1e-5)


def test_stft_loss_reacts_to_added_noise_and_to_level():
    torch.manual_seed(0)
    x = tone()
    noisy = x + 0.02 * torch.randn_like(x)
    assert float(multi_resolution_stft_loss(noisy, x)["log_magnitude"]) > 0.1
    assert float(multi_resolution_stft_loss(x * 0.5, x)["spectral_convergence"]) > 0.1


def test_stft_loss_rejects_mismatched_and_short_inputs():
    x = tone(seconds=1.0)
    with pytest.raises(ValueError, match="same shape"):
        multi_resolution_stft_loss(x, x[..., :-10])
    with pytest.raises(ValueError, match="largest FFT"):
        multi_resolution_stft_loss(x[..., :100], x[..., :100])


def test_periodicity_loss_ignores_gain_and_sign():
    x = tone()
    reference = float(periodicity_loss(x, x))
    assert float(periodicity_loss(x * 3.0, x)) == pytest.approx(reference, abs=1e-4)
    assert float(periodicity_loss(-x, x)) == pytest.approx(reference, abs=1e-4)


def test_periodicity_loss_detects_pitch_and_noise():
    x = tone(f0=220.0)
    # Resample by 6% to shift the period; linear interpolation is enough for a test probe.
    shifted = torch.nn.functional.interpolate(x, size=int(x.shape[-1] / 1.06), mode="linear",
                                              align_corners=False)
    shifted = torch.nn.functional.pad(shifted, (0, x.shape[-1] - shifted.shape[-1]))
    torch.manual_seed(1)
    breathy = x + 0.05 * torch.randn_like(x)
    baseline = float(periodicity_loss(x, x))
    assert float(periodicity_loss(shifted, x)) > 10 * baseline
    assert float(periodicity_loss(breathy, x)) > 10 * baseline


def test_energy_loss_splits_gain_from_shape():
    x = tone()
    gained = energy_envelope_loss(x * 2.0, x)
    assert float(gained["envelope_shape"]) == pytest.approx(0.0, abs=1e-5)
    assert float(gained["envelope_level"]) == pytest.approx(math.log(2.0), abs=1e-3)
    # A 20 ms shift deforms the contour (it is the term that notices moved syllables).
    shifted = torch.roll(x, int(0.02 * 24000), dims=-1)
    assert float(energy_envelope_loss(shifted, x)["envelope_shape"]) > 1e-2


def test_energy_loss_notices_a_dropped_syllable():
    x = tone(seconds=1.0)
    muted = x.clone()
    muted[..., 12000:16000] = 0.0
    assert float(energy_envelope_loss(muted, x)["envelope_shape"]) > 0.1


def test_residual_penalty_measures_the_edit_only():
    x = tone()
    assert float(residual_penalty(x, x)) == 0.0
    assert float(residual_penalty(x + 0.1, x)) == pytest.approx(0.01, rel=1e-5)


def test_humanization_loss_breakdown_matches_weighted_sum_and_backprops():
    torch.manual_seed(0)
    mechanical = tone()
    estimate = mechanical.clone().requires_grad_(True)
    target = mechanical + 0.01 * torch.randn_like(mechanical)
    total, breakdown = humanization_loss(estimate, target, mechanical)
    expected = sum(DEFAULT_WEIGHTS[name] * value for name, value in breakdown.items()
                   if name != "total")
    assert float(total.detach()) == pytest.approx(expected, rel=1e-4)
    assert set(breakdown) == set(DEFAULT_WEIGHTS) | {"total"}
    total.backward()
    assert estimate.grad is not None and torch.isfinite(estimate.grad).all()


def test_humanization_loss_rejects_unknown_weights():
    x = tone()
    with pytest.raises(ValueError, match="unknown loss terms"):
        humanization_loss(x, x, x, weights={"perceptual": 1.0})
