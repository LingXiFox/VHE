"""Loss suite for VHE v0.1 (see docs/LOSS_DESIGN.md for the rationale).

Every term is differentiable, CPU/MPS/CUDA portable, FP32-only and free of external
weights. Nothing here measures "naturalness": the terms are proxies that must be
reported together with audio, never as a single score.

Terms
-----
spectral      multi-resolution STFT (spectral convergence + log-magnitude L1) between the
              edit and the real recording. Phase-blind, so it cannot notice smearing.
periodicity   L1 between normalised frame autocorrelation *profiles* of edit and target.
              Penalises both a changed period (pitch drift) and a changed peak height
              (buzz removed or noise added), which a plain F0 tracker cannot express.
envelope      frame log-energy shape (mean-removed) plus a global level term. Protects
              syllable timing and the loudness contour, and it is the term that notices a
              dropped consonant or a new silence gap.
residual      L2 of the edit itself. Keeps "do nothing" cheap so the model only spends
              energy where a term above pays for it.
"""

from __future__ import annotations

from typing import Any

import torch


def _frames(x: torch.Tensor, frame: int, hop: int) -> torch.Tensor:
    if x.shape[-1] < frame:
        raise ValueError(f"signal of {x.shape[-1]} samples is shorter than frame {frame}")
    return x.unfold(-1, frame, hop)


def multi_resolution_stft_loss(
    estimate: torch.Tensor, target: torch.Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048), hop_ratio: int = 4,
) -> dict[str, torch.Tensor]:
    """Parallel WaveGAN style MRSTFT: spectral convergence + log-magnitude L1.

    Both inputs must be [B, 1, N] with N >= max(fft_sizes). The two sub-terms are returned
    separately because spectral convergence is dominated by loud frames while log-magnitude
    L1 is closer to a perceptual weighting; mixing them silently hides which one moved.
    """
    if estimate.shape != target.shape:
        raise ValueError("estimate and target must have the same shape")
    if estimate.shape[-1] < max(fft_sizes):
        raise ValueError("signal shorter than the largest FFT size")
    convergence: list[torch.Tensor] = []
    log_magnitude: list[torch.Tensor] = []
    for size in fft_sizes:
        hop = max(1, size // hop_ratio)
        window = torch.hann_window(size, device=estimate.device, dtype=estimate.dtype)
        spec_est = torch.stft(estimate.squeeze(1), n_fft=size, hop_length=hop, win_length=size,
                              window=window, center=True, return_complex=True)
        spec_tgt = torch.stft(target.squeeze(1), n_fft=size, hop_length=hop, win_length=size,
                              window=window, center=True, return_complex=True)
        magnitude_est = spec_est.abs()
        magnitude_tgt = spec_tgt.abs()
        convergence.append(
            torch.linalg.vector_norm(magnitude_est - magnitude_tgt)
            / (torch.linalg.vector_norm(magnitude_tgt) + 1e-7)
        )
        log_magnitude.append(
            torch.mean(torch.abs(torch.log(magnitude_est + 1e-5) - torch.log(magnitude_tgt + 1e-5)))
        )
    return {
        "spectral_convergence": torch.stack(convergence).mean(),
        "log_magnitude": torch.stack(log_magnitude).mean(),
    }


def _autocorrelation_profile(
    x: torch.Tensor, frame: int, hop: int, max_lag: int,
) -> torch.Tensor:
    """Normalised autocorrelation profile of every frame, shape [B, frames, max_lag].

    Frames are transformed with padding to the next power of two so the circular
    autocorrelation of the windowed frame cannot wrap real lag content into the low lags.
    r(0) is the normaliser, so the profile is invariant to gain and sign.
    """
    frames = _frames(x, frame, hop)
    n_fft = 1
    while n_fft < 2 * frame:
        n_fft *= 2
    window = torch.hann_window(frame, device=x.device, dtype=x.dtype)
    spectrum = torch.fft.rfft(frames * window, n=n_fft)
    autocorrelation = torch.fft.irfft(spectrum.real ** 2 + spectrum.imag ** 2, n=n_fft)
    autocorrelation = autocorrelation[..., :max_lag]
    return autocorrelation / (autocorrelation[..., :1].abs() + 1e-8)


def periodicity_loss(
    estimate: torch.Tensor, target: torch.Tensor,
    sample_rate: int = 24000, frame_ms: float = 32.0, hop_ms: float = 8.0,
    f0_min_hz: float = 65.0,
) -> torch.Tensor:
    """L1 distance between autocorrelation profiles, weighted by target frame energy.

    A pitch tracker returns one lag per frame and throws away the *height* of the
    periodicity peak; that height is exactly what changes when a buzzy excitation becomes a
    breathy one. Comparing the whole profile keeps both, and because the profile is
    normalised it does not fight the level term.
    """
    frame = int(round(frame_ms * sample_rate / 1000))
    hop = int(round(hop_ms * sample_rate / 1000))
    max_lag = int(round(sample_rate / f0_min_hz))
    if estimate.shape[-1] < frame + max_lag:
        raise ValueError("signal too short for the requested frame/max_lag")
    profile_est = _autocorrelation_profile(estimate, frame, hop, max_lag)
    profile_tgt = _autocorrelation_profile(target, frame, hop, max_lag)
    frames = _frames(target, frame, hop)
    energy = frames.pow(2).mean(dim=-1).sqrt()
    weight = (energy / (energy.max(dim=-1, keepdim=True).values + 1e-8)).clamp(max=1.0)
    weight = weight.detach()  # weights are a mask, not something the model should game
    distance = torch.abs(profile_est - profile_tgt).mean(dim=-1)
    return torch.sum(distance * weight) / (torch.sum(weight) + 1e-8)


def energy_envelope_loss(
    estimate: torch.Tensor, target: torch.Tensor,
    sample_rate: int = 24000, frame_ms: float = 25.0, hop_ms: float = 10.0,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Shape + level of the frame-energy contour.

    shape: L1 between mean-removed log energies, i.e. "did the syllables stay where they
    were". level: |log rms ratio|, i.e. "did the whole thing get louder". Splitting them
    stops a uniform gain error from being paid for by deforming the contour.
    """
    frame = int(round(frame_ms * sample_rate / 1000))
    hop = int(round(hop_ms * sample_rate / 1000))
    energy_est = _frames(estimate, frame, hop).pow(2).mean(dim=-1).clamp_min(eps).sqrt()
    energy_tgt = _frames(target, frame, hop).pow(2).mean(dim=-1).clamp_min(eps).sqrt()
    log_est = torch.log(energy_est)
    log_tgt = torch.log(energy_tgt)
    shape = torch.mean(torch.abs((log_est - log_est.mean(dim=-1, keepdim=True))
                                 - (log_tgt - log_tgt.mean(dim=-1, keepdim=True))))
    level = torch.mean(torch.abs(log_est.mean(dim=-1) - log_tgt.mean(dim=-1)))
    return {"envelope_shape": shape, "envelope_level": level}


def residual_penalty(estimate: torch.Tensor, mechanical: torch.Tensor) -> torch.Tensor:
    """L2 of the edit. Keeps the identity solution cheap; too large a weight suppresses the
    micro-variation the model is supposed to add, so it is swept in EXP-0009, not guessed."""
    return torch.mean((estimate - mechanical) ** 2)


DEFAULT_WEIGHTS: dict[str, float] = {
    "spectral_convergence": 1.0,
    "log_magnitude": 1.0,
    "periodicity": 1.0,
    "envelope_shape": 1.0,
    "envelope_level": 1.0,
    "residual": 1e-3,
}


def humanization_loss(
    estimate: torch.Tensor, target: torch.Tensor, mechanical: torch.Tensor,
    sample_rate: int = 24000, weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weighted sum plus the unweighted per-term breakdown for logging."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown loss terms: {sorted(unknown)}")
    terms: dict[str, torch.Tensor] = {}
    terms.update(multi_resolution_stft_loss(estimate, target))
    terms["periodicity"] = periodicity_loss(estimate, target, sample_rate=sample_rate)
    terms.update(energy_envelope_loss(estimate, target, sample_rate=sample_rate))
    terms["residual"] = residual_penalty(estimate, mechanical)
    total = torch.zeros((), device=estimate.device, dtype=estimate.dtype)
    for name, value in terms.items():
        total = total + weights[name] * value
    breakdown = {name: float(value.detach().cpu()) for name, value in terms.items()}
    breakdown["total"] = float(total.detach().cpu())
    return total, breakdown
