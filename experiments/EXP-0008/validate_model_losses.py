"""EXP-0008: numerical validation of the v0.1 model and the loss suite.

No training, no real recordings. Two things are measured:

1. model properties that the architecture doc claims (identity start, length contract,
   receptive field, parameter count, CPU/MPS agreement, gradient flow);
2. whether each loss term reacts to the distortion it is supposed to see, on a synthetic
   "mechanical vs pseudo-natural" fixture, including the ratios a training signal needs.

Usage:
    .venv/bin/python experiments/EXP-0008/validate_model_losses.py \
        --out experiments/EXP-0008/logs/validate.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from vhe.losses import (DEFAULT_WEIGHTS, energy_envelope_loss, humanization_loss,
                        multi_resolution_stft_loss, periodicity_loss, residual_penalty)
from vhe.model import ResidualUNet

SR = 24000


def mechanical_fixture(seconds: float = 1.0, seed: int = 0) -> torch.Tensor:
    """Synthetic 'mechanical' voice: perfect harmonics, no jitter, no noise, flat envelope."""
    generator = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    t = torch.arange(n, dtype=torch.float64) / SR
    f0 = 180.0
    signal = torch.zeros(n, dtype=torch.float64)
    for k in range(1, 12):
        signal += (1.0 / k) * torch.sin(2 * math.pi * f0 * k * t)
    # Buzzy: a hard-edged sawtooth-like excitation, then a rectangular syllable envelope.
    envelope = torch.ones(n, dtype=torch.float64)
    envelope[int(0.10 * SR):int(0.12 * SR)] = 0.0   # 20 ms 'stop' closure
    envelope[int(0.55 * SR):int(0.62 * SR)] = 0.0   # 70 ms pause
    signal = signal * envelope
    signal = signal / signal.abs().max() * 0.85
    return signal.to(torch.float32).view(1, 1, -1)


def pseudo_natural(mechanical: torch.Tensor, seed: int = 1) -> torch.Tensor:
    """Stand-in for the real recording: reference implementation of *what we want*.

    Adds the four things the mechanical fixture lacks (period jitter, shimmer, breath
    noise, damped high harmonics) using signal processing, not a neural network. It is a
    fixture, not a claim about real voices.
    """
    generator = torch.Generator().manual_seed(seed)
    x = mechanical.double()
    n = x.shape[-1]
    phase = torch.cumsum(torch.full((n,), 2 * math.pi * 180.0 / SR, dtype=torch.float64), 0)
    jitter = 1.0 + 0.004 * torch.randn(n, generator=generator, dtype=torch.float64).cumsum(0).clamp(-20, 20) / 20.0
    phase = phase * jitter
    timbre = torch.zeros(n, dtype=torch.float64)
    for k in range(1, 12):
        damp = math.exp(-0.08 * k)  # high harmonics damped: less buzz
        timbre += damp / k * torch.sin(k * phase)
    shimmer = 1.0 + 0.05 * torch.randn(max(1, n // 200), generator=generator,
                                       dtype=torch.float64).repeat_interleave(200)[:n]
    breath = 0.02 * torch.randn(n, generator=generator, dtype=torch.float64)
    envelope = (x.flatten() != 0).double()
    smooth = torch.nn.functional.conv1d(envelope.view(1, 1, -1),
                                        torch.ones(1, 1, 48, dtype=torch.float64) / 48,
                                        padding=24).view(-1)[:n]
    out = (timbre * shimmer + breath * smooth) * smooth
    out = out / out.abs().max() * 0.85
    return out.to(torch.float32).view(1, 1, -1)


def distortion(name: str, target: torch.Tensor) -> torch.Tensor:
    if name == "identity":
        return target.clone()
    if name == "noise_snr20":
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(target.shape, generator=generator)
        noise = noise / noise.std() * (target.std() * 0.1)
        return target + noise
    if name == "pitch_plus_2pct":
        size = int(target.shape[-1] / 1.02)
        shifted = torch.nn.functional.interpolate(target, size=size, mode="linear",
                                                  align_corners=False)
        return torch.nn.functional.pad(shifted, (0, target.shape[-1] - size))
    if name == "shift_20ms":
        return torch.roll(target, int(0.02 * SR), dims=-1)
    if name == "gain_2x":
        return target * 2.0
    if name == "drop_100ms":
        out = target.clone()
        out[..., int(0.3 * SR):int(0.4 * SR)] = 0.0
        return out
    if name == "dull_3tap":
        kernel = torch.ones(1, 1, 8) / 8
        return torch.nn.functional.conv1d(target, kernel, padding=4)[..., :target.shape[-1]]
    if name == "bitcrush_8bit":
        step = 2.0 / 256
        return torch.round(target / step) * step
    raise ValueError(name)


def loss_row(estimate: torch.Tensor, target: torch.Tensor, mechanical: torch.Tensor) -> dict:
    stft = multi_resolution_stft_loss(estimate, target)
    envelope = energy_envelope_loss(estimate, target)
    total, breakdown = humanization_loss(estimate, target, mechanical)
    return {
        "spectral_convergence": float(stft["spectral_convergence"]),
        "log_magnitude": float(stft["log_magnitude"]),
        "periodicity": float(periodicity_loss(estimate, target)),
        "envelope_shape": float(envelope["envelope_shape"]),
        "envelope_level": float(envelope["envelope_level"]),
        "residual": float(residual_penalty(estimate, mechanical)),
        "total": float(total.detach()),
    }


def receptive_field(model: torch.nn.Module, length: int = 16384) -> dict:
    """Influence width of a single impulse, measured, not derived from layer arithmetic.

    The shipped model starts at identity (zero-init head), so its impulse response is a
    delta of width 1 and says nothing about the architecture. Weights are therefore
    randomised with a fixed generator first; the width depends only on the conv geometry.
    """
    import copy

    probe = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    centre = length // 2
    impulse = torch.zeros(1, 1, length)
    impulse[..., centre] = 1.0
    with torch.no_grad():
        base = probe(torch.zeros(1, 1, length))
        response = probe(impulse) - probe(torch.zeros(1, 1, length))
    magnitude = response[0, 0].abs()
    threshold = float(magnitude.max()) * 1e-6
    nonzero = torch.nonzero(magnitude > threshold).flatten()
    width = int(nonzero.max() - nonzero.min()) + 1
    return {
        "left_samples": int(nonzero.min()) - centre,
        "right_samples": int(nonzero.max()) - centre,
        "width_samples": width,
        "width_ms": round(width / SR * 1000.0, 2),
        "periods_at_65hz": round(width / (SR / 65.0), 2),
        "periods_at_180hz": round(width / (SR / 180.0), 2),
    }


def _grad_scales(model: torch.nn.Module) -> dict:
    per_module: dict[str, float] = {}
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        key = name.split(".")[0]
        per_module[key] = round(per_module.get(key, 0.0) + float(parameter.grad.abs().sum()), 8)
    return {"per_block_grad_abs_sum": per_module, "parameters_without_grad": missing}


def gradient_flow(model: torch.nn.Module, mechanical: torch.Tensor, target: torch.Tensor) -> dict:
    """Gradient reach at initialisation and after one head-only step.

    The output head is zero-initialised, so d(loss)/d(anything upstream) is exactly zero at
    step 0: the first update can only move the head. This is the standard price of an
    identity-preserving start, and it has to be measured, not assumed.
    """
    import copy

    probe = copy.deepcopy(model)  # the caller's weights must stay at the identity start
    probe.zero_grad(set_to_none=True)
    total, _ = humanization_loss(probe(mechanical), target, mechanical)
    total.backward()
    at_init = _grad_scales(probe)
    all_finite = all(torch.isfinite(p.grad).all() for p in probe.parameters()
                     if p.grad is not None)

    optimizer = torch.optim.SGD(probe.parameters(), lr=1e-2)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    total, _ = humanization_loss(probe(mechanical), target, mechanical)
    total.backward()
    after_head_step = _grad_scales(probe)
    return {"at_init": at_init, "after_one_head_step": after_head_step,
            "all_finite": all_finite}


def device_agreement(model: torch.nn.Module, mechanical: torch.Tensor) -> dict:
    out = {}
    import copy

    probe = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
        cpu_edit = probe(mechanical) - mechanical
    out["cpu_edit_absmax"] = float(cpu_edit.abs().max())
    out["note"] = "weights randomised: at identity init every backend outputs an exact no-op"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_model = ResidualUNet()
        mps_model.load_state_dict(probe.state_dict())
        mps_model = mps_model.to("mps")
        with torch.no_grad():
            mps_edit = (mps_model(mechanical.to("mps")) - mechanical.to("mps")).cpu()
        out["mps_available"] = True
        out["cpu_vs_mps_edit_max_abs_diff"] = float((cpu_edit - mps_edit).abs().max())
        out["mps_finite"] = bool(torch.isfinite(mps_edit).all())
    else:
        out["mps_available"] = False
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model = ResidualUNet()
    mechanical = mechanical_fixture()
    target = pseudo_natural(mechanical)

    lengths = {}
    with torch.no_grad():
        for length in (777, 1000, 24000, 24013):
            x = torch.randn(1, 1, length, dtype=torch.float32)
            y = model(x)
            lengths[str(length)] = {
                "shape": list(y.shape),
                "max_abs_edit": float((y - x).abs().max()),
            }

    report = {
        "experiment": "EXP-0008",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "identity_and_length": lengths,
        "receptive_field": receptive_field(model),
        "loss_matrix": {},
        "loss_at_init": None,
        "gradient_flow": gradient_flow(model, mechanical, target),
        "device_agreement": device_agreement(model, mechanical),
        "weights": DEFAULT_WEIGHTS,
        "fixture": {"sample_rate": SR, "seconds": 1.0, "f0_hz": 180.0,
                    "closures_ms": [20, 70], "note": "synthetic; not a real recording"},
    }

    names = ["identity", "noise_snr20", "pitch_plus_2pct", "shift_20ms", "gain_2x",
             "drop_100ms", "dull_3tap", "bitcrush_8bit"]
    for name in names:
        report["loss_matrix"][name] = loss_row(distortion(name, target), target, mechanical)

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        identity_loss, identity_breakdown = humanization_loss(model(mechanical), target,
                                                              mechanical)
    report["loss_at_init"] = identity_breakdown
    report["loss_at_init"]["note"] = "model output is exactly the mechanical input here"

    # Margin: the loss of a small, clearly audible distortion vs the loss of a perfect pair.
    identity_total = report["loss_matrix"]["identity"]["total"]
    # The identity row is ~0 for every paired term by construction, so ratios against it are
    # ill-conditioned; absolute margins are the meaningful numbers.
    report["margin_identity_vs_distortion"] = {
        name: {"absolute": round(row["total"] - identity_total, 5),
               "ratio": round(row["total"] / max(identity_total, 1e-9), 1)}
        for name, row in report["loss_matrix"].items() if name != "identity"
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("parameters", "receptive_field", "identity_and_length")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
