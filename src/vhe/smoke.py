"""Tiny cross-platform forward/backward probe, NOT VHE training on real voice."""

from __future__ import annotations

import math
from typing import Any

from .backend import inspect_runtime, load_torch, resolve_device


def run_smoke(
    device: str = "auto", *, batch_size: int = 1, samples: int = 2048,
    steps: int = 3, seed: int = 42,
) -> dict[str, Any]:
    """Exercise Conv1d forward/backprop/optimizer with synthetic tensors.

    The synthetic target is deliberately an amplitude-scaled sine, never a
    natural-speech reference. Passing does NOT establish humanization quality.
    """
    if not 1 <= batch_size <= 8 or not 256 <= samples <= 48000 or not 1 <= steps <= 10:
        raise ValueError("For smoke, require batch_size 1..8, samples 256..48000, steps 1..10")
    torch = load_torch()
    from .model import ResidualVHE

    target_device = resolve_device(device, torch_module=torch)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    model = ResidualVHE().to(device=target_device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    time = torch.arange(samples, dtype=torch.float32, device=target_device) / 24000
    # Synthetic fixture is audio-like, but not a real recording or a learning dataset.
    wave = (0.3 * torch.sin(2 * math.pi * 220.0 * time)).view(1, 1, -1)
    x = wave.expand(batch_size, -1, -1).contiguous()
    fake_target = x * 0.85
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        if y.shape != x.shape or y.device != x.device or y.dtype != x.dtype:
            raise AssertionError("VHE output shape/device/dtype mismatch")
        loss = torch.mean((y - fake_target) ** 2)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite smoke loss")
        loss.backward()
        last_grad = model.network[-1].weight.grad
        if last_grad is None or not torch.isfinite(last_grad).all().item():
            raise FloatingPointError("Missing or invalid gradients")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    report = inspect_runtime(device)
    report.update({
        "status": "backend-smoke-passed-not-humanization",
        "batch_size": batch_size,
        "samples": samples,
        "steps": steps,
        "seed": seed,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "losses": losses,
        "output_shape": list(y.shape),
        "output_dtype": str(y.dtype),
    })
    if target_device.type == "cuda":
        report["peak_cuda_memory_mb"] = round(torch.cuda.max_memory_allocated(target_device) / 1048576, 2)
    return report
