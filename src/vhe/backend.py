"""Runtime portability layer for CPU, NVIDIA CUDA and Apple MPS.

A device is selected once, then tensors and models use exactly that device.
Missing explicitly requested accelerators are errors, not silent CPU fallbacks.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

DEVICES = ("auto", "cpu", "cuda", "mps")


def load_torch() -> Any:
    """Keep the mechanical source-filter CLI usable without installing PyTorch."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the correct CUDA/macOS build first; "
            "see docs/PLATFORMS.md."
        ) from exc
    return torch


def resolve_device(requested: str = "auto", *, torch_module: Any = None) -> Any:
    """Return torch.device, rejecting unavailable explicit accelerators.

    Priority auto: CUDA > MPS > CPU. CUDA/MPS never silently downgrade.
    """
    if requested not in DEVICES:
        raise ValueError(f"Unknown device {requested!r}; choose from {DEVICES}.")
    torch = torch_module if torch_module is not None else load_torch()
    cuda_ok = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "auto":
        selected = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    else:
        selected = requested
    if selected == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA requested but unavailable. Check NVIDIA driver and CUDA-enabled PyTorch wheel.")
    if selected == "mps" and not mps_ok:
        raise RuntimeError("MPS requested but unavailable. Check Apple Silicon, macOS and MPS-enabled PyTorch.")
    return torch.device(selected)


def inspect_runtime(requested: str = "auto") -> dict[str, Any]:
    """Read-only report. Useful even if torch has not been installed."""
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "requested_device": requested,
        "torch_installed": False,
        "selected_device": None,
    }
    try:
        torch = load_torch()
    except RuntimeError as exc:
        result["error"] = str(exc)
        return result

    result["torch_installed"] = True
    result["torch_version"] = str(torch.__version__)
    result["torch_cuda_runtime"] = str(torch.version.cuda) if torch.version.cuda else None
    result["cuda_available"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    result["mps_built"] = bool(mps is not None and mps.is_built())
    result["mps_available"] = bool(mps is not None and mps.is_available())
    if result["cuda_available"]:
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
    try:
        result["selected_device"] = str(resolve_device(requested, torch_module=torch))
    except (RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    result["precision_policy"] = "float32 (all platforms; mixed precision disabled in MVP)"
    return result
