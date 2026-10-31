"""Device-neutral prototype model weights; no optimizer/resume support yet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend import load_torch, resolve_device


def save_weights(path: str | Path, model: Any) -> Path:
    """Write portable CPU state_dict, refusing to overwrite an experiment artifact."""
    torch = load_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents clobbering research artifacts, including on Windows.
    state = {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}
    with path.open("xb") as output:
        torch.save(state, output)
    return path


def load_weights(path: str | Path, model: Any, *, device: str = "auto") -> Any:
    """Load weights safely on CPU then move model to selected CUDA/MPS/CPU backend."""
    torch = load_torch()
    selected = resolve_device(device, torch_module=torch)
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device=selected, dtype=torch.float32)
