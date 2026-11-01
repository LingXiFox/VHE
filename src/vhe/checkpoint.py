"""Device-neutral checkpoints: weights for inference, full state for resumable training.

Two artifacts, deliberately different:

* save_weights / load_weights carry only the model state_dict, so an inference artifact never
  depends on an optimizer implementation or a torch version's optimizer layout.
* save_training_state / load_training_state carry the optimizer state and the step counter, and
  are written to step-named files that are never overwritten. Resume therefore picks a specific
  file, and a crashed run cannot destroy the history of the runs before it.
"""

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


def save_training_state(path: str | Path, model: Any, optimizer: Any, *, step: int,
                        history: list, config: dict, torch_module: Any = None) -> Path:
    """Write model + optimizer + step to a new step-named file; refuse to overwrite."""
    import json

    torch = torch_module or load_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {key: value.detach().to("cpu").clone()
                  for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "history": list(history),
        "config": dict(config),
        "channels": int(config.get("channels", 0)),
    }
    with path.open("xb") as output:
        torch.save(payload, output)
    # A JSON twin makes the checkpoint inspectable without torch (and without pickles).
    summary = {"step": payload["step"], "channels": payload["channels"],
               "config": payload["config"], "history_length": len(history)}
    (path.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def load_training_state(path: str | Path, model: Any, optimizer: Any, *,
                        device: str = "auto") -> dict:
    """Restore model and optimizer; the caller keeps its own history if the file has none."""
    torch = load_torch()
    selected = resolve_device(device, torch_module=torch)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=selected, dtype=torch.float32)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(selected)
    return payload