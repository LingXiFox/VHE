"""Training loop for VHE (P2).

Scope: make one small batch trainable end to end, with resumable checkpoints and a log that
records what the loss terms did. This is deliberately not a research-grade trainer: no
distributed launch, no mixed precision (PLATFORMS.md keeps FP32 as the default), no LR schedule.

What the loop guarantees:
* every run writes into a fresh output directory; checkpoints are step-named and never
  overwritten, so a resumed run cannot silently lose its history;
* the dev set is evaluated with the same crop length as training and reported separately;
* the loss is vhe.losses.humanization_loss and its per-term breakdown is logged, because a
  falling total alone is not evidence of more natural speech (AGENTS.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .backend import load_torch, resolve_device
from .checkpoint import load_training_state, save_training_state
from .data.pair_cache import PairCache
from .losses import DEFAULT_WEIGHTS, humanization_loss
from .model import build_model


@dataclass
class TrainConfig:
    cache: str
    output: str
    sample_rate: int = 24000
    crop_s: float = 1.6
    batch_size: int = 4
    steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    model: str = "unet"
    channels: int = 16
    checkpoint_every: int = 50
    log_every: int = 10
    eval_every: int = 50
    eval_batches: int = 4
    dev_cache: Optional[str] = None
    overfit: int = 0
    losses: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def as_dict(self) -> dict:
        return asdict(self)


def _crop(mechanical: np.ndarray, real: np.ndarray, length: int, start: int
          ) -> tuple[np.ndarray, np.ndarray]:
    if mechanical.size < length:
        pad = length - mechanical.size
        mechanical = np.pad(mechanical, (0, pad))
        real = np.pad(real, (0, pad))
        return mechanical, real
    return mechanical[start:start + length], real[start:start + length]


class PairCrops:
    """Deterministic random crops of cached pairs, sized so a batch stacks without padding."""

    def __init__(self, cache: PairCache, *, crop_s: float, sample_rate: int,
                 limit: int = 0, seed: int = 42) -> None:
        self.cache = cache
        self.length = int(round(crop_s * sample_rate))
        self.rows = list(cache)[: limit or None]
        if not self.rows:
            raise ValueError(f"pair cache {cache.path} has no pairs")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[int(self.rng.integers(len(self.rows)))]
        mechanical, real = row.load()
        start = int(self.rng.integers(max(1, mechanical.size - self.length + 1)))
        return _crop(mechanical, real, self.length, start)

    def fixed(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[index]
        mechanical, real = row.load()
        start = max(0, (mechanical.size - self.length) // 2)
        return _crop(mechanical, real, self.length, start)


def _tensor_batch(samples: list[tuple[np.ndarray, np.ndarray]], torch, device):
    mechanical = torch.from_numpy(np.stack([item[0] for item in samples])).unsqueeze(1)
    real = torch.from_numpy(np.stack([item[1] for item in samples])).unsqueeze(1)
    return mechanical.to(device), real.to(device)


def _evaluate(model, batches, torch, device, config: TrainConfig) -> dict:
    model.eval()
    totals: list[dict] = []
    with torch.no_grad():
        for mechanical, real in batches:
            estimate = model(mechanical)
            _, breakdown = humanization_loss(estimate, real, mechanical,
                                             sample_rate=config.sample_rate,
                                             weights=config.losses)
            totals.append(breakdown)
    model.train()
    return {key: float(np.mean([item[key] for item in totals])) for key in totals[0]}


def train_model(config: TrainConfig, *, resume: Optional[str] = None,
                progress: bool = False) -> dict:
    torch = load_torch()
    device = resolve_device(config.device, torch_module=torch)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    cache = PairCache(config.cache)
    limit = config.overfit or 0
    train_crops = PairCrops(cache, crop_s=config.crop_s, sample_rate=config.sample_rate,
                            limit=limit, seed=config.seed)
    dev_batches = None
    if config.dev_cache:
        dev_cache = PairCache(config.dev_cache)
        dev_crops = PairCrops(dev_cache, crop_s=config.crop_s, sample_rate=config.sample_rate,
                              limit=limit, seed=config.seed + 1)
        dev_batches = []
        for batch_index in range(config.eval_batches):
            positions = [position % len(dev_crops) for position in
                         range(batch_index * config.batch_size,
                               (batch_index + 1) * config.batch_size)]
            samples = [dev_crops.fixed(position) for position in positions]
            dev_batches.append(_tensor_batch(samples, torch, device))

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(config.model, channels=config.channels)
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    step = 0
    history: list[dict] = []
    if resume:
        state = load_training_state(resume, model, optimizer, device=str(device))
        step, history = int(state["step"]), list(state.get("history", []))
        if int(state.get("channels", config.channels)) != config.channels:
            raise ValueError("resume checkpoint was written with a different channel count")

    log_path = output / "log.jsonl"
    # Overfit mode reuses one fixed batch so a falling loss means the model fit those frames,
    # not that the sampler wandered. Full mode draws a fresh random crop every step.
    fixed_samples = ([train_crops.fixed(position % len(train_crops))
                      for position in range(config.batch_size)] if config.overfit else None)
    model.train()
    while step < config.steps:
        samples = fixed_samples or [train_crops.sample() for _ in range(config.batch_size)]
        mechanical, real = _tensor_batch(samples, torch, device)
        estimate = model(mechanical)
        loss, breakdown = humanization_loss(estimate, real, mechanical,
                                            sample_rate=config.sample_rate,
                                            weights=config.losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        step += 1
        record = {"step": step, **breakdown}
        if config.eval_every and dev_batches and step % config.eval_every == 0:
            record["dev"] = _evaluate(model, dev_batches, torch, device, config)
        history.append(record)
        if step % config.log_every == 0 or step == config.steps:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress:
                dev = record.get("dev", {})
                print(f"step {step:5d} total {breakdown['total']:.4f} "
                      f"sc {breakdown['spectral_convergence']:.4f} "
                      f"mag {breakdown['log_magnitude']:.4f} "
                      f"per {breakdown['periodicity']:.4f}"
                      + (f" dev {dev.get('total', float('nan')):.4f}" if dev else ""))
        if config.checkpoint_every and step % config.checkpoint_every == 0:
            save_training_state(output / "checkpoints" / f"step-{step:06d}.pt", model,
                                optimizer, step=step, history=history,
                                config=config.as_dict(), torch_module=torch)

    summary = {"steps": step, "device": str(device), "pairs": len(train_crops),
               "cache": config.cache, "dev_cache": config.dev_cache,
               "last": history[-1] if history else None,
               "best_total": min((item["total"] for item in history), default=None),
               "first_total": history[0]["total"] if history else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
