#!/usr/bin/env python3
"""Measure GPU memory for the P2 training recipe (EXP-0028).

Reports torch.cuda.max_memory_allocated / max_memory_reserved for a matrix of
channel and batch settings, using the same data, crop length and loss weights as
EXP-0027. It measures memory only: no claim is made about audio quality, and the
20-step runs are not training results. Each run directory must be new; the probe
refuses to reuse one so a failed attempt is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
if len(_HERE.parents) >= 3 and (_HERE.parents[2] / "src").is_dir():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))

from vhe.model import build_model  # noqa: E402
from vhe.train import TrainConfig, train_model  # noqa: E402


VARIANTS = (
    ("ch16_b8", {"channels": 16, "batch_size": 8}),
    ("ch32_b8", {"channels": 32, "batch_size": 8}),
    ("ch16_b16", {"channels": 16, "batch_size": 16}),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--crop-s", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", help="JSON report path")
    args = parser.parse_args()

    import torch  # noqa: E402

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this probe only reports GPU memory")

    root = Path(args.out_root)
    records = []
    for tag, overrides in VARIANTS:
        output = root / tag
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing run directory: {output}")
        config = TrainConfig(cache=args.cache, output=str(output), steps=args.steps,
                             crop_s=args.crop_s, seed=args.seed, device=args.device,
                             model="unet", overfit=8, checkpoint_every=0,
                             log_every=max(1, args.steps), eval_every=0, **overrides)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        summary = train_model(config)
        torch.cuda.synchronize()
        parameter_count = sum(parameter.numel()
                                for parameter in build_model(config.model,
                                                             channels=config.channels).parameters())
        record = {
            "tag": tag,
            "channels": config.channels,
            "batch_size": config.batch_size,
            "steps": config.steps,
            "crop_s": config.crop_s,
            "pairs": 8,
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
            "parameter_count": parameter_count,
            "last_total": summary["last"]["total"],
            "output": str(output),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    report = {
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_mib": total_mib,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "crop_s": args.crop_s,
        "overfit_pairs": 8,
        "runs": records,
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"device_total_memory_mib": total_mib}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
