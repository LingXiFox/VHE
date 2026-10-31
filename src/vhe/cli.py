"""CLI for mechanical baseline and CUDA/MPS/CPU runtime diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

from .synth import PRESETS, synthesize, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="VHE research tools (mechanical baseline; neural backend smoke, not trained humanization)")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synth", help="Generate deterministic mechanical vowel audio")
    synth.add_argument("--phonemes", default="a,i,u", help="comma-separated: a,i,u only")
    synth.add_argument("--preset", choices=tuple(PRESETS), default="yang")
    synth.add_argument("--f0", type=float, default=220.0)
    synth.add_argument("--duration", type=float, default=1.2)
    synth.add_argument("--sr", type=int, default=24000)
    synth.add_argument("--seed", type=int, default=42)
    synth.add_argument("--output", required=True)
    doctor = sub.add_parser("doctor", help="Inspect local PyTorch CUDA/MPS/CPU availability")
    doctor.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    smoke = sub.add_parser("smoke", help="Test neural model forward/backward; NOT humanization")
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--samples", type=int, default=2048)
    smoke.add_argument("--steps", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "doctor":
        from .backend import inspect_runtime
        report = inspect_runtime(args.device)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if "error" in report:
            parser.exit(2, "Device unavailable or torch not installed; see docs/PLATFORMS.md\n")
    elif args.command == "smoke":
        from .smoke import run_smoke
        try:
            report = run_smoke(args.device, batch_size=args.batch_size, samples=args.samples,
                               steps=args.steps, seed=args.seed)
        except (RuntimeError, ValueError, FloatingPointError, AssertionError) as exc:
            parser.exit(2, f"Backend smoke failed: {exc}\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "synth":
        phones = [s.strip() for s in args.phonemes.split(",")]
        audio = synthesize(phones, f0_hz=args.f0, duration_s=args.duration,
                           sample_rate=args.sr, seed=args.seed, template=PRESETS[args.preset])
        target = Path(args.output)
        write_wav(target, audio, args.sr)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        print(json.dumps({"output": str(target), "samples": len(audio), "sr": args.sr,
                          "sha256": sha, "status": "mechanical-baseline-only"}, indent=2))


if __name__ == "__main__":
    main()
