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
    manifest = sub.add_parser("manifest", help="Build a speaker-grouped manifest from a local AISHELL-3 root")
    manifest.add_argument("--root", required=True, help="AISHELL-3 directory holding train/ and test/")
    manifest.add_argument("--output", required=True, help="JSONL manifest path; refuses to overwrite")
    manifest.add_argument("--dev-fraction", type=float, default=0.1,
                          help="share of official train speakers held out as dev (speaker-grouped)")
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--no-hashes", action="store_true", help="skip per-file SHA256 (fast inventory)")
    manifest.add_argument("--fail-on-errors", action="store_true",
                          help="exit 2 if any utterance was rejected")
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
    elif args.command == "manifest":
        from .data import (apply_speaker_split, build_manifest, carve_speaker_split,
                           check_no_speaker_leak, write_manifest)
        try:
            built = build_manifest(args.root, compute_hashes=not args.no_hashes)
            grouped = apply_speaker_split(built, carve_speaker_split(
                built, dev_fraction=args.dev_fraction, seed=args.seed))
            check_no_speaker_leak(grouped)  # raises instead of writing a leaky manifest
            digest = write_manifest(args.output, grouped)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            parser.exit(2, f"Manifest build failed: {exc}\n")
        stats = grouped.stats()
        print(json.dumps({
            "root": grouped.root,
            "output": args.output,
            "manifest_sha256": digest,
            "content_sha256": grouped.content_hash(),
            "stats": stats,
            "issues": [issue.as_dict() for issue in grouped.issues[:20]],
            "status": "manifest-only; no audio copied",
        }, indent=2, ensure_ascii=False))
        if args.fail_on_errors and grouped.errors:
            parser.exit(2, f"{len(grouped.errors)} utterance(s) rejected; see {args.output}\n")
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
