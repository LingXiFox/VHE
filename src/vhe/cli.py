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
    manifest.add_argument("--split-sizes", default="auto",
                          help="speaker counts per split as train/dev/test (default: auto, "
                               "which gives 180/19/19 for the 218 AISHELL-3 speakers)")
    manifest.add_argument("--seed", type=int, default=None,
                          help="speaker-split seed (default: vhe.data.SPLIT_SEED)")
    manifest.add_argument("--no-hashes", action="store_true", help="skip per-file SHA256 (fast inventory)")
    manifest.add_argument("--fail-on-errors", action="store_true",
                          help="exit 2 if any utterance was rejected")
    scan = sub.add_parser("scan", help="Scan annotations + WAV headers into a light manifest and a coverage report")
    scan.add_argument("--root", required=True, help="AISHELL-3 directory holding train/ and test/")
    scan.add_argument("--manifest", required=True, help="JSONL light manifest path; refuses to overwrite")
    scan.add_argument("--report", required=True, help="JSON coverage report path; refuses to overwrite")
    scan.add_argument("--split-sizes", default="auto",
                      help="speaker counts per split as train/dev/test (default: auto)")
    scan.add_argument("--seed", type=int, default=None,
                      help="speaker-split seed (default: vhe.data.SPLIT_SEED)")
    scan.add_argument("--top", type=int, default=12,
                      help="how many missing units to rank for the next inventory expansion")
    scan.add_argument("--fail-on-errors", action="store_true",
                      help="exit 2 if any utterance was rejected")
    audit = sub.add_parser("audit", help="Cross-check train/label_train-set.txt against train/content.txt")
    audit.add_argument("--root", required=True)
    audit.add_argument("--output", required=True, help="JSON audit path; refuses to overwrite")
    annotate = sub.add_parser("annotate", help="Copy a light manifest with annotation flags merged in")
    annotate.add_argument("--manifest", required=True, help="Existing JSONL manifest to read")
    annotate.add_argument("--output", required=True, help="New JSONL manifest; refuses to overwrite")
    annotate.add_argument("--root", default=None,
                          help="Corpus root for the audit (default: the manifest meta root)")
    coverage = sub.add_parser("coverage", help="Recompute coverage from a manifest under the current inventory")
    coverage.add_argument("--manifest", required=True)
    coverage.add_argument("--output", required=True, help="JSON coverage report; refuses to overwrite")
    coverage.add_argument("--top", type=int, default=12)
    refresh = sub.add_parser("refresh", help="Re-score a manifest's renderable flags under the current inventory")
    refresh.add_argument("--manifest", required=True, help="Existing JSONL manifest to read")
    refresh.add_argument("--output", required=True, help="New JSONL manifest; refuses to overwrite")
    refresh.add_argument("--root", default=None,
                         help="Corpus root override (default: the manifest meta root)")
    aligner_train = sub.add_parser(
        "aligner-train",
        help="Train the chain-HMM forced aligner on a manifest subset and save the model")
    aligner_train.add_argument("--manifest", required=True, help="Light JSONL manifest")
    aligner_train.add_argument("--root", default=None,
                               help="Corpus root (default: the manifest meta root)")
    aligner_train.add_argument("--output-model", required=True, help="NPZ model; refuses overwrite")
    aligner_train.add_argument("--output-report", required=True,
                               help="JSON training report; refuses overwrite")
    aligner_train.add_argument("--splits", default="train", help="comma-separated split names")
    aligner_train.add_argument("--speakers", type=int, default=None,
                               help="number of speakers to sample (default: all in the splits)")
    aligner_train.add_argument("--per-speaker", type=int, default=1)
    aligner_train.add_argument("--iterations", type=int, default=4)
    aligner_train.add_argument("--states-per-phone", type=int, default=3)
    aligner_train.add_argument("--components", type=int, default=1)
    aligner_train.add_argument("--frame-cap", type=int, default=2500)
    aligner_train.add_argument("--estimate-transitions", action="store_true",
                               help="re-estimate per-state stay probabilities (EXP-0022 ablation)")
    aligner_train.add_argument("--include-blocked", action="store_true",
                               help="also sample utterances the inventory cannot spell")
    aligner_train.add_argument("--seed", type=int, default=42)
    align = sub.add_parser(
        "align", help="Force-align utterances with a trained model and write a span index")
    align.add_argument("--manifest", required=True)
    align.add_argument("--root", default=None)
    align.add_argument("--model", required=True, help="NPZ aligner from aligner-train")
    align.add_argument("--output", required=True, help="JSONL span index; refuses overwrite")
    align.add_argument("--splits", default="train")
    align.add_argument("--speakers", type=int, default=None)
    align.add_argument("--per-speaker", type=int, default=1)
    align.add_argument("--limit", type=int, default=None,
                       help="cap utterances actually aligned (the index records skips)")
    align.add_argument("--include-blocked", action="store_true",
                       help="also align utterances the inventory cannot spell (they skip)")
    align.add_argument("--seed", type=int, default=42)
    pair_cache = sub.add_parser(
        "pair-cache", help="Render aligned utterances into a training-ready pair cache")
    pair_cache.add_argument("--index", required=True, help="JSONL span index from 'align'")
    pair_cache.add_argument("--manifest", required=True, help="Light JSONL manifest (transcripts)")
    pair_cache.add_argument("--root", required=True)
    pair_cache.add_argument("--output", required=True, help="New pair cache directory")
    pair_cache.add_argument("--splits", default="train")
    pair_cache.add_argument("--template", default="yang")
    pair_cache.add_argument("--backend", default="mandarin_formant",
                            help="synth backend name (default mandarin_formant, whose inventory "
                                 "matches the manifest's renderable flags)")
    pair_cache.add_argument("--sample-rate", type=int, default=24000)
    pair_cache.add_argument("--limit", type=int, default=None)
    pair_cache.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Train VHE on a pair cache (P2 small-scale runs)")
    train.add_argument("--cache", required=True, help="Pair cache built by 'pair-cache'")
    train.add_argument("--dev-cache", default=None)
    train.add_argument("--output", required=True,
                       help="Run directory for logs and step-named checkpoints")
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--crop-s", type=float, default=1.6)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--channels", type=int, default=16)
    train.add_argument("--device", default="auto")
    train.add_argument("--eval-every", type=int, default=50)
    train.add_argument("--checkpoint-every", type=int, default=50)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--overfit", type=int, default=0,
                       help="reuse one fixed batch of N pairs instead of random crops")
    train.add_argument("--resume", default=None, help="step-named checkpoint to continue from")
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "aligner-train":
        from .data.align_index import (build_training_items, read_manifest, resolve_root,
                                       select_rows)
        from .data.aligner import train_aligner

        meta, rows = read_manifest(args.manifest)
        root = resolve_root(meta, args.root)
        picked = select_rows(rows, splits=args.splits.split(","),
                             speakers=args.speakers, per_speaker=args.per_speaker,
                             seed=args.seed, renderable_only=not args.include_blocked)
        skipped: list[dict] = []
        items = build_training_items(picked, root, on_skip=skipped.append)
        if not items:
            parser.exit(3, "no utterance could be prepared for aligner training\n")
        model, report = train_aligner(
            items, states_per_phone=args.states_per_phone, n_components=args.components,
            iterations=args.iterations, seed=args.seed, frame_cap=args.frame_cap,
            estimate_transitions=args.estimate_transitions)
        report.update({"manifest": str(args.manifest), "root": str(root),
                       "splits": args.splits.split(","), "speakers_requested": args.speakers,
                       "per_speaker": args.per_speaker, "rows_selected": len(picked),
                       "renderable_only": not args.include_blocked, "skipped": skipped})
        if Path(args.output_model).exists() or Path(args.output_report).exists():
            parser.exit(2, "refusing to overwrite an existing model or report\n")
        model.to_npz(args.output_model, meta={"trained_from": str(args.manifest),
                                              "report": report})
        Path(args.output_report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: report[key] for key in
                          ("utterances", "frames", "hours", "states", "history")},
                         indent=2, ensure_ascii=False))
    elif args.command == "align":
        from .data.align_index import (align_rows, index_fingerprint, read_manifest,
                                       resolve_root, select_rows, write_index)
        from .data.aligner import PhoneAligner

        meta, rows = read_manifest(args.manifest)
        root = resolve_root(meta, args.root)
        picked = select_rows(rows, splits=args.splits.split(","),
                             speakers=args.speakers, per_speaker=args.per_speaker,
                             seed=args.seed, renderable_only=not args.include_blocked)
        if args.limit is not None:
            picked = picked[:args.limit]
        aligner = PhoneAligner.from_npz(args.model)
        skipped: list[dict] = []

        def on_skip(record: dict) -> None:
            skipped.append(record)

        entries = [entry.as_dict() if hasattr(entry, "as_dict") else entry
                   for entry in align_rows(aligner, picked, root, on_skip=on_skip)]
        records = [entry for entry in entries if entry.get("type") == "utterance"]
        target = write_index(args.output, entries,
                             meta={"manifest": str(args.manifest), "root": str(root),
                                   "model": str(args.model), "splits": args.splits.split(","),
                                   "seed": args.seed, "selected": len(picked),
                                   "renderable_only": not args.include_blocked})
        print(json.dumps({"index": str(target), "selected": len(picked),
                          "aligned": len(records), "skipped": len(entries) - len(records),
                          "sha256": index_fingerprint(target)}, indent=2))
    elif args.command == "pair-cache":
        from .data.pair_cache import build_pair_cache

        try:
            meta = build_pair_cache(args.index, args.manifest, args.root, args.output,
                                    template=args.template, backend_name=args.backend,
                                    sample_rate=args.sample_rate,
                                    splits=args.splits.split(","), limit=args.limit,
                                    seed=args.seed)
        except FileExistsError as exc:
            parser.exit(2, f"{exc}\n")
        print(json.dumps({key: meta[key] for key in
                          ("requested", "written", "rejected", "rejected_reasons")},
                         indent=2, ensure_ascii=False))
    elif args.command == "train":
        from .train import TrainConfig, train_model

        config = TrainConfig(cache=args.cache, dev_cache=args.dev_cache, output=args.output,
                             steps=args.steps, batch_size=args.batch_size, crop_s=args.crop_s,
                             learning_rate=args.learning_rate, channels=args.channels,
                             device=args.device, eval_every=args.eval_every,
                             checkpoint_every=args.checkpoint_every, log_every=args.log_every,
                             overfit=args.overfit, seed=args.seed)
        try:
            summary = train_model(config, resume=args.resume, progress=True)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(2, f"Training failed: {exc}\n")
        print(json.dumps({key: summary[key] for key in
                          ("steps", "device", "pairs", "first_total", "best_total")},
                         indent=2, ensure_ascii=False))
    elif args.command == "doctor":
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
        from .data import (SPLIT_SEED, build_manifest, check_no_speaker_leak,
                           parse_split_sizes, write_manifest)
        try:
            sizes = parse_split_sizes(args.split_sizes)
            grouped = build_manifest(args.root, compute_hashes=not args.no_hashes,
                                     split_sizes=sizes,
                                     split_seed=SPLIT_SEED if args.seed is None else args.seed)
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
    elif args.command == "scan":
        from .data import SPLIT_SEED, parse_split_sizes
        from .data.corpus import scan_corpus, write_corpus_manifest
        target = Path(args.report)
        try:
            # Check both outputs before the 36 s scan, otherwise a pre-existing report
            # aborts the run after the manifest was already written.
            for existing in (Path(args.manifest), target):
                if existing.exists():
                    raise FileExistsError(f"refusing to overwrite an existing file: {existing}")
            index = scan_corpus(args.root, split_sizes=parse_split_sizes(args.split_sizes),
                                split_seed=SPLIT_SEED if args.seed is None else args.seed)
            digest = write_corpus_manifest(args.manifest, index)
            report = index.report(top_units=args.top)
            report["manifest_sha256"] = digest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            parser.exit(2, f"Corpus scan failed: {exc}\n")
        print(json.dumps({
            "manifest": args.manifest, "manifest_sha256": digest, "report": str(target),
            "utterances": len(index.entries), "errors": len(index.errors),
            "syllable_coverage": round(index.stats.syllable_coverage, 4),
            "utterance_coverage": round(index.stats.utterance_coverage, 4),
            "status": "header-only scan; no audio decoded",
        }, indent=2, ensure_ascii=False))
        if args.fail_on_errors and index.errors:
            parser.exit(2, f"{len(index.errors)} utterance(s) rejected; see {args.report}\n")
    elif args.command == "audit":
        from .data.corpus import audit_prosody_labels
        try:
            report = audit_prosody_labels(args.root, limit=50)
        except FileNotFoundError as exc:
            parser.exit(2, f"Annotation audit failed: {exc}\n")
        target = Path(args.output)
        if target.exists():
            parser.exit(2, f"refusing to overwrite an existing audit: {target}\n")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(json.dumps({key: report[key] for key in
                          ("checked", "matched", "mismatched", "format_errors",
                           "missing_text", "header_comments")} |
                         {"flagged_utterances": len(report["flags"]),
                          "output": str(target)}, indent=2, ensure_ascii=False))
    elif args.command == "annotate":
        from .data.corpus import annotate_manifest
        try:
            summary = annotate_manifest(args.manifest, args.output, root=args.root)
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            parser.exit(2, f"Annotation flags not applied: {exc}\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "coverage":
        from .data.corpus import coverage_report, read_corpus_manifest
        target = Path(args.output)
        if target.exists():
            parser.exit(2, f"refusing to overwrite an existing report: {target}\n")
        try:
            entries, _meta = read_corpus_manifest(args.manifest)
        except (FileNotFoundError, ValueError, OSError) as exc:
            parser.exit(2, f"Coverage recompute failed: {exc}\n")
        report = coverage_report(entries, top_units=args.top)
        report["manifest"] = str(args.manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(json.dumps({"output": str(target),
                          "utterances": report["stats"]["utterances"],
                          "syllables": report["stats"]["syllables"],
                          "syllable_coverage": report["stats"]["syllable_coverage"],
                          "utterance_coverage": report["stats"]["utterance_coverage"],
                          "quality_flags": report["quality_flags"]},
                         indent=2, ensure_ascii=False))
    elif args.command == "refresh":
        from .data.corpus import refresh_manifest
        try:
            summary = refresh_manifest(args.manifest, args.output, root=args.root)
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            parser.exit(2, f"Manifest refresh failed: {exc}\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
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
