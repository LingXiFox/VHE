"""Re-score AISHELL-3 coverage against the *current* phoneme inventory.

Reads the v3 manifest only (no WAV is touched), so every phoneme batch can be re-measured
in a few seconds instead of a second 26 GB header pass.

    python experiments/EXP-0016/probe_coverage.py \
        --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v3/manifest.jsonl \
        --stage 0 --out experiments/EXP-0016/logs/coverage_stage0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vhe.data.corpus import coverage_report, read_corpus_manifest  # noqa: E402
from vhe.data.g2p import SYNTH_FINAL_TO_PHONEME, SUPPORTED_ONSETS  # noqa: E402
from vhe.synth import MANDARIN_PHONEME_IDS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if args.out.exists():
        parser.error(f"refusing to overwrite an existing report: {args.out}")

    entries, meta = read_corpus_manifest(args.manifest)
    report = coverage_report(entries, top_units=args.top)
    report["stage"] = args.stage
    report["manifest"] = str(args.manifest)
    report["manifest_entries"] = len(entries)
    report["manifest_root"] = meta.get("root")
    report["inventory"] = {
        "size": len(MANDARIN_PHONEME_IDS),
        "phonemes": list(MANDARIN_PHONEME_IDS),
        "onsets": sorted(SUPPORTED_ONSETS),
        "finals": sorted(SYNTH_FINAL_TO_PHONEME),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    stats = report["stats"]
    print(f"stage {args.stage}: inventory {report['inventory']['size']} phonemes")
    print(f"  utterances {stats['utterances']}  syllables {stats['syllables']}")
    print(f"  syllable coverage {stats['syllable_coverage']:.6f} "
          f"({stats['renderable_syllables']}/{stats['syllables']})")
    print(f"  utterance coverage {stats['utterance_coverage']:.6f} "
          f"({stats['renderable_utterances']}/{stats['utterances']})")
    print("  blocked units (top): " + ", ".join(
        f"{unit}:{count}" for unit, count in list(stats["blocked_units"].items())[:8]))
    print("  next units:")
    for row in report["next_units"]:
        print(f"    {row['unit']:<24} +{row['syllables_unblocked']:<7} "
              f"cum {row['cumulative_unblocked']:<8} also_needs={row['also_needs']}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
