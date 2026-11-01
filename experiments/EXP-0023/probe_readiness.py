"""Measure how many dev utterances yield a training-ready forced pair (EXP-0023).

The listening package stops early once a speaker has enough ready pairs, so its
ready/attempted ratio is biased. This probe walks a fixed, pre-declared sample (12 dev speakers
x 5 utterances, seed 42, renderable only) and records every pair, its issues and its readiness, so
the corpus-level yield is measurable and reproducible.

What it proves: the share of dev utterances that the v3 aligner plus the current pitch policy can
turn into a training pair, and which issue codes block the rest.
What it does not prove: that a training-ready pair is perceptually good. That is the listening gate.

Usage:
    python experiments/EXP-0023/probe_readiness.py --manifest ... --root ... --model ... \
        --out experiments/EXP-0023/logs/readiness_dev60.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vhe.data.align_index import align_rows, read_manifest, resolve_root, select_rows
from vhe.data.aligner import PhoneAligner
from vhe.data.pairing import build_pair
from vhe.synth.contract import PRESETS
from vhe.synth.mandarin import MandarinFormantSynth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--speakers", type=int, default=12)
    parser.add_argument("--per-speaker", type=int, default=5)
    parser.add_argument("--template", default="yang")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        parser.exit(2, f"{out} exists; probes are never overwritten\n")
    meta, rows = read_manifest(Path(args.manifest))
    root = resolve_root(meta, args.root)
    picked = select_rows(rows, splits=tuple(args.splits.split(",")), speakers=args.speakers,
                         per_speaker=args.per_speaker, seed=args.seed)
    aligner = PhoneAligner.from_npz(args.model)
    backend = MandarinFormantSynth()
    by_id = {str(row["utt_id"]): row for row in picked}

    records, align_skips = [], []
    for entry in align_rows(aligner, picked, root, on_skip=align_skips.append):
        if isinstance(entry, dict):
            continue
        pinyins = [str(token) for token in by_id[entry.utt_id]["pinyin"]]
        pair = build_pair(root / entry.wav_path, pinyins, template=PRESETS[args.template],
                          backend=backend, alignment="forced", phone_spans=entry.spans)
        syllables = pair.pitch.get("syllables", [])
        gaps = [float(item["longest_gap_s"]) for item in syllables]
        coverage = [float(item["coverage"]) for item in syllables]
        records.append({
            "utt_id": entry.utt_id, "speaker": entry.speaker,
            "status": pair.status, "ready": bool(pair.readiness.get("training_ready")),
            "reasons": list(pair.readiness.get("reasons", [])),
            "issues": sorted({issue["code"] for issue in pair.issues}),
            "alignment_confidence": round(float(entry.confidence), 4),
            "syllables": len(pinyins),
            "words": pair.syllables and len(pair.syllables),
            "max_gap_s": round(max(gaps), 4) if gaps else None,
            "min_coverage": round(min(coverage), 4) if coverage else None,
        })

    def summary(items):
        return {"count": len(items),
                "codes": dict(Counter(code for item in items for code in item["issues"])),
                "reasons": dict(Counter(code for item in items for code in item["reasons"]))}

    attempted = len(records) + len(align_skips)
    report = {
        "probe": "readiness", "model": str(args.model), "template": args.template,
        "splits": args.splits, "speakers": args.speakers, "per_speaker": args.per_speaker,
        "seed": args.seed, "requested": len(picked), "attempted": attempted,
        "aligned": len(records), "align_skips": len(align_skips),
        "skip_reasons": dict(Counter(item["reason"] for item in align_skips)),
        "ready": sum(1 for item in records if item["ready"]),
        "ready_rate_aligned": round(sum(1 for item in records if item["ready"])
                                    / max(1, len(records)), 4),
        "ready_rate_attempted": round(sum(1 for item in records if item["ready"])
                                      / max(1, attempted), 4),
        "blocked": summary([item for item in records if not item["ready"]]),
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
