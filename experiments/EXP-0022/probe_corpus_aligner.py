"""EXP-0021 probe: does the chain-HMM aligner behave on real AISHELL-3 speech?

What this proves: on a fixed, speaker-disjoint subset of AISHELL-3 the aligner (a) trains
without divergence, (b) produces monotone, non-degenerate phone spans whose syllable starts
land near the energy-valley DP boundaries, and (c) is stable between a flat start and four
Baum-Welch iterations.

What this does NOT prove: there is no human-annotated boundary reference in this run. The
energy-valley DP is itself an unverified heuristic, so agreement with it measures
compatibility, not correctness. Phone duration statistics are descriptive; they cannot show
that any single boundary is right.

Usage:
    python experiments/EXP-0021/probe_corpus_aligner.py \
        --manifest .../aishell3-scan/v4/manifest.jsonl \
        --root /Volumes/LingXiWorkspace/VHE-Datasets/AISHELL-3/extracted \
        --train-speakers 16 --eval-speakers 4 --per-speaker 20 \
        --iterations 0 4 --out experiments/EXP-0021/logs/corpus_aligner.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
#: Running from the checkout; a script copied to /tmp relies on PYTHONPATH instead.
if len(_HERE.parents) >= 3 and (_HERE.parents[2] / "src").is_dir():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))

from vhe.audio import read_wav  # noqa: E402
from vhe.data.align import align_syllables  # noqa: E402
from vhe.data.aligner import (  # noqa: E402
    PHONES,
    PhoneAligner,
    pinyin_phones,
    prepare_utterance,
    train_aligner,
)
from vhe.data.corpus import entry_renderable, read_corpus_manifest  # noqa: E402
from vhe.data.pairing import parse_syllable  # noqa: E402
from vhe.synth.phonology import MANDARIN_PHONEMES  # noqa: E402


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def select_utterances(entries, *, split: str, speakers: int, per_speaker: int,
                      seed: int) -> list:
    """Pick speakers and utterances deterministically from one split."""
    pool = [e for e in entries if e.split == split and entry_renderable(e)]
    names = sorted({e.speaker for e in pool})
    if speakers < len(names):
        picked = sorted(_rng(seed).choice(names, size=speakers, replace=False).tolist())
    else:
        picked = names
    chosen = []
    for name in picked:
        items = sorted((e for e in pool if e.speaker == name), key=lambda e: e.utt_id)
        if per_speaker < len(items):
            idx = np.sort(_rng(seed + 1).choice(len(items), size=per_speaker, replace=False))
            items = [items[int(i)] for i in idx]
        chosen.extend(items)
    return chosen


def prepare(entries, root: Path) -> tuple[list, list]:
    items, skips = [], []
    for entry in entries:
        try:
            items.append(prepare_utterance(root / entry.wav_path, entry.pinyin))
        except Exception as exc:  # noqa: BLE001 - every failure is reported, never dropped
            skips.append({"utt_id": entry.utt_id, "reason": f"{type(exc).__name__}: {exc}"})
    return items, skips


def syllable_starts(result, pinyins) -> tuple[list[float], str | None]:
    """Map phone spans back onto syllable starts; returns (starts, problem)."""
    counts = []
    for token in pinyins:
        syllable = parse_syllable(str(token), "mandarin")
        if syllable.reason is not None or not syllable.phonemes:
            return [], f"pinyin {token!r} unresolved: {syllable.reason}"
        counts.append(len(syllable.phonemes))
    if sum(counts) != len(result.spans):
        return [], f"phone count mismatch: {sum(counts)} vs {len(result.spans)}"
    starts, offset = [], 0
    for count in counts:
        starts.append(result.spans[offset].start_s)
        offset += count
    return starts, None


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "max": round(float(array.max()), 4),
    }


def compare(alignment, pinyins, audio, rate: int, min_frames: int) -> dict:
    """One utterance: aligner spans against the energy-valley DP and silence gate."""
    starts, problem = syllable_starts(alignment, pinyins)
    if problem is not None:
        return {"problem": problem}
    dp = align_syllables(np.asarray(audio, dtype=np.float64), rate, len(pinyins))
    if any(issue["severity"] == "error" for issue in dp.issues):
        return {"problem": f"energy_dp: {dp.issues[0]['code']}"}
    if len(dp.boundaries) != len(starts):
        return {"problem": f"syllable count mismatch: {len(dp.boundaries)} vs {len(starts)}"}
    reference = [boundary.start_s for boundary in dp.boundaries]
    deltas = [abs(a - b) for a, b in zip(starts[1:], reference[1:])]
    signed = [a - b for a, b in zip(starts[1:], reference[1:])]
    durations = [span.end_s - span.start_s for span in alignment.spans]
    return {
        "syllables": len(starts),
        "phones": len(alignment.spans),
        "abs_delta_s": deltas,
        "signed_delta_s": signed,
        "confidence": alignment.confidence,
        "logprob": alignment.logprob,
        "lead_s": round(alignment.offset_s, 4),
        "tail_s": round(alignment.speech_s[1] - alignment.spans[-1].end_s, 4),
        "min_phone_s": round(float(min(durations)), 4),
        "first_start_s": round(starts[0], 4),
        "last_end_s": round(alignment.spans[-1].end_s, 4),
        "phones_detail": [(span.phoneme, round(span.end_s - span.start_s, 4))
                          for span in alignment.spans],
        #: Spans pinned at one frame per state: the signature of a duration prior that is
        #: too short for the phone (see EXP-0021 README).
        "degenerate": sum(1 for span in alignment.spans if span.frames <= min_frames),
    }


def evaluate(aligner: PhoneAligner, entries, root: Path, limit: int | None) -> dict:
    rows, problems, phone_durations = [], [], {}
    kind_durations: dict[str, list[float]] = {}
    used = entries if limit is None else entries[:limit]
    min_frames = aligner.states_per_phone
    started = time.perf_counter()
    for entry in used:
        audio, rate = read_wav(root / entry.wav_path)
        alignment = aligner.align(audio, rate, entry.pinyin)
        row = compare(alignment, entry.pinyin, audio, rate, min_frames)
        row["utt_id"] = entry.utt_id
        row["speaker"] = entry.speaker
        if "problem" in row:
            problems.append({"utt_id": entry.utt_id, "problem": row["problem"]})
            continue
        for name, duration in row["phones_detail"]:
            phone_durations.setdefault(name, []).append(duration)
            kind = MANDARIN_PHONEMES[name].kind
            kind_durations.setdefault(kind, []).append(duration)
        rows.append(row)
    elapsed = time.perf_counter() - started
    deltas = [d for row in rows for d in row["abs_delta_s"]]
    signed = [d for row in rows for d in row["signed_delta_s"]]
    return {
        "utterances": len(rows),
        "seconds": round(elapsed, 2),
        "seconds_per_utterance": round(elapsed / max(1, len(rows)), 3),
        "abs_delta_s": summarise(deltas),
        "signed_delta_s": summarise(signed),
        "within_50ms": round(float(np.mean([d <= 0.050 for d in deltas])), 3) if deltas else None,
        "within_100ms": round(float(np.mean([d <= 0.100 for d in deltas])), 3) if deltas else None,
        "confidence": summarise([row["confidence"] for row in rows]),
        "lead_s": summarise([row["lead_s"] for row in rows]),
        "first_start_s": summarise([row["first_start_s"] for row in rows]),
        "min_phone_s": summarise([row["min_phone_s"] for row in rows]),
        "phones": {name: summarise(values) for name, values in sorted(phone_durations.items())},
        "kinds": {kind: summarise(values) for kind, values in sorted(kind_durations.items())},
        "degenerate_fraction": round(
            sum(row["degenerate"] for row in rows)
            / max(1, sum(row["phones"] for row in rows)), 3),
        "problems": problems,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-speakers", type=int, default=16)
    parser.add_argument("--eval-speakers", type=int, default=4)
    parser.add_argument("--per-speaker", type=int, default=20)
    parser.add_argument("--eval-utterances", type=int, default=40)
    parser.add_argument("--iterations", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--states-per-phone", type=int, default=3)
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-cap", type=int, default=2500)
    parser.add_argument("--estimate-transitions", action="store_true",
                        help="re-estimate one stay probability per state (EXP-0022 arm)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    root = Path(args.root)
    entries, meta = read_corpus_manifest(args.manifest)
    train_entries = select_utterances(entries, split="train", speakers=args.train_speakers,
                                      per_speaker=args.per_speaker, seed=args.seed)
    eval_entries = select_utterances(entries, split="dev", speakers=args.eval_speakers,
                                     per_speaker=args.per_speaker, seed=args.seed + 7)
    eval_entries = eval_entries[: args.eval_utterances]
    print(f"train: {len(train_entries)} utterances / "
          f"{len({e.speaker for e in train_entries})} speakers", flush=True)
    print(f"eval:  {len(eval_entries)} utterances / "
          f"{len({e.speaker for e in eval_entries})} speakers", flush=True)

    started = time.perf_counter()
    items, skips = prepare(train_entries, root)
    prepare_s = time.perf_counter() - started
    print(f"prepared {len(items)} train utterances in {prepare_s:.1f}s "
          f"({len(skips)} skipped)", flush=True)

    report: dict = {
        "experiment": "EXP-0022",
        "manifest": str(args.manifest),
        "manifest_inventory": meta.get("inventory", {}).get("sha256"),
        "root": str(root),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "config": {
            "train_speakers": args.train_speakers, "eval_speakers": args.eval_speakers,
            "per_speaker": args.per_speaker, "eval_utterances": args.eval_utterances,
            "iterations": args.iterations, "states_per_phone": args.states_per_phone,
            "components": args.components, "seed": args.seed, "frame_cap": args.frame_cap,
            "phone_inventory": len(PHONES),
            "estimate_transitions": bool(args.estimate_transitions),
        },
        "data": {
            "train_utterances": len(items),
            "train_speakers": sorted({e.speaker for e in train_entries}),
            "eval_speakers": sorted({e.speaker for e in eval_entries}),
            "prepare_seconds": round(prepare_s, 2),
            "prepare_skips": skips,
        },
        "runs": {},
    }

    aligners: dict[int, PhoneAligner] = {}
    for iterations in args.iterations:
        started = time.perf_counter()
        aligner, train_report = train_aligner(
            items, states_per_phone=args.states_per_phone, n_components=args.components,
            iterations=iterations, seed=args.seed, frame_cap=args.frame_cap,
            estimate_transitions=args.estimate_transitions)
        train_s = time.perf_counter() - started
        aligners[iterations] = aligner
        history = train_report["history"]
        tail = (f"loglik/frame={history[-1]['loglik_per_frame']:.2f}" if history
                else "loglik/frame=flat-start")
        print(f"iterations={iterations}: {train_s:.1f}s monotone={train_report['monotone']} "
              f"{tail}", flush=True)
        result = evaluate(aligner, eval_entries, root, args.eval_utterances)
        report["runs"][str(iterations)] = {
            "train_seconds": round(train_s, 2),
            "train": train_report,
            "eval": result,
        }

    if len(aligners) > 1:
        keys = sorted(aligners)
        first, last = aligners[keys[0]], aligners[keys[-1]]
        deltas = []
        for entry in eval_entries:
            audio, rate = read_wav(root / entry.wav_path)
            left = first.align(audio, rate, entry.pinyin)
            right = last.align(audio, rate, entry.pinyin)
            if len(left.spans) != len(right.spans):
                continue
            deltas.extend(abs(a.start_s - b.start_s)
                          for a, b in zip(left.spans[1:], right.spans[1:]))
        report["flat_start_vs_em"] = {
            "iterations": [keys[0], keys[-1]],
            "abs_delta_s": summarise(deltas),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
