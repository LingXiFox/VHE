"""EXP-0015: multi-candidate CMND pitch tracking with Viterbi continuity.

Same synthetic ground truth (8 contours) and the same 12 real utterances as EXP-0014, so the
numbers are directly comparable with the single-candidate estimator recorded there. The new
tracker always runs with the default f0 range (65-900 Hz); a narrowed range is only run as a
control, never as the reported result, because EXP-0014 showed the old estimator only behaved
after fmax was cut to 500 Hz. Frames the tracker rejects are reported as rejections and are
not counted as measurements.

Usage:
    python experiments/EXP-0015/probe_pitch.py --manifest <manifest.jsonl> --output <report.json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.audio import read_wav  # noqa: E402
from vhe.data.corpus import read_corpus_manifest  # noqa: E402
from vhe.data.f0 import estimate_f0  # noqa: E402
from vhe.data.pitch import octave_error_kind, track_pitch  # noqa: E402
from vhe.data.resample import resample  # noqa: E402
from vhe.synth import (  # noqa: E402
    PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve, SynthesisRequest, Timing,
)
from vhe.synth.mandarin import MandarinFormantSynth  # noqa: E402

SR = 24000
SYNTH = MandarinFormantSynth()


def render_a(curve: PitchCurve, duration_s: float) -> np.ndarray:
    request = SynthesisRequest(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("a", duration_s),), inventory="mandarin"),
        pitch=curve, timing=Timing(0.0, 0.0), sample_rate=SR)
    return SYNTH.render(request).audio


def contours() -> list[tuple[str, float, PitchCurve]]:
    """Identical to EXP-0014 so the two reports can be compared line by line."""
    out = []
    for f0 in (110.0, 165.0, 210.0, 300.0, 350.0):
        out.append((f"constant_{int(f0)}", 0.60, PitchCurve.constant(f0, 0.60)))
    times = np.linspace(0.0, 0.60, 61)
    out.append(("glide_110_220", 0.60,
                PitchCurve(times_s=tuple(times), f0_hz=tuple(np.linspace(110.0, 220.0, 61)))))
    out.append(("glide_300_130", 0.60,
                PitchCurve(times_s=tuple(times), f0_hz=tuple(np.linspace(300.0, 130.0, 61)))))
    vibrato_t = np.linspace(0.0, 0.80, 81)
    vibrato = 200.0 + 12.0 * np.sin(2.0 * math.pi * 5.0 * vibrato_t)
    out.append(("vibrato_200_pm12", 0.80,
                PitchCurve(times_s=tuple(vibrato_t), f0_hz=tuple(vibrato))))
    return out


def cents(estimate: float, reference: float) -> float:
    return 1200.0 * math.log2(estimate / reference)


def score_frames(pairs: list[tuple[float, float]], curve: PitchCurve, *,
                 frames: int) -> dict:
    """pairs = (time_s, f0) for frames the estimator was willing to report."""
    if not pairs:
        return {"voiced_frames": 0, "hit_50c": 0.0, "octave_error_rate": 0.0,
                "rejection_rate": 1.0}
    errors = [cents(f0, float(np.interp(t, curve.times_s, curve.f0_hz))) for t, f0 in pairs]
    abs_err = np.abs(np.asarray(errors))
    kinds = Counter()
    for (t, f0), err in zip(pairs, errors):
        reference = float(np.interp(t, curve.times_s, curve.f0_hz))
        kind = octave_error_kind(f0, reference) if abs(err) > 550.0 else None
        kinds[kind or "none"] += 1
    return {
        "frames": frames,
        "voiced_frames": len(pairs),
        "rejection_rate": round(1.0 - len(pairs) / frames, 4) if frames else 0.0,
        "hit_50c": float(np.mean(abs_err <= 50.0)),
        "hit_100c": float(np.mean(abs_err <= 100.0)),
        "octave_error_rate": float(np.mean(abs_err > 550.0)),
        "octave_kinds": dict(sorted(kinds.items())),
        "median_abs_cents": float(np.median(abs_err)),
        "p90_abs_cents": float(np.percentile(abs_err, 90)),
        "signed_median_cents": float(np.median(errors)),
    }


def legacy_pairs(track) -> list[tuple[float, float]]:
    return [(f.time_s, f.f0_hz) for f in track.frames if f.voicing >= 0.45 and f.f0_hz > 0.0]


def synthetic_part() -> list[dict]:
    rows = []
    for name, duration, curve in contours():
        audio = render_a(curve, duration)
        legacy_default = estimate_f0(audio, SR)
        legacy_bounded = estimate_f0(audio, SR, fmax_hz=500.0)
        tracked = track_pitch(audio, SR)
        new_pairs = [(f.time_s, f.f0_hz) for f in tracked.frames if f.f0_hz > 0.0]
        row = {
            "case": name,
            "frames_new": len(tracked.frames),
            "legacy_default": score_frames(legacy_pairs(legacy_default), curve,
                                           frames=len(legacy_default.frames)),
            "legacy_fmax500": score_frames(legacy_pairs(legacy_bounded), curve,
                                           frames=len(legacy_bounded.frames)),
            "new_default": score_frames(new_pairs, curve, frames=len(tracked.frames)),
            "new_reasons": tracked.reasons(),
            "new_median_abs_step_cents": round(tracked.stats()["median_abs_step_cents"], 2),
        }
        rows.append(row)
        print(f"{name:18s} legacy hit50={row['legacy_default']['hit_50c']:.2f} "
              f"oct={row['legacy_default']['octave_error_rate']:.2f} | "
              f"fmax500 hit50={row['legacy_fmax500']['hit_50c']:.2f} "
              f"oct={row['legacy_fmax500']['octave_error_rate']:.2f} | "
              f"new hit50={row['new_default']['hit_50c']:.2f} "
              f"rej={row['new_default']['rejection_rate']:.2f} "
              f"step={row['new_median_abs_step_cents']:.1f}")
    return rows


def octave_jump_rate(f0s: list[float]) -> float:
    if len(f0s) < 2:
        return 0.0
    ratios = [b / a for a, b in zip(f0s, f0s[1:]) if a > 0.0 and b > 0.0]
    if not ratios:
        return 0.0
    jumps = sum(1 for r in ratios if 1.9 <= r <= 2.1 or 0.47 <= r <= 0.53)
    return jumps / len(ratios)


def real_part(manifest: Path, limit: int, max_duration_s: float) -> dict:
    entries, meta = read_corpus_manifest(manifest)
    root = Path(meta["root"])
    picks, seen = [], set()
    for entry in entries:
        if entry.split != "test" or entry.duration_s > max_duration_s:
            continue
        if entry.speaker in seen:
            continue
        seen.add(entry.speaker)
        picks.append(entry)
        if len(picks) >= limit:
            break
    rows = []
    for entry in picks:
        audio, sr = read_wav(root / entry.wav_path)
        audio = resample(audio, sr, SR)
        legacy = estimate_f0(audio, SR, fmax_hz=500.0)
        tracked = track_pitch(audio, SR)
        legacy_f0 = legacy_pairs(legacy)
        new_f0 = [(f.time_s, f.f0_hz) for f in tracked.frames if f.f0_hz > 0.0]
        row = {
            "utt_id": entry.utt_id,
            "speaker": entry.speaker,
            "duration_s": round(entry.duration_s, 3),
            "frames": len(tracked.frames),
            "legacy_median_f0_hz": round(float(np.median([f for _, f in legacy_f0])), 1)
            if legacy_f0 else None,
            "legacy_voiced_ratio": round(len(legacy_f0) / max(1, len(legacy.frames)), 3),
            "new_median_f0_hz": round(float(np.median([f for _, f in new_f0])), 1)
            if new_f0 else None,
            "new_voiced_ratio": round(len(new_f0) / max(1, len(tracked.frames)), 3),
            "new_rejection_rate": round(tracked.stats()["rejection_rate"], 4),
            "new_reasons": tracked.reasons(),
            "new_median_abs_step_cents": round(tracked.stats()["median_abs_step_cents"], 2),
            "new_octave_jump_rate": round(octave_jump_rate([f for _, f in new_f0]), 4),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    new_median = [r["new_median_f0_hz"] for r in rows if r["new_median_f0_hz"]]
    legacy_median = [r["legacy_median_f0_hz"] for r in rows if r["legacy_median_f0_hz"]]
    reasons: Counter = Counter()
    for row in rows:
        for name, count in row["new_reasons"].items():
            reasons[name] += count
    total_frames = sum(row["frames"] for row in rows)
    return {
        "utterances": len(rows),
        "median_of_medians_hz": round(float(np.median(new_median)), 1) if new_median else None,
        "legacy_median_of_medians_hz": round(float(np.median(legacy_median)), 1)
        if legacy_median else None,
        "mean_rejection_rate": round(float(np.mean([r["new_rejection_rate"] for r in rows])), 4)
        if rows else None,
        "frames": total_frames,
        "reasons": dict(sorted(reasons.items())),
        "rejected_share": round(float(sum(count for name, count in reasons.items()
                                           if name != "ok") / max(1, total_frames)), 4),
        "mean_octave_jump_rate": round(float(np.mean([r["new_octave_jump_rate"] for r in rows])), 4)
        if rows else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-duration-s", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite an existing report: {args.output}")
    report = {"synthetic": synthetic_part()}
    report["real"] = real_part(args.manifest, args.limit, args.max_duration_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report["real"].items() if k != "rows"}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
