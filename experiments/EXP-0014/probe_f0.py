"""EXP-0014: validate the f0 estimator on synthetic ground truth and a tiny real subset.

Two parts, both CPU only and bounded in time:

1. Synthetic ground truth. The formant synthesiser renders a sustained /a/ with a known
   PitchCurve; estimate_f0 is then scored against that curve in cents. This isolates
   estimator defects (octave errors at high f0, glide tracking) from corpus problems.
2. A tiny real subset (<= 12 utterances from the test split, <= 3 s each). There is no
   ground truth here, so the report is descriptive: voiced ratio, f0 range, and the rate of
   octave jumps between adjacent frames. Nothing is written back to the corpus.

Usage:
    python experiments/EXP-0014/probe_f0.py --manifest <manifest.jsonl> --output <report.json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.audio import read_wav  # noqa: E402
from vhe.data.corpus import read_corpus_manifest  # noqa: E402
from vhe.data.f0 import estimate_f0  # noqa: E402
from vhe.data.resample import resample  # noqa: E402
from vhe.synth import (  # noqa: E402
    PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve, SynthesisRequest, Timing,
)
from vhe.synth.mandarin import MandarinFormantSynth  # noqa: E402

SR = 24000
SYNTH = MandarinFormantSynth()
REAL_FRAME_S, REAL_HOP_S = 0.040, 0.010
VOICING_THRESHOLD = 0.45


def render_a(curve: PitchCurve, duration_s: float) -> np.ndarray:
    request = SynthesisRequest(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("a", duration_s),), inventory="mandarin"),
        pitch=curve, timing=Timing(0.0, 0.0), sample_rate=SR)
    return SYNTH.render(request).audio


def contours() -> list[tuple[str, float, Path | None]]:
    """(name, duration_s, curve) triples covering the melodic range the corpus actually uses."""
    out = []
    for f0 in (110.0, 165.0, 210.0, 300.0, 350.0):
        out.append((f"constant_{int(f0)}", 0.60, PitchCurve.constant(f0, 0.60)))
    glide_up = np.linspace(110.0, 220.0, 61)
    glide_down = np.linspace(300.0, 130.0, 61)
    times = np.linspace(0.0, 0.60, 61)
    out.append(("glide_110_220", 0.60, PitchCurve(times_s=tuple(times), f0_hz=tuple(glide_up))))
    out.append(("glide_300_130", 0.60, PitchCurve(times_s=tuple(times), f0_hz=tuple(glide_down))))
    vibrato_t = np.linspace(0.0, 0.80, 81)
    vibrato = 200.0 + 12.0 * np.sin(2.0 * math.pi * 5.0 * vibrato_t)
    out.append(("vibrato_200_pm12", 0.80, PitchCurve(times_s=tuple(vibrato_t), f0_hz=tuple(vibrato))))
    return out


def cents(estimate: float, reference: float) -> float:
    return 1200.0 * math.log2(estimate / reference)


def score(track, curve: PitchCurve) -> dict:
    frames = [(fr.time_s, fr.f0_hz) for fr in track.frames if np.isfinite(fr.f0_hz)]
    if not frames:
        return {"voiced_frames": 0}
    errors = [cents(f0, float(np.interp(t, curve.times_s, curve.f0_hz))) for t, f0 in frames]
    abs_err = np.abs(np.asarray(errors))
    return {
        "voiced_frames": len(errors),
        "hit_50c": float(np.mean(abs_err <= 50.0)),
        "hit_100c": float(np.mean(abs_err <= 100.0)),
        "octave_error_rate": float(np.mean(abs_err > 550.0)),
        "median_abs_cents": float(np.median(abs_err)),
        "p90_abs_cents": float(np.percentile(abs_err, 90)),
        "signed_median_cents": float(np.median(errors)),
    }


def synthetic_part() -> list[dict]:
    rows = []
    for name, duration, curve in contours():
        audio = render_a(curve, duration)
        default = score(estimate_f0(audio, SR), curve)
        bounded = score(estimate_f0(audio, SR, fmax_hz=500.0), curve)
        rows.append({"case": name, "default": default, "fmax_500": bounded})
        print(f"{name:18s} default hit50={default.get('hit_50c', 0):.2f} "
              f"oct={default.get('octave_error_rate', 0):.2f} | "
              f"fmax500 hit50={bounded.get('hit_50c', 0):.2f} "
              f"oct={bounded.get('octave_error_rate', 0):.2f}")
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
    # One utterance per speaker so a single voice cannot dominate the descriptive statistics.
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
        track = estimate_f0(audio, SR, fmax_hz=500.0)
        # estimate_f0 reports f0_hz = 0 with voicing = 0 on unvoiced frames; counting those
        # as measurements is what made the first version of this report useless.
        f0s = [fr.f0_hz for fr in track.frames
               if fr.voicing >= VOICING_THRESHOLD and fr.f0_hz > 0.0]
        rows.append({
            "utt_id": entry.utt_id,
            "speaker": entry.speaker,
            "duration_s": round(entry.duration_s, 3),
            "frames": len(track.frames),
            "voiced_ratio": round(len(f0s) / max(1, len(track.frames)), 3),
            "median_f0_hz": round(float(np.median(f0s)), 1) if f0s else None,
            "p05_hz": round(float(np.percentile(f0s, 5)), 1) if f0s else None,
            "p95_hz": round(float(np.percentile(f0s, 95)), 1) if f0s else None,
            "octave_jump_rate": round(octave_jump_rate(f0s), 4),
            "frames_out_of_range": sum(1 for f in f0s if f < 65.0 or f > 500.0),
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    medians = [r["median_f0_hz"] for r in rows if r["median_f0_hz"]]
    jumps = [r["octave_jump_rate"] for r in rows]
    return {
        "utterances": len(rows),
        "median_of_medians_hz": round(float(np.median(medians)), 1) if medians else None,
        "mean_octave_jump_rate": round(float(np.mean(jumps)), 4) if jumps else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-duration", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 1
    report = {
        "sample_rate": SR,
        "estimator": {"frame_s": REAL_FRAME_S, "hop_s": REAL_HOP_S, "fmin_hz": 65.0,
                      "voicing_threshold": VOICING_THRESHOLD},
        "synthetic": synthetic_part(),
        "real": real_part(args.manifest, args.limit, args.max_duration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
