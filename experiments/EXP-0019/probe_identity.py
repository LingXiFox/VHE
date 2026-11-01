"""EXP-0019: is the template choice a timbre decision, and how far are the presets?

The mechanical half of a pair is rendered by MandarinFormantSynth from one of two original
presets ("yang": formant scale 1.08, "yin": 0.87). EXP-0018 picked between them by median F0
(>= 200 Hz -> yang). This probe asks whether that choice has anything to do with timbre.

It measures, for the 16 pairs EXP-0018 actually rendered, a level-normalised long-term
average spectrum (LTAS, 24 mel bands, 100-7000 Hz, loud frames only) and three distances:

  d(real, yang)   how far the real speaker is from each preset
  d(real, yin)
  d(yang, yin)    how much the two presets differ from each other

If the presets differ by less than either differs from real speech, "picking the closer
preset" is not a timbre match: both are wrong, and identity has to be defined against the
mechanical input the model actually sees, never against the real recording.

A diagnostic grid then fits a single formant scale s (0.75..1.20, spectral tilt fixed at 1.35)
that minimises d(real, render(s)). That number says where a speaker sits relative to the two
presets. It is a *measurement of the gap*, never a way to build pairs: fitting the synthesizer
to the target speaker is exactly the substitution AGENTS.md forbids.

Run: python experiments/EXP-0019/probe_identity.py --manifest <v4 manifest.jsonl> \
        --pairs-report experiments/EXP-0018/logs/pair_report_v3.json \
        --out experiments/EXP-0019/logs/identity_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from vhe.data.corpus import read_corpus_manifest
from vhe.data.pairing import build_pair
from vhe.synth import PRESETS
from vhe.synth.mandarin import MandarinFormantSynth

SR = 24000
FRAME = 1024
HOP = 256
N_BANDS = 24
LOW_HZ = 100.0
HIGH_HZ = 7000.0
LEVEL_FLOOR_RATIO = 0.05
#: 0.70 .. 1.30 keeps the optimum off the boundary for most speakers; a 1-D fit cannot
#: separate formant scale from spectral tilt, so this stays a gap diagnostic, not a fit.
GRID = tuple(round(0.70 + 0.05 * index, 2) for index in range(13))
FIT_TILT = 1.35
FIT_BREATHINESS = 0.009
HEURISTIC_F0_HZ = 200.0


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def band_edges() -> np.ndarray:
    mels = np.linspace(hz_to_mel(LOW_HZ), hz_to_mel(HIGH_HZ), N_BANDS + 1)
    return mel_to_hz(mels)


def ltas_db(audio: np.ndarray, sample_rate: int = SR) -> np.ndarray:
    """Level-normalised log band energies over the loud frames (one value per band)."""
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size < FRAME:
        samples = np.pad(samples, (0, FRAME - samples.size))
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
    rms = np.sqrt((frames ** 2).mean(axis=1))
    floor = max(float(rms.max()) * LEVEL_FLOOR_RATIO, 1e-6)
    keep = rms >= floor
    if not keep.any():
        keep = rms >= float(rms.max())
    window = np.hanning(FRAME)
    power = np.abs(np.fft.rfft(frames[keep] * window, axis=1)) ** 2
    mean_power = power.mean(axis=0)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sample_rate)
    edges = band_edges()
    bands = np.array([mean_power[(freqs >= low) & (freqs < high)].sum()
                      for low, high in zip(edges[:-1], edges[1:])])
    db = 10.0 * np.log10(np.maximum(bands, 1e-20))
    return db - db.mean()


def distance(left: np.ndarray, right: np.ndarray) -> float:
    """RMS difference in dB between two level-normalised LTAS vectors."""
    return float(np.sqrt(np.mean((left - right) ** 2)))


def median_f0(report_row: dict) -> float | None:
    value = report_row.get("measured_median_f0_hz")
    return None if value is None else float(value)


def choose_template(median: float | None) -> str:
    if median is None:
        return "yin"
    return "yang" if median >= HEURISTIC_F0_HZ else "yin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pairs-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--grid", action="store_true",
                        help="also fit an implied formant scale (diagnostic only)")
    args = parser.parse_args()

    args.out = args.out.resolve()
    if args.out.exists():
        parser.error(f"refusing to overwrite an existing report: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    report = json.loads(Path(args.pairs_report).read_text(encoding="utf-8"))
    rows = [row for row in report["samples"] if row["status"] == "ok"]
    entries, meta = read_corpus_manifest(args.manifest)
    root = Path(meta["root"])
    by_id = {entry.utt_id: entry for entry in entries}
    backend = MandarinFormantSynth()

    samples = []
    for row in rows:
        utt_id = row["utt_id"]
        entry = by_id[utt_id]
        rendered = {}
        for name in ("yang", "yin"):
            rendered[name] = build_pair(root / entry.wav_path, list(entry.pinyin),
                                        template=PRESETS[name], backend=backend,
                                        alignment="dp")
        real_sha = {name: result.real_sha256 for name, result in rendered.items()}
        if len(set(real_sha.values())) != 1:
            raise SystemExit(f"{utt_id}: the two renders disagree about the real audio")
        real = rendered["yang"].real
        if real is None or rendered["yang"].mechanical is None \
                or rendered["yin"].mechanical is None:
            raise SystemExit(f"{utt_id}: a rendered half is missing")
        real_ltas = ltas_db(real)
        yang_ltas = ltas_db(rendered["yang"].mechanical)
        yin_ltas = ltas_db(rendered["yin"].mechanical)
        d_yang = distance(real_ltas, yang_ltas)
        d_yin = distance(real_ltas, yin_ltas)
        separation = distance(yang_ltas, yin_ltas)
        median = median_f0(row)
        heuristic = choose_template(median)
        closer = "yang" if d_yang <= d_yin else "yin"
        record = {
            "sample_id": row["sample_id"],
            "utt_id": utt_id,
            "speaker": row["speaker"],
            "gender": row["gender"],
            "pinyin": row["pinyin"],
            "median_f0_hz": median,
            "heuristic_template": heuristic,
            "closer_template": closer,
            "agrees": heuristic == closer,
            "d_real_yang_db": round(d_yang, 3),
            "d_real_yin_db": round(d_yin, 3),
            "d_yang_yin_db": round(separation, 3),
            "margin_db": round(abs(d_yang - d_yin), 3),
            "gap_over_separation": round(min(d_yang, d_yin) / separation, 3),
        }
        if args.grid:
            curve = []
            for scale in GRID:
                template = replace(PRESETS["yang"], formant_scale=scale,
                                   spectral_tilt=FIT_TILT, breathiness=FIT_BREATHINESS)
                result = build_pair(root / entry.wav_path, list(entry.pinyin),
                                    template=template, backend=backend, alignment="dp")
                if result.mechanical is None:
                    raise SystemExit(f"{utt_id}: grid render at scale {scale} failed")
                curve.append((scale, round(distance(real_ltas, ltas_db(result.mechanical)), 3)))
            best_scale, best_distance = min(curve, key=lambda item: item[1])
            record["implied_scale"] = best_scale
            record["implied_scale_distance_db"] = best_distance
            record["scale_curve"] = curve
            record["inside_preset_range"] = bool(0.87 <= best_scale <= 1.08)
            record["at_grid_edge"] = best_scale in (GRID[0], GRID[-1])
        samples.append(record)
        print(f"{record['sample_id']} {utt_id} f0={median} heuristic={heuristic} "
              f"closer={closer} d_yang={d_yang:.2f} d_yin={d_yin:.2f} "
              f"sep={separation:.2f}"
              + (f" scale={record.get('implied_scale')}" if args.grid else ""))

    def median_of(values: list[float]) -> float | None:
        return None if not values else round(float(np.median(values)), 3)

    agreements = sum(1 for sample in samples if sample["agrees"])
    payload = {
        "experiment": "EXP-0019",
        "hypothesis": ("median F0 does not select a timbre match: the two presets differ by "
                       "less than either differs from a real speaker"),
        "source_report": str(Path(args.pairs_report).resolve()),
        "manifest": str(args.manifest.resolve()),
        "metric": {"name": "log-mel LTAS distance", "bands": N_BANDS,
                   "low_hz": LOW_HZ, "high_hz": HIGH_HZ, "frame": FRAME, "hop": HOP,
                   "level_floor_ratio": LEVEL_FLOOR_RATIO,
                   "unit": "dB RMS over level-normalised band energies"},
        "settings": {"heuristic_f0_hz": HEURISTIC_F0_HZ, "grid": list(GRID) if args.grid else [],
                     "fit_tilt": FIT_TILT, "fit_breathiness": FIT_BREATHINESS,
                     "sample_rate": SR},
        "samples": len(samples),
        "agreement": {"heuristic_equals_closer": agreements,
                      "rate": round(agreements / len(samples), 3) if samples else None},
        "d_real_yang_db": median_of([sample["d_real_yang_db"] for sample in samples]),
        "d_real_yin_db": median_of([sample["d_real_yin_db"] for sample in samples]),
        "d_yang_yin_db": median_of([sample["d_yang_yin_db"] for sample in samples]),
        "margin_db": median_of([sample["margin_db"] for sample in samples]),
        "gap_over_separation": median_of([sample["gap_over_separation"]
                                          for sample in samples]),
        "rows": samples,
    }
    if args.grid:
        scales = [sample["implied_scale"] for sample in samples]
        payload["implied_scale"] = {
            "median": median_of(scales),
            "min": min(scales) if scales else None,
            "max": max(scales) if scales else None,
            "inside_preset_range": sum(1 for sample in samples
                                       if sample["inside_preset_range"]),
            "at_grid_edge": sum(1 for sample in samples if sample["at_grid_edge"]),
            "note": ("diagnostic of the speaker/preset gap only; the synthesizer is never "
                     "fitted to the target speaker when building pairs"),
        }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + chr(10),
                        encoding="utf-8")
    summary = {key: payload[key] for key in
               ("samples", "agreement", "d_real_yang_db", "d_real_yin_db", "d_yang_yin_db",
                "margin_db", "gap_over_separation")}
    if args.grid:
        summary["implied_scale"] = payload["implied_scale"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
