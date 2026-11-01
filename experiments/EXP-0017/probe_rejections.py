"""EXP-0017: what are the F0 tracker's no_candidate frames?

The P1.4 tracker rejects 281 of 2052 real frames as no_candidate (no CMND dip below
0.55 between 65 and 900 Hz). A rejection can mean two very different things:

* the frame carries no periodicity at all (fricative, stop closure, breath, silence), or
* the frame is voiced but the candidate generator missed it (weak/creaky voice, strong
  formant structure, jitter).

The tracker's own CMND curve cannot make that distinction, because that curve is exactly
what failed. This probe therefore measures independent, frequency-domain statistics on
*every* frame and calibrates the thresholds on the frames the tracker did accept:

* harmonic product spectrum (HPS) peak and its z-score over a 65-400 Hz grid,
* real-cepstrum peak prominence in the 65-400 Hz quefrency band,
* spectral flatness and level relative to the file peak,
* the raw minimum CMND over 65-400 Hz (how far the frame was from being a candidate).

A rejected frame is called a suspected miss only when all of those statistics are inside
the interquartile range of the accepted (ok) frames, i.e. the frame looks like voicing by
every measure except the one that rejected it. Nothing is retuned: the CMND gate stays at
0.55, no rejected frame is interpolated, and the classifier is a diagnostic, so the report
states which fraction of frames it could not decide.

Usage:
    python experiments/EXP-0017/probe_rejections.py --manifest <manifest.jsonl> \
        --output experiments/EXP-0017/logs/rejection_report_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.audio import read_wav  # noqa: E402
from vhe.data.corpus import read_corpus_manifest  # noqa: E402
from vhe.data.f0 import cmnd_curve, estimate_f0  # noqa: E402
from vhe.data.pitch import track_pitch  # noqa: E402
from vhe.data.resample import resample  # noqa: E402

SR = 24000
FRAME_MS = 30.0
HOP_MS = 10.0
HPS_FMIN, HPS_FMAX = 65.0, 400.0
HPS_HARMONICS = 5
GRID = np.arange(HPS_FMIN, HPS_FMAX + 1e-6, 0.5)
HARMONICS = np.arange(1, HPS_HARMONICS + 1)


def frame_grid(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same framing as track_pitch, so frame k of both refers to the same samples."""
    window = max(64, int(round(sample_rate * FRAME_MS / 1000.0)))
    hop = max(1, int(round(sample_rate * HOP_MS / 1000.0)))
    signal = np.asarray(audio, dtype=np.float64)
    if window > signal.size:
        signal = np.pad(signal, (0, window - signal.size), mode="reflect")
    starts = np.arange(0, signal.size - window + 1, hop)
    frames = np.stack([signal[start:start + window] for start in starts])
    times = (starts + window / 2.0) / sample_rate
    return frames, times, np.array([float(np.sqrt(np.mean(f * f))) for f in frames])


def spectral_features(frame: np.ndarray, sample_rate: int) -> dict:
    x = np.asarray(frame, dtype=np.float64) - float(np.mean(frame))
    tapered = x * np.hanning(x.size)
    magnitude = np.abs(np.fft.rfft(tapered))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
    power = magnitude ** 2
    band = (freqs >= 80.0) & (freqs <= 4000.0)
    selected = power[band]
    if selected.size == 0 or selected.mean() <= 0.0:
        flatness = 1.0
    else:
        flatness = float(np.exp(np.mean(np.log(selected + 1e-20))) / selected.mean())

    floor = max(float(magnitude.max()) * 1e-6, 1e-12)
    log_magnitude = np.log(magnitude + floor)
    queries = (GRID[:, None] * HARMONICS[None, :]).ravel()
    scores = np.interp(queries, freqs, log_magnitude).reshape(GRID.size, HARMONICS.size)
    totals = scores.sum(axis=1)
    spread = float(totals.std())
    best = int(np.argmax(totals))
    hps_z = float((totals[best] - float(np.median(totals))) / spread) if spread > 1e-12 else 0.0

    cepstrum = np.fft.irfft(log_magnitude)
    q_low = max(1, int(sample_rate / HPS_FMAX))
    q_high = min(cepstrum.size - 1, int(sample_rate / HPS_FMIN))
    band_cepstrum = cepstrum[q_low:q_high]
    if band_cepstrum.size:
        peak = int(np.argmax(band_cepstrum))
        baseline = float(np.median(band_cepstrum))
        mad = float(np.median(np.abs(band_cepstrum - baseline))) + 1e-12
        cepstrum_f0 = float(sample_rate / (q_low + peak))
        cepstrum_prominence = float((band_cepstrum[peak] - baseline) / mad)
    else:
        cepstrum_f0, cepstrum_prominence = 0.0, 0.0

    return {"flatness": round(flatness, 4),
            "hps_f0_hz": round(float(GRID[best]), 1),
            "hps_z": round(hps_z, 3),
            "cepstrum_f0_hz": round(cepstrum_f0, 1),
            "cepstrum_prominence": round(cepstrum_prominence, 2)}


def min_cmnd(frame: np.ndarray, sample_rate: int) -> float:
    centered = np.asarray(frame, dtype=np.float64) - float(np.mean(frame))
    if float(np.dot(centered, centered)) <= 1e-12:
        return 1.0
    lag_max = min(centered.size - 3, int(np.ceil(sample_rate / 65.0)))
    lags, _, cmnd = cmnd_curve(centered, lag_max)
    low = max(2, int(np.floor(sample_rate / HPS_FMAX)))
    selected = cmnd[(lags >= low) & (lags <= lag_max)]
    return float(selected.min()) if selected.size else 1.0


def analyse_utterance(entry, root: Path) -> dict:
    audio, sample_rate = read_wav(root / entry.wav_path)
    audio = resample(audio, sample_rate, SR)
    track = track_pitch(audio, SR)
    frames, times, levels = frame_grid(audio, SR)
    peak_level = float(np.max(levels)) if levels.size else 0.0
    count = min(len(track.frames), frames.shape[0])
    rows = []
    for index in range(count):
        level_db = 20.0 * np.log10(max(levels[index], 1e-9) / max(peak_level, 1e-12))
        features = spectral_features(frames[index], SR)
        features.update({"time_s": round(float(times[index]), 3),
                         "level_db": round(float(level_db), 1),
                         "min_cmnd": round(min_cmnd(frames[index], SR), 3),
                         "reason": track.frames[index].reason,
                         "f0_hz": round(track.frames[index].f0_hz, 1)})
        rows.append(features)
    return {"utt_id": entry.utt_id, "speaker": entry.speaker,
            "duration_s": round(entry.duration_s, 3), "frames": rows,
            "reasons": track.reasons()}


def quantiles(values: list[float]) -> dict:
    array = np.asarray([value for value in values if value == value], dtype=float)
    if array.size == 0:
        return {}
    points = (10, 25, 50, 75, 90)
    return {f"p{point}": round(float(np.percentile(array, point)), 3) for point in points}


def label_rejected(rows: list[dict], calibration: dict) -> None:
    """Attach a diagnostic label to every no_candidate frame, using ok-frame quantiles."""
    for row in rows:
        if row["reason"] != "no_candidate":
            row["label"] = None
            continue
        looks_voiced = (row["hps_z"] >= calibration["hps_z"]["p25"]
                        and row["cepstrum_prominence"] >= calibration["cepstrum_prominence"]["p25"]
                        and row["flatness"] <= calibration["flatness"]["p75"])
        borderline = (row["hps_z"] >= calibration["hps_z"]["p10"]
                      and row["flatness"] <= calibration["flatness"]["p90"])
        row["label"] = "suspected_miss" if looks_voiced else (
            "boundary_or_weak" if borderline else "aperiodic")


def level_bucket(level_db: float) -> str:
    if level_db > -20.0:
        return ">-20dB"
    if level_db > -30.0:
        return ">-30dB"
    if level_db > -45.0:
        return ">-45dB"
    return "<=-45dB"


def cmnd_bucket(value: float) -> str:
    if value <= 0.60:
        return "<=0.60"
    if value <= 0.70:
        return "<=0.70"
    if value <= 0.85:
        return "<=0.85"
    return ">0.85"


def cross_tab(rows: list[dict], key) -> dict:
    table: dict[str, Counter] = {}
    for row in rows:
        table.setdefault(key(row), Counter())[row["label"]] += 1
    return {bucket: dict(sorted(counts.items())) for bucket, counts in sorted(table.items())}


def median_by_label(rows: list[dict], field: str) -> dict:
    table: dict[str, list[float]] = {}
    for row in rows:
        table.setdefault(row["label"], []).append(row[field])
    return {label: round(float(np.median(values)), 3)
            for label, values in sorted(table.items())}


def mark_runs(rows: list[dict]) -> None:
    run = 0
    for row in rows:
        if row["reason"] == "no_candidate":
            run += 1
        else:
            run = 0
        row["run_length"] = run
    run = 0
    for row in reversed(rows):
        if row["reason"] == "no_candidate":
            run = max(run, row.get("run_length", 1))
            row["run_after"] = run
        else:
            run = 0


def compare_ssb04350040(entry, root: Path) -> dict:
    """The one speaker where the two estimators disagree by 0.18 octave (250.0 vs 280.3 Hz)."""
    audio, sample_rate = read_wav(root / entry.wav_path)
    audio = resample(audio, sample_rate, SR)
    legacy = estimate_f0(audio, SR, fmax_hz=500.0)
    legacy_times, legacy_f0 = legacy.contour()
    track = track_pitch(audio, SR)
    frames, _, _ = frame_grid(audio, SR)
    offsets = []
    for frame in track.frames:
        if frame.f0_hz > 0.0:
            legacy_value = float(np.interp(frame.time_s, legacy_times, legacy_f0))
            if legacy_value > 0.0:
                offsets.append(abs(1200.0 * np.log2(frame.f0_hz / legacy_value)))
    # Report where the two medians disagree, not the first frame that happens to differ.
    examples = []
    for index, frame in enumerate(track.frames):
        if frame.f0_hz <= 0.0 or index >= frames.shape[0] or len(examples) >= 3:
            continue
        legacy_value = float(np.interp(frame.time_s, legacy_times, legacy_f0))
        if legacy_value <= 0.0 or abs(1200.0 * np.log2(frame.f0_hz / legacy_value)) < 50.0:
            continue
        magnitude = np.abs(np.fft.rfft(frames[index] * np.hanning(frames[index].size)))
        freqs = np.fft.rfftfreq(frames[index].size, 1.0 / SR)
        threshold = 0.05 * float(magnitude.max())
        peaks = [float(freqs[i]) for i in range(1, magnitude.size - 1)
                 if magnitude[i] >= threshold and magnitude[i] >= magnitude[i - 1]
                 and magnitude[i] >= magnitude[i + 1]]
        examples.append({"time_s": round(frame.time_s, 3),
                         "new_f0_hz": round(frame.f0_hz, 1),
                         "legacy_f0_hz": round(legacy_value, 1),
                         "harmonic_peaks_hz": [round(value, 1) for value in peaks[:10]],
                         **spectral_features(frames[index], SR)})
    return {"utt_id": entry.utt_id, "speaker": entry.speaker,
            "frames": len(track.frames),
            "new_median_f0_hz": round(float(np.median(track.voiced_f0)), 1)
            if track.voiced_f0.size else None,
            "legacy_median_f0_hz": round(float(np.median(legacy.voiced_f0)), 1)
            if legacy.voiced_f0.size else None,
            "median_abs_offset_cents": round(float(np.median(offsets)), 1) if offsets else None,
            "share_frames_above_50_cents": round(float(np.mean(np.asarray(offsets) > 50.0)), 3)
            if offsets else None,
            "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-duration-s", type=float, default=3.0)
    parser.add_argument("--special", default="SSB04350040")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite an existing report: {args.output}")

    entries, meta = read_corpus_manifest(args.manifest)
    root = Path(meta["root"])
    picks, seen = [], set()
    for entry in entries:
        if entry.split != "test" or entry.duration_s > args.max_duration_s:
            continue
        if entry.speaker in seen:
            continue
        seen.add(entry.speaker)
        picks.append(entry)
        if len(picks) >= args.limit:
            break

    per_utterance = [analyse_utterance(entry, root) for entry in picks]
    rows = [row for utterance in per_utterance for row in utterance["frames"]]
    for utterance in per_utterance:
        mark_runs(utterance["frames"])

    ok_rows = [row for row in rows if row["reason"] == "ok"]
    calibration = {key: quantiles([row[key] for row in ok_rows])
                   for key in ("hps_z", "cepstrum_prominence", "flatness", "level_db",
                               "min_cmnd")}
    label_rejected(rows, calibration)

    rejected = [row for row in rows if row["reason"] == "no_candidate"]
    labels = Counter(row["label"] for row in rejected)
    between = Counter(row["label"] for row in rejected if row["run_after"] >= 2)
    # A frame is "between voiced" when it sits between two accepted frames, not merely after
    # one: the onset/offset of voicing is where voicing genuinely starts and stops.
    voiced_neighbourhood = Counter()
    for utterance in per_utterance:
        frames = utterance["frames"]
        for index, row in enumerate(frames):
            if row["reason"] != "no_candidate":
                continue
            previous = frames[index - 1]["reason"] if index > 0 else None
            following = frames[index + 1]["reason"] if index + 1 < len(frames) else None
            row["previous"] = previous
            row["next"] = following
            if previous == "ok" and following == "ok":
                voiced_neighbourhood[row["label"]] += 1

    suspicious = []
    for utterance in per_utterance:
        for row in utterance["frames"]:
            if row["label"] == "suspected_miss" and row.get("run_after", 0) >= 2 \
                    and len(suspicious) < 12:
                suspicious.append({"utt_id": utterance["utt_id"], **row})

    reasons = Counter(row["reason"] for row in rows)
    special = next((entry for entry in picks if entry.utt_id == args.special), None)
    report = {
        "experiment": "EXP-0017",
        "manifest": str(args.manifest),
        "utterances": len(per_utterance),
        "frames": len(rows),
        "reason_totals": dict(sorted(reasons.items())),
        "no_candidate_frames": len(rejected),
        "calibration_on_ok_frames": calibration,
        "labels": dict(sorted(labels.items())),
        "labels_inside_a_run_of_two_or_more": dict(sorted(between.items())),
        "labels_between_two_ok_frames": dict(sorted(voiced_neighbourhood.items())),
        "min_cmnd_buckets": dict(sorted(Counter(cmnd_bucket(row["min_cmnd"])
                                                for row in rejected).items())),
        "level_buckets": dict(sorted(Counter(level_bucket(row["level_db"])
                                             for row in rejected).items())),
        "label_by_level": cross_tab(rejected, lambda row: level_bucket(row["level_db"])),
        "label_by_min_cmnd": cross_tab(rejected, lambda row: cmnd_bucket(row["min_cmnd"])),
        "median_flatness_by_label": median_by_label(rejected, "flatness"),
        "median_min_cmnd_by_label": median_by_label(rejected, "min_cmnd"),
        "median_level_db_by_label": median_by_label(rejected, "level_db"),
        "suspected_miss_examples": suspicious,
        "special_case": compare_ssb04350040(special, root) if special is not None else None,
        "per_utterance": [{"utt_id": utterance["utt_id"], "speaker": utterance["speaker"],
                           "frames": len(utterance["frames"]),
                           "reasons": utterance["reasons"],
                           "labels": dict(sorted(Counter(
                               row["label"] for row in utterance["frames"]
                               if row["label"]).items()))}
                          for utterance in per_utterance],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("utterances", "frames", "reason_totals", "calibration_on_ok_frames",
                       "labels", "labels_inside_a_run_of_two_or_more",
                       "labels_between_two_ok_frames", "min_cmnd_buckets",
                       "level_buckets", "label_by_level", "label_by_min_cmnd",
                       "median_flatness_by_label", "median_min_cmnd_by_label",
                       "median_level_db_by_label")}, indent=2, ensure_ascii=False))
    if report["special_case"]:
        print(json.dumps(report["special_case"], indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
