"""Can interior F0 gaps be bridged? (EXP-0024)

Every pair blocked by the readiness probe (EXP-0023) fails for exactly one reason:
pitch_gap_interpolated, i.e. an interior run of frames the tracker refused, longer than
max_pitch_gap_s = 20 ms, which build_pair bridges by linear interpolation and flags.

The threshold is a policy, not a measurement. This probe measures the quantity the policy should
depend on: how much F0 error a linear bridge introduces, as a function of gap length.

Arm A (self-calibration): inside long voiced runs, delete k centre frames, bridge them by linear
interpolation in log-F0 from the surviving neighbours and compare against the tracker's own values.
This measures information lost by the bridge, assuming the tracker is right on voiced frames; it
cannot detect a tracker that is confidently wrong.

Arm B (independent check on real gaps): for every rejected frame that sits between two voiced
frames, estimate F0 by normalised autocorrelation restricted to +-1/2 octave around the linear
bridge, so the independent estimator cannot jump octaves. Frames whose peak correlation is below
0.30 are reported as aperiodic instead of periodic.

Decision rule, fixed before running: bridge gaps up to G*, the largest length whose 90th percentile
error is <= 50 cents in arm A and <= 50 cents for periodic frames in arm B, provided the aperiodic
share of arm B frames at that length stays <= 10%. If no length qualifies, keep 20 ms.

Usage:
    python experiments/EXP-0024/probe_gap_policy.py --manifest ... --root ... --ready-log ... \
        --out experiments/EXP-0024/logs/gap_policy_dev60.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vhe.audio import read_wav
from vhe.synth.phonology import MANDARIN_PHONEMES
from vhe.data.aligner import PhoneAligner
from vhe.data.pitch import track_pitch
from vhe.data.resample import resample

SAMPLE_RATE = 24000
MAX_GAP_FRAMES = 10
APERIODIC_AC = 0.30
OCTAVE_HALF = 2.0 ** 0.5
WINDOW_S = 0.040


def cents(a: float, b: float) -> float:
    return 1200.0 * float(np.log2(a / b))


def bridge_anchors_valid(f0: np.ndarray, run_start: int, run_stop: int) -> bool:
    left, right = run_start - 1, run_stop
    return 0 <= left and right < f0.size and f0[left] > 0 and f0[right] > 0


def bridge_errors(f0: np.ndarray, truth: np.ndarray, run_start: int, run_stop: int) -> list[float]:
    """Per-frame error (cents) of a linear log-F0 bridge across [run_start, run_stop)."""
    if not bridge_anchors_valid(f0, run_start, run_stop):
        return []
    log_left, log_right = np.log2(f0[run_start - 1]), np.log2(f0[run_stop])
    errors = []
    for offset, index in enumerate(range(run_start, run_stop), 1):
        fraction = offset / (run_stop - run_start + 1)
        predicted = 2.0 ** (log_left + fraction * (log_right - log_left))
        errors.append(abs(cents(predicted, float(truth[index]))))
    return errors


def normalised_autocorrelation(frame: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= frame.size:
        return 0.0
    left = frame[: frame.size - lag]
    right = frame[lag:]
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right))) + 1e-12
    return float(np.dot(left, right) / denominator)


def independent_estimate(audio: np.ndarray, center_s: float, reference_hz: float) -> tuple[float, float]:
    """Best autocorrelation peak within +-1/2 octave of reference_hz; returns (f0, peak)."""
    half = int(WINDOW_S * SAMPLE_RATE / 2)
    center = int(round(center_s * SAMPLE_RATE))
    start, stop = center - half, center + half
    if start < 0 or stop > audio.size:
        return 0.0, 0.0
    frame = audio[start:stop].astype(np.float64)
    frame = frame - frame.mean()
    window = np.hanning(frame.size)
    frame = frame * window
    low = int(np.floor(SAMPLE_RATE / (reference_hz * OCTAVE_HALF)))
    high = int(np.ceil(SAMPLE_RATE / (reference_hz / OCTAVE_HALF)))
    low, high = max(2, low), min(frame.size - 2, high)
    if high <= low:
        return 0.0, 0.0
    peaks = [(normalised_autocorrelation(frame, lag), lag) for lag in range(low, high + 1)]
    peak, lag = max(peaks)
    return SAMPLE_RATE / lag, peak


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, position = [], 0
    while position < mask.size:
        if mask[position]:
            stop = position
            while stop < mask.size and mask[stop]:
                stop += 1
            runs.append((position, stop))
            position = stop
        else:
            position += 1
    return runs


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {"count": int(array.size), "mean": round(float(array.mean()), 2),
            "p50": round(float(np.percentile(array, 50)), 2),
            "p90": round(float(np.percentile(array, 90)), 2),
            "p99": round(float(np.percentile(array, 99)), 2),
            "max": round(float(array.max()), 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ready-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None,
                        help="aligner model; without it the phoneme-context arm is skipped")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        parser.exit(2, f"{out} exists; probes are never overwritten\n")
    ready = json.loads(Path(args.ready_log).read_text(encoding="utf-8"))
    utt_ids = [item["utt_id"] for item in ready["records"]]
    if args.limit:
        utt_ids = utt_ids[: args.limit]

    rows = {}
    with open(args.manifest, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "meta":
                continue
            if str(row["utt_id"]) in utt_ids:
                rows[str(row["utt_id"])] = row

    aligner = PhoneAligner.from_npz(args.model) if args.model else None
    sonorant_gap_frames = voiceless_gap_frames = unknown_gap_frames = 0
    arm_a: dict[int, list[float]] = defaultdict(list)
    arm_b: dict[int, list[float]] = defaultdict(list)
    arm_b_aperiodic: Counter = Counter()
    arm_b_total: Counter = Counter()
    per_utterance = []

    for utt_id in utt_ids:
        row = rows.get(utt_id)
        if row is None:
            continue
        audio, rate = read_wav(Path(args.root) / row["wav_path"])
        if rate != SAMPLE_RATE:
            audio = resample(np.asarray(audio, dtype=np.float32), rate, SAMPLE_RATE)
        track = track_pitch(audio, SAMPLE_RATE)
        f0 = np.array([frame.f0_hz for frame in track.frames])
        confidence = np.array([frame.confidence for frame in track.frames])
        times = np.array([frame.time_s for frame in track.frames])
        reasons = Counter(frame.reason for frame in track.frames)

        # Arm C: which phoneme does each rejected frame sit in? A bridge inside a sonorant
        # replaces real phonation; a bridge inside a voiceless phone replaces silence or noise.
        covers = []
        if aligner is not None:
            pinyins = [str(token) for token in row["pinyin"]]
            alignment = aligner.align(np.asarray(audio, dtype=np.float32), SAMPLE_RATE, pinyins)
            covers = [(span.start_s, span.end_s, span.phoneme) for span in alignment.spans]
        sonorant = {name for name, phone in MANDARIN_PHONEMES.items()
                    if phone.kind in ("vowel", "diphthong", "nasal", "lateral", "approximant")}

        # Arm A: mask k centre frames inside long voiced runs.
        for start, stop in runs_of(f0 > 0):
            length = stop - start
            if length < 2 * MAX_GAP_FRAMES + 3:
                continue
            centre = start + length // 2
            for k in range(1, MAX_GAP_FRAMES + 1):
                mask_start, mask_stop = centre - k // 2, centre - k // 2 + k
                if mask_start - 1 < start or mask_stop + 1 > stop:
                    continue
                if confidence[centre] < 0.5:
                    continue
                errors = bridge_errors(f0, f0, mask_start, mask_stop)
                if errors:
                    arm_a[k].append(float(np.mean(errors)))

        # Arm B: real rejected runs enclosed by voiced frames.
        for start, stop in runs_of(f0 <= 0):
            if start == 0 or stop >= f0.size:
                continue
            length = stop - start
            if covers:
                for index in range(start, stop):
                    moment = float(times[index])
                    name = next((phone for begin, end, phone in covers
                                 if begin <= moment < end), None)
                    if name is None:
                        unknown_gap_frames += 1
                    elif name in sonorant:
                        sonorant_gap_frames += 1
                    else:
                        voiceless_gap_frames += 1
            if length > MAX_GAP_FRAMES:
                continue
            if not bridge_anchors_valid(f0, start, stop):
                continue
            log_left, log_right = np.log2(f0[start - 1]), np.log2(f0[stop])
            for offset, index in enumerate(range(start, stop), 1):
                fraction = offset / (length + 1)
                reference = 2.0 ** (log_left + fraction * (log_right - log_left))
                estimate, peak = independent_estimate(audio, float(times[index]), reference)
                arm_b_total[length] += 1
                if peak < APERIODIC_AC or estimate <= 0.0:
                    arm_b_aperiodic[length] += 1
                else:
                    arm_b[length].append(abs(cents(estimate, reference)))
        per_utterance.append({"utt_id": utt_id, "frames": len(track.frames),
                              "voiced_ratio": round(float(track.voiced_ratio), 4),
                              "reasons": dict(reasons), "arm_a_samples": sum(len(v) for v in arm_a.values())})

    summary = {
        "probe": "gap_policy", "seed": args.seed, "utterances": len(per_utterance),
        "sample_rate": SAMPLE_RATE, "aperiodic_ac_threshold": APERIODIC_AC,
        "arm_a_by_gap_frames": {str(k): describe(v) for k, v in sorted(arm_a.items())},
        "arm_b_periodic_error_by_gap_frames": {str(k): describe(v) for k, v in sorted(arm_b.items())},
        "arm_b_frames_by_gap_frames": {str(k): arm_b_total[k] for k in sorted(arm_b_total)},
        "arm_b_aperiodic_by_gap_frames": {str(k): arm_b_aperiodic[k] for k in sorted(arm_b_total)},
        "arm_c_gap_frames": {"sonorant": sonorant_gap_frames,
                             "voiceless": voiceless_gap_frames,
                             "uncovered": unknown_gap_frames},
        "utterances_detail": per_utterance,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "utterances_detail"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
