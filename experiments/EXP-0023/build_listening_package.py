"""EXP-0023: render the listening package (real vs forced-aligned vs DP mechanical).

One variable: where the per-phoneme durations come from. Arm B uses chain-HMM phone spans
from the aligner; arm A keeps the energy-valley DP windows plus g2p nominal onset/nucleus
durations. Everything else is shared, including the voice template and the pitch tracker.

Run (4090 box):
    PYTHONPATH=~/VHE/src .venv/bin/python experiments/EXP-0023/build_listening_package.py \
        --manifest ~/datasets/VHE-Datasets/derived/aishell3-scan/v4/manifest.jsonl \
        --root ~/datasets/VHE-Datasets/AISHELL-3/extracted \
        --model ~/datasets/VHE-Datasets/derived/aishell3-aligner/v1/model.npz \
        --out ~/datasets/VHE-Datasets/derived/aishell3-listen/EXP-0023 \
        --splits dev --speakers 4 --per-speaker 5 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
if len(_HERE.parents) >= 3 and (_HERE.parents[2] / "src").is_dir():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))

from vhe.audio import read_wav, write_wav                       # noqa: E402
from vhe.data.align_index import read_manifest, resolve_root, select_rows  # noqa: E402
from vhe.data.aligner import PhoneAligner                      # noqa: E402
from vhe.data.features import log_mel_spectrogram              # noqa: E402
from vhe.data.pairing import build_pair                        # noqa: E402
from vhe.synth import PRESETS                                  # noqa: E402
from vhe.synth.mandarin import MandarinFormantSynth            # noqa: E402


def spoke_spans(aligner: PhoneAligner, wav_path: Path, pinyins):
    """Align one file and shift the spans into the file's own time base."""
    audio, sample_rate = read_wav(wav_path)
    result = aligner.align(audio, sample_rate, pinyins)
    offset = float(result.offset_s)
    spans = [(span.phoneme, span.start_s + offset, span.end_s + offset,
              float(span.confidence)) for span in result.spans]
    return spans, float(result.confidence)


def logmel_distance(reference: np.ndarray, other: np.ndarray, sample_rate: int) -> float:
    """Mean absolute difference of log-mel spectrograms over the shared frame count."""
    frames = min(reference.size, other.size)
    kw = dict(sample_rate=sample_rate, n_fft=1024, n_mels=40)
    a = log_mel_spectrogram(reference[:frames], **kw)
    b = log_mel_spectrogram(other[:frames], **kw)
    length = min(a.shape[0], b.shape[0])
    return float(np.mean(np.abs(a[:length] - b[:length])))


def onset_fraction(result) -> float:
    """Share of each syllable's duration spent on its onset phoneme, averaged."""
    values = []
    for allocation in result.allocations:
        total = sum(entry["duration_s"] for entry in allocation)
        if len(allocation) > 1 and total > 0:
            values.append(allocation[0]["duration_s"] / total)
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="Output directory; must not exist")
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--speakers", type=int, default=4)
    parser.add_argument("--per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", default="yang")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        parser.exit(2, f"{out} exists; listening packages are never overwritten\n")
    meta, rows = read_manifest(args.manifest)
    root = resolve_root(meta, args.root)
    picked = select_rows(rows, splits=args.splits.split(","), speakers=args.speakers,
                         per_speaker=args.per_speaker, seed=args.seed)
    aligner = PhoneAligner.from_npz(args.model)
    backend = MandarinFormantSynth()
    template = PRESETS[args.template]
    out.mkdir(parents=True)

    records, allocations = [], []
    for index, row in enumerate(picked, 1):
        utt_id = str(row["utt_id"])
        wav_path = root / str(row["wav_path"])
        pinyins = [str(token) for token in row["pinyin"]]
        record = {"utt_id": utt_id, "speaker": row.get("speaker"), "split": row.get("split"),
                  "wav_path": str(row["wav_path"]), "syllables": len(pinyins)}
        try:
            spans, alignment_confidence = spoke_spans(aligner, wav_path, pinyins)
        except Exception as error:                      # noqa: BLE001 - recorded, not hidden
            record["spans_error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            continue
        record["alignment_confidence"] = round(alignment_confidence, 4)

        common = dict(template=template, backend=backend)
        forced = build_pair(wav_path, pinyins, alignment="forced", phone_spans=spans, **common)
        dp = build_pair(wav_path, pinyins, alignment="dp", **common)
        record["forced_status"] = forced.status
        record["dp_status"] = dp.status
        record["forced_ready"] = forced.readiness.get("training_ready")
        record["dp_ready"] = dp.readiness.get("training_ready")
        record["forced_reasons"] = ",".join(forced.readiness.get("reasons", []))
        record["dp_reasons"] = ",".join(dp.readiness.get("reasons", []))
        record["forced_issues"] = ",".join(sorted({issue["code"] for issue in forced.issues}))
        record["dp_issues"] = ",".join(sorted({issue["code"] for issue in dp.issues}))
        record["forced_onset_fraction"] = round(onset_fraction(forced), 4)
        record["dp_onset_fraction"] = round(onset_fraction(dp), 4)

        if forced.mechanical is not None and forced.real is not None:
            rate = forced.sample_rate
            speaker_dir = out / utt_id
            speaker_dir.mkdir()
            write_wav(speaker_dir / "real.wav", forced.real, rate)
            write_wav(speaker_dir / "mech_forced.wav", forced.mechanical, rate)
            record["logmel_forced"] = round(logmel_distance(forced.real, forced.mechanical, rate), 5)
            for allocation in forced.allocations:
                allocations.append({"utt_id": utt_id, "arm": "forced", "phones": allocation})
        if dp.mechanical is not None and dp.real is not None:
            rate = dp.sample_rate
            speaker_dir = out / utt_id
            if not (speaker_dir / "mech_dp.wav").exists():
                speaker_dir.mkdir(exist_ok=True)
                write_wav(speaker_dir / "real.wav", dp.real, rate)
            write_wav(speaker_dir / "mech_dp.wav", dp.mechanical, rate)
            record["logmel_dp"] = round(logmel_distance(dp.real, dp.mechanical, rate), 5)
            for allocation in dp.allocations:
                allocations.append({"utt_id": utt_id, "arm": "dp", "phones": allocation})
        records.append(record)
        print(f"[{index}/{len(picked)}] {utt_id} forced={forced.status} dp={dp.status}",
              flush=True)

    fields = ["utt_id", "speaker", "split", "syllables", "wav_path", "alignment_confidence",
              "forced_status", "dp_status", "forced_ready", "dp_ready", "forced_reasons",
              "dp_reasons", "forced_issues", "dp_issues", "forced_onset_fraction",
              "dp_onset_fraction", "logmel_forced", "logmel_dp", "spans_error"]
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "allocations.json").write_text(json.dumps(allocations, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    summary = {
        "selected": len(picked), "rendered": sum(1 for r in records
                                                 if r.get("forced_status") == "ok"),
        "forced_ok": sum(1 for r in records if r.get("forced_status") == "ok"),
        "dp_ok": sum(1 for r in records if r.get("dp_status") == "ok"),
        "forced_ready": sum(1 for r in records if r.get("forced_ready")),
        "dp_ready": sum(1 for r in records if r.get("dp_ready")),
        "spans_errors": sum(1 for r in records if "spans_error" in r),
        "logmel_forced_mean": round(float(np.mean([r["logmel_forced"] for r in records
                                                   if "logmel_forced" in r])), 5)
        if any("logmel_forced" in r for r in records) else None,
        "logmel_dp_mean": round(float(np.mean([r["logmel_dp"] for r in records
                                               if "logmel_dp" in r])), 5)
        if any("logmel_dp" in r for r in records) else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
