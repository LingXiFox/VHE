"""EXP-0029: render the analysis-resynthesis mechanical source for listening.

Runs robotize() on a handful of corpus utterances, writes real/robot pairs to the
derived workspace (never into Git) and measures the same descriptors used to
diagnose the formant synthesiser in EXP-0023, so the two mechanical sources can be
compared on identical numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if len(Path(__file__).resolve().parents) >= 3 and (Path(__file__).resolve().parents[2] / "src").is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.audio import read_wav, write_wav  # noqa: E402
from vhe.data.align_index import read_manifest  # noqa: E402
from vhe.data.features import log_mel_spectrogram  # noqa: E402
from vhe.data.f0 import estimate_f0  # noqa: E402
from vhe.data.resample import resample  # noqa: E402
from vhe.data.robotize import robotize  # noqa: E402

SAMPLE_RATE = 24000


def logmel_distance(reference: np.ndarray, other: np.ndarray, sample_rate: int) -> float:
    frames = min(reference.size, other.size)
    kwargs = dict(sample_rate=sample_rate, n_fft=1024, n_mels=40)
    a = log_mel_spectrogram(reference[:frames], **kwargs)
    b = log_mel_spectrogram(other[:frames], **kwargs)
    length = min(a.shape[0], b.shape[0])
    return float(np.mean(np.abs(a[:length] - b[:length])))


def periodicity(signal: np.ndarray, sample_rate: int, fmin: float = 65.0,
                fmax: float = 400.0) -> tuple[float, float]:
    centred = signal - signal.mean()
    size = len(centred)
    lags = np.arange(int(sample_rate / fmax), int(sample_rate / fmin) + 1)
    correlation = np.array([np.dot(centred[:size - lag], centred[lag:]) for lag in lags])
    correlation = correlation / np.linalg.norm(centred) ** 2 * size / (size - lags)
    best = int(np.argmax(correlation))
    return float(sample_rate / lags[best]), float(correlation[best])


def high_frequency_ratio(signal: np.ndarray, sample_rate: int, cut_hz: float = 6000.0) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    total = spectrum.sum()
    return float(spectrum[freqs >= cut_hz].sum() / total) if total > 0 else 0.0


def spectral_flatness(signal: np.ndarray, sample_rate: int, low: float = 300.0,
                      high: float = 4000.0) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) + 1e-12
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    band = spectrum[(freqs >= low) & (freqs <= high)]
    return float(np.exp(np.log(band).mean()) / band.mean())


def describe(signal: np.ndarray, sample_rate: int) -> dict:
    track = estimate_f0(signal, sample_rate)
    voiced = track.voiced_f0
    voiced = voiced[np.isfinite(voiced)]
    median_f0 = float(np.median(voiced)) if voiced.size else 0.0
    ac_f0, ac_peak = periodicity(signal, sample_rate)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    return {
        "seconds": round(signal.size / sample_rate, 3),
        "rms_db": round(20 * np.log10(rms + 1e-12), 2),
        "peak": round(float(np.max(np.abs(signal))), 3),
        "median_f0_hz": round(median_f0, 1),
        "ac_f0_hz": round(ac_f0, 1),
        "periodicity": round(ac_peak, 3),
        "hf_ratio": round(high_frequency_ratio(signal, sample_rate), 5),
        "flatness": round(spectral_flatness(signal, sample_rate), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--speakers", type=int, default=3)
    parser.add_argument("--per-speaker", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bands", type=int, default=24,
                        help="mel bands the spectral envelope is flattened onto")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")

    _, rows = read_manifest(args.manifest)
    pool = [row for row in rows if row.get("split") == args.split and row.get("renderable")]
    if not pool:
        raise SystemExit(f"no renderable rows in split {args.split!r}")
    speakers: list[str] = []
    for row in pool:
        if row["speaker"] not in speakers:
            speakers.append(row["speaker"])
    rng = np.random.default_rng(args.seed)
    chosen = speakers if len(speakers) <= args.speakers else list(rng.choice(speakers, args.speakers, replace=False))

    pairs = []
    args.out.mkdir(parents=True)
    for speaker in chosen:
        owned = [row for row in pool if row["speaker"] == speaker][: args.per_speaker]
        for row in owned:
            audio, rate = read_wav(args.root / row["wav_path"])
            audio = np.asarray(audio, dtype=np.float64)
            if rate != SAMPLE_RATE:
                audio = np.asarray(resample(audio, rate, SAMPLE_RATE), dtype=np.float64)
            mechanical = robotize(audio, SAMPLE_RATE, bands=args.bands)
            directory = args.out / row["utt_id"]
            directory.mkdir()
            write_wav(directory / "real.wav", audio.astype(np.float32), SAMPLE_RATE)
            write_wav(directory / "robot.wav", mechanical, SAMPLE_RATE)
            record = {
                "utt_id": row["utt_id"],
                "speaker": speaker,
                "pinyin": row.get("pinyin", []),
                "logmel": round(logmel_distance(audio, mechanical, SAMPLE_RATE), 3),
                "real": describe(audio, SAMPLE_RATE),
                "robot": describe(np.asarray(mechanical, dtype=np.float64), SAMPLE_RATE),
            }
            pairs.append(record)
            print(json.dumps(record, ensure_ascii=False))

    summary = {
        "pairs": len(pairs),
        "speakers": chosen,
        "seed": args.seed,
        "bands": args.bands,
        "logmel_mean": round(float(np.mean([p["logmel"] for p in pairs])), 3),
        "real": {key: round(float(np.mean([p["real"][key] for p in pairs])), 3)
                 for key in ("rms_db", "peak", "median_f0_hz", "periodicity", "hf_ratio", "flatness")},
        "robot": {key: round(float(np.mean([p["robot"][key] for p in pairs])), 3)
                  for key in ("rms_db", "peak", "median_f0_hz", "periodicity", "hf_ratio", "flatness")},
    }
    (args.out / "report.json").write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
