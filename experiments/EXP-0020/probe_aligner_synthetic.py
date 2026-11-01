"""EXP-0020 probe: how well does the chain-HMM aligner recover synthetic phone boundaries?

iterations=0 means the flat-start parameters are used as-is (no Baum-Welch), which is the
control for "did EM actually help".

The synthetic speaker is six band-limited noises (a=400-900, i=1800-2600, s=4000-6000,
sh=2500-3200, t=900-1400, n=200-350 Hz) with 10 ms raised-cosine edges inside a 150 ms silence
pad. Each utterance is 3-5 phones of 100-200 ms, so the only cue a boundary exists is the change
of spectral band: there is no coarticulation, no F0, no formant transition, no noise.

What it measures, per configuration: the fraction of training utterances whose Baum-Welch
likelihood sequence is monotone, the maximum and mean boundary error over held-out utterances,
the minimum span confidence, and the energy-gate crop error. Phone sequences with two adjacent
identical phones are excluded from scoring because their internal boundary is not observable.

Usage: python experiments/EXP-0020/probe_aligner_synthetic.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.data.aligner import (PreparedUtterance, crop_speech,  # noqa: E402
                              train_aligner)
from vhe.data.features import SAMPLE_RATE, acoustic_features  # noqa: E402

BANDS = {"a": (400.0, 900.0), "i": (1800.0, 2600.0), "s": (4000.0, 6000.0),
         "sh": (2500.0, 3200.0), "t": (900.0, 1400.0), "n": (200.0, 350.0)}
VOCABULARY = tuple(BANDS)


def colour(phone: str, duration: float, rng: np.random.Generator) -> np.ndarray:
    samples = int(duration * SAMPLE_RATE)
    spectrum = np.fft.rfft(rng.normal(size=samples))
    frequencies = np.fft.rfftfreq(samples, 1.0 / SAMPLE_RATE)
    low, high = BANDS[phone]
    shaped = np.fft.irfft(spectrum * ((frequencies >= low) & (frequencies <= high)), n=samples)
    shaped /= np.abs(shaped).max() + 1e-9
    ramp = int(0.01 * SAMPLE_RATE)
    envelope = np.ones(samples)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return (0.3 * shaped * envelope).astype(np.float32)


def synthetic(phones, seed, lead=0.15, tail=0.15):
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead * SAMPLE_RATE), dtype=np.float32)]
    boundaries = []
    cursor = lead
    for phone in phones:
        duration = float(rng.uniform(0.10, 0.20))
        parts.append(colour(phone, duration, rng))
        boundaries.append((cursor, cursor + duration))
        cursor += duration
    parts.append(np.zeros(int(tail * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(parts), boundaries


def prepare(identifier, phones, seed):
    audio, _boundaries = synthetic(phones, seed)
    cropped, (lead, tail) = crop_speech(audio, SAMPLE_RATE)
    return PreparedUtterance(identifier, acoustic_features(cropped, SAMPLE_RATE), tuple(phones),
                             (lead, tail), audio.size / SAMPLE_RATE, lead * 0.01)


def evaluate(aligner, sequences, seeds):
    errors, confidences = [], []
    for phones, seed in zip(sequences, seeds):
        audio, boundaries = synthetic(phones, seed)
        cropped, (lead, _tail) = crop_speech(audio, SAMPLE_RATE)
        result = aligner.align_phones(acoustic_features(cropped, SAMPLE_RATE), phones,
                                      offset_s=lead * 0.01)
        predicted = np.array([span.start_s for span in result.spans[1:]])
        truth = np.array([start for start, _end in boundaries[1:]])
        errors.append(float(np.abs(predicted - truth).max()))
        confidences.append(result.confidence)
    return errors, confidences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-utterances", type=int, default=30)
    parser.add_argument("--eval-utterances", type=int, default=8)
    args = parser.parse_args()

    rng = np.random.default_rng(11)
    items = []
    for index in range(args.train_utterances):
        phones = tuple(rng.choice(VOCABULARY, size=int(rng.integers(3, 6))))
        items.append(prepare(f"train{index}", phones, 1000 + index))

    evaluation_rng = np.random.default_rng(31)
    sequences = [tuple(evaluation_rng.choice(VOCABULARY, size=4, replace=False))
                 for _ in range(args.eval_utterances)]
    seeds = [2000 + index for index in range(args.eval_utterances)]

    runs = []
    for components in (1, 2):
        for iterations in (0, 1, 4):
            aligner, report = train_aligner(items, n_components=components,
                                            iterations=iterations, seed=42)
            errors, confidences = evaluate(aligner, sequences, seeds)
            runs.append({"components": components, "iterations": iterations,
                         "monotone": report["monotone"],
                         "trained_loglik": (report["history"][-1]["loglik"]
                                           if report["history"] else None),
                         "max_boundary_error_s": round(max(errors), 4),
                         "mean_boundary_error_s": round(float(np.mean(errors)), 4),
                         "min_confidence": round(float(np.min(confidences)), 4),
                         "errors_s": [round(value, 4) for value in errors]})
            print(runs[-1])

    audio, _boundaries = synthetic(("a", "i", "s"), 4000, lead=0.15, tail=0.15)
    _cropped, (lead, tail) = crop_speech(audio, SAMPLE_RATE)
    payload = {"platform": platform.platform(), "python": sys.version.split()[0],
               "sample_rate": SAMPLE_RATE, "hop_s": 0.010, "seed": 42,
               "train_utterances": args.train_utterances,
               "eval_utterances": args.eval_utterances,
               "crop_frames": [lead, tail], "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
