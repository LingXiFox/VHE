"""EXP-0011 probe: octave errors of the F0 estimator on harmonic-rich frames.

Stimulus family: a harmonic comb shaped by one resonance (a stand-in for a vowel whose
first formant sits at F). Deterministic phases, so every run sees the identical waveform.
Plus: the /a/ vowel of the Mandarin physical synthesizer (the signal that exposed the
failure), pure tones as a regression guard, silence and white noise as gating guards.

Classification of one frame: within 5% of the true f0 -> ok; within 5% of 2*f0 -> octave
up; within 5% of f0/2 -> octave down; anything else -> other.

    python experiments/EXP-0011/probe_octave.py --out experiments/EXP-0011/logs/probe_before.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

SR = 24000


def harmonic_comb(f0: float, formant_hz: float, duration_s: float = 0.5,
                  sample_rate: int = SR, jitter: float = 0.0, seed: int = 7) -> np.ndarray:
    """Harmonic comb, amplitudes peaked at formant_hz, fixed phases."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_s * sample_rate))) / sample_rate
    out = np.zeros_like(t)
    for k in range(1, 60):
        freq = k * f0
        if freq >= 0.45 * sample_rate:
            break
        amp = (1.0 / k ** 1.1) * np.exp(-0.5 * ((freq - formant_hz) / 90.0) ** 2) + 0.02 / k
        drift = 0.0 if jitter == 0.0 else jitter * f0 * np.sin(2.0 * np.pi * 5.0 * t + k)
        out += amp * np.cos(2.0 * np.pi * freq * t + drift * t + rng.uniform(0.0, 2.0 * np.pi))
    return (0.5 * out / np.max(np.abs(out))).astype(np.float64)


def mandarin_vowel(f0: float, phoneme: str = "a", template: str = "yang") -> np.ndarray:
    from vhe.synth import (PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve,
                           SynthesisRequest, Timing, get_backend)

    request = SynthesisRequest(
        template=PRESETS[template],
        phonemes=PhonemeSequence((PhonemeSegment(phoneme, 0.4),), inventory="mandarin"),
        pitch=PitchCurve.constant(f0, 0.4),
        timing=Timing(0.0, 0.0), sample_rate=SR)
    return get_backend("mandarin_formant").render(request).audio.astype(np.float64)


def add_noise(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(audio.size)
    scale = np.sqrt(np.mean(audio * audio)) / (10.0 ** (snr_db / 20.0))
    return audio + scale * noise / np.sqrt(np.mean(noise * noise))


def classify(measured: np.ndarray, true_f0: float) -> dict:
    valid = measured[measured > 0.0]
    counts = {"ok": 0, "octave_up": 0, "octave_down": 0, "other": 0}
    for value in valid:
        if abs(value - true_f0) <= 0.05 * true_f0:
            counts["ok"] += 1
        elif abs(value - 2.0 * true_f0) <= 0.05 * 2.0 * true_f0:
            counts["octave_up"] += 1
        elif abs(value - 0.5 * true_f0) <= 0.05 * 0.5 * true_f0:
            counts["octave_down"] += 1
        else:
            counts["other"] += 1
    total = max(1, int(valid.size))
    entry = {key: counts[key] for key in counts}
    entry["frames"] = int(valid.size)
    entry["median_hz"] = round(float(np.median(valid)), 2) if valid.size else None
    entry["ok_ratio"] = round(counts["ok"] / total, 4)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from vhe.data.f0 import estimate_f0

    result: dict = {"numpy": np.__version__, "python": sys.version.split()[0],
                    "platform": platform.platform(), "sr": SR, "comb": {}, "vowel": {},
                    "guards": {}}
    for f0 in (100.0, 150.0, 200.0, 220.0, 240.0, 300.0):
        for formant in (500.0, 700.0, 800.0, 1000.0, 1250.0):
            track = estimate_f0(harmonic_comb(f0, formant), SR)
            entry = classify(track.voiced_f0, f0)
            entry["expected_hz"] = f0
            entry["formant_hz"] = formant
            result["comb"][f"{f0:.0f}Hz_F{formant:.0f}"] = entry
    for f0 in (110.0, 200.0, 220.0, 240.0, 350.0):
        for phoneme in ("a", "i", "u"):
            track = estimate_f0(mandarin_vowel(f0, phoneme), SR)
            entry = classify(track.voiced_f0, f0)
            entry["expected_hz"] = f0
            result["vowel"][f"{phoneme}_{f0:.0f}Hz"] = entry

    silence = np.zeros(SR // 2)
    result["guards"]["silence"] = classify(estimate_f0(silence, SR).voiced_f0, 200.0)
    rng = np.random.default_rng(3)
    noise = 0.2 * rng.standard_normal(SR // 2)
    result["guards"]["white_noise"] = classify(estimate_f0(noise, SR).voiced_f0, 200.0)
    for f0 in (65.0, 220.0, 440.0, 880.0):
        t = np.arange(SR // 2) / SR
        tone = (0.5 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float64)
        result["guards"][f"sine_{f0:.0f}"] = classify(estimate_f0(tone, SR).voiced_f0, f0)

    # Robustness: the fix must not only work on clean synthetic tones. White noise at a
    # known SNR is added to the two signals whose half-period dip was the problem.
    result["robustness"] = {}
    clean = {"comb_220Hz_F500": harmonic_comb(220.0, 500.0),
             "vowel_a_220Hz": mandarin_vowel(220.0)}
    for name, audio in clean.items():
        for snr in (30.0, 20.0, 10.0):
            for seed in (11, 12):
                noisy = add_noise(audio, snr, seed)
                track = estimate_f0(noisy, SR)
                entry = classify(track.voiced_f0, 220.0)
                entry["snr_db"] = snr
                entry["seed"] = seed
                result["robustness"][f"{name}_snr{snr:.0f}_s{seed}"] = entry

    numeric = {key: entry for key, entry in result["comb"].items()}
    total_frames = sum(entry["frames"] for entry in numeric.values())
    total_up = sum(entry["octave_up"] for entry in numeric.values())
    total_down = sum(entry["octave_down"] for entry in numeric.values())
    vowel_frames = sum(entry["frames"] for entry in result["vowel"].values())
    vowel_up = sum(entry["octave_up"] for entry in result["vowel"].values())
    result["summary"] = {
        "comb_frames": total_frames,
        "comb_octave_up": total_up,
        "comb_octave_down": total_down,
        "comb_octave_up_ratio": round(total_up / max(1, total_frames), 4),
        "comb_clean_cases": sum(1 for e in numeric.values() if e["octave_up"] == 0),
        "comb_cases": len(numeric),
        "vowel_frames": vowel_frames,
        "vowel_octave_up": vowel_up,
        "vowel_octave_up_ratio": round(vowel_up / max(1, vowel_frames), 4),
        "robust_frames": sum(e["frames"] for e in result["robustness"].values()),
        "robust_octave_up": sum(e["octave_up"] for e in result["robustness"].values()),
        "robust_octave_down": sum(e["octave_down"] for e in result["robustness"].values()),
        "robust_cases_clean": sum(1 for e in result["robustness"].values()
                                  if not e["octave_up"] and not e["octave_down"]),
        "robust_cases": len(result["robustness"]),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print("guards:", json.dumps(result["guards"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
