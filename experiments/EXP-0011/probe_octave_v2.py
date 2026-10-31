"""EXP-0011 probe v2: F0 octave errors, with the identifiability of every stimulus reported.

v1 (probe_octave.py) scored a harmonic comb as "wrong" whenever the estimate was not within
5% of the nominal f0, but its combs let the fundamental sit up to 29 dB below the dominant
partial. A frame whose waveform is essentially a pure tone at 2*f0 *is* identifiable at 2*f0;
those frames were scored as errors ("other"), which inflated the v1 numbers (799/1410 frames
of the comb family). v1 logs are kept unchanged; this file is the corrected design.

Changes against v1:
  * comb amplitudes keep the fundamental at a controlled level (h1_relative_db), so the
    nominal f0 is actually present; the achieved ratio is written to the JSON;
  * every case reports the measured CMND at the nominal period and at half of it, which is
    the quantity the estimator's first-dip rule actually looks at;
  * summary counts include "other", so an octave-error fix cannot hide behind a re-labelled
    failure mode.

    python experiments/EXP-0011/probe_octave_v2.py --out experiments/EXP-0011/logs/v2_before.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

SR = 24000
H1_RELATIVE_DB = -12.0


def harmonic_comb(f0: float, formant_hz: float, duration_s: float = 0.5,
                  sample_rate: int = SR, h1_relative_db: float = H1_RELATIVE_DB,
                  seed: int = 7) -> tuple[np.ndarray, float]:
    """Harmonic comb with a guaranteed fundamental. Returns (audio, h1_level_db)."""
    rng = np.random.default_rng(seed)
    freqs, amps = [], []
    for k in range(1, 80):
        freq = k * f0
        if freq >= 0.45 * sample_rate:
            break
        freqs.append(freq)
        amps.append((1.0 / k ** 1.1) * np.exp(-0.5 * ((freq - formant_hz) / 90.0) ** 2) + 0.02 / k)
    amps = np.asarray(amps)
    amps /= amps.max()
    floor = 10.0 ** (h1_relative_db / 20.0)
    amps[0] = max(float(amps[0]), floor)
    t = np.arange(int(round(duration_s * sample_rate))) / sample_rate
    out = np.zeros_like(t)
    for freq, amp in zip(freqs, amps):
        out += amp * np.cos(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
    out = 0.5 * out / np.max(np.abs(out))
    return out.astype(np.float64), float(20.0 * np.log10(amps[0]))


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
    entry["accuracy"] = round(counts["ok"] / total, 4)
    return entry


def cmnd_dips(audio: np.ndarray, f0: float) -> dict:
    """CMND at the nominal period and at its half, for identifiability context."""
    from vhe.data.f0 import _difference

    frame = audio[audio.size // 4: audio.size // 4 + 960]
    frame = frame - frame.mean()
    lags, _diff, cmnd = _difference(frame, 1, min(frame.size - 3, 370))
    out = {}
    for name, lag in (("half_period", SR / (2.0 * f0)), ("period", SR / f0)):
        index = int(round(lag)) - 1
        out[name] = round(float(cmnd[index]), 5) if 0 <= index < cmnd.size else None
        out[name + "_lag"] = int(round(lag))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from vhe.data.f0 import estimate_f0

    result: dict = {"numpy": np.__version__, "python": sys.version.split()[0],
                    "platform": platform.platform(), "sr": SR,
                    "h1_relative_db": H1_RELATIVE_DB, "groups": {}}
    stimuli: dict[str, dict] = {"comb": {}, "vowel": {}, "robust": {}, "guards": {}}
    for f0 in (100.0, 150.0, 200.0, 220.0, 240.0, 300.0):
        for formant in (500.0, 700.0, 800.0, 1000.0, 1250.0):
            audio, h1 = harmonic_comb(f0, formant)
            stimuli["comb"][f"{f0:.0f}Hz_F{formant:.0f}"] = {
                "audio": audio, "truth": f0, "h1_level_db": h1}
    for f0 in (110.0, 200.0, 220.0, 240.0, 350.0):
        for phoneme in ("a", "i", "u"):
            stimuli["vowel"][f"{phoneme}_{f0:.0f}Hz"] = {
                "audio": mandarin_vowel(f0, phoneme), "truth": f0, "h1_level_db": None}
    for name, key in (("comb_220Hz_F500", "220Hz_F500"), ("vowel_a_220Hz", "a_220Hz")):
        clean = stimuli["comb"][key]["audio"] if key in stimuli["comb"] else stimuli["vowel"][key]["audio"]
        for snr in (30.0, 20.0, 10.0):
            for seed in (11, 12):
                stimuli["robust"][f"{name}_snr{snr:.0f}_s{seed}"] = {
                    "audio": add_noise(clean, snr, seed), "truth": 220.0, "h1_level_db": None}
    rng = np.random.default_rng(3)
    stimuli["guards"]["silence"] = {"audio": np.zeros(SR // 2), "truth": 200.0, "h1_level_db": None}
    stimuli["guards"]["white_noise"] = {"audio": 0.2 * rng.standard_normal(SR // 2),
                                        "truth": 200.0, "h1_level_db": None}
    for f0 in (65.0, 220.0, 440.0, 880.0):
        t = np.arange(SR // 2) / SR
        stimuli["guards"][f"sine_{f0:.0f}"] = {
            "audio": (0.5 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float64),
            "truth": f0, "h1_level_db": 0.0}

    for group, items in stimuli.items():
        result["groups"][group] = {}
        for name, item in items.items():
            track = estimate_f0(item["audio"], SR)
            entry = classify(track.voiced_f0, item["truth"])
            entry["expected_hz"] = item["truth"]
            if item["h1_level_db"] is not None:
                entry["h1_level_db"] = round(item["h1_level_db"], 1)
            if group in ("comb", "vowel"):
                entry.update(cmnd_dips(item["audio"], item["truth"]))
            result["groups"][group][name] = entry

    summary = {}
    for group, items in result["groups"].items():
        frames = sum(e["frames"] for e in items.values())
        summary[group] = {
            "cases": len(items),
            "frames": frames,
            "ok": sum(e["ok"] for e in items.values()),
            "octave_up": sum(e["octave_up"] for e in items.values()),
            "octave_down": sum(e["octave_down"] for e in items.values()),
            "other": sum(e["other"] for e in items.values()),
        }
        summary[group]["accuracy"] = round(summary[group]["ok"] / max(1, frames), 4)
    result["summary"] = summary
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
