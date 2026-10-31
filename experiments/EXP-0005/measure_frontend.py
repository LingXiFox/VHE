"""EXP-0005 measurements: resampler response, F0 accuracy, segmentation behaviour.

Synthetic signals only (see the experiment README): the corpus is not on this machine, so
nothing here can claim real-speech accuracy. Run from the repository root:

    .venv/bin/python experiments/EXP-0005/measure_frontend.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vhe.data.f0 import estimate_f0
from vhe.data.resample import resample
from vhe.data.segment import segment_by_energy

SR = 24000
OUT = Path(__file__).resolve().parent / "logs" / "frontend_report.json"


def tone(freq_hz: float, seconds: float = 1.0, sample_rate: int = SR,
         amplitude: float = 0.6) -> np.ndarray:
    times = np.arange(int(round(sample_rate * seconds))) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * times)).astype(np.float32)


def linear_resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    # Negative control: what a naive np.interp resampler does to the same signals.
    n_out = int(round(audio.size * dst / src))
    return np.interp(np.arange(n_out) / dst, np.arange(audio.size) / src, audio).astype(np.float32)


def bursts(freqs, *, burst_s=0.12, gap_s=0.10, lead_s=0.05) -> np.ndarray:
    parts = [np.zeros(int(round(lead_s * SR)), dtype=np.float32)]
    for freq in freqs:
        parts.append(tone(freq, burst_s))
        parts.append(np.zeros(int(round(gap_s * SR)), dtype=np.float32))
    return np.concatenate(parts)


def resampler_report() -> dict:
    report = {"length": {}, "passband": {}, "stopband_db": None}
    for src in (44100, 16000, 48000):
        audio = tone(1000.0, 0.5, src)
        out = resample(audio, src, SR)
        report["length"][f"{src}->{SR}"] = {
            "input": int(audio.size), "output": int(out.size),
            "expected": int(round(audio.size * SR / src)),
            "exact": int(out.size) == int(round(audio.size * SR / src)),
        }
    for freq in (500.0, 1000.0, 2000.0, 4000.0):
        out = resample(tone(freq, 0.5, 44100), 44100, SR)[1000:-1000]
        # RMS over whole periods: peak-picking is phase dependent (4 kHz at 24 kHz gives
        # only 6 samples per period, so max|y| reads up to 1.25 dB low even with a flat filter).
        period = int(round(SR / freq))
        whole = (out.size // period) * period
        rms = float(np.sqrt(np.mean(out[:whole] ** 2)))
        report["passband"][f"{freq:g}"] = round(
            float(20.0 * np.log10(rms / (0.6 / np.sqrt(2.0)))), 4)
    out = resample(tone(15000.0, 0.5, 44100), 44100, SR)[1000:-1000]
    report["stopband_db"] = round(float(20.0 * np.log10(max(np.max(np.abs(out)), 1e-12) / 0.6)), 2)

    control = {"passband_rms_db": {}, "stopband_db": None}
    for freq in (500.0, 1000.0, 2000.0, 4000.0):
        out = linear_resample(tone(freq, 0.5, 44100), 44100, SR)[1000:-1000]
        period = int(round(SR / freq))
        whole = (out.size // period) * period
        rms = float(np.sqrt(np.mean(out[:whole] ** 2)))
        control["passband_rms_db"][f"{freq:g}"] = round(
            float(20.0 * np.log10(rms / (0.6 / np.sqrt(2.0)))), 4)
    out = linear_resample(tone(15000.0, 0.5, 44100), 44100, SR)[1000:-1000]
    control["stopband_db"] = round(
        float(20.0 * np.log10(max(np.max(np.abs(out)), 1e-12) / 0.6)), 2)
    report["naive_linear_control"] = control
    return report


def f0_report() -> dict:
    steady = {}
    for freq in (100.0, 120.0, 220.0, 300.0, 400.0):
        track = estimate_f0(tone(freq), SR)
        median = float(np.median(track.voiced_f0)) if track.voiced_f0.size else 0.0
        steady[f"{freq:g}"] = {
            "voiced_ratio": round(track.voiced_ratio, 3),
            "median_hz": round(median, 2),
            "rel_error": round(abs(median - freq) / freq, 4),
        }
    times = np.arange(SR) / SR
    contour = np.linspace(100.0, 400.0, times.size)
    phase = 2.0 * np.pi * np.cumsum(contour) / SR
    track = estimate_f0((0.6 * np.sin(phase)).astype(np.float32), SR)
    frame_times, values = track.contour()
    voiced = values > 0
    expected = np.interp(frame_times[voiced], times, contour)
    rel = np.abs(values[voiced] - expected) / expected
    glide = {"voiced_ratio": round(track.voiced_ratio, 3),
             "mean_rel_error": round(float(rel.mean()), 4),
             "max_rel_error": round(float(rel.max()), 4),
             "start_hz": round(float(values[voiced][0]), 2),
             "end_hz": round(float(values[voiced][-1]), 2)}
    silent = estimate_f0(np.zeros(SR // 2, dtype=np.float32), SR)
    return {"steady": steady, "glide": glide,
            "silence_voiced_ratio": round(silent.voiced_ratio, 3)}


def segmentation_report() -> dict:
    audio = bursts([220.0, 260.0, 300.0, 240.0])
    four = segment_by_energy(audio, SR, f0_track=estimate_f0(audio, SR))
    click = np.concatenate([np.zeros(SR // 10, dtype=np.float32), tone(220.0, 0.15),
                            np.zeros(SR // 10, dtype=np.float32), tone(260.0, 0.01),
                            np.zeros(SR // 10, dtype=np.float32), tone(300.0, 0.15),
                            np.zeros(SR // 10, dtype=np.float32)])
    click_result = segment_by_energy(click, SR)
    silent = segment_by_energy(np.zeros(SR, dtype=np.float32), SR)
    return {
        "four_syllables": {
            "threshold": round(four.threshold, 4),
            "segments": [{"start_s": round(s.start_s, 3), "end_s": round(s.end_s, 3),
                          "voiced": s.voiced} for s in four.segments],
            "stats": four.stats(),
        },
        "click_10ms": {"segments": len(click_result.segments),
                       "codes": [i["code"] for i in click_result.issues]},
        "silence": {"segments": len(silent.segments),
                    "codes": [i["code"] for i in silent.issues],
                    "severities": [i["severity"] for i in silent.issues]},
    }


def main() -> None:
    report = {"resampler": resampler_report(), "f0": f0_report(),
              "segmentation": segmentation_report()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
