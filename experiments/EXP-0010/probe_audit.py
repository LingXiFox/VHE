"""EXP-0010: algorithm audit of the Mandarin backend.

Measures five things that a "sounds plausible" check cannot: out-of-band harmonic energy
(aliasing), the spectral shape of each excitation type, envelope discontinuities at phoneme
boundaries, formant-trajectory smoothness inside diphthongs, and level consistency of the same
nucleus across consonant contexts.

Run: PYTHONPATH=src .venv/bin/python experiments/EXP-0010/probe_audit.py \
        --out experiments/EXP-0010/logs/audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from vhe.synth import (PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve,
                       SynthesisRequest, Timing)
from vhe.synth.mandarin import MandarinFormantSynth

SR = 24000
BACKEND = MandarinFormantSynth()
FRAME = 120  # 5 ms
HOP = 120


def render(phonemes, f0=200.0, template="yang"):
    if isinstance(phonemes, str):
        phonemes = ((phonemes, 0.20),)
    total = float(sum(duration for _name, duration in phonemes))
    request = SynthesisRequest(
        template=PRESETS[template],
        phonemes=PhonemeSequence(tuple(PhonemeSegment(name, duration)
                                       for name, duration in phonemes), inventory="mandarin"),
        pitch=PitchCurve.constant(f0, total),
        timing=Timing(), sample_rate=SR)
    return BACKEND.render(request).audio


def band_fractions(audio, edges=(0.0, 500.0, 2000.0, 4000.0, 8000.0, 12000.0)):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / SR)
    total = float(spectrum.sum())
    return {f"{int(lo)}-{int(hi)}": float(spectrum[(freqs >= lo) & (freqs < hi)].sum()) / total
            for lo, hi in zip(edges[:-1], edges[1:])}


def above_fraction(audio, cutoff_hz):
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / SR)
    return float(spectrum[freqs > cutoff_hz].sum() / max(spectrum.sum(), 1e-20))


def frame_rms(audio):
    frames = [audio[start:start + FRAME] for start in range(0, audio.size - FRAME + 1, HOP)]
    return np.array([float(np.sqrt(np.mean(frame ** 2))) for frame in frames])


def envelope_jumps(audio, boundary_samples):
    rms = frame_rms(audio)
    scale = max(float(rms.max()), 1e-9)
    jumps = np.abs(np.diff(rms)) / scale
    boundary_frames = {int(round(sample / HOP)) for sample in boundary_samples}
    near = sorted({index + offset for index in boundary_frames for offset in (-1, 0, 1)
                   if 0 <= index + offset < jumps.size})
    interior = [index for index in range(jumps.size) if index not in set(near)]
    return {
        "boundary_max": float(jumps[near].max()) if near else 0.0,
        "interior_max": float(jumps[interior].max()) if interior else 0.0,
        "boundary_mean": float(jumps[near].mean()) if near else 0.0,
        "interior_mean": float(jumps[interior].mean()) if interior else 0.0,
        "frames": int(jumps.size),
    }


def track_peaks(audio, window=512, hop=120, count=3):
    track = []
    for start in range(0, audio.size - window + 1, hop):
        frame = audio[start:start + window] * np.hanning(window)
        spectrum = np.log(np.maximum(np.abs(np.fft.rfft(frame)), 1e-9))
        smooth = np.convolve(spectrum, np.ones(5) / 5.0, mode="same")
        freqs = np.fft.rfftfreq(window, 1.0 / SR)
        peaks = [(float(smooth[i]), float(freqs[i])) for i in range(2, smooth.size - 2)
                 if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]]
        peaks.sort(reverse=True)
        track.append(sorted(freq for _level, freq in peaks[:count]))
    return track


def max_formant_step(audio):
    """Two trackers, both reported: index-matched (naive) and nearest-frequency matched."""
    track = [peaks for peaks in track_peaks(audio) if len(peaks) >= 2]
    naive = []
    for previous, current in zip(track, track[1:]):
        n = min(len(previous), len(current))
        naive.append(max(abs(current[i] - previous[i]) for i in range(n)))
    matched = []
    for previous, current in zip(track, track[1:]):
        # Each peak follows the closest peak of the previous frame. A peak that appears or
        # disappears then contributes a small step instead of an index-shift artefact.
        matched.append(max(min(abs(value - other) for other in previous) for value in current))
    return {
        "naive_max_step_hz": float(max(naive)) if naive else 0.0,
        "naive_median_step_hz": float(np.median(naive)) if naive else 0.0,
        "matched_max_step_hz": float(max(matched)) if matched else 0.0,
        "matched_median_step_hz": float(np.median(matched)) if matched else 0.0,
    }


def spectral_flux(audio, window=1024, hop=120):
    """Cosine distance between consecutive L2-normalised magnitude spectra."""
    frames = [audio[start:start + window] * np.hanning(window)
              for start in range(0, audio.size - window + 1, hop)]
    spectra = np.array([np.abs(np.fft.rfft(frame)) for frame in frames])
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    flux = 1.0 - np.sum(spectra[1:] * spectra[:-1], axis=1)
    return {"max_flux": float(flux.max()), "median_flux": float(np.median(flux))}


def click_ratio(audio, boundary_samples, half_window_ms=2.0):
    """Largest sample-to-sample step near a boundary, relative to the signal's own p99.9 step."""
    diff = np.abs(np.diff(audio))
    reference = float(np.percentile(diff, 99.9))
    half = int(round(half_window_ms * 1e-3 * SR))
    near = []
    for sample in boundary_samples:
        lo = max(0, sample - half)
        hi = min(diff.size, sample + half)
        if hi > lo:
            near.append(float(diff[lo:hi].max()))
    worst = max(near) if near else 0.0
    return {"boundary_max_step": worst, "p99_9_step": reference,
            "ratio": worst / reference if reference > 0 else float("inf")}


def rms_of(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {"probe": "exp0010_mandarin_audit", "python": sys.version.split()[0],
              "platform": platform.platform(), "sample_rate": SR,
              "backend": {"name": BACKEND.name, "version": BACKEND.version}}

    # (a) aliasing: harmonic energy must stay below the 0.45 * sr ceiling.
    alias = {}
    for phoneme in ("a", "i", "u", "m", "s", "sh", "ai"):
        for f0 in (65.0, 200.0, 440.0, 880.0):
            audio = render(((phoneme, 0.20),), f0=f0)
            alias[f"{phoneme}@{f0:.0f}"] = above_fraction(audio, 0.45 * SR)
    report["aliasing"] = {"cutoff_hz": 0.45 * SR, "fractions": alias,
                          "max": float(max(alias.values()))}

    # (b) excitation spectra: band shares per phoneme.
    spectra = {}
    for phoneme in ("a", "i", "u", "m", "n", "l", "f", "s", "sh", "x", "b", "p", "k"):
        audio = render(((phoneme, 0.20),))
        spectra[phoneme] = band_fractions(audio)
    report["band_fractions"] = spectra
    report["sibilance_ratio_2_8k"] = {
        phoneme: spectra[phoneme]["2000-4000"] + spectra[phoneme]["4000-8000"]
        for phoneme in spectra}

    # (c) envelope jumps at phoneme boundaries.
    boundaries = {
        "ma": [0.06],
        "ba": [0.04],
        "sa": [0.10],
        "ka": [0.06],
        "ma_ba_na": [0.06, 0.20, 0.30, 0.44, 0.56],
        "ai": [0.12],
    }
    jumps = {}
    for name, points in boundaries.items():
        if name == "ma":
            phonemes = (("m", 0.06), ("a", 0.24))
        elif name == "ba":
            phonemes = (("b", 0.04), ("a", 0.26))
        elif name == "sa":
            phonemes = (("s", 0.10), ("a", 0.22))
        elif name == "ka":
            phonemes = (("k", 0.06), ("a", 0.26))
        elif name == "ai":
            phonemes = (("ai", 0.32),)
        else:
            phonemes = (("m", 0.06), ("a", 0.14), ("b", 0.04), ("a", 0.14),
                        ("n", 0.06), ("a", 0.14))
        audio = render(phonemes)
        jumps[name] = envelope_jumps(audio, [int(round(p * SR)) for p in points])
    report["envelope_jumps"] = jumps

    # (d) formant continuity inside diphthongs and across a hard splice (control).
    continuous = {}
    for phoneme in ("ai", "au", "ei", "ou"):
        audio = render(((phoneme, 0.32),))
        continuous[phoneme] = max_formant_step(audio)
    splice = np.concatenate((render((("a", 0.20),)), render((("i", 0.20),))))
    continuous["spliced_a_i_control"] = max_formant_step(splice)
    report["formant_steps"] = continuous

    flux = {phoneme: spectral_flux(render(((phoneme, 0.32),)))
            for phoneme in ("ai", "au", "ei", "ou")}
    flux["spliced_a_i_control"] = spectral_flux(splice)
    report["spectral_flux"] = flux

    clicks = {}
    for name, points in boundaries.items():
        if name == "ma":
            phonemes = (("m", 0.06), ("a", 0.24))
        elif name == "ba":
            phonemes = (("b", 0.04), ("a", 0.26))
        elif name == "sa":
            phonemes = (("s", 0.10), ("a", 0.22))
        elif name == "ka":
            phonemes = (("k", 0.06), ("a", 0.26))
        elif name == "ai":
            phonemes = (("ai", 0.32),)
        else:
            phonemes = (("m", 0.06), ("a", 0.14), ("b", 0.04), ("a", 0.14),
                        ("n", 0.06), ("a", 0.14))
        audio = render(phonemes)
        clicks[name] = click_ratio(audio, [int(round(p * SR)) for p in points])
    report["boundary_clicks"] = clicks

    # (e) level of the same nucleus across contexts.
    contexts = {"a": (("a", 0.24),), "ma": (("m", 0.06), ("a", 0.24)),
                "ba": (("b", 0.04), ("a", 0.26)), "sa": (("s", 0.10), ("a", 0.22)),
                "ai": (("ai", 0.24),), "an": (("an", 0.24),)}
    levels = {}
    for name, phonemes in contexts.items():
        audio = render(phonemes)
        total = float(sum(duration for _name, duration in phonemes))
        tail = audio[int(round(0.05 * SR)):int(round((total - 0.01) * SR))]
        levels[name] = {"rms_full": rms_of(audio), "rms_tail": rms_of(tail),
                        "peak": float(np.max(np.abs(audio))),
                        "dbfs_tail": 20.0 * np.log10(max(rms_of(tail), 1e-12))}
    spread = max(v["dbfs_tail"] for v in levels.values()) - min(
        v["dbfs_tail"] for v in levels.values())
    report["levels"] = {"contexts": levels, "tail_dbfs_spread": float(spread)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"aliasing_max": report["aliasing"]["max"],
                      "spectral_flux": flux,
                      "envelope_jumps": jumps,
                      "formant_steps": continuous,
                      "tail_dbfs_spread": report["levels"]["tail_dbfs_spread"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
