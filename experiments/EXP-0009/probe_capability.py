"""EXP-0009: capability acceptance probe for the Mandarin physical backend.

Renders every advertised phoneme plus multi-phoneme sequences and measures the properties
the VHE data pipeline depends on. Writes a JSON report and audition WAVs into artifacts/
(both are git-ignored: no audio enters Git).

Run: PYTHONPATH=src .venv/bin/python experiments/EXP-0009/probe_capability.py \
        --out experiments/EXP-0009/logs/report.json --audio-dir experiments/EXP-0009/artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from vhe.audio import write_wav
from vhe.data.f0 import estimate_f0
from vhe.synth import (MANDARIN_PHONEME_IDS, PRESETS, PhonemeSegment, PhonemeSequence,
                       PitchCurve, SynthesisRequest, Timing)
from vhe.synth.mandarin import MandarinFormantSynth

SR = 24000
BACKEND = MandarinFormantSynth()
F0 = 200.0


def render(phonemes, f0=F0, template="yang", sample_rate=SR):
    if isinstance(phonemes, str):
        phonemes = ((phonemes, 0.15),)
    total = float(sum(duration for _name, duration in phonemes))
    request = SynthesisRequest(
        template=PRESETS[template],
        phonemes=PhonemeSequence(tuple(PhonemeSegment(name, duration)
                                       for name, duration in phonemes), inventory="mandarin"),
        pitch=PitchCurve.constant(f0, total),
        timing=Timing(), sample_rate=sample_rate)
    return BACKEND.render(request), total


def autocorrelation_peak(audio, sample_rate=SR):
    centre = audio[audio.size // 4: audio.size // 4 + int(0.100 * sample_rate)]
    centre = centre - centre.mean()
    if float(np.dot(centre, centre)) <= 0.0:
        return 0.0
    spectrum = np.fft.rfft(centre, 1 << (2 * centre.size - 1).bit_length())
    corr = np.fft.irfft(spectrum * np.conj(spectrum))[:centre.size]
    return float(np.max(corr[int(sample_rate / 900.0):int(sample_rate / 65.0)] / corr[0]))


def spectral_centroid(audio, sample_rate=SR):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    return float((spectrum * freqs).sum() / max(spectrum.sum(), 1e-12))


def formant_peaks(audio, sample_rate=SR, count=3):
    freqs = np.fft.rfftfreq(1024, 1.0 / sample_rate)
    spectrum = np.log(np.maximum(np.abs(np.fft.rfft(audio[:1024] * np.hanning(1024))), 1e-9))
    smooth = np.convolve(spectrum, np.ones(9) / 9.0, mode="same")
    peaks = [(float(smooth[i]), float(freqs[i])) for i in range(2, smooth.size - 2)
             if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    peaks.sort(reverse=True)
    return sorted(freq for _level, freq in peaks[:count])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--audio-dir", default=None)
    args = parser.parse_args()
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "probe": "exp0009_mandarin_capability",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backend": {"name": BACKEND.name, "version": BACKEND.version,
                    "inventory": BACKEND.inventory,
                    "phonemes": list(BACKEND.supported_phonemes)},
        "sample_rate": SR,
        "phonemes": {},
        "sequences": {},
        "f0": {},
        "templates": {},
        "tensor_handoff": {},
    }

    for phoneme in MANDARIN_PHONEME_IDS:
        result, total = render(((phoneme, 0.15),))
        audio = result.audio
        track = estimate_f0(audio, SR, fmax_hz=500.0)
        entry = {
            "samples": int(audio.size),
            "expected_samples": int(round(total * SR)),
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "peak": float(np.max(np.abs(audio))),
            "finite": bool(np.all(np.isfinite(audio))),
            "voiced_ratio": float(track.voiced_ratio),
            "autocorrelation_peak": autocorrelation_peak(audio),
            "spectral_centroid_hz": spectral_centroid(audio),
            "fingerprint": result.request_fingerprint,
        }
        report["phonemes"][phoneme] = entry
        if audio_dir is not None:
            write_wav(audio_dir / f"phoneme_{phoneme}.wav", audio, SR)

    sequences = {
        "ma": (("m", 0.06), ("a", 0.24)),
        "ba": (("b", 0.04), ("a", 0.26)),
        "sa": (("s", 0.10), ("a", 0.22)),
        "a_ai": (("a", 0.15), ("ai", 0.25)),
        "ma_ba_na": (("m", 0.06), ("a", 0.14), ("b", 0.04), ("a", 0.14),
                     ("n", 0.06), ("a", 0.14)),
        "ai_au_ei_ou": (("ai", 0.15), ("au", 0.15), ("ei", 0.15), ("ou", 0.15)),
    }
    for name, phonemes in sequences.items():
        result, total = render(phonemes, f0=200.0)
        audio = result.audio
        track = estimate_f0(audio, SR, fmax_hz=500.0)
        voiced = track.voiced_f0
        report["sequences"][name] = {
            "phonemes": [[p, d] for p, d in phonemes],
            "samples": int(audio.size),
            "expected_samples": int(round(total * SR)),
            "voiced_ratio": float(track.voiced_ratio),
            "f0_median_hz": float(np.median(voiced)) if voiced.size else None,
            "f0_error_fraction": (abs(float(np.median(voiced)) - 200.0) / 200.0
                                  if voiced.size else None),
            "peak": float(np.max(np.abs(audio))),
            "fingerprint": result.request_fingerprint,
        }
        if audio_dir is not None:
            write_wav(audio_dir / f"sequence_{name}.wav", audio, SR)

    for f0 in (65.0, 110.0, 200.0, 350.0, 440.0, 880.0):
        result, _total = render((("a", 0.30),), f0=f0)
        track = estimate_f0(result.audio, SR, fmax_hz=min(900.0, 1.5 * f0))
        median = float(np.median(track.voiced_f0)) if track.voiced_f0.size else None
        report["f0"][f"{f0:.0f}"] = {
            "median_hz": median,
            "error_fraction": abs(median - f0) / f0 if median else None,
            "voiced_ratio": float(track.voiced_ratio),
        }

    for template in ("yang", "yin"):
        result, _total = render((("a", 0.30),), template=template)
        report["templates"][template] = {
            "peaks_hz": formant_peaks(result.audio),
            "rms": float(np.sqrt(np.mean(result.audio ** 2))),
            "fingerprint": result.request_fingerprint,
        }
        if audio_dir is not None:
            write_wav(audio_dir / f"template_{template}.wav", result.audio, SR)

    try:
        import torch
        audio = render((("a", 0.20),))[0].audio
        tensor = torch.from_numpy(audio)
        entry = {"torch": torch.__version__, "mps_available": bool(torch.backends.mps.is_available())}
        entry["cpu_roundtrip_bit_equal"] = bool(np.array_equal(tensor.numpy(), audio))
        if torch.backends.mps.is_available():
            moved = tensor.to("mps").to("cpu").numpy()
            entry["mps_roundtrip_bit_equal"] = bool(np.array_equal(moved, audio))
            entry["mps_dtype"] = str(tensor.to("mps").dtype)
        report["tensor_handoff"] = entry
    except ImportError:  # pragma: no cover - torch is a hard dependency here
        report["tensor_handoff"] = {"error": "torch unavailable"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "phonemes"}, indent=2)[:2000])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
