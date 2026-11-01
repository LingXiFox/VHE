"""Render and measure one phoneme batch of EXP-0016.

For every unit in the batch this probe:
* builds the syllable through vhe.data.g2p (so the orthography and the inventory are the
  same path the pair builder uses), renders it with two voice templates,
* writes 16-bit PCM WAVs (small, synthetic, reproducible) into the stage artefacts,
* measures length, level, pitch (EXP-0015 tracker), formant glide direction, template
  dependence and coda nasal resonance,
* re-renders the stage-0 regression set and compares it byte for byte with
  logs/render_baseline.json, so growing the inventory cannot quietly retune the
  phonemes that already existed.

    python experiments/EXP-0016/probe_phonemes.py --stage 1 --units e ian ing iou \
        --json experiments/EXP-0016/logs/phonemes_stage1.json \
        --wav-dir experiments/EXP-0016/artifacts/stage1
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vhe.data.g2p import syllable_plan  # noqa: E402
from vhe.data.pitch import track_pitch  # noqa: E402
from vhe.synth import (PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve,  # noqa: E402
                       SynthesisRequest, Timing, phonology)
from vhe.synth.mandarin import MandarinFormantSynth  # noqa: E402

SR = 24000
F0 = 200.0
SYLLABLE_S = 0.30
#: Carrier onset per final, chosen so the rendered string is a real Mandarin syllable.
CARRIERS = {"e": "", "ian": "d", "ing": "b", "iou": "l", "uo": "d", "ong": "d",
            "en": "b", "apical_retroflex": "sh", "apical_dental": "z", "zh": "zh",
            "h": "h", "uei": "d", "iang": "l", "in": "b", "v": "n", "uai": "g"}
BACKEND = MandarinFormantSynth()

#: Stage-0 regression set, byte-identical to the renderer that produced
#: artifacts/render_baseline.json (recorded before the first inventory change).
REGRESSION_CASES = (
    ("ma1", 200.0, (("m", 0.070), ("a", 0.230)), "yang"),
    ("sha1", 200.0, (("sh", 0.080), ("a", 0.220)), "yang"),
    ("ban3", 180.0, (("b", 0.052), ("an", 0.248)), "yin"),
    ("dei3", 220.0, (("d", 0.050), ("ei", 0.250)), "yang"),
    ("gou3", 160.0, (("g", 0.052), ("ou", 0.248)), "yin"),
    ("bi1", 240.0, (("b", 0.050), ("i", 0.250)), "yang"),
    ("dan1", 195.0, (("d", 0.050), ("an", 0.200)), "yang"),
    ("ai4", 210.0, (("a", 0.120), ("i", 0.130)), "yin"),
)


def render(phonemes, *, f0=F0, template="yang", sr=SR):
    total = float(sum(duration for _name, duration in phonemes))
    request = SynthesisRequest(
        template=PRESETS[template],
        phonemes=PhonemeSequence(tuple(PhonemeSegment(name, duration)
                                       for name, duration in phonemes), inventory="mandarin"),
        pitch=PitchCurve.constant(f0, total), timing=Timing(0.0, 0.0), sample_rate=sr)
    return BACKEND.render(request).audio


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> str:
    pcm = np.clip(audio, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(data)
    return hashlib.sha256(data).hexdigest()


def band_energy(window: np.ndarray, low: float, high: float, sr: int = SR) -> float:
    n = 1 << max(0, window.size - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size), n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


def harmonic_peak(window: np.ndarray, low: float, high: float, f0: float, sr: int = SR) -> float:
    n = 1 << max(0, window.size - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size), n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    best_amp, best_hz = -1.0, 0.0
    for k in range(int(math.ceil(low / f0)), int(high / f0) + 1):
        index = int(np.searchsorted(freqs, k * f0))
        amp = float(np.max(spectrum[max(0, index - 1):index + 2]))
        if amp > best_amp:
            best_amp, best_hz = amp, k * f0
    return best_hz


def analyse_syllable(onset: str, final: str, template: str) -> dict:
    plan, reason = syllable_plan(onset, final, SYLLABLE_S)
    if reason is not None:
        return {"error": reason}
    audio = render(plan, template=template)
    track = track_pitch(audio, SR)
    voiced = track.voiced_f0
    onset_s = plan[0][1] if len(plan) > 1 else 0.0
    nucleus_start = onset_s
    return {
        "syllable": f"{onset}{final}",
        "plan_s": [[name, round(seconds, 4)] for name, seconds in plan],
        "samples": int(audio.size),
        "duration_s": round(audio.size / SR, 4),
        "rms": round(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 5),
        "peak": round(float(np.max(np.abs(audio))), 4),
        "f0_median_hz": round(float(np.median(voiced)) if voiced.size else 0.0, 2),
        "voiced_ratio": round(track.voiced_ratio, 3),
        "rejections": track.stats()["reasons"],
        "nucleus_start_s": round(nucleus_start, 4),
    }


def analyse_glide(unit: str) -> dict:
    """Formant trajectory of the unit on its own, probed at f0=100 Hz."""
    audio = render(((unit, SYLLABLE_S),), f0=100.0)
    early = audio[int(0.055 * SR):int(0.095 * SR)]
    late = audio[int(0.225 * SR):int(0.265 * SR)]
    return {
        "f1_early_hz": harmonic_peak(early, 200.0, 900.0, 100.0),
        "f1_late_hz": harmonic_peak(late, 200.0, 900.0, 100.0),
        "f2_early_hz": harmonic_peak(early, 900.0, 3000.0, 100.0),
        "f2_late_hz": harmonic_peak(late, 900.0, 3000.0, 100.0),
        "low_share_early": round(band_energy(early, 200.0, 320.0)
                                 / max(band_energy(early, 300.0, 3000.0), 1e-12), 5),
        "low_share_late": round(band_energy(late, 200.0, 320.0)
                                / max(band_energy(late, 300.0, 3000.0), 1e-12), 5),
    }


def analyse_nasal(unit: str) -> dict:
    spec = phonology.MANDARIN_PHONEMES[unit]
    if spec.nasal_from is None:
        return {}
    nasal = render(((unit, SYLLABLE_S),))
    phonology.MANDARIN_PHONEMES[unit] = replace(spec, nasal_from=None)
    try:
        oral = render(((unit, SYLLABLE_S),))
    finally:
        phonology.MANDARIN_PHONEMES[unit] = spec

    def share(audio: np.ndarray, start: float, stop: float) -> float:
        window = audio[int(start * SR):int(stop * SR)]
        return band_energy(window, 200.0, 320.0) / max(band_energy(window, 300.0, 3000.0), 1e-12)

    return {"nasal_from": spec.nasal_from,
            "coda_low_share_nasal": round(share(nasal, 0.225, 0.265), 5),
            "coda_low_share_oral": round(share(oral, 0.225, 0.265), 5),
            "head_low_share_nasal": round(share(nasal, 0.055, 0.095), 5),
            "head_low_share_oral": round(share(oral, 0.055, 0.095), 5)}


def regression(baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    out = {"baseline_inventory_size": baseline["inventory_size"],
           "current_inventory_size": len(BACKEND.supported_phonemes),
           "cases": {}}
    for name, f0, phonemes, template in REGRESSION_CASES:
        row = baseline["renders"][name]
        audio = render(phonemes, f0=f0, template=template)
        digest = hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()
        out["cases"][name] = {"same_bytes": digest == row["sha256"],
                              "same_samples": int(audio.size) == row["samples"],
                              "sha256": digest,
                              "expected_sha256": row["sha256"]}
    out["all_same"] = all(row["same_bytes"] and row["same_samples"]
                          for row in out["cases"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int)
    parser.add_argument("--units", required=True, nargs="+")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--wav-dir", required=True, type=Path)
    # The baseline lives under logs/ because artifacts/ is git-ignored (audio only): the
    # regression check has to work from a fresh clone.
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "experiments/EXP-0016/logs/render_baseline.json")
    args = parser.parse_args(argv)
    args.json = args.json.resolve()
    args.wav_dir = args.wav_dir.resolve()
    args.baseline = args.baseline.resolve()
    if args.json.exists():
        parser.error(f"refusing to overwrite an existing report: {args.json}")

    report = {"stage": args.stage, "units": {}, "regression": regression(args.baseline)}
    for unit in args.units:
        onset = CARRIERS.get(unit, "")
        plan, reason = syllable_plan(onset, unit, SYLLABLE_S)
        if reason is not None:
            raise SystemExit(f"{onset}{unit}: {reason}")
        # "l+iou" rather than "liou": the pinyin spelling is liu, so a concatenation here
        # would look like orthography and mislead the next reader.
        entry = {"carrier": onset,
                 "onset_plus_final": f"{onset}+{unit}",
                 "phonemes": [name for name, _seconds in plan]}
        for template in ("yang", "yin"):
            entry[template] = analyse_syllable(onset, unit, template)
            audio = render(plan, template=template)
            path = args.wav_dir / f"{onset or 'zero'}{unit}_{template}.wav"
            entry[template]["wav"] = str(path.relative_to(ROOT))
            entry[template]["wav_sha256"] = write_wav(path, audio)
        entry["glide"] = analyse_glide(unit) if unit not in ("zh", "h") else {}
        entry["nasal"] = analyse_nasal(unit)
        report["units"][unit] = entry
        glide = entry["glide"]
        print(f"{entry['onset_plus_final']:<10} f2 {glide.get('f2_early_hz')}->{glide.get('f2_late_hz')} Hz"
              f"  f1 {glide.get('f1_early_hz')}->{glide.get('f1_late_hz')} Hz"
              f"  f0 {entry['yang'].get('f0_median_hz')} Hz"
              f"  voiced {entry['yang'].get('voiced_ratio')}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"baseline renders unchanged: {report['regression']['all_same']}")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
