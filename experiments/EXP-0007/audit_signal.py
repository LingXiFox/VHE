"""EXP-0007 audit probes: measure the seven suspected signal-front-end defects.

Run against a checkout's src tree:
    PYTHONPATH=<checkout>/src python experiments/EXP-0007/audit_signal.py --out logs/audit_before.json

Every probe is independent and reports numbers, not pass/fail, so the same script can be
run before and after the fixes and the two JSON files compared line by line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _envelope_jump(audio, f0, sr, limit_s=0.12):
    """Max envelope discontinuity, normalised by the envelope peak."""
    head = np.asarray(audio, dtype=np.float64)[: int(limit_s * sr)]
    window = max(int(round(sr / f0)), 5)
    hop = max(window // 8, 1)
    if head.size < window + hop:
        return None
    frames = [head[start:start + window] for start in range(0, head.size - window + 1, hop)]
    env = np.array([np.sqrt(np.mean(frame ** 2)) for frame in frames])
    peak = float(env.max())
    if peak <= 0:
        return None
    return round(float(np.max(np.abs(np.diff(env)))) / peak, 4)


def _synth_probes() -> dict:
    from vhe.synth import (PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve,
                           SynthesisRequest, Timing)
    from vhe.synth.registry import get_backend

    backend = get_backend()
    out = {"short_voiced": {}, "envelope_step": {}}

    def render(ph, f0, dur, tmpl=PRESETS["yang"], peak=1.0, sr=24000):
        request = SynthesisRequest(
            template=tmpl,
            phonemes=PhonemeSequence((PhonemeSegment(ph, dur),)),
            pitch=PitchCurve.constant(f0, dur),
            timing=Timing(),
            sample_rate=sr,
            seed=42,
        )
        return backend.render(request).audio

    # f0 = 200 Hz at 24 kHz is exactly 120 samples per period, so a rectangular
    # one-period window measures the envelope without phase-dependent ripple.
    for dur in (0.010, 0.020, 0.029, 0.030, 0.045, 0.061, 0.200):
        entry = {}
        try:
            audio = render("a", 200.0, dur)
            entry["ok"] = True
            entry["n"] = int(audio.size)
            entry["first"] = float(audio[0])
            entry["peak"] = float(np.max(np.abs(audio)))
            # Largest single-sample jump in the first 60 ms relative to the peak; a correct
            # 30 ms fade keeps consecutive differences near the waveform slope, a broken
            # envelope shows a step.
            # Envelope continuity: one-pitch-period Hann-windowed RMS with a period/8 hop,
            # so the estimate follows the fade without following the waveform. A correct
            # 30 ms fade moves the envelope by ~0.04 of full scale per hop; an overwritten
            # fade turns the envelope discontinuity into a jump near 0.5.
            entry["env_jump"] = _envelope_jump(audio, 200.0, 24000)
        except Exception as error:  # noqa: BLE001 - the point is to record the failure
            entry = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        out["short_voiced"][f"{dur:.3f}"] = entry

    # Per-syllable peak normalisation: two different templates must not end up equally loud
    # only because each render is peak-normalised.
    quiet = render("u", 180.0, 0.25, tmpl=PRESETS["yin"])
    loud = render("a", 220.0, 0.25, tmpl=PRESETS["yang"])
    out["peak_normalisation"] = {
        "yin_peak": float(np.max(np.abs(quiet))),
        "yang_peak": float(np.max(np.abs(loud))),
        "yin_rms": float(np.sqrt(np.mean(quiet.astype(np.float64) ** 2))),
        "yang_rms": float(np.sqrt(np.mean(loud.astype(np.float64) ** 2))),
    }

    # Spectral sanity: harmonic amplitude must fall with spectral_tilt and the formant
    # peak must move with formant_scale.
    def spectral_centroid(audio, sr=24000):
        spec = np.abs(np.fft.rfft(audio.astype(np.float64)))
        freqs = np.fft.rfftfreq(audio.size, 1 / sr)
        return float(np.sum(freqs * spec) / max(np.sum(spec), 1e-12))

    from vhe.synth import VoiceTemplate
    centroids = {}
    for tilt in (0.8, 1.3, 2.2):
        centroids[f"tilt_{tilt}"] = spectral_centroid(
            render("a", 220.0, 0.4, tmpl=VoiceTemplate(spectral_tilt=tilt)))
    for scale in (0.75, 1.35):
        centroids[f"scale_{scale}"] = spectral_centroid(
            render("a", 220.0, 0.4, tmpl=VoiceTemplate(formant_scale=scale)))
    out["spectral"] = centroids
    return out


def _f0_probes() -> dict:
    from vhe.data.f0 import estimate_f0

    sr = 24000
    out = {"pure_tones": [], "harmonic_stack": [], "silence": [], "edge_cases": {}}

    def tone(f0, fs, seconds=1.0):
        return np.sin(2 * np.pi * f0 * np.arange(int(fs * seconds)) / fs)

    for fs in (16000, 24000, 48000):
        for f0 in (65.0, 80.0, 100.0, 150.0, 220.0, 300.0, 440.0, 600.0, 880.0):
            track = estimate_f0(tone(f0, fs), fs)
            measured = float(np.median(track.voiced_f0)) if track.voiced_f0.size else 0.0
            out["pure_tones"].append({
                "sr": fs, "f0": f0, "measured": round(measured, 2),
                "error_pct": round((measured - f0) / f0 * 100.0, 2) if measured else None,
                "voiced": round(track.voiced_ratio, 3),
            })

    # Harmonic stack with a 1/h^1.3 tilt: the realistic x_mechanical case.
    for f0 in (100.0, 220.0, 390.0):
        signal = sum(np.sin(2 * np.pi * k * f0 * np.arange(sr) / sr) / k ** 1.3
                     for k in range(1, int(4000 / f0)))
        track = estimate_f0(signal, sr)
        measured = float(np.median(track.voiced_f0))
        out["harmonic_stack"].append({"f0": f0, "measured": round(measured, 2),
                                      "error_pct": round((measured - f0) / f0 * 100, 2),
                                      "voiced": round(track.voiced_ratio, 3)})

    # Absolute vs relative voicing gate: a -120 dB periodic residue after digital silence.
    for amp in (1e-6, 1e-3):
        signal = np.zeros(sr)
        signal[sr // 2:] = amp * np.sin(2 * np.pi * 220 * np.arange(sr // 2) / sr)
        track = estimate_f0(signal, sr)
        out["silence"].append({"amp": amp, "voiced": round(track.voiced_ratio, 3)})

    # Octave-down regression case (was 73.33 Hz for a 220 Hz tone before EXP-0005).
    track = estimate_f0(tone(220.0, sr), sr)
    out["edge_cases"]["220hz_at_24k"] = round(float(np.median(track.voiced_f0)), 2)

    # Missing fundamental: 200 Hz plus a 4x stronger 400 Hz component. The waveform is
    # genuinely ambiguous; record what the estimator says instead of asserting.
    signal = (np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
              + 4.0 * np.sin(2 * np.pi * 400 * np.arange(sr) / sr))
    track = estimate_f0(signal, sr)
    out["edge_cases"]["missing_fundamental_200"] = round(float(np.median(track.voiced_f0)), 2)

    # One-frame artefact at a pitch step (200 -> 300 Hz).
    phase = 2 * np.pi * np.cumsum(np.where(np.arange(sr) < sr // 2, 200.0, 300.0)) / sr
    track = estimate_f0(np.sin(phase), sr)
    times, values = track.contour()
    window = values[(times > 0.49) & (times < 0.52)]
    out["edge_cases"]["step_transition_frames"] = [round(float(v), 1) for v in window]
    return out


def _segment_probes() -> dict:
    from vhe.data.segment import segment_by_energy

    sr = 24000
    out = {"bursts": [], "gaps": [], "clicks": []}

    def bursts(onset, duration, amplitude=0.5, f0=200.0, fade=0.0, total=2.0):
        signal = np.zeros(int(total * sr))
        start = int(onset * sr)
        stop = start + int(duration * sr)
        body = amplitude * np.sin(2 * np.pi * f0 * np.arange(stop - start) / sr)
        if fade > 0:
            n = int(fade * sr)
            body[:n] *= np.linspace(0, 1, n)
        signal[start:stop] = body
        return signal

    for onset, duration, amplitude in ((0.37, 0.30, 0.5), (1.00, 0.15, 0.5),
                                       (0.50, 0.08, 0.05), (0.50, 0.30, 0.9)):
        seg = segment_by_energy(bursts(onset, duration, amplitude), sr)
        if not seg.segments:
            out["bursts"].append({"true_onset": onset, "true_duration": duration,
                                  "detected": False})
            continue
        got = seg.segments[0]
        out["bursts"].append({
            "true_onset": onset, "true_duration": duration, "detected": True,
            "onset": round(got.start_s, 4), "end": round(got.end_s, 4),
            "onset_error_ms": round((got.start_s - onset) * 1000, 1),
            "duration_error_ms": round((got.duration_s - duration) * 1000, 1),
        })

    for gap in (0.0, 0.02, 0.06):
        signal = np.zeros(sr)
        for start in (0.2, 0.4 + gap):
            index = int(start * sr)
            signal[index:index + int(0.18 * sr)] = 0.4 * np.sin(
                2 * np.pi * 200 * np.arange(int(0.18 * sr)) / sr)
        seg = segment_by_energy(signal, sr)
        out["gaps"].append({"gap_s": gap, "segments": len(seg.segments)})

    for length in (0.010, 0.020):
        signal = np.zeros(sr)
        index = int(0.5 * sr)
        signal[index:index + int(length * sr)] = 0.8
        seg = segment_by_energy(signal, sr)
        out["clicks"].append({"length_s": length, "segments": len(seg.segments),
                              "issues": [issue["code"] for issue in seg.issues]})
    return out


def _pitch_probes() -> dict:
    from vhe.synth import PitchCurve

    curve = PitchCurve(times_s=(0.0, 1.0), f0_hz=(100.0, 400.0))
    values = curve.sample(24000, 24000)
    midpoint = float(values[12000])
    return {
        "midpoint_hz": round(midpoint, 2),
        "midpoint_cents_vs_geometric_mean": round(1200 * np.log2(midpoint / 200.0), 1),
        "first": float(values[0]),
        "last": float(values[-1]),
    }


def _resample_probes() -> dict:
    from vhe.data.resample import resample

    out = {"passband_db": {}, "impulse": {}}

    def gain(freq, src, dst):
        signal = np.sin(2 * np.pi * freq * np.arange(src) / src)
        resampled = resample(signal, src, dst)
        middle = resampled[dst // 4: 3 * dst // 4]
        return float(20 * np.log10(max(np.sqrt(2 * np.mean(middle ** 2)), 1e-12)))

    for freq in (1000, 8000, 10000, 11000, 15000, 20000):
        out["passband_db"][f"{freq}"] = {
            "44100->24000": round(gain(freq, 44100, 24000), 3),
            "48000->24000": round(gain(freq, 48000, 24000), 3),
        }

    for src, dst in ((44100, 24000), (48000, 24000), (16000, 24000)):
        signal = np.zeros(src)
        signal[int(0.5 * src)] = 1.0
        resampled = resample(signal, src, dst)
        out["impulse"][f"{src}->{dst}"] = {
            "peak_index": int(np.argmax(resampled)),
            "expected_index": 0.5 * src * dst / src,
            "peak": round(float(resampled.max()), 4),
        }
    return out


def _pairing_probes() -> dict:
    from vhe.audio import write_wav
    from vhe.data.pairing import build_pair
    from vhe.synth import PRESETS

    sr = 24000
    from vhe.synth import PhonemeSegment, PhonemeSequence, PitchCurve, SynthesisRequest, Timing
    from vhe.synth.registry import get_backend

    backend = get_backend()

    def render(ph, f0, dur):
        return backend.render(SynthesisRequest(
            template=PRESETS["yang"],
            phonemes=PhonemeSequence((PhonemeSegment(ph, dur),)),
            pitch=PitchCurve.constant(f0, dur), timing=Timing(),
            sample_rate=sr, seed=42)).audio

    gains = (1.0, 0.5, 0.8)
    onsets = (0.10, 0.45, 0.80)
    parts = []
    for gain, onset in zip(gains, onsets):
        parts.append(np.zeros(int(0.10 * sr), dtype=np.float32))
        parts.append((render("aiu"[len(parts) // 2], 200.0 + 20 * len(parts), 0.25) * gain)
                     .astype(np.float32))
    parts.append(np.zeros(int(0.09 * sr), dtype=np.float32))
    real = np.concatenate(parts)
    path = Path("/tmp/vhe_exp0007_fixture.wav")
    if path.exists():
        path.unlink()
    write_wav(path, real, sr)

    class Spy:
        def __init__(self):
            self.fingerprints = []

        def render(self, request):
            result = backend.render(request)
            self.fingerprints.append(result.request_fingerprint)
            return result

    spy = Spy()
    pair = build_pair(path, ["a1", "i1", "u1"], template=PRESETS["yang"], backend=spy)
    composite = hashlib.sha256("\n".join(spy.fingerprints).encode()).hexdigest()
    out = {
        "status": pair.status,
        "n_real": int(pair.real.size) if pair.real is not None else None,
        "n_mechanical": int(pair.mechanical.size) if pair.mechanical is not None else None,
        "fingerprint_is_last_syllable": pair.request_fingerprint == (spy.fingerprints[-1]
                                                                   if spy.fingerprints else None),
        "fingerprint_is_composite": pair.request_fingerprint == composite,
        "renders": len(spy.fingerprints),
        "issues": [issue["code"] for issue in pair.issues],
        "gains_db": pair.gains_db,
        "mechanical_peak": (float(np.max(np.abs(pair.mechanical)))
                            if pair.mechanical is not None else None),
    }
    if pair.mechanical is not None:
        for index, (onset, gain) in enumerate(zip(onsets, gains)):
            start = int((onset - 0.02) * sr)
            stop = int((onset + 0.27) * sr)
            real_rms = float(np.sqrt(np.mean(pair.real[start:stop] ** 2)))
            mech_rms = float(np.sqrt(np.mean(pair.mechanical[start:stop] ** 2)))
            out[f"syllable_{index}_rms_ratio"] = round(mech_rms / max(real_rms, 1e-9), 3)
        active = np.abs(pair.real) > 1e-2
        out["mechanical_zero_where_real_active"] = round(
            float(np.mean(np.abs(pair.mechanical[active]) < 1e-4)), 4)
        out["mechanical_active_where_real_silent"] = round(
            float(np.mean(np.abs(pair.mechanical[~active]) > 1e-2)), 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {
        "src": str(Path(sys.path[0] if sys.path else ".")),
        "numpy": np.__version__,
        "synth": _synth_probes(),
        "f0": _f0_probes(),
        "segment": _segment_probes(),
        "pitch_curve": _pitch_probes(),
        "resample": _resample_probes(),
        "pairing": _pairing_probes(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
