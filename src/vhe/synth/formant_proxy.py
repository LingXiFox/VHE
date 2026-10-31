"""Harmonic + formant SOURCE-FILTER PROXY backend.

A deliberately mechanical baseline, NOT an articulatory/physical model and NOT a
neural vocoder. It renders vowels by summing harmonics of an f0 contour and
emphasising formant bands; consonants, tones and prosody are absent. Its purpose is
to be a reproducible `x_mechanical` source for VHE experiments, never to be
presented as human-sounding output.
"""

from __future__ import annotations

import numpy as np

from .contract import (
    PHONEME_INVENTORY,
    SynthesisRequest,
    SynthesisResult,
    supported_phonemes,
)


class FormantProxySynth:
    """Deterministic source-filter proxy: [request] -> float32 [N] waveform."""

    name = "formant_proxy"
    version = "0.1.0"
    supported_phonemes = supported_phonemes()

    def render(self, request: SynthesisRequest) -> SynthesisResult:
        request.validate()
        sample_rate = request.sample_rate
        template = request.template
        segments = request.phonemes.segments

        n_voiced = round(request.voiced_duration_s * sample_rate)
        n_lead = round(request.timing.lead_silence_s * sample_rate)
        n_tail = round(request.timing.tail_silence_s * sample_rate)
        time = np.arange(n_voiced, dtype=np.float64) / sample_rate

        # Segment centres drive a piecewise-linear formant trajectory; equal-length
        # segments reproduce the original single-vowel interpolation exactly.
        durations = np.asarray([s.duration_s for s in segments], dtype=np.float64)
        ends = np.cumsum(durations)
        centres = ends - durations / 2.0
        targets = np.asarray([PHONEME_INVENTORY[s.phoneme].as_tuple() for s in segments],
                             dtype=np.float64)
        formants = [
            np.interp(time, centres, targets[:, index]) * template.formant_scale
            for index in range(3)
        ]

        base_f0 = request.pitch.sample(n_voiced, sample_rate)
        vibrato = template.vibrato_cents * np.sin(2.0 * np.pi * template.vibrato_hz * time)
        f0 = base_f0 * np.power(2.0, vibrato / 1200.0)
        phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate

        signal = np.zeros(n_voiced, dtype=np.float64)
        # Harmonic limit uses the highest *base* f0 so aliasing cannot creep in when
        # the contour rises; with a constant curve this matches the legacy baseline.
        max_harmonic = int((sample_rate * 0.43) // (float(np.max(base_f0)) * 1.06))
        bandwidths = (110.0, 150.0, 210.0)
        weights = (1.0, 0.72, 0.38)
        for harmonic in range(1, max_harmonic + 1):
            frequency = harmonic * f0
            formant_gain = 0.008 + sum(
                weight * np.exp(-0.5 * ((frequency - centre) / bandwidth) ** 2)
                for centre, bandwidth, weight in zip(formants, bandwidths, weights)
            )
            amplitude = formant_gain * harmonic ** (-template.spectral_tilt)
            signal += amplitude * np.sin(harmonic * phase)

        rng = np.random.default_rng(request.seed)
        signal += template.breathiness * rng.standard_normal(n_voiced)
        fade = max(1, int(0.03 * sample_rate))
        envelope = np.ones(n_voiced)
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        signal *= envelope
        peak = float(np.max(np.abs(signal)))
        if peak <= 1e-10 or not np.isfinite(peak):
            raise RuntimeError("Invalid/silent synthesis output")
        voiced = (0.85 * signal / peak).astype(np.float32)

        audio = np.zeros(n_lead + n_voiced + n_tail, dtype=np.float32)
        audio[n_lead:n_lead + n_voiced] = voiced
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            backend_name=self.name,
            backend_version=self.version,
            request_fingerprint=request.fingerprint(),
        )
