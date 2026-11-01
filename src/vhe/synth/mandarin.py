"""Articulatory Mandarin backend: source-filter synthesis with real consonants.

Why a second backend instead of extending FormantProxySynth: the proxy is frozen as the
baseline of EXP-0005/0006 (its WAV bytes are pinned by BASELINE_SHA256) and it can only
say "a", "i", "u". Content-bearing Mandarin needs stops with closures and bursts,
fricatives with noise bands, affricates with both, nasal resonance, diphthong formant
glides and coarticulation between them. That is a different synthesizer, so it lives in a
different module with its own version and its own inventory.

Signal model (all numpy, no scipy; scipy is not a runtime dependency):

    voiced source   : harmonic additive synthesis, per-frame amplitudes taken from a
                      formant envelope evaluated at k*f0, so harmonics always follow f0
    noise source    : white noise filtered by a time-varying band-pass in the STFT domain
                      (overlap-add), plus short release bursts
    filter          : three formant resonances + (optional) nasal pole and anti-formant,
                      with coarticulation smoothing of the formant tracks
    timing          : closure / voice-onset-time per stop, frication per affricate

Mechanical by construction: perfectly periodic source, no jitter, no shimmer, no
amplitude modulation beyond the phoneme envelope, straight-line tone contours. What the
backend does model is phonetics, because VHE cannot learn to restore contrasts that were
never synthesised. "Mechanical" must mean "lacking human micro-variation", not "missing
consonants".
"""

from __future__ import annotations

import numpy as np

from .contract import (
    MAX_DURATION_S,
    SynthesisRequest,
    SynthesisResult,
)
from .phonology import MANDARIN_PHONEMES, MANDARIN_PHONEME_IDS, PhonemeSpec

# Analysis-frame hop for the articulatory tracks. 5 ms resolves a 6 ms burst position and
# a 12 ms voice-onset time without making the track arrays huge.
HOP_S = 0.005
# Coarticulation: neighbouring phonemes overlap by roughly one articulation gesture.
FORMANT_SMOOTH_S = 0.025
AMPLITUDE_SMOOTH_S = 0.008
# Highest harmonic as a fraction of the sample rate; below Nyquist on purpose.
BANDWIDTH_FRACTION = 0.45
MAX_HARMONICS = 250

FORMANT_BANDWIDTHS_HZ = (90.0, 110.0, 170.0)
FORMANT_WEIGHTS = (1.0, 0.45, 0.25)
FORMANT_FLOOR = 0.012
NASAL_POLE_HZ = 250.0
NASAL_POLE_BW_HZ = 120.0
NASAL_ZERO_HZ = 900.0
NASAL_ZERO_BW_HZ = 350.0
NASAL_ZERO_DEPTH = 0.55

# Loudness target for the voiced body, in RMS (-14 dBFS). Every utterance is scaled so
# that its voiced region sits here, which keeps the VHE input level independent of which
# phonemes happen to be present; without it a single loud fricative re-scaled the whole
# utterance and the vowel level wandered by 4 dB between contexts.
TARGET_VOICED_RMS = 0.20
PEAK_CEILING = 0.98


def _smooth(values: np.ndarray, window_frames: int) -> np.ndarray:
    """Hann-window moving average with edge padding (no boundary dip)."""
    window_frames = max(1, int(window_frames))
    if window_frames <= 1 or values.size < 2:
        return values
    kernel = np.hanning(window_frames + 2)[1:-1]
    kernel = kernel / kernel.sum()
    pad = window_frames // 2
    padded = np.pad(values, (pad, window_frames - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _hertz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


class MandarinFormantSynth:
    """Limited-inventory Mandarin source-filter synthesizer."""

    name = "mandarin_formant"
    version = "0.1.0"
    supported_phonemes: tuple[str, ...] = MANDARIN_PHONEME_IDS
    inventory = "mandarin"

    # -- public API ---------------------------------------------------------------

    def render(self, request: SynthesisRequest) -> SynthesisResult:
        request.validate()
        if request.phonemes.inventory != self.inventory:
            raise ValueError(
                f"{self.name} renders the {self.inventory!r} inventory, "
                f"got {request.phonemes.inventory!r}"
            )
        sample_rate = request.sample_rate
        hop = max(1, int(round(HOP_S * sample_rate)))
        total_samples = int(round(request.total_duration_s * sample_rate))
        lead_samples = int(round(request.timing.lead_silence_s * sample_rate))
        voiced_samples = int(round(request.voiced_duration_s * sample_rate))
        if voiced_samples < 1:
            raise ValueError("voiced region is empty")

        rng = np.random.default_rng(request.seed)
        n_frames = int(np.ceil(total_samples / hop)) + 2
        frame_times = np.arange(n_frames, dtype=np.float64) * hop / sample_rate

        tracks = self._build_tracks(request, frame_times, total_samples, lead_samples,
                                    voiced_samples, sample_rate, hop)
        audio, tone_rms = self._synthesise(request, tracks, total_samples, lead_samples,
                                           voiced_samples, sample_rate, hop)
        reference = tone_rms if tone_rms > 1e-4 else TARGET_VOICED_RMS
        audio = self._add_bursts(request, audio, lead_samples, sample_rate, rng, reference)
        audio = self._add_breath(request, audio, tracks, lead_samples, voiced_samples,
                                 sample_rate, rng, reference)
        audio = self._finish(audio, reference, voiced_samples, lead_samples, sample_rate)
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            backend_name=self.name,
            backend_version=self.version,
            request_fingerprint=request.fingerprint(),
        )

    # -- articulatory tracks -------------------------------------------------------

    def _build_tracks(self, request, frame_times, total_samples, lead_samples,
                      voiced_samples, sample_rate, hop) -> dict[str, np.ndarray]:
        n_frames = frame_times.size
        f1 = np.zeros(n_frames)
        f2 = np.zeros(n_frames)
        f3 = np.zeros(n_frames)
        voiced_amp = np.zeros(n_frames)
        noise_amp = np.zeros(n_frames)
        noise_lo = np.zeros(n_frames)
        noise_hi = np.zeros(n_frames)
        nasal = np.zeros(n_frames)

        start_s = lead_samples / sample_rate
        cursor_s = start_s
        for segment in request.phonemes.segments:
            spec = MANDARIN_PHONEMES[segment.phoneme]
            stop_s = cursor_s + segment.duration_s
            mask = (frame_times >= cursor_s) & (frame_times < stop_s)
            if not np.any(mask):
                cursor_s = stop_s
                continue
            local = (frame_times[mask] - cursor_s) / segment.duration_s
            f1[mask], f2[mask], f3[mask] = self._formants(spec, local)
            kind_voiced, kind_noise = self._source(spec, frame_times[mask] - cursor_s,
                                                   segment.duration_s)
            voiced_amp[mask] = kind_voiced
            noise_amp[mask] = kind_noise
            if spec.noise_band is not None:
                noise_lo[mask], noise_hi[mask] = spec.noise_band
            if spec.nasal_from is not None:
                if spec.nasal_from <= 0.0:
                    nasal[mask] = 1.0  # /m/ /n/ are nasal for their whole duration
                else:
                    # Coda nasals fade in over ~20% of the segment (a velar/alveolar
                    # closure takes about that long), rather than ramping over the rest.
                    nasal[mask] = np.clip((local - spec.nasal_from) / 0.20, 0.0, 1.0)
            cursor_s = stop_s

        # Coarticulation: smooth the articulatory targets, never the source flags, so a
        # closure stays a closure while the formants glide through it.
        formant_window = int(round(FORMANT_SMOOTH_S / HOP_S))
        amplitude_window = int(round(AMPLITUDE_SMOOTH_S / HOP_S))
        f1 = _smooth(f1, formant_window)
        f2 = _smooth(f2, formant_window)
        f3 = _smooth(f3, formant_window)
        nasal = _smooth(nasal, formant_window)
        voiced_amp = _smooth(voiced_amp, amplitude_window)
        noise_amp = _smooth(noise_amp, amplitude_window)
        noise_lo = _smooth(noise_lo, formant_window)
        noise_hi = _smooth(noise_hi, formant_window)

        scale = request.template.formant_scale
        return {
            "f1": f1 * scale,
            "f2": np.minimum(f2 * scale, BANDWIDTH_FRACTION * sample_rate),
            "f3": np.minimum(f3 * scale, BANDWIDTH_FRACTION * sample_rate),
            "voiced": voiced_amp,
            "noise": noise_amp,
            "noise_lo": np.maximum(noise_lo, 80.0),
            "noise_hi": np.minimum(noise_hi, BANDWIDTH_FRACTION * sample_rate),
            "nasal": nasal,
            "hop": np.asarray(hop),
            "lead": np.asarray(lead_samples),
            "voiced_samples": np.asarray(voiced_samples),
        }

    @staticmethod
    def _formants(spec: PhonemeSpec, local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(spec.targets, dtype=np.float64)
        fractions = points[:, 0]
        values = points[:, 1:]
        if values.shape[0] == 1:
            ones = np.ones_like(local)
            return values[0, 0] * ones, values[0, 1] * ones, values[0, 2] * ones
        return (
            np.interp(local, fractions, values[:, 0]),
            np.interp(local, fractions, values[:, 1]),
            np.interp(local, fractions, values[:, 2]),
        )

    @staticmethod
    def _source(spec: PhonemeSpec, t_rel: np.ndarray, duration_s: float
                ) -> tuple[np.ndarray, np.ndarray]:
        """Voiced and noise amplitude envelopes for one segment (0..1)."""
        voiced = np.zeros_like(t_rel)
        noise = np.zeros_like(t_rel)
        if spec.closure_s > 0.0:
            open_s = spec.closure_s
        else:
            open_s = 0.0
        if spec.burst is not None:
            open_s = max(open_s, spec.closure_s + spec.burst[2])
        vot_end = open_s + spec.vot_s

        if spec.kind in ("vowel", "diphthong"):
            voiced[:] = 1.0
        elif spec.kind in ("nasal", "lateral"):
            # Nasal murmur and lateral are a few dB below the vowel body; the nasal pole
            # in the filter adds some of it back, this keeps the balance from flipping.
            voiced[:] = 0.50 if spec.kind == "nasal" else 0.62
        elif spec.kind == "approximant":
            # A syllabic approximant ([ʅ], [ɿ]) runs both sources: the voiced body is the
            # syllable nucleus, the noise is the residual frication of the retroflex/dental
            # gesture. Voicing is full, not the nasal 0.5, because this segment carries the
            # syllable's level; the noise gain is small enough not to take over.
            voiced[:] = 1.0
            noise[:] = spec.noise_gain
        elif spec.kind == "fricative":
            noise[:] = spec.noise_gain
        elif spec.kind in ("stop", "affricate"):
            # Closure: silent for voiceless stops, weakly voiced for the voiced series.
            closed = t_rel < open_s
            voiced[closed] = 0.35 if spec.voiced else 0.0
            if spec.kind == "stop":
                aspiration = (t_rel >= open_s) & (t_rel < vot_end)
                noise[aspiration] = spec.noise_gain
                release = t_rel >= vot_end
                voiced[release] = 1.0
            else:
                frication = t_rel >= open_s
                noise[frication] = spec.noise_gain
                # The vowel after an affricate starts inside the segment's tail so the
                # transition does not fall exactly on the segment boundary.
                tail = t_rel >= max(open_s, duration_s - 0.35 * duration_s)
                voiced[tail] = 0.6
        else:  # pragma: no cover - Kind is a closed literal
            raise ValueError(f"unhandled phoneme kind {spec.kind!r}")
        return voiced, noise

    # -- sources -------------------------------------------------------------------

    def _synthesise(self, request, tracks, total_samples, lead_samples, voiced_samples,
                    sample_rate, hop) -> np.ndarray:
        f0_voiced = request.pitch.sample(voiced_samples, sample_rate)
        template = request.template
        if template.vibrato_cents > 0.0 and template.vibrato_hz > 0.0:
            time = np.arange(voiced_samples, dtype=np.float64) / sample_rate
            depth = template.vibrato_cents / 1200.0
            f0_voiced = f0_voiced * np.power(2.0, depth * np.sin(
                2.0 * np.pi * template.vibrato_hz * time))
        f0 = np.zeros(total_samples)
        f0[lead_samples:lead_samples + voiced_samples] = f0_voiced

        if not np.any(f0 > 0.0):
            return np.zeros(total_samples)

        phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
        frame_index = np.arange(tracks["f1"].size)
        sample_times = np.arange(total_samples, dtype=np.float64) / sample_rate
        frames_at_sample = sample_times / HOP_S

        f0_min = float(np.min(f0[f0 > 0.0]))
        k_max = int(min(MAX_HARMONICS, np.floor(BANDWIDTH_FRACTION * sample_rate / f0_min)))
        k_max = max(1, k_max)
        harmonics = np.arange(1, k_max + 1, dtype=np.float64)
        nyquist = BANDWIDTH_FRACTION * sample_rate

        # (frames, K) formant envelope; f0 is per frame for the envelope lookup.
        f0_frames = np.zeros(tracks["f1"].size)
        frame_times = frame_index * hop / sample_rate
        f0_frames[:] = np.interp(frame_times, sample_times, f0, left=0.0, right=0.0)
        envelope = self._envelope(harmonics, tracks, f0_frames)
        tilt = np.power(harmonics, -template.spectral_tilt)[None, :]
        # Per-frame harmonic RMS normalisation. Without it the absolute k^-tilt puts the
        # whole level at the mercy of where F1 happens to sit relative to the harmonics:
        # a nasal with F1=250 and f0=200 radiated 17 dB more than the /a/ that followed
        # it purely because its fundamental landed in the formant peak. The formant
        # envelope is supposed to shape timbre, the voiced track is supposed to set level,
        # so the envelope is normalised and the level track is applied afterwards.
        shaped = envelope * tilt
        norm = np.sqrt(np.mean(shaped ** 2, axis=1, keepdims=True))
        amp_frames = shaped / np.maximum(norm, 1e-9) * tracks["voiced"][:, None]

        tone = np.zeros(total_samples)
        for index in range(k_max):
            harmonic = index + 1
            amp = np.interp(frames_at_sample, frame_index, amp_frames[:, index],
                            left=0.0, right=0.0)
            mask = (harmonic * f0) < nyquist
            if not np.any(mask):
                continue
            tone += amp * mask * np.sin(harmonic * phase)

        voiced_region = tone[lead_samples:lead_samples + voiced_samples]
        tone_rms = float(np.sqrt(np.mean(voiced_region.astype(np.float64) ** 2))) if voiced_samples else 0.0
        noise = self._noise(tracks, total_samples, sample_rate, tone_rms, request.seed)
        return tone + noise, tone_rms

    @staticmethod
    def _envelope(harmonics: np.ndarray, tracks, f0_frames: np.ndarray) -> np.ndarray:
        """Formant envelope evaluated at k*f0 for every frame: shape (frames, K)."""
        freqs = f0_frames[:, None] * harmonics[None, :]
        envelope = np.full_like(freqs, FORMANT_FLOOR)
        for number, (centre, bandwidth, weight) in enumerate(
                zip((tracks["f1"], tracks["f2"], tracks["f3"]),
                    FORMANT_BANDWIDTHS_HZ, FORMANT_WEIGHTS)):
            envelope += weight * np.exp(-0.5 * ((freqs - centre[:, None]) / bandwidth) ** 2)
        nasal = tracks["nasal"][:, None]
        pole = np.exp(-0.5 * ((freqs - NASAL_POLE_HZ) / NASAL_POLE_BW_HZ) ** 2)
        zero = np.exp(-0.5 * ((freqs - NASAL_ZERO_HZ) / NASAL_ZERO_BW_HZ) ** 2)
        return envelope * (1.0 + nasal * (1.0 * pole - NASAL_ZERO_DEPTH * zero))

    def _noise(self, tracks, total_samples, sample_rate, tone_rms: float,
               seed: int) -> np.ndarray:
        # noise_gain is a level relative to the voiced body, not an absolute variance:
        # without this calibration a fricative at gain 0.55 is ~8 dB louder than the vowel
        # it precedes, and peak normalisation then buries the whole utterance.
        rng = np.random.default_rng(seed * 7919 + 13)
        white = rng.standard_normal(total_samples)
        n_fft = int(2 ** round(np.log2(max(64.0, 0.0213 * sample_rate))))
        hop = n_fft // 4
        spectrum, window = _stft(white, n_fft, hop)
        n_stft = spectrum.shape[0]
        stft_times = (np.arange(n_stft) * hop + n_fft // 2) / sample_rate
        frame_index = np.arange(tracks["f1"].size)
        frame_times = frame_index * HOP_S
        gain = np.interp(stft_times, frame_times, tracks["noise"], left=0.0, right=0.0)
        low = np.interp(stft_times, frame_times, tracks["noise_lo"], left=80.0, right=80.0)
        high = np.interp(stft_times, frame_times, tracks["noise_hi"], left=80.0, right=80.0)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)[None, :]
        # A band-pass, not a bump at "low": the earlier 0.5*(1+cos(...)) form peaked at the
        # lower edge and notched at the upper one, so /s/ stopped at 6.4 kHz instead of 10 kHz
        # and /sh/ lost everything above 2.7 kHz (EXP-0010 audit). Rise below "low", flat in
        # between, fall above "high", with the edge widths kept from the previous version.
        edge = np.maximum(0.6 * low, 150.0)[:, None]
        rise = 0.5 * (1.0 - np.cos(np.pi * np.clip((freqs - low[:, None]) / edge, 0.0, 1.0)))
        high_edge = np.maximum(0.35 * high, 200.0)[:, None]
        # The fall has to close before the 0.45 * sr harmonic ceiling: /s/ tops out at 10 kHz
        # and its 3.5 kHz-wide transition used to reach 13.5 kHz, so 10% of its energy sat
        # above the cutoff (EXP-0010). Clamping the stop band only narrows /s/; /sh/, /x/ and
        # /f/ already end below the ceiling.
        ceiling = 0.45 * sample_rate
        stop = np.minimum(high[:, None] + high_edge, ceiling)
        width = np.maximum(stop - high[:, None], 1.0)
        fall = 0.5 * (1.0 + np.cos(np.pi * np.clip(
            (freqs - high[:, None]) / width, 0.0, 1.0)))
        mask = rise * fall
        # A gentle +3 dB/octave rise keeps fricatives from sounding dull; the band edges
        # above still dominate, this is only a tilt inside the band.
        tilt = np.power(np.maximum(freqs, 200.0) / 1000.0, 0.5)
        shaped = spectrum * mask * tilt
        unit = _istft(shaped, n_fft, hop, total_samples)
        unit_rms = float(np.sqrt(np.mean(unit.astype(np.float64) ** 2)))
        reference = tone_rms if tone_rms > 1e-4 else 0.25
        scale = reference / unit_rms if unit_rms > 1e-9 else 0.0
        # Gain stays in the STFT domain so noise_gain still controls the relative level;
        # only the *unit* noise is calibrated against the voiced body.
        return _istft(shaped * (gain[:, None] * scale), n_fft, hop, total_samples)

    # -- events --------------------------------------------------------------------

    def _add_bursts(self, request, audio, lead_samples, sample_rate, rng,
                    reference: float) -> np.ndarray:
        cursor_s = lead_samples / sample_rate
        for segment in request.phonemes.segments:
            spec = MANDARIN_PHONEMES[segment.phoneme]
            if spec.burst is not None:
                centre, bandwidth, duration_s, gain = spec.burst
                start = int(round((cursor_s + spec.closure_s) * sample_rate))
                n_burst = max(2, int(round(duration_s * sample_rate)))
                burst = _band_noise(n_burst, centre, bandwidth, sample_rate, rng)
                window = np.hanning(n_burst + 2)[1:-1]
                # Burst gain is relative to the voiced body, not an absolute amplitude:
                # an absolute burst vanished once the vowel was normalised properly.
                burst = burst * window * gain * reference
                stop = min(audio.size, start + n_burst)
                if stop > start:
                    audio[start:stop] += burst[: stop - start]
            cursor_s += segment.duration_s
        return audio

    def _add_breath(self, request, audio, tracks, lead_samples, voiced_samples,
                    sample_rate, rng, reference: float) -> np.ndarray:
        breathiness = request.template.breathiness
        if breathiness <= 0.0:
            return audio
        n = audio.size
        frame_index = np.arange(tracks["voiced"].size)
        frame_times = frame_index * HOP_S
        times = np.arange(n, dtype=np.float64) / sample_rate
        envelope = np.interp(times, frame_times, tracks["voiced"], left=0.0, right=0.0)
        audio = audio + 4.0 * breathiness * reference * envelope * rng.standard_normal(n)
        return audio

    def _finish(self, audio, reference, voiced_samples, lead_samples,
                sample_rate) -> np.ndarray:
        audio = audio * (TARGET_VOICED_RMS / reference)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > PEAK_CEILING:
            audio = audio * (PEAK_CEILING / peak)
        fade = min(int(round(0.005 * sample_rate)), max(1, voiced_samples // 2))
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
            start = lead_samples
            stop = min(audio.size, start + fade)
            audio[start:stop] *= ramp[: stop - start]
            tail = min(audio.size, start + voiced_samples)
            audio[max(start, tail - fade):tail] *= ramp[: tail - max(start, tail - fade)][::-1]
        return np.asarray(audio, dtype=np.float32)


def _stft(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(n_fft + 2)[1:-1]
    pad = n_fft // 2
    padded = np.pad(x, (pad, pad))
    n_frames = 1 + (padded.size - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = padded[index] * window
    return np.fft.rfft(frames, axis=1), window


def _istft(spectrum: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(n_fft + 2)[1:-1]
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1) * window
    n_frames = frames.shape[0]
    out = np.zeros((n_frames - 1) * hop + n_fft)
    norm = np.zeros_like(out)
    for index in range(n_frames):
        start = index * hop
        out[start:start + n_fft] += frames[index]
        norm[start:start + n_fft] += window ** 2
    out = out / np.maximum(norm, 1e-8)
    pad = n_fft // 2
    out = out[pad:pad + length]
    if out.size < length:
        out = np.pad(out, (0, length - out.size))
    return out


def _band_noise(n: int, centre_hz: float, bandwidth_hz: float, sample_rate: int,
                rng: np.random.Generator) -> np.ndarray:
    """Gaussian-windowed noise band around centre_hz (FFT filtering, no recursion)."""
    noise = rng.standard_normal(n)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mask = np.exp(-0.5 * ((freqs - centre_hz) / max(bandwidth_hz, 1.0)) ** 2)
    mask += 0.15 * np.exp(-0.5 * ((freqs - 2.0 * centre_hz) / max(bandwidth_hz, 1.0)) ** 2)
    filtered = np.fft.irfft(spectrum * mask, n=n)
    peak = float(np.max(np.abs(filtered)))
    return filtered / peak if peak > 0.0 else filtered
