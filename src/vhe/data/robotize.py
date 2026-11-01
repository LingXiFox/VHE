"""Mechanical source by analysis-resynthesis (EXP-0029).

The formant synthesiser in vhe.synth renders from phonemes and a pitch track. Its
intelligibility is limited: the owner's listening check on 2026-11-01 reported
"unintelligible, with electronic noise", and the measured fricative place cues are
wrong (EXP-0010: /f/ centroid 6156 Hz sits above /sh/ at 5051 Hz). This module
builds a different, deliberately coarser mechanical source from the human
recording itself, which is allowed for pair construction only: the VHE model never
sees the reference, only the rendered x_mechanical.

What it does, in order:
  1. estimate F0 on the reference and build a pulse-train excitation in voiced
     frames, white noise in unvoiced frames;
  2. take the reference magnitude spectrum, flatten it onto a small number of mel
     bands, and impose that band envelope on the excitation's phase;
  3. overlap-add back to exactly the input length.

The result keeps timing, F0 and every band-level cue, so it stays intelligible,
while the excitation and the flattened envelope make it sound like a vocoder. It
is not a claim that this is a good mechanical voice: that is a listening question,
and the log-mel distance it produces is recorded as evidence only.
"""
from __future__ import annotations

import numpy as np

from .features import mel_filterbank

DEFAULT_N_FFT = 1024
DEFAULT_HOP_S = 0.010
DEFAULT_BANDS = 24
DEFAULT_F0_FLOOR = 65.0
DEFAULT_F0_CEILING = 500.0
EPS = 1e-8


def _frame_signal(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(audio) < frame:
        padded = np.pad(audio, (0, frame - len(audio)))
    else:
        padded = audio
    count = 1 + (len(padded) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return padded[index]


def _stft(audio: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray, int]:
    """STFT with n_fft samples of padding on both sides.

    The padding is not cosmetic: without it the first and last frames are covered
    by a single window, so dividing by the window square sum at the overlap-add
    stage amplifies the edges by more than an order of magnitude (measured peak 87
    for an input peaking at 4.6). With padding the normalisation is constant.
    """
    window = np.hanning(n_fft + 2)[1:-1]
    pad = n_fft
    padded = np.pad(audio, (pad, pad))
    frames = _frame_signal(padded, n_fft, hop)
    return np.fft.rfft(frames * window, axis=1), window, pad


def _istft(spectrum: np.ndarray, n_fft: int, hop: int, length: int,
           window: np.ndarray, pad: int) -> np.ndarray:
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1) * window
    out = np.zeros(hop * (len(frames) - 1) + n_fft)
    norm = np.zeros_like(out)
    for index, frame in enumerate(frames):
        start = index * hop
        out[start:start + n_fft] += frame
        norm[start:start + n_fft] += window ** 2
    out = out / np.maximum(norm, 1e-6)
    out = out[pad:pad + length]
    if len(out) < length:
        out = np.pad(out, (0, length - len(out)))
    return out


def _f0_track(audio: np.ndarray, sample_rate: int, *, hop_s: float = 0.010,
              fmin_hz: float = DEFAULT_F0_FLOOR,
              fmax_hz: float = DEFAULT_F0_CEILING) -> np.ndarray:
    """Per-sample F0 (0.0 where unvoiced) from the estimator in vhe.data.f0.

    Reusing the existing estimator keeps one implementation of pitch tracking in
    the project; the octave errors it shows on corpus speech are tolerable here
    because the excitation only needs a plausible period.
    """
    from .f0 import estimate_f0

    track = estimate_f0(audio, sample_rate, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
    times = track.times()
    values = np.array([frame.f0_hz for frame in track.frames], dtype=np.float64)
    if times.size == 0:
        return np.zeros(len(audio))
    sample_times = np.arange(len(audio), dtype=np.float64) / sample_rate
    f0 = np.interp(sample_times, times, values, left=0.0, right=0.0)

    # A frame the estimator marked unvoiced must not be bridged into voice by
    # interpolation, so hold the decision on the frame grid.
    voiced = values > 0.0
    frame_of_sample = np.clip((sample_times / hop_s).astype(int), 0, len(values) - 1)
    f0[~voiced[frame_of_sample]] = 0.0
    f0[f0 < fmin_hz] = 0.0
    return f0


def _excitation(f0: np.ndarray, sample_rate: int, rng: np.random.Generator,
                noise_gain: float = 0.0) -> np.ndarray:
    """Pulse train in voiced frames, white noise in unvoiced frames."""
    phase = 0.0
    pulses = np.zeros(len(f0))
    for index in range(len(f0)):
        frequency = f0[index]
        if frequency <= 0.0:
            continue
        phase += frequency / sample_rate
        if phase >= 1.0:
            phase -= 1.0
            pulses[index] = 1.0
    # A one-sample impulse has a flat spectrum, which the envelope then shapes.
    # noise_gain adds breathiness inside voiced frames, but it costs periodicity
    # (measured: 0.00 -> 0.358, 0.12 -> 0.190 autocorrelation peak against 0.462
    # for the reference), so the default keeps voiced frames purely periodic.
    noise = rng.standard_normal(len(f0)) * noise_gain
    voiced = f0 > 0.0
    return np.where(voiced, pulses + noise, rng.standard_normal(len(f0)))


def _flat_envelope(magnitude: np.ndarray, sample_rate: int, n_fft: int,
                   bands: int) -> np.ndarray:
    """Flatten the magnitude onto mel bands: every bin takes its band mean.

    The first version mapped band energies back through the filterbank weights,
    which silently shrank wide (high) bands: the rendered pairs kept 0.1 % of
    their energy above 6 kHz against 1.5 % in the reference (EXP-0029 run 1).
    Assigning each bin to its strongest band and using the band mean keeps the
    magnitude in every band, so fricative energy survives.
    """
    if bands <= 0:
        return magnitude
    bank = mel_filterbank(sample_rate, n_fft, bands)
    assignment = bank.argmax(axis=0)
    # Per-frame band mean, so the envelope still follows the input over time.
    frame_means = np.zeros((magnitude.shape[0], bands))
    for band in range(bands):
        mask = assignment == band
        if mask.any():
            frame_means[:, band] = magnitude[:, mask].mean(axis=1)
    return frame_means[:, assignment]


def robotize(audio: np.ndarray, sample_rate: int, *, n_fft: int = DEFAULT_N_FFT,
             hop_s: float = DEFAULT_HOP_S, bands: int = DEFAULT_BANDS,
             seed: int = 42, noise_gain: float = 0.0) -> np.ndarray:
    """Render a mechanical copy of audio with the same length and sample rate.

    Args:
        audio: 1-D float signal (the human reference; used for analysis only).
        sample_rate: sample rate of audio.
        n_fft: STFT size, rounded to a power of two.
        hop_s: STFT hop in seconds.
        bands: mel bands the magnitude envelope is flattened onto; 0 disables it.
        seed: seed of the excitation noise, so renders are reproducible.
        noise_gain: noise mixed into voiced frames, relative to the pulse train.

    Returns:
        Mechanical signal, same length as audio, peak below 1.0.
    """
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("robotize expects a 1-D signal")
    if signal.size == 0:
        return signal.astype(np.float32)

    n_fft = int(2 ** round(np.log2(max(64, n_fft))))
    hop = max(1, int(round(hop_s * sample_rate)))
    rng = np.random.default_rng(seed)

    f0 = _f0_track(signal, sample_rate)
    excitation = _excitation(f0, sample_rate, rng, noise_gain=noise_gain)

    reference_spec, window, pad = _stft(signal, n_fft, hop)
    excitation_spec, _, _ = _stft(excitation, n_fft, hop)
    envelope = _flat_envelope(np.abs(reference_spec), sample_rate, n_fft, bands)

    shaped = envelope * np.exp(1j * np.angle(excitation_spec))
    output = _istft(shaped, n_fft, hop, len(signal), window, pad)

    # Match loudness to the reference before clipping: an impulse train is far
    # peakier than speech, so peak normalisation alone would render it too quiet.
    source_rms = float(np.sqrt(np.mean(signal ** 2)))
    output_rms = float(np.sqrt(np.mean(output ** 2)))
    if output_rms > EPS and source_rms > EPS:
        output = output * (source_rms / output_rms)

    peak = float(np.max(np.abs(output)))
    if peak > 0.99:
        output = output * (0.99 / peak)
    return output.astype(np.float32)
