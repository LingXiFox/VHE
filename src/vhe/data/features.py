"""Mel-frequency cepstra for the P2 forced aligner (NumPy only, no scipy).

The aligner needs a frame-synchronous representation that survives the 44.1 kHz corpus
and the 16/24 kHz synthesizer. CMVN inside the utterance removes channel gain and the
speaker's average spectral tilt, which is what makes a *single* GMM-HMM usable across
speakers before any speaker adaptation exists.

Units: frames hop by HOP_S = 10 ms. Frame t covers [t*hop, t*hop + frame) in samples,
so a boundary reported at frame t is accurate to +- one hop. Everything is float32 on
output because the corpus loader stores float32 PCM.
"""

from __future__ import annotations

import numpy as np

#: All acoustic modelling happens at one rate; 16 kHz keeps the fricative band alive
#: up to 8 kHz without the cost of 44.1 kHz frames.
SAMPLE_RATE = 16000
FRAME_S = 0.025
HOP_S = 0.010
N_FFT = 512
N_MELS = 26
N_MFCC = 13
PREEMPHASIS = 0.97
LOG_FLOOR = 1e-10
#: Deltas of both orders triple the feature width: 13 -> 39.
N_DELTAS = 2


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                   n_mels: int = N_MELS, fmin: float = 20.0,
                   fmax: float | None = None) -> np.ndarray:
    """[n_mels, n_fft//2+1] HTK-style triangular filters, DC-normalised per band."""
    fmax = sample_rate / 2.0 if fmax is None else float(fmax)
    edges = mel_to_hz(np.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bank = np.zeros((n_mels, bins.size), dtype=np.float64)
    for band in range(n_mels):
        left, centre, right = edges[band], edges[band + 1], edges[band + 2]
        rising = (bins - left) / max(centre - left, 1e-9)
        falling = (right - bins) / max(right - centre, 1e-9)
        bank[band] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def frame_count(n_samples: int, frame: int, hop: int) -> int:
    if n_samples < frame:
        return 1
    return 1 + (n_samples - frame) // hop


def frame_signal(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """[T, frame] view of the signal; the tail is zero-padded, never dropped."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("frame_signal expects mono audio with shape [N]")
    if signal.size == 0:
        raise ValueError("frame_signal expects a non-empty signal")
    count = frame_count(signal.size, frame, hop)
    needed = frame + hop * (count - 1)
    if needed > signal.size:
        signal = np.pad(signal, (0, needed - signal.size))
    return np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop][:count]


def _dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """DCT-II, orthonormal: rows are the cepstral basis vectors."""
    n = np.arange(n_in, dtype=np.float64)
    k = np.arange(n_out, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / n_in * (n + 0.5) * k)
    scale = np.full((n_out, 1), np.sqrt(2.0 / n_in))
    scale[0] = np.sqrt(1.0 / n_in)
    return basis * scale


def log_mel_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
                        frame_s: float = FRAME_S, hop_s: float = HOP_S,
                        n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """[T, n_mels] log mel energies after pre-emphasis and a Hamming window."""
    frame = int(round(frame_s * sample_rate))
    hop = int(round(hop_s * sample_rate))
    signal = np.asarray(audio, dtype=np.float64)
    emphasised = np.append(signal[:1], signal[1:] - PREEMPHASIS * signal[:-1])
    frames = frame_signal(emphasised, frame, hop)
    window = np.hamming(frame)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    bank = mel_filterbank(sample_rate, n_fft, n_mels)
    return np.log(power @ bank.T + LOG_FLOOR)


def mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
         n_mfcc: int = N_MFCC, **kwargs) -> np.ndarray:
    """[T, n_mfcc] cepstra, c0 included (CMVN removes its mean anyway)."""
    spectrum = log_mel_spectrogram(audio, sample_rate, **kwargs)
    return spectrum @ _dct_matrix(spectrum.shape[-1], n_mfcc).T


def deltas(feature: np.ndarray, width: int = 2) -> np.ndarray:
    """Standard regression deltas, edge-replicated so the frame count never changes."""
    data = np.asarray(feature, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("deltas expects [T, D]")
    if width < 1:
        raise ValueError("width must be >= 1")
    padded = np.pad(data, ((width, width), (0, 0)), mode="edge")
    denominator = 2.0 * sum(step * step for step in range(1, width + 1))
    accumulator = np.zeros_like(data)
    for step in range(1, width + 1):
        accumulator += step * (padded[width + step: width + step + data.shape[0]]
                               - padded[width - step: width - step + data.shape[0]])
    return accumulator / denominator


def cmvn(feature: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    """Per-utterance mean/variance normalisation; the std floor keeps silent frames finite."""
    data = np.asarray(feature, dtype=np.float64)
    mean = data.mean(axis=0, keepdims=True)
    std = np.maximum(data.std(axis=0, keepdims=True), floor)
    return (data - mean) / std


def acoustic_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
                      n_deltas: int = N_DELTAS, normalise: bool = True) -> np.ndarray:
    """[T, 13*(1+n_deltas)] float32 features for the aligner.

    CMVN is applied to the static coefficients *before* the deltas are taken: normalising
    after differencing would rescale the velocity by the static variance, which differs
    per utterance.
    """
    static = mfcc(audio, sample_rate)
    if normalise:
        static = cmvn(static)
    blocks = [static]
    for _ in range(n_deltas):
        blocks.append(deltas(blocks[-1]))
    return np.concatenate(blocks, axis=1).astype(np.float32)
