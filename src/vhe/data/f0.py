"""Frame-wise F0 tracking by normalised autocorrelation (NumPy only).

Used only to *build* paired data and diagnostics: VHE never sees F0 as an input. Frames
too quiet or too aperiodic are marked unvoiced instead of being given an invented pitch;
callers must decide what to do and report the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class F0Frame:
    time_s: float
    f0_hz: float  # 0.0 means unvoiced
    voicing: float  # peak normalised autocorrelation, in [0, 1]


@dataclass
class F0Track:
    frames: list[F0Frame] = field(default_factory=list)
    sample_rate: int = 0
    frame_s: float = 0.0
    hop_s: float = 0.0
    fmin_hz: float = 0.0
    fmax_hz: float = 0.0

    @property
    def voiced_ratio(self) -> float:
        if not self.frames:
            return 0.0
        return sum(1 for frame in self.frames if frame.f0_hz > 0) / len(self.frames)

    @property
    def voiced_f0(self) -> np.ndarray:
        return np.array([f.f0_hz for f in self.frames if f.f0_hz > 0], dtype=np.float64)

    def times(self) -> np.ndarray:
        return np.array([frame.time_s for frame in self.frames], dtype=np.float64)

    def contour(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, f0) with unvoiced frames as 0.0; gaps are the caller's problem."""
        return self.times(), np.array([frame.f0_hz for frame in self.frames], dtype=np.float64)

    def stats(self) -> dict:
        voiced = self.voiced_f0
        return {
            "frames": len(self.frames),
            "voiced_frames": int(voiced.size),
            "voiced_ratio": round(self.voiced_ratio, 4),
            "median_f0_hz": round(float(np.median(voiced)), 2) if voiced.size else None,
            "p05_f0_hz": round(float(np.percentile(voiced, 5)), 2) if voiced.size else None,
            "p95_f0_hz": round(float(np.percentile(voiced, 95)), 2) if voiced.size else None,
        }


def _estimate_frame(frame: np.ndarray, sample_rate: int, fmin_hz: float, fmax_hz: float,
                    threshold: float) -> tuple[float, float]:
    centered = frame - frame.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return 0.0, 0.0
    size = 1 << (2 * centered.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), size)[:centered.size]
    if autocorr[0] <= 0:
        return 0.0, 0.0
    autocorr = autocorr / autocorr[0]
    # Unbiased correction: the raw estimate decays with lag and biases towards short lags.
    lags = np.arange(centered.size)
    autocorr = autocorr * centered.size / (centered.size - lags)

    lag_min = max(2, int(np.floor(sample_rate / fmax_hz)))
    lag_max = min(centered.size - 2, int(np.ceil(sample_rate / fmin_hz)))
    if lag_max <= lag_min:
        return 0.0, 0.0
    window = autocorr[lag_min:lag_max + 1]
    best_max = float(window.max())
    if best_max <= 0.0:
        return 0.0, 0.0
    # Octave-down guard: the unbiased correction favours long lags, so a pure tone can win
    # at 3T instead of T. Pick the SHORTEST lag that is a local maximum within 85% of the
    # global peak; a true period always has a strong peak at its shortest multiple.
    padded = np.concatenate(([-np.inf], window, [-np.inf]))
    is_local = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:])
    strong = is_local & (window >= 0.85 * best_max)
    if bool(strong.any()):
        best = int(np.argmax(strong)) + lag_min
    else:
        best = int(np.argmax(window)) + lag_min
    peak = float(autocorr[best])
    voicing = float(np.clip(peak, 0.0, 1.0))
    if peak < threshold:
        return 0.0, voicing

    # Parabolic refinement around the peak (sub-sample lag).
    left, center, right = autocorr[best - 1], autocorr[best], autocorr[best + 1]
    denom = left - 2.0 * center + right
    shift = 0.5 * (left - right) / denom if abs(denom) > 1e-12 else 0.0
    lag = best + float(np.clip(shift, -0.5, 0.5))
    return float(sample_rate / lag), voicing


def estimate_f0(audio: np.ndarray, sample_rate: int, *, fmin_hz: float = 65.0,
                fmax_hz: float = 900.0, frame_ms: float = 40.0, hop_ms: float = 10.0,
                voicing_threshold: float = 0.45) -> F0Track:
    """Track F0 on a mono signal. Short signals yield a single (padded) frame."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("estimate_f0 expects mono audio with shape [N]")
    if signal.size == 0:
        raise ValueError("estimate_f0 expects a non-empty signal")
    if not 0.0 < fmin_hz < fmax_hz:
        raise ValueError(f"need 0 < fmin < fmax, got {fmin_hz}..{fmax_hz}")
    window = max(64, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if window > signal.size:
        pad = window - signal.size
        signal = np.pad(signal, (0, pad), mode="reflect")

    starts = list(range(0, signal.size - window + 1, hop))
    track = F0Track(sample_rate=sample_rate, frame_s=window / sample_rate,
                    hop_s=hop / sample_rate, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
    for start in starts:
        frame = signal[start:start + window]
        f0, voicing = _estimate_frame(frame, sample_rate, fmin_hz, fmax_hz, voicing_threshold)
        track.frames.append(F0Frame(time_s=(start + window / 2.0) / sample_rate,
                                    f0_hz=f0, voicing=voicing))
    return track
