"""Frame-wise F0 tracking by YIN (NumPy only).

Used only to *build* paired data and diagnostics: VHE never sees F0 as an input. Frames
too quiet or too aperiodic are marked unvoiced instead of being given an invented pitch;
callers must decide what to do and report the ratio.

The estimator is the cumulative-mean-normalised difference function (CMND) of de Cheveigne
and Kawahara (2002): for lag tau,

    d(tau)  = sum_{j<W} (x[j] - x[j+tau])^2 = e0 + e1(tau) - 2 r(tau)
    d'(tau) = d(tau) * tau / sum_{k<=tau} d(k)

r(tau) is the raw autocorrelation (one FFT), e0/e1 are prefix sums of x^2, and W is a fixed
integration width so every lag averages over the same number of samples. A pure tone of
period T gives d(kT) = 0 exactly, so d'(T) = 0 and the *shortest* lag with d' below the
absolute threshold is the period. This removes two failure modes of the previous
unbiased-autocorrelation estimate: half-period locking at low F0 (65-80 Hz read as ~900 Hz)
and the shortest-strong-local-peak heuristic that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class F0Frame:
    time_s: float
    f0_hz: float  # 0.0 means unvoiced
    voicing: float  # 1 - CMND at the chosen lag, in [0, 1]


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


def _difference(x: np.ndarray, lag_min: int, lag_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lags, d(tau), d'(tau)) for lag_min..lag_max."""
    n = x.size
    lags = np.arange(lag_min, lag_max + 1)
    prefix = np.concatenate(([0.0], np.cumsum(x * x)))
    # Variable integration window: lag tau averages over n - tau terms on both sides, which
    # keeps e_head, e_tail and r(tau) mutually consistent (a fixed width would need a second
    # FFT to restrict r(tau) to the same terms). The CMND normalisation is what makes the
    # shrinking window harmless.
    energy_head = prefix[n - lags]
    energy_tail = prefix[n] - prefix[lags]
    spectrum = np.fft.rfft(x, 1 << (2 * n - 1).bit_length())
    corr = np.fft.irfft(spectrum * np.conj(spectrum), 1 << (2 * n - 1).bit_length())[:n]
    diff = np.maximum(energy_head + energy_tail - 2.0 * corr[lags], 0.0)
    running = np.cumsum(diff)
    cmnd = diff * lags / np.maximum(running, 1e-12)
    return lags, diff, cmnd


#: Absolute CMND margin by which the second dip must beat the first one before the estimator
#: accepts the longer period (EXP-0011; swept 0.008..0.08, see the experiment README).
OCTAVE_TOLERANCE = 0.02
#: Relative depth: the global minimum must also be at most this fraction of the first dip.
OCTAVE_DEPTH_RATIO = 0.5


def _estimate_frame(frame: np.ndarray, sample_rate: int, fmin_hz: float, fmax_hz: float,
                    threshold: float) -> tuple[float, float]:
    centered = frame - frame.mean()
    if float(np.dot(centered, centered)) <= 1e-12:
        return 0.0, 0.0
    lag_min = max(2, int(np.floor(sample_rate / fmax_hz)))
    lag_max = min(centered.size - 3, int(np.ceil(sample_rate / fmin_hz)))
    if lag_max <= lag_min or centered.size - lag_max < 16:
        return 0.0, 0.0  # too few samples left for even one period of the lowest F0
    # The cumulative mean must start at lag 1, not at lag_min: normalising by a sum that
    # already contains only large lags inflates CMND by an order of magnitude and hides the
    # period of high voices (880 Hz at 24 kHz was read as 440 Hz for exactly this reason).
    lags, diff, cmnd = _difference(centered, 1, lag_max)
    offset = max(0, lag_min - 1)
    search = cmnd[offset:]
    # YIN absolute threshold: the first dip below it is the period; a deeper later dip is a
    # multiple of the same period and must not win.
    below = np.flatnonzero(search < 0.15)
    if below.size:
        best = int(below[0])
        while best + 1 < search.size and search[best + 1] < search[best]:
            best += 1
    else:
        best = int(np.argmin(search))
    # The first dip is not enough on its own: a frame whose energy is dominated by an even
    # harmonic of f0 also matches at half the period, and that dip comes first. /a/ at 220 Hz
    # with F1 near 440 Hz was read as 440 Hz for exactly this reason (EXP-0011). The true
    # period matches the *whole* waveform, so its dip is strictly deeper than any sub-multiple
    # dip. Look for it in the only place it can be: between the chosen dip and twice that lag.
    #
    # The search is deliberately bounded. An unbounded "restart from the global minimum" rule
    # was tried first and made things worse: several lags reach ~0 for a periodic frame, so
    # floating point decides, and under 10 dB noise the global minimum drifts to twice the
    # period (EXP-0011 measured 87 octave-down frames on clean combs, 40 on noisy ones).
    # Both the relative and the absolute condition are needed; each alone re-labels the error
    # instead of fixing it.
    current_cmnd = float(search[best])
    if current_cmnd > 1e-9:
        # Lags, not array indices: search starts at lag_min, so doubling the index would only
        # reach half the period that has to be inspected.
        current_lag = int(lags[best + offset])
        upper_lag = min(int(lags[-1]), 2 * current_lag + 2)
        upper = int(np.searchsorted(lags, upper_lag, side="right")) - 1 - offset
        upper = min(upper, search.size - 1)
        window = search[best + 1:upper + 1]
        if window.size:
            local = best + 1 + int(np.argmin(window))
            candidate = float(search[local])
            deep_enough = candidate <= OCTAVE_DEPTH_RATIO * current_cmnd
            clearly_better = current_cmnd - candidate >= OCTAVE_TOLERANCE
            if deep_enough and clearly_better:
                best = local
    best += offset
    voicing = float(np.clip(1.0 - float(cmnd[best]), 0.0, 1.0))
    if voicing < threshold:
        return 0.0, voicing

    # Parabolic refinement on d(tau) around the chosen lag (sub-sample period).
    if 0 < best < diff.size - 1:
        left, center, right = diff[best - 1], diff[best], diff[best + 1]
        denom = left - 2.0 * center + right
        shift = 0.5 * (left - right) / denom if abs(denom) > 1e-12 else 0.0
        lag = float(lags[best]) + float(np.clip(shift, -0.5, 0.5))
    else:
        lag = float(lags[best])
    return float(sample_rate / lag), voicing


def estimate_f0(audio: np.ndarray, sample_rate: int, *, fmin_hz: float = 65.0,
                fmax_hz: float = 900.0, frame_ms: float = 40.0, hop_ms: float = 10.0,
                voicing_threshold: float = 0.45, silence_floor_db: float = -60.0,
                absolute_floor: float = 1e-3) -> F0Track:
    """Track F0 on a mono signal. Short signals yield a single (padded) frame.

    A frame is gated to unvoiced when its RMS is below either silence_floor_db relative to
    the loudest frame of the same signal, or absolute_floor (about -60 dBFS). Without a
    relative gate any digital-silence residue counts as fully voiced.
    """
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
    frames = [signal[start:start + window] for start in starts]
    levels = np.array([float(np.sqrt(np.mean(frame * frame))) for frame in frames])
    gate = max(absolute_floor,
               float(levels.max()) * 10.0 ** (silence_floor_db / 20.0)) if levels.size else 0.0
    track = F0Track(sample_rate=sample_rate, frame_s=window / sample_rate,
                    hop_s=hop / sample_rate, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
    for start, frame, level in zip(starts, frames, levels):
        if level < gate:
            f0, voicing = 0.0, 0.0
        else:
            f0, voicing = _estimate_frame(frame, sample_rate, fmin_hz, fmax_hz,
                                          voicing_threshold)
        track.frames.append(F0Frame(time_s=(start + window / 2.0) / sample_rate,
                                    f0_hz=f0, voicing=voicing))
    return track
