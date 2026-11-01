"""Energy-valley alignment of a known syllable count (P1.5, experimental).

The pairing pipeline needs one boundary per transcript syllable, but read speech rarely has
a pause between syllables and the backend has no silence phoneme. This module places N-1
boundaries with a global dynamic program over frame energy: every syllable must last between
min_duration_s and max_duration_s, and boundaries are pushed towards the deepest energy
valleys relative to their local peak.

It is **not** forced alignment. It has no acoustic model, no phonetic context and no
knowledge of where a stop closure ends, so a boundary can land in the middle of a nasal coda.
Every internal boundary therefore reports how deep its valley was in dB and a confidence in
[0, 1]; the pairing record refuses to call a pair training-ready when those confidences are
low. Manual boundaries can be supplied instead (see vhe.data.pairing.build_pair), which is
the only way to get phoneme-level truth without an external aligner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .f0 import F0Track
from .segment import frame_rms, segmentation_threshold

DEFAULT_MIN_DURATION_S = 0.08
DEFAULT_MAX_DURATION_S = 1.20
DEFAULT_DIP_WINDOW_S = 0.15
DEFAULT_CONFIDENT_DIP_DB = 12.0
#: How strongly the cost prefers a position where energy is *rising*. Without it the DP
#: parks boundaries in the middle of a long stop closure, where the dip is deepest, and the
#: syllable window can end up containing no voiced frame at all.
DEFAULT_RISE_WEIGHT = 0.5
DEFAULT_RISE_WINDOW_S = 0.04


@dataclass(frozen=True)
class Boundary:
    """One syllable window; dip_db/confidence describe its *start* boundary."""

    start_s: float
    end_s: float
    dip_db: Optional[float] = None
    confidence: Optional[float] = None
    voiced_ratio: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict:
        return {"start_s": round(self.start_s, 4), "end_s": round(self.end_s, 4),
                "duration_s": round(self.duration_s, 4),
                "dip_db": None if self.dip_db is None else round(self.dip_db, 2),
                "confidence": None if self.confidence is None else round(self.confidence, 3),
                "voiced_ratio": None if self.voiced_ratio is None
                else round(self.voiced_ratio, 3)}


@dataclass
class Alignment:
    boundaries: list[Boundary] = field(default_factory=list)
    method: str = "energy_dp"
    issues: list[dict] = field(default_factory=list)
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    settings: dict = field(default_factory=dict)

    def add_issue(self, code: str, detail: str, severity: str = "warning") -> None:
        self.issues.append({"code": code, "detail": detail, "severity": severity})

    def stats(self) -> dict:
        internal = [item for item in self.boundaries[1:] if item.confidence is not None]
        confidences = [float(item.confidence) for item in internal]
        durations = [item.duration_s for item in self.boundaries]
        return {
            "method": self.method,
            "syllables": len(self.boundaries),
            "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
            "min_confidence": round(float(np.min(confidences)), 3) if confidences else None,
            "low_confidence_boundaries": sum(1 for value in confidences if value < 0.35),
            "mean_duration_s": round(float(np.mean(durations)), 3) if durations else None,
            "min_duration_s": round(float(np.min(durations)), 3) if durations else None,
            "max_duration_s": round(float(np.max(durations)), 3) if durations else None,
            "issues": [[issue["code"], issue["detail"]] for issue in self.issues],
            "settings": dict(self.settings),
        }


def _local_max(values: np.ndarray, half: int) -> np.ndarray:
    padded = np.pad(values, (half, half), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return view.max(axis=1)


def _mean_after(values: np.ndarray, half: int) -> np.ndarray:
    padded = np.pad(values, (0, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, half)
    return windows[1:].mean(axis=1)


def _mean_before(values: np.ndarray, half: int) -> np.ndarray:
    padded = np.pad(values, (half, 0), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, half)
    return windows[: values.size].mean(axis=1)


def _voiced_ratio(f0_track: Optional[F0Track], start_s: float, end_s: float) -> Optional[float]:
    if f0_track is None:
        return None
    times, f0 = f0_track.contour()
    inside = (times >= start_s) & (times < end_s)
    if not bool(np.any(inside)):
        return None
    return float(np.mean(f0[inside] > 0))


def align_syllables(audio: np.ndarray, sample_rate: int, count: int, *,
                    frame_ms: float = 25.0, hop_ms: float = 10.0,
                    min_duration_s: float = DEFAULT_MIN_DURATION_S,
                    max_duration_s: float = DEFAULT_MAX_DURATION_S,
                    dip_window_s: float = DEFAULT_DIP_WINDOW_S,
                    confident_dip_db: float = DEFAULT_CONFIDENT_DIP_DB,
                    rise_weight: float = DEFAULT_RISE_WEIGHT,
                    rise_window_s: float = DEFAULT_RISE_WINDOW_S,
                    f0_track: Optional[F0Track] = None) -> Alignment:
    """Place count syllable windows inside the speech region of one recording.

    Deterministic: the same signal and parameters always produce the same boundaries.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if min_duration_s <= 0.0 or max_duration_s < min_duration_s:
        raise ValueError("require 0 < min_duration_s <= max_duration_s")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("align_syllables expects a non-empty mono signal")

    result = Alignment(method="energy_dp", min_duration_s=min_duration_s,
                       max_duration_s=max_duration_s)
    rms, hop_s = frame_rms(signal, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    if rms.size < 3:
        result.add_issue("alignment_signal_too_short",
                         f"{signal.size / sample_rate:.3f}s is shorter than three frames",
                         "error")
        return result
    smooth = np.convolve(rms, np.ones(3) / 3.0, mode="same")
    threshold = segmentation_threshold(smooth)
    if threshold <= 0.0:
        result.add_issue("alignment_no_speech", "signal is silent; nothing to align", "error")
        return result
    active = np.flatnonzero(smooth > threshold)
    if active.size == 0:
        result.add_issue("alignment_no_speech", "no frame exceeded the energy threshold",
                         "error")
        return result
    region_start = int(active[0])
    region_stop = int(active[-1]) + 1
    n = region_stop - region_start
    if n < 3:
        result.add_issue("alignment_signal_too_short",
                         f"speech region is {n * hop_s:.3f}s long", "error")
        return result
    min_frames = max(1, int(round(min_duration_s / hop_s)))
    max_frames = max(min_frames, int(round(max_duration_s / hop_s)))
    if n < count * min_frames:
        result.add_issue(
            "alignment_region_too_short",
            f"{n * hop_s:.2f}s of speech cannot hold {count} syllables of at least "
            f"{min_duration_s:.2f}s", "error")
        return result

    if rise_weight < 0.0:
        raise ValueError("rise_weight must be >= 0")
    half = max(1, int(round(dip_window_s / hop_s)))
    local_peak = _local_max(smooth[region_start:region_stop], half)
    energy_ratio = smooth[region_start:region_stop] / np.maximum(local_peak, 1e-12)
    rise_half = max(1, int(round(rise_window_s / hop_s)))
    rise = (_mean_after(smooth, rise_half) - _mean_before(smooth, rise_half))
    rise_ratio = np.clip(rise[region_start:region_stop]
                         / np.maximum(local_peak, 1e-12), -1.0, 1.0)
    cost = energy_ratio - rise_weight * rise_ratio

    def _ratio(start_s: float, end_s: float) -> Optional[float]:
        return _voiced_ratio(f0_track, start_s, end_s)

    if count == 1:
        start_s = region_start * hop_s
        end_s = min(signal.size / sample_rate, region_stop * hop_s)
        result.boundaries = [Boundary(start_s, end_s, None, None, _ratio(start_s, end_s))]
        return result

    infinity = float("inf")
    dp = np.full((count - 1, n), infinity)
    back = np.full((count - 1, n), -1, dtype=int)
    for position in range(min_frames, min(n, max_frames + 1)):
        dp[0, position] = cost[position]
    for row in range(1, count - 1):
        previous = dp[row - 1]
        for position in range(n):
            low = max(0, position - max_frames)
            high = position - min_frames
            if high < low:
                continue
            window = previous[low:high + 1]
            if not np.isfinite(window).any():
                continue
            offset = int(np.argmin(window))
            dp[row, position] = float(window[offset]) + float(cost[position])
            back[row, position] = low + offset

    best_total = infinity
    best_position = -1
    for position in range(n):
        if not np.isfinite(dp[count - 2, position]):
            continue
        remaining = (n - 1) - position
        if not min_frames <= remaining <= max_frames:
            continue
        if dp[count - 2, position] < best_total:
            best_total = float(dp[count - 2, position])
            best_position = position
    if best_position < 0:
        result.add_issue(
            "alignment_infeasible",
            f"no boundary placement fits {count} syllables into {n * hop_s:.2f}s with "
            f"{min_duration_s:.2f}-{max_duration_s:.2f}s per syllable", "error")
        return result

    edges = [best_position]
    for row in range(count - 2, 0, -1):
        edges.append(int(back[row, edges[-1]]))
    edges.reverse()

    positions = [0] + edges + [n]
    windows: list[Boundary] = []
    for index in range(count):
        start_position, stop_position = positions[index], positions[index + 1]
        start_s = (region_start + start_position) * hop_s
        end_s = min(signal.size / sample_rate, (region_start + stop_position) * hop_s)
        if index == 0:
            dip_db = confidence = None
        else:
            dip_db = float(20.0 * np.log10(
                local_peak[start_position]
                / max(float(smooth[region_start + start_position]), 1e-12)))
            confidence = float(np.clip(dip_db / confident_dip_db, 0.0, 1.0))
        windows.append(Boundary(start_s=start_s, end_s=end_s, dip_db=dip_db,
                               confidence=confidence,
                               voiced_ratio=_ratio(start_s, end_s)))
    result.boundaries = windows
    result.settings.update({"rise_weight": rise_weight, "rise_window_s": rise_window_s,
                            "dip_window_s": dip_window_s,
                            "confident_dip_db": confident_dip_db})
    return result
