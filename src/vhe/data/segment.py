"""Energy-based syllable segmentation and per-utterance quality flags.

Chinese read speech has no reliable silence between syllables, so this is deliberately a
coarse segmentation: it splits on energy valleys and is used to build timing for the
synthesizer and to *count* syllables against the transcript. Anything that does not fit
(minimum duration, syllable-count mismatch, clipping) becomes a reported issue; nothing is
silently dropped from the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .f0 import F0Track


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    rms: float
    voiced_ratio: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def voiced(self) -> Optional[bool]:
        if self.voiced_ratio is None:
            return None
        return self.voiced_ratio >= 0.5


@dataclass
class Segmentation:
    segments: list[Segment] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    threshold: float = 0.0
    hop_s: float = 0.0

    @property
    def speech_duration_s(self) -> float:
        return sum(seg.duration_s for seg in self.segments)

    def add_issue(self, code: str, detail: str, severity: str = "warning") -> None:
        self.issues.append({"code": code, "detail": detail, "severity": severity})

    def stats(self) -> dict:
        voiced = [seg for seg in self.segments if seg.voiced]
        return {
            "segments": len(self.segments),
            "speech_duration_s": round(self.speech_duration_s, 3),
            "voiced_segments": len(voiced),
            "issues": len(self.issues),
            "issue_codes": sorted({issue["code"] for issue in self.issues}),
        }


def _frame_rms(signal: np.ndarray, window: int, hop: int) -> np.ndarray:
    if signal.size < window:
        signal = np.pad(signal, (0, window - signal.size))
    count = 1 + (signal.size - window) // hop
    index = np.arange(window)[None, :] + hop * np.arange(count)[:, None]
    frames = signal[index]
    return np.sqrt(np.mean(frames * frames, axis=1))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def segmentation_threshold(rms: np.ndarray, *, floor_percentile: float = 10.0,
                           peak_percentile: float = 95.0,
                           relative: float = 0.12) -> float:
    floor = float(np.percentile(rms, floor_percentile))
    peak = float(np.percentile(rms, peak_percentile))
    if peak <= 0.0:
        # Very short utterances can put the only voiced frames above the percentile.
        peak = float(rms.max())
    if peak <= 0.0:
        return 0.0
    return floor + relative * (peak - floor)


def segment_by_energy(audio: np.ndarray, sample_rate: int, *, frame_ms: float = 25.0,
                      hop_ms: float = 10.0, floor_percentile: float = 10.0,
                      peak_percentile: float = 95.0, relative_threshold: float = 0.12,
                      min_segment_s: float = 0.06, merge_gap_s: float = 0.05,
                      f0_track: Optional[F0Track] = None,
                      report_clipping: bool = True) -> Segmentation:
    """Split a mono signal into energy bursts; unvoiced bursts stay in the list."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("segment_by_energy expects mono audio with shape [N]")
    if signal.size == 0:
        raise ValueError("segment_by_energy expects a non-empty signal")
    window = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    rms = _frame_rms(signal, window, hop)
    if rms.size >= 3:
        rms = np.convolve(rms, np.ones(3) / 3.0, mode="same")

    result = Segmentation(hop_s=hop / sample_rate)
    if report_clipping:
        clipped = float(np.mean(np.abs(signal) >= 0.999))
        if clipped > 0.001:
            result.add_issue("clipping", f"{clipped:.4%} of samples at full scale")

    threshold = segmentation_threshold(rms, floor_percentile=floor_percentile,
                                       peak_percentile=peak_percentile,
                                       relative=relative_threshold)
    result.threshold = threshold
    if threshold <= 0.0:
        result.add_issue("no_speech", "signal is silent; no segments produced", "error")
        return result

    runs = _runs(rms > threshold)
    if not runs:
        result.add_issue("no_speech", "no frame exceeded the energy threshold", "error")
        return result

    merge_frames = max(1, int(round(merge_gap_s / result.hop_s)))
    merged: list[list[int]] = [list(runs[0])]
    for start, stop in runs[1:]:
        if start - merged[-1][1] <= merge_frames:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])

    times, f0 = (f0_track.contour() if f0_track is not None else (None, None))
    for start_frame, stop_frame in merged:
        # A frame is window samples long but frames are counted on hop; using hop * count
        # keeps durations unbiased (adding the window would inflate every burst by ~25 ms).
        start_s = start_frame * result.hop_s
        end_s = min(signal.size / sample_rate, stop_frame * result.hop_s)
        end_s = max(end_s, start_s + result.hop_s)
        duration = end_s - start_s
        if duration < min_segment_s:
            result.add_issue("segment_too_short",
                             f"burst at {start_s:.3f}s dropped: {duration * 1000:.0f} ms "
                             f"< {min_segment_s * 1000:.0f} ms")
            continue
        rms_value = float(np.max(rms[start_frame:stop_frame]))
        voiced_ratio = None
        if f0_track is not None:
            inside = (times >= start_s) & (times <= end_s)
            if bool(np.any(inside)):
                voiced_ratio = float(np.mean(f0[inside] > 0))
        result.segments.append(Segment(start_s=start_s, end_s=end_s, rms=rms_value,
                                       voiced_ratio=voiced_ratio))

    if not result.segments:
        result.add_issue("no_speech", "all detected bursts were shorter than the minimum",
                         "error")
    return result


def check_syllable_count(segmentation: Segmentation, expected: int, *, tolerance: int = 0) -> bool:
    """Compare detected bursts against the transcript; mismatch is recorded, never hidden."""
    detected = len(segmentation.segments)
    if abs(detected - expected) <= tolerance:
        return True
    segmentation.add_issue(
        "syllable_count_mismatch",
        f"transcript has {expected} syllables, energy segmentation found {detected}")
    return False
