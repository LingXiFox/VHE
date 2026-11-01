"""Multi-candidate YIN/CMND pitch tracking with Viterbi continuity (NumPy only).

EXP-0015. vhe.data.f0.estimate_f0 picks a single period per frame from the first CMND dip
and then tries to repair octave errors with two hand-tuned constants. That works on
sustained vowels but fails on glides and vibrato, and it has no way to say "I do not know".
This module keeps the same CMND core and adds three things:

1. Multiple period candidates. Every local dip of the CMND curve contributes its own lag
   plus the integer multiples (2x, 3x, 4x: sub-harmonic / octave-down errors) and the
   integer fractions (1/2, 1/3, 1/4: octave-up errors), so 2x/3x/4x and half-frequency
   mistakes are represented instead of being decided by a threshold.
2. Harmonic consistency. A true period is also a period at 2T, 3T, 4T; a half period is
   not a period at 3T/2, so the mean normalised correlation at the higher multiples
   separates them without a frequency-dependent threshold. The score is
   (1 - aperiodicity) * (0.5 + 0.5 * harmonic_support).
3. Frame-to-frame continuity. A Viterbi pass over per-frame states (candidates plus one
   unvoiced state) with a cost on |log2(f_t/f_{t-1})| removes isolated octave jumps that a
   per-frame decision cannot see.

Every frame carries an explicit reason. Frames the model cannot vouch for are rejected
(f0_hz = 0.0) with the reason recorded; nothing is interpolated, so downstream paired-data
code can count rejections instead of silently training on invented pitch.

Only used to build paired data and diagnostics: VHE itself never sees F0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .f0 import F0Frame, F0Track, cmnd_curve

#: Dips shallower than this are still candidates; anything above is not a period at all.
CANDIDATE_MAX_CMND = 0.55
#: A dip at or below this CMND value counts as a genuine near-period of the frame.
STRONG_DIP_CMND = 0.15
#: How many distinct CMND dips to seed the candidate set from (before multiples/fractions).
MAX_BASE_DIPS = 4
#: Multiples and fractions of each base lag that are scored, to represent octave errors.
HARMONIC_ORDERS = (2, 3, 4)
#: Candidates whose scores differ by less than this are treated as an octave tie; the shorter
#: period wins (frame-wise YIN fails towards 2T far more often than towards T/2, because the
#: shorter lag is the one that is always a genuine dip for a periodic signal).
OCTAVE_TIE_MARGIN = 0.02
#: A voiced candidate is rejected as ambiguous when another candidate at a non-integer octave
#: ratio is within this factor of its aperiodicity (with an absolute floor so that two
#: near-perfect candidates are still told apart by one being twice as aperiodic).
AMBIGUITY_AP_FACTOR = 1.5
AMBIGUITY_AP_FLOOR = 0.010
#: Weight of harmonic support inside the candidate score. Support must be able to veto a
#: half-period hypothesis but must not outvote the CMND depth itself: with an equal weight the
#: formant of /a/ (~825 Hz, the 4th harmonic of 206 Hz) tied with the true period under vibrato
#: and won the octave tie-break, because every multiple of 29 samples is also a dip.
SUPPORT_WEIGHT = 0.3
#: Below this candidate score the frame is rejected instead of reported.
MIN_CONFIDENCE = 0.30
#: Viterbi costs. Emission is -log(score); the unvoiced state has a fixed cost. The continuity
#: cost is per octave of frame-to-frame pitch change, capped at MAX_JUMP_OCTAVES.
UNVOICED_COST = 2.5
CONTINUITY_COST = 3.0
SWITCH_COST = 1.0
MAX_JUMP_OCTAVES = 2.0

REASONS = (
    "ok",
    "silence",
    "no_candidate",
    "low_confidence",
    "octave_ambiguous",
    "continuity_gated",
)


@dataclass(frozen=True)
class PeriodCandidate:
    """One period hypothesis for one frame."""

    f0_hz: float
    lag: float
    aperiodicity: float
    harmonic_support: float
    score: float

    def as_dict(self) -> dict:
        return {
            "f0_hz": round(self.f0_hz, 3),
            "lag": round(self.lag, 2),
            "aperiodicity": round(self.aperiodicity, 4),
            "harmonic_support": round(self.harmonic_support, 4),
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class PitchFrame:
    time_s: float
    f0_hz: float  # 0.0 means rejected or unvoiced, never an interpolated guess
    confidence: float
    reason: str = "ok"
    candidates: tuple[PeriodCandidate, ...] = ()

    @property
    def voiced(self) -> bool:
        return self.f0_hz > 0.0

    def as_dict(self) -> dict:
        return {
            "time_s": round(self.time_s, 4),
            "f0_hz": round(self.f0_hz, 2),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass
class PitchTrack:
    frames: list[PitchFrame] = field(default_factory=list)
    sample_rate: int = 0
    frame_s: float = 0.0
    hop_s: float = 0.0
    fmin_hz: float = 0.0
    fmax_hz: float = 0.0
    settings: dict = field(default_factory=dict)

    @property
    def voiced_ratio(self) -> float:
        if not self.frames:
            return 0.0
        return sum(1 for frame in self.frames if frame.voiced) / len(self.frames)

    @property
    def voiced_f0(self) -> np.ndarray:
        return np.array([f.f0_hz for f in self.frames if f.voiced], dtype=np.float64)

    def times(self) -> np.ndarray:
        return np.array([frame.time_s for frame in self.frames], dtype=np.float64)

    def contour(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, f0) with rejected frames as 0.0; gaps are the caller's problem."""
        return self.times(), np.array([f.f0_hz for f in self.frames], dtype=np.float64)

    def reasons(self) -> dict[str, int]:
        counts = {name: 0 for name in REASONS}
        for frame in self.frames:
            counts[frame.reason] = counts.get(frame.reason, 0) + 1
        return counts

    def stats(self) -> dict:
        voiced = self.voiced_f0
        counts = self.reasons()
        frames = len(self.frames)
        return {
            "frames": frames,
            "voiced_frames": int(voiced.size),
            "voiced_ratio": round(self.voiced_ratio, 4),
            "rejected_frames": frames - int(voiced.size),
            "rejection_rate": round((frames - int(voiced.size)) / frames, 4) if frames else 0.0,
            "reasons": counts,
            "median_f0_hz": round(float(np.median(voiced)), 2) if voiced.size else None,
            "p05_f0_hz": round(float(np.percentile(voiced, 5)), 2) if voiced.size else None,
            "p95_f0_hz": round(float(np.percentile(voiced, 95)), 2) if voiced.size else None,
            "median_abs_step_cents": round(median_step_cents(self), 2),
        }

    def to_f0_track(self) -> F0Track:
        """Legacy view so existing pairing code keeps working; voicing = candidate score."""
        track = F0Track(sample_rate=self.sample_rate, frame_s=self.frame_s, hop_s=self.hop_s,
                        fmin_hz=self.fmin_hz, fmax_hz=self.fmax_hz)
        track.frames = [F0Frame(time_s=f.time_s, f0_hz=f.f0_hz, voicing=f.confidence)
                        for f in self.frames]
        return track


def cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(a / b)


def octave_error_kind(estimate: float, reference: float, *, tolerance: float = 0.03) -> str | None:
    """Classify an octave-class error: '2x', '3x', '4x', '1/2x', ... or None."""
    ratio = estimate / reference
    for name, value in (("2x", 2.0), ("3x", 3.0), ("4x", 4.0),
                        ("1/2x", 1 / 2.0), ("1/3x", 1 / 3.0), ("1/4x", 1 / 4.0)):
        if abs(ratio - value) <= tolerance * value:
            return name
    return None


def median_step_cents(track: PitchTrack) -> float:
    """Median |frame-to-frame change| in cents over adjacent voiced frames."""
    steps = []
    previous = None
    for frame in track.frames:
        if not frame.voiced:
            previous = None
            continue
        if previous is not None:
            steps.append(abs(cents(frame.f0_hz, previous)))
        previous = frame.f0_hz
    return float(np.median(steps)) if steps else 0.0


def _refine_lag(diff: np.ndarray, index: int) -> float:
    """Parabolic refinement of the integer lag index using the raw difference curve."""
    if 0 < index < diff.size - 1:
        left, center, right = diff[index - 1], diff[index], diff[index + 1]
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-12:
            shift = 0.5 * (left - right) / denom
            return float(index) + float(np.clip(shift, -0.5, 0.5))
    return float(index)


def frame_candidates(frame: np.ndarray, sample_rate: int, *, fmin_hz: float = 65.0,
                     fmax_hz: float = 900.0, max_dips: int = MAX_BASE_DIPS,
                     max_cmnd: float = CANDIDATE_MAX_CMND) -> tuple[PeriodCandidate, ...]:
    """All period hypotheses for one frame, octave relatives included."""
    centered = np.asarray(frame, dtype=np.float64) - float(np.mean(frame))
    if float(np.dot(centered, centered)) <= 1e-12:
        return ()
    lag_min = max(2, int(np.floor(sample_rate / fmax_hz)))
    lag_max = min(centered.size - 3, int(np.ceil(sample_rate / fmin_hz)))
    if lag_max <= lag_min or centered.size - lag_max < 16:
        return ()
    lags, diff, cmnd = cmnd_curve(centered, lag_max)

    # cmnd_curve returns arrays aligned with lags 1..lag_max, so array index i is lag i + 1.
    sub = cmnd[lag_min - 1:lag_max]
    if sub.size < 3:
        return ()
    dips = [i for i in range(1, sub.size - 1)
            if sub[i] <= sub[i - 1] and sub[i] <= sub[i + 1] and sub[i] <= max_cmnd]
    deepest = int(np.argmin(sub))
    if sub[deepest] <= max_cmnd and deepest not in dips:
        dips.append(deepest)
    if not dips:
        return ()
    # Two-step selection, because neither pure rule survives the corpus:
    # - deepest-first drops the fundamental on tones (all multiples tie near zero, floating
    #   point picks 4T for a 350 Hz sine);
    # - shortest-first drops the fundamental on formant-heavy vowels (/a/ at 165 Hz has shallow
    #   dips at 29/58/87/116 caused by the ~825 Hz formant, and the real dip at 145 is fifth).
    # So first keep the dips that are actually deep (absolute YIN-like gate), then prefer the
    # shortest of those; a periodic signal keeps its fundamental among them, a vowel keeps the
    # true period. Only when nothing is deep do we fall back to the shallowest minima, which
    # is what noisy frames need to produce rejectable candidates instead of nothing.
    strong = [i for i in dips if sub[i] <= STRONG_DIP_CMND]
    if strong:
        base_lags = [lag_min + i for i in sorted(strong)[:max_dips]]
    else:
        dips.sort(key=lambda i: sub[i])
        base_lags = [lag_min + i for i in dips[:max_dips]]

    lag_set: set[int] = set()
    for base in base_lags:
        for order in (1, *HARMONIC_ORDERS):
            lag = base * order
            if lag_min <= lag <= lag_max:
                lag_set.add(lag)
        for order in HARMONIC_ORDERS:
            lag = int(round(base / order))
            if lag_min <= lag <= lag_max:
                lag_set.add(lag)
    if not lag_set:
        return ()

    candidates: list[PeriodCandidate] = []
    for lag in sorted(lag_set):
        aperiodicity = float(cmnd[lag - 1])
        support_terms = []
        for order in HARMONIC_ORDERS:
            higher = lag * order
            if higher <= lag_max:
                support_terms.append(float(np.clip(1.0 - cmnd[higher - 1], 0.0, 1.0)))
        support = float(np.mean(support_terms)) if support_terms else float(1.0 - aperiodicity)
        score = float(np.clip(1.0 - aperiodicity, 0.0, 1.0)) * (
            1.0 - SUPPORT_WEIGHT + SUPPORT_WEIGHT * support)
        refined = _refine_lag(diff, lag - 1) + 1.0
        candidates.append(PeriodCandidate(f0_hz=float(sample_rate / refined), lag=refined,
                                          aperiodicity=aperiodicity, harmonic_support=support,
                                          score=score))

    return tuple(prefer_shortest_period(candidates))


def prefer_shortest_period(candidates: list[PeriodCandidate]) -> list[PeriodCandidate]:
    """Drop sub-harmonic duplicates when a shorter period explains the frame nearly as well.

    A periodic signal makes every multiple of its period an equally deep CMND dip, so T, 2T,
    3T and 4T score the same and floating point decides which one wins. The resolution is a
    prior, not a threshold: parse the frame with the shortest period that fits. The opposite
    error (picking T/2 because the waveform is dominated by its second harmonic) is *not*
    resolved here: that candidate scores lower through harmonic support, because 1.5T is not
    a dip. Ordering by lag before the greedy pass is what makes the result independent of the
    score ordering.
    """
    kept: list[PeriodCandidate] = []
    # Shortest period first: descending f0 is ascending lag.
    for candidate in sorted(candidates, key=lambda c: -c.f0_hz):
        redundant = False
        for shorter in kept:
            multiple = shorter.f0_hz / candidate.f0_hz  # period ratio, >= 1
            nearest = int(round(multiple))
            # Any integer multiple counts, not just 2..4: with fmin/fmax = 65/900 a frame can
            # contain up to 13 periods, and a 6x candidate at 83.77 Hz beat a correct 500 Hz
            # estimate before this range was widened.
            if 2 <= nearest <= 16 and abs(multiple - nearest) <= 0.03 * nearest:
                if candidate.score - shorter.score < OCTAVE_TIE_MARGIN:
                    redundant = True
                    break
        if not redundant:
            kept.append(candidate)
    kept.sort(key=lambda c: -c.score)
    return kept


def _emission(candidate: PeriodCandidate) -> float:
    return -math.log(max(min(candidate.score, 1.0), 1e-3))


def _transition(previous: PeriodCandidate | None, current: PeriodCandidate | None) -> float:
    if previous is None and current is None:
        return 0.0
    if previous is None or current is None:
        return SWITCH_COST
    jump = abs(math.log2(current.f0_hz / previous.f0_hz))
    return CONTINUITY_COST * min(jump, MAX_JUMP_OCTAVES)


def _viterbi(per_frame: list[tuple[PeriodCandidate, ...]], *, unvoiced_cost: float) -> list[int]:
    """Decode the cheapest state path; -1 means the unvoiced/rejected state."""
    costs: list[list[float]] = []
    back: list[list[int]] = []
    for frame_index, candidates in enumerate(per_frame):
        states: list[PeriodCandidate | None] = list(candidates) + [None]
        frame_cost: list[float] = []
        frame_back: list[int] = []
        for state in states:
            emission = unvoiced_cost if state is None else _emission(state)
            if frame_index == 0:
                frame_cost.append(emission)
                frame_back.append(-1)
                continue
            previous_states: list[PeriodCandidate | None] = list(per_frame[frame_index - 1]) + [None]
            best_cost, best_index = math.inf, 0
            for previous_index, previous_state in enumerate(previous_states):
                total = costs[frame_index - 1][previous_index] + _transition(previous_state, state)
                if total < best_cost:
                    best_cost, best_index = total, previous_index
            frame_cost.append(emission + best_cost)
            frame_back.append(best_index)
        costs.append(frame_cost)
        back.append(frame_back)

    last = int(np.argmin(costs[-1]))
    path = [last]
    for frame_index in range(len(per_frame) - 1, 0, -1):
        last = back[frame_index][last]
        path.append(last)
    path.reverse()
    # Map state index to candidate index: the unvoiced state is always last.
    return [-1 if index == len(per_frame[i]) else index for i, index in enumerate(path)]


def _same_period_family(ratio: float, *, tolerance: float = 0.03) -> bool:
    """True when two lags are integer multiples of each other (T vs 2T, 3T, T/2, ...).

    Both are genuine dips of a periodic signal, so their scores tie by construction; that tie
    is resolved by preferring the shortest period, not by rejecting the frame. Rejecting it
    would throw away every clean vowel, which is exactly what the first draft of this module
    did on a 210 Hz sine.
    """
    if ratio <= 0.0:
        return False
    if ratio < 1.0:
        ratio = 1.0 / ratio
    for value in (1, 2, 3, 4):
        if abs(ratio - value) <= tolerance * value:
            return True
    return False


def _ambiguous(candidate: PeriodCandidate, candidates: tuple[PeriodCandidate, ...]) -> bool:
    """Another, structurally different period explains the frame about as well as the chosen one.

    The comparison is on aperiodicity, not on the score: a formant artefact can score 0.93
    against 0.99 because the score mixes depth with harmonic support, while the depth itself
    (0.053 vs 0.005) shows it is ten times less periodic. Using the score window rejected 18%
    of a clean 300 Hz synthetic /a/ on a spurious 90 Hz dip produced by its ~900 Hz formant.
    """
    limit = max(AMBIGUITY_AP_FLOOR, AMBIGUITY_AP_FACTOR * candidate.aperiodicity)
    for other in candidates:
        if other is candidate or other.aperiodicity > limit:
            continue
        ratio = candidate.f0_hz / other.f0_hz
        if _same_period_family(ratio):
            continue
        if 1.9 <= ratio <= 4.1 or 1 / 4.1 <= ratio <= 1 / 1.9:
            return True
    return False


def track_pitch(audio: np.ndarray, sample_rate: int, *, fmin_hz: float = 65.0,
                fmax_hz: float = 900.0, frame_ms: float = 30.0, hop_ms: float = 10.0,
                silence_floor_db: float = -60.0, absolute_floor: float = 1e-3,
                min_confidence: float = MIN_CONFIDENCE, unvoiced_cost: float = UNVOICED_COST,
                continuity_cost: float = CONTINUITY_COST, switch_cost: float = SWITCH_COST,
                max_dips: int = MAX_BASE_DIPS,
                max_cmnd: float = CANDIDATE_MAX_CMND) -> PitchTrack:
    """Frame-wise pitch track with candidates, Viterbi continuity and explicit rejections.

    The 30 ms analysis window is deliberate. YIN's usual 25-40 ms is fine for sustained vowels,
    but a 40 ms window smears a fast glide: 300->130 Hz over 0.6 s changes the period by ~6%
    inside one frame, the true period stops being a dip, and the stationary formant of /a/
    (~880 Hz) wins instead (EXP-0015 measured 16% octave errors at 40 ms and 0% at 30 ms,
    with every other synthetic contour unchanged).
    """
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("track_pitch expects mono audio with shape [N]")
    if signal.size == 0:
        raise ValueError("track_pitch expects a non-empty signal")
    if not 0.0 < fmin_hz < fmax_hz:
        raise ValueError(f"need 0 < fmin < fmax, got {fmin_hz}..{fmax_hz}")
    window = max(64, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if window > signal.size:
        signal = np.pad(signal, (0, window - signal.size), mode="reflect")

    starts = list(range(0, signal.size - window + 1, hop))
    frames = [signal[start:start + window] for start in starts]
    levels = np.array([float(np.sqrt(np.mean(frame * frame))) for frame in frames])
    gate = max(absolute_floor,
               float(levels.max()) * 10.0 ** (silence_floor_db / 20.0)) if levels.size else 0.0

    per_frame: list[tuple[PeriodCandidate, ...]] = []
    gated: list[bool] = []
    for frame, level in zip(frames, levels):
        if level < gate:
            per_frame.append(())
            gated.append(True)
            continue
        gated.append(False)
        per_frame.append(frame_candidates(frame, sample_rate, fmin_hz=fmin_hz, fmax_hz=fmax_hz,
                                          max_dips=max_dips, max_cmnd=max_cmnd))

    path = _viterbi(per_frame, unvoiced_cost=unvoiced_cost)

    track = PitchTrack(sample_rate=sample_rate, frame_s=window / sample_rate,
                       hop_s=hop / sample_rate, fmin_hz=fmin_hz, fmax_hz=fmax_hz,
                       settings={"frame_ms": frame_ms, "hop_ms": hop_ms,
                                 "min_confidence": min_confidence,
                                 "unvoiced_cost": unvoiced_cost,
                                 "continuity_cost": continuity_cost,
                                 "switch_cost": switch_cost,
                                 "max_dips": max_dips, "max_cmnd": max_cmnd,
                                 "octave_tie_margin": OCTAVE_TIE_MARGIN,
                                 "ambiguity_ap_factor": AMBIGUITY_AP_FACTOR,
                                 "ambiguity_ap_floor": AMBIGUITY_AP_FLOOR,
                                 "silence_floor_db": silence_floor_db,
                                 "absolute_floor": absolute_floor})
    for index, (start, candidates, choice) in enumerate(zip(starts, per_frame, path)):
        time_s = (start + window / 2.0) / sample_rate
        if choice < 0:
            if gated[index]:
                reason, confidence = "silence", 0.0
            elif not candidates:
                reason, confidence = "no_candidate", 0.0
            else:
                reason = "continuity_gated"
                confidence = candidates[0].score
            track.frames.append(PitchFrame(time_s=time_s, f0_hz=0.0, confidence=confidence,
                                           reason=reason, candidates=candidates))
            continue
        candidate = candidates[choice]
        if candidate.score < min_confidence:
            track.frames.append(PitchFrame(time_s=time_s, f0_hz=0.0, confidence=candidate.score,
                                           reason="low_confidence", candidates=candidates))
            continue
        if _ambiguous(candidate, candidates):
            track.frames.append(PitchFrame(time_s=time_s, f0_hz=0.0, confidence=candidate.score,
                                           reason="octave_ambiguous", candidates=candidates))
            continue
        track.frames.append(PitchFrame(time_s=time_s, f0_hz=candidate.f0_hz,
                                       confidence=candidate.score, reason="ok",
                                       candidates=candidates))
    return track
