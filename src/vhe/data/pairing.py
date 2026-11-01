"""Build (x_mechanical, x_real) pairs from one real utterance.

The pipeline is deliberately strict about what it can render. The pinyin token is mapped
to phoneme ids by :mod:`vhe.data.g2p` against the *backend's own* inventory, and a syllable
whose onset or nucleus is missing from that inventory BLOCKS the pair instead of being
approximated: "guang3" is not silently rendered as "uang", and an apical "si" is not
silently rendered as /si/. The blocked reason is part of the record.

Timing fidelity: each syllable is rendered on its own and placed at the onset measured
from the real utterance, with no lead/tail silence of its own, so the two signals share a
length and share the pauses. Inside a syllable the onset/nucleus split is the fixed
heuristic from g2p (nominal onset length, nucleus takes the remainder), not a forced
alignment; the allocation is recorded per syllable. The cost is a fade at every syllable
edge, which the pair record reports as an issue.

Nothing here is committed to Git: real recordings never enter the repository, and the
mechanical signal is derived from the real one only through F0/timing analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..audio import read_wav, write_wav
from ..synth import (PitchCurve, PhonemeSegment, PhonemeSequence, SynthesisRequest,
                     SynthBackend, Timing, VoiceTemplate, get_backend)
from ..synth.contract import F0_MAX_HZ, F0_MIN_HZ
from .mandarin import PinyinError, split_token
from .g2p import syllable_plan, syllable_phonemes
from .aishell3 import file_sha256
from .align import align_syllables
from .f0 import F0Track, estimate_f0
from .pitch import PitchTrack, track_pitch
from .resample import resample
from .segment import check_syllable_count, segment_by_energy

INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "r", "z", "c", "s", "y", "w")
PAIR_STATUS = ("ok", "blocked", "rejected")
#: 'viterbi' is the EXP-0015 multi-candidate tracker; 'legacy' keeps the old YIN estimate
#: available as an ablation. Neither may narrow fmax_hz to hide octave errors.
PITCH_ESTIMATORS = ("viterbi", "legacy")
ALIGNMENT_MODES = ("segments", "dp", "forced")
#: Confidence below which a DP boundary is reported as unreliable (0.35 == 4.2 dB valley).
DEFAULT_MIN_BOUNDARY_CONFIDENCE = 0.35
#: An interior rejection gap longer than this is reported as interpolated, not smoothed over.
DEFAULT_MAX_PITCH_GAP_S = 0.05


@dataclass(frozen=True)
class Syllable:
    pinyin: str
    #: orthographic onset as written ('y'/'w' included, '' for a vowel-initial syllable)
    initial: str
    #: orthographic final as written after the onset ('ue' in yue4, 'i' in zhi1)
    final: str
    tone: int
    phonemes: tuple[str, ...] = ()
    reason: Optional[str] = None
    #: resolved final: 've' for yue4, 'apical_retroflex' for zhi1, 'iou' for liu2
    underlying_final: str = ""
    #: trailing r is a modified final, not a coda; never rendered as base + /r/
    erhua: bool = False

    @property
    def phoneme(self) -> Optional[str]:
        """The single phoneme of a one-phoneme syllable, else None (kept for callers)."""
        return self.phonemes[0] if len(self.phonemes) == 1 else None

    def as_dict(self) -> dict:
        return {"pinyin": self.pinyin, "initial": self.initial, "final": self.final,
                "underlying_final": self.underlying_final, "tone": self.tone,
                "erhua": self.erhua, "phonemes": list(self.phonemes),
                "phoneme": self.phoneme, "reason": self.reason}


def parse_syllable(pinyin: str, inventory: str = "vowels") -> Syllable:
    """Split an AISHELL-3 token like "guang3" into initial/final/tone (no fuzzy matching).

    The orthographic split (what the annotation writes) and the resolved final (what the
    syllable actually is) are both kept: "yue4" is initial y + final ue, underlying ve;
    "zhi1" is zh + i, underlying apical_retroflex; "liu2" is l + iu, underlying iou.
    Resolution is shared with the corpus scanner via vhe.data.mandarin, so coverage
    statistics and rendering cannot disagree.

    "vowels" keeps the original behaviour of the {a, i, u} backend; "mandarin" resolves
    onsets and finals through vhe.data.g2p.
    """
    try:
        token = split_token(pinyin)
    except PinyinError as error:
        raise ValueError(error.detail) from None
    # g2p resolves the *orthographic* final back to its underlying form, so it must be given
    # the written spelling. Passing the underlying final here built fake pinyin ("shi" ->
    # "sh" + "apical_retroflex" -> "shapical_retroflex"), which blocked 3160 of the 5314
    # manifest-renderable utterances at render time while coverage said they were fine.
    written = token.written or token.final
    phonemes, reason = syllable_phonemes(token.initial, written, inventory,
                                         erhua=token.erhua)
    return Syllable(token.pinyin, token.initial, written, token.tone,
                    phonemes, reason, underlying_final=token.final, erhua=token.erhua)


@dataclass
class PairResult:
    utterance_id: str
    status: str
    sample_rate: int
    source_sample_rate: int
    issues: list[dict] = field(default_factory=list)
    syllables: list[dict] = field(default_factory=list)
    f0_stats: dict = field(default_factory=dict)
    segmentation_stats: dict = field(default_factory=dict)
    real: Optional[np.ndarray] = None
    mechanical: Optional[np.ndarray] = None
    request_fingerprint: Optional[str] = None
    request_fingerprints: list[str] = field(default_factory=list)
    gains_db: list[float] = field(default_factory=list)
    #: per syllable, the (phoneme, seconds) allocation handed to the synthesizer
    allocations: list[list[dict]] = field(default_factory=list)
    real_sha256: Optional[str] = None
    mechanical_sha256: Optional[str] = None
    template: Optional[dict] = None
    #: F0 estimator identity, per-syllable coverage and every interior rejection gap.
    pitch: dict = field(default_factory=dict)
    #: Alignment method, per-boundary confidence and why the method was chosen.
    alignment: dict = field(default_factory=dict)
    #: {'training_ready': bool, 'reasons': [...]} -- never inferred from status alone.
    readiness: dict = field(default_factory=dict)

    def add_issue(self, code: str, detail: str, severity: str = "warning") -> None:
        self.issues.append({"code": code, "detail": detail, "severity": severity})

    def as_dict(self) -> dict:
        return {
            "utterance_id": self.utterance_id,
            "status": self.status,
            "sample_rate": self.sample_rate,
            "source_sample_rate": self.source_sample_rate,
            "real_sha256": self.real_sha256,
            "mechanical_sha256": self.mechanical_sha256,
            "request_fingerprint": self.request_fingerprint,
            "request_fingerprints": self.request_fingerprints,
            "gains_db": self.gains_db,
            "allocations": self.allocations,
            "template": self.template,
            "f0_stats": self.f0_stats,
            "pitch": self.pitch,
            "alignment": self.alignment,
            "readiness": self.readiness,
            "segmentation_stats": self.segmentation_stats,
            "syllables": self.syllables,
            "issues": self.issues,
        }


def _audio_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _analyse_pitch(audio: np.ndarray, sample_rate: int, estimator: str
                   ) -> tuple[F0Track | PitchTrack, dict]:
    """Run the selected F0 estimator and return its track plus a flat stats record."""
    if estimator == "legacy":
        track = estimate_f0(audio, sample_rate)
        return track, {"estimator": "legacy_yin", **track.stats()}
    if estimator == "viterbi":
        track = track_pitch(audio, sample_rate)
        stats = {"estimator": "viterbi_multicandidate", **track.stats()}
        # The coverage report already consumed median_f0_hz from the legacy track.
        stats["median_f0_hz"] = stats.get("median")
        return track, stats
    raise ValueError(f"unknown pitch estimator {estimator!r}; expected one of "
                     f"{', '.join(PITCH_ESTIMATORS)}")


def _pitch_curve(track: F0Track | PitchTrack, start_s: float, voiced_s: float, *,
                 index: int, max_gap_s: float,
                 result: PairResult) -> tuple[Optional[PitchCurve], dict]:
    """Fit a PitchCurve to one syllable window without hiding rejected frames.

    Rejected frames are never silently bridged: the count and the longest interior run of
    unvoiced frames are recorded and any run above max_gap_s raises pitch_gap_interpolated,
    which keeps the pair out of the training-ready set.
    """
    times, f0 = track.contour()
    inside = (times >= start_s) & (times < start_s + voiced_s)
    frames = int(inside.sum())
    voiced = inside & (f0 > 0)
    voiced_frames = int(voiced.sum())
    info = {"index": index, "frames": frames, "voiced_frames": voiced_frames,
            "coverage": round(voiced_frames / frames, 4) if frames else 0.0,
            "interior_gaps": 0, "longest_gap_s": 0.0, "lead_s": None, "tail_s": None,
            "octave_ambiguous_frames": 0}
    if voiced_frames < 2:
        result.add_issue("insufficient_pitch_points",
                         f"syllable {index}: only {voiced_frames} voiced F0 frames inside "
                         f"{voiced_s:.3f}s", "error")
        return None, info
    hop_s = float(getattr(track, "hop_s", 0.01))
    order = np.flatnonzero(inside)
    mask = f0[order] > 0
    gaps: list[float] = []
    position = 0
    while position < mask.size:
        if not mask[position]:
            stop = position
            while stop < mask.size and not mask[stop]:
                stop += 1
            if position > 0 and stop < mask.size:
                gaps.append((stop - position) * hop_s)
            position = stop
        else:
            position += 1
    first_frame = int(np.argmax(mask))
    last_frame = int(mask.size - 1 - np.argmax(mask[::-1]))
    info["interior_gaps"] = len(gaps)
    info["longest_gap_s"] = round(max(gaps, default=0.0), 4)
    info["lead_s"] = round(max(0.0, float(times[order[first_frame]]) - start_s), 4)
    info["tail_s"] = round(max(0.0, start_s + voiced_s - float(times[order[last_frame]])), 4)
    if frames and voiced_frames / frames < 0.35:
        result.add_issue("low_pitch_coverage",
                         f"syllable {index}: {voiced_frames}/{frames} frames voiced "
                         f"({100.0 * voiced_frames / frames:.0f}%)", "warning")
    if info["longest_gap_s"] > max_gap_s:
        result.add_issue("pitch_gap_interpolated",
                         f"syllable {index}: {len(gaps)} interior rejection gap(s), longest "
                         f"{1000.0 * info['longest_gap_s']:.0f} ms > "
                         f"{1000.0 * max_gap_s:.0f} ms; the curve is linearly interpolated "
                         "across it and the pair is not training-ready", "warning")
    if info["lead_s"] > 0.03 or info["tail_s"] > 0.03:
        result.add_issue("pitch_edge_held",
                         f"syllable {index}: first voiced frame {1000.0 * info['lead_s']:.0f} ms "
                         f"after / last {1000.0 * info['tail_s']:.0f} ms before the window edge; "
                         "the synthesizer holds the edge value", "warning")
    t = np.asarray(times[voiced]) - start_s
    values = np.clip(np.asarray(f0[voiced]), F0_MIN_HZ, F0_MAX_HZ)
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t, values = t[keep], values[keep]
    if t[-1] < voiced_s:
        t = np.append(t, voiced_s)
        values = np.append(values, values[-1])
    return (PitchCurve(times_s=tuple(float(v) for v in t),
                       f0_hz=tuple(float(v) for v in values)), info)


def _manual_windows(boundaries_s, count: int, duration_s: float
                    ) -> list[tuple[float, float, None, None]]:
    pairs = [(float(start), float(stop)) for start, stop in boundaries_s]
    if len(pairs) != count:
        raise ValueError(f"{len(pairs)} manual boundaries for {count} syllables")
    previous = 0.0
    for index, (start, stop) in enumerate(pairs):
        if not (0.0 <= start < stop <= duration_s + 1e-6):
            raise ValueError(f"boundary {index} ({start:.3f}, {stop:.3f}) is outside "
                             f"0..{duration_s:.3f}s")
        if start < previous - 1e-6:
            raise ValueError(f"boundary {index} starts at {start:.3f}s, before the previous "
                             "boundary ends")
        if stop - start < 0.02:
            raise ValueError(f"boundary {index} lasts {stop - start:.3f}s, below the 0.02s floor")
        previous = stop
    return [(start, stop, None, None) for start, stop in pairs]


def _span_windows(spans, syllables, duration_s: float, result: "PairResult"):
    """Group forced-alignment phone spans into per-syllable windows and phone plans.

    The aligner emits one (phone, start, end) triple per transcript phoneme in order.
    Turning those into a pair is deliberately strict: a sequence that does not match the
    transcript phone-for-phone is an aligner/transcript disagreement, not something to be
    repaired by dropping or interpolating phones, so it fails the pair instead.
    """
    if any(len(span) not in (3, 4) for span in spans):
        raise ValueError("phone spans must be (phone, start, end) or "
                         "(phone, start, end, confidence) tuples")
    expected = [phone for syllable in syllables for phone in syllable.phonemes]
    names = [str(span[0]) for span in spans]
    if len(spans) != len(expected):
        result.add_issue("phone_sequence_mismatch",
                         f"alignment has {len(spans)} spans for {len(expected)} transcript "
                         f"phonemes", "error")
        return None, None, {"method": "forced_alignment", "syllables": 0}
    if names != expected:
        first = next((index for index, (a, b) in enumerate(zip(names, expected)) if a != b),
                     None)
        result.add_issue("phone_sequence_mismatch",
                         f"aligned phones differ from the transcript at index {first}: "
                         f"{names[first]!r} vs {expected[first]!r}"
                         if first is not None else "aligned phones differ from the transcript",
                         "error")
        return None, None, {"method": "forced_alignment", "syllables": 0}
    spans = [(str(span[0]), float(span[1]), float(span[2]),
              None if len(span) < 4 or span[3] is None else float(span[3])) for span in spans]
    for index, (name, start, stop, _) in enumerate(spans):
        if not (0.0 <= start < stop <= duration_s + 1e-6):
            result.add_issue("bad_phone_span",
                             f"span {index} {name!r} ({start:.3f}, {stop:.3f}) is outside "
                             f"0..{duration_s:.3f}s", "error")
            return None, None, {"method": "forced_alignment", "syllables": 0}
        if index and start < spans[index - 1][2] - 1e-6:
            result.add_issue("bad_phone_span",
                             f"span {index} {name!r} starts at {start:.3f}s, before span "
                             f"{index - 1} ends ({spans[index - 1][2]:.3f}s)", "error")
            return None, None, {"method": "forced_alignment", "syllables": 0}

    windows, plans, confidences = [], [], []
    cursor = 0
    for syllable in syllables:
        count = len(syllable.phonemes)
        block = spans[cursor:cursor + count]
        cursor += count
        start, stop = block[0][1], block[-1][2]
        if stop - start < 0.02:
            result.add_issue("bad_syllable_span",
                             f"syllable {syllable.pinyin!r} spans {stop - start:.3f}s, "
                             "below the 0.02s floor", "error")
            return None, None, {"method": "forced_alignment", "syllables": 0}
        block_confidences = [value for _, _, _, value in block if value is not None]
        confidence = min(block_confidences) if block_confidences else None
        windows.append((start, stop, None, confidence))
        if confidence is not None:
            confidences.append(confidence)
        plans.append([(name, max(0.005, end - begin)) for name, begin, end, _ in block])
    stats = _window_stats(windows, method="forced_alignment", confidences=confidences)
    stats["source"] = "chain-HMM forced alignment (10 ms frames)"
    stats["phones"] = [[name, round(begin, 4), round(end, 4)] for name, begin, end, _ in spans]
    return windows, plans, stats


def _window_stats(windows, *, method: str, confidences: list[float]) -> dict:
    durations = [stop - start for start, stop, _, _ in windows]
    return {
        "method": method,
        "syllables": len(windows),
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
        "min_confidence": round(float(np.min(confidences)), 3) if confidences else None,
        "low_confidence_boundaries": sum(1 for value in confidences if value < 0.35),
        "mean_duration_s": round(float(np.mean(durations)), 3) if durations else None,
        "min_duration_s": round(float(np.min(durations)), 3) if durations else None,
        "max_duration_s": round(float(np.max(durations)), 3) if durations else None,
        # The windows themselves, so a pair record can be audited or re-rendered without
        # recomputing the alignment.
        "boundaries": [{"start_s": round(float(start), 4), "end_s": round(float(stop), 4),
                        "duration_s": round(float(stop - start), 4),
                        # Per-boundary evidence, so a pair record can show *which* join was
                        # weak instead of only the syllable-level minimum.
                        "dip_db": None if dip_db is None else round(float(dip_db), 2),
                        "confidence": None if confidence is None
                        else round(float(confidence), 3)}
                       for start, stop, dip_db, confidence in windows],
    }


def _resolve_windows(audio: np.ndarray, sample_rate: int, count: int, track,
                     *, alignment: str, boundaries_s, result: PairResult):
    """Return one (start, end, dip_db, confidence) window per syllable, or None."""
    duration_s = audio.size / sample_rate
    if boundaries_s is not None:
        try:
            windows = _manual_windows(boundaries_s, count, duration_s)
        except ValueError as error:
            result.add_issue("bad_manual_boundaries", str(error), "error")
            return None, {"method": "manual", "syllables": 0}
        # Keep the automatic view in the record so a manual run can be audited against it.
        seg = segment_by_energy(audio, sample_rate, f0_track=track)
        result.segmentation_stats = seg.stats()
        stats = _window_stats(windows, method="manual", confidences=[])
        stats["source"] = "operator-specified boundaries"
        return windows, stats
    if alignment not in ALIGNMENT_MODES:
        raise ValueError(f"unknown alignment {alignment!r}; expected one of "
                         f"{', '.join(ALIGNMENT_MODES)}")
    seg = segment_by_energy(audio, sample_rate, f0_track=track)
    result.segmentation_stats = seg.stats()
    result.issues.extend(seg.issues)
    if not seg.segments:
        result.add_issue("no_segments", "energy segmentation produced no syllables", "error")
        return None, {"method": "energy_segments", "syllables": 0}
    check_syllable_count(seg, count)
    result.issues.extend(issue for issue in seg.issues if issue not in result.issues)
    if alignment == "segments":
        windows = [(item.start_s, item.end_s, None, None) for item in seg.segments]
        stats = _window_stats(windows, method="energy_segments", confidences=[])
        if len(windows) != count:
            result.add_issue("segment_syllable_mismatch",
                             f"{len(windows)} segments vs {count} syllables", "error")
            return None, stats
        return windows, stats
    aligned = align_syllables(audio, sample_rate, count, f0_track=track)
    result.issues.extend(aligned.issues)
    if len(aligned.boundaries) != count:
        result.add_issue("alignment_failed",
                         f"energy DP returned {len(aligned.boundaries)} boundaries for "
                         f"{count} syllables", "error")
        return None, aligned.stats()
    windows = [(item.start_s, item.end_s, item.dip_db, item.confidence)
               for item in aligned.boundaries]
    confidences = [float(item.confidence) for item in aligned.boundaries[1:]
                   if item.confidence is not None]
    stats = _window_stats(windows, method="energy_dp", confidences=confidences)
    stats["dips_db"] = [None if item.dip_db is None else round(item.dip_db, 2)
                        for item in aligned.boundaries]
    return windows, stats


def _readiness(result: PairResult, *, min_boundary_confidence: float,
               max_gap_s: float) -> dict:
    """Decide whether a pair may enter training; never derived from status alone."""
    reasons: list[str] = []
    if result.status != "ok":
        reasons.append(f"status_{result.status}")
    if any(issue.get("severity", "error") == "error" for issue in result.issues):
        reasons.append("error_issues")
    alignment = result.alignment or {}
    method = alignment.get("method")
    if method == "energy_dp":
        minimum = alignment.get("min_confidence")
        if (alignment.get("low_confidence_boundaries") or 0) > 0 or minimum is None \
                or minimum < min_boundary_confidence:
            reasons.append("low_boundary_confidence")
    elif method == "forced_alignment":
        minimum = alignment.get("min_confidence")
        if minimum is None or minimum < min_boundary_confidence:
            reasons.append("low_boundary_confidence")
    elif method == "energy_segments":
        reasons.append("no_boundary_confidence")
    elif method is None:
        reasons.append("no_alignment")
    if method != "manual":
        # An independent energy segmentation is the only cross-check available without an
        # acoustic model. When it counts a different number of bursts than the transcript,
        # the DP is distributing syllables over a region the signal does not support, and a
        # deep valley between two vowels can still be a vowel transition, not a boundary.
        segments = (result.segmentation_stats or {}).get("segments")
        expected = len(result.syllables)
        if segments is not None and expected and segments != expected:
            reasons.append("segmentation_count_disagrees")
    infos = (result.pitch or {}).get("syllables") or []
    if not infos:
        reasons.append("no_pitch_frames")
    else:
        coverage = float(np.mean([item["coverage"] for item in infos]))
        if coverage < 0.5:
            reasons.append("low_pitch_coverage")
        if max(item["longest_gap_s"] for item in infos) > max_gap_s:
            reasons.append("pitch_gap_interpolated")
    codes = {issue["code"] for issue in result.issues}
    for code in ("clipping", "gain_limited_by_peak", "segment_truncated"):
        if code in codes:
            reasons.append(code)
    unique = list(dict.fromkeys(reasons))
    return {"training_ready": not unique, "reasons": unique,
            "min_boundary_confidence": min_boundary_confidence,
            "max_gap_s": max_gap_s}


def build_pair(wav_path: str | Path, pinyins: list[str], *, template: VoiceTemplate,
               backend: Optional[SynthBackend] = None, sample_rate: int = 24000,
               seed: int = 42, max_duration_s: float = 30.0,
               pitch_estimator: str = "viterbi", alignment: str = "segments",
               boundaries_s: Optional[Sequence[tuple[float, float]]] = None,
               phone_spans: Optional[Sequence[tuple]] = None,
               min_boundary_confidence: float = DEFAULT_MIN_BOUNDARY_CONFIDENCE,
               max_pitch_gap_s: float = DEFAULT_MAX_PITCH_GAP_S) -> PairResult:
    """Analyse one real utterance and render its mechanical counterpart when possible.

    pitch_estimator: 'viterbi' (EXP-0015 multi-candidate tracker) or 'legacy' (old YIN).
    alignment: 'segments' keeps the original energy-segment count check; 'dp' places one
        boundary per transcript syllable with the energy-valley dynamic program; 'forced'
        takes phone_spans from the chain-HMM aligner; passing boundaries_s overrides all
        three with operator-specified (start, end) seconds.
    phone_spans: (phone, start, end[, confidence]) per transcript phoneme in order, in the
        time base of the loaded audio. Only valid with alignment='forced'; the phone sequence
        must match the transcript exactly, otherwise the pair is rejected rather than
        repaired.
    """
    if alignment == "forced" and phone_spans is None:
        raise ValueError("alignment='forced' requires phone_spans")
    if phone_spans is not None and boundaries_s is not None:
        raise ValueError("phone_spans and boundaries_s are mutually exclusive")
    if pitch_estimator not in PITCH_ESTIMATORS:
        raise ValueError(f"unknown pitch estimator {pitch_estimator!r}; expected one of "
                         f"{', '.join(PITCH_ESTIMATORS)}")
    if boundaries_s is None and phone_spans is None and alignment not in ALIGNMENT_MODES:
        raise ValueError(f"unknown alignment {alignment!r}; expected one of "
                         f"{', '.join(ALIGNMENT_MODES)}")
    path = Path(wav_path)
    audio, source_rate = read_wav(path)
    result = PairResult(utterance_id=path.stem, status="rejected",
                        sample_rate=sample_rate, source_sample_rate=source_rate,
                        real_sha256=file_sha256(path))

    def finish() -> PairResult:
        result.readiness = _readiness(result, min_boundary_confidence=min_boundary_confidence,
                                      max_gap_s=max_pitch_gap_s)
        return result

    if audio.size / source_rate > max_duration_s:
        result.add_issue("too_long", f"{audio.size / source_rate:.1f}s > {max_duration_s}s",
                         "error")
        return finish()
    if source_rate != sample_rate:
        audio = resample(audio, source_rate, sample_rate)
        result.add_issue("resampled", f"{source_rate} Hz -> {sample_rate} Hz")
    result.real = audio
    result.real_sha256 = _audio_sha256(audio)

    track, result.f0_stats = _analyse_pitch(audio, sample_rate, pitch_estimator)
    result.pitch = {"estimator": result.f0_stats["estimator"],
                    "estimation": pitch_estimator, "max_gap_s": max_pitch_gap_s,
                    "syllables": []}
    if track.voiced_ratio < 0.3:
        result.add_issue("mostly_unvoiced",
                         f"voiced frame ratio {track.voiced_ratio:.4f} < 0.3000", "error")
        return finish()

    synth = backend or get_backend()
    inventory = getattr(synth, "inventory", "vowels")

    def parse_all():
        try:
            parsed = [parse_syllable(token, inventory) for token in pinyins]
        except ValueError as error:
            result.add_issue("bad_pinyin", str(error), "error")
            return None
        result.syllables = [syllable.as_dict() for syllable in parsed]
        blocked = [syllable for syllable in parsed if not syllable.phonemes]
        if blocked:
            unique = sorted({syllable.pinyin for syllable in blocked})
            result.add_issue(
                "backend_inventory_insufficient",
                f"{len(blocked)}/{len(parsed)} syllables cannot be rendered by the backend "
                f"inventory {inventory!r} ({', '.join(unique[:8])}); "
                f"first reason: {blocked[0].reason}",
                "error")
            result.status = "blocked"
            return None
        return parsed

    plans = None
    if alignment == "forced" and phone_spans is not None:
        syllables = parse_all()
        if syllables is None:
            return finish()
        windows, plans, alignment_stats = _span_windows(
            phone_spans, syllables, audio.size / sample_rate, result)
        result.alignment = alignment_stats
        if windows is None:
            return finish()
    else:
        windows, alignment_stats = _resolve_windows(
            audio, sample_rate, len(pinyins), track, alignment=alignment,
            boundaries_s=boundaries_s, result=result)
        result.alignment = alignment_stats
        if windows is None:
            return finish()
        syllables = parse_all()
        if syllables is None:
            return finish()

    # One request per syllable, placed at the onset measured from the real utterance.
    # A single concatenated request cannot express the pauses between syllables
    # (PhonemeSequence has no silence slots), so it produced a shorter signal and the
    # real signal had to be truncated. Per-syllable placement keeps both signals the
    # same length and keeps every pause where it was.
    mechanical = np.zeros_like(audio)
    fingerprints: list[str] = []
    for index, (syllable, window) in enumerate(zip(syllables, windows)):
        start_s, end_s, dip_db, confidence = window
        duration = max(0.01, float(end_s - start_s))
        if plans is not None:
            plan, plan_reason = plans[index], None
        else:
            plan, plan_reason = syllable_plan(syllable.initial, syllable.final, duration,
                                              inventory, erhua=syllable.erhua)
        if plan_reason is not None:
            result.add_issue("syllable_plan_blocked",
                             f"syllable {index} {syllable.pinyin!r}: {plan_reason}", "error")
            result.status = "blocked"
            return finish()
        result.allocations.append([{"phoneme": phoneme, "duration_s": round(seconds, 4)}
                                   for phoneme, seconds in plan])
        pitch, pitch_info = _pitch_curve(track, start_s, duration, index=index,
                                         max_gap_s=max_pitch_gap_s, result=result)
        pitch_info["dip_db"] = None if dip_db is None else round(float(dip_db), 2)
        pitch_info["confidence"] = None if confidence is None else round(float(confidence), 3)
        result.pitch["syllables"].append(pitch_info)
        if pitch is None:
            return finish()
        request = SynthesisRequest(
            template=template,
            phonemes=PhonemeSequence(
                tuple(PhonemeSegment(phoneme=phoneme, duration_s=seconds)
                      for phoneme, seconds in plan), inventory=inventory),
            pitch=pitch,
            # No lead/tail silence: the pauses are already in the real signal, and a 20 ms
            # lead would shift every syllable 20 ms late while a 50 ms tail would overlap
            # the next syllable's onset and cut the previous fade-out.
            timing=Timing(lead_silence_s=0.0, tail_silence_s=0.0),
            sample_rate=sample_rate,
            # Vary the seed per syllable: a shared seed would repeat the same breath-noise
            # draw in every syllable, which the model could learn as a position cue.
            seed=seed + index,
        )
        try:
            request.validate()
        except ValueError as error:
            result.add_issue("invalid_request", str(error), "error")
            return finish()
        rendered = synth.render(request)
        fingerprints.append(rendered.request_fingerprint)
        offset = int(round(start_s * sample_rate))
        stop = offset + rendered.audio.size
        if offset >= mechanical.size:
            result.add_issue("segment_out_of_range",
                             f"syllable {index} at {start_s:.3f}s starts past the signal end",
                             "error")
            return finish()
        if stop > mechanical.size:
            result.add_issue("segment_truncated",
                             f"syllable {index} at {start_s:.3f}s overruns the signal by "
                             f"{stop - mechanical.size} samples and is truncated")
        stop = min(mechanical.size, stop)
        piece = rendered.audio[: stop - offset].astype(np.float64)
        # The synthesizer peak-normalises every render, so placing the syllables as-is erases
        # the loudness contour of the utterance (a syllable 6 dB quieter in the recording came
        # back 6 dB louder). Match each syllable's RMS to the real segment it replaces; the
        # gain is recorded so it can be audited.
        real_rms = float(np.sqrt(np.mean(np.square(
            audio[offset:stop].astype(np.float64))))) if stop > offset else 0.0
        synth_rms = float(np.sqrt(np.mean(np.square(piece)))) if piece.size else 0.0
        gain = 1.0
        if real_rms > 1e-6 and synth_rms > 1e-9:
            gain = float(np.clip(real_rms / synth_rms, 0.05, 20.0))
        # Never let loudness matching push the pair outside [-1, 1]: clipping would add a
        # distortion cue that is neither in the recording nor in the synthesizer, and the
        # dataset consumer cannot tell it from the model's own artefact.
        peak = float(np.max(np.abs(piece))) if piece.size else 0.0
        if gain * peak > 0.999:
            clipped_gain = 0.999 / peak if peak > 0 else gain
            if clipped_gain < gain:
                result.add_issue("gain_limited_by_peak",
                                 f"syllable {index} gain {20.0 * np.log10(gain / clipped_gain):.2f} dB "
                                 "below the RMS match to keep the pair below full scale")
                gain = clipped_gain
        result.gains_db.append(round(float(20.0 * np.log10(gain)), 3) if gain > 0 else 0.0)
        mechanical[offset:stop] = (piece * gain).astype(np.float32)
    result.add_issue("syllablewise_render",
                     f"{len(syllables)} syllables rendered separately at measured onsets; "
                     "each render applies a short fade in/out, so boundaries dip more than in "
                     "real speech, and the onset/nucleus split inside a syllable is the g2p "
                     "nominal-length heuristic, not a forced alignment")
    # The composite fingerprint covers every syllable request; overwriting it with the last
    # syllable's fingerprint (as this function used to) made two different utterances with the
    # same final syllable indistinguishable.
    result.request_fingerprints = fingerprints
    result.request_fingerprint = hashlib.sha256(
        "\n".join(fingerprints).encode("utf-8")).hexdigest()
    result.mechanical = mechanical
    result.mechanical_sha256 = _audio_sha256(mechanical)
    result.template = {"formant_scale": template.formant_scale,
                       "spectral_tilt": template.spectral_tilt,
                       "breathiness": template.breathiness,
                       "vibrato_cents": template.vibrato_cents,
                       "vibrato_hz": template.vibrato_hz}
    result.status = "ok"
    return finish()


def write_pair(out_dir: str | Path, result: PairResult, *, overwrite: bool = False) -> dict:
    """Write real/mechanical WAVs and a JSON record. Refuses to clobber by default."""
    if result.status != "ok" or result.real is None or result.mechanical is None:
        raise ValueError(f"cannot write pair with status {result.status!r}")
    target = Path(out_dir)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite pair directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    write_wav(target / "real.wav", result.real, result.sample_rate)
    write_wav(target / "mechanical.wav", result.mechanical, result.sample_rate)
    record = result.as_dict()
    record["files"] = {"real": "real.wav", "mechanical": "mechanical.wav"}
    (target / "pair.json").write_text(
        __import__("json").dumps(record, ensure_ascii=False, indent=2) + chr(10))
    return record
