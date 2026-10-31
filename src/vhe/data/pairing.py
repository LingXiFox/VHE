"""Build (x_mechanical, x_real) pairs from one real utterance.

The pipeline is deliberately strict about what it can render: the only synthesizer
backend in this repository speaks the phoneme inventory {a, i, u} without consonants, so
a Mandarin syllable such as "guang3" cannot be reproduced without silently deleting its
onset and coda. Such a syllable BLOCKS the pair instead of being approximated; the
blocked reason is part of the record.

Timing fidelity: each syllable is rendered on its own and placed at the onset measured
from the real utterance, so the two signals share a length and share the pauses. The cost
is a 30 ms fade at every syllable edge, which the pair record reports as an issue.

Nothing here is committed to Git: real recordings never enter the repository, and the
mechanical signal is derived from the real one only through F0/timing analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..audio import read_wav, write_wav
from ..synth import (PHONEME_INVENTORY, PitchCurve, PhonemeSegment, PhonemeSequence,
                     SynthesisRequest, SynthBackend, Timing, VoiceTemplate, get_backend)
from ..synth.contract import F0_MAX_HZ, F0_MIN_HZ, MAX_SILENCE_S
from .aishell3 import file_sha256
from .f0 import F0Track, estimate_f0
from .resample import resample
from .segment import Segmentation, Segment, check_syllable_count, segment_by_energy

INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "r", "z", "c", "s", "y", "w")
PAIR_STATUS = ("ok", "blocked", "rejected")


@dataclass(frozen=True)
class Syllable:
    pinyin: str
    initial: str
    final: str
    tone: int
    phoneme: Optional[str]
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {"pinyin": self.pinyin, "initial": self.initial, "final": self.final,
                "tone": self.tone, "phoneme": self.phoneme, "reason": self.reason}


def parse_syllable(pinyin: str) -> Syllable:
    """Split an AISHELL-3 token like "guang3" into initial/final/tone (no fuzzy matching)."""
    token = pinyin.strip().lower()
    if not token or token[-1] not in "12345":
        raise ValueError(f"pinyin token must end with a tone digit 1-5: {pinyin!r}")
    tone = int(token[-1])
    body = token[:-1]
    if not body or not body.isalpha():
        raise ValueError(f"pinyin body must be alphabetic: {pinyin!r}")
    initial = next((item for item in INITIALS if body.startswith(item)), "")
    final = body[len(initial):]
    if not final:
        raise ValueError(f"pinyin token has no final: {pinyin!r}")
    if initial == "" and final in PHONEME_INVENTORY:
        return Syllable(pinyin, initial, final, tone, final)
    missing = "onset" if initial else "final"
    return Syllable(pinyin, initial, final, tone, None,
                    f"{missing} {initial or final!r} not in backend inventory "
                    f"{list(PHONEME_INVENTORY)}")


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
    real_sha256: Optional[str] = None
    mechanical_sha256: Optional[str] = None
    template: Optional[dict] = None

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
            "template": self.template,
            "f0_stats": self.f0_stats,
            "segmentation_stats": self.segmentation_stats,
            "syllables": self.syllables,
            "issues": self.issues,
        }


def _audio_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def _pitch_curve(track: F0Track, start_s: float, voiced_s: float,
                 result: PairResult) -> Optional[PitchCurve]:
    times, f0 = track.contour()
    inside = (times >= start_s) & (times <= start_s + voiced_s) & (f0 > 0)
    if int(inside.sum()) < 2:
        result.add_issue("insufficient_pitch_points",
                         f"only {int(inside.sum())} voiced F0 frames inside the voiced region",
                         "error")
        return None
    t = np.asarray(times[inside]) - start_s
    values = np.clip(np.asarray(f0[inside]), F0_MIN_HZ, F0_MAX_HZ)
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t, values = t[keep], values[keep]
    if t[-1] < voiced_s:
        t = np.append(t, voiced_s)
        values = np.append(values, values[-1])
    return PitchCurve(times_s=tuple(float(v) for v in t), f0_hz=tuple(float(v) for v in values))


def _timing(first: Segment, last: Segment, total_s: float, result: PairResult) -> Timing:
    lead = max(0.0, first.start_s)
    tail = max(0.0, total_s - last.end_s)
    if lead > MAX_SILENCE_S or tail > MAX_SILENCE_S:
        result.add_issue("silence_clipped",
                         f"lead {lead:.3f}s / tail {tail:.3f}s exceeds {MAX_SILENCE_S}s; clamped")
        lead, tail = min(lead, MAX_SILENCE_S), min(tail, MAX_SILENCE_S)
    return Timing(lead_silence_s=lead, tail_silence_s=tail)


def build_pair(wav_path: str | Path, pinyins: list[str], *, template: VoiceTemplate,
               backend: Optional[SynthBackend] = None, sample_rate: int = 24000,
               seed: int = 42, max_duration_s: float = 30.0) -> PairResult:
    """Analyse one real utterance and render its mechanical counterpart when possible."""
    path = Path(wav_path)
    audio, source_rate = read_wav(path)
    if audio.size / source_rate > max_duration_s:
        return PairResult(utterance_id=path.stem, status="rejected",
                          sample_rate=sample_rate, source_sample_rate=source_rate,
                          issues=[{"code": "too_long",
                                   "detail": f"{audio.size / source_rate:.1f}s > {max_duration_s}s",
                                   "severity": "error"}],
                          real_sha256=file_sha256(path))
    result = PairResult(utterance_id=path.stem, status="rejected",
                        sample_rate=sample_rate, source_sample_rate=source_rate,
                        real_sha256=file_sha256(path))
    if source_rate != sample_rate:
        audio = resample(audio, source_rate, sample_rate)
        result.add_issue("resampled", f"{source_rate} Hz -> {sample_rate} Hz")
    result.real = audio
    result.real_sha256 = _audio_sha256(audio)

    track = estimate_f0(audio, sample_rate)
    result.f0_stats = track.stats()
    if track.voiced_ratio < 0.3:
        result.add_issue("mostly_unvoiced",
                         f"voiced frame ratio {track.voiced_ratio:.2f} < 0.30", "error")
        return result

    seg = segment_by_energy(audio, sample_rate, f0_track=track)
    result.segmentation_stats = seg.stats()
    result.issues.extend(seg.issues)
    if not seg.segments:
        result.add_issue("no_segments", "energy segmentation produced no syllables", "error")
        return result
    check_syllable_count(seg, len(pinyins))
    result.issues.extend(issue for issue in seg.issues if issue not in result.issues)

    try:
        syllables = [parse_syllable(token) for token in pinyins]
    except ValueError as error:
        result.add_issue("bad_pinyin", str(error), "error")
        return result
    result.syllables = [syllable.as_dict() for syllable in syllables]
    blocked = [syllable for syllable in syllables if syllable.phoneme is None]
    if blocked:
        unique = sorted({syllable.pinyin for syllable in blocked})
        result.add_issue(
            "backend_inventory_insufficient",
            f"{len(blocked)}/{len(syllables)} syllables cannot be rendered by the backend "
            f"({', '.join(unique[:8])}); consonants/diphthongs would be silently dropped",
            "error")
        result.status = "blocked"
        return result
    if len(seg.segments) != len(syllables):
        result.add_issue("segment_syllable_mismatch",
                         f"{len(seg.segments)} segments vs {len(syllables)} syllables", "error")
        return result

    # One request per syllable, placed at the onset measured from the real utterance.
    # A single concatenated request cannot express the pauses between syllables
    # (PhonemeSequence has no silence slots), so it produced a shorter signal and the
    # real signal had to be truncated. Per-syllable placement keeps both signals the
    # same length and keeps every pause where it was.
    synth = backend or get_backend()
    mechanical = np.zeros_like(audio)
    fingerprints: list[str] = []
    for syllable, segment in zip(syllables, seg.segments):
        duration = max(0.01, float(segment.duration_s))
        pitch = _pitch_curve(track, segment.start_s, duration, result)
        if pitch is None:
            return result
        request = SynthesisRequest(
            template=template,
            phonemes=PhonemeSequence((PhonemeSegment(phoneme=syllable.phoneme,
                                                     duration_s=duration),)),
            pitch=pitch,
            timing=Timing(),
            sample_rate=sample_rate,
            seed=seed,
        )
        try:
            request.validate()
        except ValueError as error:
            result.add_issue("invalid_request", str(error), "error")
            return result
        rendered = synth.render(request)
        fingerprints.append(rendered.request_fingerprint)
        offset = int(round(segment.start_s * sample_rate))
        stop = min(mechanical.size, offset + rendered.audio.size)
        if stop <= offset:
            result.add_issue("segment_out_of_range",
                             f"segment at {segment.start_s:.3f}s starts past the signal end",
                             "error")
            return result
        mechanical[offset:stop] = rendered.audio[: stop - offset]
    result.add_issue("syllablewise_render",
                     f"{len(syllables)} syllables rendered separately at measured onsets; "
                     "each render applies a 30 ms fade, so boundaries dip more than in real speech")
    result.request_fingerprint = hashlib.sha256(
        "\n".join(fingerprints).encode("utf-8")).hexdigest()
    result.mechanical = mechanical
    result.mechanical_sha256 = _audio_sha256(mechanical)
    result.request_fingerprint = rendered.request_fingerprint
    result.template = {"formant_scale": template.formant_scale,
                       "spectral_tilt": template.spectral_tilt,
                       "breathiness": template.breathiness,
                       "vibrato_cents": template.vibrato_cents,
                       "vibrato_hz": template.vibrato_hz}
    result.status = "ok"
    return result


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
