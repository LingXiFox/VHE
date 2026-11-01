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
from typing import Optional

import numpy as np

from ..audio import read_wav, write_wav
from ..synth import (PitchCurve, PhonemeSegment, PhonemeSequence, SynthesisRequest,
                     SynthBackend, Timing, VoiceTemplate, get_backend)
from ..synth.contract import F0_MAX_HZ, F0_MIN_HZ
from .mandarin import PinyinError, split_token
from .g2p import syllable_plan, syllable_phonemes
from .aishell3 import file_sha256
from .f0 import F0Track, estimate_f0
from .resample import resample
from .segment import Segmentation, check_syllable_count, segment_by_energy

INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "r", "z", "c", "s", "y", "w")
PAIR_STATUS = ("ok", "blocked", "rejected")


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
    phonemes, reason = syllable_phonemes(token.initial, token.final, inventory,
                                         erhua=token.erhua)
    return Syllable(token.pinyin, token.initial, token.written or token.final, token.tone,
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

    synth = backend or get_backend()
    inventory = getattr(synth, "inventory", "vowels")
    try:
        syllables = [parse_syllable(token, inventory) for token in pinyins]
    except ValueError as error:
        result.add_issue("bad_pinyin", str(error), "error")
        return result
    result.syllables = [syllable.as_dict() for syllable in syllables]
    blocked = [syllable for syllable in syllables if not syllable.phonemes]
    if blocked:
        unique = sorted({syllable.pinyin for syllable in blocked})
        result.add_issue(
            "backend_inventory_insufficient",
            f"{len(blocked)}/{len(syllables)} syllables cannot be rendered by the backend "
            f"inventory {inventory!r} ({', '.join(unique[:8])}); "
            f"first reason: {blocked[0].reason}",
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
    mechanical = np.zeros_like(audio)
    fingerprints: list[str] = []
    for index, (syllable, segment) in enumerate(zip(syllables, seg.segments)):
        duration = max(0.01, float(segment.duration_s))
        plan, plan_reason = syllable_plan(syllable.initial, syllable.final, duration,
                                          inventory, erhua=syllable.erhua)
        if plan_reason is not None:
            result.add_issue("syllable_plan_blocked",
                             f"syllable {index} {syllable.pinyin!r}: {plan_reason}", "error")
            result.status = "blocked"
            return result
        result.allocations.append([{"phoneme": phoneme, "duration_s": round(seconds, 4)}
                                   for phoneme, seconds in plan])
        pitch = _pitch_curve(track, segment.start_s, duration, result)
        if pitch is None:
            return result
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
            return result
        rendered = synth.render(request)
        fingerprints.append(rendered.request_fingerprint)
        offset = int(round(segment.start_s * sample_rate))
        stop = offset + rendered.audio.size
        if offset >= mechanical.size:
            result.add_issue("segment_out_of_range",
                             f"segment at {segment.start_s:.3f}s starts past the signal end",
                             "error")
            return result
        if stop > mechanical.size:
            result.add_issue("segment_truncated",
                             f"syllable {index} at {segment.start_s:.3f}s overruns the signal by "
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
