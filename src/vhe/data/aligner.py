"""Corpus-trained monophone forced alignment (P2 prerequisite).

What this is: one speaker-independent chain HMM over the 36 phonemes the mechanical backend can
render, trained by flat start + Baum-Welch on the authorised corpus itself. Given a waveform and
the transcript's pinyin it returns one boundary per phoneme — including where inside a syllable
the onset stops and the nucleus starts, which the g2p plan can only guess from nominal durations.

What this is not: an ASR system. No language model, no triphone tying, no speaker adaptation. The
aligner only has to *place boundaries* for a transcript it is handed; recognition accuracy is not
a goal, and nothing it produces is ever fed to VHE as an input. Its output is a timing annotation
used to build (mechanical, real) pairs.

Silence is not modelled by the HMM. Leading and trailing silence come from the shared energy gate
(`segment.frame_rms` + `segment.segmentation_threshold`) and the chain is run over the speech
region only: an utterance-level Gaussian for silence drifts with per-utterance CMVN and, measured
on synthetic data, lost the leading 150 ms to the first phone. Gate + crop is deterministic and
reports what it did (`silence_frames`, `offset_s`) instead of hiding it in a posterior.

Cost: training is CPU NumPy, ~4 Baum-Welch passes over a few hundred utterances, seconds to a
couple of minutes. Default emissions are one diagonal Gaussian per state: 8-component GMMs on
this much data overfit and drove variances to the floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..audio import read_wav
from ..synth.phonology import MANDARIN_PHONEME_IDS
from .features import HOP_S, SAMPLE_RATE, acoustic_features
from .hmm import ChainHmm, HmmAccumulator, kmeans_init
from .resample import resample
from .segment import frame_rms, segmentation_threshold

SILENCE = "sil"
DEFAULT_STATES_PER_PHONE = 3
DEFAULT_COMPONENTS = 1
DEFAULT_ITERATIONS = 4
DEFAULT_FRAME_CAP = 2500

PHONES: tuple[str, ...] = tuple(MANDARIN_PHONEME_IDS)


class AlignmentError(RuntimeError):
    """Raised when an utterance cannot be aligned; carries the reason, never drops silently."""


@dataclass(frozen=True)
class PhoneSpan:
    phoneme: str
    start_s: float
    end_s: float
    frames: int
    #: Mean posterior of the Viterbi state over this span. Low means the path is a guess.
    confidence: float

    def as_dict(self) -> dict:
        return {"phoneme": self.phoneme, "start_s": round(self.start_s, 4),
                "end_s": round(self.end_s, 4), "frames": self.frames,
                "confidence": round(self.confidence, 4)}


@dataclass(frozen=True)
class AlignmentResult:
    phones: tuple[str, ...]
    logprob: float
    frames: int
    hop_s: float
    spans: tuple[PhoneSpan, ...]
    offset_s: float = 0.0
    silence_frames: tuple[int, int] = (0, 0)

    @property
    def boundary_s(self) -> tuple[float, ...]:
        """Internal phone boundaries in seconds from the start of the audio."""
        return tuple(span.start_s for span in self.spans[1:])

    @property
    def confidence(self) -> float:
        """Worst span posterior: the number a readiness rule should use."""
        return float(min((span.confidence for span in self.spans), default=0.0))

    @property
    def speech_s(self) -> tuple[float, float]:
        """(start, end) of the region the chain was run over, in audio time."""
        return (self.offset_s, self.offset_s + self.frames * self.hop_s)

    def as_dict(self) -> dict:
        return {"phones": list(self.phones), "logprob": round(self.logprob, 4),
                "frames": self.frames, "hop_s": self.hop_s,
                "offset_s": round(self.offset_s, 4),
                "confidence": round(self.confidence, 4),
                "silence_frames": list(self.silence_frames),
                "spans": [span.as_dict() for span in self.spans]}


def pinyin_phones(pinyins: Sequence[str]) -> tuple[tuple[str, ...], Optional[str]]:
    """Flatten a transcript into phoneme ids, or return the first blocking reason.

    Uses the same resolver as rendering, so an utterance the backend refuses to render is not
    aligned either — there is no point timing a phoneme the synthesizer will not produce.
    """
    from .pairing import parse_syllable  # local import keeps the module graph acyclic

    phones: list[str] = []
    for token in pinyins:
        syllable = parse_syllable(str(token), "mandarin")
        if syllable.reason is not None:
            return (), f"pinyin {token!r}: {syllable.reason}"
        if not syllable.phonemes:
            return (), f"pinyin {token!r} resolves to no phonemes"
        phones.extend(syllable.phonemes)
    return tuple(phones), None


def silence_frame_edges(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                        min_run_frames: int = 3) -> tuple[int, int]:
    """(leading, trailing) silence frames on the feature grid, under the shared energy gate."""
    rms, _hop = frame_rms(audio, sample_rate, frame_ms=1000.0 * HOP_S * 2.5,
                          hop_ms=1000.0 * HOP_S)
    threshold = segmentation_threshold(rms)
    quiet = rms <= threshold
    lead = 0
    while lead < quiet.size and quiet[lead]:
        lead += 1
    tail = 0
    while tail < quiet.size and quiet[quiet.size - 1 - tail]:
        tail += 1
    if lead < min_run_frames:
        lead = 0
    if tail < min_run_frames:
        tail = 0
    if lead + tail >= quiet.size:
        lead, tail = 0, 0
    return int(lead), int(tail)


def crop_speech(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Trim leading/trailing silence frames; returns the crop and the frames removed."""
    lead, tail = silence_frame_edges(audio, sample_rate)
    hop = int(round(HOP_S * sample_rate))
    start = lead * hop
    stop = audio.size - tail * hop
    if stop - start < hop:
        return audio, (0, 0)
    return audio[start:stop], (lead, tail)


@dataclass
class PreparedUtterance:
    utterance_id: str
    features: np.ndarray
    phones: tuple[str, ...]
    silence: tuple[int, int]
    duration_s: float
    offset_s: float = 0.0


def prepare_utterance(wav_path: str | Path, pinyins: Sequence[str]) -> PreparedUtterance:
    """Read one recording and turn it into aligner input; raises instead of guessing."""
    phones, reason = pinyin_phones(pinyins)
    if reason is not None:
        raise AlignmentError(reason)
    audio, source_rate = read_wav(wav_path)
    if source_rate != SAMPLE_RATE:
        audio = resample(audio, source_rate, SAMPLE_RATE)
    duration = audio.size / SAMPLE_RATE
    speech, (lead, tail) = crop_speech(audio, SAMPLE_RATE)
    features = acoustic_features(speech, SAMPLE_RATE)
    if features.shape[0] < 1:
        raise AlignmentError("the whole utterance sits under the silence gate")
    return PreparedUtterance(utterance_id=Path(wav_path).stem, features=features, phones=phones,
                             silence=(lead, tail), duration_s=duration,
                             offset_s=lead * HOP_S)


def _state_layout(phones: Sequence[str], states_per_phone: int) -> dict[str, tuple[int, int]]:
    """One block per distinct phoneme, in first-appearance order.

    A transcript that repeats a phoneme (是是) reuses the same block: flat start pools frames by
    phoneme, not by position. Building a block per occurrence and overwriting the key would leave
    the earlier blocks with no frames and shift every later offset.
    """
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for phone in phones:
        if phone in layout:
            continue
        layout[phone] = (offset, states_per_phone)
        offset += states_per_phone
    return layout


def _flat_start_segmentation(item: PreparedUtterance, layout: dict[str, tuple[int, int]],
                             total_states: int) -> np.ndarray:
    """Assign every frame to a chain state without using any acoustics.

    Frames are split across phone blocks in proportion to their state counts, and each block is
    split evenly across its own states. Ids are *vocabulary* state ids, not per-utterance ones:
    flat start pools frames across utterances by state id.
    """
    frames = item.features.shape[0]
    blocks = [(phone, layout[phone][1]) for phone in item.phones]
    speech_states = sum(count for _phone, count in blocks)
    if speech_states == 0:
        raise AlignmentError("no states to fill")
    if frames < speech_states:
        raise AlignmentError(f"{frames} frames cannot hold {speech_states} states")
    assigned = [max(1, int(round(frames * count / speech_states))) for _p, count in blocks]
    drift = frames - sum(assigned)
    if drift:
        assigned[-1] = max(1, assigned[-1] + drift)
    if sum(assigned) > frames:
        raise AlignmentError(f"{frames} frames cannot hold {sum(assigned)} phone frames")
    per_frame = np.zeros(frames, dtype=np.int64)
    cursor = 0
    for (phone, count), frames_for_phone in zip(blocks, assigned):
        offset, _states = layout[phone]
        for state in range(count):
            start = cursor + int(round(frames_for_phone * state / count))
            stop = cursor + int(round(frames_for_phone * (state + 1) / count))
            per_frame[start:stop] = offset + state
        cursor += frames_for_phone
    return per_frame


def _initial_hmm(items: Sequence[PreparedUtterance], *, phones: Sequence[str],
                 states_per_phone: int, n_components: int, seed: int,
                 frame_cap: int = DEFAULT_FRAME_CAP) -> ChainHmm:
    rng = np.random.default_rng(seed)
    layout = _state_layout(phones, states_per_phone)
    n_states = len(layout) * states_per_phone
    dim = items[0].features.shape[1]
    samples: list[list[np.ndarray]] = [[] for _ in range(n_states)]
    for item in items:
        assignment = _flat_start_segmentation(item, layout, n_states)
        for state in range(n_states):
            picked = item.features[assignment == state]
            if picked.size:
                samples[state].append(picked)
    global_frames = np.concatenate([chunk for group in samples for chunk in group])
    global_mean = global_frames.mean(axis=0)
    global_var = np.maximum(global_frames.var(axis=0), 1e-3)
    weights = np.zeros((n_states, n_components))
    means = np.zeros((n_states, n_components, dim))
    variances = np.zeros((n_states, n_components, dim))
    for state, chunks in enumerate(samples):
        frames = np.concatenate(chunks) if chunks else global_mean[None, :]
        if frames.shape[0] > frame_cap:
            step = int(np.ceil(frames.shape[0] / frame_cap))
            frames = frames[::step][:frame_cap]
        if frames.shape[0] < n_components * 2:
            weights[state] = 1.0 / n_components
            means[state] = global_mean + rng.normal(0.0, 0.01, size=(n_components, dim))
            variances[state] = global_var
            continue
        weights[state], means[state], variances[state] = kmeans_init(
            frames, n_components, rng)
    return ChainHmm(means=means, variances=variances, weights=weights,
                    global_variance=global_var)


@dataclass
class PhoneAligner:
    """A trained monophone chain HMM plus the bookkeeping needed to align one utterance."""

    hmm: ChainHmm
    states_per_phone: int = DEFAULT_STATES_PER_PHONE
    sample_rate: int = SAMPLE_RATE
    hop_s: float = HOP_S
    phones: tuple[str, ...] = PHONES
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layout = _state_layout(self.phones, self.states_per_phone)

    # ------------------------------------------------------------------ inference

    def chain(self, phones: Sequence[str]) -> tuple[list[int], list[tuple[str, int]]]:
        """State indices used by one utterance plus (phone, state_count) in order."""
        indices: list[int] = []
        blocks: list[tuple[str, int]] = []
        for phone in phones:
            if phone not in self.layout:
                raise AlignmentError(f"phone {phone!r} is not in the aligner vocabulary")
            offset, count = self.layout[phone]
            indices.extend(range(offset, offset + count))
            blocks.append((phone, count))
        return indices, blocks

    def align_phones(self, features: np.ndarray, phones: Sequence[str],
                     offset_s: float = 0.0) -> AlignmentResult:
        """Viterbi-align a feature matrix against an explicit phone sequence."""
        sequence = tuple(phones)
        indices, blocks = self.chain(sequence)
        if features.shape[0] < len(indices):
            raise AlignmentError(f"{features.shape[0]} frames cannot cover {len(indices)} states "
                                 f"({len(sequence)} phones)")
        sub = self.hmm.subset(indices)
        state_ll = sub.state_loglik(features)
        logprob, path = sub.viterbi(state_ll)
        _total, log_posterior = sub.occupancy(state_ll)
        posterior = np.exp(log_posterior)
        spans: list[PhoneSpan] = []
        frame = 0
        first_state = 0
        # The path is non-decreasing and visits every state, so each phone owns exactly the
        # frames whose state falls in its block.
        for phone, count in blocks:
            start = frame
            confidences: list[float] = []
            while frame < path.size and path[frame] < first_state + count:
                confidences.append(posterior[frame, path[frame]])
                frame += 1
            spans.append(PhoneSpan(
                phoneme=phone,
                start_s=offset_s + start * self.hop_s,
                end_s=offset_s + frame * self.hop_s,
                frames=frame - start,
                confidence=float(np.mean(confidences)) if confidences else 0.0))
            first_state += count
        return AlignmentResult(phones=sequence, logprob=float(logprob), frames=int(path.size),
                               hop_s=self.hop_s, spans=tuple(spans), offset_s=offset_s)

    def align(self, audio: np.ndarray, sample_rate: int, pinyins: Sequence[str]
              ) -> AlignmentResult:
        """Align one recording against its transcript, trimming silence by the energy gate."""
        phones, reason = pinyin_phones(pinyins)
        if reason is not None:
            raise AlignmentError(reason)
        signal = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            signal = resample(signal, sample_rate, self.sample_rate)
        speech, (lead, tail) = crop_speech(signal, self.sample_rate)
        features = acoustic_features(speech, self.sample_rate)
        result = self.align_phones(features, phones, offset_s=lead * self.hop_s)
        return AlignmentResult(phones=result.phones, logprob=result.logprob,
                               frames=result.frames, hop_s=result.hop_s, spans=result.spans,
                               offset_s=result.offset_s, silence_frames=(lead, tail))

    # ------------------------------------------------------------------ persistence

    def to_npz(self, path: str | Path, *, meta: dict | None = None) -> Path:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        payload = {f"hmm_{key}": value for key, value in self.hmm.to_npz().items()}
        payload["meta"] = np.asarray(json.dumps({
            "phones": list(self.phones), "states_per_phone": self.states_per_phone,
            "sample_rate": self.sample_rate, "hop_s": self.hop_s,
            "n_components": self.hmm.n_components, "dim": self.hmm.dim,
            **(meta or {}),
        }))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        return target

    @classmethod
    def from_npz(cls, path: str | Path) -> "PhoneAligner":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            hmm = ChainHmm.from_npz({key[4:]: data[key] for key in data.files
                                     if key.startswith("hmm_")})
        return cls(hmm=hmm, states_per_phone=int(meta["states_per_phone"]),
                   sample_rate=int(meta["sample_rate"]), hop_s=float(meta["hop_s"]),
                   phones=tuple(meta["phones"]), meta=meta)


def train_aligner(items: Sequence[PreparedUtterance], *, phones: Sequence[str] = PHONES,
                  states_per_phone: int = DEFAULT_STATES_PER_PHONE,
                  n_components: int = DEFAULT_COMPONENTS,
                  iterations: int = DEFAULT_ITERATIONS, seed: int = 42,
                  frame_cap: int = DEFAULT_FRAME_CAP) -> tuple[PhoneAligner, dict]:
    """Flat start plus Baum-Welch; returns the aligner and a per-iteration report."""
    if not items:
        raise ValueError("train_aligner needs at least one prepared utterance")
    usable: list[PreparedUtterance] = []
    skipped: list[dict] = []
    for item in items:
        needed = states_per_phone * len(item.phones)
        if item.features.shape[0] < needed:
            skipped.append({"utterance_id": item.utterance_id,
                            "frames": int(item.features.shape[0]), "needed": needed})
        else:
            usable.append(item)
    if not usable:
        raise AlignmentError(f"no utterance has enough frames for its phone count: "
                             f"{len(skipped)} skipped")
    items = usable
    hmm = _initial_hmm(items, phones=phones, states_per_phone=states_per_phone,
                       n_components=n_components, seed=seed, frame_cap=frame_cap)
    aligner = PhoneAligner(hmm=hmm, states_per_phone=states_per_phone, phones=tuple(phones))
    history: list[dict] = []
    # iterations=0 keeps the flat-start parameters, which is the baseline an ablation needs.
    for iteration in range(max(0, iterations)):
        accumulator = HmmAccumulator(hmm.n_states, hmm.n_components, hmm.dim)
        for item in items:
            indices, _blocks = aligner.chain(item.phones)
            accumulator.add(hmm, item.features, indices)
        history.append({"iteration": iteration, "loglik": round(accumulator.loglik, 4),
                        "frames": accumulator.frames,
                        "loglik_per_frame": round(accumulator.loglik
                                                  / max(1, accumulator.frames), 5)})
        hmm = hmm.update(accumulator.weights_sum, accumulator.mean_sum,
                         accumulator.var_sum, accumulator.occupancy_sum)
        aligner = PhoneAligner(hmm=hmm, states_per_phone=states_per_phone,
                               phones=tuple(phones))
    report = {"utterances": len(items), "skipped": skipped, "states": hmm.n_states,
              "components": hmm.n_components, "dim": hmm.dim,
              "states_per_phone": states_per_phone, "seed": seed,
              "frames": history[-1]["frames"] if history else 0,
              "hours": round(sum(item.duration_s for item in items) / 3600.0, 4),
              "history": history,
              "monotone": all(second["loglik"] >= first["loglik"] - 1e-6
                              for first, second in zip(history, history[1:]))}
    return aligner, report
