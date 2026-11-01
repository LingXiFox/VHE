"""Corpus-level forced-alignment index (P2 prerequisite).

What this module is for: turning a light manifest (vhe.data.corpus) plus a trained
chain-HMM aligner into a per-utterance phone-span index that the pairing pipeline can
consume. It is the bridge between "the aligner works" (EXP-0020/0021/0022) and
"build_pair can use real boundaries instead of g2p nominal durations".

Design rules, all of them load-bearing:

* Spans are written in the *file* time base, not the cropped-speech time base. The gate
  that trims leading/trailing silence records its offset, and every span is shifted by it,
  so a consumer never has to know the gate existed.
* Failures are indexed, not filtered. An utterance the aligner cannot handle (unrenderable
  transcript, no voiced speech, too few frames) becomes a 'skipped' record with a reason,
  so a downstream count of usable pairs is auditable and a silent 30% loss cannot happen.
* The index refuses to overwrite. Re-running an alignment is a new file; that matches the
  experiment policy in AGENTS.md.
* No audio is written. The index holds paths, spans and confidences only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from ..audio import read_wav
from .aligner import AlignmentError, PhoneAligner, PreparedUtterance, prepare_utterance

INDEX_VERSION = 1


@dataclass(frozen=True)
class AlignedUtterance:
    """One aligned utterance: phone spans in the file's own time base."""

    utt_id: str
    speaker: str
    split: str
    wav_path: str
    #: (phone, start_s, end_s, confidence) in file time, in transcript order.
    spans: tuple[tuple[str, float, float, float], ...]
    offset_s: float
    frames: int
    logprob: float
    confidence: float
    duration_s: float

    def as_dict(self) -> dict:
        return {
            "type": "utterance", "utt_id": self.utt_id, "speaker": self.speaker,
            "split": self.split, "wav_path": self.wav_path,
            "spans": [[phone, round(start, 4), round(stop, 4), round(value, 4)]
                      for phone, start, stop, value in self.spans],
            "offset_s": round(self.offset_s, 4), "frames": self.frames,
            "logprob": round(self.logprob, 3), "confidence": round(self.confidence, 4),
            "duration_s": round(self.duration_s, 4),
        }


def read_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    """Split a light manifest into (meta, rows); the meta record is the first JSONL line."""
    meta: dict = {}
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "meta" or (not rows and "root" in record
                                                and "utt_id" not in record):
                meta = record
                continue
            rows.append(record)
    return meta, rows


def resolve_root(meta: dict, root: Optional[str | Path]) -> Path:
    """Corpus root: explicit argument first, then the manifest's recorded root."""
    candidate = Path(root) if root is not None else Path(str(meta.get("root", "")))
    if not str(candidate) or not candidate.exists():
        raise FileNotFoundError(
            f"corpus root {str(candidate)!r} does not exist; pass root= for a manifest "
            "built on another machine")
    return candidate


def select_rows(rows: Iterable[dict], *, splits: Sequence[str] = ("train",),
                speakers: Optional[int] = None, per_speaker: int = 1,
                seed: int = 42, renderable_only: bool = True) -> list[dict]:
    """Pick rows deterministically: whole speakers, a fixed number of rows per speaker.

    Speakers are chosen, not utterances, so a training subset never mixes a speaker's
    utterances across arms of an ablation.

    'renderable_only' drops utterances the current inventory cannot spell *before*
    sampling, because a blocked utterance never reaches the aligner: sampling first
    would silently shrink the subset (EXP-0023: 320 rows requested, 21 prepared, 93%
    of the loss came from the five units the inventory lacks). Rows from manifests that
    predate the 'renderable' column are kept.
    """
    wanted = set(splits)
    pool = [row for row in rows if row.get("split") in wanted]
    if renderable_only:
        pool = [row for row in pool if row.get("renderable", True)]
    by_speaker: dict[str, list[dict]] = {}
    for row in pool:
        by_speaker.setdefault(str(row.get("speaker")), []).append(row)
    names = sorted(by_speaker)
    if speakers is not None:
        if speakers > len(names):
            hint = " with renderable utterances" if renderable_only else ""
            raise ValueError(f"{speakers} speakers requested, manifest has {len(names)} "
                             f"speaker(s){hint} in splits {sorted(wanted)}")
        rng = np.random.default_rng(seed)
        names = sorted(rng.choice(names, size=speakers, replace=False).tolist())
    picked: list[dict] = []
    for name in names:
        group = sorted(by_speaker[name], key=lambda row: str(row.get("utt_id")))
        rng = np.random.default_rng([seed, int(hashlib.sha256(name.encode()).hexdigest()[:8],
                                               16)])
        indices = rng.permutation(len(group))[:per_speaker]
        picked.extend(group[int(index)] for index in sorted(indices))
    return picked


def build_training_items(rows: Iterable[dict], root: Path, *,
                         on_skip=None) -> list[PreparedUtterance]:
    """Prepare aligner training items, counting every skipped utterance with its reason."""
    items: list[PreparedUtterance] = []
    for row in rows:
        path = root / str(row.get("wav_path"))
        try:
            items.append(prepare_utterance(path, [str(token) for token in row["pinyin"]]))
        except (AlignmentError, OSError, ValueError, KeyError) as error:
            if on_skip is not None:
                on_skip({"type": "skipped", "utt_id": row.get("utt_id"),
                         "wav_path": str(row.get("wav_path")),
                         "reason": f"{type(error).__name__}: {error}"})
    return items


def align_rows(aligner: PhoneAligner, rows: Iterable[dict], root: Path, *,
               on_skip=None) -> Iterator[object]:
    """Yield AlignedUtterance or a skip record; never raise on a single bad utterance."""
    for row in rows:
        utt_id = str(row.get("utt_id"))
        wav_path = root / str(row.get("wav_path"))
        pinyins = [str(token) for token in row.get("pinyin", ())]
        try:
            audio, sample_rate = read_wav(wav_path)
            result = aligner.align(audio, sample_rate, pinyins)
        except (AlignmentError, OSError, ValueError) as error:
            record = {"type": "skipped", "utt_id": utt_id, "speaker": row.get("speaker"),
                      "split": row.get("split"), "wav_path": str(row.get("wav_path")),
                      "reason": f"{type(error).__name__}: {error}"}
            if on_skip is not None:
                on_skip(record)
            yield record
            continue
        offset = float(result.offset_s)
        spans = tuple((span.phoneme, span.start_s + offset, span.end_s + offset,
                       float(span.confidence)) for span in result.spans)
        yield AlignedUtterance(
            utt_id=utt_id, speaker=str(row.get("speaker")), split=str(row.get("split")),
            wav_path=str(row.get("wav_path")), spans=spans, offset_s=offset,
            frames=int(result.frames), logprob=float(result.logprob),
            confidence=float(result.confidence),
            duration_s=float(audio.size) / float(sample_rate))


def write_index(path: str | Path, entries: Iterable[dict], *, meta: dict) -> Path:
    """Write the JSONL index; refuses to overwrite an existing file."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target} exists; alignment runs are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = {"utterance": 0, "skipped": 0}
    with open(target, "w", encoding="utf-8") as handle:
        header = dict(meta, type="meta", index_version=INDEX_VERSION)
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count[entry["type"]] = count.get(entry["type"], 0) + 1
    return target


def load_index(path: str | Path) -> tuple[dict, dict[str, list[tuple[str, float, float,
                                                                     float]]]]:
    """Read an index into (meta, {utt_id: [(phone, start, end, confidence), ...]})."""
    meta: dict = {}
    spans: dict[str, list[tuple[str, float, float, float]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "meta":
                meta = record
            elif record.get("type") == "utterance":
                spans[str(record["utt_id"])] = [
                    (str(phone), float(start), float(stop), float(value))
                    for phone, start, stop, value in record["spans"]]
    return meta, spans


def index_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
