"""Corpus-level scan of AISHELL-3 annotations (P1.3).

What this module does, and what it deliberately does not do:

* It reads the text annotation (content.txt) and the WAV *header* of every utterance
  (channels, sample rate, width, frames). It never decodes audio, never resamples, never
  estimates F0 and never writes a mechanical signal. A full-corpus pass therefore costs
  seconds of CPU instead of becoming a hidden preprocessing job.
* It resolves every pinyin token through vhe.data.mandarin and records, per utterance,
  the onsets, the underlying finals, the tones and the erhua flags. A syllable the
  26-phoneme inventory cannot render is counted with the *units* it is missing
  ('final:iang', 'onset:h', 'erhua'), never repaired.
* It writes one JSONL line per utterance: the light manifest that later stages stream
  from. No audio bytes, no absolute path outside the corpus root.

The split stored here is the speaker-grouped one from vhe.data.assign_speaker_split. The
official directory ('train'/'test') is kept in source_dir because it is corpus provenance,
and -- because it splits sentences, not speakers -- it is never used as the training split.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import wave
from typing import Callable

from .aishell3 import (
    NATIVE_CHANNELS,
    NATIVE_SAMPLE_RATE,
    NATIVE_SAMPLE_WIDTH,
    ParseIssue,
    SPLIT_NAMES,
    SPLIT_SEED,
    assign_speaker_split,
    auto_split_sizes,
    discover_splits,
    parse_content_line,
    parse_spk_info,
    read_wav_header,
    speaker_of,
)
from .g2p import SUPPORTED_FINALS, SUPPORTED_ONSETS
from .mandarin import FINAL_SEGMENTS, PinyinError, Token, missing_units, split_token
from ..synth.contract import INVENTORIES

#: v2 adds per-utterance quality_flags (annotation conflicts); v3 adds the inventory
#: fingerprint in the meta record and re-scores 'renderable'. Older manifests still load.
MANIFEST_VERSION = 3


@dataclass(frozen=True)
class CorpusEntry:
    """One accepted utterance: metadata plus the resolved syllable inventory."""

    utt_id: str
    speaker: str
    split: str
    source_dir: str
    wav_path: str
    text: str
    pinyin: tuple[str, ...]
    onsets: tuple[str, ...]
    finals: tuple[str, ...]
    tones: tuple[int, ...]
    erhua: tuple[bool, ...]
    sample_rate: int
    n_samples: int
    duration_s: float
    renderable: bool
    #: Annotation quality flags, e.g. 'prosody_conflict'. Empty for a clean utterance.
    quality_flags: tuple[str, ...] = ()

    @property
    def segments(self) -> tuple[str, ...]:
        """Phonemic segments (medial + nucleus + coda), for future inventory work."""
        phones: list[str] = []
        for onset, final in zip(self.onsets, self.finals):
            if onset:
                phones.append(onset)
            phones.extend(FINAL_SEGMENTS[final])
        return tuple(phones)

    def as_dict(self) -> dict:
        return {
            "type": "entry",
            "utt_id": self.utt_id, "speaker": self.speaker, "split": self.split,
            "source_dir": self.source_dir, "wav_path": self.wav_path, "text": self.text,
            "pinyin": list(self.pinyin), "onsets": list(self.onsets),
            "finals": list(self.finals), "tones": list(self.tones),
            "erhua": list(self.erhua), "sample_rate": self.sample_rate,
            "n_samples": self.n_samples, "duration_s": round(self.duration_s, 6),
            "renderable": self.renderable, "quality_flags": list(self.quality_flags),
        }


@dataclass
class CorpusStats:
    """Token-weighted corpus statistics; every count is over accepted utterances."""

    utterances: int = 0
    speakers: int = 0
    syllables: int = 0
    total_duration_s: float = 0.0
    tokens: Counter = field(default_factory=Counter)
    onsets: Counter = field(default_factory=Counter)
    finals: Counter = field(default_factory=Counter)
    segments: Counter = field(default_factory=Counter)
    tones: Counter = field(default_factory=Counter)
    tone_types: Counter = field(default_factory=Counter)
    erhua: int = 0
    blocked_syllables: int = 0
    blocked_utterances: int = 0
    renderable_syllables: int = 0
    renderable_utterances: int = 0
    blocked_units: Counter = field(default_factory=Counter)
    missing_sets: Counter = field(default_factory=Counter)
    issues: Counter = field(default_factory=Counter)

    def add_syllable(self, token: Token, missing: tuple[str, ...]) -> None:
        self.syllables += 1
        self.tokens[token.pinyin] += 1
        self.onsets[token.initial or "-"] += 1
        self.finals[token.final] += 1
        self.tones[token.tone] += 1
        self.tone_types[(token.pinyin, token.tone)] += 1
        if token.erhua:
            self.erhua += 1
        for segment in FINAL_SEGMENTS[token.final]:
            self.segments[segment] += 1
        if missing:
            self.blocked_syllables += 1
            for unit in missing:
                self.blocked_units[unit] += 1
            self.missing_sets[frozenset(missing)] += 1
        else:
            self.renderable_syllables += 1

    @property
    def syllable_coverage(self) -> float:
        return self.renderable_syllables / self.syllables if self.syllables else 0.0

    @property
    def utterance_coverage(self) -> float:
        return self.renderable_utterances / self.utterances if self.utterances else 0.0

    def as_dict(self) -> dict:
        return {
            "utterances": self.utterances,
            "speakers": self.speakers,
            "syllables": self.syllables,
            "total_hours": round(self.total_duration_s / 3600.0, 4),
            "distinct_tokens": len(self.tokens),
            "distinct_onsets": len(self.onsets),
            "distinct_finals": len(self.finals),
            "distinct_segments": len(self.segments),
            "distinct_tone_forms": len(self.tone_types),
            "erhua_syllables": self.erhua,
            "blocked_syllables": self.blocked_syllables,
            "blocked_utterances": self.blocked_utterances,
            "renderable_syllables": self.renderable_syllables,
            "renderable_utterances": self.renderable_utterances,
            "syllable_coverage": round(self.syllable_coverage, 6),
            "utterance_coverage": round(self.utterance_coverage, 6),
            "tones": {str(tone): self.tones[tone] for tone in sorted(self.tones)},
            "onsets": dict(self.onsets.most_common()),
            "finals": dict(self.finals.most_common()),
            "segments": dict(self.segments.most_common()),
            "blocked_units": dict(self.blocked_units.most_common()),
            "missing_sets": [
                {"units": sorted(units), "syllables": count}
                for units, count in sorted(self.missing_sets.items(),
                                           key=lambda item: (-item[1], sorted(item[0])))],
            "issue_codes": dict(self.issues.most_common()),
        }


@dataclass
class CorpusIndex:
    root: str
    entries: list[CorpusEntry] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)
    stats: CorpusStats = field(default_factory=CorpusStats)
    split_seed: int = SPLIT_SEED
    split_sizes: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[ParseIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def by_split(self) -> dict[str, list[CorpusEntry]]:
        table: dict[str, list[CorpusEntry]] = {name: [] for name in SPLIT_NAMES}
        for entry in self.entries:
            table.setdefault(entry.split, []).append(entry)
        return table

    def speakers_by_split(self) -> dict[str, set[str]]:
        return {name: {entry.speaker for entry in entries}
                for name, entries in self.by_split().items()}

    def report(self, *, top_units: int = 10) -> dict:
        speakers = self.speakers_by_split()
        return {
            "version": MANIFEST_VERSION,
            "root": self.root,
            "split_seed": self.split_seed,
            "split_sizes": dict(self.split_sizes),
            "stats": self.stats.as_dict(),
            "speakers_per_split": {name: len(names) for name, names in speakers.items()},
            "utterances_per_split": {name: len(entries)
                                     for name, entries in self.by_split().items()},
            "next_units": rank_next_units(self.stats, top=top_units),
            "errors": [issue.as_dict() for issue in self.errors],
        }


def rank_next_units(stats: CorpusStats, *, top: int = 10) -> list[dict]:
    """Greedy inventory expansion: which missing units unblock the most real syllables.

    Each step picks the unit that newly renders the largest number of syllables, given
    every unit picked so far (a syllable blocked by {'onset:zh', 'final:i'} needs both).
    Erhua is reported separately: it is a modified final, not a missing phoneme.
    """
    if top <= 0:
        return []
    pending: dict[frozenset[str], int] = {}
    for units, count in stats.missing_sets.items():
        units = units - {"erhua"}
        if units:
            pending[units] = pending.get(units, 0) + count
    held: set[str] = set()
    ranking: list[dict] = []
    while pending and len(ranking) < top:
        # Marginal gain is measured exactly: a syllable counts only once the candidate
        # completes its whole missing set. Counting partial coverage instead would rank a
        # unit that still leaves the syllable blocked ahead of one that finishes it.
        candidates = sorted({unit for units in pending for unit in units if unit not in held})
        if not candidates:
            break
        gains = {unit: sum(count for units, count in pending.items()
                           if units <= held | {unit}) for unit in candidates}
        unit = min(candidates, key=lambda item: (-gains[item], item))
        newly = [units for units in pending if units <= held | {unit}]
        unblocked = sum(pending.pop(units) for units in newly)
        held.add(unit)
        also_needs = sorted({other for units in pending if unit in units
                             for other in units - held})
        ranking.append({"unit": unit, "syllables_unblocked": unblocked,
                        "also_needs": also_needs,
                        "cumulative_unblocked": unblocked +
                        (ranking[-1]["cumulative_unblocked"] if ranking else 0)})
    for item in ranking:
        item["syllables_unblocked_alone"] = stats.blocked_units[item["unit"]]
    erhua = stats.blocked_units.get("erhua", 0)
    if erhua:
        ranking.append({"unit": "erhua", "syllables_unblocked": erhua,
                        "also_needs": [], "cumulative_unblocked": erhua +
                        (ranking[-1]["cumulative_unblocked"] if ranking else 0),
                        "syllables_unblocked_alone": erhua,
                        "note": "needs an erhua final unit, not a phoneme"})
    return ranking


# --------------------------------------------------------------------------- scanning


def _wav_metadata(path: Path) -> tuple[int, int, float]:
    """(sample_rate, frames, duration_s) from the header; no sample is decoded."""
    channels, sample_rate, width, frames = read_wav_header(path)
    if channels != NATIVE_CHANNELS:
        raise ValueError(f"expected {NATIVE_CHANNELS} channel(s), header says {channels}")
    if width != NATIVE_SAMPLE_WIDTH:
        raise ValueError(f"expected {NATIVE_SAMPLE_WIDTH * 8}-bit PCM, header says "
                         f"{width * 8}-bit")
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate {sample_rate}")
    return sample_rate, frames, frames / float(sample_rate)


@dataclass(frozen=True)
class _Annotation:
    """One parsed content.txt line, before the WAV header is touched."""

    source_dir: str
    utt_id: str
    speaker: str
    text: str
    pinyin: tuple[str, ...]


def _read_annotations(root_path: Path, splits: dict[str, Path],
                      issues: list[ParseIssue]) -> list[_Annotation]:
    rows: list[_Annotation] = []
    for source_dir in sorted(splits):
        content = splits[source_dir] / "content.txt"
        with content.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    entry = parse_content_line(line)
                except ValueError as exc:
                    issues.append(ParseIssue(source_dir, f"content.txt:{line_no}",
                                             "malformed_line", str(exc)))
                    continue
                if entry is None:
                    continue
                if not entry.pinyin:
                    issues.append(ParseIssue(source_dir, entry.utt_id, "no_pinyin",
                                             "annotation carries no pinyin", "warn"))
                rows.append(_Annotation(source_dir, entry.utt_id,
                                        speaker_of(entry.utt_id), entry.text, entry.pinyin))
    return rows


def scan_corpus(root: str | Path, *, split_sizes: dict[str, int] | None = None,
                split_seed: int = SPLIT_SEED,
                progress: Callable[[int], None] | None = None) -> CorpusIndex:
    """Scan annotations and WAV headers; return the index and its statistics.

    The speaker split is assigned over the full speaker set *before* any utterance is
    accepted, so a rejected recording can never change which split its speaker lands in.
    """
    root_path = Path(root)
    splits = discover_splits(root_path)
    issues: list[ParseIssue] = []
    rows = _read_annotations(root_path, splits, issues)

    spk_path = root_path / "spk-info.txt"
    attributes = parse_spk_info(spk_path) if spk_path.is_file() else {}
    speakers = sorted({row.speaker for row in rows})
    sizes = dict(split_sizes if split_sizes is not None else auto_split_sizes(len(speakers)))
    assignment = assign_speaker_split(speakers, split_sizes=sizes, seed=split_seed,
                                      attributes=attributes)

    stats = CorpusStats(speakers=len(speakers))
    entries: list[CorpusEntry] = []
    for row in rows:
        split = assignment[row.speaker]
        wav_path = splits[row.source_dir] / "wav" / row.speaker / f"{row.utt_id}.wav"
        if not wav_path.is_file():
            issues.append(ParseIssue(split, row.utt_id, "missing_wav",
                                     f"no audio at {row.source_dir}/wav/{row.speaker}/"))
            continue
        try:
            sample_rate, frames, duration_s = _wav_metadata(wav_path)
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            issues.append(ParseIssue(split, row.utt_id, "bad_wav_header", str(exc)))
            continue

        tokens: list[Token] = []
        rejected = False
        for written in row.pinyin:
            try:
                tokens.append(split_token(written))
            except PinyinError as exc:
                issues.append(ParseIssue(split, row.utt_id, "bad_pinyin",
                                         f"{written!r}: {exc.detail}"))
                rejected = True
                break
        if rejected or not tokens:
            continue

        missing_per_syllable: list[tuple[str, ...]] = []
        for token in tokens:
            missing_per_syllable.append(
                missing_units(token.initial, token.final, SUPPORTED_ONSETS,
                              SUPPORTED_FINALS, erhua=token.erhua))
        renderable = not any(missing_per_syllable)

        for token, missing in zip(tokens, missing_per_syllable):
            stats.add_syllable(token, missing)
        stats.utterances += 1
        stats.total_duration_s += duration_s
        if renderable:
            stats.renderable_utterances += 1
        else:
            stats.blocked_utterances += 1

        entries.append(CorpusEntry(
            utt_id=row.utt_id, speaker=row.speaker, split=split,
            source_dir=row.source_dir,
            wav_path=wav_path.relative_to(root_path).as_posix(),
            text=row.text, pinyin=tuple(token.pinyin for token in tokens),
            onsets=tuple(token.initial for token in tokens),
            finals=tuple(token.final for token in tokens),
            tones=tuple(token.tone for token in tokens),
            erhua=tuple(token.erhua for token in tokens),
            sample_rate=sample_rate, n_samples=frames,
            duration_s=duration_s, renderable=renderable))
        if progress is not None and len(entries) % 5000 == 0:
            progress(len(entries))

    for issue in issues:
        stats.issues[issue.code] += 1
    return CorpusIndex(root=str(root_path), entries=entries, issues=issues, stats=stats,
                       split_seed=split_seed, split_sizes=sizes)


# --------------------------------------------------------------------- manifest I/O


def inventory_fingerprint() -> dict:
    """Identity of the phoneme inventory that decides renderability *right now*.

    A manifest stores the fingerprint it was scored under; comparing it with this function
    tells the loader whether the stored 'renderable' booleans are still trustworthy.
    """
    payload = {"size": len(INVENTORIES["mandarin"]),
               "onsets": sorted(SUPPORTED_ONSETS), "finals": sorted(SUPPORTED_FINALS)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(blob).hexdigest()}


def entry_renderable(entry: CorpusEntry) -> bool:
    """Re-decide renderability of one entry under the current inventory (no audio read)."""
    for written in entry.pinyin:
        token = split_token(written)
        if missing_units(token.initial, token.final, SUPPORTED_ONSETS, SUPPORTED_FINALS,
                         erhua=token.erhua):
            return False
    return True


def refresh_renderable(entries: list[CorpusEntry]
                       ) -> tuple[list[CorpusEntry], dict]:
    """Re-score every entry's 'renderable' flag under the current inventory.

    The flag is a scan-time boolean, so it goes stale as soon as the phoneme inventory
    grows (EXP-0016 added ten units and made thousands of utterances renderable without
    touching the manifest). Returns new entries plus a summary of what changed.
    """
    refreshed: list[CorpusEntry] = []
    stale = newly = lost = unscorable = 0
    examples: list[dict] = []
    for entry in entries:
        try:
            current = entry_renderable(entry)
        except PinyinError:
            unscorable += 1
            refreshed.append(replace(entry, renderable=False))
            continue
        if current != entry.renderable:
            stale += 1
            if current:
                newly += 1
            else:
                lost += 1
            if len(examples) < 5:
                examples.append({"utt_id": entry.utt_id, "was": entry.renderable,
                                 "now": current})
            refreshed.append(replace(entry, renderable=current))
        else:
            refreshed.append(entry)
    summary = {"inventory": inventory_fingerprint(), "checked": len(entries),
               "stale": stale, "newly_renderable": newly,
               "no_longer_renderable": lost, "unscorable": unscorable,
               "examples": examples}
    return refreshed, summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_corpus_manifest(path: str | Path, index: CorpusIndex, *,
                          extra_meta: dict | None = None) -> str:
    """Write the light manifest (meta line + one JSON line per utterance). Never overwrites."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        meta = {"type": "meta", "version": MANIFEST_VERSION, "root": index.root,
                "split_seed": index.split_seed, "split_sizes": dict(index.split_sizes),
                "utterances": len(index.entries), "syllables": index.stats.syllables,
                "inventory": inventory_fingerprint()}
        if extra_meta:
            meta.update(extra_meta)
        for payload in [meta] + [entry.as_dict() for entry in index.entries]:
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def read_corpus_manifest(path: str | Path) -> tuple[list[CorpusEntry], dict]:
    """Stream a light manifest back into entries plus its meta record."""
    target = Path(path)
    meta: dict = {}
    entries: list[CorpusEntry] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload
                continue
            try:
                entries.append(CorpusEntry(
                    utt_id=payload["utt_id"], speaker=payload["speaker"],
                    split=payload["split"], source_dir=payload.get("source_dir", ""),
                    wav_path=payload["wav_path"], text=payload.get("text", ""),
                    pinyin=tuple(payload["pinyin"]), onsets=tuple(payload["onsets"]),
                    finals=tuple(payload["finals"]), tones=tuple(payload["tones"]),
                    erhua=tuple(payload["erhua"]), sample_rate=payload["sample_rate"],
                    n_samples=payload["n_samples"], duration_s=payload["duration_s"],
                    renderable=payload["renderable"],
                    quality_flags=tuple(payload.get("quality_flags", ()))))
            except (KeyError, TypeError) as exc:
                raise ValueError(f"{target}:{line_no}: malformed manifest line: {exc}") from exc
    if not meta:
        raise ValueError(f"{target}: manifest has no meta record")
    return entries, meta


# ------------------------------------------------------- re-scoring and annotation

def stats_from_entries(entries: list[CorpusEntry]) -> CorpusStats:
    """Recompute corpus statistics from manifest entries under the *current* inventory.

    No audio is read: every unit needed to decide renderability is already in the entry, so
    growing the phoneme inventory never requires a second pass over 26 GB of WAV headers.
    """
    stats = CorpusStats(speakers=len({entry.speaker for entry in entries}))
    for entry in entries:
        tokens: list[Token] = []
        try:
            tokens = [split_token(written) for written in entry.pinyin]
        except PinyinError:
            stats.issues["bad_pinyin"] += 1
            continue
        missing_per_syllable = [
            missing_units(token.initial, token.final, SUPPORTED_ONSETS, SUPPORTED_FINALS,
                          erhua=token.erhua)
            for token in tokens]
        for token, missing in zip(tokens, missing_per_syllable):
            stats.add_syllable(token, missing)
        stats.utterances += 1
        stats.total_duration_s += entry.duration_s
        if any(missing_per_syllable):
            stats.blocked_utterances += 1
        else:
            stats.renderable_utterances += 1
    return stats


def coverage_report(entries: list[CorpusEntry], *, top_units: int = 12) -> dict:
    """Coverage under the current inventory plus the ranking of the next units to add."""
    stats = stats_from_entries(entries)
    flagged = Counter(flag for entry in entries for flag in entry.quality_flags)
    splits = Counter(entry.split for entry in entries)
    return {
        "version": MANIFEST_VERSION,
        "stats": stats.as_dict(),
        "utterances_per_split": dict(sorted(splits.items())),
        "quality_flags": dict(sorted(flagged.items())),
        "next_units": rank_next_units(stats, top=top_units),
    }


def apply_quality_flags(entries: list[CorpusEntry],
                        flags: dict[str, tuple[str, ...] | list[str]]) -> list[CorpusEntry]:
    """Return entries with per-utterance annotation flags merged in (input is not mutated)."""
    merged: list[CorpusEntry] = []
    for entry in entries:
        extra = tuple(flags.get(entry.utt_id, ()))
        if not extra:
            merged.append(entry)
            continue
        merged.append(replace(entry, quality_flags=entry.quality_flags + tuple(
            flag for flag in extra if flag not in entry.quality_flags)))
    return merged


def annotate_manifest(manifest: str | Path, output: str | Path, *,
                      root: str | Path | None = None) -> dict:
    """Write a new manifest that carries the prosody/annotation flags found by the audit.

    The corpus is not rescanned: entries are streamed from an existing manifest and only the
    flag field changes, so the speaker split and every measurement stay bit-identical.
    """
    source = Path(manifest)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing manifest: {target}")
    entries, meta = read_corpus_manifest(source)
    base = Path(root) if root is not None else Path(meta.get("root", "."))
    if not base.is_dir():
        raise FileNotFoundError(f"corpus root {base} does not exist; pass root= explicitly")
    audit = audit_prosody_labels(base)
    annotated = apply_quality_flags(entries, audit["flags"])
    index = CorpusIndex(root=str(base), entries=annotated, split_seed=meta.get("split_seed"),
                        split_sizes=meta.get("split_sizes", {}),
                        stats=stats_from_entries(annotated))
    extra = {
        "annotated_from": source.name,
        "annotated_from_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prosody_audit": {key: audit[key] for key in
                          ("checked", "matched", "mismatched", "format_errors",
                           "missing_text", "header_comments")},
    }
    digest = write_corpus_manifest(target, index, extra_meta=extra)
    return {"output": str(target), "manifest_sha256": digest, "utterances": len(annotated),
            "flagged_utterances": sum(1 for entry in annotated if entry.quality_flags),
            "audit": extra["prosody_audit"]}


def refresh_manifest(manifest: str | Path, output: str | Path, *,
                     root: str | Path | None = None) -> dict:
    """Write a new manifest whose renderable flags match the current inventory.

    The corpus is not rescanned: 'renderable' is a pure function of the stored pinyin, so
    the caller gets a manifest whose flags agree with the coverage report of the same
    inventory. Audio is never opened, so the speaker split and every measurement survive.
    """
    source = Path(manifest)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing manifest: {target}")
    entries, meta = read_corpus_manifest(source)
    refreshed, summary = refresh_renderable(entries)
    index = CorpusIndex(root=str(root) if root is not None else meta.get("root", ""),
                        entries=refreshed, split_seed=meta.get("split_seed", SPLIT_SEED),
                        split_sizes=meta.get("split_sizes", {}),
                        stats=stats_from_entries(refreshed))
    extra = {"refreshed_from": {"manifest": source.name, "version": meta.get("version"),
                                "inventory": meta.get("inventory"),
                                "sha256": _sha256_file(source)},
             "refresh": {key: value for key, value in summary.items() if key != "examples"},
             "refresh_examples": summary["examples"]}
    digest = write_corpus_manifest(target, index, extra_meta=extra)
    return {"output": str(target), "manifest_sha256": digest,
            "utterances": len(refreshed), "stale": summary["stale"],
            "newly_renderable": summary["newly_renderable"],
            "no_longer_renderable": summary["no_longer_renderable"],
            "unscorable": summary["unscorable"],
            "renderable_utterances": sum(1 for entry in refreshed if entry.renderable),
            "inventory": summary["inventory"]}


# ---------------------------------------------------------------- annotation audit


def parse_label_pinyin(payload: str) -> tuple[str, ...]:
    """Split a label_train-set.txt pinyin field, dropping the % and $ pause markers."""
    return tuple(token for token in payload.split() if token not in ("%", "$"))


#: Anomaly classes the audit distinguishes. None of them is repaired automatically.
ANOMALY_KINDS = ("prosody_conflict", "format_error", "missing_text", "header_comment")


def _anomaly(kind: str, *, utt_id: str | None, where: str, line_no: int, raw: str,
             detail: str, **extra) -> dict:
    record = {"kind": kind, "utt_id": utt_id, "file": where, "line": line_no,
              "raw": raw.rstrip("\n"), "detail": detail}
    record.update(extra)
    return record


def audit_prosody_labels(root: str | Path, *,
                         limit: int = 20) -> dict:
    """Cross-check train/label_train-set.txt against train/content.txt.

    The label file is the only place where phrase-level pauses survive; it is used as a
    second reading of the same syllables. Disagreements are reported, never resolved
    silently: neither file is authoritative about a tone the other contradicts.

    Every line that cannot be matched is classified, with its line number and its raw text
    kept verbatim, because a record that vanishes from the count also vanishes from the
    evidence. The classes are:

    * `prosody_conflict` - the line parses and the utterance exists, but the two readings
      disagree. The utterance is still usable audio; the disagreement is a flag.
    * `format_error` - the line does not carry the three '|'-separated fields. Nothing can be
      matched, so the raw line is the only record.
    * `missing_text` - the line parses, but its utterance is absent from content.txt.
    * `header_comment` - a documented '#' header line, not an annotation record at all.
    """
    root_path = Path(root)
    label_path = root_path / "train" / "label_train-set.txt"
    content_path = root_path / "train" / "content.txt"
    if not label_path.is_file() or not content_path.is_file():
        raise FileNotFoundError(f"missing annotation file under {root_path / 'train'}")

    content: dict[str, tuple[str, ...]] = {}
    with content_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                entry = parse_content_line(line)
            except ValueError:
                continue
            if entry is not None:
                content[entry.utt_id] = entry.pinyin

    where = label_path.relative_to(root_path).as_posix()
    checked = 0
    matched = 0
    mismatches: list[dict] = []
    anomalies: list[dict] = []
    flags: dict[str, list[str]] = {}
    counts = {kind: 0 for kind in ANOMALY_KINDS}
    with label_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                counts["header_comment"] += 1
                anomalies.append(_anomaly(
                    "header_comment", utt_id=None, where=where, line_no=line_no, raw=line,
                    detail="documented header line; carries no utterance id and no label"))
                continue
            fields = stripped.split("|")
            if len(fields) != 3:
                counts["format_error"] += 1
                leading = stripped.split("|", 1)[0].strip()
                guess = leading[:-4] if leading.endswith(".wav") else leading
                guess = guess or None
                if guess:
                    flags.setdefault(guess, []).append("label_format_error")
                anomalies.append(_anomaly(
                    "format_error", utt_id=guess, where=where, line_no=line_no, raw=line,
                    detail=f"expected 3 '|'-separated fields, found {len(fields)}"))
                continue
            utt_id = fields[0].strip()
            if utt_id.endswith(".wav"):
                utt_id = utt_id[:-4]
            if not utt_id:
                counts["format_error"] += 1
                anomalies.append(_anomaly(
                    "format_error", utt_id=None, where=where, line_no=line_no, raw=line,
                    detail="empty utterance id field"))
                continue
            reference = content.get(utt_id)
            if reference is None:
                counts["missing_text"] += 1
                flags.setdefault(utt_id, []).append("missing_content")
                anomalies.append(_anomaly(
                    "missing_text", utt_id=utt_id, where=where, line_no=line_no, raw=line,
                    detail="no line for this utterance in train/content.txt"))
                continue
            spoken = parse_label_pinyin(fields[1])
            checked += 1
            if spoken == reference:
                matched += 1
                continue
            differences = []
            for index in range(max(len(spoken), len(reference))):
                left = spoken[index] if index < len(spoken) else None
                right = reference[index] if index < len(reference) else None
                if left != right:
                    differences.append({"index": index, "label": left, "content": right})
            mismatches.append({"utt_id": utt_id, "line": line_no,
                               "label": list(spoken), "content": list(reference),
                               "differences": differences[:4]})
            counts["prosody_conflict"] += 1
            flags.setdefault(utt_id, []).append("prosody_conflict")
            anomalies.append(_anomaly(
                "prosody_conflict", utt_id=utt_id, where=where, line_no=line_no, raw=line,
                detail=f"{len(differences)} syllable position(s) disagree",
                label=list(spoken), content=list(reference),
                differences=differences[:4]))

    return {"label_file": where,
            "checked": checked, "matched": matched,
            "mismatched": len(mismatches),
            "mismatches": mismatches[:limit], "mismatch_total": len(mismatches),
            "anomaly_counts": dict(counts),
            "anomalies": anomalies,
            "flags": {utt: tuple(codes) for utt, codes in sorted(flags.items())},
            "format_errors": counts["format_error"],
            "missing_text": counts["missing_text"],
            "header_comments": counts["header_comment"]}
