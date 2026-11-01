"""AISHELL-3 corpus adapter: layout discovery, parsing and manifest building.

Verified against the published corpus (OpenSLR SLR93, Apache-2.0, ~19 GB, 218
speakers, 88035 utterances) and a public mirror of the original directory tree:

    AISHELL-3/
      spk-info.txt              # speaker<TAB>age_group<TAB>gender<TAB>accent
      phone_set.txt             # pinyin phone inventory
      train/
        content.txt             # <utt>.wav<TAB>char pinyin char pinyin ...
        label_train-set.txt     # prosody-labelled variant (optional)
        wav/<speaker>/<utt>.wav # 44.1 kHz, mono, 16-bit PCM
      test/
        content.txt
        wav/<speaker>/<utt>.wav

Nothing here downloads data. The corpus is never copied into Git: manifests store
relative paths plus a SHA256, and the audio itself stays outside the repository.

Failure policy: every rejected utterance becomes a ParseIssue with a stable code.
Nothing is dropped silently -- callers must be able to count and print the reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import wave
from typing import Iterable, Iterator

# Native format of the corpus; anything else is recorded and flagged, not silently fixed.
NATIVE_SAMPLE_RATE = 44100
NATIVE_CHANNELS = 1
NATIVE_SAMPLE_WIDTH = 2  # bytes -> 16-bit PCM

MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0

SPLIT_NAMES = ("train", "dev", "test")

#: Speaker split for the full 218-speaker corpus. Recorded in the manifest meta line.
SPLIT_SEED = 20261101
DEFAULT_SPLIT_SIZES = {"train": 180, "dev": 19, "test": 19}
#: dev and test share the same fraction; small corpora (tests) use it to scale the sizes.
DEV_SHARE = DEFAULT_SPLIT_SIZES["dev"] / sum(DEFAULT_SPLIT_SIZES.values())

_TONE_SYLLABLE = re.compile(r"[a-zü]+[1-5]")


@dataclass(frozen=True)
class ParseIssue:
    """One rejected or suspicious record. severity 'error' drops, 'warn' keeps."""

    split: str
    where: str
    code: str
    detail: str
    severity: str = "error"

    def as_dict(self) -> dict:
        return {"split": self.split, "where": self.where, "code": self.code,
                "detail": self.detail, "severity": self.severity}


@dataclass(frozen=True)
class ContentEntry:
    """One line of content.txt."""

    utt_id: str
    text: str
    pinyin: tuple[str, ...]


@dataclass(frozen=True)
class SpeakerInfo:
    speaker: str
    age_group: str
    gender: str
    accent: str


@dataclass(frozen=True)
class Utterance:
    """Accepted recording with everything needed to build a paired sample."""

    utt_id: str
    speaker: str
    #: speaker-grouped split assigned by assign_speaker_split ('train'/'dev'/'test')
    split: str
    wav_path: str  # relative to the corpus root, POSIX style: never absolute
    text: str
    pinyin: tuple[str, ...]
    sample_rate: int
    n_samples: int
    duration_s: float
    sha256: str
    #: directory the recording ships in ('train'/'test'): the official sentence-level
    #: split, which leaks speakers and is therefore NOT used as the training split.
    source_dir: str = ""
    age_group: str | None = None
    gender: str | None = None
    accent: str | None = None

    def as_dict(self) -> dict:
        return {
            "utt_id": self.utt_id,
            "speaker": self.speaker,
            "split": self.split,
            "source_dir": self.source_dir,
            "wav_path": self.wav_path,
            "text": self.text,
            "pinyin": list(self.pinyin),
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "duration_s": round(self.duration_s, 6),
            "sha256": self.sha256,
            "age_group": self.age_group,
            "gender": self.gender,
            "accent": self.accent,
        }


@dataclass
class Manifest:
    """Accepted utterances plus every diagnostic produced while building them."""

    root: str
    entries: list[Utterance] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)
    #: seed and sizes of the speaker split, so a manifest can be regenerated exactly
    split_seed: int = SPLIT_SEED
    split_sizes: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[ParseIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ParseIssue]:
        return [issue for issue in self.issues if issue.severity == "warn"]

    def speakers_by_split(self) -> dict[str, set[str]]:
        table: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
        for entry in self.entries:
            table.setdefault(entry.split, set()).add(entry.speaker)
        return table

    def stats(self) -> dict:
        durations = [entry.duration_s for entry in self.entries]
        table = self.speakers_by_split()
        return {
            "utterances": len(self.entries),
            "speakers": len({entry.speaker for entry in self.entries}),
            "total_hours": round(sum(durations) / 3600.0, 4) if durations else 0.0,
            "min_duration_s": round(min(durations), 3) if durations else None,
            "max_duration_s": round(max(durations), 3) if durations else None,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "speakers_per_split": {name: len(spk) for name, spk in table.items()},
            "utterances_per_split": {
                name: sum(1 for entry in self.entries if entry.split == name)
                for name in SPLIT_NAMES},
            "split_seed": self.split_seed,
            "split_sizes": dict(self.split_sizes),
            "error_codes": _count_codes(self.errors),
            "warning_codes": _count_codes(self.warnings),
        }

    def content_hash(self) -> str:
        """sha256 over the canonical manifest content (independent of file order)."""
        payload = json.dumps([entry.as_dict() for entry in sorted(self.entries,
                                                                  key=lambda e: (e.split, e.utt_id))],
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _count_codes(issues: Iterable[ParseIssue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.code] = counts.get(issue.code, 0) + 1
    return counts


# --------------------------------------------------------------------------- parsing


def discover_splits(root: str | Path) -> dict[str, Path]:
    """Map split name -> directory containing content.txt. Never guesses silently."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"AISHELL-3 root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"AISHELL-3 root is not a directory: {root_path}")
    found: dict[str, Path] = {}
    for name in ("train", "test", "dev"):
        candidate = root_path / name
        if (candidate / "content.txt").is_file():
            found[name] = candidate
    if (root_path / "content.txt").is_file() and not found:
        found[root_path.name or "train"] = root_path
    if not found:
        raise FileNotFoundError(
            f"No content.txt found under {root_path}. Expected AISHELL-3 layout: "
            "train/content.txt, test/content.txt and wav/<speaker>/<utt>.wav."
        )
    return found


def parse_content_line(line: str) -> ContentEntry | None:
    """Parse 'SSB00050001.wav<TAB>广 guang3 州 zhou1 ...' (blank/comment -> None).

    A missing transcript ('SSB00110002.wav<TAB>') is preserved as an empty entry so that
    build_manifest reports an explicit empty_text issue; stripping the tab first would
    turn it into a malformed line and hide the real defect.
    """
    text = line.rstrip("\r\n").lstrip("\ufeff")
    if not text.strip() or text.lstrip().startswith("#"):
        return None
    if "\t" in text:
        name, rest = text.split("\t", 1)
    else:  # tolerate single-space separation between filename and transcript
        parts = text.strip().split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"content.txt line has no transcript: {line!r}")
        name, rest = parts
    utt_id = name.strip()
    if utt_id.endswith(".wav"):
        utt_id = utt_id[:-4]
    if not utt_id:
        raise ValueError(f"content.txt line has no utterance id: {line!r}")

    chars: list[str] = []
    pinyin: list[str] = []
    for token in rest.split():
        if _TONE_SYLLABLE.fullmatch(token) and len(pinyin) < len(chars):
            pinyin.append(token)
        else:
            chars.append(token)
    return ContentEntry(utt_id=utt_id, text="".join(chars), pinyin=tuple(pinyin))


def parse_spk_info(path: Path) -> dict[str, SpeakerInfo]:
    """Parse 'SSB1837<TAB>B<TAB>female<TAB>north' lines, skipping # comments."""
    speakers: dict[str, SpeakerInfo] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [field.strip() for field in stripped.split("\t")]
        if len(fields) < 4:
            continue
        speakers[fields[0]] = SpeakerInfo(fields[0], fields[1], fields[2], fields[3])
    return speakers


def parse_pinyin_file(path: Path) -> dict[str, tuple[str, ...]]:
    """Optional standalone pinyin.txt: '<utt>.wav<TAB>guang3 zhou1 ...'."""
    table: dict[str, tuple[str, ...]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "\t" not in stripped:
            continue
        name, phones = stripped.split("\t", 1)
        utt_id = name.strip()
        if utt_id.endswith(".wav"):
            utt_id = utt_id[:-4]
        table[utt_id] = tuple(phones.split())
    return table


def read_wav_header(path: Path) -> tuple[int, int, int, int]:
    """Return (channels, sample_rate, sample_width_bytes, frames) from the header only."""
    with wave.open(str(path), "rb") as handle:
        return (handle.getnchannels(), handle.getframerate(),
                handle.getsampwidth(), handle.getnframes())


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speaker_of(utt_id: str) -> str:
    """AISHELL-3 utterance ids are SSB00050001: speaker = first 7 characters."""
    return utt_id[:7]


# --------------------------------------------------------------------------- building


def build_manifest(
    root: str | Path,
    *,
    compute_hashes: bool = True,
    min_duration_s: float = MIN_DURATION_S,
    max_duration_s: float = MAX_DURATION_S,
    split_sizes: dict[str, int] | None = None,
    split_seed: int = SPLIT_SEED,
) -> Manifest:
    """Walk the corpus and return accepted utterances plus explicit issues.

    The returned manifest is already re-partitioned by speaker (split_sizes=None scales
    180/19/19 to the corpus: exactly 180/19/19 for all 218 AISHELL-3 speakers). The
    official directory ('train'/'test') is kept per utterance in source_dir and is never
    used as the training split.
    """
    root_path = Path(root)
    splits = discover_splits(root_path)
    manifest = Manifest(root=str(root_path))

    speaker_table: dict[str, SpeakerInfo] = {}
    for candidate in (root_path / "spk-info.txt",):
        if candidate.is_file():
            speaker_table = parse_spk_info(candidate)
    if not speaker_table:
        manifest.issues.append(ParseIssue(split="*", where="spk-info.txt", code="missing_spk_info",
                                          detail="speaker attributes unavailable", severity="warn"))

    for split, directory in sorted(splits.items()):
        standalone_pinyin = directory / "pinyin.txt"
        pinyin_table = parse_pinyin_file(standalone_pinyin) if standalone_pinyin.is_file() else {}

        for line_no, line in enumerate(
                directory.joinpath("content.txt").read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1):
            try:
                entry = parse_content_line(line)
            except ValueError as exc:
                manifest.issues.append(ParseIssue(split=split, where=f"content.txt:{line_no}",
                                                  code="bad_line", detail=str(exc)))
                continue
            if entry is None:
                continue
            _accept_utterance(root_path, manifest, split, directory, entry, pinyin_table,
                              speaker_table, compute_hashes, min_duration_s, max_duration_s)

    speakers = {entry.speaker for entry in manifest.entries}
    sizes = dict(split_sizes) if split_sizes is not None else auto_split_sizes(len(speakers))
    assignment = assign_speaker_split(speakers, split_sizes=sizes, seed=split_seed,
                                      attributes=speaker_table)
    return apply_speaker_split(manifest, assignment, seed=split_seed, split_sizes=sizes)


def _accept_utterance(root_path: Path, manifest: Manifest, split: str, directory: Path,
                      entry: ContentEntry, pinyin_table: dict[str, tuple[str, ...]],
                      speaker_table: dict[str, SpeakerInfo], compute_hashes: bool,
                      min_duration_s: float, max_duration_s: float) -> None:
    speaker = speaker_of(entry.utt_id)
    wav_path = directory / "wav" / speaker / f"{entry.utt_id}.wav"

    if not entry.text:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="empty_text",
                                          detail="content.txt has no characters"))
        return
    pinyin = pinyin_table.get(entry.utt_id) or entry.pinyin
    if len(pinyin) != len(entry.text):
        manifest.issues.append(ParseIssue(
            split=split, where=entry.utt_id, code="pinyin_mismatch",
            detail=f"{len(entry.text)} characters vs {len(pinyin)} syllables",
            severity="warn"))
    if not wav_path.is_file():
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="missing_wav",
                                          detail=str(wav_path.relative_to(root_path))))
        return
    try:
        channels, sample_rate, width, frames = read_wav_header(wav_path)
    except (wave.Error, EOFError, OSError) as exc:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="unreadable_wav",
                                          detail=f"{type(exc).__name__}: {exc}"))
        return

    if channels != NATIVE_CHANNELS:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="not_mono",
                                          detail=f"{channels} channels; downmixing is a silent content change"))
        return
    if width != NATIVE_SAMPLE_WIDTH:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="not_16bit",
                                          detail=f"{width * 8}-bit PCM"))
        return
    if sample_rate != NATIVE_SAMPLE_RATE:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="unexpected_sample_rate",
                                          detail=f"{sample_rate} Hz, native is {NATIVE_SAMPLE_RATE}; "
                                                 "resample at pairing time and record it",
                                          severity="warn"))

    duration_s = frames / float(sample_rate) if sample_rate else 0.0
    if duration_s < min_duration_s:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="too_short",
                                          detail=f"{duration_s:.3f}s < {min_duration_s}s"))
        return
    if duration_s > max_duration_s:
        manifest.issues.append(ParseIssue(split=split, where=entry.utt_id, code="too_long",
                                          detail=f"{duration_s:.3f}s > {max_duration_s}s"))
        return

    info = speaker_table.get(speaker)
    manifest.entries.append(Utterance(
        utt_id=entry.utt_id,
        speaker=speaker,
        split=split,
        source_dir=split,
        wav_path=wav_path.relative_to(root_path).as_posix(),
        text=entry.text,
        pinyin=pinyin,
        sample_rate=sample_rate,
        n_samples=frames,
        duration_s=duration_s,
        sha256=file_sha256(wav_path) if compute_hashes else "",
        age_group=info.age_group if info else None,
        gender=info.gender if info else None,
        accent=info.accent if info else None,
    ))


# --------------------------------------------------------------------------- splits


def parse_split_sizes(value: str) -> dict[str, int] | None:
    """'180/19/19' (or 180:19:19, 180,19,19) -> sizes; 'auto'/'' -> None."""
    text = str(value).strip().lower()
    if text in ("", "auto"):
        return None
    parts = text.replace(":", "/").replace(",", "/").split("/")
    if len(parts) != 3 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"split sizes must be train/dev/test integers: {value!r}")
    return {"train": int(parts[0]), "dev": int(parts[1]), "test": int(parts[2])}


def auto_split_sizes(n_speakers: int) -> dict[str, int]:
    """Split sizes for n speakers: the 180/19/19 shape scaled by DEV_SHARE.

    218 speakers give exactly 180/19/19; a 20-speaker test corpus gives 16/2/2.
    """
    if n_speakers < 0:
        raise ValueError("n_speakers must be non-negative")
    dev = int(round(n_speakers * DEV_SHARE))
    test = dev
    return {"train": n_speakers - dev - test, "dev": dev, "test": test}


def _balanced_sequence(n_items: int, sizes: dict[str, int]) -> list[str]:
    """A split label per item whose counts equal sizes exactly, spread evenly.

    Greedy on the smallest (assigned + 1) / size ratio. A split that reached its size has
    ratio > 1 and is never picked again, so with n_items == sum(sizes) every split ends
    exactly at its target while the labels stay interleaved instead of blocked together.
    """
    assigned = {name: 0 for name in sizes}
    sequence: list[str] = []
    for _ in range(n_items):
        name = min(
            SPLIT_NAMES,
            key=lambda split: ((assigned[split] + 1) / sizes[split] if sizes[split] else
                               float("inf"), SPLIT_NAMES.index(split)))
        sequence.append(name)
        assigned[name] += 1
    return sequence


def assign_speaker_split(speakers: Iterable[str], *,
                         split_sizes: dict[str, int] | None = None,
                         seed: int = SPLIT_SEED,
                         attributes: dict[str, SpeakerInfo] | None = None,
                         ) -> dict[str, str]:
    """Assign every speaker to exactly one split (speaker is the unit, never an utterance).

    The official AISHELL-3 layout splits *sentences*, not speakers: 170 of the 214
    speakers under test/ also appear under train/. Re-using that boundary would leak
    speakers into validation, so this function ignores it entirely and re-partitions all
    speakers. Utterance paths are untouched; only the split label changes.

    Allocation is stratified by gender when speaker attributes are available, so dev and
    test keep the corpus ratio instead of depending on the shuffle, and deterministic
    given (speakers, split_sizes, seed): re-running reproduces the same assignment.
    """
    names = sorted(set(speakers))
    sizes = dict(split_sizes if split_sizes is not None else auto_split_sizes(len(names)))
    unknown = set(sizes) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown split name(s): {sorted(unknown)}")
    if any(size < 0 for size in sizes.values()):
        raise ValueError(f"split sizes must be non-negative: {sizes}")
    if sum(sizes.values()) != len(names):
        raise ValueError(f"split sizes {sizes} do not cover {len(names)} speakers")
    missing = [name for name in SPLIT_NAMES if name not in sizes]
    if missing:
        raise ValueError(f"split sizes must name every split, missing {missing}")

    import random
    rng = random.Random(seed)
    strata: dict[str, list[str]] = {}
    for name in names:
        info = (attributes or {}).get(name)
        key = info.gender if info is not None and info.gender else "unknown"
        strata.setdefault(key, []).append(name)

    # Deterministic stratum order: biggest first, ties by name; members shuffled by seed.
    groups = [strata[key] for key in sorted(strata, key=lambda item: (-len(strata[item]), item))]
    for members in groups:
        rng.shuffle(members)
    # Deal round-robin across strata so every split receives the corpus gender mix, then
    # hand out the exactly-sized label sequence in that order.
    ordered: list[str] = []
    for index in range(len(names)):
        for members in groups:
            if index < len(members):
                ordered.append(members[index])
    return dict(zip(ordered, _balanced_sequence(len(ordered), sizes)))


def apply_speaker_split(manifest: Manifest, assignment: dict[str, str], *,
                        seed: int = SPLIT_SEED,
                        split_sizes: dict[str, int] | None = None) -> Manifest:
    """Return a new manifest whose entries carry the speaker-grouped split."""
    unassigned = sorted({entry.speaker for entry in manifest.entries} - set(assignment))
    if unassigned:
        raise ValueError(f"no split assigned for speaker(s) {unassigned[:5]}")
    entries = []
    for entry in manifest.entries:
        split = assignment[entry.speaker]
        entries.append(Utterance(**{**entry.__dict__, "split": split}))
    return Manifest(root=manifest.root, entries=entries, issues=list(manifest.issues),
                    split_seed=seed,
                    split_sizes=dict(split_sizes if split_sizes is not None else {}))


def check_no_speaker_leak(manifest: Manifest) -> dict[str, list[str]]:
    """Raise if one speaker appears in more than one split; else return the table."""
    table: dict[str, set[str]] = {}
    for entry in manifest.entries:
        table.setdefault(entry.speaker, set()).add(entry.split)
    leaked = {speaker: sorted(splits) for speaker, splits in table.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"speaker leakage across splits: {leaked}")
    return {split: sorted(speaker for speaker, splits in table.items() if split in splits)
            for split in SPLIT_NAMES}


# --------------------------------------------------------------------------- manifest I/O


def write_manifest(path: str | Path, manifest: Manifest) -> str:
    """Write JSONL (one utterance per line) + 'meta' first line; return sha256.

    Refuses to overwrite: a manifest is an experiment input, and silently replacing it
    would invalidate every recorded content hash pointing at this path.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "meta", "root": manifest.root,
                         "stats": manifest.stats(),
                         "split_seed": manifest.split_seed,
                         "split_sizes": dict(manifest.split_sizes),
                         "content_sha256": manifest.content_hash()},
                        sort_keys=True, ensure_ascii=False)]
    for entry in sorted(manifest.entries, key=lambda e: (e.split, e.utt_id)):
        lines.append(json.dumps({"type": "utterance", **entry.as_dict()},
                                sort_keys=True, ensure_ascii=False))
    for issue in manifest.issues:
        lines.append(json.dumps({"type": "issue", **issue.as_dict()},
                                sort_keys=True, ensure_ascii=False))
    payload = "\n".join(lines) + "\n"
    target.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_manifest(path: str | Path) -> tuple[Manifest, dict]:
    """Inverse of write_manifest; returns (manifest, meta line)."""
    target = Path(path)
    manifest = Manifest(root="")
    meta: dict = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.pop("type")
        if kind == "meta":
            meta = record
            manifest.root = record.get("root", "")
            manifest.split_seed = int(record.get("split_seed", SPLIT_SEED))
            manifest.split_sizes = dict(record.get("split_sizes", {}))
        elif kind == "utterance":
            record["pinyin"] = tuple(record.get("pinyin", ()))
            manifest.entries.append(Utterance(**record))
        elif kind == "issue":
            manifest.issues.append(ParseIssue(**record))
        else:
            raise ValueError(f"unknown manifest record type {kind!r}")
    return manifest, meta


def iter_utterance_paths(manifest: Manifest, root: str | Path | None = None) -> Iterator[tuple[Utterance, Path]]:
    base = Path(root) if root is not None else Path(manifest.root)
    for entry in manifest.entries:
        yield entry, base / entry.wav_path
