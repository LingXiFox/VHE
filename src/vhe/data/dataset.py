"""On-demand paired-sample access over the light corpus manifest (P1.3).

The manifest stores annotations and WAV headers only; nothing here pre-renders the corpus.
One item is built when it is requested: read the recording, resample, estimate F0, segment
by energy, render each syllable with the mechanical backend at the measured onset. That
keeps the repository free of derived audio, lets a 4090 box stream the corpus from disk, and
means a change to the synthesizer invalidates nothing on disk.

The class is a plain mapping (__len__/__getitem__), so torch DataLoader can wrap it while
this module stays importable without torch -- the data path must work on a CPU-only host.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .corpus import CorpusEntry, read_corpus_manifest
from .pairing import PAIR_STATUS, PairResult, build_pair

DEFAULT_MAX_DURATION_S = 30.0


@dataclass
class PairSample:
    """One paired training unit: the real recording and its mechanical counterpart."""

    entry: CorpusEntry
    result: PairResult

    @property
    def status(self) -> str:
        return self.result.status

    @property
    def quality_flags(self) -> tuple[str, ...]:
        """Annotation flags inherited from the manifest (e.g. 'prosody_conflict')."""
        return self.entry.quality_flags

    @property
    def flagged(self) -> bool:
        return bool(self.entry.quality_flags)

    @property
    def usable(self) -> bool:
        """True only when both signals exist, have equal length and no error was raised."""
        if self.result.status not in PAIR_STATUS or self.result.real is None:
            return False
        if self.result.mechanical is None:
            return False
        if self.result.real.shape != self.result.mechanical.shape:
            return False
        return not [issue for issue in self.result.issues
                    if issue.get("severity", "error") == "error"]

    def summary(self) -> dict:
        return {"utt_id": self.entry.utt_id, "speaker": self.entry.speaker,
                "split": self.entry.split, "status": self.status,
                "usable": self.usable, "duration_s": round(self.entry.duration_s, 3),
                "syllables": len(self.entry.pinyin),
                "quality_flags": list(self.entry.quality_flags),
                "issues": [issue.get("code") for issue in self.result.issues],
                "f0_stats": self.result.f0_stats}


class CorpusPairs(Sequence[PairSample]):
    """Map index -> PairSample, rendering the mechanical signal lazily.

    Determinism: the seed for utterance i is base_seed + i, so the mechanical rendering of a
    given index does not depend on iteration order or worker count. Sharding is index-based
    (index % world_size) for the same reason.
    """

    def __init__(self, entries: Sequence[CorpusEntry], root: str | Path, *,
                 template=None, sample_rate: int = 24000, seed: int = 42,
                 max_duration_s: float = DEFAULT_MAX_DURATION_S,
                 world_size: int = 1, rank: int = 0) -> None:
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.seed = seed
        self.max_duration_s = max_duration_s
        if template is None:
            from ..synth import PRESETS
            template = PRESETS["yang"]
        self.template = template
        selected = list(entries)
        if world_size > 1:
            selected = [entry for position, entry in enumerate(selected)
                        if position % world_size == rank]
        self.entries: list[CorpusEntry] = selected
        self._world_size = world_size
        self._rank = rank

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_manifest(cls, manifest: str | Path, *, root: str | Path | None = None,
                      split: str | None = "train", renderable_only: bool = True,
                      exclude_flagged: bool = False,
                      limit: int | None = None, **kwargs) -> "CorpusPairs":
        """Load a light manifest and keep only the utterances this shard will use."""
        entries, meta = read_corpus_manifest(manifest)
        base = Path(root) if root is not None else Path(meta.get("root", "."))
        if not base.exists():
            raise FileNotFoundError(
                f"corpus root {base} does not exist; pass root= for a manifest built elsewhere")
        if split is not None:
            entries = [entry for entry in entries if entry.split == split]
        if renderable_only:
            entries = [entry for entry in entries if entry.renderable]
        if exclude_flagged:
            # Flags are surfaced, not silently dropped: this is an explicit opt-in filter.
            entries = [entry for entry in entries if not entry.quality_flags]
        if limit is not None:
            entries = entries[:limit]
        return cls(entries, base, **kwargs)

    # --------------------------------------------------------------------- sequence

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> PairSample:
        entry = self.entries[index]
        path = self.root / entry.wav_path
        result = build_pair(path, list(entry.pinyin), template=self.template,
                            sample_rate=self.sample_rate,
                            seed=self.seed + index,
                            max_duration_s=self.max_duration_s)
        return PairSample(entry=entry, result=result)

    def seed_for(self, index: int) -> int:
        """The rendering seed of one index, independent of sharding."""
        return self.seed + index

    # --------------------------------------------------------------------- helpers

    def inventory(self) -> dict:
        """What this view holds, for logs and experiment metadata."""
        speakers = {entry.speaker for entry in self.entries}
        durations = [entry.duration_s for entry in self.entries]
        return {"utterances": len(self.entries), "speakers": len(speakers),
                "hours": round(sum(durations) / 3600.0, 4),
                "split": sorted({entry.split for entry in self.entries}),
                "sample_rate": self.sample_rate, "seed": self.seed,
                "world_size": self._world_size, "rank": self._rank,
                "flagged_utterances": sum(1 for entry in self.entries
                                          if entry.quality_flags),
                "root": str(self.root)}

    def iter_samples(self, start: int = 0, stop: int | None = None) -> Iterator[PairSample]:
        for index in range(start, len(self) if stop is None else stop):
            yield self[index]


def summarize(samples: Sequence[PairSample]) -> dict:
    """Aggregate a batch of samples without hiding failures."""
    by_status: dict[str, int] = {}
    reasons: dict[str, int] = {}
    usable = 0
    for sample in samples:
        by_status[sample.status] = by_status.get(sample.status, 0) + 1
        usable += int(sample.usable)
        for issue in sample.result.issues:
            code = issue.get("code", "unknown")
            reasons[code] = reasons.get(code, 0) + 1
    flags: dict[str, int] = {}
    for sample in samples:
        for flag in sample.quality_flags:
            flags[flag] = flags.get(flag, 0) + 1
    return {"samples": len(samples), "usable": usable, "by_status": by_status,
            "quality_flags": flags, "issue_codes": reasons}
