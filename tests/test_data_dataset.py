"""Tests for the lazy paired dataset view.

Rendering is stubbed: these tests pin selection, sharding, seed derivation and the
usable/issue bookkeeping, while tests/test_data_pairing.py keeps covering the real render.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vhe.data import dataset as dataset_module
from vhe.data.corpus import CorpusEntry, write_corpus_manifest
from vhe.data.dataset import CorpusPairs, PairSample, summarize
from vhe.data.pairing import PairResult


def make_entry(index: int, *, split: str = "train", renderable: bool = True,
               blocked: bool = False) -> CorpusEntry:
    if blocked:
        # chuang2 needs an onset (ch) and a final (uang) that the inventory lacks, so the
        # entry is genuinely unrenderable and revalidation cannot flip it.
        text, pinyin, onsets, finals, tones = "窗", ("chuang2",), ("ch",), ("uang",), (2,)
    else:
        text, pinyin, onsets, finals, tones = "大", ("da4",), ("d",), ("a",), (4,)
    return CorpusEntry(
        utt_id=f"SSB000{index:04d}", speaker=f"SSB000{index % 3}", split=split,
        source_dir="train", wav_path=f"train/wav/SSB000{index % 3}/SSB000{index:04d}.wav",
        text=text, pinyin=pinyin, onsets=onsets, finals=finals, tones=tones,
        erhua=(False,), sample_rate=44100, n_samples=44100, duration_s=1.0,
        renderable=renderable)


class FakeResult(PairResult):
    def __init__(self, utt_id: str, *, levels: tuple[float, ...] = (0.1, 0.2),
                 issues: list[dict] | None = None, status: str = "ok") -> None:
        super().__init__(utterance_id=utt_id, status=status, sample_rate=24000,
                         source_sample_rate=44100)
        self.real = np.zeros(len(levels), dtype=np.float32) + 0.5
        self.mechanical = np.array(levels, dtype=np.float32)
        self.issues = list(issues or [])


@pytest.fixture()
def manifest(tmp_path, monkeypatch):
    entries = [make_entry(1), make_entry(2, split="dev"),
               make_entry(3, renderable=False, blocked=True), make_entry(4)]
    target = tmp_path / "manifest.jsonl"
    write_corpus_manifest(target, _index(entries, tmp_path))
    captured: list[dict] = []

    def fake_build_pair(path, pinyins, **kwargs):
        captured.append({"path": str(path), "pinyins": list(pinyins), **kwargs})
        return FakeResult(path.stem)

    monkeypatch.setattr(dataset_module, "build_pair", fake_build_pair)
    return target, entries, captured


def _index(entries, tmp_path):
    from vhe.data.corpus import CorpusIndex, CorpusStats

    stats = CorpusStats()
    stats.utterances = len(entries)
    return CorpusIndex(root=str(tmp_path), entries=list(entries), issues=[], stats=stats,
                       split_seed=20261101, split_sizes={"train": 3, "dev": 1, "test": 0})


def test_from_manifest_filters_split_and_renderable(manifest, tmp_path) -> None:
    target, entries, _ = manifest
    view = CorpusPairs.from_manifest(target, root=tmp_path)
    assert [entry.utt_id for entry in view.entries] == ["SSB0000001", "SSB0000004"]

    dev = CorpusPairs.from_manifest(target, root=tmp_path, split="dev")
    assert [entry.utt_id for entry in dev.entries] == ["SSB0000002"]

    everything = CorpusPairs.from_manifest(target, root=tmp_path, split=None,
                                           renderable_only=False)
    assert len(everything) == 4

    limited = CorpusPairs.from_manifest(target, root=tmp_path, limit=1)
    assert len(limited) == 1


def test_sharding_by_index_is_stable(manifest, tmp_path) -> None:
    target, entries, _ = manifest
    full = CorpusPairs.from_manifest(target, root=tmp_path, split=None, renderable_only=False)
    first = CorpusPairs.from_manifest(target, root=tmp_path, split=None, renderable_only=False,
                                      world_size=2, rank=0)
    second = CorpusPairs.from_manifest(target, root=tmp_path, split=None, renderable_only=False,
                                       world_size=2, rank=1)
    assert [entry.utt_id for entry in first.entries] == [
        full.entries[0].utt_id, full.entries[2].utt_id]
    assert [entry.utt_id for entry in second.entries] == [
        full.entries[1].utt_id, full.entries[3].utt_id]


def test_getitem_renders_at_the_pinned_seed(manifest, tmp_path) -> None:
    target, _, captured = manifest
    view = CorpusPairs.from_manifest(target, root=tmp_path, seed=100)
    sample = view[1]
    assert isinstance(sample, PairSample)
    assert captured[-1]["seed"] == 101  # base seed + index, independent of sharding
    assert captured[-1]["pinyins"] == ["da4"]
    assert captured[-1]["path"].endswith("SSB0000004.wav")
    assert view.seed_for(1) == 101
    assert sample.usable


def test_usable_requires_equal_lengths_and_no_error(monkeypatch) -> None:
    entry = make_entry(1)
    ok = FakeResult(entry.utt_id)
    assert PairSample(entry, ok).usable
    broken = FakeResult(entry.utt_id, levels=(0.1, 0.2, 0.3))
    broken.mechanical = np.zeros(2, dtype=np.float32)
    assert not PairSample(entry, broken).usable
    warned = FakeResult(entry.utt_id, issues=[{"code": "resampled", "severity": "warn"}])
    assert PairSample(entry, warned).usable
    error = FakeResult(entry.utt_id, issues=[{"code": "syllable_plan_blocked"},
                                             {"severity": "error"}])
    assert not PairSample(entry, error).usable
    missing = FakeResult(entry.utt_id, status="blocked")
    missing.mechanical = None
    assert not PairSample(entry, missing).usable


def test_quality_flags_are_surfaced_and_filterable(tmp_path, monkeypatch) -> None:
    entries = [replace(make_entry(1), quality_flags=("prosody_conflict",)), make_entry(2)]
    target = tmp_path / "flagged.jsonl"
    write_corpus_manifest(target, _index(entries, tmp_path))
    monkeypatch.setattr(dataset_module, "build_pair",
                        lambda path, pinyins, **kwargs: FakeResult(path.stem))

    view = CorpusPairs.from_manifest(target, root=tmp_path, split=None)
    assert view[0].flagged and view[0].quality_flags == ("prosody_conflict",)
    assert not view[1].flagged
    assert view.inventory()["flagged_utterances"] == 1
    assert "quality_flags" in view[0].summary()

    filtered = CorpusPairs.from_manifest(target, root=tmp_path, split=None,
                                         exclude_flagged=True)
    assert [entry.utt_id for entry in filtered.entries] == ["SSB0000002"]
    assert summarize([view[0], view[1]])["quality_flags"] == {"prosody_conflict": 1}

    entry = make_entry(1)
    samples = [PairSample(entry, FakeResult(entry.utt_id)),
               PairSample(entry, FakeResult(entry.utt_id, status="blocked",
                                            issues=[{"code": "bad_pinyin"}])),
               PairSample(entry, FakeResult(entry.utt_id,
                                            issues=[{"code": "resampled",
                                                     "severity": "warn"}])),
               ]
    report = summarize(samples)
    assert report["samples"] == 3
    assert report["usable"] == 2
    assert report["by_status"] == {"ok": 2, "blocked": 1}
    assert report["issue_codes"] == {"bad_pinyin": 1, "resampled": 1}


def test_revalidation_repairs_a_stale_renderable_flag(tmp_path, monkeypatch) -> None:
    # A manifest written before EXP-0016 stored renderable=False for a syllable the current
    # inventory renders. The loader must not trust the stored boolean.
    entries = [replace(make_entry(1), renderable=False), make_entry(2)]
    target = tmp_path / "stale.jsonl"
    write_corpus_manifest(target, _index(entries, tmp_path))

    def fake_build_pair(path, pinyins, **kwargs):
        return FakeResult(path.stem)

    monkeypatch.setattr(dataset_module, "build_pair", fake_build_pair)
    view = CorpusPairs.from_manifest(target, root=tmp_path)
    assert [entry.utt_id for entry in view.entries] == ["SSB0000001", "SSB0000002"]
    staleness = view.inventory()["staleness"]
    assert staleness["revalidated"] is True
    assert staleness["newly_renderable"] == 1
    assert staleness["changed"] == 1
    assert staleness["renderable_in_manifest"] == 1
    assert staleness["renderable_now"] == 2

    frozen = CorpusPairs.from_manifest(target, root=tmp_path, revalidate=False)
    assert [entry.utt_id for entry in frozen.entries] == ["SSB0000002"]
    assert frozen.inventory()["staleness"]["revalidated"] is False


def test_the_loader_asks_for_the_mandarin_backend_and_viterbi_by_default(tmp_path,
                                                                        monkeypatch) -> None:
    target = tmp_path / "defaults.jsonl"
    write_corpus_manifest(target, _index([make_entry(1)], tmp_path))
    captured: list[dict] = []

    def fake_build_pair(path, pinyins, **kwargs):
        captured.append(kwargs)
        return FakeResult(path.stem)

    monkeypatch.setattr(dataset_module, "build_pair", fake_build_pair)
    view = CorpusPairs.from_manifest(target, root=tmp_path)
    _ = view[0]
    assert captured[-1]["pitch_estimator"] == "viterbi"
    assert captured[-1]["alignment"] == "dp"
    assert captured[-1]["backend"].name == "mandarin_formant"
    assert view.inventory()["backend"] == "mandarin_formant"
    assert view.inventory()["pitch_estimator"] == "viterbi"
    assert view.inventory()["alignment"] == "dp"

