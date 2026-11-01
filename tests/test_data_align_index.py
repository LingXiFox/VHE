"""Tests for the corpus alignment index.

What these tests prove: the index separates meta from utterances, samples whole speakers
deterministically, shifts aligner spans into file time, keeps skipped utterances as explicit
records, and refuses to overwrite an existing index.

What they do not prove: that the spans are correct speech boundaries. That is EXP-0021/0022
plus the listening check, not a unit test. The aligner here is a stub, so nothing about the
HMM is under test in this file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vhe.audio import write_wav
from vhe.data.align_index import (align_rows, build_training_items, index_fingerprint,
                                  load_index, read_manifest, select_rows, write_index)
from vhe.data.aligner import AlignmentResult, PhoneSpan


def make_manifest(tmp_path, splits=(("s1", "train"), ("s2", "train"), ("s3", "test"))):
    rows, seen = [], {}
    for speaker, split in splits:
        for index in range(3):
            seen[(speaker, index)] = f"{speaker}/utt{index}.wav"
            rows.append({"utt_id": f"{speaker}_utt{index}", "speaker": speaker, "split": split,
                         "wav_path": f"{speaker}/utt{index}.wav",
                         "pinyin": ["ma1", "ba2"]})
    path = tmp_path / "manifest.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "meta", "version": 3, "root": str(tmp_path)}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path, rows


def write_wav_row(root, row, sample_rate=16000, duration=0.4, seed=0):
    target = root / row["wav_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    write_wav(target, (0.2 * rng.standard_normal(int(sample_rate * duration))).astype(np.float32),
              sample_rate)
    return target


class StubAligner:
    """Returns fixed spans and a fixed gate offset, so the time-base shift is testable."""

    def __init__(self, offset_s=0.15):
        self.offset_s = offset_s
        self.calls = 0

    def align(self, audio, sample_rate, pinyins):
        self.calls += 1
        spans = (PhoneSpan("m", 0.2, 0.25, 5, 0.9), PhoneSpan("a", 0.25, 0.4, 15, 0.8))
        return AlignmentResult(phones=("m", "a"), logprob=-3.5, frames=40, hop_s=0.01,
                               spans=spans, offset_s=self.offset_s, silence_frames=(3, 2))


def test_read_manifest_separates_meta_from_rows(tmp_path):
    path, rows = make_manifest(tmp_path)
    meta, loaded = read_manifest(path)
    assert meta["version"] == 3
    assert [row["utt_id"] for row in loaded] == [row["utt_id"] for row in rows]


def test_select_rows_is_deterministic_and_keeps_speakers_whole(tmp_path):
    path, _ = make_manifest(tmp_path)
    _, rows = read_manifest(path)
    first = select_rows(rows, splits=["train"], speakers=2, per_speaker=2, seed=7)
    second = select_rows(rows, splits=["train"], speakers=2, per_speaker=2, seed=7)
    assert [row["utt_id"] for row in first] == [row["utt_id"] for row in second]
    speakers = [row["speaker"] for row in first]
    assert set(speakers) <= {"s1", "s2"}
    assert len(set(speakers)) == 2
    for speaker in set(speakers):
        assert speakers.count(speaker) == 2
    # A different seed picks a different speaker set or different utterances.
    other = select_rows(rows, splits=["train"], speakers=1, per_speaker=3, seed=11)
    assert all(row["speaker"] in {"s1", "s2"} for row in other)


def test_select_rows_refuses_a_speaker_count_above_the_manifest(tmp_path):
    path, _ = make_manifest(tmp_path)
    _, rows = read_manifest(path)
    with pytest.raises(ValueError, match="speakers requested"):
        select_rows(rows, splits=["train"], speakers=5)


def test_select_rows_drops_unrenderable_rows_before_sampling(tmp_path):
    """Sampling must happen inside the renderable pool, not before it.

    EXP-0023 asked for 320 rows and got 21: the 36-unit inventory blocks most sentences,
    so a per-speaker sample drawn from all rows wastes almost every slot.
    """
    path, _ = make_manifest(tmp_path)
    _, rows = read_manifest(path)
    blocked = {"s1_utt0", "s2_utt0", "s2_utt1"}
    for row in rows:
        row["renderable"] = row["utt_id"] not in blocked
    picked = select_rows(rows, splits=["train"], speakers=2, per_speaker=3, seed=7)
    assert [row["utt_id"] for row in picked if row["utt_id"] in blocked] == []
    assert len(picked) == 3
    # The blocked rows stay reachable for reproduction of the older experiments.
    unfiltered = select_rows(rows, splits=["train"], speakers=2, per_speaker=3, seed=7,
                             renderable_only=False)
    assert len(unfiltered) == 6


def test_select_rows_keeps_rows_from_manifests_without_the_renderable_column(tmp_path):
    path, _ = make_manifest(tmp_path)
    _, rows = read_manifest(path)
    assert all("renderable" not in row for row in rows)
    assert len(select_rows(rows, splits=["train"], speakers=2, per_speaker=3, seed=7)) == 6


def test_select_rows_counts_only_speakers_with_renderable_rows(tmp_path):
    """A speaker whose every utterance is blocked cannot be sampled at all."""
    path, _ = make_manifest(tmp_path)
    _, rows = read_manifest(path)
    for row in rows:
        row["renderable"] = row["speaker"] != "s2"
    picked = select_rows(rows, splits=["train"], speakers=1, per_speaker=2, seed=7)
    assert {row["speaker"] for row in picked} == {"s1"}
    with pytest.raises(ValueError, match="with renderable utterances"):
        select_rows(rows, splits=["train"], speakers=2, per_speaker=2, seed=7)


def test_align_rows_shifts_spans_into_file_time_and_records_skips(tmp_path):
    path, rows = make_manifest(tmp_path)
    meta, loaded = read_manifest(path)
    for index, row in enumerate(loaded):
        write_wav_row(tmp_path, row, seed=index)
    loaded[1]["wav_path"] = "s1/missing.wav"
    records = list(align_rows(StubAligner(offset_s=0.15), loaded, tmp_path))
    aligned = [record for record in records if not isinstance(record, dict)]
    skipped = [record for record in records if isinstance(record, dict)]
    assert len(aligned) == len(loaded) - 1
    assert len(skipped) == 1 and skipped[0]["type"] == "skipped"
    assert "FileNotFoundError" in skipped[0]["reason"]
    entry = aligned[0]
    # File time = crop time + gate offset, so consumers never see the gate.
    assert entry.spans[0][1] == pytest.approx(0.35)
    assert entry.spans[0][2] == pytest.approx(0.40)
    assert entry.offset_s == pytest.approx(0.15)
    assert [phone for phone, *_ in entry.spans] == ["m", "a"]


def test_write_index_refuses_overwrite_and_round_trips(tmp_path):
    path, rows = make_manifest(tmp_path)
    meta, loaded = read_manifest(path)
    for index, row in enumerate(loaded):
        write_wav_row(tmp_path, row, seed=index)
    entries = [entry.as_dict() if not isinstance(entry, dict) else entry
               for entry in align_rows(StubAligner(), loaded, tmp_path)]
    target = tmp_path / "index.jsonl"
    write_index(target, entries, meta={"manifest": str(path)})
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_index(target, entries, meta={})
    index_meta, spans = load_index(target)
    assert index_meta["manifest"] == str(path)
    assert index_meta["index_version"] == 1
    assert set(spans) == {row["utt_id"] for row in loaded}
    assert spans[loaded[0]["utt_id"]][0][0] == "m"
    assert len(index_fingerprint(target)) == 64


def test_build_training_items_reports_unreadable_rows(tmp_path):
    path, rows = make_manifest(tmp_path)
    meta, loaded = read_manifest(path)
    write_wav_row(tmp_path, loaded[0], seed=0)
    skipped = []
    items = build_training_items(loaded, tmp_path, on_skip=skipped.append)
    assert len(items) == 1
    assert len(skipped) == len(loaded) - 1
    assert all("reason" in record for record in skipped)
