"""Tests for the pair cache (src/vhe/data/pair_cache.py).

What these prove: only training-ready pairs are written, every rejected row is recorded with
its reason, the cache refuses to overwrite itself, and PairCache reads back exactly what was
written (speaker, split, arrays).

What these do not prove: that the written pairs are acoustically good. That is the listening
gate (EXP-0023), not a unit test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vhe.data.pair_cache import PairCache, build_pair_cache


class StubResult:
    def __init__(self, *, ready: bool, status: str = "ok", reasons=(), length: int = 800):
        self.real = np.zeros(length, dtype=np.float32) + 0.25 if status == "ok" else None
        self.mechanical = np.zeros(length, dtype=np.float32) - 0.25 if status == "ok" else None
        self.status = status
        self.readiness = {"training_ready": ready, "reasons": list(reasons)}


def write_index(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "meta", "root": "/tmp"}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def write_manifest(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "meta", "root": "/tmp"}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def aligned_row(utt_id: str, speaker: str = "S1", split: str = "train") -> dict:
    return {"type": "utterance", "utt_id": utt_id, "speaker": speaker, "split": split,
            "wav_path": f"wav/{utt_id}.wav", "spans": [["a", 0.0, 0.2, 0.9]],
            "offset_s": 0.0, "frames": 20, "logprob": -1.0, "confidence": 0.9,
            "duration_s": 0.2}


def manifest_row(utt_id: str) -> dict:
    return {"utt_id": utt_id, "pinyin": ["a1"]}


def test_writes_only_training_ready_pairs_and_records_rejections(tmp_path: Path) -> None:
    index = write_index(tmp_path / "index.jsonl",
                        [aligned_row("u1"), aligned_row("u2"), aligned_row("u3")])
    manifest = write_manifest(tmp_path / "manifest.jsonl",
                              [manifest_row(name) for name in ("u1", "u2", "u3")])
    results = {"u1": StubResult(ready=True), "u2": StubResult(ready=False, reasons=["low"]),
               "u3": StubResult(ready=True)}
    calls = []

    def builder(wav_path, pinyins, **kwargs):
        calls.append((wav_path, tuple(pinyins), kwargs["alignment"]))
        return results[Path(wav_path).stem]

    meta = build_pair_cache(index, manifest, tmp_path, tmp_path / "cache",
                            pair_builder=builder)

    assert meta["requested"] == 3 and meta["written"] == 2 and meta["rejected"] == 1
    assert meta["rejected_rows"][0]["utt_id"] == "u2"
    assert meta["rejected_reasons"] == {"low": 1}
    assert calls[0][2] == "forced"
    assert sorted(path.name for path in (tmp_path / "cache" / "pairs").iterdir()) == \
        ["u1.npz", "u3.npz"]


def test_cache_round_trips_metadata_and_arrays(tmp_path: Path) -> None:
    index = write_index(tmp_path / "index.jsonl", [aligned_row("u1", speaker="S9")])
    manifest = write_manifest(tmp_path / "manifest.jsonl", [manifest_row("u1")])
    build_pair_cache(index, manifest, tmp_path, tmp_path / "cache",
                     pair_builder=lambda *_, **__: StubResult(ready=True, length=400))
    cache = PairCache(tmp_path / "cache")
    assert cache.meta["template"] == "yang" and cache.meta["sample_rate"] == 24000
    assert len(cache) == 1 and cache.speakers() == {"S9"}
    pair = next(iter(cache))
    mechanical, real = pair.load()
    assert mechanical.shape == real.shape == (400,)
    assert float(mechanical[0]) == pytest.approx(-0.25)
    assert float(real[0]) == pytest.approx(0.25)
    assert pair.duration_s == pytest.approx(400 / 24000, abs=1e-4)


def test_cache_refuses_to_overwrite(tmp_path: Path) -> None:
    index = write_index(tmp_path / "index.jsonl", [aligned_row("u1")])
    manifest = write_manifest(tmp_path / "manifest.jsonl", [manifest_row("u1")])
    builder = lambda *_, **__: StubResult(ready=True)  # noqa: E731
    build_pair_cache(index, manifest, tmp_path, tmp_path / "cache", pair_builder=builder)
    with pytest.raises(FileExistsError):
        build_pair_cache(index, manifest, tmp_path, tmp_path / "cache", pair_builder=builder)


def test_rows_without_transcript_are_rejected_not_dropped(tmp_path: Path) -> None:
    index = write_index(tmp_path / "index.jsonl", [aligned_row("u1")])
    manifest = write_manifest(tmp_path / "manifest.jsonl", [])
    meta = build_pair_cache(index, manifest, tmp_path, tmp_path / "cache",
                            pair_builder=lambda *_, **__: StubResult(ready=True))
    assert meta["written"] == 0 and meta["rejected"] == 1
    assert meta["rejected_rows"][0]["reason"] == "no transcript in manifest"
