"""Tests for the P2 training loop (src/vhe/train.py).

What these prove: crops are deterministic and centred, the pipeline runs end to end on a
synthetic pair cache, an overfit run lowers the loss it optimises, checkpoints are written at
step-named paths and a resumed run continues from the saved step instead of restarting.

What these do not prove: that a lower humanization loss sounds more natural, or that the
default hyper-parameters fit real speech. Only an experiment with an audio comparison can say
that (AGENTS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vhe.data.pair_cache import PairCache, build_pair_cache
from vhe.train import PairCrops, TrainConfig, train_model

pytest.importorskip("torch")

SAMPLE_RATE = 24000


class StubResult:
    def __init__(self, length: int, phase: float) -> None:
        t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
        self.real = (0.3 * np.sin(2 * np.pi * (220 + phase) * t)).astype(np.float32)
        self.mechanical = (0.3 * np.sin(2 * np.pi * (220 + phase) * t + 0.4)
                           + 0.05 * np.cos(2 * np.pi * 3000 * t)).astype(np.float32)
        self.status = "ok"
        self.readiness = {"training_ready": True, "reasons": []}


def make_cache(root: Path, count: int = 2, seconds: float = 1.0) -> Path:
    index = root / "index.jsonl"
    manifest = root / "manifest.jsonl"
    with index.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "meta"}) + "\n")
        for position in range(count):
            handle.write(json.dumps({
                "type": "utterance", "utt_id": f"u{position}", "speaker": "S1",
                "split": "train", "wav_path": f"wav/u{position}.wav",
                "spans": [["a", 0.0, 0.2, 0.9]], "offset_s": 0.0, "frames": 20,
                "logprob": -1.0, "confidence": 0.9, "duration_s": seconds}) + "\n")
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "meta"}) + "\n")
        for position in range(count):
            handle.write(json.dumps({"utt_id": f"u{position}", "pinyin": ["a1"]}) + "\n")
    length = int(seconds * SAMPLE_RATE)

    def builder(wav_path, pinyins, **kwargs):
        return StubResult(length, float(Path(wav_path).stem[1:]))

    build_pair_cache(index, manifest, root, root / "cache", pair_builder=builder)
    return root / "cache"


def make_config(cache: Path, output: Path, **overrides) -> TrainConfig:
    settings = dict(cache=str(cache), output=str(output), sample_rate=SAMPLE_RATE,
                    crop_s=0.1, batch_size=2, steps=8, learning_rate=3e-3, channels=8,
                    checkpoint_every=0, log_every=1, eval_every=0, device="cpu", seed=7)
    settings.update(overrides)
    return TrainConfig(**settings)


def test_pair_crops_are_deterministic_and_centred(tmp_path: Path) -> None:
    cache = PairCache(make_cache(tmp_path, count=2))
    first = PairCrops(cache, crop_s=0.05, sample_rate=SAMPLE_RATE, seed=11)
    second = PairCrops(cache, crop_s=0.05, sample_rate=SAMPLE_RATE, seed=11)
    assert first.length == 1200
    for _ in range(3):
        left_mech, left_real = first.sample()
        right_mech, right_real = second.sample()
        assert left_mech.shape == left_real.shape == (1200,)
        assert np.array_equal(left_mech, right_mech) and np.array_equal(left_real, right_real)
    row = cache.rows[0]
    with np.load(cache.path / row["file"]) as archive:
        real = archive["real"]
    start = (real.size - 1200) // 2
    _, centre_real = first.fixed(0)
    assert np.array_equal(centre_real, real[start:start + 1200])


def test_pair_crops_reject_an_empty_cache(tmp_path: Path) -> None:
    cache_dir = make_cache(tmp_path, count=1)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "index.jsonl").write_text(json.dumps({"type": "meta"}) + "\n", encoding="utf-8")
    assert len(PairCache(cache_dir)) == 1
    with pytest.raises(ValueError):
        PairCrops(PairCache(empty), crop_s=0.05, sample_rate=SAMPLE_RATE)


def test_overfit_run_lowers_the_loss_and_writes_checkpoints(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    output = tmp_path / "run"
    # 24 steps is the smallest budget that reliably beats the untrained loss: at 8 steps the
    # Adam warm-up still dominates and the total can drift upward.
    summary = train_model(make_config(cache, output, overfit=1, steps=24, checkpoint_every=8))
    assert summary["steps"] == 24 and summary["device"].startswith("cpu")
    assert summary["last"]["total"] < summary["first_total"]
    assert (output / "checkpoints" / "step-000008.pt").is_file()
    assert (output / "checkpoints" / "step-000024.pt").is_file()
    records = [json.loads(line) for line in
               (output / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == list(range(1, 25))
    assert "spectral_convergence" in records[-1] and "periodicity" in records[-1]
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["best_total"] == summary["best_total"]


def test_resume_continues_from_the_saved_step(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    first = train_model(make_config(cache, tmp_path / "run-a", steps=2, checkpoint_every=1))
    assert first["steps"] == 2
    checkpoint = tmp_path / "run-a" / "checkpoints" / "step-000002.pt"
    resumed = train_model(make_config(cache, tmp_path / "run-b", steps=4, checkpoint_every=2),
                          resume=str(checkpoint))
    assert resumed["steps"] == 4
    steps = [json.loads(line)["step"] for line in
             (tmp_path / "run-b" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert steps == [3, 4]
    assert (tmp_path / "run-b" / "checkpoints" / "step-000004.pt").is_file()
