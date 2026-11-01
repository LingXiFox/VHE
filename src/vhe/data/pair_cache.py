"""Materialize real/mechanical training pairs from an alignment index.

Why this exists: training needs (x_mechanical, y_real) tensors, while the pairing pipeline
works one utterance at a time from a manifest row, a phone-span index and the synthesizer.
Rendering on demand inside the training loop would re-run the synthesizer and the F0 tracker
every epoch, which is slower than the model. So pairs are rendered once into a cache and the
loader only reads arrays.

Rules:
* Only training_ready pairs are written by default; rejected ones are counted and listed with
  their readiness reasons, never silently dropped.
* The cache refuses to overwrite: a new dataset is a new directory, matching the experiment
  policy that history is append-only.
* Arrays are stored as float32 at the requested rate; nothing else is stored, because the
  manifest and the alignment index are the provenance.
* The speaker/split columns are copied from the index so a split never mixes speakers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from .align_index import read_manifest as read_align_index

CACHE_VERSION = 1


@dataclass(frozen=True)
class CachedPair:
    utt_id: str
    speaker: str
    split: str
    duration_s: float
    sample_rate: int
    path: Path

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.path) as archive:
            return archive["mechanical"], archive["real"]


def _spans(row: dict) -> tuple[tuple[str, float, float, float], ...]:
    return tuple((str(phone), float(start), float(stop), float(confidence))
                 for phone, start, stop, confidence in row["spans"])


def build_pair_cache(index: str | Path, manifest: str | Path, root: str | Path,
                     output: str | Path, *, template: str = "yang",
                     sample_rate: int = 24000, splits: Sequence[str] = ("train",),
                     limit: Optional[int] = None, seed: int = 42,
                     pair_builder: Optional[Callable] = None) -> dict:
    """Render pairs for every training-ready row and write an index beside them."""
    from ..synth import PRESETS
    from .pairing import build_pair

    if pair_builder is None:
        pair_builder = build_pair
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing pair cache: {out}")
    pairs_dir = out / "pairs"
    pairs_dir.mkdir(parents=True)

    _, index_rows = read_align_index(index)
    _, manifest_rows = read_align_index(manifest)
    pinyin = {str(row["utt_id"]): [str(token) for token in row.get("pinyin", ())]
              for row in manifest_rows}
    wanted = set(splits)
    rows = [row for row in index_rows
            if row.get("type", "utterance") != "skipped" and row.get("split") in wanted]
    rows.sort(key=lambda row: str(row["utt_id"]))
    if limit is not None:
        rows = rows[:limit]

    rejected: list[dict] = []
    written: list[dict] = []
    for row in rows:
        utt_id = str(row["utt_id"])
        transcript = pinyin.get(utt_id)
        if not transcript:
            rejected.append({"utt_id": utt_id, "reason": "no transcript in manifest"})
            continue
        result = pair_builder(Path(root) / row["wav_path"], transcript,
                             template=PRESETS[template], sample_rate=sample_rate,
                             seed=seed, alignment="forced", phone_spans=_spans(row))
        ready = bool(result.readiness.get("training_ready"))
        if not ready or result.real is None or result.mechanical is None:
            rejected.append({"utt_id": utt_id, "speaker": row.get("speaker"),
                             "status": result.status,
                             "reasons": list(result.readiness.get("reasons", []))})
            continue
        real = np.ascontiguousarray(result.real, dtype=np.float32)
        mechanical = np.ascontiguousarray(result.mechanical, dtype=np.float32)
        if real.shape != mechanical.shape:
            rejected.append({"utt_id": utt_id, "reason": "length mismatch"})
            continue
        target = pairs_dir / f"{utt_id}.npz"
        with target.open("xb") as handle:
            np.savez(handle, real=real, mechanical=mechanical)
        written.append({"type": "pair", "utt_id": utt_id, "speaker": row.get("speaker"),
                        "split": row.get("split"), "file": f"pairs/{utt_id}.npz",
                        "duration_s": round(real.size / sample_rate, 4),
                        "sample_rate": sample_rate, "template": template,
                        "alignment_confidence": row.get("confidence")})

    meta = {"type": "meta", "version": CACHE_VERSION, "index": str(index),
            "manifest": str(manifest), "root": str(root), "template": template,
            "sample_rate": sample_rate, "splits": list(splits), "seed": seed,
            "requested": len(rows), "written": len(written), "rejected": len(rejected),
            "rejected_reasons": dict(Counter(code for item in rejected
                                             for code in item.get("reasons", ["(skipped)"]))),
            "rejected_rows": rejected}
    with (out / "index.jsonl").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for item in written:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return meta


class PairCache:
    """Read a pair cache: metadata, rows, and array loading in row order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta: dict = {}
        self.rows: list[dict] = []
        with (self.path / "index.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "meta":
                    self.meta = record
                else:
                    self.rows.append(record)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[CachedPair]:
        for row in self.rows:
            yield CachedPair(utt_id=row["utt_id"], speaker=row["speaker"],
                             split=row["split"], duration_s=float(row["duration_s"]),
                             sample_rate=int(row["sample_rate"]),
                             path=self.path / row["file"])

    def speakers(self) -> set[str]:
        return {row["speaker"] for row in self.rows}

    def durations(self) -> np.ndarray:
        return np.asarray([float(row["duration_s"]) for row in self.rows])
