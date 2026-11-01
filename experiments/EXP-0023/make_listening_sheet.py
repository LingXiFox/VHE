#!/usr/bin/env python
"""Turn a listening package into a single A/B sheet and one audio file per pair.

The package from build_listening_package.py has one directory per utterance with real.wav and
mech_forced.wav. Listening to 48 separate files invites ordering mistakes, so this script writes
ab/<utt>.wav = human (1.0 s gap) mechanical and a sheet.md that lists, per pair, the speaker, the
pinyin transcript and the automatic metrics recorded by the builder.

Only pairs whose forced arm is training-ready are listed: the P2 listening gate judges pairs that
could actually enter training. Rejected pairs stay in the package for diagnosis, not for the gate.

Usage:
    python experiments/EXP-0023/make_listening_sheet.py \
        --package /path/derived/aishell3-listen/EXP-0023-run5 \
        --manifest /path/derived/aishell3-scan/v4/manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vhe.audio import read_wav, write_wav

GAP_S = 1.0


def load_table(package: Path) -> list[dict]:
    with open(package / "table.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def transcripts(manifest: Path) -> dict[str, list[str]]:
    mapping = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "meta":
                continue
            mapping[str(row["utt_id"])] = [str(token) for token in row.get("pinyin", ())]
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    package = Path(args.package)
    rows = [row for row in load_table(package) if row.get("forced_ready") == "True"]
    rows.sort(key=lambda row: row["utt_id"])
    if not rows:
        parser.exit(2, "no training-ready pairs in this package\n")
    pinyin = transcripts(Path(args.manifest))
    ab = package / "ab"
    ab.mkdir(exist_ok=True)

    lines = ["# EXP-0023 listening sheet", "",
             f"{len(rows)} training-ready pairs. Each ab/<utt>.wav is the human recording "
             f"followed by {GAP_S:.1f} s of silence and then the mechanical rendering.",
             "",
             "| # | utt | speaker | pinyin | align conf | log-mel dist |",
             "|---|---|---|---|---|---|"]
    for index, row in enumerate(rows, 1):
        utt = row["utt_id"]
        real, rate = read_wav(package / utt / "real.wav")
        mech, mech_rate = read_wav(package / utt / "mech_forced.wav")
        if mech_rate != rate:
            parser.exit(2, f"{utt}: rate mismatch {rate} vs {mech_rate}\n")
        gap = np.zeros(int(GAP_S * rate), dtype=np.float32)
        write_wav(ab / f"{utt}.wav", np.concatenate([real, gap, mech]), rate)
        words = " ".join(pinyin.get(utt, []))
        lines.append(f"| {index} | {utt} | {row['speaker']} | {words} | "
                     f"{row['alignment_confidence']} | {row.get('logmel_forced', '')} |")
    (package / "sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} A/B files and sheet.md in {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
