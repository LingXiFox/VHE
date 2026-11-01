"""Join real and mechanical renders into one A/B file per utterance (EXP-0029).

Writes ab/<utt>.wav = real + 0.7 s of silence + robot, so a listener can compare
without switching players. Refuses to overwrite an existing ab/ directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if len(Path(__file__).resolve().parents) >= 3 and (Path(__file__).resolve().parents[2] / "src").is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.audio import read_wav, write_wav  # noqa: E402

GAP_SECONDS = 0.7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (args.package / "ab")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)

    lines = []
    for directory in sorted(p for p in args.package.iterdir() if p.is_dir() and p.name != "ab"):
        real, rate = read_wav(directory / "real.wav")
        robot, other_rate = read_wav(directory / "robot.wav")
        if other_rate != rate:
            raise SystemExit(f"sample rate mismatch in {directory}")
        gap = np.zeros(int(GAP_SECONDS * rate), dtype=np.float32)
        combined = np.concatenate([np.asarray(real, dtype=np.float32), gap,
                                   np.asarray(robot, dtype=np.float32)])
        write_wav(out / f"{directory.name}.wav", combined, rate)
        lines.append(f"| {directory.name} | {len(combined) / rate:.2f} s |")
        print("wrote", out / f"{directory.name}.wav")

    (out / "sheet.md").write_text(
        "# EXP-0029 A/B sheet\n\nEach file: real speech, 0.7 s silence, then the\n"
        "analysis-resynthesis mechanical copy.\n\n| utterance | length |\n|---|---|\n"
        + "\n".join(lines) + "\n")
    (args.package / "ab_manifest.json").write_text(json.dumps(
        {"gap_seconds": GAP_SECONDS, "files": [str(p.name) for p in sorted(out.glob('*.wav'))]},
        indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
