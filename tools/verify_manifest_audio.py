"""Verify that the audio behind a manifest is present and byte-identical.

Hashes every entry marked renderable and prints a single aggregate digest, so two
machines can be compared without moving hashes around. Used to check the WSL2
training mirror on 2026-11-01 (5314 files, digest d26b7f68...).

    python tools/verify_manifest_audio.py --manifest m.jsonl --root /data/aishell3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vhe.data.corpus import read_corpus_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True, help="corpus root that wav_path is relative to")
    parser.add_argument("--all", action="store_true",
                        help="hash every entry, not only the ones marked renderable")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    manifest, root = Path(args.manifest), Path(args.root)
    entries, _meta = read_corpus_manifest(manifest)
    selected = [e for e in entries if args.all or e.renderable]
    selected.sort(key=lambda e: e.wav_path)
    if args.limit is not None:
        selected = selected[: args.limit]

    digest = hashlib.sha256()
    missing: list[str] = []
    total = 0
    for entry in selected:
        path = root / entry.wav_path
        if not path.is_file():
            missing.append(entry.wav_path)
            continue
        data = path.read_bytes()
        total += len(data)
        digest.update(f"{entry.wav_path} {len(data)} {hashlib.sha256(data).hexdigest()}\n".encode())

    report = {
        "manifest": str(manifest),
        "root": str(root),
        "hashed": len(selected) - len(missing),
        "missing": len(missing),
        "bytes": total,
        "digest": digest.hexdigest(),
        "first_missing": missing[:5],
    }
    print(json.dumps(report, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
