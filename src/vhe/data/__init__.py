"""Dataset adapters and manifest builders (P1).

Only metadata plumbing lives here: no audio is downloaded, copied into the
repository or fed to VHE from this package.
"""

from .aishell3 import (
    Manifest,
    ParseIssue,
    Utterance,
    apply_speaker_split,
    build_manifest,
    carve_speaker_split,
    check_no_speaker_leak,
    discover_splits,
    read_manifest,
    write_manifest,
)

__all__ = [
    "Manifest", "ParseIssue", "Utterance", "apply_speaker_split", "build_manifest",
    "carve_speaker_split", "check_no_speaker_leak", "discover_splits", "read_manifest",
    "write_manifest",
]
