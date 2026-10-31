"""Dataset adapters and manifest builders (P1).

Only metadata plumbing lives here: no audio is downloaded, copied into the
repository or fed to VHE from this package.
"""

from .f0 import F0Frame, F0Track, estimate_f0
from .resample import resample
from .segment import Segment, Segmentation, check_syllable_count, segment_by_energy
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
    "F0Frame", "F0Track", "estimate_f0", "resample", "Segment", "Segmentation",
    "check_syllable_count", "segment_by_energy", "Manifest", "ParseIssue", "Utterance", "apply_speaker_split", "build_manifest",
    "carve_speaker_split", "check_no_speaker_leak", "discover_splits", "read_manifest",
    "write_manifest",
]
