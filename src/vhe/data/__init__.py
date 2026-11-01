"""Dataset adapters and manifest builders (P1).

Only metadata plumbing lives here: no audio is downloaded, copied into the
repository or fed to VHE from this package.
"""

from .f0 import F0Frame, F0Track, estimate_f0
from .resample import resample
from .segment import Segment, Segmentation, check_syllable_count, segment_by_energy
from .pairing import (
    PairResult,
    Syllable,
    build_pair,
    parse_syllable,
    write_pair,
)
from .aishell3 import (
    DEFAULT_SPLIT_SIZES,
    SPLIT_NAMES,
    SPLIT_SEED,
    Manifest,
    ParseIssue,
    Utterance,
    apply_speaker_split,
    assign_speaker_split,
    auto_split_sizes,
    build_manifest,
    check_no_speaker_leak,
    discover_splits,
    parse_split_sizes,
    read_manifest,
    write_manifest,
)

__all__ = [
    "F0Frame", "F0Track", "estimate_f0", "resample", "Segment", "Segmentation",
    "check_syllable_count", "segment_by_energy", "Manifest", "ParseIssue", "Utterance",
    "apply_speaker_split", "build_manifest", "assign_speaker_split", "auto_split_sizes",
    "parse_split_sizes", "DEFAULT_SPLIT_SIZES", "SPLIT_SEED", "SPLIT_NAMES",
    "check_no_speaker_leak", "discover_splits", "read_manifest", "write_manifest",
    "PairResult", "Syllable", "build_pair", "parse_syllable", "write_pair",
]
