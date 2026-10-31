"""Measure the EXP-0006 pairing pipeline on synthetic vowels and probe the real corpus.

The "real" signal here is produced by the same formant synthesizer, so this report only
proves plumbing and timing alignment. It says nothing about real-speech alignment quality
and nothing about naturalness.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from vhe.audio import write_wav
from vhe.data.aishell3 import file_sha256
from vhe.data.pairing import build_pair, write_pair
from vhe.data.resample import resample
from vhe.synth import (PRESETS, PhonemeSegment, PhonemeSequence, PitchCurve,
                       SynthesisRequest, Timing, get_backend)

SAMPLE_RATE = 24000
ROOT = Path(__file__).resolve().parents[2]


def render_vowel(phoneme: str, duration: float = 0.2, f0_hz: float = 200.0) -> np.ndarray:
    request = SynthesisRequest(
        template=PRESETS["yin"],
        phonemes=PhonemeSequence((PhonemeSegment(phoneme, duration),)),
        pitch=PitchCurve.constant(f0_hz, duration),
        timing=Timing(),
        sample_rate=SAMPLE_RATE,
    )
    return get_backend().render(request).audio


def make_fixture(path: Path, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Three isolated vowels with 100 ms gaps, as a stand-in for three real syllables."""
    lead = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
    gap = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
    chunks = [lead]
    for phoneme in ("a", "i", "u"):
        chunks.extend([render_vowel(phoneme), gap])
    audio = np.concatenate(chunks)
    if rate != SAMPLE_RATE:
        audio = resample(audio, SAMPLE_RATE, rate)
    write_wav(path, audio, rate)
    return audio


def summarise(result) -> dict:
    return {
        "status": result.status,
        "real_samples": None if result.real is None else int(result.real.size),
        "mechanical_samples": None if result.mechanical is None else int(result.mechanical.size),
        "equal_length": (result.real is not None and result.mechanical is not None
                         and result.real.size == result.mechanical.size),
        "issues": result.issues,
        "f0_stats": result.f0_stats,
        "segmentation_stats": result.segmentation_stats,
        "request_fingerprint": result.request_fingerprint,
        "real_sha256": result.real_sha256,
        "mechanical_sha256": result.mechanical_sha256,
    }


def gap_rms(audio: np.ndarray, start: int, stop: int) -> float:
    return float(np.sqrt(np.mean(np.square(audio[start:stop], dtype=np.float64))))


def main() -> int:
    backend = get_backend()
    report: dict = {
        "backend": {"name": backend.name, "version": backend.version,
                    "phonemes": list(backend.supported_phonemes)},
        "sample_rate": SAMPLE_RATE,
    }
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "vowels24.wav"
        make_fixture(source)
        result = build_pair(source, ["a1", "i2", "u3"], template=PRESETS["yang"])
        report["vowel_pair"] = summarise(result)
        if result.real is not None and result.mechanical is not None:
            report["vowel_pair"]["gap_window"] = [6000, 7200]
            report["vowel_pair"]["gap_rms"] = {
                "real": gap_rms(result.real, 6000, 7200),
                "mechanical": gap_rms(result.mechanical, 6000, 7200),
            }
        if result.status == "ok":
            pair_dir = tmp / "pair"
            record = write_pair(pair_dir, result)
            report["written_pair"] = {
                "pair_json_sha256": hashlib.sha256(
                    (pair_dir / "pair.json").read_bytes()).hexdigest(),
                "files": record["files"],
                "mechanical_sha256": record["mechanical_sha256"],
            }

        wide = tmp / "vowels44.wav"
        make_fixture(wide, rate=44100)
        report["resampled_pair"] = summarise(
            build_pair(wide, ["a1", "i2", "u3"], template=PRESETS["yang"]))

        report["blocked_pair"] = summarise(
            build_pair(source, ["ma1", "ni3", "guang3"], template=PRESETS["yang"]))
        report["blocked_pair"]["syllables"] = build_pair(
            source, ["ma1", "ni3", "guang3"], template=PRESETS["yang"]).syllables

    candidates = [os.environ.get("AISHELL3_ROOT"), str(ROOT / "data/raw/aishell3"),
                  str(ROOT / "data/raw/AISHELL-3"), str(Path.home() / "data/aishell3")]
    report["aishell3_probe"] = {
        str(candidate): (Path(candidate).is_dir() if candidate else False)
        for candidate in candidates if candidate
    }
    report["real_corpus_available"] = any(report["aishell3_probe"].values())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
