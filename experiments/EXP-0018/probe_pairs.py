"""EXP-0018: a diverse, small set of real/mechanical pairs with diagnostic previews.

The P1.5 brief allows at most twenty real pairs, forbids batch generation, and forbids
treating energy segmentation as a phoneme-level forced alignment. This probe therefore:

* picks one utterance per speaker from dev/test (never train), greedily maximising the
  number of Mandarin finals and onsets that the sample exercises, so the twenty pairs
  cover the syllable inventory instead of repeating the same vowel twenty times;
* builds each pair with the viterbi tracker and the energy-DP aligner, records every
  boundary confidence, per-syllable pitch coverage and readiness reason, and never marks
  a pair training-ready on the strength of status alone;
* writes one preview folder per pair (real audio, mechanical audio, diagnostic image) plus
  a listening sheet that stays explicitly "not yet accepted" until a human listens.

Audio and images live outside the repository, under the dataset workspace, because the
real half is licensed AISHELL-3 audio and must not enter Git.

Usage:
    python experiments/EXP-0018/probe_pairs.py --manifest <manifest.jsonl> \
        --out-root /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-pairs/v1 \
        --report experiments/EXP-0018/logs/pair_report_v1.json \
        --sheet experiments/EXP-0018/logs/listening_sheet_v1.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vhe.audio import read_wav, write_wav  # noqa: E402
from vhe.data.corpus import read_corpus_manifest  # noqa: E402
from vhe.data.pairing import build_pair  # noqa: E402
from vhe.data.pitch import track_pitch  # noqa: E402
from vhe.data.resample import resample  # noqa: E402
from vhe.synth import PRESETS  # noqa: E402
from vhe.synth.registry import get_backend  # noqa: E402

SR = 24000
BACKGROUND = (12, 12, 18)
PANEL = (24, 24, 32)
WHITE = (235, 235, 240)
GREY = (120, 120, 130)
ORANGE = (250, 170, 60)
CYAN = (90, 200, 235)
RED = (235, 90, 90)
GREEN = (120, 220, 130)

# A 5x7 bitmap font: without Pillow or matplotlib a label has to be drawn by hand. Only the
# characters used by the panel labels are present; unknown characters render as blanks.
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def write_png(path: Path, rgb: np.ndarray) -> str:
    """Write an HxWx3 uint8 array as a PNG and return its sha256."""
    height, width, _ = rgb.shape
    rows = b"".join(bytes([0]) + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (bytes([137, 80, 78, 71, 13, 10, 26, 10])
            + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 6))
            + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def fill_rect(img: np.ndarray, x: int, y: int, width: int, height: int, color) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1 = min(img.shape[1], x + width)
    y1 = min(img.shape[0], y + height)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = color


def draw_text(img: np.ndarray, x: int, y: int, text: str, scale: int = 2, color=WHITE) -> None:
    cursor = x
    for character in text.upper():
        glyph = FONT.get(character, FONT[" "])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    fill_rect(img, cursor + column * scale, y + row * scale, scale, scale, color)
        cursor += (5 + 1) * scale


def hline(img: np.ndarray, x0: int, x1: int, y: int, color) -> None:
    fill_rect(img, x0, y, max(0, x1 - x0), 1, color)


def vline(img: np.ndarray, x: int, y0: int, y1: int, color, width: int = 1) -> None:
    fill_rect(img, x, y0, width, max(0, y1 - y0), color)


def colormap(values: np.ndarray) -> np.ndarray:
    """Map 0..1 to a dark-blue to orange to white ramp, without a plotting library."""
    stops = np.array([[0.0, 12.0, 12.0, 40.0], [0.35, 40.0, 60.0, 140.0],
                      [0.60, 200.0, 120.0, 40.0], [0.85, 240.0, 200.0, 80.0],
                      [1.0, 255.0, 255.0, 255.0]])
    positions = stops[:, 0]
    channels = [np.interp(values, positions, stops[:, index]) for index in (1, 2, 3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def spectrogram(audio: np.ndarray, sample_rate: int, *, window_ms: float = 25.0,
                hop_ms: float = 5.0, max_hz: float = 8000.0):
    window = int(round(sample_rate * window_ms / 1000.0))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    signal = np.asarray(audio, dtype=np.float64)
    if signal.size < window:
        signal = np.pad(signal, (0, window - signal.size))
    starts = np.arange(0, signal.size - window + 1, hop)
    tapered = np.stack([signal[start:start + window] * np.hanning(window) for start in starts])
    magnitude = np.abs(np.fft.rfft(tapered, axis=1))
    freqs = np.fft.rfftfreq(window, 1.0 / sample_rate)
    keep = freqs <= max_hz
    db = 20.0 * np.log10(magnitude[:, keep] + 1e-9)
    db -= db.max()
    return np.clip((db + 80.0) / 80.0, 0.0, 1.0).T, (starts + window / 2.0) / sample_rate, freqs[keep]


def draw_waveform(img: np.ndarray, box, audio: np.ndarray, sample_rate: int,
                  boundaries: list, color) -> None:
    x0, y0, width, height = box
    fill_rect(img, x0, y0, width, height, PANEL)
    duration = audio.size / sample_rate
    mid = y0 + height // 2
    hline(img, x0, x0 + width, mid, GREY)
    samples = np.asarray(audio, dtype=np.float64)
    peak = max(float(np.max(np.abs(samples))), 1e-6)
    for column in range(width):
        start = int(column / width * samples.size)
        stop = max(start + 1, int((column + 1) / width * samples.size))
        piece = samples[start:min(stop, samples.size)]
        if piece.size == 0:
            continue
        high = mid - int(piece.max() / peak * (height / 2 - 2))
        low = mid - int(piece.min() / peak * (height / 2 - 2))
        fill_rect(img, x0 + column, min(high, low), 1, max(1, abs(low - high)), color)
    for start_s, end_s in boundaries:
        vline(img, x0 + int(start_s / duration * width), y0, y0 + height, ORANGE, 2)


def draw_f0(img: np.ndarray, box, tracks: list, rejected: np.ndarray,
            duration: float) -> None:
    x0, y0, width, height = box
    fill_rect(img, x0, y0, width, height, PANEL)
    low, high = 60.0, 500.0

    def to_y(f0):
        return y0 + height - int((np.log(f0) - np.log(low)) / (np.log(high) - np.log(low))
                                 * height)

    for reference in (65, 100, 150, 200, 300, 400):
        hline(img, x0, x0 + width, to_y(reference), (40, 40, 52))
        draw_text(img, x0 + 2, to_y(reference) - 14, str(reference), 1, GREY)
    for times, values, color in tracks:
        for time_s, f0 in zip(times, values):
            if f0 <= 0.0:
                continue
            x = x0 + int(time_s / duration * width)
            y = to_y(float(f0))
            fill_rect(img, x, y, 2, 2, color)
    for time_s in rejected:
        x = x0 + int(time_s / duration * width)
        fill_rect(img, x, y0 + height - 4, 1, 4, RED)


def draw_spectrogram(img: np.ndarray, box, audio: np.ndarray, sample_rate: int,
                     boundaries: list) -> None:
    x0, y0, width, height = box
    fill_rect(img, x0, y0, width, height, PANEL)
    values, _, _ = spectrogram(audio, sample_rate)
    rows = np.linspace(0, values.shape[0] - 1, height).astype(int)
    columns = np.linspace(0, values.shape[1] - 1, width).astype(int)
    image = colormap(values[np.ix_(rows, columns)])
    img[y0:y0 + height, x0:x0 + width] = image
    duration = audio.size / sample_rate
    for start_s, end_s in boundaries:
        vline(img, x0 + int(start_s / duration * width), y0, y0 + height, CYAN, 1)
    draw_text(img, x0 - 62, y0, "8 KHZ", 1, GREY)
    draw_text(img, x0 - 62, y0 + height - 8, "0", 1, GREY)


def diagnostic_image(path: Path, real: np.ndarray, mechanical: np.ndarray, sample_rate: int,
                     result, label: str) -> str:
    """Four panels: real waveform, F0s, real spectrogram, mechanical spectrogram."""
    width, panel_height, margin, top = 1000, 190, 76, 40
    height = top + 4 * panel_height + 3 * 12 + 30
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = BACKGROUND
    draw_text(img, margin, 12, label, 2, WHITE)
    draw_text(img, margin, 28,
              "REAL WAVEFORM AND F0: GREEN=VITERBI ORANGE=BOUNDARIES RED=REJECTED",
              1, GREY)
    boundaries = [(item["start_s"], item["end_s"])
                  for item in (result.alignment or {}).get("boundaries", [])]
    duration = real.size / sample_rate
    draw_waveform(img, (margin, top, width - 2 * margin, panel_height), real,
                  sample_rate, boundaries, WHITE)
    track = track_pitch(real, sample_rate)
    times = np.array([frame.time_s for frame in track.frames])
    f0 = np.array([frame.f0_hz for frame in track.frames])
    rejected = np.array([frame.time_s for frame in track.frames if frame.reason != "ok"])
    mechanical_track = track_pitch(mechanical, sample_rate)
    mechanical_times = np.array([frame.time_s for frame in mechanical_track.frames])
    mechanical_f0 = np.array([frame.f0_hz for frame in mechanical_track.frames])
    draw_f0(img, (margin, top + panel_height + 12, width - 2 * margin, panel_height),
            [(times, f0, GREEN), (mechanical_times, mechanical_f0, CYAN)], rejected,
            duration)
    draw_text(img, margin, top + panel_height + 12 + 4,
              "F0 HZ: GREEN=REAL CYAN=MECHANICAL", 1, GREY)
    draw_spectrogram(img, (margin, top + 2 * (panel_height + 12),
                           width - 2 * margin, panel_height), real, sample_rate, boundaries)
    draw_spectrogram(img, (margin, top + 3 * (panel_height + 12),
                           width - 2 * margin, panel_height), mechanical, sample_rate,
                     boundaries)
    draw_text(img, margin, height - 20,
              "PANELS: REAL WAVE, F0, REAL SPECTROGRAM, MECHANICAL SPECTROGRAM", 1, GREY)
    return write_png(path, img)


def load_gender(spk_info: Path) -> dict:
    genders = {}
    if not spk_info.exists():
        return genders
    for line in spk_info.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(chr(9)) if part.strip()]
        if len(parts) >= 3:
            genders[parts[0]] = parts[2]
    return genders


def select_entries(entries, genders: dict, count: int, seed: int):
    """Greedy coverage of finals and onsets, one utterance per speaker first."""
    rng = np.random.default_rng(seed)
    pool = [entry for entry in entries
            if entry.split in ("dev", "test") and entry.renderable and not entry.quality_flags
            and 1.0 <= entry.duration_s <= 3.0 and 2 <= len(entry.pinyin) <= 6]
    if not pool:
        raise SystemExit("no eligible dev/test utterances; is the manifest v4 renderable?")
    order = rng.permutation(len(pool))
    pool = [pool[index] for index in order]
    chosen, seen_finals, seen_onsets, speakers = [], set(), set(), set()
    gender_counts = Counter()
    while len(chosen) < count and pool:
        best, best_score = None, -1.0
        for entry in pool:
            if entry.speaker in speakers:
                continue
            new_finals = len(set(entry.finals) - seen_finals)
            new_onsets = len(set(entry.onsets) - seen_onsets)
            balance = 0.5 if gender_counts[genders.get(entry.speaker, "unknown")] == 0 else 0.0
            score = 2.0 * new_finals + new_onsets + balance - 0.01 * entry.duration_s
            if score > best_score:
                best, best_score = entry, score
        if best is None:
            break
        chosen.append(best)
        speakers.add(best.speaker)
        seen_finals.update(best.finals)
        seen_onsets.update(best.onsets)
        gender_counts[genders.get(best.speaker, "unknown")] += 1
        pool = [entry for entry in pool if entry.utt_id != best.utt_id]
    return chosen, sorted(seen_finals), sorted(seen_onsets)


def choose_template(median_f0: float) -> str:
    """Crude pitch-guided template choice, recorded so EXP-0019 can test it properly."""
    return "yang" if median_f0 >= 200.0 else "yin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--spk-info", type=Path, default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="v1", help="version tag written into the sheet")
    args = parser.parse_args()
    for path in (args.report, args.sheet):
        if path.exists():
            parser.error(f"refusing to overwrite an existing record: {path}")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        parser.error(f"refusing to write previews into a non-empty directory: {args.out_root}")

    entries, meta = read_corpus_manifest(args.manifest)
    root = Path(meta["root"])
    spk_info = args.spk_info or (root / "spk-info.txt")
    genders = load_gender(spk_info)
    chosen, finals, onsets = select_entries(entries, genders, args.count, args.seed)

    records = []
    for index, entry in enumerate(chosen, start=1):
        sample_id = f"sample_{index:02d}"
        folder = args.out_root / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        real, source_rate = read_wav(root / entry.wav_path)
        real = resample(real, source_rate, SR)
        preview_track = track_pitch(real, SR)
        median_f0 = (float(np.median(preview_track.voiced_f0))
                     if preview_track.voiced_f0.size else 0.0)
        template_name = choose_template(median_f0)
        # The Mandarin backend must be named explicitly: the registry default is the
        # FormantProxySynth with the five-vowel inventory, which blocks every real syllable
        # (this exact mistake produced the v1 run preserved beside v2 in the dataset folder).
        result = build_pair(root / entry.wav_path, list(entry.pinyin),
                            template=PRESETS[template_name],
                            backend=get_backend("mandarin_formant"), sample_rate=SR,
                            seed=args.seed + index, pitch_estimator="viterbi",
                            alignment="dp")
        record = {
            "sample_id": sample_id, "utt_id": entry.utt_id, "speaker": entry.speaker,
            "gender": genders.get(entry.speaker, "unknown"),
            "split": entry.split, "duration_s": round(entry.duration_s, 3),
            "pinyin": list(entry.pinyin), "text": entry.text,
            "onsets": list(entry.onsets), "finals": list(entry.finals),
            "tones": list(entry.tones),
            "template": template_name, "template_choice_basis": "median f0 >= 200 Hz",
            "measured_median_f0_hz": round(median_f0, 1),
            "status": result.status,
            "readiness": result.readiness,
            "issues": result.issues,
            "alignment": {key: value for key, value in (result.alignment or {}).items()
                          if key != "boundaries"},
            "boundaries": (result.alignment or {}).get("boundaries", []),
            "pitch": result.pitch,
            "f0_stats": result.f0_stats,
            "gains_db": result.gains_db,
            "allocations": result.allocations,
            "real_sha256": result.real_sha256,
            "mechanical_sha256": result.mechanical_sha256,
        }
        if result.real is not None and result.mechanical is not None:
            real_path = folder / "real.wav"
            mechanical_path = folder / "mechanical.wav"
            if not real_path.exists():
                write_wav(real_path, result.real, SR)
            if not mechanical_path.exists():
                write_wav(mechanical_path, result.mechanical, SR)
            record["real_wav"] = str(real_path)
            record["mechanical_wav"] = str(mechanical_path)
            record["image"] = str(folder / "diagnostics.png")
            record["image_sha256"] = diagnostic_image(
                folder / "diagnostics.png", result.real, result.mechanical, SR, result,
                f"{sample_id} {entry.utt_id} {entry.speaker} {template_name}")
        (folder / "notes.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
        records.append(record)
        print(f"{sample_id} {entry.utt_id} {entry.speaker} {result.status} "
              f"ready={result.readiness.get('training_ready')} "
              f"reasons={','.join(result.readiness.get('reasons', []))}")

    ready = [record for record in records if record["readiness"].get("training_ready")]
    reasons = Counter(reason for record in records
                      for reason in record["readiness"].get("reasons", []))
    coverages = [window.get("coverage") for record in records
                 for window in record["pitch"].get("syllables", [])
                 if window.get("coverage") is not None]
    confidences = [float(boundary.get("confidence")) for record in records
                   for boundary in record["boundaries"]
                   if boundary.get("confidence") is not None]
    report = {
        "experiment": "EXP-0018",
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "out_root": str(args.out_root),
        "requested": args.count, "built": len(records),
        "seed": args.seed,
        "selection": {"finals_covered": finals, "onsets_covered": onsets,
                      "speakers": len({record["speaker"] for record in records}),
                      "genders": dict(Counter(record["gender"] for record in records)),
                      "pitch_estimator": "viterbi", "alignment": "dp",
                      "split": "dev+test only"},
        "status_counts": dict(Counter(record["status"] for record in records)),
        "training_ready": len(ready),
        "training_ready_ids": [record["sample_id"] for record in ready],
        "readiness_reasons": dict(sorted(reasons.items())),
        "boundary_confidence": {
            "count": len(confidences),
            "median": round(float(np.median(confidences)), 3) if confidences else None,
            "min": round(float(np.min(confidences)), 3) if confidences else None,
            "below_0_35": sum(1 for value in confidences if value < 0.35)},
        "pitch_coverage": {
            "windows": len(coverages),
            "median": round(float(np.median(coverages)), 3) if coverages else None,
            "below_0_5": sum(1 for value in coverages if value < 0.5),
            "zero": sum(1 for value in coverages if value == 0.0)},
        "median_f0_hz": {record["sample_id"]: record["measured_median_f0_hz"]
                         for record in records},
        "gains_db_range": [round(min((value for record in records
                                      for value in record["gains_db"]), default=0.0), 2),
                           round(max((value for record in records
                                      for value in record["gains_db"]), default=0.0), 2)],
        "samples": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + chr(10),
                           encoding="utf-8")

    lines = [f"# EXP-0018 listening sheet ({args.label})", "",
             "Status: **not yet accepted by a human listener**. Every row below is unscored "
             "until the owner listens; no claim about naturalness may be made before that.",
             "",
             f"Previews: {args.out_root} (outside Git; the real half is licensed audio).",
             "Each folder holds real.wav, mechanical.wav, diagnostics.png (real waveform, "
             "F0: green=real / cyan=mechanical / red=rejected frames, real spectrogram, "
             "mechanical spectrogram) and notes.json.",
             "",
             "Rate each row 1-5 (1 unusable, 5 indistinguishable) or write a comment; "
             "empty means untested.",
             "",
             "| id | utt | speaker | gender | pinyin | template | ready | timbre | naturalness | "
             "content | notes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for record in records:
        lines.append(
            f"| {record['sample_id']} | {record['utt_id']} | {record['speaker']} | "
            f"{record['gender']} | {' '.join(record['pinyin'])} | {record['template']} | "
            f"{'yes' if record['readiness'].get('training_ready') else 'no'} | | | | |")
    lines += ["", "## Failure reasons recorded for each sample", ""]
    for record in records:
        reasons_text = ", ".join(record["readiness"].get("reasons", [])) or "none"
        codes = ", ".join(sorted({issue["code"] for issue in record["issues"]})) or "none"
        lines.append(f"- {record['sample_id']} {record['utt_id']}: {reasons_text}; "
                     f"issues: {codes}")
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    args.sheet.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("built", "status_counts", "training_ready", "readiness_reasons",
                       "boundary_confidence", "pitch_coverage", "gains_db_range")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
