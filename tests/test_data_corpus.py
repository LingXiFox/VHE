"""Corpus scan tests on a synthetic AISHELL-3 layout.

The fixture is four speakers spread over the official train/ and test/ directories, so the
tests can prove three things without the 26 GB corpus: the speaker split ignores the
official directory, the 26-phoneme inventory blocks a syllable without repairing it, and
the light manifest round-trips. Audio is written as a real 44.1 kHz PCM header because the
scan reads headers, not samples.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vhe.audio import write_wav
from vhe.data.corpus import (
    MANIFEST_VERSION,
    CorpusStats,
    annotate_manifest,
    apply_quality_flags,
    audit_prosody_labels,
    coverage_report,
    parse_label_pinyin,
    rank_next_units,
    read_corpus_manifest,
    scan_corpus,
    write_corpus_manifest,
)
from vhe.data.mandarin import split_token

CONTENT_TRAIN = """\
SSB00050001.wav\t大 da4 马 ma3
SSB00050002.wav\t江 jiang1 山 shan1
SSB00060001.wav\t爸 ba4 妈 ma1
"""
CONTENT_TEST = """\
SSB00070001.wav\t他 ta1 走 zou3
SSB00080001.wav\t你 ni3 拿 na2 吗 ma5
"""
SPK_INFO = """\
# voice-file name; age group; gender; accent
SSB0005\tB\tfemale\tnorth
SSB0006\tC\tmale\tnorth
SSB0007\tB\tfemale\tsouth
SSB0008\tC\tmale\tsouth
"""
LABELS = """\
SSB00050001|da4 ma3|大马
SSB00050002|jiang1 shan2|江山
SSB00060001|ba4 ma1|爸妈
"""


def write_wav_header(path: Path, *, seconds: float = 0.5, sample_rate: int = 44100,
                     channels: int = 1, sample_width: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    if (channels, sample_width, sample_rate) == (1, 2, 44100):
        time = np.arange(frames) / sample_rate
        write_wav(path, 0.2 * np.sin(2 * np.pi * 200.0 * time), sample_rate)
        return
    payload = np.zeros((frames, channels), dtype="<i2")
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(payload.tobytes())


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "aishell3"
    for name, content in (("train", CONTENT_TRAIN), ("test", CONTENT_TEST)):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "content.txt").write_text(content, encoding="utf-8")
    (root / "spk-info.txt").write_text(SPK_INFO, encoding="utf-8")
    (root / "train" / "label_train-set.txt").write_text(LABELS, encoding="utf-8")
    for source_dir, body in (("train", CONTENT_TRAIN), ("test", CONTENT_TEST)):
        for line in body.strip().splitlines():
            name = line.split("\t", 1)[0].replace(".wav", "")
            write_wav_header(root / source_dir / "wav" / name[:7] / f"{name}.wav")
    return root


def test_scan_assigns_speaker_split_ignoring_official_dir(corpus_root: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    assert index.split_seed == 20261101
    assert len(index.entries) == 5
    assert index.stats.speakers == 4

    splits = index.speakers_by_split()
    assert {name: len(names) for name, names in splits.items()} == {
        "train": 2, "dev": 1, "test": 1}
    # A speaker's utterances never straddle two splits, whatever their official directory.
    per_speaker: dict[str, set[str]] = {}
    for entry in index.entries:
        per_speaker.setdefault(entry.speaker, set()).add(entry.split)
    assert all(len(names) == 1 for names in per_speaker.values())
    # The official directories are kept only as provenance.
    assert {entry.source_dir for entry in index.entries} == {"train", "test"}


def test_scan_blocks_unsupported_final_without_repair(corpus_root: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    blocked = [entry for entry in index.entries if not entry.renderable]
    assert [entry.utt_id for entry in blocked] == ["SSB00050002"]
    assert blocked[0].finals == ("iang", "an")
    assert index.stats.blocked_units["final:iang"] == 1
    assert index.stats.syllables == 11
    assert index.stats.blocked_syllables == 1
    assert index.stats.syllable_coverage == pytest.approx(10 / 11)
    assert index.stats.utterance_coverage == pytest.approx(4 / 5)


def test_scan_counts_tones_onsets_and_segments(corpus_root: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    stats = index.stats
    assert stats.tones[1] + stats.tones[2] + stats.tones[3] + stats.tones[4] + \
        stats.tones[5] == stats.syllables
    assert stats.tokens["ma3"] == 1 and stats.tokens["ma1"] == 1 and stats.tokens["ma5"] == 1
    assert stats.onsets["-"] == 0
    assert stats.finals["a"] == 7  # da4 ma3 ba4 ma1 ta1 na2 ma5
    assert stats.segments["n"] == 1  # shan1 coda
    assert stats.erhua == 0
    assert sum(entry.duration_s for entry in index.entries) == pytest.approx(
        stats.total_duration_s)


def test_scan_reports_missing_wav_and_keeps_speaker(corpus_root: Path) -> None:
    (corpus_root / "train" / "wav" / "SSB0005" / "SSB00050001.wav").unlink()
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    codes = [issue.code for issue in index.errors]
    assert codes == ["missing_wav"]
    assert index.stats.speakers == 4  # the split is decided over all speakers first
    assert len(index.entries) == 4


def test_scan_reports_bad_pinyin(corpus_root: Path) -> None:
    lines = CONTENT_TRAIN.replace("大 da4", "大 dax4")
    (corpus_root / "train" / "content.txt").write_text(lines, encoding="utf-8")
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    assert [issue.code for issue in index.errors] == ["bad_pinyin"]
    assert "dax4" in index.errors[0].detail


def test_scan_is_deterministic(corpus_root: Path) -> None:
    first = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    second = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    assert [(entry.utt_id, entry.split) for entry in first.entries] == \
        [(entry.utt_id, entry.split) for entry in second.entries]


def test_manifest_round_trip_and_refuses_overwrite(corpus_root: Path, tmp_path: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    target = tmp_path / "manifest.jsonl"
    digest = write_corpus_manifest(target, index)
    assert len(digest) == 64
    entries, meta = read_corpus_manifest(target)
    assert entries == index.entries
    assert meta["split_sizes"] == {"train": 2, "dev": 1, "test": 1}
    assert meta["utterances"] == 5
    assert meta["version"] == MANIFEST_VERSION
    with pytest.raises(FileExistsError):
        write_corpus_manifest(target, index)


def test_manifest_lines_hold_no_absolute_path(corpus_root: Path, tmp_path: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    target = tmp_path / "manifest.jsonl"
    write_corpus_manifest(target, index)
    for line in target.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["type"] == "meta":
            continue
        assert not Path(payload["wav_path"]).is_absolute()
        assert payload["wav_path"].startswith(("train/", "test/"))


def test_rank_next_units_prefers_the_biggest_gain() -> None:
    stats = CorpusStats()
    syllables = [("jiang1", ("final:iang",)), ("jiang1", ("final:iang",)),
                 ("zhi4", ("final:apical_retroflex", "onset:zh")),
                 ("huan4", ("onset:h",))]
    for written, missing in syllables:
        stats.add_syllable(split_token(written), tuple(missing))
    ranking = rank_next_units(stats, top=4)
    # 'final:iang' finishes two syllables; 'onset:h' finishes 'huan4'; then one of the two
    # units that finish 'zhi4' (alphabetical tie-break). No unit is picked for a syllable
    # it cannot finish on its own while a finishing unit is still available.
    assert [(item["unit"], item["syllables_unblocked"], item["cumulative_unblocked"])
            for item in ranking] == [("final:iang", 2, 2), ("onset:h", 1, 3),
                                     ("final:apical_retroflex", 0, 3), ("onset:zh", 1, 4)]
    # A unit that cannot finish a syllable alone says which partner it still needs.
    assert ranking[2]["also_needs"] == ["onset:zh"]
    assert ranking[0]["syllables_unblocked_alone"] == 2


def test_rank_next_units_reports_erhua_separately() -> None:
    stats = CorpusStats()
    stats.add_syllable(split_token("jinr1"), ("final:in", "erhua"))
    ranking = rank_next_units(stats, top=5)
    assert [item["unit"] for item in ranking] == ["final:in", "erhua"]
    assert ranking[-1]["note"].startswith("needs an erhua final unit")


def test_parse_label_pinyin_drops_pause_markers() -> None:
    assert parse_label_pinyin("chi2 qi3 % hong2 ying1 $") == (
        "chi2", "qi3", "hong2", "ying1")


def test_audit_prosody_labels_reports_the_tone_disagreement(corpus_root: Path) -> None:
    report = audit_prosody_labels(corpus_root)
    assert report["checked"] == 3
    assert report["matched"] == 2
    assert report["mismatched"] == 1
    mismatch = report["mismatches"][0]
    assert mismatch["utt_id"] == "SSB00050002"
    assert mismatch["differences"] == [{"index": 1, "label": "shan2", "content": "shan1"}]

def test_scan_cli_writes_manifest_and_report(corpus_root: Path, tmp_path: Path,
                                             capsys, monkeypatch) -> None:
    from vhe.cli import main

    manifest, report = tmp_path / "m.jsonl", tmp_path / "r.json"
    monkeypatch.setattr("sys.argv", ["vhe", "scan", "--root", str(corpus_root),
                                     "--manifest", str(manifest), "--report", str(report),
                                     "--split-sizes", "2/1/1", "--top", "3"])
    main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["utterances"] == 5
    assert printed["errors"] == 0
    assert manifest.is_file() and report.is_file()
    body = json.loads(report.read_text(encoding="utf-8"))
    assert body["speakers_per_split"] == {"train": 2, "dev": 1, "test": 1}
    assert 0.0 < body["stats"]["syllable_coverage"] < 1.0
    assert 1 <= len(body["next_units"]) <= 3  # the fixture blocks exactly one final


def test_scan_cli_checks_both_outputs_before_scanning(corpus_root: Path, tmp_path: Path,
                                                      capsys, monkeypatch) -> None:
    from vhe.cli import main

    manifest, report = tmp_path / "m.jsonl", tmp_path / "r.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["vhe", "scan", "--root", str(corpus_root),
                                     "--manifest", str(manifest), "--report", str(report)])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    # A pre-existing report must not leave a half-written manifest behind.
    assert not manifest.exists()

def test_audit_classifies_every_anomaly_with_line_and_raw_text(corpus_root: Path) -> None:
    labels = "# header comment line\n" + LABELS + "SSB00050003|ma1 ma1\n" + \
        "SSB99990001|da4 ma3|大马\n"
    (corpus_root / "train" / "label_train-set.txt").write_text(labels, encoding="utf-8")
    report = audit_prosody_labels(corpus_root)

    assert (report["checked"], report["matched"], report["mismatched"]) == (3, 2, 1)
    assert report["anomaly_counts"] == {"prosody_conflict": 1, "format_error": 1,
                                        "missing_text": 1, "header_comment": 1}
    assert [(item["kind"], item["utt_id"], item["line"]) for item in report["anomalies"]] == [
        ("header_comment", None, 1), ("prosody_conflict", "SSB00050002", 3),
        ("format_error", "SSB00050003", 5), ("missing_text", "SSB99990001", 6)]
    # A line with no legal record keeps its raw text; nothing is dropped from the evidence.
    assert report["anomalies"][3]["raw"] == "SSB99990001|da4 ma3|大马"
    assert report["anomalies"][2]["detail"].startswith("expected 3 '|'-separated fields")
    assert report["flags"] == {"SSB00050002": ("prosody_conflict",),
                               "SSB00050003": ("label_format_error",),
                               "SSB99990001": ("missing_content",)}


def test_apply_quality_flags_does_not_mutate_or_duplicate(corpus_root: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    flagged = apply_quality_flags(index.entries, {"SSB00050001": ("prosody_conflict",)})
    assert index.entries[0].quality_flags == ()
    by_id = {entry.utt_id: entry for entry in flagged}
    assert by_id["SSB00050001"].quality_flags == ("prosody_conflict",)
    again = apply_quality_flags(flagged, {"SSB00050001": ("prosody_conflict",)})
    assert [entry.quality_flags for entry in again] == [entry.quality_flags
                                                        for entry in flagged]


def test_annotate_manifest_carries_flags_without_rescanning(corpus_root: Path,
                                                            tmp_path: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    plain = tmp_path / "v1.jsonl"
    digest = write_corpus_manifest(plain, index)

    summary = annotate_manifest(plain, tmp_path / "v2.jsonl", root=corpus_root)
    entries, meta = read_corpus_manifest(tmp_path / "v2.jsonl")
    assert summary["flagged_utterances"] == 1
    assert summary["manifest_sha256"] != digest
    assert meta["annotated_from_sha256"] == digest
    assert meta["prosody_audit"]["mismatched"] == 1

    by_id = {entry.utt_id: entry for entry in entries}
    assert by_id["SSB00050002"].quality_flags == ("prosody_conflict",)
    original = {entry.utt_id: entry for entry in index.entries}
    for utt_id, entry in by_id.items():
        assert replace(entry, quality_flags=()) == original[utt_id]
    with pytest.raises(FileExistsError):
        annotate_manifest(plain, tmp_path / "v2.jsonl", root=corpus_root)


def test_annotate_and_coverage_cli_wire_the_manifest(corpus_root: Path, tmp_path: Path,
                                                    capsys, monkeypatch) -> None:
    from vhe.cli import main

    plain, flagged = tmp_path / "v1.jsonl", tmp_path / "v2.jsonl"
    monkeypatch.setattr("sys.argv", ["vhe", "scan", "--root", str(corpus_root),
                                     "--manifest", str(plain), "--report",
                                     str(tmp_path / "r.json"), "--split-sizes", "2/1/1"])
    main()
    capsys.readouterr()

    monkeypatch.setattr("sys.argv", ["vhe", "annotate", "--manifest", str(plain),
                                     "--output", str(flagged)])
    main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["flagged_utterances"] == 1

    monkeypatch.setattr("sys.argv", ["vhe", "coverage", "--manifest", str(flagged),
                                     "--output", str(tmp_path / "c.json"), "--top", "2"])
    main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["quality_flags"] == {"prosody_conflict": 1}
    assert printed["utterances"] == 5


def test_coverage_recompute_reads_no_audio(corpus_root: Path) -> None:
    index = scan_corpus(corpus_root, split_sizes={"train": 2, "dev": 1, "test": 1})
    for path in (corpus_root / "train" / "wav").rglob("*.wav"):
        path.unlink()
    report = coverage_report(index.entries, top_units=3)
    assert report["stats"]["utterances"] == 5
    assert report["stats"]["syllable_coverage"] == pytest.approx(
        index.stats.syllable_coverage)
    assert report["stats"]["utterance_coverage"] == pytest.approx(
        index.stats.utterance_coverage)
    assert report["next_units"][0]["unit"] == "final:iang"

