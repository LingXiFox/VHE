"""AISHELL-3 adapter tests on synthetic fixtures (no real corpus on this machine).

The fixtures reproduce the real directory layout and the first lines of the real
annotation files, which were verified against the published corpus (see
docs/DATA_POLICY.md). Real audio is never required: these tests pin the parser,
the failure taxonomy and the speaker-grouped split logic.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from vhe.audio import write_wav
from vhe.data import (
    apply_speaker_split,
    build_manifest,
    carve_speaker_split,
    check_no_speaker_leak,
    discover_splits,
    read_manifest,
    write_manifest,
)
from vhe.data.aishell3 import (
    parse_content_line,
    parse_pinyin_file,
    parse_spk_info,
    speaker_of,
)

# Verbatim lines from the published corpus (content.txt / spk-info.txt).
REAL_CONTENT = "SSB00050001.wav\t广 guang3 州 zhou1 女 nv3 大 da4 学 xue2 生 sheng1"
REAL_SPK_INFO = "# voice-file name; age group; gender; accent\nSSB1837\tB\tfemale\tnorth\n"


def make_wav(path: Path, *, seconds: float = 0.6, sample_rate: int = 44100,
             channels: int = 1, sample_width: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # write_wav refuses overwrite; harness rewrites fixtures
    if (channels, sample_width, sample_rate) == (1, 2, 44100):
        time = np.arange(int(seconds * sample_rate)) / sample_rate
        write_wav(path, 0.3 * np.sin(2 * np.pi * 220.0 * time), sample_rate)
        return
    frames = int(seconds * sample_rate)
    payload = np.zeros((frames, channels), dtype="<i2" if sample_width == 2 else "i1")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(payload.tobytes())


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "AISHELL-3"
    (root / "train" / "wav").mkdir(parents=True)
    (root / "test" / "wav").mkdir(parents=True)
    (root / "spk-info.txt").write_text(
        REAL_SPK_INFO + "SSB0005\tB\tfemale\tnorth\nSSB0009\tC\tmale\tsouth\n",
        encoding="utf-8")

    make_wav(root / "train" / "wav" / "SSB0005" / "SSB00050001.wav")
    make_wav(root / "train" / "wav" / "SSB0005" / "SSB00050002.wav")
    make_wav(root / "train" / "wav" / "SSB0009" / "SSB00090001.wav")
    make_wav(root / "train" / "wav" / "SSB0009" / "SSB00090003.wav", seconds=0.1)  # too_short
    make_wav(root / "test" / "wav" / "SSB0011" / "SSB00110001.wav")
    make_wav(root / "test" / "wav" / "SSB0012" / "SSB00120001.wav", channels=2)  # not_mono

    (root / "train" / "content.txt").write_text(
        REAL_CONTENT + "\n"
        "SSB00050002.wav\t尊 zhun1 重 zhong4 科 ke1 学 xue2\n"
        "SSB00090001.wav\t七 qi1 路 lu4 无 wu2 人 ren2\n"
        "SSB00090002.wav\tmissing wav file\n"
        "SSB00090003.wav\t太 tai4 短 duan3\n"
        "# a comment line\n",
        encoding="utf-8")
    (root / "test" / "content.txt").write_text(
        "SSB00110001.wav\t黑 hei1 客 ke4 宣 xuan1 布 bu4\n"
        "SSB00120001.wav\t北 bei3 京 jing1\n"
        "SSB00110002.wav\t\n",
        encoding="utf-8")
    return root


def test_parse_real_content_line():
    entry = parse_content_line(REAL_CONTENT)
    assert entry is not None
    assert entry.utt_id == "SSB00050001"
    assert entry.text == "广州女大学生"
    assert entry.pinyin == ("guang3", "zhou1", "nv3", "da4", "xue2", "sheng1")


def test_parse_content_line_rejects_and_skips():
    assert parse_content_line("\ufeff") is None
    assert parse_content_line("# comment") is None
    with pytest.raises(ValueError, match="no transcript"):
        parse_content_line("SSB00050001.wav")


def test_speaker_info_and_pinyin_files(corpus: Path):
    speakers = parse_spk_info(corpus / "spk-info.txt")
    assert speakers["SSB1837"].gender == "female"
    assert speakers["SSB0009"].accent == "south"
    pinyin = corpus / "train" / "pinyin.txt"
    pinyin.write_text("SSB00050001.wav\tguang3 zhou1\n", encoding="utf-8")
    assert parse_pinyin_file(pinyin)["SSB00050001"] == ("guang3", "zhou1")
    assert speaker_of("SSB00050001") == "SSB0005"


def test_discover_splits_and_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_splits(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="content.txt"):
        discover_splits(empty)


def test_build_manifest_taxonomy(corpus: Path):
    manifest = build_manifest(corpus)
    accepted = {(entry.split, entry.utt_id) for entry in manifest.entries}
    assert accepted == {
        ("train", "SSB00050001"), ("train", "SSB00050002"), ("train", "SSB00090001"),
        ("test", "SSB00110001"),
    }
    codes = manifest.stats()["error_codes"]
    assert codes["missing_wav"] == 1
    assert codes["too_short"] == 1
    assert codes["not_mono"] == 1
    assert codes["empty_text"] == 1
    # Nothing is dropped silently: 4 accepted + 4 rejected = 7 transcript rows + 1 comment row.
    assert len(manifest.entries) + len(manifest.errors) == 8
    assert all(issue.detail for issue in manifest.issues)


def test_manifest_records_provenance(corpus: Path):
    manifest = build_manifest(corpus)
    first = [e for e in manifest.entries if e.utt_id == "SSB00050001"][0]
    assert first.wav_path == "train/wav/SSB0005/SSB00050001.wav"
    assert first.sample_rate == 44100
    assert first.n_samples == int(0.6 * 44100)
    assert len(first.sha256) == 64
    assert first.gender == "female" and first.accent == "north"
    # SSB0012's only utterance is rejected as not_mono, so the speaker never enters the
    # manifest: rejected audio must not contribute speakers to split bookkeeping.
    assert manifest.speakers_by_split()["test"] == {"SSB0011"}


def test_hashes_are_stable_and_detect_changes(corpus: Path):
    manifest = build_manifest(corpus)
    assert manifest.content_hash() == build_manifest(corpus).content_hash()
    make_wav(corpus / "train" / "wav" / "SSB0005" / "SSB00050001.wav", seconds=0.7)
    assert manifest.content_hash() != build_manifest(corpus).content_hash()


def test_speaker_split_never_leaks(corpus: Path):
    manifest = build_manifest(corpus)
    assignment = carve_speaker_split(manifest, dev_fraction=0.5, seed=7)
    assert set(assignment) == {"SSB0005", "SSB0009", "SSB0011"}
    assert assignment["SSB0011"] == "test"
    grouped = apply_speaker_split(manifest, assignment)
    table = check_no_speaker_leak(grouped)
    assert sorted(table["train"] + table["dev"]) == ["SSB0005", "SSB0009"]
    assert table["test"] == ["SSB0011"]


def test_speaker_split_is_deterministic_and_sized(corpus: Path):
    manifest = build_manifest(corpus)
    first = carve_speaker_split(manifest, dev_fraction=0.5, seed=7)
    second = carve_speaker_split(manifest, dev_fraction=0.5, seed=7)
    assert first == second
    assert list(first.values()).count("dev") == 1  # round(2 train speakers * 0.5)
    with pytest.raises(ValueError, match="dev_fraction"):
        carve_speaker_split(manifest, dev_fraction=1.0)


def test_manifest_jsonl_roundtrip(corpus: Path, tmp_path: Path):
    manifest = build_manifest(corpus)
    path = tmp_path / "manifest.jsonl"
    digest = write_manifest(path, manifest)
    assert len(digest) == 64

    restored, meta = read_manifest(path)
    assert meta["content_sha256"] == manifest.content_hash()
    expected = [e.as_dict() for e in sorted(manifest.entries, key=lambda e: (e.split, e.utt_id))]
    assert [e.as_dict() for e in restored.entries] == expected
    assert len(restored.issues) == len(manifest.issues)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "meta"

    again = tmp_path / "manifest2.jsonl"
    assert write_manifest(again, manifest) == digest  # byte-stable across runs
    assert again.read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_manifest(path, manifest)


def test_no_train_test_overlap_by_construction(corpus: Path):
    manifest = build_manifest(corpus)
    grouped = apply_speaker_split(manifest, carve_speaker_split(manifest, dev_fraction=0.0))
    table = check_no_speaker_leak(grouped)
    assert not set(table["train"]) & set(table["test"])
    assert table["dev"] == []


def test_manifest_cli_prints_report_and_writes_jsonl(corpus: Path, tmp_path: Path,
                                                     capsys, monkeypatch):
    from vhe.cli import main

    out = tmp_path / "manifest.jsonl"
    monkeypatch.setattr("sys.argv", ["vhe", "manifest", "--root", str(corpus),
                                     "--output", str(out), "--dev-fraction", "0.5"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "manifest-only; no audio copied"
    assert report["stats"]["utterances"] == 4
    assert report["stats"]["errors"] == 4
    assert report["stats"]["speakers_per_split"] == {"train": 1, "dev": 1, "test": 1}
    assert len(report["manifest_sha256"]) == 64
    assert out.is_file()


def test_manifest_cli_fail_on_errors_and_overwrite_guard(corpus: Path, tmp_path: Path,
                                                        capsys, monkeypatch):
    from vhe.cli import main

    out = tmp_path / "manifest.jsonl"
    argv = ["vhe", "manifest", "--root", str(corpus), "--output", str(out), "--fail-on-errors"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as first:
        main()
    assert first.value.code == 2
    assert "rejected" in capsys.readouterr().err
    with pytest.raises(SystemExit) as second:
        main()
    assert second.value.code == 2
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_manifest_cli_missing_root(tmp_path: Path, capsys, monkeypatch):
    from vhe.cli import main

    monkeypatch.setattr("sys.argv", ["vhe", "manifest", "--root", str(tmp_path / "absent"),
                                     "--output", str(tmp_path / "m.jsonl")])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "does not exist" in capsys.readouterr().err
