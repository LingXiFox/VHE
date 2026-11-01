"""Tests for paired-data construction (P1-C).

The "real" fixture is rendered by the same synthesizer, so these tests only prove that the
plumbing (analysis -> request -> render -> pair record) works and that impossible pairs are
refused. They say nothing about naturalness or about real-speech alignment quality.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vhe.audio import read_wav, write_wav
from vhe.data.pairing import build_pair, parse_syllable, write_pair
from vhe.synth import (PRESETS, PitchCurve, PhonemeSegment, PhonemeSequence,
                       SynthesisRequest, Timing, get_backend)


def render_vowel(phoneme, duration=0.2, f0_hz=200.0, sample_rate=24000,
                 lead=0.0, tail=0.0):
    request = SynthesisRequest(
        template=PRESETS["yin"],
        phonemes=PhonemeSequence((PhonemeSegment(phoneme, duration),)),
        pitch=PitchCurve.constant(f0_hz, duration),
        timing=Timing(lead_silence_s=lead, tail_silence_s=tail),
        sample_rate=sample_rate,
    )
    return get_backend().render(request).audio


def make_utterance(path, phonemes=("a", "i", "u"), sample_rate=24000, duration=0.2,
                   f0_hz=200.0, lead=0.05, tail=0.05):
    # Isolated syllables: the energy segmenter needs an acoustic boundary, and the backend
    # has no silence phoneme, so syllables are rendered separately and concatenated.
    from vhe.data.resample import resample

    parts = [np.zeros(int(round(lead * 24000)), dtype=np.float32)]
    for phoneme in phonemes:
        parts.append(render_vowel(phoneme, duration, f0_hz))
        parts.append(np.zeros(int(round(0.1 * 24000)), dtype=np.float32))
    audio = np.concatenate(parts)
    if sample_rate != 24000:
        audio = resample(audio, 24000, sample_rate)
    write_wav(path, audio, sample_rate)
    return audio


def test_parse_syllable_splits_initial_final_tone():
    syllable = parse_syllable("guang3")
    assert (syllable.initial, syllable.final, syllable.tone) == ("g", "uang", 3)
    assert syllable.phoneme is None
    assert "inventory" in syllable.reason

    apical = parse_syllable("zhi1")
    assert apical.initial == "zh" and apical.final == "i" and apical.phoneme is None

    plain = parse_syllable("a1")
    assert plain.phoneme == "a"
    assert parse_syllable("wu2").phoneme is None  # w is an onset, not a vowel
    assert parse_syllable("v3").final == "v" and parse_syllable("v3").phoneme is None


@pytest.mark.parametrize("token", ["a", "1a", "a1b", "", "a0"])
def test_parse_syllable_rejects_malformed_tokens(token):
    with pytest.raises(ValueError):
        parse_syllable(token)


def test_build_pair_vowel_only_succeeds(tmp_path):
    path = tmp_path / "a_i_u.wav"
    make_utterance(path)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"])
    assert result.status == "ok", result.issues
    assert result.real is not None and result.mechanical is not None
    assert result.real.size == result.mechanical.size
    assert result.real_sha256 and result.mechanical_sha256
    assert result.real_sha256 != result.mechanical_sha256
    assert result.request_fingerprint
    assert result.f0_stats["voiced_ratio"] > 0.3
    assert result.segmentation_stats["segments"] == 3
    codes = {issue["code"] for issue in result.issues}
    assert "syllablewise_render" in codes
    # Pauses survive: the 100 ms gaps between the isolated fixture syllables stay silent.
    assert np.abs(result.mechanical[6000:7200]).max() == 0.0
    # Loudness matching is recorded per syllable and never pushes the pair past full scale.
    assert len(result.gains_db) == 3
    assert all(np.isfinite(result.gains_db))
    assert float(np.abs(result.mechanical).max()) <= 0.9990001


def test_continuous_vowels_collapse_into_one_segment(tmp_path):
    # Documented limitation: voiced vowels with no energy valley are one burst, so a
    # transcript of 3 syllables cannot be aligned by energy alone.
    path = tmp_path / "continuous.wav"
    audio = np.concatenate([render_vowel(p, 0.2) for p in ("a", "i", "u")])
    write_wav(path, audio, 24000)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"])
    assert result.status == "rejected"
    codes = {issue["code"] for issue in result.issues}
    assert "segment_syllable_mismatch" in codes
    assert "syllable_count_mismatch" in codes


def test_gain_match_is_per_syllable_and_capped_at_full_scale(tmp_path):
    # A full-scale square has much more RMS per unit peak than the sine-like render, so the
    # loudness match asks for ~3x gain; without the cap the pair would clip at ~2.7 and the
    # dataset consumer could not tell that distortion from the model's own artefact.
    audio = make_utterance(tmp_path / "base.wav").copy()
    rate = 24000
    start = int(round(0.05 * rate))
    stop = start + int(round(0.2 * rate))
    t = np.arange(stop - start) / rate
    audio[start:stop] = np.sign(np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    path = tmp_path / "loud_first.wav"
    write_wav(path, audio, rate)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"])
    assert result.status == "ok", result.issues
    codes = {issue["code"] for issue in result.issues}
    assert "gain_limited_by_peak" in codes
    assert float(np.abs(result.mechanical).max()) <= 0.9990001
    # The cap trades exact loudness matching for headroom, so that syllable stays louder
    # than the recording (documented, not silent).
    assert result.gains_db[0] < 20.0 * np.log10(2.0)


def test_build_pair_blocks_mandarin_syllables_with_consonants(tmp_path):
    path = tmp_path / "mandarin.wav"
    make_utterance(path)
    result = build_pair(path, ["ma1", "ni3", "guang3"], template=PRESETS["yang"])
    assert result.status == "blocked"
    assert result.mechanical is None
    assert result.request_fingerprint is None
    codes = {issue["code"] for issue in result.issues}
    assert "backend_inventory_insufficient" in codes
    blocked = [s for s in result.syllables if s["phoneme"] is None]
    assert len(blocked) == 3


def test_build_pair_rejects_silence(tmp_path):
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(24000, dtype=np.float32), 24000)
    result = build_pair(path, ["a1"], template=PRESETS["yang"])
    assert result.status == "rejected"
    assert "mostly_unvoiced" in {issue["code"] for issue in result.issues}


def test_build_pair_rejects_oversized_utterance(tmp_path):
    path = tmp_path / "long.wav"
    make_utterance(path, duration=0.3)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"],
                        max_duration_s=0.1)
    assert result.status == "rejected"
    assert result.issues[0]["code"] == "too_long"


def test_build_pair_records_resampling(tmp_path):
    path = tmp_path / "at_44100.wav"
    make_utterance(path, sample_rate=44100)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"])
    assert result.status == "ok", result.issues
    assert result.source_sample_rate == 44100
    assert "resampled" in {issue["code"] for issue in result.issues}
    assert result.real.size == 22800
    assert result.mechanical.size == result.real.size


def test_write_pair_refuses_overwrite_and_records_hashes(tmp_path):
    path = tmp_path / "src.wav"
    make_utterance(path)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"])
    target = tmp_path / "pair"
    record = write_pair(target, result)
    assert json.loads((target / "pair.json").read_text())["status"] == "ok"
    assert record["files"]["mechanical"] == "mechanical.wav"
    real, rate = read_wav(target / "real.wav")
    assert rate == 24000 and real.size == result.real.size
    with pytest.raises(FileExistsError):
        write_pair(target, result)


def test_write_pair_rejects_blocked_result(tmp_path):
    path = tmp_path / "src.wav"
    make_utterance(path)
    result = build_pair(path, ["ma1", "ni3", "guang3"], template=PRESETS["yang"])
    with pytest.raises(ValueError):
        write_pair(tmp_path / "pair", result)

def make_mandarin_utterance(path, syllables=(("m", "a"), ("b", "a"), ("n", "a")),
                            sample_rate=24000, f0_hz=200.0, lead=0.05, gap=0.10):
    """Rendered "real" audio for the Mandarin backend: the pair pipeline needs energy gaps."""
    from vhe.synth.mandarin import MandarinFormantSynth

    backend = MandarinFormantSynth()
    parts = [np.zeros(int(round(lead * sample_rate)), dtype=np.float32)]
    for onset, nucleus in syllables:
        durations = (onset, 0.06), (nucleus, 0.24)
        request = SynthesisRequest(
            template=PRESETS["yang"],
            phonemes=PhonemeSequence(
                tuple(PhonemeSegment(name, duration) for name, duration in durations),
                inventory="mandarin"),
            pitch=PitchCurve.constant(f0_hz, 0.30),
            timing=Timing(), sample_rate=sample_rate)
        parts.append(backend.render(request).audio)
        parts.append(np.zeros(int(round(gap * sample_rate)), dtype=np.float32))
    audio = np.concatenate(parts)
    write_wav(path, audio, sample_rate)
    return audio


def test_build_pair_with_the_mandarin_backend_records_onset_nucleus_allocations(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_e2e.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth())
    assert result.status == "ok", result.issues
    assert result.real.size == result.mechanical.size
    assert [s["phonemes"] for s in result.syllables] == [["m", "a"], ["b", "a"], ["n", "a"]]
    assert len(result.allocations) == 3
    for allocation, (onset, nucleus) in zip(result.allocations, ("ma", "ba", "na")):
        assert [entry["phoneme"] for entry in allocation] == [onset, nucleus]
        # The plan fits the *measured* syllable duration, which the segmenter trims to the
        # energy edges (0.268 s for a 0.30 s render), not the requested duration.
        total = sum(entry["duration_s"] for entry in allocation)
        assert 0.20 < total <= 0.301, total
    # One request per syllable, each with its own seed, so the fingerprints must differ.
    assert len(result.request_fingerprints) == 3
    assert len(set(result.request_fingerprints)) == 3
    assert len(result.gains_db) == 3
    assert float(np.abs(result.mechanical).max()) <= 0.9990001
    # The mechanical signal is audible where the recording speaks and silent in the gaps.
    rate = result.sample_rate
    for start in (0.05, 0.45, 0.85):
        window = result.mechanical[int(start * rate):int((start + 0.2) * rate)]
        assert float(np.sqrt(np.mean(window ** 2))) > 0.01
    assert float(np.abs(result.mechanical[int(0.36 * rate):int(0.44 * rate)]).max()) < 1e-3


def test_build_pair_blocks_a_mandarin_syllable_outside_the_inventory(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_blocked.wav"
    make_mandarin_utterance(path)
    # zi2 is the dental apical vowel [ɿ], still absent from the inventory; shi2 became
    # renderable in EXP-0016 stage 2, so the apical refusal is now covered by the dental
    # series. The assertion is unchanged: no near-substitution.
    result = build_pair(path, ["ma1", "zi2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth())
    assert result.status == "blocked"
    assert result.mechanical is None
    codes = {issue["code"] for issue in result.issues}
    assert "backend_inventory_insufficient" in codes
    blocked = [s for s in result.syllables if not s["phonemes"]]
    assert [s["pinyin"] for s in blocked] == ["zi2"]
    assert "apical" in blocked[0]["reason"]
