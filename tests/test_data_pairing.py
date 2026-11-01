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


@pytest.mark.parametrize("token,phonemes", [
    ("shi2", ("sh", "apical_retroflex")),
    ("zhi1", ("zh", "apical_retroflex")),
    ("wo3", ("uo",)),
    ("you3", ("iou",)),
    ("liu2", ("l", "iou")),
    ("yan2", ("ian",)),
    ("yue4", ()),
    ("nv3", ()),
    ("zher4", ()),
    ("chun1", ()),
])
def test_parse_syllable_looks_up_the_written_final_not_the_underlying_one(token, phonemes):
    """Regression: passing the underlying final into g2p built fake pinyin.

    'shi' used to be resolved as 'sh' + 'apical_retroflex', which g2p turned into the
    nonsense spelling 'shapical_retroflex' and blocked. 3160 of the 5314 utterances the
    manifest called renderable hit this, so coverage and rendering disagreed by 59%.
    """
    syllable = parse_syllable(token, "mandarin")
    assert syllable.phonemes == phonemes
    if not phonemes:
        assert syllable.reason


def test_parse_syllable_keeps_the_written_spelling_next_to_the_underlying_final():
    syllable = parse_syllable("you3", "mandarin")
    assert (syllable.initial, syllable.final) == ("y", "ou")
    assert syllable.underlying_final == "iou"
    assert syllable.phonemes == ("iou",)
    apical = parse_syllable("zhi1", "mandarin")
    assert apical.final == "i" and apical.underlying_final == "apical_retroflex"


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


SPAN_LAYOUT = (
    # (phone, start, end) inside make_mandarin_utterance's 0.05 lead + 0.30 render + 0.10 gap
    ("m", 0.050, 0.110, 0.95),
    ("a", 0.110, 0.330, 0.92),
    ("b", 0.450, 0.500, 0.90),
    ("a", 0.500, 0.720, 0.88),
    ("n", 0.850, 0.900, 0.94),
    ("a", 0.900, 1.120, 0.91),
)


def test_forced_alignment_spans_replace_the_nominal_phone_plan(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_forced.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="forced",
                        phone_spans=SPAN_LAYOUT)
    assert result.status == "ok", result.issues
    assert result.alignment["method"] == "forced_alignment"
    assert result.real.size == result.mechanical.size
    # The plan is the aligner's spans, not g2p's nominal onset/nucleus split.
    asserted = [entry["duration_s"] for allocation in result.allocations
                for entry in allocation]
    assert asserted == pytest.approx([0.06, 0.22, 0.05, 0.22, 0.05, 0.22], abs=1e-4)
    boundaries = result.alignment["boundaries"]
    assert [round(item["start_s"], 3) for item in boundaries] == [0.05, 0.45, 0.85]
    assert [round(item["end_s"], 3) for item in boundaries] == [0.33, 0.72, 1.12]
    # Per-syllable confidence is the weakest span in that syllable.
    assert [item["confidence"] for item in boundaries] == [0.92, 0.88, 0.91]
    assert "low_boundary_confidence" not in result.readiness["reasons"]
    # The spans themselves are kept so a pair can be audited without re-aligning.
    assert [item[0] for item in result.alignment["phones"]] == ["m", "a", "b", "a", "n", "a"]


def test_forced_alignment_rejects_a_phone_sequence_that_disagrees_with_the_transcript(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_forced_mismatch.wav"
    make_mandarin_utterance(path)
    spans = [("n",) + tuple(span[1:]) if index == 0 else span
             for index, span in enumerate(SPAN_LAYOUT)]
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="forced",
                        phone_spans=spans)
    assert result.status == "rejected"
    assert result.mechanical is None
    codes = {issue["code"] for issue in result.issues}
    assert "phone_sequence_mismatch" in codes


def test_forced_alignment_rejects_overlapping_spans(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_forced_overlap.wav"
    make_mandarin_utterance(path)
    spans = list(SPAN_LAYOUT)
    spans[1] = ("a", 0.040, 0.330, 0.92)   # starts before 'm' has finished
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="forced",
                        phone_spans=spans)
    assert result.status == "rejected"
    codes = {issue["code"] for issue in result.issues}
    assert "bad_phone_span" in codes


def test_forced_alignment_mode_requires_spans_and_excludes_manual_boundaries(tmp_path):
    path = tmp_path / "unused.wav"
    make_mandarin_utterance(path)
    with pytest.raises(ValueError, match="requires phone_spans"):
        build_pair(path, ["ma1"], template=PRESETS["yang"], alignment="forced")
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_pair(path, ["ma1"], template=PRESETS["yang"], alignment="forced",
                   phone_spans=[("m", 0.0, 0.05)], boundaries_s=[(0.0, 0.3)])


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

# --- P1.5: pitch tracker, alignment modes and training readiness ---------------------


def test_pair_pipeline_uses_the_viterbi_tracker_and_reports_readiness(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_tracker.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth())
    assert result.status == "ok", result.issues
    # EXP-0015's tracker is the default; the legacy estimator is opt-in only.
    assert result.pitch["estimator"] == "viterbi_multicandidate"
    assert result.f0_stats["estimator"] == "viterbi_multicandidate"
    assert "reasons" in result.f0_stats and "median_abs_step_cents" in result.f0_stats
    assert result.alignment["method"] == "energy_segments"
    # Energy segmentation produces no per-boundary confidence, so a pair built this way is
    # never training-ready: the boundaries are the segmenter's, not verified ones.
    assert result.readiness["training_ready"] is False
    assert "no_boundary_confidence" in result.readiness["reasons"]


def test_legacy_estimator_stays_available_as_an_ablation(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_legacy.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), pitch_estimator="legacy",
                        alignment="dp")
    assert result.status == "ok", result.issues
    assert result.pitch["estimator"] == "legacy_yin"
    assert result.f0_stats["estimator"] == "legacy_yin"
    assert result.f0_stats["median_f0_hz"] == pytest.approx(200.0, abs=5.0)


def test_dp_alignment_gives_every_isolated_syllable_a_voiced_window(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "mandarin_dp.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="dp")
    assert result.status == "ok", result.issues
    assert result.alignment["method"] == "energy_dp"
    assert len(result.syllables) == len(result.pitch["syllables"]) == 3
    # The reward for rising energy is what keeps a boundary out of the middle of /b/'s
    # closure; without it the second window contained no voiced frame at all.
    assert min(item["coverage"] for item in result.pitch["syllables"]) > 0.5
    assert result.readiness["training_ready"] is True, result.readiness


def test_dp_does_not_call_a_continuous_vowel_split_training_ready(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "cont_vowels.wav"
    audio = np.concatenate([render_vowel(p, 0.2) for p in ("a", "i", "u")])
    write_wav(path, audio, 24000)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="dp")
    assert result.status == "ok", result.issues
    # The DP finds a deep valley between two vowels, but the independent energy segmenter
    # sees one burst while the transcript claims three syllables, so the boundaries are not
    # trusted -- even though dip depth alone would report high confidence.
    assert result.alignment["min_confidence"] > 0.5
    assert "segmentation_count_disagrees" in result.readiness["reasons"]
    assert result.readiness["training_ready"] is False


def test_manual_boundaries_are_the_only_route_to_a_trusted_pair(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "cont_manual.wav"
    audio = np.concatenate([render_vowel(p, 0.2) for p in ("a", "i", "u")])
    write_wav(path, audio, 24000)
    result = build_pair(path, ["a1", "i2", "u3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="dp",
                        boundaries_s=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)])
    assert result.status == "ok", result.issues
    assert result.alignment["method"] == "manual"
    assert [round(item["duration_s"], 3) for item in result.alignment["boundaries"]] \
        == [0.2, 0.2, 0.2]
    assert result.readiness["training_ready"] is True, result.readiness


def test_energy_dp_boundaries_carry_their_own_evidence(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "evidenced.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="dp")
    assert result.status == "ok", result.issues
    bounds = result.alignment["boundaries"]
    assert all({"start_s", "end_s", "duration_s", "dip_db", "confidence"} <= set(item)
               for item in bounds)
    # The first window has no left-hand join, so it has no valley evidence of its own.
    assert bounds[0]["confidence"] is None
    assert sum(1 for item in bounds if item["confidence"] is not None) == len(bounds) - 1


def test_bad_manual_boundaries_are_reported_not_guessed(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "bad_manual.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(),
                        boundaries_s=[(0.0, 0.2), (0.2, 0.4)])
    assert result.status == "rejected"
    codes = {issue["code"] for issue in result.issues}
    assert "bad_manual_boundaries" in codes
    assert result.readiness["training_ready"] is False


def test_a_pitch_gap_is_interpolated_but_blocks_readiness(tmp_path):
    from vhe.synth.mandarin import MandarinFormantSynth

    path = tmp_path / "gap.wav"
    audio = render_vowel("a", 0.5)
    rate = 24000
    audio[int(0.25 * rate):int(0.33 * rate)] = 0.0
    write_wav(path, audio, rate)
    result = build_pair(path, ["a1"], template=PRESETS["yang"],
                        backend=MandarinFormantSynth(), alignment="dp",
                        boundaries_s=[(0.0, 0.5)])
    assert result.status == "ok", result.issues
    codes = {issue["code"] for issue in result.issues}
    assert "pitch_gap_interpolated" in codes
    assert result.pitch["syllables"][0]["longest_gap_s"] > 0.05
    assert result.readiness["training_ready"] is False
    assert "pitch_gap_interpolated" in result.readiness["reasons"]


def test_unknown_pitch_estimator_is_a_programming_error(tmp_path):
    path = tmp_path / "anything.wav"
    make_mandarin_utterance(path)
    with pytest.raises(ValueError):
        build_pair(path, ["ma1"], template=PRESETS["yang"], pitch_estimator="magic")
    with pytest.raises(ValueError):
        build_pair(path, ["ma1"], template=PRESETS["yang"], alignment="magic")


def test_pair_record_serialises_the_new_diagnostics(tmp_path):
    path = tmp_path / "record.wav"
    make_mandarin_utterance(path)
    result = build_pair(path, ["ma1", "ba2", "na3"], template=PRESETS["yang"],
                        backend=__import__("vhe.synth.mandarin",
                                           fromlist=["MandarinFormantSynth"]
                                           ).MandarinFormantSynth())
    record = result.as_dict()
    for key in ("pitch", "alignment", "readiness"):
        assert key in record
    json.dumps(record, allow_nan=False)

