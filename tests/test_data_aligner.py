"""Tests for the chain-HMM forced aligner (P2).

The fixtures are synthetic band-limited noises with known phone boundaries, so these tests prove
the pipeline (silence gate, phone chain, Viterbi spans, persistence) recovers boundaries that are
separated in the spectrum. They do not prove anything about real Mandarin: overlapping phones,
coarticulation and channel noise are exactly what a corpus experiment has to measure. The
boundary test deliberately uses distinct phone sequences, because two adjacent identical phones
have no acoustic cue between them and their split is arbitrary by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhe.data.aligner import (AlignmentError, PreparedUtterance, _state_layout, crop_speech,
                              pinyin_phones, silence_frame_edges, train_aligner)
from vhe.data.features import SAMPLE_RATE, acoustic_features

BANDS = {"a": (400.0, 900.0), "i": (1800.0, 2600.0), "s": (4000.0, 6000.0),
         "sh": (2500.0, 3200.0), "t": (900.0, 1400.0), "n": (200.0, 350.0)}


def colour(phone, duration, rng, sample_rate=SAMPLE_RATE):
    samples = int(duration * sample_rate)
    spectrum = np.fft.rfft(rng.normal(size=samples))
    frequencies = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    low, high = BANDS[phone]
    shaped = np.fft.irfft(spectrum * ((frequencies >= low) & (frequencies <= high)), n=samples)
    shaped /= np.abs(shaped).max() + 1e-9
    ramp = int(0.01 * sample_rate)
    envelope = np.ones(samples)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return (0.3 * shaped * envelope).astype(np.float32)


def synthetic(phones, seed, lead=0.15, tail=0.15):
    """Return (audio, [(start_s, end_s), ...]) with the true phone boundaries."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead * SAMPLE_RATE), dtype=np.float32)]
    boundaries = []
    cursor = lead
    for phone in phones:
        duration = float(rng.uniform(0.10, 0.20))
        parts.append(colour(phone, duration, rng))
        boundaries.append((cursor, cursor + duration))
        cursor += duration
    parts.append(np.zeros(int(tail * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(parts), boundaries


def prepared(identifier, phones, seed):
    audio, _boundaries = synthetic(phones, seed)
    cropped, (lead, _tail) = crop_speech(audio, SAMPLE_RATE)
    return PreparedUtterance(identifier, acoustic_features(cropped, SAMPLE_RATE), tuple(phones),
                             (lead, 0), audio.size / SAMPLE_RATE, lead * 0.01)


def test_silence_frame_edges_finds_padding():
    audio, _ = synthetic(("a", "i"), 1, lead=0.20, tail=0.10)
    lead, tail = silence_frame_edges(audio, SAMPLE_RATE)
    assert lead == pytest.approx(20, abs=2)
    assert tail == pytest.approx(10, abs=2)


def test_crop_speech_trims_and_reports_frames():
    audio, _ = synthetic(("a",), 2, lead=0.15, tail=0.15)
    cropped, (lead, tail) = crop_speech(audio, SAMPLE_RATE)
    assert 0 < cropped.size < audio.size
    assert lead * 160 + tail * 160 == audio.size - cropped.size


def test_crop_speech_leaves_unpadded_signal_alone():
    audio, _ = synthetic(("a",), 3, lead=0.0, tail=0.0)
    cropped, (lead, tail) = crop_speech(audio, SAMPLE_RATE)
    assert (lead, tail) == (0, 0)
    assert np.array_equal(cropped, audio)


def test_pinyin_phones_resolves_a_mandarin_phrase():
    phones, reason = pinyin_phones(["ni3", "hao3"])
    assert reason is None
    assert phones == ("n", "i", "h", "au")


def test_pinyin_phones_reports_blocking_syllables_instead_of_substituting():
    for pinyin, fragment in (("jue2", "ü"), ("er2", "er")):
        phones, reason = pinyin_phones([pinyin])
        assert phones == ()
        assert reason is not None and fragment in reason


def test_state_layout_offsets_are_global():
    layout = _state_layout(("a", "i", "a"), states_per_phone=3)
    assert layout["a"] == (0, 3)
    assert layout["i"] == (3, 3)
    # 3 phones * 3 states: the repeated vowel must not be given a second block.
    assert sum(count for _offset, count in layout.values()) == 6


def test_aligner_recovers_synthetic_boundaries():
    rng = np.random.default_rng(21)
    vocabulary = tuple(BANDS)
    items = [prepared(f"train{index}", tuple(rng.choice(vocabulary, size=3)), 100 + index)
             for index in range(20)]
    aligner, report = train_aligner(items, iterations=4)
    assert report["monotone"] is True
    assert report["skipped"] == []

    # Adjacent identical phones are acoustically indistinguishable, so their internal boundary
    # is not a fair target; the evaluation sequences therefore use distinct phones.
    errors = []
    for index in range(4):
        phones = tuple(np.random.default_rng(500 + index).choice(vocabulary, size=4,
                                                                replace=False))
        audio, boundaries = synthetic(phones, 700 + index)
        cropped, (lead, _tail) = crop_speech(audio, SAMPLE_RATE)
        result = aligner.align_phones(acoustic_features(cropped, SAMPLE_RATE), phones,
                                      offset_s=lead * 0.01)
        predicted = np.array([span.start_s for span in result.spans[1:]])
        truth = np.array([start for start, _end in boundaries[1:]])
        errors.append(float(np.abs(predicted - truth).max()))
        assert result.confidence > 0.5
        assert result.spans[0].start_s < result.spans[-1].end_s
    assert max(errors) < 0.05


def test_spans_are_contiguous_and_cover_every_frame():
    items = [prepared(f"utt{index}", ("a", "i", "s"), 300 + index) for index in range(6)]
    aligner, _report = train_aligner(items, iterations=2)
    audio, _boundaries = synthetic(("a", "i", "s"), 999)
    cropped, (lead, _tail) = crop_speech(audio, SAMPLE_RATE)
    result = aligner.align_phones(acoustic_features(cropped, SAMPLE_RATE), ("a", "i", "s"),
                                  offset_s=lead * 0.01)
    assert sum(span.frames for span in result.spans) == result.frames
    for earlier, later in zip(result.spans, result.spans[1:]):
        assert earlier.end_s == pytest.approx(later.start_s)
    assert len(result.boundary_s) == 2


def test_aligner_persistence_roundtrip_and_refusal_to_overwrite(tmp_path):
    items = [prepared(f"utt{index}", ("a", "i"), 400 + index) for index in range(5)]
    aligner, _report = train_aligner(items, iterations=2)
    path = aligner.to_npz(tmp_path / "aligner.npz", meta={"experiment": "unit-test"})
    restored = type(aligner).from_npz(path)
    assert np.allclose(restored.hmm.means, aligner.hmm.means)
    assert restored.states_per_phone == aligner.states_per_phone
    assert restored.meta["experiment"] == "unit-test"
    with pytest.raises(FileExistsError):
        aligner.to_npz(path)


def test_too_short_utterance_raises():
    items = [prepared(f"utt{index}", ("a", "i"), 500 + index) for index in range(4)]
    aligner, _report = train_aligner(items, iterations=1)
    features = np.zeros((5, 39), dtype=np.float32)
    with pytest.raises(AlignmentError):
        aligner.align_phones(features, ("a", "i", "s", "sh"))


def test_train_aligner_reports_skipped_utterances():
    items = [prepared(f"utt{index}", ("a", "i"), 600 + index) for index in range(3)]
    items.append(PreparedUtterance("tiny", np.zeros((4, 39), dtype=np.float32),
                                   ("a", "i", "s"), (0, 0), 0.04))
    _aligner, report = train_aligner(items, iterations=1)
    assert report["utterances"] == 3
    assert [entry["utterance_id"] for entry in report["skipped"]] == ["tiny"]


def test_train_aligner_refuses_empty_input():
    with pytest.raises(ValueError):
        train_aligner([])
