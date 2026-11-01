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

from vhe.data.aligner import (AlignmentError, PreparedUtterance, _flat_start_segmentation,
                              _state_layout, crop_speech, duration_by_kind, pinyin_phones,
                              silence_frame_edges, train_aligner)
from vhe.data.features import SAMPLE_RATE, acoustic_features
from vhe.data.aligner import PHONES
from vhe.data.hmm import ChainHmm

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


def test_flat_start_splits_every_frame_when_rounding_would_overflow():
    """Ratio 4.59 frames per phone used to raise '459 frames cannot hold 463 phone frames'.

    Rounding each block on its own gives 5 frames to all 100 blocks (500 > 459), and handing
    the drift to the last block clamps it at one frame, so the total still exceeds the frames.
    A real 154-phone sentence over 459 frames crashed aligner training this way (EXP-0023).
    """
    frames, repeats = 459, 100
    phones = tuple(PHONES[index % len(PHONES)] for index in range(repeats))
    item = PreparedUtterance("long", np.zeros((frames, 39), dtype=np.float32), phones, (0, 0),
                             4.59)
    layout = _state_layout(PHONES, 3)
    assignment = _flat_start_segmentation(item, layout, len(PHONES) * 3)
    assert assignment.shape == (frames,)
    assert assignment.min() >= 0 and assignment.max() < len(PHONES) * 3
    # Every phone in the sequence keeps at least one frame; none is squeezed out.
    for phone in set(phones):
        offset, count = layout[phone]
        assert np.isin(assignment, np.arange(offset, offset + count)).sum() >= 1


def test_flat_start_refuses_fewer_frames_than_states():
    phones = tuple(PHONES[index % len(PHONES)] for index in range(40))
    item = PreparedUtterance("short", np.zeros((10, 39), dtype=np.float32), phones, (0, 0), 0.1)
    with pytest.raises(AlignmentError, match="frames cannot"):
        _flat_start_segmentation(item, _state_layout(PHONES, 3), len(PHONES) * 3)


def synthetic_timed(phones, durations, seed, lead=0.15, tail=0.15):
    """Like synthetic(), but the caller fixes each phone's duration."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead * SAMPLE_RATE), dtype=np.float32)]
    boundaries = []
    cursor = lead
    for phone, duration in zip(phones, durations):
        parts.append(colour(phone, duration, rng))
        boundaries.append((cursor, cursor + duration))
        cursor += duration
    parts.append(np.zeros(int(tail * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(parts), boundaries


def timed_corpus(count=12, long_s=0.30, short_s=0.06):
    items = []
    for index in range(count):
        audio, _ = synthetic_timed(("a", "t"), (long_s, short_s), seed=100 + index)
        cropped, (lead, _tail) = crop_speech(audio, SAMPLE_RATE)
        items.append(PreparedUtterance(f"utt{index}", acoustic_features(cropped, SAMPLE_RATE),
                                       ("a", "t"), (lead, 0), audio.size / SAMPLE_RATE,
                                       lead * 0.01))
    return items


def test_estimated_transitions_agree_with_decoded_durations():
    # Fixed 0.5 makes every kind last exactly 60 ms no matter what the decoder does, so the two
    # numbers can only agree if the stays were re-estimated from this corpus. Note what this
    # does *not* test: that the estimate is closer to the true 300 ms / 60 ms split. On this
    # synthetic pair the emissions dominate the boundary (measured: ~215/157 ms either way).
    items = timed_corpus()
    fixed_aligner, fixed_report = train_aligner(items, iterations=3)
    aligner, report = train_aligner(items, iterations=3, estimate_transitions=True)
    assert fixed_report["estimate_transitions"] is False
    assert report["estimate_transitions"] is True
    fixed_kinds = fixed_report["self_probability_by_kind"]
    assert fixed_kinds["vowel"]["mean_duration_s"] == pytest.approx(0.06)
    assert fixed_kinds["stop"]["mean_duration_s"] == pytest.approx(0.06)
    layout = _state_layout(PHONES, 3)
    for phone in ("a", "t"):
        offset, count = layout[phone]
        stays = aligner.hmm.self_probabilities[offset:offset + count]
        implied = float(np.sum(1.0 / (1.0 - stays))) * 0.01
        decoded = [span.end_s - span.start_s
                   for item in items
                   for phoneme, span in zip(item.phones, aligner.align_phones(item.features,
                                                                              item.phones).spans)
                   if phoneme == phone]
        assert implied == pytest.approx(float(np.mean(decoded)), rel=0.4)
    assert np.allclose(fixed_aligner.hmm.self_probabilities[:3], 0.5)


def test_duration_by_kind_reads_model_not_decoder():
    hmm = ChainHmm(means=np.zeros((4, 1, 2)), variances=np.ones((4, 1, 2)),
                   weights=np.ones((4, 1)), self_probability=np.array([0.9, 0.9, 0.5, 0.5]))
    kinds = duration_by_kind(hmm, ("a", "a", "t", "t"), states_per_phone=2)
    # 2 states at stay 0.9 -> 2/(0.1) = 20 frames = 0.20 s; at 0.5 -> 2/(0.5) = 4 frames.
    assert kinds["vowel"]["mean_duration_s"] == pytest.approx(0.20)
    assert kinds["stop"]["mean_duration_s"] == pytest.approx(0.04)
    assert kinds["vowel"]["phones"] == 1

