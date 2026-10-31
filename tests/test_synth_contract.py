"""Contract tests: units, validation, determinism, and the byte-level regression.

The regression hash is the SHA256 of the v0.0.1 baseline WAV (EXP-0000, Linux) and was
reproduced on macOS/arm64 (EXP-0002). A change here means the mechanical baseline
changed behaviour and needs a new experiment record, not a silent test update.
"""

import hashlib
import wave

import numpy as np
import pytest

from vhe.audio import read_wav, write_wav
from vhe.synth import (
    PHONEME_INVENTORY,
    PRESETS,
    PitchCurve,
    PhonemeSegment,
    PhonemeSequence,
    SynthesisRequest,
    Timing,
    VoiceTemplate,
    get_backend,
    list_backends,
    synthesize,
)
from vhe.synth.contract import F0_MAX_HZ, F0_MIN_HZ

BASELINE_SHA256 = "cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985"


def _autocorrelation_f0(window: np.ndarray, sample_rate: int,
                        fmin: float = 70.0, fmax: float = 600.0) -> float:
    """Estimate the fundamental of one window by normalised autocorrelation.

    Normalising by the overlap energy matters: the unnormalised score always
    prefers long lags (less overlap, less accumulated energy) and would report
    a high harmonic instead of the fundamental.
    """
    signal = window.astype(np.float64)
    signal = signal - signal.mean()
    if not np.any(signal):
        return float("nan")
    best_score, best_lag = -2.0, 0
    for lag in range(int(sample_rate / fmax), int(sample_rate / fmin) + 1):
        head, tail = signal[:-lag], signal[lag:]
        norm = np.sqrt(np.dot(head, head) * np.dot(tail, tail))
        if norm <= 0.0:
            continue
        score = float(np.dot(head, tail) / norm)
        if score > best_score:
            best_score, best_lag = score, lag
    return sample_rate / best_lag if best_lag else float("nan")


def make_request(**overrides) -> SynthesisRequest:
    params = dict(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("a", 0.5),)),
        pitch=PitchCurve.constant(220.0, 0.5),
        timing=Timing(),
        sample_rate=16000,
        seed=42,
    )
    params.update(overrides)
    return SynthesisRequest(**params)


def test_legacy_wrapper_reproduces_v001_baseline_bytes(tmp_path):
    audio = synthesize(["a", "i", "u"], f0_hz=220.0, duration_s=1.2,
                       sample_rate=24000, template=PRESETS["yang"], seed=42)
    path = tmp_path / "baseline.wav"
    write_wav(path, audio, 24000)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE_SHA256


def test_request_has_no_hidden_conditioning_inputs():
    # Guards the product boundary: text, phoneme labels, speaker id, MIDI or a
    # reference recording must never become VHE (or even synth) request fields.
    assert set(SynthesisRequest.__dataclass_fields__) == {
        "template", "phonemes", "pitch", "timing", "sample_rate", "seed",
    }
    assert set(VoiceTemplate.__dataclass_fields__) == {
        "formant_scale", "spectral_tilt", "breathiness", "vibrato_cents", "vibrato_hz",
    }


def test_unsupported_phoneme_is_an_error_not_a_substitution():
    with pytest.raises(ValueError, match="a,i,u"):
        make_request(phonemes=PhonemeSequence((PhonemeSegment("yuan", 0.3),))).validate()


def test_empty_phoneme_sequence_is_rejected():
    with pytest.raises(ValueError, match="a,i,u"):
        make_request(phonemes=PhonemeSequence(())).validate()


def test_segment_and_sample_rate_validation():
    with pytest.raises(ValueError, match="duration_s"):
        PhonemeSegment("a", 0.0).validate()
    with pytest.raises(ValueError, match="sample_rate"):
        make_request(sample_rate=22050).validate()
    with pytest.raises(ValueError, match="formant_scale"):
        make_request(template=VoiceTemplate(formant_scale=2.0)).validate()


def test_pitch_curve_units_and_interpolation():
    curve = PitchCurve(times_s=(0.0, 0.5), f0_hz=(110.0, 440.0))
    samples = curve.sample(5, 4)  # t = 0, .25, .5, .75, 1.0 seconds
    assert samples[0] == pytest.approx(110.0)
    # Two octaves over 0.5 s: the midpoint is the geometric mean (220 Hz), not the arithmetic
    # mean (275 Hz). Interpolating Hz would make the glide 386 cents sharp of a linear pitch
    # sweep, which is audible; see EXP-0007.
    assert samples[1] == pytest.approx(220.0)
    assert samples[2] == pytest.approx(440.0)
    assert samples[3] == pytest.approx(440.0)  # clamped, never extrapolated
    assert samples[4] == pytest.approx(440.0)
    constant = PitchCurve.constant(180.0, 1.0).sample(100, 100)
    assert np.all(constant == 180.0)  # a flat curve stays bit-identical


def test_pitch_curve_rejects_bad_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        PitchCurve(times_s=(0.0, 0.0), f0_hz=(200.0, 210.0)).validate()
    with pytest.raises(ValueError, match="f0_hz"):
        PitchCurve(times_s=(0.0,), f0_hz=(F0_MAX_HZ + 1.0,)).validate()
    with pytest.raises(ValueError, match="f0_hz"):
        PitchCurve(times_s=(0.0,), f0_hz=(F0_MIN_HZ - 1.0,)).validate()


def test_curve_must_cover_voiced_region():
    with pytest.raises(ValueError, match="does not cover"):
        make_request(pitch=PitchCurve(times_s=(0.0, 0.2), f0_hz=(220.0, 220.0))).validate()


def test_timing_pads_silence_without_changing_voiced_length():
    backend = get_backend()
    plain = backend.render(make_request())
    padded = backend.render(make_request(timing=Timing(lead_silence_s=0.1,
                                                   tail_silence_s=0.05)))
    assert plain.audio.size == 8000
    assert padded.audio.size == 8000 + 1600 + 800
    assert np.all(padded.audio[:1600] == 0.0)
    assert np.all(padded.audio[-800:] == 0.0)
    assert np.array_equal(padded.audio[1600:9600], plain.audio)


def test_negative_or_excessive_silence_is_rejected():
    with pytest.raises(ValueError, match="lead_silence_s"):
        Timing(lead_silence_s=-0.1).validate()
    with pytest.raises(ValueError, match="tail_silence_s"):
        Timing(tail_silence_s=6.0).validate()


def test_result_layout_matches_vhe_input_contract():
    result = get_backend().render(make_request())
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert result.as_batch().shape == (1, 1, result.audio.size)
    assert 0.0 < float(np.max(np.abs(result.audio))) <= 1.0
    assert result.backend_name == "formant_proxy"
    assert len(result.request_fingerprint) == 64


def test_render_is_deterministic_and_seed_sensitive():
    backend = get_backend()
    first = backend.render(make_request())
    second = backend.render(make_request())
    other_seed = backend.render(make_request(seed=43))
    assert np.array_equal(first.audio, second.audio)
    assert not np.array_equal(first.audio, other_seed.audio)
    assert first.request_fingerprint == second.request_fingerprint
    assert first.request_fingerprint != other_seed.request_fingerprint


def test_fingerprint_tracks_the_unique_variable():
    base = make_request().fingerprint()
    assert make_request(template=VoiceTemplate(spectral_tilt=1.5)).fingerprint() != base
    assert make_request(pitch=PitchCurve.constant(230.0, 0.5)).fingerprint() != base
    assert make_request(phonemes=PhonemeSequence((PhonemeSegment("i", 0.5),))).fingerprint() != base


def test_rising_pitch_curve_raises_pitch_track():
    request = make_request(
        phonemes=PhonemeSequence((PhonemeSegment("a", 1.0),)),
        pitch=PitchCurve(times_s=(0.0, 1.0), f0_hz=(110.0, 440.0)),
        sample_rate=16000,
    )
    audio = get_backend().render(request).audio
    # Zero-crossing counts are dominated by the upper harmonics here, so estimate the
    # fundamental by autocorrelation instead (same method as the P1 F0 analyser).
    rate, width = 16000, 512  # 32 ms windows: the sweep moves < 10 Hz inside one
    estimates = []
    for index in range(4):
        centre = (index * 4000) + 2000
        window = audio[centre - width // 2:centre + width // 2]
        estimates.append(_autocorrelation_f0(window, rate))
    # 110 -> 440 Hz is two octaves, interpolated linearly in log2(f0).
    expected = [110.0 * 4.0 ** (index * 0.25 + 0.125) for index in range(4)]
    assert estimates == sorted(estimates), (estimates, expected)
    for estimate, target in zip(estimates, expected):
        assert estimate == pytest.approx(target, rel=0.08), (estimates, expected)


def test_registry_lists_backends_and_rejects_unknown():
    assert "formant_proxy" in list_backends()
    assert get_backend().supported_phonemes == tuple(PHONEME_INVENTORY)
    with pytest.raises(KeyError, match="Unknown synth backend"):
        get_backend("no_such_engine")


def test_wav_roundtrip_within_one_lsb(tmp_path):
    audio = get_backend().render(make_request()).audio
    path = tmp_path / "roundtrip.wav"
    write_wav(path, audio, 16000)
    loaded, rate = read_wav(path)
    assert rate == 16000
    assert loaded.shape == audio.shape
    # 16-bit storage costs up to two LSBs: write truncates towards zero after
    # scaling by 32767, read divides by 32768. Anything looser would hide a bug.
    assert np.max(np.abs(loaded - audio)) <= 2.0 / 32768.0


def test_read_wav_rejects_non_mono_and_non_16bit(tmp_path):
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros((10, 2), dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="mono"):
        read_wav(stereo)

    eight_bit = tmp_path / "eight.wav"
    with wave.open(str(eight_bit), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(bytes(range(1, 11)))
    with pytest.raises(ValueError, match="16-bit"):
        read_wav(eight_bit)
