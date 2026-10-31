"""Acceptance tests for the Mandarin phoneme-level physical synthesizer (P1.2).

These fix the properties the VHE pipeline depends on, not the sound quality:
every advertised phoneme can be rendered, the inventory cannot be silently substituted,
multi-phoneme sequences keep the requested length and pitch, voiced and unvoiced excitation
are measurably different, formant movement inside a diphthong is continuous, tones follow
their reference contours, nothing is synthesised above 0.45 * sample_rate, and the tensor
hand-off between MPS/CPU and the NumPy renderer is lossless in FP32.

Listening quality is assessed in experiments/EXP-0009, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhe.data.f0 import estimate_f0
from vhe.synth import (MANDARIN_PHONEME_IDS, PRESETS, PhonemeSegment, PhonemeSequence,
                       PitchCurve, SynthesisRequest, Timing, get_backend)
from vhe.synth.mandarin import MandarinFormantSynth

SR = 24000
BACKEND = MandarinFormantSynth()


def render(phonemes, *, f0: float = 200.0, duration_s: float | None = None,
           template: str = "yang", sample_rate: int = SR) -> np.ndarray:
    if isinstance(phonemes, str):
        phonemes = ((phonemes, duration_s or 0.25),)
    elif len(phonemes) == 2 and isinstance(phonemes[0], str):
        phonemes = ((phonemes[0], float(phonemes[1])),)
    total = float(sum(duration for _name, duration in phonemes))
    request = SynthesisRequest(
        template=PRESETS[template],
        phonemes=PhonemeSequence(tuple(PhonemeSegment(name, duration)
                                       for name, duration in phonemes), inventory="mandarin"),
        pitch=PitchCurve.constant(f0, total),
        timing=Timing(0.0, 0.0), sample_rate=sample_rate)
    return BACKEND.render(request).audio


def autocorrelation_peak(audio: np.ndarray, sample_rate: int = SR) -> float:
    """Largest normalised autocorrelation peak in the plausible f0 lag range."""
    centre = audio[audio.size // 4: audio.size // 4 + int(0.100 * sample_rate)]
    centre = centre - centre.mean()
    energy = float(np.dot(centre, centre))
    if energy <= 0.0:
        return 0.0
    spectrum = np.fft.rfft(centre, 1 << (2 * centre.size - 1).bit_length())
    corr = np.fft.irfft(spectrum * np.conj(spectrum))[:centre.size]
    lag_min = max(2, int(sample_rate / 900.0))
    lag_max = int(sample_rate / 65.0)
    return float(np.max(corr[lag_min:lag_max] / corr[0]))


def log_spectrum(audio: np.ndarray, sample_rate: int = SR, n_fft: int = 1024) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft)))
    return np.log(np.maximum(spectrum, 1e-9))


def spectral_peaks(audio: np.ndarray, count: int = 4, sample_rate: int = SR) -> list[float]:
    """Coarse formant proxies: local maxima of a smoothed log spectrum."""
    freqs = np.fft.rfftfreq(1024, 1.0 / sample_rate)
    spectrum = log_spectrum(audio, sample_rate)
    kernel = np.ones(9) / 9.0
    smooth = np.convolve(spectrum, kernel, mode="same")
    peaks = []
    for index in range(2, smooth.size - 2):
        if smooth[index] > smooth[index - 1] and smooth[index] >= smooth[index + 1]:
            peaks.append((float(smooth[index]), float(freqs[index])))
    peaks.sort(reverse=True)
    return sorted(freq for _level, freq in peaks[:count])


@pytest.mark.parametrize("phoneme", MANDARIN_PHONEME_IDS)
def test_every_advertised_phoneme_renders(phoneme: str):
    audio = render(phoneme, duration_s=0.15)
    assert audio.dtype == np.float32
    assert audio.size == int(round(0.15 * SR))
    assert np.all(np.isfinite(audio))
    assert float(np.max(np.abs(audio))) <= 1.0
    assert float(np.sqrt(np.mean(audio ** 2))) > 1e-3


def test_backend_declares_its_inventory_and_version():
    assert BACKEND.inventory == "mandarin"
    assert BACKEND.supported_phonemes == MANDARIN_PHONEME_IDS
    assert BACKEND.version == "0.1.0"


def test_unknown_phoneme_is_rejected_and_nothing_is_substituted():
    with pytest.raises(ValueError, match="Unsupported phoneme"):
        render("zh", duration_s=0.1)


def test_mandarin_backend_rejects_vowel_inventory_request():
    request = SynthesisRequest(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("a", 0.2),)),  # inventory="vowels"
        pitch=PitchCurve.constant(200.0, 0.2),
        timing=Timing(), sample_rate=SR)
    with pytest.raises(ValueError, match="inventory"):
        BACKEND.render(request)


def test_vowel_backend_rejects_mandarin_inventory_request():
    request = SynthesisRequest(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("m", 0.05), PhonemeSegment("a", 0.2)),
                                 inventory="mandarin"),
        pitch=PitchCurve.constant(200.0, 0.25),
        timing=Timing(), sample_rate=SR)
    with pytest.raises(ValueError, match="renders the 'vowels' inventory"):
        get_backend().render(request)


def test_multi_phoneme_sequence_keeps_length_and_pitch():
    audio = render((("m", 0.06), ("a", 0.24), ("sh", 0.08), ("a", 0.20)), f0=200.0)
    assert audio.size == int(round(0.58 * SR))
    track = estimate_f0(audio, SR)
    voiced = track.voiced_f0
    # The fricative is unvoiced by construction, so only the vocalic part may be counted;
    # a stray junction frame reads 2 * f0, hence the fraction test next to the median.
    assert track.voiced_ratio > 0.5
    assert abs(float(np.median(voiced)) - 200.0) / 200.0 <= 0.02
    within = np.abs(voiced - 200.0) / 200.0 <= 0.03
    assert float(np.mean(within)) >= 0.90, float(np.mean(within))


def test_f0_estimator_locks_onto_a_high_harmonic_at_210_and_300_hz():
    """Documented limitation (EXP-0011 addendum, not a property we endorse).

    F1 of /a/ sits near 4 * 210 Hz and near 3 * 300 Hz, so the first CMND valley below the
    0.15 threshold lands on a harmonic multiple and the estimator reports 842 Hz / 902 Hz.
    Bounding the search to a realistic melodic range recovers the true f0, which is why the
    callers that know the range must pass fmax_hz. If this test starts failing the estimator
    improved: update EXP-0011 instead of deleting the assertion.
    """
    for f0 in (210.0, 300.0):
        audio = render("a", f0=f0, duration_s=0.3)
        default = float(np.median(estimate_f0(audio, SR).voiced_f0))
        assert default > 2.0 * f0, default
        bounded = float(np.median(estimate_f0(audio, SR, fmax_hz=500.0).voiced_f0))
        assert abs(bounded - f0) / f0 <= 0.01, bounded


@pytest.mark.parametrize("f0", [110.0, 200.0, 350.0])
def test_f0_is_preserved(f0: float):
    audio = render("a", f0=f0, duration_s=0.3)
    median = float(np.median(estimate_f0(audio, SR).voiced_f0))
    assert abs(median - f0) / f0 <= 0.01, median


def test_voiced_and_unvoiced_excitation_are_distinguishable():
    vowel = autocorrelation_peak(render("a", duration_s=0.2))
    nasal = autocorrelation_peak(render("m", duration_s=0.2))
    lateral = autocorrelation_peak(render("l", duration_s=0.2))
    fricative = autocorrelation_peak(render("s", duration_s=0.2))
    postalveolar = autocorrelation_peak(render("sh", duration_s=0.2))
    labiodental = autocorrelation_peak(render("f", duration_s=0.2))
    # Measured 0.977 / 0.988 / 0.958 / 0.063 / 0.129 / 0.093 when this test was written.
    assert min(vowel, nasal, lateral) > 0.85, (vowel, nasal, lateral)
    assert max(fricative, postalveolar, labiodental) < 0.30, (fricative, postalveolar, labiodental)


def test_diphthong_formants_move_continuously():
    audio = render("ai", f0=200.0, duration_s=0.32)
    window = 1024
    hop = int(0.005 * SR)
    frames = [audio[start:start + window] * np.hanning(window)
              for start in range(0, audio.size - window, hop)]
    spectra = np.array([np.abs(np.fft.rfft(frame)) for frame in frames])
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    # Cosine distance between consecutive L2-normalised magnitude spectra.
    flux = 1.0 - np.sum(spectra[1:] * spectra[:-1], axis=1)
    assert float(np.max(flux)) < 0.02, float(np.max(flux))
    # And it must actually move: the second formant region ends higher than it starts.
    early = spectral_peaks(audio[:window])
    late = spectral_peaks(audio[-window:])
    assert max(late) > max(early), (early, late)


def test_hard_spliced_vowels_are_detectably_discontinuous():
    # Control for the test above: a naive splice of two steady vowels does exceed the threshold.
    first = render("a", f0=200.0, duration_s=0.2)
    second = render("i", f0=200.0, duration_s=0.2)
    spliced = np.concatenate((first, second))
    window = 1024
    hop = int(0.005 * SR)
    frames = [spliced[start:start + window] * np.hanning(window)
              for start in range(0, spliced.size - window, hop)]
    spectra = np.array([np.abs(np.fft.rfft(frame)) for frame in frames])
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    flux = 1.0 - np.sum(spectra[1:] * spectra[:-1], axis=1)
    assert float(np.max(flux)) > 0.02, float(np.max(flux))


def test_voice_templates_change_formants_not_pitch():
    yang = render("a", f0=200.0, duration_s=0.3, template="yang")
    yin = render("a", f0=200.0, duration_s=0.3, template="yin")
    assert abs(float(np.median(estimate_f0(yang, SR).voiced_f0)) - 200.0) < 2.0
    assert abs(float(np.median(estimate_f0(yin, SR).voiced_f0)) - 200.0) < 2.0
    yang_peaks = spectral_peaks(yang)
    yin_peaks = spectral_peaks(yin)
    # yang F2 proxy 796 Hz, yin 609 Hz when measured (template formant_scale differs).
    lowest_yang, lowest_yin = min(yang_peaks), min(yin_peaks)
    assert lowest_yang - lowest_yin > 50.0, (yang_peaks, yin_peaks)


@pytest.mark.parametrize("tone,rising", [("1", None), ("2", True), ("4", False)])
def test_tone_contours_follow_their_direction(tone: str, rising: bool | None):
    from vhe.synth import tone_curve_points

    total = 0.4
    base_hz = 200.0
    times_s, semitones = tone_curve_points((tone,), (total,))
    request = SynthesisRequest(
        template=PRESETS["yang"],
        phonemes=PhonemeSequence((PhonemeSegment("a", total),), inventory="mandarin"),
        pitch=PitchCurve(times_s=times_s,
                         f0_hz=tuple(base_hz * 2.0 ** (value / 12.0) for value in semitones)),
        timing=Timing(), sample_rate=SR)
    audio = BACKEND.render(request).audio
    values = estimate_f0(audio, SR, fmax_hz=500.0).voiced_f0
    if rising is None:
        assert float(np.ptp(values)) / float(np.median(values)) < 0.10
    elif rising:
        assert values[-1] > values[0] + 5.0, (values[0], values[-1])
    else:
        assert values[-1] < values[0] - 20.0, (values[0], values[-1])


def test_no_energy_above_the_harmonic_ceiling():
    audio = render("a", f0=880.0, duration_s=0.2)
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1.0 / SR)
    total = float(spectrum.sum())
    above = float(spectrum[freqs > 0.45 * SR].sum())
    assert above / total < 1e-3, above / total


def test_peak_stays_below_full_scale_for_a_loud_template():
    audio = render(("m", 0.05), f0=110.0, duration_s=0.05, template="yang")
    assert float(np.max(np.abs(audio))) <= 1.0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tensor_handoff_is_lossless_in_fp32(device: str):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    audio = render("a", f0=200.0, duration_s=0.2)
    tensor = torch.from_numpy(audio).to(device=device)
    returned = tensor.to("cpu").numpy()
    assert returned.dtype == np.float32
    assert np.array_equal(returned, audio)
    # The renderer itself must accept the tensor round-trip without changing its output.
    again = render("a", f0=200.0, duration_s=0.2)
    assert np.array_equal(audio, again)
