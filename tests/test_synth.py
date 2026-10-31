import numpy as np
import pytest

from vhe.synth import PRESETS, synthesize, write_wav


def test_audio_shape_dtype_and_limits():
    x = synthesize(["a", "i", "u"], sample_rate=16000, duration_s=0.3)
    assert x.shape == (4800,)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert 0.05 < np.max(np.abs(x)) <= 1.0


def test_determinism_with_seed():
    options = dict(phonemes=["a", "i"], sample_rate=16000,
                   duration_s=0.2, template=PRESETS["yin"], seed=12)
    assert np.array_equal(synthesize(**options), synthesize(**options))


def test_presets_are_distinct():
    a = synthesize(["a"], sample_rate=16000, duration_s=0.2, template=PRESETS["yang"])
    b = synthesize(["a"], sample_rate=16000, duration_s=0.2, template=PRESETS["yin"])
    assert not np.array_equal(a, b)


def test_fails_on_unsupported_phoneme():
    with pytest.raises(ValueError, match="a,i,u"):
        synthesize(["yuan"])


def test_does_not_overwrite(tmp_path):
    path = tmp_path / "example.wav"
    x = synthesize(["a"], sample_rate=16000, duration_s=0.15)
    write_wav(path, x, 16000)
    assert path.stat().st_size > 44
    with pytest.raises(FileExistsError):
        write_wav(path, x, 16000)
