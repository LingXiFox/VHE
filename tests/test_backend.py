import pytest

from vhe.backend import inspect_runtime, resolve_device


def test_runtime_has_platform_information():
    report = inspect_runtime("auto")
    assert report["python"]
    assert report["platform"]
    assert report["requested_device"] == "auto"


def test_explicit_cpu_selection():
    torch = pytest.importorskip("torch")
    assert resolve_device("cpu", torch_module=torch).type == "cpu"


def test_auto_matches_backend_availability():
    torch = pytest.importorskip("torch")
    chosen = resolve_device("auto", torch_module=torch).type
    expected = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    assert chosen == expected


def test_missing_explicit_accelerator_is_not_silent_cpu_fallback():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            resolve_device("cuda", torch_module=torch)
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS requested"):
            resolve_device("mps", torch_module=torch)


def test_invalid_device():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_auto_priority_with_simulated_backends():
    # Pure dispatch simulation, NOT an actual CUDA/MPS hardware test.
    from types import SimpleNamespace
    for cuda, mps, expected in (
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ):
        fake = SimpleNamespace(
            device=str,
            cuda=SimpleNamespace(is_available=lambda value=cuda: value),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda value=mps: value)),
        )
        assert resolve_device("auto", torch_module=fake) == expected
