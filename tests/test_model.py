import pytest

torch = pytest.importorskip("torch")
from vhe.model import ResidualVHE
from vhe.smoke import run_smoke


def test_prototype_preserves_input_at_initialization():
    model = ResidualVHE()
    x = torch.randn(2, 1, 777, dtype=torch.float32)
    y = model(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)


def test_prototype_validates_shape_and_type():
    model = ResidualVHE()
    with pytest.raises(ValueError, match="mono"):
        model(torch.randn(1, 2, 128))
    with pytest.raises(ValueError, match="64"):
        model(torch.randn(1, 1, 16))
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(1, 1, 128, dtype=torch.int16))


def test_cpu_gradient_and_optimizer_smoke():
    report = run_smoke("cpu", samples=512, steps=2)
    assert report["selected_device"] == "cpu"
    assert report["status"] == "backend-smoke-passed-not-humanization"
    assert report["output_shape"] == [1, 1, 512]
    assert len(report["losses"]) == 2
    assert report["losses"][-1] < report["losses"][0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual NVIDIA CUDA")
def test_cuda_smoke_if_available():
    assert run_smoke("cuda", samples=512, steps=2)["selected_device"] == "cuda"


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="requires actual Apple MPS")
def test_mps_smoke_if_available():
    assert run_smoke("mps", samples=512, steps=2)["selected_device"] == "mps"


def test_portable_checkpoint_roundtrip_and_no_overwrite(tmp_path):
    from vhe.checkpoint import load_weights, save_weights

    model = ResidualVHE()
    x = torch.randn(1, 1, 256)
    # Simulate learned weights, not speech training.
    with torch.no_grad():
        model.network[-1].bias.fill_(0.01)
    before = model(x).detach().clone()
    path = tmp_path / "weights.pt"
    save_weights(path, model)
    with pytest.raises(FileExistsError):
        save_weights(path, model)
    loaded = load_weights(path, ResidualVHE(), device="cpu")
    assert torch.equal(before, loaded(x))
