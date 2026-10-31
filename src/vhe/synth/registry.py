"""Backend registry: the synthesizer is swappable, the contract is not."""

from __future__ import annotations

from .contract import SynthBackend
from .formant_proxy import FormantProxySynth
from .mandarin import MandarinFormantSynth

DEFAULT_BACKEND = "formant_proxy"

_FACTORIES = {
    FormantProxySynth.name: FormantProxySynth,
    MandarinFormantSynth.name: MandarinFormantSynth,
}


def list_backends() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def get_backend(name: str = DEFAULT_BACKEND) -> SynthBackend:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(f"Unknown synth backend {name!r}; available: {list_backends()}") from None
    backend = factory()
    if not isinstance(backend, SynthBackend):  # runtime_checkable structural check
        raise TypeError(f"Backend {name!r} does not satisfy the SynthBackend protocol")
    return backend
