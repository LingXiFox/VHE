"""Band-limited rational resampling in NumPy (no scipy dependency).

AISHELL-3 is 44.1 kHz while the synthesizer contract only accepts 16/24/48 kHz, so the
pairing pipeline must resample. Naive linear interpolation would alias the fricative
energy above the new Nyquist frequency and quietly change the content; this module uses a
windowed-sinc polyphase kernel instead and refuses ratios it cannot do well.

Determinism: the kernel is built from the exact rational ratio (Fraction), not from a
floating-point approximation, so 44100 -> 24000 always uses L/M = 80/147 and produces the
same samples on every platform.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

MAX_PHASES = 2000  # guards against absurd rates; above this the kernel table gets silly


def _kaiser_window(x: np.ndarray, beta: float) -> np.ndarray:
    inside = np.clip(1.0 - x * x, 0.0, 1.0)
    return np.i0(beta * np.sqrt(inside)) / np.i0(beta)


def _kernel_table(phases: int, taps: int, cutoff: float, beta: float) -> np.ndarray:
    """Rows of FIR weights, one per fractional phase, DC-normalised to unit gain."""
    half = (taps - 1) / 2.0
    offsets = np.arange(taps) - half
    table = np.empty((phases, taps), dtype=np.float64)
    for phase in range(phases):
        frac = phase / phases
        distance = frac - offsets
        table[phase] = 2.0 * cutoff * np.sinc(2.0 * cutoff * distance) * _kaiser_window(
            distance / half, beta)
    table /= table.sum(axis=1, keepdims=True)
    return table


def resample(audio: np.ndarray, src_rate: int, dst_rate: int, *,
             taps_per_side: int = 16, beta: float = 8.6) -> np.ndarray:
    """Resample mono float audio from src_rate to dst_rate; returns float32.

    The output length is round(len(audio) * dst_rate / src_rate) exactly, so repeated
    resampling does not drift in duration.
    """
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {src_rate} -> {dst_rate}")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("resample expects mono audio with shape [N]")
    if signal.size == 0:
        raise ValueError("resample expects a non-empty signal")
    if src_rate == dst_rate:
        return signal.astype(np.float32)

    # Output sample n sits at input position n * src/dst = n * step / phases, so the
    # kernel needs exactly phases fractional offsets and advances step input samples
    # per output sample.
    ratio = Fraction(src_rate, dst_rate)
    step, phases = ratio.numerator, ratio.denominator
    if phases > MAX_PHASES:
        raise ValueError(f"unsupported ratio {dst_rate}/{src_rate}: {phases} phases")

    # Cutoff at the *lower* Nyquist frequency, expressed in cycles per input sample.
    cutoff = 0.5 * min(1.0, phases / step)
    half = int(np.ceil(taps_per_side / min(1.0, phases / step)))
    taps = 2 * half + 1
    kernel = _kernel_table(phases, taps, cutoff, beta)

    padded = np.pad(signal, (half, half), mode="edge")
    n_out = int(round(signal.size * phases / step))
    offsets = np.arange(-half, half + 1)
    positions = np.arange(n_out, dtype=np.int64) * step
    base, phase = np.divmod(positions, phases)

    out = np.empty(n_out, dtype=np.float64)
    chunk = 1 << 15
    for start in range(0, n_out, chunk):
        stop = min(start + chunk, n_out)
        indices = base[start:stop, None] + offsets[None, :] + half
        out[start:stop] = np.sum(padded[indices] * kernel[phase[start:stop]], axis=1)
    return out.astype(np.float32)
