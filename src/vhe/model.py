"""Small residual tensor-to-tensor *prototype*; NOT a trained humanizer.

No text, voice template, MIDI or speaker identity is accepted as VHE input.
The last residual layer initializes to zero: initially output == input.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualVHE(nn.Module):
    """Portable Conv1d prototype with the contract [B, 1, N] -> [B, 1, N]."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        if channels < 4:
            raise ValueError("channels must be >= 4")
        self.network = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=8, dilation=4),
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=9, padding=4),
        )
        last = self.network[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, mechanical_audio: torch.Tensor) -> torch.Tensor:
        if mechanical_audio.ndim != 3 or mechanical_audio.shape[1] != 1:
            raise ValueError("VHE requires mono float audio with shape [B,1,N]")
        if mechanical_audio.shape[-1] < 64:
            raise ValueError("VHE requires at least 64 audio samples")
        if not mechanical_audio.is_floating_point():
            raise TypeError("VHE requires floating point PCM audio")
        return mechanical_audio + self.network(mechanical_audio)

class _DilatedBlock(nn.Module):
    """Pre-activation-free dilated residual conv stack (no normalisation).

    BatchNorm is unusable here (batch 4-8, non-stationary audio) and LayerNorm over time
    would leak future context across the whole utterance, so the block keeps a plain
    residual sum and relies on the zero-initialised output head for identity start-up.
    """

    def __init__(self, channels: int, dilations) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.activation(layer(x))
        return x


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Up(nn.Module):
    """Nearest-neighbour upsample + conv instead of ConvTranspose1d.

    Transposed convolution with stride 2 and kernel 4 produces a periodic comb in the
    impulse response unless the kernel is exactly interpolating; the comb lands in the
    audio band and is indistinguishable from added buzz. Upsample+conv has one filter tap
    per phase to learn, so EXP-0008 measures its impulse response instead of assuming it.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ResidualUNet(nn.Module):
    """v0.1 humanizer: 1-D residual U-Net over the raw waveform, output = input + edit.

    Contract: [B, 1, N] float32 in, [B, 1, N] out, same sample rate. Nothing else is
    accepted: no F0, phonemes, speaker id or reference audio. The visual/measurable
    consequence is that the network can only apply a local, waveform-level correction;
    anything it cannot infer from the mechanical audio itself is out of reach by
    construction (see docs/VHE_ARCHITECTURE_RESEARCH.md).
    """

    STRIDE = 8  # 3 downsampling stages of factor 2

    def __init__(self, channels: int = 32, dilations=(1, 2, 4, 8),
                 bottleneck_dilations=(1, 2, 4, 8, 16, 32)) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be >= 8")
        c = channels
        self.stem = nn.Conv1d(1, c, kernel_size=9, padding=4)
        self.enc1 = _DilatedBlock(c, dilations)
        self.down1 = _Down(c, 2 * c)
        self.enc2 = _DilatedBlock(2 * c, dilations)
        self.down2 = _Down(2 * c, 4 * c)
        self.enc3 = _DilatedBlock(4 * c, dilations)
        self.down3 = _Down(4 * c, 4 * c)
        self.mid = _DilatedBlock(4 * c, bottleneck_dilations)
        self.up3 = _Up(4 * c, 4 * c)
        self.dec3 = _DilatedBlock(4 * c, dilations)
        self.up2 = _Up(4 * c, 2 * c)
        self.dec2 = _DilatedBlock(2 * c, dilations)
        self.up1 = _Up(2 * c, c)
        self.dec1 = _DilatedBlock(c, dilations)
        self.head = nn.Conv1d(c, 1, kernel_size=9, padding=4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        remainder = x.shape[-1] % self.STRIDE
        if remainder == 0:
            return x, 0
        pad = self.STRIDE - remainder
        # Reflect padding keeps the boundary periodic instead of introducing a step to zero,
        # which the dilated convs would otherwise smear into the first/last few ms.
        return torch.nn.functional.pad(x, (0, pad), mode="reflect"), pad

    def forward(self, mechanical_audio: torch.Tensor) -> torch.Tensor:
        if mechanical_audio.ndim != 3 or mechanical_audio.shape[1] != 1:
            raise ValueError("VHE requires mono float audio with shape [B,1,N]")
        if mechanical_audio.shape[-1] < 64:
            raise ValueError("VHE requires at least 64 audio samples")
        if not mechanical_audio.is_floating_point():
            raise TypeError("VHE requires floating point PCM audio")
        x, pad = self._pad(mechanical_audio)
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        mid = self.mid(self.down3(e3))
        d3 = self.dec3(self.up3(mid) + e3)
        d2 = self.dec2(self.up2(d3) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        edit = self.head(d1)
        if pad:
            edit = edit[..., : edit.shape[-1] - pad]
        return mechanical_audio + edit


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    """Factory used by the CLI and the training loop."""
    if name == "unet":
        return ResidualUNet(**kwargs)
    if name == "prototype":
        return ResidualVHE(**kwargs)
    raise ValueError(f"unknown model {name!r}")
