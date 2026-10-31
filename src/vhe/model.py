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
