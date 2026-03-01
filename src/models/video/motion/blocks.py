"""Building blocks for micro-kinetic temporal motion encoding."""

import torch
import torch.nn as nn


class MicroKineticBlock(nn.Module):
    """Depthwise-separable temporal block with optional residual connection."""

    def __init__(
        self,
        channels,
        kernel_size=7,
        dilation=1,
        dropout=0.1,
        residual=True,
    ):
        super().__init__()
        # Keep output length unchanged with "same" temporal padding.
        pad = ((kernel_size - 1) // 2) * dilation
        # Depthwise temporal filtering: each channel gets its own temporal kernel.
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        # Pointwise mixing lets channels interact after depthwise filtering.
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # Post-conv normalization/activation/dropout stack.
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # Residual improves optimization stability in deeper stacks.
        self.residual = bool(residual)

    def forward(self, x):
        # Temporal transform path.
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        # Residual path keeps original signal available to later layers.
        if self.residual:
            return x + h
        return h
