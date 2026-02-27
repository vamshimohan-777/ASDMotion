import torch
import torch.nn as nn


class MicroKineticBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=7,
        dilation=1,
        dropout=0.1,
        residual=True,
    ):
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = bool(residual)

    def forward(self, x):
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        if self.residual:
            return x + h
        return h

