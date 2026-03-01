"""Fusion module for motion and RGB embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class MotionRGBFusion(nn.Module):
    def __init__(self, motion_dim: int, rgb_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # Store dimensions explicitly for runtime checks/fallback handling.
        self.motion_dim = int(motion_dim)
        self.rgb_dim = int(rgb_dim)
        self.out_dim = int(out_dim)

        in_dim = self.motion_dim + self.rgb_dim
        # Late-fusion projection after concatenating motion and RGB embeddings.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.out_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, motion_embedding: torch.Tensor, rgb_embedding: torch.Tensor | None = None) -> torch.Tensor:
        # Support motion-only inference by substituting zeros for missing RGB branch.
        if rgb_embedding is None:
            rgb_embedding = torch.zeros(
                (motion_embedding.size(0), self.rgb_dim),
                dtype=motion_embedding.dtype,
                device=motion_embedding.device,
            )
        # Concatenate along feature dimension and project to fused output space.
        x = torch.cat([motion_embedding, rgb_embedding], dim=-1)
        return self.mlp(x)
