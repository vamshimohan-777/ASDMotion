"""ResNet18 RGB context branch."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18RGBBranch(nn.Module):
    """
    Frame-level RGB encoder with temporal average pooling.

    Input:
      rgb_windows: [B, T, 3, 224, 224] (float in [0,1] or uint8)
      frame_mask: [B, T] optional
    Output:
      rgb_embedding: [B, D]
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        # Optional ImageNet initialization for stronger visual priors.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Remove classification head; keep spatial feature extractor.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [N,512,1,1]
        # Project CNN features to model embedding space.
        self.proj = nn.Sequential(
            nn.Linear(512, int(embedding_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(embedding_dim)),
        )
        self.embedding_dim = int(embedding_dim)
        # ImageNet normalization buffers (broadcast over [B,T,C,H,W]).
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def freeze_backbone(self) -> None:
        # Useful for staged training where only projection/fusion is tuned.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        # Re-enable full finetuning.
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, rgb_windows: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Expected shape is windowed RGB clips.
        if rgb_windows.dim() != 5:
            raise ValueError(f"rgb_windows must be [B,T,3,224,224], got {tuple(rgb_windows.shape)}")
        b, t, c, h, w = rgb_windows.shape
        if c != 3:
            raise ValueError(f"rgb_windows channel must be 3, got {c}")

        # Convert uint8-like inputs to normalized float tensors.
        x = rgb_windows.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - self._mean) / self._std
        # Flatten time into batch for per-frame CNN inference.
        x = x.reshape(b * t, c, h, w)

        # Encode each frame then restore [B,T,D].
        feat = self.backbone(x).flatten(1)  # [BT,512]
        feat = self.proj(feat).reshape(b, t, self.embedding_dim)  # [B,T,D]

        # Aggregate frames uniformly if no validity mask is provided.
        if frame_mask is None:
            return feat.mean(dim=1)
        if frame_mask.shape != (b, t):
            raise ValueError(f"frame_mask must be [B,T]={b,t}, got {tuple(frame_mask.shape)}")
        # Masked temporal average over valid frames.
        wgt = frame_mask.float().unsqueeze(-1)
        return (feat * wgt).sum(dim=1) / wgt.sum(dim=1).clamp(min=1e-6)
