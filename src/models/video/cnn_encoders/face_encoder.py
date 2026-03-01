# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def _build_backbone(backbone_name: str, pretrained: bool):
    # Normalize user input so backbone selection is robust to case/None variants.
    name = str(backbone_name or "resnet18").lower()
    if name == "resnet50":
        # Stronger backbone for richer facial micro-pattern representation.
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return resnet50(weights=weights), 2048
    if name == "resnet18":
        # Lighter backbone for lower latency with good baseline facial cues.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet18(weights=weights), 512
    # Fail fast on unsupported names to avoid silent architecture mismatches.
    raise ValueError(f"Unsupported backbone: {backbone_name}")


class FaceN(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_fc_head: bool = True,
    ):
        super().__init__()

        # Build selected image backbone that extracts high-level face descriptors.
        backbone, feature_dim = _build_backbone(backbone_name, pretrained)
        if use_fc_head:
            # Keep a learnable FC head for task adaptation before final projection.
            backbone.fc = nn.Linear(feature_dim, feature_dim)
        else:
            # Identity head exposes raw pooled backbone features.
            backbone.fc = nn.Identity()
        self.backbone = backbone

        # Shared projection aligns face features with multimodal/event encoder width.
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

    def forward(self, x):
        # Extract per-frame facial representation.
        x = self.backbone(x)
        # Project to compact dimension expected by downstream detection modules.
        x = self.proj(x)
        return x

