import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


def _build_resnet_backbone(backbone_name: str, pretrained: bool):
    name = str(backbone_name).lower().strip()
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet50(weights=weights), 2048
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet18(weights=weights), 512
    raise ValueError(f"Unsupported backbone '{backbone_name}'. Use 'resnet18' or 'resnet50'.")


class MotionN(nn.Module):
    def __init__(self, pretrained: bool = True, backbone_name: str = "resnet18"):
        super().__init__()

        backbone, feat_dim = _build_resnet_backbone(backbone_name, pretrained)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x
