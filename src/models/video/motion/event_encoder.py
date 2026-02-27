import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.models.video.motion.blocks import MicroKineticBlock


class ResNetMicroKineticEventEncoder(nn.Module):
    """
    Frame-level ResNet18 + temporal micro-kinetic event detector.

    Input:
      motion_windows: [B, W, J, 9]
      joint_mask: [B, W, J] (optional)
      timestamps: [B, W] (optional)
    Output:
      - window_embedding: [B, D]
      - event_vectors: [B, K, D]
      - event_mask: [B, K]
      - event_times: [B, K]
      - event_frame_index: [B, K]
      - frame_event_scores: [B, W]
    """

    def __init__(
        self,
        d_model=256,
        temporal_channels=256,
        micro_blocks=3,
        kernel_size=7,
        use_dilation=True,
        residual=True,
        dropout=0.2,
        k_max=16,
    ):
        super().__init__()
        self.embedding_dim = int(d_model)
        self.k_max = int(max(1, k_max))

        backbone = models.resnet18(weights=None)
        self.resnet_backbone = nn.Sequential(*list(backbone.children())[:-1])  # [N,512,1,1]
        self.frame_proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.temporal_in = nn.Sequential(
            nn.Conv1d(d_model, temporal_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.GELU(),
        )
        blocks = []
        for i in range(int(max(1, micro_blocks))):
            dilation = (2 ** i) if use_dilation else 1
            blocks.append(
                MicroKineticBlock(
                    channels=temporal_channels,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                    residual=bool(residual),
                )
            )
        self.temporal_blocks = nn.Sequential(*blocks)
        self.temporal_out = nn.Sequential(
            nn.Conv1d(temporal_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.event_score_head = nn.Linear(d_model, 1)
        self.window_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _motion_to_rgb_image(motion_windows):
        # motion_windows: [B, W, J, 9]
        b, w, j, f = motion_windows.shape
        if f != 9:
            raise ValueError(f"Expected feature dim 9, got {f}")

        # Convert [J,9] to pseudo-image [3,J,3], then resize for ResNet.
        x = motion_windows.reshape(b * w, j, 3, 3).permute(0, 2, 1, 3).contiguous()  # [BW,3,J,3]
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return x

    def _frame_valid_mask(self, joint_mask, b, w, device):
        if joint_mask is None:
            return torch.ones((b, w), dtype=torch.bool, device=device)
        # frame valid if any joint is present.
        return (joint_mask.float().sum(dim=-1) > 0)

    def forward(self, motion_windows, joint_mask=None, timestamps=None, return_events=False):
        b, w, _, _ = motion_windows.shape
        device = motion_windows.device

        img = self._motion_to_rgb_image(motion_windows)
        feat_2d = self.resnet_backbone(img).flatten(1)  # [BW,512]
        frame_feat = self.frame_proj(feat_2d).reshape(b, w, -1)  # [B,W,D]

        x = frame_feat.transpose(1, 2)  # [B,D,W]
        x = self.temporal_in(x)
        x = self.temporal_blocks(x)
        x = self.temporal_out(x)
        x = x.transpose(1, 2)  # [B,W,D]

        frame_valid = self._frame_valid_mask(joint_mask, b, w, device=device)
        frame_scores = self.event_score_head(x).squeeze(-1)  # [B,W]
        frame_scores = frame_scores.masked_fill(~frame_valid, float("-inf"))

        k = min(int(self.k_max), int(w))
        topk_scores, topk_idx = torch.topk(frame_scores, k=k, dim=1)
        # Preserve temporal order in selected events.
        order = torch.argsort(topk_idx, dim=1)
        topk_idx = topk_idx.gather(1, order)
        topk_scores = topk_scores.gather(1, order)

        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        event_vectors = x.gather(1, idx_exp)  # [B,K,D]
        event_mask = frame_valid.gather(1, topk_idx)  # [B,K]

        if timestamps is None:
            event_times = topk_idx.float()
        else:
            event_times = timestamps.gather(1, topk_idx)

        # Window embedding: weighted average by sigmoid(event score) over valid frames.
        wgt = torch.sigmoid(self.event_score_head(x).squeeze(-1))
        wgt = wgt * frame_valid.float()
        wgt = wgt / wgt.sum(dim=1, keepdim=True).clamp(min=1e-6)
        window_embedding = (x * wgt.unsqueeze(-1)).sum(dim=1)
        window_embedding = self.window_pool(window_embedding)

        if not return_events:
            return window_embedding

        return {
            "window_embedding": window_embedding,
            "event_vectors": event_vectors,
            "event_mask": event_mask,
            "event_times": event_times,
            "event_frame_index": topk_idx,
            "event_scores": topk_scores,
            "frame_event_scores": torch.sigmoid(frame_scores),
        }

