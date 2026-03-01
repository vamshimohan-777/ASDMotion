"""Micro-kinetic motion encoder for short-range dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.video.motion.blocks import MicroKineticBlock


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int, eps: float = 1e-6) -> torch.Tensor:
    # Fast path: standard mean when no validity mask is provided.
    if mask is None:
        return x.mean(dim=dim)
    # Convert mask to float so it can be used as weights in multiplication/sum.
    w = mask.float()
    # Expand trailing dims until mask rank matches data rank for broadcasting.
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    # Weighted sum over the reduction dimension.
    num = (x * w).sum(dim=dim)
    # Sum of weights; clamp avoids divide-by-zero on fully-masked slices.
    den = w.sum(dim=dim).clamp(min=eps)
    return num / den


class MicroKineticMotionEncoder(nn.Module):
    """
    Short-range micro-kinetic encoder.

    Input:
      motion_windows: [B, T, J, F]
      joint_mask: [B, T, J] or None
    Output:
      embedding: [B, D]
    """

    def __init__(
        self,
        in_features: int = 9,
        temporal_channels: int = 128,
        num_blocks: int = 3,
        kernel_size: int = 7,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        residual: bool = True,
        use_dilation: bool = False,
        k_max: int = 8,
    ) -> None:
        super().__init__()
        # Keep normalized integer versions for stable downstream shape logic.
        self.embedding_dim = int(embedding_dim)
        self.temporal_channels = int(temporal_channels)
        # At least one event is always returned when event extraction is enabled.
        self.k_max = int(max(1, k_max))

        # Per-joint feature projection F -> C using 1x1 temporal conv.
        self.input_proj = nn.Sequential(
            nn.Conv1d(int(in_features), self.temporal_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.temporal_channels),
            nn.GELU(),
        )

        blocks = []
        # Stack temporal micro-kinetic blocks, optionally with exponential dilation.
        for i in range(int(max(1, num_blocks))):
            dilation = (2 ** i) if bool(use_dilation) else 1
            blocks.append(
                MicroKineticBlock(
                    channels=self.temporal_channels,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                    residual=bool(residual),
                )
            )
        # Temporal encoder over each joint trajectory.
        self.temporal_blocks = nn.Sequential(*blocks)
        # Final projection C -> D for window-level motion embedding.
        self.proj = nn.Sequential(
            nn.Linear(self.temporal_channels, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

        # Per-frame event saliency head from frame-level features.
        self.frame_score_head = nn.Linear(self.temporal_channels, 1)

    def forward(
        self,
        motion_windows: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        return_events: bool = False,
    ):
        # Timestamps are kept for call-signature compatibility with old callers.
        del timestamps
        if motion_windows.dim() != 4:
            raise ValueError(f"motion_windows must be [B,T,J,F], got {tuple(motion_windows.shape)}")
        b, t, j, f = motion_windows.shape

        # Flatten joints into batch so temporal conv runs per-joint sequence.
        x = motion_windows.permute(0, 2, 3, 1).reshape(b * j, f, t)  # [BJ,F,T]
        x = self.input_proj(x)
        x = self.temporal_blocks(x)  # [BJ,C,T]

        jm = None
        if joint_mask is not None:
            if joint_mask.shape != (b, t, j):
                raise ValueError(
                    f"joint_mask must be [B,T,J]={b,t,j}, got {tuple(joint_mask.shape)}"
                )
            # Match flattened joint layout [BJ,T].
            jm = joint_mask.permute(0, 2, 1).reshape(b * j, t)  # [BJ,T]

        # Temporal pooling per joint.
        joint_feat = _masked_mean(x.transpose(1, 2), jm, dim=1)  # [BJ,C]
        joint_feat = joint_feat.reshape(b, j, self.temporal_channels)  # [B,J,C]

        joint_valid = None
        if jm is not None:
            # A joint is valid if it has at least one valid frame in the window.
            joint_valid = (jm.sum(dim=1) > 0.0).reshape(b, j)
        # Joint pooling yields one motion vector per sample.
        motion_vec = _masked_mean(joint_feat, joint_valid, dim=1)  # [B,C]
        embedding = self.proj(motion_vec)  # [B,D]

        # Default mode: only return compact window embedding.
        if not return_events:
            return embedding

        # Rebuild frame-major features to score salient events over time.
        frame_feat = x.reshape(b, j, self.temporal_channels, t).permute(0, 3, 1, 2)  # [B,T,J,C]
        frame_valid = None
        if joint_mask is not None:
            # Frame is valid if any joint is valid in that frame.
            frame_valid = (joint_mask.sum(dim=-1) > 0.0)  # [B,T]
            frame_joint_mask = joint_mask.unsqueeze(-1)  # [B,T,J,1]
            # Pool joints per frame with joint validity.
            frame_feat = _masked_mean(frame_feat, frame_joint_mask, dim=2)  # [B,T,C]
        else:
            frame_feat = frame_feat.mean(dim=2)
            # All frames are valid when no mask is provided.
            frame_valid = torch.ones((b, t), dtype=torch.bool, device=motion_windows.device)

        # Raw event logits per frame; invalid frames are excluded from ranking.
        frame_logits = self.frame_score_head(frame_feat).squeeze(-1)
        frame_logits = frame_logits.masked_fill(~frame_valid, float("-inf"))
        frame_scores = torch.sigmoid(frame_logits)

        # Pick up to k_max salient frames; keep temporal order for downstream use.
        k = max(1, min(t, self.k_max))
        topk_logits, topk_idx = torch.topk(frame_logits, k=k, dim=1)
        order = torch.argsort(topk_idx, dim=1)
        topk_idx = topk_idx.gather(1, order)
        topk_logits = topk_logits.gather(1, order)
        topk_scores = torch.sigmoid(topk_logits)
        # Gather event feature vectors by selected frame indices.
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, frame_feat.size(-1))
        event_vectors = frame_feat.gather(1, idx)
        event_mask = frame_valid.gather(1, topk_idx)
        # Float representation is convenient for positional/time embedding layers.
        event_times = topk_idx.float()

        # Rich output is used by hierarchical/long-range temporal modules.
        return {
            "window_embedding": embedding,
            "event_vectors": event_vectors,
            "event_mask": event_mask,
            "event_times": event_times,
            "event_frame_index": topk_idx,
            "event_scores": topk_scores,
            "frame_event_scores": frame_scores,
            "frame_event_logits": frame_logits,
            "frame_valid_mask": frame_valid,
        }


class ResNetMicroKineticEventEncoder(MicroKineticMotionEncoder):
    """
    Backward-compatible alias used by older pipeline code.

    Old args are mapped to the new micro-kinetic motion encoder:
    - `d_model` -> `embedding_dim`
    - `micro_blocks` -> `num_blocks`
    """

    def __init__(
        self,
        d_model: int = 256,
        temporal_channels: int = 128,
        micro_blocks: int = 3,
        kernel_size: int = 7,
        use_dilation: bool = False,
        residual: bool = True,
        dropout: float = 0.1,
        k_max: int = 16,
    ) -> None:
        # Preserve old signature while delegating to the new base implementation.
        del k_max
        super().__init__(
            in_features=9,
            temporal_channels=int(temporal_channels),
            num_blocks=int(micro_blocks),
            kernel_size=int(kernel_size),
            embedding_dim=int(d_model),
            dropout=float(dropout),
            residual=bool(residual),
            use_dilation=bool(use_dilation),
            k_max=int(k_max),
        )
