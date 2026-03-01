"""Modality-aware wrappers over micro-kinetic motion encoders."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA, LandmarkSchema
from src.models.video.motion.event_encoder import MicroKineticMotionEncoder


class TemporalBranchEncoder(MicroKineticMotionEncoder):
    """Backward-compatible branch encoder name."""


class MultiBranchMotionEncoder(nn.Module):
    """
    Encodes selected landmark modalities and fuses them into one embedding.

    Supports phased rollout:
    - Phase 1: enabled_modalities=("pose",)
    - Later: add "hands" and "face"
    """

    def __init__(
        self,
        schema: LandmarkSchema = DEFAULT_SCHEMA,
        enabled_modalities: tuple[str, ...] = ("pose",),
        in_feat: int = 9,
        branch_channels: int = 128,
        branch_blocks: int = 3,
        kernel_size: int = 7,
        use_dilation: bool = False,
        residual: bool = True,
        branch_dropout: float = 0.1,
        embedding_dim: int = 256,
        fusion_dim: int = 256,
    ) -> None:
        super().__init__()
        # Shared schema exposes stable landmark index ranges per modality.
        self.schema = schema
        # Normalize names to avoid casing-related mismatches.
        self.enabled_modalities = tuple(m.lower() for m in enabled_modalities)
        # Public output dimension after multi-branch fusion.
        self.embedding_dim = int(fusion_dim)

        # Independent per-modality branches (same architecture/hyperparameters).
        self.pose_encoder = MicroKineticMotionEncoder(
            in_features=int(in_feat),
            temporal_channels=int(branch_channels),
            num_blocks=int(branch_blocks),
            kernel_size=int(kernel_size),
            embedding_dim=int(embedding_dim),
            dropout=float(branch_dropout),
            residual=bool(residual),
            use_dilation=bool(use_dilation),
        )
        self.hand_encoder = MicroKineticMotionEncoder(
            in_features=int(in_feat),
            temporal_channels=int(branch_channels),
            num_blocks=int(branch_blocks),
            kernel_size=int(kernel_size),
            embedding_dim=int(embedding_dim),
            dropout=float(branch_dropout),
            residual=bool(residual),
            use_dilation=bool(use_dilation),
        )
        self.face_encoder = MicroKineticMotionEncoder(
            in_features=int(in_feat),
            temporal_channels=int(branch_channels),
            num_blocks=int(branch_blocks),
            kernel_size=int(kernel_size),
            embedding_dim=int(embedding_dim),
            dropout=float(branch_dropout),
            residual=bool(residual),
            use_dilation=bool(use_dilation),
        )

        # Concatenate [pose, hands, face] embeddings and project to fusion space.
        self.fuse = nn.Sequential(
            nn.Linear(int(embedding_dim) * 3, int(fusion_dim)),
            nn.GELU(),
            nn.LayerNorm(int(fusion_dim)),
        )

    def _slice(self, x: torch.Tensor, m: torch.Tensor | None, name: str):
        # Select joint slice for the requested modality.
        if name == "pose":
            s = self.schema.pose_slice
        elif name == "hands":
            s = slice(self.schema.left_hand_slice.start, self.schema.right_hand_slice.stop)
        elif name == "face":
            s = self.schema.face_slice
        else:
            raise ValueError(f"Unknown modality: {name}")
        # Apply same slice to data and optional joint-validity mask.
        xs = x[:, :, s, :]
        ms = None if m is None else m[:, :, s]
        return xs, ms

    def forward(self, motion_windows: torch.Tensor, joint_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = motion_windows.size(0)
        device = motion_windows.device

        # Disabled modalities contribute zeros; keeps concat layout fixed.
        out_pose = torch.zeros((b, self.pose_encoder.embedding_dim), device=device, dtype=motion_windows.dtype)
        out_hand = torch.zeros_like(out_pose)
        out_face = torch.zeros_like(out_pose)

        # Run only enabled modality branches.
        if "pose" in self.enabled_modalities:
            x, m = self._slice(motion_windows, joint_mask, "pose")
            out_pose = self.pose_encoder(x, joint_mask=m)
        if "hands" in self.enabled_modalities:
            x, m = self._slice(motion_windows, joint_mask, "hands")
            out_hand = self.hand_encoder(x, joint_mask=m)
        if "face" in self.enabled_modalities:
            x, m = self._slice(motion_windows, joint_mask, "face")
            out_face = self.face_encoder(x, joint_mask=m)

        # Fuse all three branch slots into one embedding.
        return self.fuse(torch.cat([out_pose, out_hand, out_face], dim=-1))
