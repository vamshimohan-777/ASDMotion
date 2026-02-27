import random

import torch
import torch.nn as nn

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
from src.models.video.motion.blocks import MicroKineticBlock


class TemporalBranchEncoder(nn.Module):
    def __init__(
        self,
        n_joints,
        in_feat=9,
        channels=128,
        n_blocks=3,
        kernel_size=7,
        use_dilation=True,
        residual=True,
        dropout=0.1,
        out_dim=256,
    ):
        super().__init__()
        in_ch = int(n_joints * in_feat)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        blocks = []
        for i in range(int(n_blocks)):
            dilation = (2 ** i) if use_dilation else 1
            blocks.append(
                MicroKineticBlock(
                    channels=channels,
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                    residual=bool(residual),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: [B, W, J, F]
        B, W, J, F = x.shape
        h = x.reshape(B, W, J * F).transpose(1, 2)  # [B, J*F, W]
        h = self.input_proj(h)
        h = self.blocks(h)
        return self.out_proj(h)


class MultiBranchMotionEncoder(nn.Module):
    def __init__(
        self,
        schema=DEFAULT_SCHEMA,
        in_feat=9,
        branch_channels=128,
        branch_blocks=3,
        kernel_size=7,
        use_dilation=True,
        residual=True,
        branch_dropout=0.1,
        embedding_dim=256,
        fusion_dim=256,
        modality_dropout=0.0,
    ):
        super().__init__()
        self.schema = schema
        self.embedding_dim = int(embedding_dim)
        self.modality_dropout = float(modality_dropout)

        self.pose_encoder = TemporalBranchEncoder(
            n_joints=schema.pose_joints,
            in_feat=in_feat,
            channels=branch_channels,
            n_blocks=branch_blocks,
            kernel_size=kernel_size,
            use_dilation=use_dilation,
            residual=residual,
            dropout=branch_dropout,
            out_dim=embedding_dim,
        )
        self.hand_encoder = TemporalBranchEncoder(
            n_joints=schema.hand_joints * 2,
            in_feat=in_feat,
            channels=branch_channels,
            n_blocks=branch_blocks,
            kernel_size=kernel_size,
            use_dilation=use_dilation,
            residual=residual,
            dropout=branch_dropout,
            out_dim=embedding_dim,
        )
        self.face_encoder = TemporalBranchEncoder(
            n_joints=schema.face_joints,
            in_feat=in_feat,
            channels=branch_channels,
            n_blocks=branch_blocks,
            kernel_size=kernel_size,
            use_dilation=use_dilation,
            residual=residual,
            dropout=branch_dropout,
            out_dim=embedding_dim,
        )

        self.fuse = nn.Sequential(
            nn.Linear(embedding_dim * 3, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def _apply_modality_dropout(self, pose_z, hand_z, face_z):
        if not self.training or self.modality_dropout <= 0.0:
            return pose_z, hand_z, face_z

        # Four modes: pose, pose+hands, pose+face, all
        p = random.random()
        if p < self.modality_dropout:
            mode = random.choice(("pose", "pose_hands", "pose_face", "all"))
            if mode == "pose":
                hand_z = torch.zeros_like(hand_z)
                face_z = torch.zeros_like(face_z)
            elif mode == "pose_hands":
                face_z = torch.zeros_like(face_z)
            elif mode == "pose_face":
                hand_z = torch.zeros_like(hand_z)
        return pose_z, hand_z, face_z

    def forward(self, motion_windows, joint_mask=None):
        # motion_windows: [B, W, J, 9]
        pose_x = motion_windows[:, :, self.schema.pose_slice, :]
        hand_x = motion_windows[
            :,
            :,
            self.schema.left_hand_slice.start : self.schema.right_hand_slice.stop,
            :,
        ]
        face_x = motion_windows[:, :, self.schema.face_slice, :]

        pose_z = self.pose_encoder(pose_x)
        hand_z = self.hand_encoder(hand_x)
        face_z = self.face_encoder(face_x)
        pose_z, hand_z, face_z = self._apply_modality_dropout(pose_z, hand_z, face_z)
        fused = torch.cat([pose_z, hand_z, face_z], dim=-1)
        return self.fuse(fused)

