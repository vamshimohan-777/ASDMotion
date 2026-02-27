import torch
import torch.nn as nn

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA


class TemporalConvResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        padding = ((int(kernel_size) - 1) // 2) * int(dilation)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.drop(h)
        return self.act(x + h)


class JointWiseTemporalBranch(nn.Module):
    def __init__(
        self,
        in_features=9,
        hidden_dim=192,
        out_dim=256,
        kernel_sizes=(5, 9, 11),
        use_dilation=True,
        dropout=0.1,
        joint_pool="mean",
    ):
        super().__init__()
        self.joint_pool = str(joint_pool).lower().strip()
        if self.joint_pool not in {"mean", "attention"}:
            raise ValueError(f"Unsupported joint_pool='{joint_pool}', expected 'mean' or 'attention'.")

        self.input_proj = nn.Sequential(
            nn.Conv1d(int(in_features), int(hidden_dim), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalConvResidualBlock(
                    channels=int(hidden_dim),
                    kernel_size=int(k),
                    dilation=(2 ** idx) if use_dilation else 1,
                    dropout=float(dropout),
                )
                for idx, k in enumerate(kernel_sizes)
            ]
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.joint_attention = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.readout = nn.Sequential(
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.GELU(),
            nn.LayerNorm(int(out_dim)),
        )

    @staticmethod
    def _masked_mean(x, valid):
        mask = valid.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom

    def _joint_pooling(self, joint_features, joint_valid):
        if self.joint_pool == "mean":
            return self._masked_mean(joint_features, joint_valid)

        scores = self.joint_attention(joint_features).squeeze(-1)  # [B, J]
        scores = scores.masked_fill(~joint_valid, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = (joint_features * weights.unsqueeze(-1)).sum(dim=1)
        return pooled

    def forward(self, x, joint_valid=None):
        # x: [B, T, J, F]
        b, t, j, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * j, f, t).contiguous()  # [B*J, F, T]
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.time_pool(h).squeeze(-1)  # [B*J, H]
        h = h.reshape(b, j, -1)  # [B, J, H]

        if joint_valid is None:
            joint_valid = (x.abs().sum(dim=(1, 3)) > 1e-8)
        pooled = self._joint_pooling(h, joint_valid)
        return self.readout(pooled)


class MultiBranchMotionEncoderSSL(nn.Module):
    def __init__(
        self,
        schema=DEFAULT_SCHEMA,
        in_features=9,
        branch_hidden_dim=192,
        branch_out_dim=256,
        embedding_dim=256,
        kernel_sizes=(5, 9, 11),
        use_dilation=True,
        dropout=0.1,
        joint_pool="mean",
    ):
        super().__init__()
        self.schema = schema
        self.embedding_dim = int(embedding_dim)

        self.pose_branch = JointWiseTemporalBranch(
            in_features=in_features,
            hidden_dim=branch_hidden_dim,
            out_dim=branch_out_dim,
            kernel_sizes=kernel_sizes,
            use_dilation=use_dilation,
            dropout=dropout,
            joint_pool=joint_pool,
        )
        self.hand_branch = JointWiseTemporalBranch(
            in_features=in_features,
            hidden_dim=branch_hidden_dim,
            out_dim=branch_out_dim,
            kernel_sizes=kernel_sizes,
            use_dilation=use_dilation,
            dropout=dropout,
            joint_pool=joint_pool,
        )
        self.face_branch = JointWiseTemporalBranch(
            in_features=in_features,
            hidden_dim=branch_hidden_dim,
            out_dim=branch_out_dim,
            kernel_sizes=kernel_sizes,
            use_dilation=use_dilation,
            dropout=dropout,
            joint_pool=joint_pool,
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(branch_out_dim) * 3, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
        )

    def forward(self, x):
        # x: [B, T, J, 9]
        pose_x = x[:, :, self.schema.pose_slice, :]
        hand_x = x[
            :,
            :,
            self.schema.left_hand_slice.start : self.schema.right_hand_slice.stop,
            :,
        ]
        face_x = x[:, :, self.schema.face_slice, :]

        pose_valid = (pose_x.abs().sum(dim=(1, 3)) > 1e-8)
        hand_valid = (hand_x.abs().sum(dim=(1, 3)) > 1e-8)
        face_valid = (face_x.abs().sum(dim=(1, 3)) > 1e-8)

        pose_z = self.pose_branch(pose_x, joint_valid=pose_valid)
        hand_z = self.hand_branch(hand_x, joint_valid=hand_valid)
        face_z = self.face_branch(face_x, joint_valid=face_valid)
        fused = torch.cat([pose_z, hand_z, face_z], dim=-1)
        return self.fusion(fused)


def freeze_encoder(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    return module
