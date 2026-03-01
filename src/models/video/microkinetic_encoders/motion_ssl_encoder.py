"""Model module `src/models/video/microkinetic_encoders/motion_ssl_encoder.py` that transforms inputs into features used for prediction."""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn

# Import symbols from `src.models.video.mediapipe_layer.landmark_schema` used in this stage's output computation path.
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA


# Define class `TemporalConvResidualBlock` to package related logic in the prediction pipeline.
class TemporalConvResidualBlock(nn.Module):
    """`TemporalConvResidualBlock` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, channels, kernel_size, dilation=1, dropout=0.1):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `padding` for subsequent steps so downstream prediction heads receive the right feature signal.
        padding = ((int(kernel_size) - 1) // 2) * int(dilation)
        # Set `self.conv1` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        # Set `self.bn1` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.bn1 = nn.BatchNorm1d(channels)
        # Set `self.act` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.act = nn.GELU()
        # Set `self.conv2` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        # Set `self.bn2` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.bn2 = nn.BatchNorm1d(channels)
        # Set `self.drop` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.drop = nn.Dropout(float(dropout))

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x):
        """Maps current inputs to this module's output tensor representation."""
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.conv1(x)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.bn1(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.act(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.conv2(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.bn2(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.drop(h)
        # Return `self.act(x + h)` as this function's contribution to downstream output flow.
        return self.act(x + h)


# Define class `JointWiseTemporalBranch` to package related logic in the prediction pipeline.
class JointWiseTemporalBranch(nn.Module):
    """`JointWiseTemporalBranch` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
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
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `self.out_dim` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.out_dim = int(out_dim)
        # Set `self.joint_pool` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.joint_pool = str(joint_pool).lower().strip()
        # Branch on `self.joint_pool not in {"mean", "attention"}` to choose the correct output computation path.
        if self.joint_pool not in {"mean", "attention"}:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Unsupported joint_pool='{joint_pool}', expected 'mean' or 'attention'.")

        # Set `self.input_proj` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.input_proj = nn.Sequential(
            nn.Conv1d(int(in_features), int(hidden_dim), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.GELU(),
        )
        # Set `self.blocks` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.blocks = nn.ModuleList(
            [
                TemporalConvResidualBlock(
                    channels=int(hidden_dim),
                    kernel_size=int(k),
                    dilation=(2 ** idx) if use_dilation else 1,
                    dropout=float(dropout),
                )
                # Iterate through items to accumulate output-relevant computations.
                for idx, k in enumerate(kernel_sizes)
            ]
        )
        # Set `self.time_pool` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        # Set `self.joint_attention` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.joint_attention = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        # Set `self.readout` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.readout = nn.Sequential(
            nn.Linear(int(hidden_dim), self.out_dim),
            nn.GELU(),
            nn.LayerNorm(self.out_dim),
        )

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @staticmethod
    def _masked_mean(x, valid):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = valid.float().unsqueeze(-1)
        # Build `denom` to gate invalid timesteps/joints from influencing outputs.
        denom = mask.sum(dim=1).clamp(min=1.0)
        # Return `(x * mask).sum(dim=1) / denom` as this function's contribution to downstream output flow.
        return (x * mask).sum(dim=1) / denom

    # Define a reusable pipeline function whose outputs feed later steps.
    def _joint_pooling(self, joint_features, joint_valid):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.joint_pool == "mean"` to choose the correct output computation path.
        if self.joint_pool == "mean":
            # Return `self._masked_mean(joint_features, joint_valid)` as this function's contribution to downstream output flow.
            return self._masked_mean(joint_features, joint_valid)

        # Set `scores` for subsequent steps so downstream prediction heads receive the right feature signal.
        scores = self.joint_attention(joint_features).squeeze(-1)  # [B, J]
        # Build `scores` to gate invalid timesteps/joints from influencing outputs.
        scores = scores.masked_fill(~joint_valid, -1e4)
        # Compute `weights` as an intermediate representation used by later output layers.
        weights = torch.softmax(scores, dim=1)
        # Set `pooled` for subsequent steps so downstream prediction heads receive the right feature signal.
        pooled = (joint_features * weights.unsqueeze(-1)).sum(dim=1)
        # Return `pooled` as this function's contribution to downstream output flow.
        return pooled

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x, joint_valid=None):
        # x: [B, T, J, F]
        """Maps current inputs to this module's output tensor representation."""
        # Set `b, t, j, f` for subsequent steps so downstream prediction heads receive the right feature signal.
        b, t, j, f = x.shape
        # Branch on `j == 0` to choose the correct output computation path.
        if j == 0:
            # Return `x.new_zeros((b, self.out_dim))` as this function's contribution to downstream output flow.
            return x.new_zeros((b, self.out_dim))

        # Compute `h` as an intermediate representation used by later output layers.
        h = x.permute(0, 2, 3, 1).reshape(b * j, f, t).contiguous()  # [B*J, F, T]
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.input_proj(h)
        # Iterate over `self.blocks` so each item contributes to final outputs/metrics.
        for block in self.blocks:
            # Compute `h` as an intermediate representation used by later output layers.
            h = block(h)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.time_pool(h).squeeze(-1)  # [B*J, H]
        # Compute `h` as an intermediate representation used by later output layers.
        h = h.reshape(b, j, -1)  # [B, J, H]

        # Branch on `joint_valid is None` to choose the correct output computation path.
        if joint_valid is None:
            # Set `joint_valid` for subsequent steps so downstream prediction heads receive the right feature signal.
            joint_valid = (x.abs().sum(dim=(1, 3)) > 1e-8)
        # Set `pooled` for subsequent steps so downstream prediction heads receive the right feature signal.
        pooled = self._joint_pooling(h, joint_valid)
        # Return `self.readout(pooled)` as this function's contribution to downstream output flow.
        return self.readout(pooled)


# Define class `MultiBranchMotionEncoderSSL` to package related logic in the prediction pipeline.
class MultiBranchMotionEncoderSSL(nn.Module):
    """`MultiBranchMotionEncoderSSL` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
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
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Compute `self.schema` as an intermediate representation used by later output layers.
        self.schema = schema
        # Compute `self.embedding_dim` as an intermediate representation used by later output layers.
        self.embedding_dim = int(embedding_dim)

        # Compute `self.pose_branch` as an intermediate representation used by later output layers.
        self.pose_branch = JointWiseTemporalBranch(
            in_features=in_features,
            hidden_dim=branch_hidden_dim,
            out_dim=branch_out_dim,
            kernel_sizes=kernel_sizes,
            use_dilation=use_dilation,
            dropout=dropout,
            joint_pool=joint_pool,
        )
        # Compute `self.hand_branch` as an intermediate representation used by later output layers.
        self.hand_branch = JointWiseTemporalBranch(
            in_features=in_features,
            hidden_dim=branch_hidden_dim,
            out_dim=branch_out_dim,
            kernel_sizes=kernel_sizes,
            use_dilation=use_dilation,
            dropout=dropout,
            joint_pool=joint_pool,
        )
        # Compute `self.face_branch` as an intermediate representation used by later output layers.
        self.face_branch = JointWiseTemporalBranch(
            in_features=in_features,
            hidden_dim=branch_hidden_dim,
            out_dim=branch_out_dim,
            kernel_sizes=kernel_sizes,
            use_dilation=use_dilation,
            dropout=dropout,
            joint_pool=joint_pool,
        )
        # Set `self.fusion` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.fusion = nn.Sequential(
            nn.Linear(int(branch_out_dim) * 3, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
        )

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x):
        # x: [B, T, J, 9]
        """Maps current inputs to this module's output tensor representation."""
        # Compute `pose_x` as an intermediate representation used by later output layers.
        pose_x = x[:, :, self.schema.pose_slice, :]
        # Compute `hand_x` as an intermediate representation used by later output layers.
        hand_x = x[
            :,
            :,
            self.schema.left_hand_slice.start : self.schema.right_hand_slice.stop,
            :,
        ]
        # Compute `face_x` as an intermediate representation used by later output layers.
        face_x = x[:, :, self.schema.face_slice, :]

        # Set `pose_valid` for subsequent steps so downstream prediction heads receive the right feature signal.
        pose_valid = (pose_x.abs().sum(dim=(1, 3)) > 1e-8)
        # Compute `hand_valid` as an intermediate representation used by later output layers.
        hand_valid = (hand_x.abs().sum(dim=(1, 3)) > 1e-8)
        # Set `face_valid` for subsequent steps so downstream prediction heads receive the right feature signal.
        face_valid = (face_x.abs().sum(dim=(1, 3)) > 1e-8)

        # Compute `pose_z` as an intermediate representation used by later output layers.
        pose_z = self.pose_branch(pose_x, joint_valid=pose_valid)
        # Compute `hand_z` as an intermediate representation used by later output layers.
        hand_z = self.hand_branch(hand_x, joint_valid=hand_valid)
        # Compute `face_z` as an intermediate representation used by later output layers.
        face_z = self.face_branch(face_x, joint_valid=face_valid)
        # Compute `fused` as an intermediate representation used by later output layers.
        fused = torch.cat([pose_z, hand_z, face_z], dim=-1)
        # Return `self.fusion(fused)` as this function's contribution to downstream output flow.
        return self.fusion(fused)


# Define a reusable pipeline function whose outputs feed later steps.
def freeze_encoder(module):
    """Toggles parameter training state, which changes which parts of the model can influence outputs."""
    # Iterate over `module.parameters()` so each item contributes to final outputs/metrics.
    for p in module.parameters():
        # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
        p.requires_grad = False
    # Call `module.eval` and use its result in later steps so downstream prediction heads receive the right feature signal.
    module.eval()
    # Return `module` as this function's contribution to downstream output flow.
    return module
