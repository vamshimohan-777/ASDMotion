"""Motion encoders that convert landmark windows into fused motion embeddings for classification."""

# Used to randomly drop modality branches during training, which regularizes the representation and can improve
# generalization of final predictions.
import random

# Tensor operations and neural-network layers used to transform raw motion windows into output embeddings.
import torch
import torch.nn as nn

# Landmark schema provides deterministic joint slices, so each branch always sees the intended body region.
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
# Temporal micro-blocks are the core feature extractor that determines how local motion patterns affect embeddings.
from src.models.video.motion.blocks import MicroKineticBlock


class TemporalBranchEncoder(nn.Module):
    """Encodes one modality stream (pose, hands, or face) from [B, W, J, F] into a fixed-length vector."""

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

        # Flattens joint and feature axes into channel count for Conv1d over time.
        in_ch = int(n_joints * in_feat)

        # Projects raw per-frame joint features into branch channels before temporal modeling.
        self.input_proj = nn.Sequential(
            # 1x1 temporal conv mixes joint-feature channels at each time step.
            nn.Conv1d(in_ch, channels, kernel_size=1, bias=False),
            # BatchNorm stabilizes training, which improves consistency of learned embeddings.
            nn.BatchNorm1d(channels),
            # Nonlinearity increases expressive power of the branch representation.
            nn.GELU(),
        )

        # Builds stacked temporal blocks; deeper stacks capture richer motion dynamics.
        blocks = []
        for i in range(int(n_blocks)):
            # Exponential dilation enlarges temporal receptive field, affecting sensitivity to long motions.
            dilation = (2 ** i) if use_dilation else 1
            blocks.append(
                MicroKineticBlock(
                    # Channel width controls branch capacity and therefore output detail.
                    channels=channels,
                    # Kernel size controls local temporal context per block.
                    kernel_size=int(kernel_size),
                    # Dilation controls spacing of sampled time steps inside convolution.
                    dilation=dilation,
                    # Dropout regularizes block activations to reduce overfitting.
                    dropout=float(dropout),
                    # Residual path preserves information flow and gradient stability.
                    residual=bool(residual),
                )
            )
        # Sequential container applies the temporal feature extractor stack in order.
        self.blocks = nn.Sequential(*blocks)

        # Compresses variable-length time dimension into a single embedding vector used downstream.
        self.out_proj = nn.Sequential(
            # Pools across time so output depends on salient motion over the full window.
            nn.AdaptiveAvgPool1d(1),
            # Removes singleton time axis to get [B, C] for dense projection.
            nn.Flatten(),
            # Projects branch channels to configured embedding size.
            nn.Linear(channels, out_dim),
            # Adds nonlinearity before normalization.
            nn.GELU(),
            # Normalizes embedding scale, improving fusion behavior across modalities.
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        """Maps one modality tensor from [B, W, J, F] to [B, out_dim]."""
        # Extract batch/window/joint/feature sizes used in deterministic reshape.
        B, W, J, F = x.shape
        # Reorders to [B, J*F, W] so Conv1d operates across time for each combined joint-feature channel.
        h = x.reshape(B, W, J * F).transpose(1, 2)
        # Initial channel projection conditions the signal for temporal micro-block processing.
        h = self.input_proj(h)
        # Temporal blocks learn motion patterns that carry predictive information.
        h = self.blocks(h)
        # Final projection returns a fixed-size modality embedding consumed by fusion/classification layers.
        return self.out_proj(h)


class MultiBranchMotionEncoder(nn.Module):
    """Encodes pose, hand, and face streams, then fuses them into one motion embedding."""

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

        # Schema decides exact landmark slicing; this directly controls what each branch receives.
        self.schema = schema
        # Stored for compatibility with callers that query encoder embedding width.
        self.embedding_dim = int(embedding_dim)
        # Probability of randomly masking modalities during training for robustness.
        self.modality_dropout = float(modality_dropout)

        # Pose branch learns body-motion features that influence final fused representation.
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
        # Hand branch sees both hands (left+right), capturing fine motor cues for output decisions.
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
        # Face branch captures facial-motion cues that can shift final classification confidence.
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

        # Learns cross-modality interactions from concatenated branch embeddings.
        self.fuse = nn.Sequential(
            # Reduces/reshapes concatenated [pose|hands|face] vector into final fusion space.
            nn.Linear(embedding_dim * 3, fusion_dim),
            # Nonlinearity enables non-additive modality interactions.
            nn.GELU(),
            # Normalization stabilizes fused feature scale for downstream heads.
            nn.LayerNorm(fusion_dim),
        )

    def _apply_modality_dropout(self, pose_z, hand_z, face_z):
        """Randomly zeroes selected modality embeddings during training to reduce over-reliance."""
        # In eval mode (or when disabled), keep full signal so predictions use all available modalities.
        if not self.training or self.modality_dropout <= 0.0:
            return pose_z, hand_z, face_z

        # Sample once per batch forward; stochastic masking changes training gradients and learned robustness.
        p = random.random()
        if p < self.modality_dropout:
            # Choose which modalities remain active; this shapes which branches receive gradient signal.
            mode = random.choice(("pose", "pose_hands", "pose_face", "all"))
            if mode == "pose":
                # Zeroing hands removes their contribution to fused output for this step.
                hand_z = torch.zeros_like(hand_z)
                # Zeroing face removes its contribution to fused output for this step.
                face_z = torch.zeros_like(face_z)
            elif mode == "pose_hands":
                # Keep pose+hands only; face contributes nothing this step.
                face_z = torch.zeros_like(face_z)
            elif mode == "pose_face":
                # Keep pose+face only; hands contribute nothing this step.
                hand_z = torch.zeros_like(hand_z)

        # Returns possibly-masked embeddings; these directly determine fused vector and downstream logits.
        return pose_z, hand_z, face_z

    def forward(self, motion_windows, joint_mask=None):
        """Converts full landmark windows [B, W, J, 9] into one fused embedding per sample."""
        # `joint_mask` is accepted for interface compatibility with other encoders.
        del joint_mask

        # Slice pose landmarks so pose branch output depends only on pose joints.
        pose_x = motion_windows[:, :, self.schema.pose_slice, :]
        # Slice contiguous hand region (left start to right stop) for the hand branch.
        hand_x = motion_windows[
            :,
            :,
            self.schema.left_hand_slice.start : self.schema.right_hand_slice.stop,
            :,
        ]
        # Slice face landmarks so face branch captures facial-motion dynamics only.
        face_x = motion_windows[:, :, self.schema.face_slice, :]

        # Encode pose stream into fixed-length embedding used in final fusion.
        pose_z = self.pose_encoder(pose_x)
        # Encode hand stream into fixed-length embedding used in final fusion.
        hand_z = self.hand_encoder(hand_x)
        # Encode face stream into fixed-length embedding used in final fusion.
        face_z = self.face_encoder(face_x)

        # Apply stochastic modality masking (training only), altering branch contributions to output.
        pose_z, hand_z, face_z = self._apply_modality_dropout(pose_z, hand_z, face_z)
        # Concatenate modality embeddings; ordering fixes which dimensions correspond to each modality.
        fused = torch.cat([pose_z, hand_z, face_z], dim=-1)
        # Project concatenated features into final fused embedding consumed by downstream classifier heads.
        return self.fuse(fused)
