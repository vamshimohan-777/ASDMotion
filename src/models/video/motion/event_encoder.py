"""Model module `src/models/video/motion/event_encoder.py` that transforms inputs into features used for prediction."""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn
# Import `torch.nn.functional as F` to support computations in this stage of output generation.
import torch.nn.functional as F
# Import symbols from `torchvision` used in this stage's output computation path.
from torchvision import models

# Import symbols from `src.models.video.motion.blocks` used in this stage's output computation path.
from src.models.video.motion.blocks import MicroKineticBlock


# Define class `ResNetMicroKineticEventEncoder` to package related logic in the prediction pipeline.
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

    # Define a reusable pipeline function whose outputs feed later steps.
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
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Compute `self.embedding_dim` as an intermediate representation used by later output layers.
        self.embedding_dim = int(d_model)
        # Compute `self.k_max` as an intermediate representation used by later output layers.
        self.k_max = int(max(1, k_max))

        # Set `backbone` for subsequent steps so downstream prediction heads receive the right feature signal.
        backbone = models.resnet18(weights=None)
        # Set `self.resnet_backbone` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.resnet_backbone = nn.Sequential(*list(backbone.children())[:-1])  # [N,512,1,1]
        # Set `self.frame_proj` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.frame_proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Set `self.temporal_in` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.temporal_in = nn.Sequential(
            nn.Conv1d(d_model, temporal_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.GELU(),
        )
        # Set `blocks` for subsequent steps so downstream prediction heads receive the right feature signal.
        blocks = []
        # Iterate over `range(int(max(1, micro_blocks)))` so each item contributes to final outputs/metrics.
        for i in range(int(max(1, micro_blocks))):
            # Set `dilation` for subsequent steps so downstream prediction heads receive the right feature signal.
            dilation = (2 ** i) if use_dilation else 1
            # Call `blocks.append` and use its result in later steps so downstream prediction heads receive the right feature signal.
            blocks.append(
                MicroKineticBlock(
                    channels=temporal_channels,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                    residual=bool(residual),
                )
            )
        # Set `self.temporal_blocks` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.temporal_blocks = nn.Sequential(*blocks)
        # Set `self.temporal_out` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.temporal_out = nn.Sequential(
            nn.Conv1d(temporal_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Compute `self.event_score_head` as an intermediate representation used by later output layers.
        self.event_score_head = nn.Linear(d_model, 1)
        # Compute `self.window_pool` as an intermediate representation used by later output layers.
        self.window_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @staticmethod
    def _motion_to_rgb_image(motion_windows):
        # motion_windows: [B, W, J, 9]
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `b, w, j, f` for subsequent steps so downstream prediction heads receive the right feature signal.
        b, w, j, f = motion_windows.shape
        # Branch on `f != 9` to choose the correct output computation path.
        if f != 9:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Expected feature dim 9, got {f}")

        # Convert [J,9] to pseudo-image [3,J,3], then resize for ResNet.
        # Compute `x` as an intermediate representation used by later output layers.
        x = motion_windows.reshape(b * w, j, 3, 3).permute(0, 2, 1, 3).contiguous()  # [BW,3,J,3]
        # Compute `x` as an intermediate representation used by later output layers.
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        # Return `x` as this function's contribution to downstream output flow.
        return x

    # Define evaluation logic used to measure prediction quality.
    def _frame_valid_mask(self, joint_mask, b, w, device):
        """Computes evaluation outputs/metrics used to assess prediction quality."""
        # Branch on `joint_mask is None` to choose the correct output computation path.
        if joint_mask is None:
            # Return `torch.ones((b, w), dtype=torch.bool, device=device)` as this function's contribution to downstream output flow.
            return torch.ones((b, w), dtype=torch.bool, device=device)
        # frame valid if any joint is present.
        # Return `(joint_mask.float().sum(dim=-1) > 0)` as this function's contribution to downstream output flow.
        return (joint_mask.float().sum(dim=-1) > 0)

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, motion_windows, joint_mask=None, timestamps=None, return_events=False):
        """Maps current inputs to this module's output tensor representation."""
        # Set `b, w, _, _` for subsequent steps so downstream prediction heads receive the right feature signal.
        b, w, _, _ = motion_windows.shape
        # Set `device` to the execution device used for this computation path.
        device = motion_windows.device

        # Set `img` for subsequent steps so downstream prediction heads receive the right feature signal.
        img = self._motion_to_rgb_image(motion_windows)
        # Compute `feat_2d` as an intermediate representation used by later output layers.
        feat_2d = self.resnet_backbone(img).flatten(1)  # [BW,512]
        # Compute `frame_feat` as an intermediate representation used by later output layers.
        frame_feat = self.frame_proj(feat_2d).reshape(b, w, -1)  # [B,W,D]

        # Compute `x` as an intermediate representation used by later output layers.
        x = frame_feat.transpose(1, 2)  # [B,D,W]
        # Compute `x` as an intermediate representation used by later output layers.
        x = self.temporal_in(x)
        # Compute `x` as an intermediate representation used by later output layers.
        x = self.temporal_blocks(x)
        # Compute `x` as an intermediate representation used by later output layers.
        x = self.temporal_out(x)
        # Compute `x` as an intermediate representation used by later output layers.
        x = x.transpose(1, 2)  # [B,W,D]

        # Build `frame_valid` to gate invalid timesteps/joints from influencing outputs.
        frame_valid = self._frame_valid_mask(joint_mask, b, w, device=device)
        # Frame-level gate logits drive both hard event picking (top-k) and soft window pooling.
        # Store raw score tensor in `frame_logits` before probability/decision conversion.
        frame_logits = self.event_score_head(x).squeeze(-1)  # [B,W]
        # Store raw score tensor in `masked_frame_logits` before probability/decision conversion.
        masked_frame_logits = frame_logits.masked_fill(~frame_valid, float("-inf"))
        # Store raw score tensor in `frame_scores` before probability/decision conversion.
        frame_scores = torch.sigmoid(masked_frame_logits)

        # Set `k` for subsequent steps so downstream prediction heads receive the right feature signal.
        k = min(int(self.k_max), int(w))
        # Store raw score tensor in `topk_logits, topk_idx` before probability/decision conversion.
        topk_logits, topk_idx = torch.topk(masked_frame_logits, k=k, dim=1)
        # Preserve temporal order in selected events.
        # Set `order` for subsequent steps so downstream prediction heads receive the right feature signal.
        order = torch.argsort(topk_idx, dim=1)
        # Compute `topk_idx` as an intermediate representation used by later output layers.
        topk_idx = topk_idx.gather(1, order)
        # Store raw score tensor in `topk_scores` before probability/decision conversion.
        topk_scores = torch.sigmoid(topk_logits.gather(1, order))

        # Compute `idx_exp` as an intermediate representation used by later output layers.
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        # Set `event_vectors` for subsequent steps so downstream prediction heads receive the right feature signal.
        event_vectors = x.gather(1, idx_exp)  # [B,K,D]
        # Build `event_mask` to gate invalid timesteps/joints from influencing outputs.
        event_mask = frame_valid.gather(1, topk_idx)  # [B,K]

        # Branch on `timestamps is None` to choose the correct output computation path.
        if timestamps is None:
            # Set `event_times` for subsequent steps so downstream prediction heads receive the right feature signal.
            event_times = topk_idx.float()
        else:
            # Set `event_times` for subsequent steps so downstream prediction heads receive the right feature signal.
            event_times = timestamps.gather(1, topk_idx)

        # Differentiable soft aggregation path for scorer learning.
        # Set `wgt` for subsequent steps so downstream prediction heads receive the right feature signal.
        wgt = frame_scores * frame_valid.float()
        # Set `wgt` for subsequent steps so downstream prediction heads receive the right feature signal.
        wgt = wgt / wgt.sum(dim=1, keepdim=True).clamp(min=1e-6)
        # Compute `window_embedding` as an intermediate representation used by later output layers.
        window_embedding = (x * wgt.unsqueeze(-1)).sum(dim=1)
        # Compute `window_embedding` as an intermediate representation used by later output layers.
        window_embedding = self.window_pool(window_embedding)

        # Branch on `not return_events` to choose the correct output computation path.
        if not return_events:
            # Return `window_embedding` as this function's contribution to downstream output flow.
            return window_embedding

        # Return `{` as this function's contribution to downstream output flow.
        return {
            "window_embedding": window_embedding,
            "event_vectors": event_vectors,
            "event_mask": event_mask,
            "event_times": event_times,
            "event_frame_index": topk_idx,
            "event_scores": topk_scores,
            "frame_event_scores": frame_scores,
            "frame_event_logits": masked_frame_logits,
            "frame_valid_mask": frame_valid,
        }
