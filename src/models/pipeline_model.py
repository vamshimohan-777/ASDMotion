"""Model module `src/models/pipeline_model.py` that transforms inputs into features used for prediction."""

# Import `copy` to support computations in this stage of output generation.
import copy

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn

# Import symbols from `src.models.video.mediapipe_layer.landmark_schema` used in this stage's output computation path.
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
# Import symbols from `src.models.video.motion.behavior_transformer` used in this stage's output computation path.
from src.models.video.motion.behavior_transformer import BehavioralTransformer
# Import symbols from `src.models.video.motion.event_encoder` used in this stage's output computation path.
from src.models.video.motion.event_encoder import ResNetMicroKineticEventEncoder


# Define class `ASDPipeline` to package related logic in the prediction pipeline.
class ASDPipeline(nn.Module):
    """
    Landmark motion pipeline:
    motion windows -> ResNet18 frame encoding -> micro-kinetic event detection
    -> event vectors + time series -> transformer ASD reasoning.
    """

    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        alpha=1.0,
        K_max=32,
        d_model=256,
        dropout=0.2,
        theta_high=0.7,
        theta_low=0.3,
        cnn_backbone="unused",
        nas_search_space=None,
        num_event_types=0,
        train_event_scorer_when_frozen=True,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Execute this statement so downstream prediction heads receive the right feature signal.
        del alpha, cnn_backbone, nas_search_space, num_event_types
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Compute `self.schema` as an intermediate representation used by later output layers.
        self.schema = DEFAULT_SCHEMA
        # Compute `self.theta_high` as an intermediate representation used by later output layers.
        self.theta_high = float(theta_high)
        # Compute `self.theta_low` as an intermediate representation used by later output layers.
        self.theta_low = float(theta_low)
        # Compute `self.train_event_scorer_whe...` as an intermediate representation used by later output layers.
        self.train_event_scorer_when_frozen = bool(train_event_scorer_when_frozen)
        # Compute `self.aggregation_method` as an intermediate representation used by later output layers.
        self.aggregation_method = "attention"
        # Compute `self._encoder_frozen` as an intermediate representation used by later output layers.
        self._encoder_frozen = False
        # Compute `self.k_max` as an intermediate representation used by later output layers.
        self.k_max = int(max(1, K_max))

        # Compute `self.architecture` as an intermediate representation used by later output layers.
        self.architecture = {
            "encoder": {
                "branch_blocks": 3,
                "branch_channels": 256,
                "kernel_size": 7,
                "use_dilation": True,
                "residual": True,
                "fusion_dim": int(d_model),
                "k_max": int(self.k_max),
            },
            "transformer": {
                "layers": 3,
                "heads": 4,
                "ff_dim": 512,
                "dropout": float(dropout),
            },
            "window": {
                "aggregation": "attention",
            },
        }
        # Call `self._build_from_architecture` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self._build_from_architecture(self.architecture)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _build_from_architecture(self, arch):
        """Constructs components whose structure controls later training or inference outputs."""
        # Set `encoder_cfg` for subsequent steps so downstream prediction heads receive the right feature signal.
        encoder_cfg = arch["encoder"]
        # Set `transformer_cfg` for subsequent steps so downstream prediction heads receive the right feature signal.
        transformer_cfg = arch["transformer"]
        # Compute `self.aggregation_method` as an intermediate representation used by later output layers.
        self.aggregation_method = arch.get("window", {}).get("aggregation", "attention")

        # Set `d_model` for subsequent steps so downstream prediction heads receive the right feature signal.
        d_model = int(encoder_cfg["fusion_dim"])
        # Compute `n_heads` as an intermediate representation used by later output layers.
        n_heads = int(transformer_cfg["heads"])
        # Branch on `d_model % max(n_heads, 1) != 0` to choose the correct output computation path.
        if d_model % max(n_heads, 1) != 0:
            # keep transformer valid after NAS mutations
            # Call `in` and use its result in later steps so downstream prediction heads receive the right feature signal.
            candidates = [h for h in (2, 4, 8) if h > 0 and d_model % h == 0]
            # Compute `n_heads` as an intermediate representation used by later output layers.
            n_heads = candidates[0] if candidates else 1

        # Set `self.motion_encoder` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.motion_encoder = ResNetMicroKineticEventEncoder(
            d_model=d_model,
            temporal_channels=int(encoder_cfg["branch_channels"]),
            micro_blocks=int(encoder_cfg["branch_blocks"]),
            kernel_size=int(encoder_cfg["kernel_size"]),
            use_dilation=bool(encoder_cfg["use_dilation"]),
            residual=bool(encoder_cfg["residual"]),
            dropout=float(transformer_cfg.get("dropout", 0.2)),
            k_max=int(encoder_cfg.get("k_max", self.k_max)),
        )
        # Compute `self.behavior` as an intermediate representation used by later output layers.
        self.behavior = BehavioralTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(transformer_cfg["layers"]),
            dim_ff=int(transformer_cfg["ff_dim"]),
            dropout=float(transformer_cfg["dropout"]),
        )

    # Define a reusable pipeline function whose outputs feed later steps.
    def _rebuild_with_arch(self, arch):
        """Constructs components whose structure controls later training or inference outputs."""
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Set `device` to the execution device used for this computation path.
            device = next(self.parameters()).device
        # Handle exceptions and keep output behavior controlled under error conditions.
        except StopIteration:
            # Set `device` to the execution device used for this computation path.
            device = torch.device("cpu")
        # Compute `self.architecture` as an intermediate representation used by later output layers.
        self.architecture = copy.deepcopy(arch)
        # Call `self._build_from_architecture` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self._build_from_architecture(self.architecture)
        # Call `self.to` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self.to(device)

    # Define a reusable pipeline function whose outputs feed later steps.
    def freeze_motion_encoder(self, train_event_scorer=None):
        """Toggles parameter training state, which changes which parts of the model can influence outputs."""
        # Branch on `train_event_scorer is None` to choose the correct output computation path.
        if train_event_scorer is None:
            # Set `train_event_scorer` for subsequent steps so downstream prediction heads receive the right feature signal.
            train_event_scorer = self.train_event_scorer_when_frozen
        # Iterate over `self.motion_encoder.parameters()` so each item contributes to final outputs/metrics.
        for p in self.motion_encoder.parameters():
            # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
            p.requires_grad = False
        # Keep scorer trainable so frame-level auxiliary gate loss can still update salience head.
        # Branch on `bool(train_event_scorer) and hasattr(self.motion_...` to choose the correct output computation path.
        if bool(train_event_scorer) and hasattr(self.motion_encoder, "event_score_head"):
            # Iterate over `self.motion_encoder.event_score_head....` so each item contributes to final outputs/metrics.
            for p in self.motion_encoder.event_score_head.parameters():
                # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                p.requires_grad = True
        # Call `self.motion_encoder.eval` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self.motion_encoder.eval()
        # Compute `self._encoder_frozen` as an intermediate representation used by later output layers.
        self._encoder_frozen = True

    # Define a reusable pipeline function whose outputs feed later steps.
    def unfreeze_upper_motion_layers(self, num_blocks=1):
        """Toggles parameter training state, which changes which parts of the model can influence outputs."""
        # Set `num_blocks` for subsequent steps so downstream prediction heads receive the right feature signal.
        num_blocks = int(max(1, num_blocks))
        # New event encoder path
        # Branch on `hasattr(self.motion_encoder, "temporal_blocks")` to choose the correct output computation path.
        if hasattr(self.motion_encoder, "temporal_blocks"):
            # Iterate over `self.motion_encoder.parameters()` so each item contributes to final outputs/metrics.
            for p in self.motion_encoder.parameters():
                # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                p.requires_grad = False
            # Set `blocks` for subsequent steps so downstream prediction heads receive the right feature signal.
            blocks = list(self.motion_encoder.temporal_blocks)
            # Iterate over `blocks[-num_blocks:]` so each item contributes to final outputs/metrics.
            for block in blocks[-num_blocks:]:
                # Iterate over `block.parameters()` so each item contributes to final outputs/metrics.
                for p in block.parameters():
                    # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                    p.requires_grad = True
            # Iterate over `("temporal_out", "window_pool", "even...` so each item contributes to final outputs/metrics.
            for name in ("temporal_out", "window_pool", "event_score_head", "frame_proj"):
                # Branch on `hasattr(self.motion_encoder, name)` to choose the correct output computation path.
                if hasattr(self.motion_encoder, name):
                    # Iterate over `getattr(self.motion_encoder, name).pa...` so each item contributes to final outputs/metrics.
                    for p in getattr(self.motion_encoder, name).parameters():
                        # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                        p.requires_grad = True
            # Compute `self._encoder_frozen` as an intermediate representation used by later output layers.
            self._encoder_frozen = False
            # Return control/value to the caller for the next output-processing step.
            return

        # Backward fallback for old branch encoder.
        # Branch on `hasattr(self.motion_encoder, "pose_encoder")` to choose the correct output computation path.
        if hasattr(self.motion_encoder, "pose_encoder"):
            # Compute `branches` as an intermediate representation used by later output layers.
            branches = [
                self.motion_encoder.pose_encoder,
                self.motion_encoder.hand_encoder,
                self.motion_encoder.face_encoder,
            ]
            # Iterate over `branches` so each item contributes to final outputs/metrics.
            for branch in branches:
                # Iterate over `branch.parameters()` so each item contributes to final outputs/metrics.
                for p in branch.parameters():
                    # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                    p.requires_grad = False
                # Set `block_list` for subsequent steps so downstream prediction heads receive the right feature signal.
                block_list = list(branch.blocks)
                # Iterate over `block_list[-num_blocks:]` so each item contributes to final outputs/metrics.
                for block in block_list[-num_blocks:]:
                    # Iterate over `block.parameters()` so each item contributes to final outputs/metrics.
                    for p in block.parameters():
                        # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                        p.requires_grad = True
                # Iterate over `branch.out_proj.parameters()` so each item contributes to final outputs/metrics.
                for p in branch.out_proj.parameters():
                    # Set `p.requires_grad` for subsequent steps so downstream prediction heads receive the right feature signal.
                    p.requires_grad = True
        # Compute `self._encoder_frozen` as an intermediate representation used by later output layers.
        self._encoder_frozen = False

    # Backward-compatible hook name.
    # Define a reusable pipeline function whose outputs feed later steps.
    def freeze_cnns(self, train_projection_heads=False):
        """Toggles parameter training state, which changes which parts of the model can influence outputs."""
        # Execute this statement so downstream prediction heads receive the right feature signal.
        del train_projection_heads
        # Call `self.freeze_motion_encoder` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self.freeze_motion_encoder()

    # Define a training routine that updates parameters and changes future outputs.
    def train(self, mode=True):
        """Executes a training step/loop that updates parameters and directly changes model output behavior."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().train(mode)
        # Branch on `self._encoder_frozen` to choose the correct output computation path.
        if self._encoder_frozen:
            # Call `self.motion_encoder.eval` and use its result in later steps so downstream prediction heads receive the right feature signal.
            self.motion_encoder.eval()
        # Return `self` as this function's contribution to downstream output flow.
        return self

    # Define a training routine that updates parameters and changes future outputs.
    def trainable_parameters(self):
        """Executes a training step/loop that updates parameters and directly changes model output behavior."""
        # Return `[p for p in self.parameters() if p.requires_grad]` as this function's contribution to downstream output flow.
        return [p for p in self.parameters() if p.requires_grad]

    # Define a reusable pipeline function whose outputs feed later steps.
    def arch_parameters(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `[]` as this function's contribution to downstream output flow.
        return []

    # Define a reusable pipeline function whose outputs feed later steps.
    def model_parameters(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `self.trainable_parameters()` as this function's contribution to downstream output flow.
        return self.trainable_parameters()

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @staticmethod
    def get_random_config(nas_search_space=None):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Execute this statement so downstream prediction heads receive the right feature signal.
        del nas_search_space
        # Return `None` as this function's contribution to downstream output flow.
        return None

    # Define a reusable pipeline function whose outputs feed later steps.
    def get_current_config(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `copy.deepcopy(self.architecture)` as this function's contribution to downstream output flow.
        return copy.deepcopy(self.architecture)

    # Define a reusable pipeline function whose outputs feed later steps.
    def discretize_nas(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `self.get_current_config()` as this function's contribution to downstream output flow.
        return self.get_current_config()

    # Define a reusable pipeline function whose outputs feed later steps.
    def apply_nas_architecture(self, nas_arch):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `not nas_arch` to choose the correct output computation path.
        if not nas_arch:
            # Return control/value to the caller for the next output-processing step.
            return
        # Compute `arch` as an intermediate representation used by later output layers.
        arch = copy.deepcopy(self.architecture)
        # Set `enc` for subsequent steps so downstream prediction heads receive the right feature signal.
        enc = nas_arch.get("encoder", {})
        # Set `tr` for subsequent steps so downstream prediction heads receive the right feature signal.
        tr = nas_arch.get("transformer", {})
        # Set `wn` for subsequent steps so downstream prediction heads receive the right feature signal.
        wn = nas_arch.get("window", {})

        # Branch on `"branch_blocks" in enc` to choose the correct output computation path.
        if "branch_blocks" in enc:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["branch_blocks"] = int(enc["branch_blocks"])
        # Branch on `"branch_channels" in enc` to choose the correct output computation path.
        if "branch_channels" in enc:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["branch_channels"] = int(enc["branch_channels"])
        # Branch on `"kernel_size" in enc` to choose the correct output computation path.
        if "kernel_size" in enc:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["kernel_size"] = int(enc["kernel_size"])
        # Branch on `"use_dilation" in enc` to choose the correct output computation path.
        if "use_dilation" in enc:
            # Call `bool` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["use_dilation"] = bool(enc["use_dilation"])
        # Branch on `"residual" in enc` to choose the correct output computation path.
        if "residual" in enc:
            # Call `bool` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["residual"] = bool(enc["residual"])
        # Branch on `"embedding_dim" in enc` to choose the correct output computation path.
        if "embedding_dim" in enc:
            # compatibility: map embedding_dim to fusion_dim
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["fusion_dim"] = int(enc["embedding_dim"])
        # Branch on `"fusion_dim" in enc` to choose the correct output computation path.
        if "fusion_dim" in enc:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["fusion_dim"] = int(enc["fusion_dim"])
        # Branch on `"k_max" in enc` to choose the correct output computation path.
        if "k_max" in enc:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["encoder"]["k_max"] = int(enc["k_max"])

        # Branch on `"layers" in tr` to choose the correct output computation path.
        if "layers" in tr:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["transformer"]["layers"] = int(tr["layers"])
        # Branch on `"heads" in tr` to choose the correct output computation path.
        if "heads" in tr:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["transformer"]["heads"] = int(tr["heads"])
        # Branch on `"ff_dim" in tr` to choose the correct output computation path.
        if "ff_dim" in tr:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["transformer"]["ff_dim"] = int(tr["ff_dim"])
        # Branch on `"dropout" in tr` to choose the correct output computation path.
        if "dropout" in tr:
            # Call `float` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["transformer"]["dropout"] = float(tr["dropout"])

        # Branch on `"aggregation" in wn` to choose the correct output computation path.
        if "aggregation" in wn:
            # Call `str` and use its result in later steps so downstream prediction heads receive the right feature signal.
            arch["window"]["aggregation"] = str(wn["aggregation"])

        # Call `self._rebuild_with_arch` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self._rebuild_with_arch(arch)

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, inputs):
        # motion_windows: [B, S, W, J, 9]
        """Maps current inputs to this module's output tensor representation."""
        # Set `motion` for subsequent steps so downstream prediction heads receive the right feature signal.
        motion = inputs["motion_windows"]
        # Branch on `motion.dim() != 5` to choose the correct output computation path.
        if motion.dim() != 5:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(
                f"motion_windows must be [B,S,W,J,9], got shape={tuple(motion.shape)}"
            )
        # Set `B, S, W, J, F` for subsequent steps so downstream prediction heads receive the right feature signal.
        B, S, W, J, F = motion.shape
        # Branch on `J != self.schema.total_joints` to choose the correct output computation path.
        if J != self.schema.total_joints:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Expected J={self.schema.total_joints}, got J={J}")
        # Branch on `F != 9` to choose the correct output computation path.
        if F != 9:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Expected feature dim 9, got {F}")

        # Build `joint_mask` to gate invalid timesteps/joints from influencing outputs.
        joint_mask = inputs.get("joint_mask")
        # Branch on `joint_mask is not None` to choose the correct output computation path.
        if joint_mask is not None:
            # Branch on `joint_mask.dim() != 4` to choose the correct output computation path.
            if joint_mask.dim() != 4:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError(
                    f"joint_mask must be [B,S,W,J], got shape={tuple(joint_mask.shape)}"
                )
            # Build `window_valid` to gate invalid timesteps/joints from influencing outputs.
            window_valid = (joint_mask.float().sum(dim=(2, 3)) > 0)
        else:
            # Compute `window_valid` as an intermediate representation used by later output layers.
            window_valid = torch.ones((B, S), dtype=torch.bool, device=motion.device)

        # Set `win_timestamps` for subsequent steps so downstream prediction heads receive the right feature signal.
        win_timestamps = inputs.get("window_timestamps")
        # Branch on `win_timestamps is not None and win_timestamps.dim...` to choose the correct output computation path.
        if win_timestamps is not None and win_timestamps.dim() != 3:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(
                f"window_timestamps must be [B,S,W], got shape={tuple(win_timestamps.shape)}"
            )

        # Set `flat_motion` for subsequent steps so downstream prediction heads receive the right feature signal.
        flat_motion = motion.reshape(B * S, W, J, F)
        # Build `flat_mask` to gate invalid timesteps/joints from influencing outputs.
        flat_mask = None if joint_mask is None else joint_mask.reshape(B * S, W, J)
        # Set `flat_ts` for subsequent steps so downstream prediction heads receive the right feature signal.
        flat_ts = None if win_timestamps is None else win_timestamps.reshape(B * S, W)

        # Set `enc_out` for subsequent steps so downstream prediction heads receive the right feature signal.
        enc_out = self.motion_encoder(
            flat_motion,
            joint_mask=flat_mask,
            timestamps=flat_ts,
            # Return `_events=True,` as this function's contribution to downstream output flow.
            return_events=True,
        )
        # Compute `flat_window_emb` as an intermediate representation used by later output layers.
        flat_window_emb = enc_out["window_embedding"]  # [BS,D]
        # Set `flat_event_vec` for subsequent steps so downstream prediction heads receive the right feature signal.
        flat_event_vec = enc_out["event_vectors"]  # [BS,K,D]
        # Build `flat_event_mask` to gate invalid timesteps/joints from influencing outputs.
        flat_event_mask = enc_out["event_mask"]  # [BS,K]
        # Set `flat_event_times` for subsequent steps so downstream prediction heads receive the right feature signal.
        flat_event_times = enc_out["event_times"]  # [BS,K]
        # Compute `flat_event_frame_idx` as an intermediate representation used by later output layers.
        flat_event_frame_idx = enc_out["event_frame_index"]  # [BS,K]
        # Set `flat_frame_event_scores` for subsequent steps so downstream prediction heads receive the right feature signal.
        flat_frame_event_scores = enc_out["frame_event_scores"]  # [BS,W]
        # Store raw score tensor in `flat_frame_event_logits` before probability/decision conversion.
        flat_frame_event_logits = enc_out.get("frame_event_logits")
        # Build `flat_frame_valid` to gate invalid timesteps/joints from influencing outputs.
        flat_frame_valid = enc_out.get("frame_valid_mask")
        # Branch on `flat_frame_valid is None` to choose the correct output computation path.
        if flat_frame_valid is None:
            # Branch on `flat_mask is None` to choose the correct output computation path.
            if flat_mask is None:
                # Set `flat_frame_valid` for subsequent steps so downstream prediction heads receive the right feature signal.
                flat_frame_valid = torch.ones((B * S, W), dtype=torch.bool, device=motion.device)
            else:
                # Build `flat_frame_valid` to gate invalid timesteps/joints from influencing outputs.
                flat_frame_valid = (flat_mask.float().sum(dim=-1) > 0)

        # Set `D` for subsequent steps so downstream prediction heads receive the right feature signal.
        D = flat_window_emb.shape[-1]
        # Set `K` for subsequent steps so downstream prediction heads receive the right feature signal.
        K = flat_event_vec.shape[1]
        # Compute `window_embeddings` as an intermediate representation used by later output layers.
        window_embeddings = flat_window_emb.reshape(B, S, D)
        # Set `event_vectors` for subsequent steps so downstream prediction heads receive the right feature signal.
        event_vectors = flat_event_vec.reshape(B, S, K, D)
        # Build `event_mask` to gate invalid timesteps/joints from influencing outputs.
        event_mask = flat_event_mask.reshape(B, S, K)
        # Set `event_times` for subsequent steps so downstream prediction heads receive the right feature signal.
        event_times = flat_event_times.reshape(B, S, K)
        # Compute `event_frame_idx` as an intermediate representation used by later output layers.
        event_frame_idx = flat_event_frame_idx.reshape(B, S, K)
        # Set `frame_event_scores` for subsequent steps so downstream prediction heads receive the right feature signal.
        frame_event_scores = flat_frame_event_scores.reshape(B, S, W)
        # Build `frame_valid_mask` to gate invalid timesteps/joints from influencing outputs.
        frame_valid_mask = flat_frame_valid.reshape(B, S, W)
        # Store raw score tensor in `frame_event_logits` before probability/decision conversion.
        frame_event_logits = None
        # Branch on `flat_frame_event_logits is not None` to choose the correct output computation path.
        if flat_frame_event_logits is not None:
            # Store raw score tensor in `frame_event_logits` before probability/decision conversion.
            frame_event_logits = flat_frame_event_logits.reshape(B, S, W)

        # Flatten event tokens over windows for global transformer reasoning.
        # Compute `token_embeddings` as an intermediate representation used by later output layers.
        token_embeddings = event_vectors.reshape(B, S * K, D)
        # Build `token_mask` to gate invalid timesteps/joints from influencing outputs.
        token_mask = event_mask.reshape(B, S * K) & window_valid.unsqueeze(-1).expand(B, S, K).reshape(B, S * K)
        # Set `token_times` for subsequent steps so downstream prediction heads receive the right feature signal.
        token_times = event_times.reshape(B, S * K)

        # Set `out` for subsequent steps so downstream prediction heads receive the right feature signal.
        out = self.behavior(
            token_embeddings,
            window_mask=token_mask,
            event_times=token_times,
            aggregation=self.aggregation_method,
        )
        # Store raw score tensor in `logit` before probability/decision conversion.
        logit = out["logit"]
        # Store raw score tensor in `prob` before probability/decision conversion.
        prob = torch.sigmoid(logit)
        # Set `confidence` for subsequent steps so downstream prediction heads receive the right feature signal.
        confidence = torch.max(prob, 1.0 - prob)
        # Set `decision` for subsequent steps so downstream prediction heads receive the right feature signal.
        decision = torch.where(
            prob >= self.theta_high,
            torch.ones_like(prob),
            torch.where(prob <= self.theta_low, torch.zeros_like(prob), -torch.ones_like(prob)),
        )

        # Aggregate token scores back to per-window scores for compatibility/reporting.
        # Set `token_scores` for subsequent steps so downstream prediction heads receive the right feature signal.
        token_scores = out.get("window_scores")
        # Compute `window_scores` as an intermediate representation used by later output layers.
        window_scores = None
        # Branch on `token_scores is not None` to choose the correct output computation path.
        if token_scores is not None:
            # Set `ts` for subsequent steps so downstream prediction heads receive the right feature signal.
            ts = token_scores.reshape(B, S, K)
            # Build `tm` to gate invalid timesteps/joints from influencing outputs.
            tm = event_mask & window_valid.unsqueeze(-1)
            # Build `ts` to gate invalid timesteps/joints from influencing outputs.
            ts = ts.masked_fill(~tm, -1.0)
            # Compute `window_scores` as an intermediate representation used by later output layers.
            window_scores = ts.max(dim=-1).values

        # Compute `attn_weights` as an intermediate representation used by later output layers.
        attn_weights = out.get("attention_weights")
        # Compute `window_attention` as an intermediate representation used by later output layers.
        window_attention = None
        # Branch on `attn_weights is not None` to choose the correct output computation path.
        if attn_weights is not None:
            # Set `aw` for subsequent steps so downstream prediction heads receive the right feature signal.
            aw = attn_weights.reshape(B, S, K)
            # Build `am` to gate invalid timesteps/joints from influencing outputs.
            am = event_mask & window_valid.unsqueeze(-1)
            # Set `aw` for subsequent steps so downstream prediction heads receive the right feature signal.
            aw = aw * am.float()
            # Compute `window_attention` as an intermediate representation used by later output layers.
            window_attention = aw.sum(dim=-1)

        # Return `{` as this function's contribution to downstream output flow.
        return {
            "logit_final": logit,
            "prob_final": prob,
            "p_final": prob,
            "p_video": prob,
            "p_image": torch.zeros_like(prob),
            "alpha": torch.ones_like(prob),
            "confidence": confidence,
            "decision": decision,
            "window_scores": window_scores,
            "attention_weights": window_attention,
            "token_scores": token_scores,
            "token_attention_weights": attn_weights,
            "window_embeddings": window_embeddings,
            "event_vector_series": event_vectors,
            "event_time_series": event_times,
            "event_mask_series": event_mask,
            "event_frame_index_series": event_frame_idx,
            "frame_event_scores": frame_event_scores,
            "frame_event_logits": frame_event_logits,
            "frame_valid_mask": frame_valid_mask,
        }
