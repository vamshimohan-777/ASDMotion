import copy

import torch
import torch.nn as nn

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
from src.models.video.motion.behavior_transformer import BehavioralTransformer
from src.models.video.motion.event_encoder import ResNetMicroKineticEventEncoder


class ASDPipeline(nn.Module):
    """
    Landmark motion pipeline:
    motion windows -> ResNet18 frame encoding -> micro-kinetic event detection
    -> event vectors + time series -> transformer ASD reasoning.
    """

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
    ):
        del alpha, cnn_backbone, nas_search_space, num_event_types
        super().__init__()
        self.schema = DEFAULT_SCHEMA
        self.theta_high = float(theta_high)
        self.theta_low = float(theta_low)
        self.aggregation_method = "attention"
        self._encoder_frozen = False
        self.k_max = int(max(1, K_max))

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
        self._build_from_architecture(self.architecture)

    def _build_from_architecture(self, arch):
        encoder_cfg = arch["encoder"]
        transformer_cfg = arch["transformer"]
        self.aggregation_method = arch.get("window", {}).get("aggregation", "attention")

        d_model = int(encoder_cfg["fusion_dim"])
        n_heads = int(transformer_cfg["heads"])
        if d_model % max(n_heads, 1) != 0:
            # keep transformer valid after NAS mutations
            candidates = [h for h in (2, 4, 8) if h > 0 and d_model % h == 0]
            n_heads = candidates[0] if candidates else 1

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
        self.behavior = BehavioralTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(transformer_cfg["layers"]),
            dim_ff=int(transformer_cfg["ff_dim"]),
            dropout=float(transformer_cfg["dropout"]),
        )

    def _rebuild_with_arch(self, arch):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.architecture = copy.deepcopy(arch)
        self._build_from_architecture(self.architecture)
        self.to(device)

    def freeze_motion_encoder(self):
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        self.motion_encoder.eval()
        self._encoder_frozen = True

    def unfreeze_upper_motion_layers(self, num_blocks=1):
        num_blocks = int(max(1, num_blocks))
        # New event encoder path
        if hasattr(self.motion_encoder, "temporal_blocks"):
            for p in self.motion_encoder.parameters():
                p.requires_grad = False
            blocks = list(self.motion_encoder.temporal_blocks)
            for block in blocks[-num_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            for name in ("temporal_out", "window_pool", "event_score_head", "frame_proj"):
                if hasattr(self.motion_encoder, name):
                    for p in getattr(self.motion_encoder, name).parameters():
                        p.requires_grad = True
            self._encoder_frozen = False
            return

        # Backward fallback for old branch encoder.
        if hasattr(self.motion_encoder, "pose_encoder"):
            branches = [
                self.motion_encoder.pose_encoder,
                self.motion_encoder.hand_encoder,
                self.motion_encoder.face_encoder,
            ]
            for branch in branches:
                for p in branch.parameters():
                    p.requires_grad = False
                block_list = list(branch.blocks)
                for block in block_list[-num_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in branch.out_proj.parameters():
                    p.requires_grad = True
        self._encoder_frozen = False

    # Backward-compatible hook name.
    def freeze_cnns(self, train_projection_heads=False):
        del train_projection_heads
        self.freeze_motion_encoder()

    def train(self, mode=True):
        super().train(mode)
        if self._encoder_frozen:
            self.motion_encoder.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def arch_parameters(self):
        return []

    def model_parameters(self):
        return self.trainable_parameters()

    @staticmethod
    def get_random_config(nas_search_space=None):
        del nas_search_space
        return None

    def get_current_config(self):
        return copy.deepcopy(self.architecture)

    def discretize_nas(self):
        return self.get_current_config()

    def apply_nas_architecture(self, nas_arch):
        if not nas_arch:
            return
        arch = copy.deepcopy(self.architecture)
        enc = nas_arch.get("encoder", {})
        tr = nas_arch.get("transformer", {})
        wn = nas_arch.get("window", {})

        if "branch_blocks" in enc:
            arch["encoder"]["branch_blocks"] = int(enc["branch_blocks"])
        if "branch_channels" in enc:
            arch["encoder"]["branch_channels"] = int(enc["branch_channels"])
        if "kernel_size" in enc:
            arch["encoder"]["kernel_size"] = int(enc["kernel_size"])
        if "use_dilation" in enc:
            arch["encoder"]["use_dilation"] = bool(enc["use_dilation"])
        if "residual" in enc:
            arch["encoder"]["residual"] = bool(enc["residual"])
        if "embedding_dim" in enc:
            # compatibility: map embedding_dim to fusion_dim
            arch["encoder"]["fusion_dim"] = int(enc["embedding_dim"])
        if "fusion_dim" in enc:
            arch["encoder"]["fusion_dim"] = int(enc["fusion_dim"])
        if "k_max" in enc:
            arch["encoder"]["k_max"] = int(enc["k_max"])

        if "layers" in tr:
            arch["transformer"]["layers"] = int(tr["layers"])
        if "heads" in tr:
            arch["transformer"]["heads"] = int(tr["heads"])
        if "ff_dim" in tr:
            arch["transformer"]["ff_dim"] = int(tr["ff_dim"])
        if "dropout" in tr:
            arch["transformer"]["dropout"] = float(tr["dropout"])

        if "aggregation" in wn:
            arch["window"]["aggregation"] = str(wn["aggregation"])

        self._rebuild_with_arch(arch)

    def forward(self, inputs):
        # motion_windows: [B, S, W, J, 9]
        motion = inputs["motion_windows"]
        if motion.dim() != 5:
            raise ValueError(
                f"motion_windows must be [B,S,W,J,9], got shape={tuple(motion.shape)}"
            )
        B, S, W, J, F = motion.shape
        if J != self.schema.total_joints:
            raise ValueError(f"Expected J={self.schema.total_joints}, got J={J}")
        if F != 9:
            raise ValueError(f"Expected feature dim 9, got {F}")

        joint_mask = inputs.get("joint_mask")
        if joint_mask is not None:
            if joint_mask.dim() != 4:
                raise ValueError(
                    f"joint_mask must be [B,S,W,J], got shape={tuple(joint_mask.shape)}"
                )
            window_valid = (joint_mask.float().sum(dim=(2, 3)) > 0)
        else:
            window_valid = torch.ones((B, S), dtype=torch.bool, device=motion.device)

        win_timestamps = inputs.get("window_timestamps")
        if win_timestamps is not None and win_timestamps.dim() != 3:
            raise ValueError(
                f"window_timestamps must be [B,S,W], got shape={tuple(win_timestamps.shape)}"
            )

        flat_motion = motion.reshape(B * S, W, J, F)
        flat_mask = None if joint_mask is None else joint_mask.reshape(B * S, W, J)
        flat_ts = None if win_timestamps is None else win_timestamps.reshape(B * S, W)

        enc_out = self.motion_encoder(
            flat_motion,
            joint_mask=flat_mask,
            timestamps=flat_ts,
            return_events=True,
        )
        flat_window_emb = enc_out["window_embedding"]  # [BS,D]
        flat_event_vec = enc_out["event_vectors"]  # [BS,K,D]
        flat_event_mask = enc_out["event_mask"]  # [BS,K]
        flat_event_times = enc_out["event_times"]  # [BS,K]
        flat_event_frame_idx = enc_out["event_frame_index"]  # [BS,K]
        flat_frame_event_scores = enc_out["frame_event_scores"]  # [BS,W]

        D = flat_window_emb.shape[-1]
        K = flat_event_vec.shape[1]
        window_embeddings = flat_window_emb.reshape(B, S, D)
        event_vectors = flat_event_vec.reshape(B, S, K, D)
        event_mask = flat_event_mask.reshape(B, S, K)
        event_times = flat_event_times.reshape(B, S, K)
        event_frame_idx = flat_event_frame_idx.reshape(B, S, K)
        frame_event_scores = flat_frame_event_scores.reshape(B, S, W)

        # Flatten event tokens over windows for global transformer reasoning.
        token_embeddings = event_vectors.reshape(B, S * K, D)
        token_mask = event_mask.reshape(B, S * K) & window_valid.unsqueeze(-1).expand(B, S, K).reshape(B, S * K)
        token_times = event_times.reshape(B, S * K)

        out = self.behavior(
            token_embeddings,
            window_mask=token_mask,
            event_times=token_times,
            aggregation=self.aggregation_method,
        )
        logit = out["logit"]
        prob = torch.sigmoid(logit)
        confidence = torch.max(prob, 1.0 - prob)
        decision = torch.where(
            prob >= self.theta_high,
            torch.ones_like(prob),
            torch.where(prob <= self.theta_low, torch.zeros_like(prob), -torch.ones_like(prob)),
        )

        # Aggregate token scores back to per-window scores for compatibility/reporting.
        token_scores = out.get("window_scores")
        window_scores = None
        if token_scores is not None:
            ts = token_scores.reshape(B, S, K)
            tm = event_mask & window_valid.unsqueeze(-1)
            ts = ts.masked_fill(~tm, -1.0)
            window_scores = ts.max(dim=-1).values

        attn_weights = out.get("attention_weights")
        window_attention = None
        if attn_weights is not None:
            aw = attn_weights.reshape(B, S, K)
            am = event_mask & window_valid.unsqueeze(-1)
            aw = aw * am.float()
            window_attention = aw.sum(dim=-1)

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
        }
