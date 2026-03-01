"""Top-level multimodal ASD pipeline."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.models.video.motion.behavior_transformer import BehavioralTransformer
from src.models.video.motion.event_encoder import MicroKineticMotionEncoder
from src.models.video.motion.fusion import MotionRGBFusion
from src.models.video.motion.rgb_branch import ResNet18RGBBranch


class ASDPipeline(nn.Module):
    """
    Motion encoder (short-range) + RGB branch + fusion + temporal transformer + classifier.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        K_max: int = 8,
        d_model: int = 256,
        dropout: float = 0.2,
        theta_high: float = 0.7,
        theta_low: float = 0.3,
        cnn_backbone: str = "resnet18",
        nas_search_space=None,
        num_event_types: int = 0,
        train_event_scorer_when_frozen: bool = True,
        use_rgb: bool = True,
        rgb_pretrained: bool = True,
    ) -> None:
        del alpha, cnn_backbone, nas_search_space, num_event_types
        super().__init__()
        self.theta_high = float(theta_high)
        self.theta_low = float(theta_low)
        self.k_max = int(max(1, K_max))
        self.train_event_scorer_when_frozen = bool(train_event_scorer_when_frozen)
        self._encoder_frozen = False
        self._rgb_enabled = bool(use_rgb)

        self.architecture = {
            "encoder": {
                "branch_blocks": 3,
                "branch_channels": 128,
                "kernel_size": 7,
                "use_dilation": False,
                "residual": True,
                "embedding_dim": int(d_model),
            },
            "rgb": {
                "embedding_dim": int(d_model),
                "pretrained": bool(rgb_pretrained),
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
            "model": {
                "d_model": int(d_model),
                "dropout": float(dropout),
            },
        }
        self._build_from_architecture(self.architecture)

    def _build_from_architecture(self, arch: dict) -> None:
        enc = arch["encoder"]
        rgb = arch["rgb"]
        tr = arch["transformer"]
        mdl = arch["model"]

        d_model = int(mdl["d_model"])
        n_heads = int(tr["heads"])
        if d_model % max(1, n_heads) != 0:
            valid_heads = [h for h in (2, 4, 8) if d_model % h == 0]
            n_heads = valid_heads[0] if valid_heads else 1

        self.aggregation_method = arch.get("window", {}).get("aggregation", "attention")

        self.motion_encoder = MicroKineticMotionEncoder(
            in_features=9,
            temporal_channels=int(enc["branch_channels"]),
            num_blocks=int(enc["branch_blocks"]),
            kernel_size=int(enc["kernel_size"]),
            embedding_dim=int(enc["embedding_dim"]),
            dropout=float(tr.get("dropout", mdl.get("dropout", 0.2))),
            residual=bool(enc.get("residual", True)),
            use_dilation=bool(enc.get("use_dilation", False)),
            k_max=self.k_max,
        )
        self.rgb_branch = ResNet18RGBBranch(
            embedding_dim=int(rgb["embedding_dim"]),
            pretrained=bool(rgb.get("pretrained", True)),
            dropout=float(mdl.get("dropout", 0.2)),
        )
        self.fusion = MotionRGBFusion(
            motion_dim=int(enc["embedding_dim"]),
            rgb_dim=int(rgb["embedding_dim"]),
            out_dim=d_model,
            dropout=float(mdl.get("dropout", 0.2)),
        )
        self.behavior = BehavioralTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(tr["layers"]),
            dim_ff=int(tr["ff_dim"]),
            dropout=float(tr["dropout"]),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(float(mdl.get("dropout", 0.2))),
            nn.Linear(d_model // 2, 1),
        )

    def _rebuild_with_arch(self, arch: dict) -> None:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.architecture = copy.deepcopy(arch)
        self._build_from_architecture(self.architecture)
        self.to(device)

    def set_use_rgb(self, enabled: bool) -> None:
        self._rgb_enabled = bool(enabled)

    def freeze_rgb_backbone(self) -> None:
        self.rgb_branch.freeze_backbone()

    def unfreeze_rgb_backbone(self) -> None:
        self.rgb_branch.unfreeze_backbone()

    def freeze_motion_encoder(self, train_event_scorer=None) -> None:
        if train_event_scorer is None:
            train_event_scorer = self.train_event_scorer_when_frozen
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        if bool(train_event_scorer):
            for p in self.motion_encoder.frame_score_head.parameters():
                p.requires_grad = True
        self.motion_encoder.eval()
        self._encoder_frozen = True

    def unfreeze_upper_motion_layers(self, num_blocks: int = 1) -> None:
        num_blocks = int(max(1, num_blocks))
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        blocks = list(self.motion_encoder.temporal_blocks)
        for blk in blocks[-num_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.motion_encoder.proj.parameters():
            p.requires_grad = True
        for p in self.motion_encoder.frame_score_head.parameters():
            p.requires_grad = True
        self._encoder_frozen = False

    def freeze_cnns(self, train_projection_heads: bool = False) -> None:
        del train_projection_heads
        self.freeze_motion_encoder()

    def train(self, mode: bool = True):
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
            arch["encoder"]["embedding_dim"] = int(enc["embedding_dim"])

        if "layers" in tr:
            arch["transformer"]["layers"] = int(tr["layers"])
        if "heads" in tr:
            arch["transformer"]["heads"] = int(tr["heads"])
        if "ff_dim" in tr:
            arch["transformer"]["ff_dim"] = int(tr["ff_dim"])
        if "dropout" in tr:
            arch["transformer"]["dropout"] = float(tr["dropout"])
            arch["model"]["dropout"] = float(tr["dropout"])
        if "aggregation" in wn:
            arch["window"]["aggregation"] = str(wn["aggregation"])

        self._rebuild_with_arch(arch)

    def forward(self, inputs: dict) -> dict:
        motion = inputs["motion_windows"]  # [B,S,W,J,F]
        if motion.dim() != 5:
            raise ValueError(f"motion_windows must be [B,S,W,J,F], got {tuple(motion.shape)}")
        b, s, w, j, f = motion.shape
        if f != 9:
            raise ValueError(f"Expected motion feature dim F=9, got {f}")

        joint_mask = inputs.get("joint_mask")
        if joint_mask is not None and joint_mask.shape != (b, s, w, j):
            raise ValueError(f"joint_mask must be [B,S,W,J], got {tuple(joint_mask.shape)}")

        timestamps = inputs.get("window_timestamps")
        if timestamps is not None and timestamps.shape != (b, s, w):
            raise ValueError(f"window_timestamps must be [B,S,W], got {tuple(timestamps.shape)}")

        flat_motion = motion.reshape(b * s, w, j, f)
        flat_mask = None if joint_mask is None else joint_mask.reshape(b * s, w, j)
        enc_out = self.motion_encoder(flat_motion, joint_mask=flat_mask, return_events=True)
        motion_emb = enc_out["window_embedding"]  # [BS,Dm]

        rgb_emb = None
        if self._rgb_enabled:
            rgb_windows = inputs.get("rgb_windows")
            if rgb_windows is not None:
                if rgb_windows.dim() != 6:
                    raise ValueError(
                        f"rgb_windows must be [B,S,W,3,H,W], got {tuple(rgb_windows.shape)}"
                    )
                fb, fs, ft, fc, fh, fw = rgb_windows.shape
                if (fb, fs, ft) != (b, s, w):
                    raise ValueError(
                        f"rgb_windows first dims must match motion [B,S,W]={b,s,w}, got {tuple(rgb_windows.shape)}"
                    )
                if fc != 3:
                    raise ValueError(f"rgb_windows channel must be 3, got {fc}")
                flat_rgb = rgb_windows.reshape(b * s, w, 3, fh, fw)
                frame_mask = None if flat_mask is None else (flat_mask.sum(dim=-1) > 0.0)
                rgb_emb = self.rgb_branch(flat_rgb, frame_mask=frame_mask)

        fused_emb = self.fusion(motion_emb, rgb_emb)  # [BS,D]
        window_emb = fused_emb.reshape(b, s, -1)

        if joint_mask is None:
            window_valid = torch.ones((b, s), dtype=torch.bool, device=motion.device)
        else:
            window_valid = (joint_mask.float().sum(dim=(2, 3)) > 0.0)

        event_times = None
        if timestamps is not None:
            event_times = timestamps.float().mean(dim=-1)  # [B,S]

        tr_out = self.behavior(
            window_embeddings=window_emb,
            window_mask=window_valid,
            event_times=event_times,
            aggregation=self.aggregation_method,
        )
        video_emb = tr_out["video_embedding"]
        logit = self.classifier(video_emb).squeeze(-1)
        prob = torch.sigmoid(logit)
        confidence = torch.max(prob, 1.0 - prob)
        decision = torch.where(
            prob >= self.theta_high,
            torch.ones_like(prob),
            torch.where(prob <= self.theta_low, torch.zeros_like(prob), -torch.ones_like(prob)),
        )

        frame_event_scores = enc_out.get("frame_event_scores")
        frame_event_logits = enc_out.get("frame_event_logits")
        frame_valid_mask = enc_out.get("frame_valid_mask")
        if frame_event_scores is not None:
            frame_event_scores = frame_event_scores.reshape(b, s, w)
        if frame_event_logits is not None:
            frame_event_logits = frame_event_logits.reshape(b, s, w)
        if frame_valid_mask is not None:
            frame_valid_mask = frame_valid_mask.reshape(b, s, w)

        k = enc_out["event_vectors"].size(1)
        event_vectors = enc_out["event_vectors"].reshape(b, s, k, -1)
        event_times_series = enc_out["event_times"].reshape(b, s, k)
        event_mask_series = enc_out["event_mask"].reshape(b, s, k)
        event_frame_idx = enc_out["event_frame_index"].reshape(b, s, k)

        return {
            "logit_final": logit,
            "prob_final": prob,
            "p_final": prob,
            "p_video": prob,
            "p_image": torch.zeros_like(prob),
            "alpha": torch.ones_like(prob),
            "confidence": confidence,
            "decision": decision,
            "window_scores": tr_out.get("window_scores"),
            "attention_weights": tr_out.get("attention_weights"),
            "token_scores": tr_out.get("window_scores"),
            "token_attention_weights": tr_out.get("attention_weights"),
            "window_embeddings": window_emb,
            "event_vector_series": event_vectors,
            "event_time_series": event_times_series,
            "event_mask_series": event_mask_series,
            "event_frame_index_series": event_frame_idx,
            "frame_event_scores": frame_event_scores,
            "frame_event_logits": frame_event_logits,
            "frame_valid_mask": frame_valid_mask,
        }
