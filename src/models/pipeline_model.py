
"""
ASD Pipeline model: face, pose, hand encoders + temporal transformer.
"""

import torch
import torch.nn as nn

from src.models.video.cnn_encoders.face_encoder import FaceN
from src.models.video.cnn_encoders.hand_encoder import HandN
from src.models.video.cnn_encoders.pose_encoder import PoseN
from src.models.nas_controller import MicroNASController


class StreamGating(nn.Module):
    def __init__(self, temperature=5.0):
        super().__init__()
        # High temperature makes the best-quality stream dominate more strongly.
        self.temperature = temperature

    def forward(self, face_feat, pose_feat, hand_feat, qualities):
        # Per-stream quality scores arrive as [B, T] and are expanded for concat.
        face_s = qualities["face_score"].unsqueeze(-1)
        pose_s = qualities["pose_score"].unsqueeze(-1)
        hand_s = qualities["hand_score"].unsqueeze(-1)

        # Softmax converts quality scores to normalized stream weights.
        scores = torch.cat([face_s, pose_s, hand_s], dim=-1)
        weights = torch.softmax(scores * self.temperature, dim=-1)

        # Scale each stream by reliability to suppress noisy evidence.
        wf = face_feat * weights[..., 0:1]
        wp = pose_feat * weights[..., 1:2]
        wh = hand_feat * weights[..., 2:3]

        # Concatenated gated feature is the video token used for event prediction.
        return torch.cat([wf, wp, wh], dim=-1), weights


class ASDPipeline(nn.Module):
    def __init__(self, alpha=0.6, K_max=32, d_model=256, dropout=0.3,
                 theta_high=0.7, theta_low=0.3,
                 cnn_backbone="resnet18",
                 face_use_fc_head=True,
                 num_event_types=12,
                 encoder_kernel_candidates=None,
                 transformer_heads_candidates=None,
                 transformer_layers_candidates=None,
                 transformer_ff_candidates=None):
        super().__init__()
        if theta_low > theta_high:
            raise ValueError("theta_low must be <= theta_high")
        self.theta_high = float(theta_high)
        self.theta_low = float(theta_low)

        # Dynamic branch backbones for face, pose, and hand cues.
        self.face_cnn = FaceN(
            pretrained=True,
            backbone_name=cnn_backbone,
            use_fc_head=face_use_fc_head,
        )
        self.pose_cnn = PoseN(pretrained=True, backbone_name=cnn_backbone)
        self.hand_cnn = HandN(pretrained=True, backbone_name=cnn_backbone)
        # Reliability-aware stream fusion before temporal/event modeling.
        self.stream_gating = StreamGating()
        # Track freeze state so train()/eval() toggles stay consistent.
        self._cnn_frozen = False
        self._train_projection_heads = False

        # NAS controller handles event extraction + temporal transformer reasoning.
        self.nas_controller = MicroNASController(
            d_in=768,
            d_model=d_model,
            K_max=K_max,
            num_event_types=num_event_types,
            num_scalars=8,
            dropout=dropout,
            encoder_kernel_candidates=encoder_kernel_candidates,
            transformer_heads_candidates=transformer_heads_candidates,
            transformer_layers_candidates=transformer_layers_candidates,
            transformer_ff_candidates=transformer_ff_candidates,
        )

    def freeze_cnns(self, train_projection_heads: bool = False):
        # Useful for staged training: freeze heavy CNNs and train temporal heads.
        self._train_projection_heads = bool(train_projection_heads)
        for module in [self.face_cnn, self.pose_cnn, self.hand_cnn]:
            for p in module.parameters():
                p.requires_grad = False
            # Optional: keep projection heads trainable for lightweight adaptation.
            if self._train_projection_heads and hasattr(module, "proj"):
                for p in module.proj.parameters():
                    p.requires_grad = True
            module.eval()
        self._cnn_frozen = True

    def train(self, mode: bool = True):
        # Enforce frozen-backbone eval mode even when parent enters train mode.
        super().train(mode)
        if self._cnn_frozen:
            for module in [self.face_cnn, self.pose_cnn, self.hand_cnn]:
                module.eval()
                if mode and self._train_projection_heads and hasattr(module, "proj"):
                    module.proj.train(mode)
        return self

    def trainable_parameters(self):
        # Returns only currently active parameters for optimizer setup.
        return [p for p in self.parameters() if p.requires_grad]

    def arch_parameters(self):
        # NAS logits are separated from standard model parameters.
        return self.nas_controller.arch_parameters()

    def model_parameters(self):
        # Exclude architecture logits to support dual-optimizer training loops.
        arch_set = set(id(p) for p in self.arch_parameters())
        return [p for p in self.parameters() if p.requires_grad and id(p) not in arch_set]

    def discretize_nas(self):
        # Lock architecture once search converges for stable inference.
        return self.nas_controller.discretize()

    @staticmethod
    def get_random_config():
        return MicroNASController.get_random_config()

    def get_current_config(self):
        return self.nas_controller.get_current_config()

    def apply_nas_architecture(self, nas_arch):
        # Apply a precomputed NAS architecture without re-running search.
        self.nas_controller.apply_config(nas_arch)

    def forward(self, inputs):
        # B=batch size, T=number of frames per clip.
        B, T = inputs["face_crops"].shape[:2]
        device = inputs["face_crops"].device

        # Flatten [B, T, C, H, W] -> [B*T, C, H, W] for frame-wise CNN encoding.
        face_x = inputs["face_crops"].reshape(-1, 3, 224, 224)
        pose_x = inputs["pose_maps"].reshape(-1, 3, 224, 224)
        # Primary key is hand_maps; motion_maps is accepted for backward compatibility.
        hand_x = inputs.get("hand_maps", inputs.get("motion_maps", inputs["pose_maps"])).reshape(-1, 3, 224, 224)

        # Restore [B, T, F] features after per-frame encoding.
        face_feat = self.face_cnn(face_x).view(B, T, -1)
        pose_feat = self.pose_cnn(pose_x).view(B, T, -1)
        hand_feat = self.hand_cnn(hand_x).view(B, T, -1)

        # Default qualities keep face/pose active and hand stream muted.
        qualities = inputs.get("qualities", {
            "face_score": torch.ones(B, T, device=device),
            "pose_score": torch.ones(B, T, device=device),
            "hand_score": torch.zeros(B, T, device=device),
        })
        # Quality-gated token sequence is the dynamic evidence for prediction.
        video_tokens, gating_weights = self.stream_gating(
            face_feat, pose_feat, hand_feat, qualities
        )

        # Valid-time mask supports variable-length clips with padding.
        mask = inputs["mask"].bool()
        # Optional timing metadata improves temporal reasoning fidelity.
        timestamps = inputs.get("timestamps", None)
        delta_t = inputs.get("delta_t", None)

        # NAS temporal branch outputs the primary video-level logit.
        transformer_out = self.nas_controller(video_tokens, mask, timestamps=timestamps, delta_t=delta_t)
        logit_video = transformer_out["logit"]
        p_video = torch.sigmoid(logit_video.clamp(-12.0, 12.0))
        confidence = (2.0 * (p_video - 0.5)).abs()

        decision = torch.full_like(p_video, -1, dtype=torch.long)
        decision = torch.where(p_video >= self.theta_high, torch.ones_like(decision), decision)
        decision = torch.where(p_video <= self.theta_low, torch.zeros_like(decision), decision)

        return {
            "logit_final": logit_video,
            "p_final": p_video,
            "p_video": p_video,
            "confidence": confidence,
            "decision": decision,
            "logit_video": logit_video,
            "gating_weights": gating_weights,
            "confidence_score": transformer_out.get("confidence_score", None),
            "event_type_id": transformer_out.get("event_type_id", None),
            "event_mask": transformer_out.get("event_mask", None),
            "event_confidence": transformer_out.get("event_confidence", None),
        }
