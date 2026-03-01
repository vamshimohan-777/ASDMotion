# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
ASD Pipeline model: face, pose, micro-motion encoders + temporal transformer + fusion.
"""

import torch
import torch.nn as nn

from src.models.video.cnn_encoders.face_encoder import FaceN
from src.models.video.cnn_encoders.motion_encoder import MotionN
from src.models.video.cnn_encoders.pose_encoder import PoseN
from src.models.nas_controller import MicroNASController

from src.models.image.perception import PerceptionCNN
from src.models.image.static_encoder import StaticEvidenceEncoder

from src.models.fusion import EvidenceFusion


class StreamGating(nn.Module):
    def __init__(self, temperature=5.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, face_feat, pose_feat, hand_feat, qualities):
        face_s = qualities["face_score"].unsqueeze(-1)
        pose_s = qualities["pose_score"].unsqueeze(-1)
        hand_s = qualities["hand_score"].unsqueeze(-1)

        scores = torch.cat([face_s, pose_s, hand_s], dim=-1)
        weights = torch.softmax(scores * self.temperature, dim=-1)

        wf = face_feat * weights[..., 0:1]
        wp = pose_feat * weights[..., 1:2]
        wh = hand_feat * weights[..., 2:3]

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

        self.face_cnn = FaceN(
            pretrained=True,
            backbone_name=cnn_backbone,
            use_fc_head=face_use_fc_head,
        )
        self.pose_cnn = PoseN(pretrained=True, backbone_name=cnn_backbone)
        self.motion_cnn = MotionN(pretrained=True, backbone_name=cnn_backbone)
        self.stream_gating = StreamGating()
        self._cnn_frozen = False
        self._train_projection_heads = False

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

        self.perception_cnn = PerceptionCNN(pretrained=True, backbone_name=cnn_backbone)
        self.static_encoder = StaticEvidenceEncoder(input_dim=256)
        self.image_head = nn.Linear(256, 1)

        self.fusion = EvidenceFusion(alpha=alpha, theta_high=theta_high, theta_low=theta_low)

    def freeze_cnns(self, train_projection_heads: bool = False):
        self._train_projection_heads = bool(train_projection_heads)
        for module in [self.face_cnn, self.pose_cnn, self.motion_cnn, self.perception_cnn]:
            for p in module.parameters():
                p.requires_grad = False
            if self._train_projection_heads and hasattr(module, "proj"):
                for p in module.proj.parameters():
                    p.requires_grad = True
            module.eval()
        self._cnn_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._cnn_frozen:
            for module in [self.face_cnn, self.pose_cnn, self.motion_cnn, self.perception_cnn]:
                module.eval()
                if mode and self._train_projection_heads and hasattr(module, "proj"):
                    module.proj.train(mode)
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def arch_parameters(self):
        return self.nas_controller.arch_parameters()

    def model_parameters(self):
        arch_set = set(id(p) for p in self.arch_parameters())
        return [p for p in self.parameters() if p.requires_grad and id(p) not in arch_set]

    def discretize_nas(self):
        return self.nas_controller.discretize()

    @staticmethod
    def get_random_config():
        return MicroNASController.get_random_config()

    def get_current_config(self):
        return self.nas_controller.get_current_config()

    def apply_nas_architecture(self, nas_arch):
        self.nas_controller.apply_config(nas_arch)

    def forward(self, inputs):
        B, T = inputs["face_crops"].shape[:2]
        device = inputs["face_crops"].device

        face_x = inputs["face_crops"].reshape(-1, 3, 224, 224)
        pose_x = inputs["pose_maps"].reshape(-1, 3, 224, 224)
        motion_x = inputs.get("motion_maps", inputs["pose_maps"]).reshape(-1, 3, 224, 224)

        face_feat = self.face_cnn(face_x).view(B, T, -1)
        pose_feat = self.pose_cnn(pose_x).view(B, T, -1)
        motion_feat = self.motion_cnn(motion_x).view(B, T, -1)

        qualities = inputs.get("qualities", {
            "face_score": torch.ones(B, T, device=device),
            "pose_score": torch.ones(B, T, device=device),
            "hand_score": torch.zeros(B, T, device=device),
        })
        video_tokens, gating_weights = self.stream_gating(
            face_feat, pose_feat, motion_feat, qualities
        )

        mask = inputs["mask"].bool()
        timestamps = inputs.get("timestamps", None)
        delta_t = inputs.get("delta_t", None)

        transformer_out = self.nas_controller(video_tokens, mask, timestamps=timestamps, delta_t=delta_t)
        logit_video = transformer_out["logit"]

        mid = T // 2
        img_feat = self.perception_cnn(inputs["face_crops"][:, mid])
        static_ev = self.static_encoder(img_feat)
        logit_image = self.image_head(static_ev).squeeze(-1)

        route_mask = inputs.get("route_mask", None)
        if route_mask is not None:
            route_mask = route_mask.to(device)
            if route_mask.dim() > 1:
                route_mask = route_mask.view(-1)
        fusion_out = self.fusion(logit_video, logit_image, route_mask=route_mask)
        fusion_out["logit_video"] = logit_video
        fusion_out["logit_image"] = logit_image
        fusion_out["gating_weights"] = gating_weights
        fusion_out["confidence_score"] = transformer_out.get("confidence_score", None)
        fusion_out["event_type_id"] = transformer_out.get("event_type_id", None)
        fusion_out["event_mask"] = transformer_out.get("event_mask", None)
        fusion_out["event_confidence"] = transformer_out.get("event_confidence", None)

        return fusion_out

