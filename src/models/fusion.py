# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Evidence Fusion - Combine Video + Image path evidence.

Fuses per-path probabilities with a learnable alpha weight and derives a
stable logit for loss computation. Supports abstention via confidence
thresholds and optional route masking for image-only inputs.
"""

import torch
import torch.nn as nn


class EvidenceFusion(nn.Module):
    """
    Fuses logits from the Video Path and Image Path.

    Training: returns logit_final + p_final + confidence (no decision thresholds).
    Eval:     applies abstain logic using theta_high/theta_low.
    """

    def __init__(self, alpha=0.6, theta_high=0.7, theta_low=0.3):
        super().__init__()
        # Learnable fusion weight (initialized near the given alpha)
        self.log_alpha = nn.Parameter(torch.tensor(alpha).logit())
        if theta_low > theta_high:
            raise ValueError("theta_low must be <= theta_high")
        self.theta_high = float(theta_high)
        self.theta_low = float(theta_low)

    @property
    def alpha(self):
        """Sigmoid-constrained to (0, 1)."""
        return torch.sigmoid(self.log_alpha)

    def forward(self, logit_video, logit_image, route_mask=None):
        """
        Args:
            logit_video: [B] raw logit from Video Path
            logit_image: [B] raw logit from Image Path
        route_mask:  [B] optional float/bool mask (1=use video path, 0=image only)

        Returns:
            dict with keys:
                logit_final - [B] fused logit (for BCEWithLogitsLoss)
                p_final     - [B] sigmoid(logit_final) - probability
                p_video     - [B] sigmoid(logit_video)
                p_image     - [B] sigmoid(logit_image)
                alpha       - scalar or [B], current fusion weight
                confidence  - [B] how far p_final is from 0.5 (0-1)
                decision    - [B] int tensor: 1=ASD, 0=Non-ASD, -1=Abstain
        """
        if logit_video is None:
            logit_video = torch.zeros_like(logit_image)
            route_mask = torch.zeros_like(logit_image)

        if route_mask is not None:
            route_mask = route_mask.float()
            if route_mask.dim() > 1:
                route_mask = route_mask.view(-1)
            # Hard gate by route: 1=video only, 0=image only
            a = route_mask
        else:
            a = self.alpha

        # Convert logits to probabilities
        p_video = torch.sigmoid(logit_video.clamp(-12.0, 12.0))
        p_image = torch.sigmoid(logit_image.clamp(-12.0, 12.0))

        # Fuse in probability space (blueprint-consistent)
        p_final = a * p_video + (1.0 - a) * p_image

        # Stable logit for BCEWithLogitsLoss
        eps = 1e-6
        p_final_clamped = p_final.clamp(eps, 1.0 - eps)
        logit_final = torch.log(p_final_clamped) - torch.log1p(-p_final_clamped)

        # Confidence = how far from the uncertain boundary (0.5)
        confidence = (2.0 * (p_final - 0.5)).abs()

        # Decision with abstention thresholds
        decision = torch.full_like(p_final, -1, dtype=torch.long)
        decision = torch.where(
            p_final >= self.theta_high,
            torch.ones_like(decision),
            decision,
        )
        decision = torch.where(
            p_final <= self.theta_low,
            torch.zeros_like(decision),
            decision,
        )

        return {
            "logit_final": logit_final,
            "p_final": p_final,
            "p_video": p_video,
            "p_image": p_image,
            "alpha": a,
            "confidence": confidence,
            "decision": decision,
        }

