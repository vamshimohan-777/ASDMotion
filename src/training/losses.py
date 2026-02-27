"""
Loss functions for ASD Pipeline training.

Includes:
  - WeightedBCELoss: handles class imbalance via pos_weight
  - NAS entropy regularization for architecture convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with stable pos_weight for class imbalance.

    IMPORTANT: pos_weight must be computed ONCE from the full dataset,
    NOT per-batch. Per-batch computation causes wild loss fluctuations
    because each mini-batch has a different pos/neg ratio.
    """

    def __init__(
        self,
        pos_weight: float = None,
        label_smoothing: float = 0.0,
        logit_clip: float = 12.0,
        brier_weight: float = 0.0,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

        self.label_smoothing = max(0.0, min(0.2, float(label_smoothing)))
        self.logit_clip = float(logit_clip)
        self.brier_weight = max(0.0, float(brier_weight))

    @staticmethod
    def compute_from_labels(labels, pos_weight_cap: float = 10.0) -> float:
        """Compute pos_weight from full dataset labels (call once)."""
        import numpy as np

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        labels = np.asarray(labels, dtype=float)
        n_pos = max(labels.sum(), 1.0)
        n_neg = max(len(labels) - n_pos, 1.0)
        return min(n_neg / n_pos, float(pos_weight_cap))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B] raw model outputs (before sigmoid)
            target: [B] ground truth labels (0.0 or 1.0)
        """
        if self.logit_clip > 0:
            logits = logits.clamp(-self.logit_clip, self.logit_clip)

        target_raw = target.float()
        target = target_raw
        if self.label_smoothing > 0:
            # Binary label smoothing toward the uncertain target 0.5.
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pw = self._pos_weight
        if pw is None:
            pw = torch.tensor(1.0, device=logits.device)
        else:
            pw = pw.to(logits.device)

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        if self.brier_weight <= 0:
            return bce

        # Brier regularization improves probability calibration and reduces overconfidence.
        probs = torch.sigmoid(logits)
        brier = F.mse_loss(probs, target_raw)
        return bce + self.brier_weight * brier


def pairwise_auc_loss(logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Pairwise ranking surrogate for AUC.
    Penalizes negative samples ranked above positive samples.
    """
    logits = logits.float().view(-1)
    target = target.float().view(-1)

    pos = logits[target > 0.5]
    neg = logits[target <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)

    tau = max(float(temperature), 1e-6)
    diff = (pos.unsqueeze(1) - neg.unsqueeze(0)) / tau
    return F.softplus(-diff).mean()


def sens_at_spec_surrogate(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_spec: float = 0.95,
    margin: float = 0.02,
    detach_threshold: bool = True,
) -> torch.Tensor:
    """
    Differentiable surrogate that pushes positive probabilities above the
    high-specificity negative quantile and suppresses negatives above it.
    """
    logits = logits.float().view(-1)
    target = target.float().view(-1)
    probs = torch.sigmoid(logits.clamp(-12.0, 12.0))

    pos = probs[target > 0.5]
    neg = probs[target <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return probs.new_tensor(0.0)

    q = min(max(float(target_spec), 0.0), 1.0)
    if neg.numel() == 1:
        thr = neg[0]
    else:
        thr = torch.quantile(neg, q=q)
    if detach_threshold:
        thr = thr.detach()

    m = max(float(margin), 0.0)
    pos_loss = F.relu((thr + m) - pos).mean()
    neg_loss = F.relu(neg - thr).mean()
    return pos_loss + 0.5 * neg_loss


def nas_entropy_regularization(controller, weight: float = 0.01) -> torch.Tensor:
    """
    Entropy regularization on NAS architecture parameters.

    Lower entropy -> more decisive architecture selection -> better convergence.

    Args:
        controller: MicroNASController instance
        weight: scaling factor for the entropy term

    Returns:
        Scalar loss term to add to total loss
    """
    return weight * controller.arch_entropy_loss()
