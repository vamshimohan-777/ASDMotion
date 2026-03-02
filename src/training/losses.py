# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

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
        # Invoke `super().__init__` to advance this processing stage.
        super().__init__()
        # Branch behavior based on the current runtime condition.
        if pos_weight is not None:
            # Invoke `self.register_buffer` to advance this processing stage.
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            # Compute `self._pos_weight` for the next processing step.
            self._pos_weight = None

        # Compute `self.label_smoothing` for the next processing step.
        self.label_smoothing = max(0.0, min(0.2, float(label_smoothing)))
        # Compute `self.logit_clip` for the next processing step.
        self.logit_clip = float(logit_clip)
        # Compute `self.brier_weight` for the next processing step.
        self.brier_weight = max(0.0, float(brier_weight))

    @staticmethod
    def compute_from_labels(labels, pos_weight_cap: float = 10.0) -> float:
        """Compute pos_weight from full dataset labels (call once)."""
        import numpy as np

        # Branch behavior based on the current runtime condition.
        if isinstance(labels, torch.Tensor):
            # Compute `labels` for the next processing step.
            labels = labels.numpy()
        # Compute `labels` for the next processing step.
        labels = np.asarray(labels, dtype=float)
        # Compute `n_pos` for the next processing step.
        n_pos = max(labels.sum(), 1.0)
        # Compute `n_neg` for the next processing step.
        n_neg = max(len(labels) - n_pos, 1.0)
        # Return the result expected by the caller.
        return min(n_neg / n_pos, float(pos_weight_cap))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B] raw model outputs (before sigmoid)
            target: [B] ground truth labels (0.0 or 1.0)
        """
        # Branch behavior based on the current runtime condition.
        if self.logit_clip > 0:
            # Compute `logits` for the next processing step.
            logits = logits.clamp(-self.logit_clip, self.logit_clip)

        # Compute `target_raw` for the next processing step.
        target_raw = target.float()
        # Compute `target` for the next processing step.
        target = target_raw
        # Branch behavior based on the current runtime condition.
        if self.label_smoothing > 0:
            # Binary label smoothing toward the uncertain target 0.5.
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute `pw` for the next processing step.
        pw = self._pos_weight
        # Branch behavior based on the current runtime condition.
        if pw is None:
            # Compute `pw` for the next processing step.
            pw = torch.tensor(1.0, device=logits.device)
        else:
            # Compute `pw` for the next processing step.
            pw = pw.to(logits.device)

        # Compute `bce` for the next processing step.
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        # Branch behavior based on the current runtime condition.
        if self.brier_weight <= 0:
            # Return the result expected by the caller.
            return bce

        # Brier regularization improves probability calibration and reduces overconfidence.
        probs = torch.sigmoid(logits)
        # Compute `brier` for the next processing step.
        brier = F.mse_loss(probs, target_raw)
        # Return the result expected by the caller.
        return bce + self.brier_weight * brier


def pairwise_auc_loss(logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Pairwise ranking surrogate for AUC.
    Penalizes negative samples ranked above positive samples.
    """
    # Compute `logits` for the next processing step.
    logits = logits.float().view(-1)
    # Compute `target` for the next processing step.
    target = target.float().view(-1)

    # Compute `pos` for the next processing step.
    pos = logits[target > 0.5]
    # Compute `neg` for the next processing step.
    neg = logits[target <= 0.5]
    # Branch behavior based on the current runtime condition.
    if pos.numel() == 0 or neg.numel() == 0:
        # Return the result expected by the caller.
        return logits.new_tensor(0.0)

    # Compute `tau` for the next processing step.
    tau = max(float(temperature), 1e-6)
    # Compute `diff` for the next processing step.
    diff = (pos.unsqueeze(1) - neg.unsqueeze(0)) / tau
    # Return the result expected by the caller.
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
    # Compute `logits` for the next processing step.
    logits = logits.float().view(-1)
    # Compute `target` for the next processing step.
    target = target.float().view(-1)
    # Compute `probs` for the next processing step.
    probs = torch.sigmoid(logits.clamp(-12.0, 12.0))

    # Compute `pos` for the next processing step.
    pos = probs[target > 0.5]
    # Compute `neg` for the next processing step.
    neg = probs[target <= 0.5]
    # Branch behavior based on the current runtime condition.
    if pos.numel() == 0 or neg.numel() == 0:
        # Return the result expected by the caller.
        return probs.new_tensor(0.0)

    # Compute `q` for the next processing step.
    q = min(max(float(target_spec), 0.0), 1.0)
    # Branch behavior based on the current runtime condition.
    if neg.numel() == 1:
        # Compute `thr` for the next processing step.
        thr = neg[0]
    else:
        # Compute `thr` for the next processing step.
        thr = torch.quantile(neg, q=q)
    # Branch behavior based on the current runtime condition.
    if detach_threshold:
        # Compute `thr` for the next processing step.
        thr = thr.detach()

    # Compute `m` for the next processing step.
    m = max(float(margin), 0.0)
    # Compute `pos_loss` for the next processing step.
    pos_loss = F.relu((thr + m) - pos).mean()
    # Compute `neg_loss` for the next processing step.
    neg_loss = F.relu(neg - thr).mean()
    # Return the result expected by the caller.
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
    # Return the result expected by the caller.
    return weight * controller.arch_entropy_loss()

