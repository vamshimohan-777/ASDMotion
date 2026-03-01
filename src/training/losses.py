"""
Loss functions for ASD Pipeline training.

Includes:
  - WeightedBCELoss: handles class imbalance via pos_weight
  - NAS entropy regularization for architecture convergence
"""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn
# Import `torch.nn.functional as F` to support computations in this stage of output generation.
import torch.nn.functional as F


# Define class `WeightedBCELoss` to package related logic in the prediction pipeline.
class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with stable pos_weight for class imbalance.

    IMPORTANT: pos_weight must be computed ONCE from the full dataset,
    NOT per-batch. Per-batch computation causes wild loss fluctuations
    because each mini-batch has a different pos/neg ratio.
    """

    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        pos_weight: float = None,
        label_smoothing: float = 0.0,
        logit_clip: float = 12.0,
        brier_weight: float = 0.0,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so gradient updates improve future predictions.
        super().__init__()
        # Branch on `pos_weight is not None` to choose the correct output computation path.
        if pos_weight is not None:
            # Call `self.register_buffer` and use its result in later steps so gradient updates improve future predictions.
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            # Compute `self._pos_weight` as an intermediate representation used by later output layers.
            self._pos_weight = None

        # Compute `self.label_smoothing` as an intermediate representation used by later output layers.
        self.label_smoothing = max(0.0, min(0.2, float(label_smoothing)))
        # Store raw score tensor in `self.logit_clip` before probability/decision conversion.
        self.logit_clip = float(logit_clip)
        # Compute `self.brier_weight` as an intermediate representation used by later output layers.
        self.brier_weight = max(0.0, float(brier_weight))

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def compute_from_labels(labels, pos_weight_cap: float = 10.0) -> float:
        """Compute pos_weight from full dataset labels (call once)."""
        # Import `numpy as np` to support computations in this stage of output generation.
        import numpy as np

        # Branch on `isinstance(labels, torch.Tensor)` to choose the correct output computation path.
        if isinstance(labels, torch.Tensor):
            # Set `labels` for subsequent steps so gradient updates improve future predictions.
            labels = labels.numpy()
        # Set `labels` for subsequent steps so gradient updates improve future predictions.
        labels = np.asarray(labels, dtype=float)
        # Set `n_pos` for subsequent steps so gradient updates improve future predictions.
        n_pos = max(labels.sum(), 1.0)
        # Set `n_neg` for subsequent steps so gradient updates improve future predictions.
        n_neg = max(len(labels) - n_pos, 1.0)
        # Return `min(n_neg / n_pos, float(pos_weight_cap))` as this function's contribution to downstream output flow.
        return min(n_neg / n_pos, float(pos_weight_cap))

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B] raw model outputs (before sigmoid)
            target: [B] ground truth labels (0.0 or 1.0)
        """
        # Branch on `self.logit_clip > 0` to choose the correct output computation path.
        if self.logit_clip > 0:
            # Store raw score tensor in `logits` before probability/decision conversion.
            logits = logits.clamp(-self.logit_clip, self.logit_clip)

        # Set `target_raw` for subsequent steps so gradient updates improve future predictions.
        target_raw = target.float()
        # Set `target` for subsequent steps so gradient updates improve future predictions.
        target = target_raw
        # Branch on `self.label_smoothing > 0` to choose the correct output computation path.
        if self.label_smoothing > 0:
            # Binary label smoothing toward the uncertain target 0.5.
            # Set `target` for subsequent steps so gradient updates improve future predictions.
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Set `pw` for subsequent steps so gradient updates improve future predictions.
        pw = self._pos_weight
        # Branch on `pw is None` to choose the correct output computation path.
        if pw is None:
            # Store raw score tensor in `pw` before probability/decision conversion.
            pw = torch.tensor(1.0, device=logits.device)
        else:
            # Store raw score tensor in `pw` before probability/decision conversion.
            pw = pw.to(logits.device)

        # Store raw score tensor in `bce` before probability/decision conversion.
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        # Branch on `self.brier_weight <= 0` to choose the correct output computation path.
        if self.brier_weight <= 0:
            # Return `bce` as this function's contribution to downstream output flow.
            return bce

        # Brier regularization improves probability calibration and reduces overconfidence.
        # Store raw score tensor in `probs` before probability/decision conversion.
        probs = torch.sigmoid(logits)
        # Update `brier` with a loss term that drives backpropagation and output improvement.
        brier = F.mse_loss(probs, target_raw)
        # Return `bce + self.brier_weight * brier` as this function's contribution to downstream output flow.
        return bce + self.brier_weight * brier


# Define a loss computation that guides optimization toward better outputs.
def pairwise_auc_loss(logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Pairwise ranking surrogate for AUC.
    Penalizes negative samples ranked above positive samples.
    """
    # Store raw score tensor in `logits` before probability/decision conversion.
    logits = logits.float().view(-1)
    # Set `target` for subsequent steps so gradient updates improve future predictions.
    target = target.float().view(-1)

    # Store raw score tensor in `pos` before probability/decision conversion.
    pos = logits[target > 0.5]
    # Execute this statement so gradient updates improve future predictions.
    neg = logits[target <= 0.5]
    # Branch on `pos.numel() == 0 or neg.numel() == 0` to choose the correct output computation path.
    if pos.numel() == 0 or neg.numel() == 0:
        # Return `logits.new_tensor(0.0)` as this function's contribution to downstream output flow.
        return logits.new_tensor(0.0)

    # Set `tau` for subsequent steps so gradient updates improve future predictions.
    tau = max(float(temperature), 1e-6)
    # Set `diff` for subsequent steps so gradient updates improve future predictions.
    diff = (pos.unsqueeze(1) - neg.unsqueeze(0)) / tau
    # Return `F.softplus(-diff).mean()` as this function's contribution to downstream output flow.
    return F.softplus(-diff).mean()


# Define a loss computation that guides optimization toward better outputs.
def event_gate_bag_loss(
    frame_event_scores: torch.Tensor,
    target: torch.Tensor,
    frame_valid_mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Auxiliary loss for frame-level event gates using video labels.

    Treats a video as a bag of frame events:
      p(video has event) = 1 - prod(1 - p_frame) over valid frames/windows.
    Then applies BCE against the video label.
    """
    # Set `scores` for subsequent steps so gradient updates improve future predictions.
    scores = frame_event_scores.float()
    # Branch on `scores.dim() == 2` to choose the correct output computation path.
    if scores.dim() == 2:
        # Set `scores` for subsequent steps so gradient updates improve future predictions.
        scores = scores.unsqueeze(1)  # [B,1,W]
    # Branch on `scores.dim() != 3` to choose the correct output computation path.
    if scores.dim() != 3:
        # Raise explicit error to stop invalid state from producing misleading outputs.
        raise ValueError(f"frame_event_scores must be [B,S,W] or [B,W], got {tuple(scores.shape)}")

    # Branch on `frame_valid_mask is None` to choose the correct output computation path.
    if frame_valid_mask is None:
        # Set `valid` for subsequent steps so gradient updates improve future predictions.
        valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        # Build `valid` to gate invalid timesteps/joints from influencing outputs.
        valid = frame_valid_mask.bool()
        # Branch on `valid.dim() == 2` to choose the correct output computation path.
        if valid.dim() == 2:
            # Set `valid` for subsequent steps so gradient updates improve future predictions.
            valid = valid.unsqueeze(1)
        # Branch on `valid.shape != scores.shape` to choose the correct output computation path.
        if valid.shape != scores.shape:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(
                f"frame_valid_mask shape {tuple(valid.shape)} must match scores {tuple(scores.shape)}"
            )

    # Set `p` for subsequent steps so gradient updates improve future predictions.
    p = scores.clamp(min=float(eps), max=1.0 - float(eps))
    # Set `log_no_event_frame` for subsequent steps so gradient updates improve future predictions.
    log_no_event_frame = torch.log1p(-p)
    # Set `log_no_event_frame` for subsequent steps so gradient updates improve future predictions.
    log_no_event_frame = torch.where(valid, log_no_event_frame, torch.zeros_like(log_no_event_frame))
    # Compute `log_no_event_window` as an intermediate representation used by later output layers.
    log_no_event_window = log_no_event_frame.sum(dim=-1).clamp(min=-60.0, max=0.0)
    # Compute `no_event_window` as an intermediate representation used by later output layers.
    no_event_window = torch.exp(log_no_event_window)

    # Compute `valid_window` as an intermediate representation used by later output layers.
    valid_window = valid.any(dim=-1)
    # Compute `log_no_event_window` as an intermediate representation used by later output layers.
    log_no_event_window = torch.where(
        valid_window, torch.log(no_event_window.clamp(min=float(eps))), torch.zeros_like(no_event_window)
    )
    # Set `log_no_event_video` for subsequent steps so gradient updates improve future predictions.
    log_no_event_video = log_no_event_window.sum(dim=-1).clamp(min=-60.0, max=0.0)
    # Compute `video_event_prob` as confidence values used in final prediction decisions.
    video_event_prob = 1.0 - torch.exp(log_no_event_video)

    # Set `y` for subsequent steps so gradient updates improve future predictions.
    y = target.float().view(-1)
    # Return `F.binary_cross_entropy(video_event_prob.clamp(min=f...` as this function's contribution to downstream output flow.
    return F.binary_cross_entropy(video_event_prob.clamp(min=float(eps), max=1.0 - float(eps)), y)


# Define a reusable pipeline function whose outputs feed later steps.
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
    # Store raw score tensor in `logits` before probability/decision conversion.
    logits = logits.float().view(-1)
    # Set `target` for subsequent steps so gradient updates improve future predictions.
    target = target.float().view(-1)
    # Store raw score tensor in `probs` before probability/decision conversion.
    probs = torch.sigmoid(logits.clamp(-12.0, 12.0))

    # Set `pos` for subsequent steps so gradient updates improve future predictions.
    pos = probs[target > 0.5]
    # Execute this statement so gradient updates improve future predictions.
    neg = probs[target <= 0.5]
    # Branch on `pos.numel() == 0 or neg.numel() == 0` to choose the correct output computation path.
    if pos.numel() == 0 or neg.numel() == 0:
        # Return `probs.new_tensor(0.0)` as this function's contribution to downstream output flow.
        return probs.new_tensor(0.0)

    # Set `q` for subsequent steps so gradient updates improve future predictions.
    q = min(max(float(target_spec), 0.0), 1.0)
    # Branch on `neg.numel() == 1` to choose the correct output computation path.
    if neg.numel() == 1:
        # Compute `thr` as an intermediate representation used by later output layers.
        thr = neg[0]
    else:
        # Compute `thr` as an intermediate representation used by later output layers.
        thr = torch.quantile(neg, q=q)
    # Branch on `detach_threshold` to choose the correct output computation path.
    if detach_threshold:
        # Compute `thr` as an intermediate representation used by later output layers.
        thr = thr.detach()

    # Set `m` for subsequent steps so gradient updates improve future predictions.
    m = max(float(margin), 0.0)
    # Update `pos_loss` with a loss term that drives backpropagation and output improvement.
    pos_loss = F.relu((thr + m) - pos).mean()
    # Update `neg_loss` with a loss term that drives backpropagation and output improvement.
    neg_loss = F.relu(neg - thr).mean()
    # Return `pos_loss + 0.5 * neg_loss` as this function's contribution to downstream output flow.
    return pos_loss + 0.5 * neg_loss


# Define a reusable pipeline function whose outputs feed later steps.
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
    # Return `weight * controller.arch_entropy_loss()` as this function's contribution to downstream output flow.
    return weight * controller.arch_entropy_loss()
