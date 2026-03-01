"""Training module `src/training/motion_ssl_landmark_losses.py` that optimizes model weights and output quality."""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn
# Import `torch.nn.functional as F` to support computations in this stage of output generation.
import torch.nn.functional as F


# Define a reusable pipeline function whose outputs feed later steps.
def temporal_contrastive_infonce(z1, z2, temperature=0.1):
    """
    Symmetric InfoNCE over batch positives:
    - positive pairs are aligned indices between z1 and z2
    - all other samples are negatives
    """
    # Set `t` for subsequent steps so gradient updates improve future predictions.
    t = max(float(temperature), 1e-6)
    # Compute `z1` as an intermediate representation used by later output layers.
    z1 = F.normalize(z1, dim=-1)
    # Compute `z2` as an intermediate representation used by later output layers.
    z2 = F.normalize(z2, dim=-1)
    # Store raw score tensor in `logits_12` before probability/decision conversion.
    logits_12 = (z1 @ z2.transpose(0, 1)) / t
    # Store raw score tensor in `logits_21` before probability/decision conversion.
    logits_21 = (z2 @ z1.transpose(0, 1)) / t
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = torch.arange(z1.shape[0], device=z1.device)
    # Update `loss` with a loss term that drives backpropagation and output improvement.
    loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
    # Return `loss` as this function's contribution to downstream output flow.
    return loss


# Define class `FutureMotionPredictor` to package related logic in the prediction pipeline.
class FutureMotionPredictor(nn.Module):
    """`FutureMotionPredictor` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, embedding_dim, hidden_dim=512, max_horizon=4):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so gradient updates improve future predictions.
        super().__init__()
        # Compute `self.horizon_emb` as an intermediate representation used by later output layers.
        self.horizon_emb = nn.Embedding(int(max_horizon) + 1, int(embedding_dim))
        # Set `self.net` for subsequent steps so gradient updates improve future predictions.
        self.net = nn.Sequential(
            nn.Linear(int(embedding_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, anchor_embedding, horizon):
        """Maps current inputs to this module's output tensor representation."""
        # Compute `h` as an intermediate representation used by later output layers.
        h = horizon.clamp(min=0, max=self.horizon_emb.num_embeddings - 1)
        # Compute `h_emb` as an intermediate representation used by later output layers.
        h_emb = self.horizon_emb(h)
        # Compute `x` as an intermediate representation used by later output layers.
        x = torch.cat([anchor_embedding, h_emb], dim=-1)
        # Return `self.net(x)` as this function's contribution to downstream output flow.
        return self.net(x)


# Define inference logic that produces the prediction returned to callers.
def future_motion_prediction_loss(predicted, target):
    """Builds inference outputs from inputs and returns values consumed by users or services."""
    # Return `F.smooth_l1_loss(predicted, target.detach())` as this function's contribution to downstream output flow.
    return F.smooth_l1_loss(predicted, target.detach())

