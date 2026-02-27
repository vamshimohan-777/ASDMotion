import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_contrastive_infonce(z1, z2, temperature=0.1):
    """
    Symmetric InfoNCE over batch positives:
    - positive pairs are aligned indices between z1 and z2
    - all other samples are negatives
    """
    t = max(float(temperature), 1e-6)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits_12 = (z1 @ z2.transpose(0, 1)) / t
    logits_21 = (z2 @ z1.transpose(0, 1)) / t
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
    return loss


class FutureMotionPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, max_horizon=4):
        super().__init__()
        self.horizon_emb = nn.Embedding(int(max_horizon) + 1, int(embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(int(embedding_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )

    def forward(self, anchor_embedding, horizon):
        h = horizon.clamp(min=0, max=self.horizon_emb.num_embeddings - 1)
        h_emb = self.horizon_emb(h)
        x = torch.cat([anchor_embedding, h_emb], dim=-1)
        return self.net(x)


def future_motion_prediction_loss(predicted, target):
    return F.smooth_l1_loss(predicted, target.detach())

