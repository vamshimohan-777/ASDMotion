import torch
import torch.nn as nn
from src.models.video.transformer_reasoning.event_transformer import TemporalTransformer


class ImageTemporalTransformer(nn.Module):
    """
    Lightweight wrapper that reuses TemporalTransformer for image/short-clip inputs.
    """
    def __init__(self, d_model=256, n_heads=4, num_layers=2, dim_ff=512, dropout=0.2):
        super().__init__()
        self.transformer = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            scalars_dim=0,
            num_event_types=2,
        )

    def forward(self, x):
        return self.transformer(x)
