import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]


class BehavioralTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=4,
        n_layers=3,
        dim_ff=512,
        dropout=0.2,
    ):
        super().__init__()
        self.pos = PositionalEncoding(d_model=d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, window_embeddings, window_mask=None, event_times=None, aggregation="attention"):
        # window_embeddings: [B, T, D]
        h = self.pos(window_embeddings)
        if event_times is not None:
            # Add continuous-time embedding for event-aligned tokens.
            t = torch.log1p(event_times.float()).unsqueeze(-1)
            h = h + self.time_mlp(t)
        key_padding_mask = None
        if window_mask is not None:
            key_padding_mask = ~window_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        if window_mask is None:
            window_mask = torch.ones(
                (h.size(0), h.size(1)), device=h.device, dtype=torch.bool
            )
        valid = window_mask.float().unsqueeze(-1)

        if aggregation == "topk":
            scores = torch.sigmoid(self.head(h)).squeeze(-1)
            scores = scores.masked_fill(~window_mask.bool(), -1.0)
            k = max(1, min(h.shape[1] // 3, h.shape[1]))
            topk_vals, _ = torch.topk(scores, k=k, dim=1)
            pooled = topk_vals.mean(dim=1, keepdim=True)
            logit = torch.logit(torch.clamp(pooled.squeeze(-1), 1e-5, 1 - 1e-5))
            return {
                "logit": logit,
                "window_scores": scores,
                "attention_weights": None,
                "pooled": pooled.squeeze(-1),
            }

        # Attention pooling.
        a = self.attn_pool(h).squeeze(-1)
        a = a.masked_fill(~window_mask.bool(), float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        w = w * valid
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return {
            "logit": logit,
            "window_scores": torch.sigmoid(self.head(h)).squeeze(-1),
            "attention_weights": w.squeeze(-1),
            "pooled": pooled,
        }
