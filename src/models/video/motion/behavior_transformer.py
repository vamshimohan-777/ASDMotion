"""Temporal transformer for long-range behavioral reasoning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        # Precompute sinusoidal encodings once; reused on every forward pass.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Non-persistent: follows device moves, but not saved in checkpoints.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add position signal up to current sequence length S.
        return x + self.pe[:, : x.size(1), :]


class BehavioralTransformer(nn.Module):
    """
    Input:
      window_embeddings: [B, S, D]
      window_mask: [B, S] optional
    Output:
      dict with video embedding, window scores, optional attention weights.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Multi-head attention requires equal-sized head splits.
        if d_model % max(1, int(n_heads)) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = int(d_model)
        # Fixed positional encoding + learnable event-time projection.
        self.pos = PositionalEncoding(d_model=self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Standard transformer encoder stack over window sequence [B,S,D].
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(dim_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)

        # Learned attention-pooling logits for sequence aggregation.
        self.attn_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1),
        )
        # Per-window saliency/classification confidence head.
        self.window_score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model // 2, 1),
        )

    def forward(
        self,
        window_embeddings: torch.Tensor,
        window_mask: torch.Tensor | None = None,
        event_times: torch.Tensor | None = None,
        aggregation: str = "attention",
        topk_ratio: float = 0.33,
    ) -> dict:
        # Inject positional structure before temporal reasoning.
        h = self.pos(window_embeddings)
        if event_times is not None:
            # Log-scale event time to compress large indices/time gaps.
            t = torch.log1p(event_times.float()).unsqueeze(-1)
            h = h + self.time_mlp(t)

        if window_mask is None:
            # If mask is missing, treat all windows as valid.
            window_mask = torch.ones(
                (h.size(0), h.size(1)),
                dtype=torch.bool,
                device=h.device,
            )
        # PyTorch encoder uses True for padding positions.
        key_padding_mask = ~window_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        # Score each encoded window; invalid windows are marked out of range.
        raw_window_scores = torch.sigmoid(self.window_score_head(h)).squeeze(-1)
        raw_window_scores = raw_window_scores.masked_fill(~window_mask.bool(), -1.0)
        valid = window_mask.float().unsqueeze(-1)

        # Alternative aggregation: weighted average over top-k windows only.
        if aggregation == "topk":
            s = raw_window_scores.masked_fill(~window_mask.bool(), float("-inf"))
            k = max(1, min(h.size(1), int(round(h.size(1) * float(topk_ratio)))))
            topk_vals, topk_idx = torch.topk(s, k=k, dim=1)
            topk_w = torch.softmax(topk_vals, dim=1)
            topk_feat = h.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            pooled = (topk_feat * topk_w.unsqueeze(-1)).sum(dim=1)
            full_w = torch.zeros_like(s).masked_fill(~window_mask.bool(), 0.0)
            full_w.scatter_(1, topk_idx, topk_w)
            return {
                "video_embedding": pooled,
                "window_scores": raw_window_scores,
                "attention_weights": full_w,
                "encoded_windows": h,
            }

        # Default aggregation: learned global attention over valid windows.
        a = self.attn_pool(h).squeeze(-1)
        a = a.masked_fill(~window_mask.bool(), float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        w = w * valid
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1)
        return {
            "video_embedding": pooled,
            "window_scores": raw_window_scores,
            "attention_weights": w.squeeze(-1),
            "encoded_windows": h,
        }
