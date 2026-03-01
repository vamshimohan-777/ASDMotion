# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Temporal Transformer with sinusoidal positional encoding and learned time-gap embedding.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_positions: torch.Tensor) -> torch.Tensor:
        t = time_positions.unsqueeze(-1).float()
        freqs = t * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return emb


class TimeGapEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [B, K]
        dt = torch.log1p(delta_t.float()).unsqueeze(-1)
        return self.mlp(dt)


class HookableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        capture_list = getattr(self, "_attention_capture_list", None)
        need_weights = (capture_list is not None)

        out = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
            is_causal=is_causal,
        )

        if need_weights:
            x, weights = out
            capture_list.append(weights.detach())
        else:
            x = out[0]

        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        scalars_dim: int = 8,
        num_encoder_layers: int = 3,
        dim_ff: int = 2048,
        dropout: float = 0.3,
        activation: str = "gelu",
        num_event_types: int = 32,
        event_type_emb_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_gap_emb = TimeGapEmbedding(d_model)

        self.type_emb = nn.Embedding(num_event_types, event_type_emb_dim)

        fuse_in = d_model + scalars_dim + 1 + event_type_emb_dim
        self.token_fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        encoder_layer = HookableTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        self._capturing_attention = False
        self._captured_attention = []

        self._init_weights()

    def _init_weights(self):
        for module in [self.token_fuse, self.cls_head, self.conf_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.cls_head[-1].bias.fill_(-0.1)

    @contextmanager
    def capture_attention(self):
        self._capturing_attention = True
        self._captured_attention = []
        for layer in self.encoder.layers:
            layer._attention_capture_list = self._captured_attention
        try:
            yield
        finally:
            for layer in self.encoder.layers:
                if hasattr(layer, "_attention_capture_list"):
                    del layer._attention_capture_list
            self._capturing_attention = False

    def forward(self, x: dict) -> dict:
        tokens = x["tokens"]
        attn_mask = x["attn_mask"]
        time_positions = x["time_positions"]
        event_type_id = x["event_type_id"]
        token_conf = x["token_conf"]
        event_scalars = x["event_scalars"]
        delta_t = x.get("delta_t", None)

        B, K, D = tokens.shape
        if delta_t is None:
            delta_t = torch.zeros(B, K, device=tokens.device)

        conf = token_conf.unsqueeze(-1)
        type_emb = self.type_emb(event_type_id)
        fused = torch.cat([tokens, event_scalars, conf, type_emb], dim=-1)
        fused = self.token_fuse(fused)

        # Positional + time-gap embedding
        fused = fused + self.time_emb(time_positions) + self.time_gap_emb(delta_t)

        src_key_padding_mask = ~attn_mask
        X = self.encoder(fused, src_key_padding_mask=src_key_padding_mask)
        X = self.norm(X)

        valid = attn_mask.float().unsqueeze(-1)
        X = X * valid
        z = X.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        logit = self.cls_head(z).squeeze(-1)
        prob = torch.sigmoid(logit)
        conf_logit = self.conf_head(z).squeeze(-1)
        conf_score = torch.sigmoid(conf_logit)

        return {
            "z": z,
            "logit": logit,
            "prob": prob,
            "confidence_score": conf_score,
        }

