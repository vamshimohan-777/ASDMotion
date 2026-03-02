
"""
Temporal Transformer with sinusoidal positional encoding and learned time-gap embedding.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Embedding width must match the token width so both can be summed.
        self.d_model = d_model
        # Standard transformer frequency basis; lower frequencies capture coarse timing,
        # higher frequencies capture fine temporal ordering.
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Register as buffer so it moves with the module device/state but is not trainable.
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_positions: torch.Tensor) -> torch.Tensor:
        # time_positions: [B, K] token timestamps/indices per sequence.
        t = time_positions.unsqueeze(-1).float()
        # Broadcast per-token times against all sinusoidal frequencies.
        freqs = t * self.inv_freq.unsqueeze(0).unsqueeze(0)
        # Concatenate sin/cos channels to preserve relative position information.
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return emb


class TimeGapEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Learn nonlinear encoding of elapsed time between events.
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [B, K]
        # log1p compresses large gaps so rare long pauses do not dominate prediction.
        dt = torch.log1p(delta_t.float()).unsqueeze(-1)
        return self.mlp(dt)


class HookableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Preserve base behavior while exposing attention weights for diagnostics.
        x = src
        if self.norm_first:
            # Pre-norm path tends to stabilize training in deeper stacks.
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # Optional list used when capture_attention() is active.
        capture_list = getattr(self, "_attention_capture_list", None)
        need_weights = (capture_list is not None)

        # Self-attention is the core temporal reasoning step for event interactions.
        out = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
            is_causal=is_causal,
        )

        if need_weights:
            # Save detached maps to inspect what temporal evidence the model used.
            x, weights = out
            capture_list.append(weights.detach())
        else:
            x = out[0]

        # Dropout regularizes attention pathways for more robust generalization.
        return self.dropout1(x)

    def _ff_block(self, x):
        # Position-wise MLP refines token features after attention mixing.
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
        # Keep transform dimensions explicit for architecture search compatibility.
        self.d_model = d_model

        # Absolute temporal location signal.
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        # Relative/elapsed-time signal.
        self.time_gap_emb = TimeGapEmbedding(d_model)

        # Learned event-type semantics (blink, gaze shift, etc. categories).
        self.type_emb = nn.Embedding(num_event_types, event_type_emb_dim)

        # Token fusion mixes dynamic token features with scalar cues and confidence.
        fuse_in = d_model + scalars_dim + 1 + event_type_emb_dim
        self.token_fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Encoder layer supports optional attention capture during analysis.
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
        # Final normalization before temporal pooling/classification.
        self.norm = nn.LayerNorm(d_model)

        # Binary ASD/video-level classification head.
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Auxiliary confidence head estimates trust in current temporal evidence.
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
        # Xavier init keeps early activations and gradients in a stable range.
        for module in [self.token_fuse, self.cls_head, self.conf_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        with torch.no_grad():
            # Slight negative prior reduces false-positive spikes at initialization.
            self.cls_head[-1].bias.fill_(-0.1)

    @contextmanager
    def capture_attention(self):
        # Context manager for explainability/debugging without changing inference path.
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
        # Tokens are event-level descriptors from the micro-kinetic encoder.
        tokens = x["tokens"]
        # True where event tokens are valid; False for padding.
        attn_mask = x["attn_mask"]
        # Absolute positions/timestamps for positional encoding.
        time_positions = x["time_positions"]
        # Event type IDs used to inject categorical micro-event priors.
        event_type_id = x["event_type_id"]
        # Upstream token reliability estimate per event.
        token_conf = x["token_conf"]
        # Hand-crafted/derived scalar features per event.
        event_scalars = x["event_scalars"]
        # Optional elapsed-time gaps between events.
        delta_t = x.get("delta_t", None)

        B, K, D = tokens.shape
        if delta_t is None:
            # Default to zero gap when timing metadata is unavailable.
            delta_t = torch.zeros(B, K, device=tokens.device)

        # Treat confidence as an input feature so uncertain events are down-weighted implicitly.
        conf = token_conf.unsqueeze(-1)
        # Convert event types to learned dense vectors.
        type_emb = self.type_emb(event_type_id)
        # Fuse all event descriptors into transformer token width.
        fused = torch.cat([tokens, event_scalars, conf, type_emb], dim=-1)
        fused = self.token_fuse(fused)

        # Positional + time-gap embedding
        # Temporal signals are additive so content representation stays aligned.
        fused = fused + self.time_emb(time_positions) + self.time_gap_emb(delta_t)

        # Transformer expects True where padding should be ignored.
        src_key_padding_mask = ~attn_mask
        # Multi-layer temporal reasoning over event sequence.
        X = self.encoder(fused, src_key_padding_mask=src_key_padding_mask)
        X = self.norm(X)

        # Zero out padding tokens before global pooling to avoid biasing logits.
        valid = attn_mask.float().unsqueeze(-1)
        X = X * valid
        # Masked mean pooling creates a sequence-level embedding z.
        z = X.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        # Main ASD/micro-event decision logit.
        logit = self.cls_head(z).squeeze(-1)
        prob = torch.sigmoid(logit)
        # Confidence branch helps downstream fusion/routing trust this branch appropriately.
        conf_logit = self.conf_head(z).squeeze(-1)
        conf_score = torch.sigmoid(conf_logit)

        return {
            "z": z,
            "logit": logit,
            "prob": prob,
            "confidence_score": conf_score,
        }

