"""
Temporal Transformer with sinusoidal positional encoding and learned time-gap embedding.
"""

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn
# Import symbols from `contextlib` used in this stage's output computation path.
from contextlib import contextmanager
# Import symbols from `src.models.video.microkinetic_encoders.event_types` used in this stage's output computation path.
from src.models.video.microkinetic_encoders.event_types import NUM_EVENT_TYPES


# Define class `SinusoidalTimeEmbedding` to package related logic in the prediction pipeline.
class SinusoidalTimeEmbedding(nn.Module):
    """`SinusoidalTimeEmbedding` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, d_model: int):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `self.d_model` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.d_model = d_model
        # Set `inv_freq` for subsequent steps so downstream prediction heads receive the right feature signal.
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Call `self.register_buffer` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self.register_buffer("inv_freq", inv_freq)

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, time_positions: torch.Tensor) -> torch.Tensor:
        """Maps current inputs to this module's output tensor representation."""
        # Set `t` for subsequent steps so downstream prediction heads receive the right feature signal.
        t = time_positions.unsqueeze(-1).float()
        # Set `freqs` for subsequent steps so downstream prediction heads receive the right feature signal.
        freqs = t * self.inv_freq.unsqueeze(0).unsqueeze(0)
        # Compute `emb` as an intermediate representation used by later output layers.
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        # Return `emb` as this function's contribution to downstream output flow.
        return emb


# Define class `TimeGapEmbedding` to package related logic in the prediction pipeline.
class TimeGapEmbedding(nn.Module):
    """`TimeGapEmbedding` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, d_model: int):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `self.mlp` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [B, K]
        """Maps current inputs to this module's output tensor representation."""
        # Set `dt` for subsequent steps so downstream prediction heads receive the right feature signal.
        dt = torch.log1p(delta_t.float()).unsqueeze(-1)
        # Return `self.mlp(dt)` as this function's contribution to downstream output flow.
        return self.mlp(dt)


# Define class `HookableTransformerEncoderLayer` to package related logic in the prediction pipeline.
class HookableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """`HookableTransformerEncoderLayer` groups related operations that shape intermediate and final outputs."""
    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """Maps current inputs to this module's output tensor representation."""
        # Compute `x` as an intermediate representation used by later output layers.
        x = src
        # Branch on `self.norm_first` to choose the correct output computation path.
        if self.norm_first:
            # Build `x` to gate invalid timesteps/joints from influencing outputs.
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            # Compute `x` as an intermediate representation used by later output layers.
            x = x + self._ff_block(self.norm2(x))
        else:
            # Build `x` to gate invalid timesteps/joints from influencing outputs.
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            # Compute `x` as an intermediate representation used by later output layers.
            x = self.norm2(x + self._ff_block(x))
        # Return `x` as this function's contribution to downstream output flow.
        return x

    # Define a reusable pipeline function whose outputs feed later steps.
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `capture_list` for subsequent steps so downstream prediction heads receive the right feature signal.
        capture_list = getattr(self, "_attention_capture_list", None)
        # Compute `need_weights` as an intermediate representation used by later output layers.
        need_weights = (capture_list is not None)

        # Set `out` for subsequent steps so downstream prediction heads receive the right feature signal.
        out = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
            is_causal=is_causal,
        )

        # Branch on `need_weights` to choose the correct output computation path.
        if need_weights:
            # Compute `x, weights` as an intermediate representation used by later output layers.
            x, weights = out
            # Call `capture_list.append` and use its result in later steps so downstream prediction heads receive the right feature signal.
            capture_list.append(weights.detach())
        else:
            # Compute `x` as an intermediate representation used by later output layers.
            x = out[0]

        # Return `self.dropout1(x)` as this function's contribution to downstream output flow.
        return self.dropout1(x)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _ff_block(self, x):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `x` as an intermediate representation used by later output layers.
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Return `self.dropout2(x)` as this function's contribution to downstream output flow.
        return self.dropout2(x)


# Define class `TemporalTransformer` to package related logic in the prediction pipeline.
class TemporalTransformer(nn.Module):
    """`TemporalTransformer` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        scalars_dim: int = 8,
        num_encoder_layers: int = 3,
        dim_ff: int = 2048,
        dropout: float = 0.3,
        activation: str = "gelu",
        num_event_types: int = NUM_EVENT_TYPES,
        event_type_emb_dim: int = 32,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `self.d_model` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.d_model = d_model

        # Compute `self.time_emb` as an intermediate representation used by later output layers.
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        # Compute `self.time_gap_emb` as an intermediate representation used by later output layers.
        self.time_gap_emb = TimeGapEmbedding(d_model)

        # Compute `self.type_emb` as an intermediate representation used by later output layers.
        self.type_emb = nn.Embedding(num_event_types, event_type_emb_dim)

        # Set `fuse_in` for subsequent steps so downstream prediction heads receive the right feature signal.
        fuse_in = d_model + scalars_dim + 1 + event_type_emb_dim
        # Set `self.token_fuse` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.token_fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Set `encoder_layer` for subsequent steps so downstream prediction heads receive the right feature signal.
        encoder_layer = HookableTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        # Set `self.encoder` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        # Set `self.norm` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.norm = nn.LayerNorm(d_model)

        # Compute `self.cls_head` as an intermediate representation used by later output layers.
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Compute `self.conf_head` as an intermediate representation used by later output layers.
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        # Set `self._capturing_attention` for subsequent steps so downstream prediction heads receive the right feature signal.
        self._capturing_attention = False
        # Set `self._captured_attention` for subsequent steps so downstream prediction heads receive the right feature signal.
        self._captured_attention = []

        # Call `self._init_weights` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self._init_weights()

    # Define a reusable pipeline function whose outputs feed later steps.
    def _init_weights(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Iterate over `[self.token_fuse, self.cls_head, self...` so each item contributes to final outputs/metrics.
        for module in [self.token_fuse, self.cls_head, self.conf_head]:
            # Iterate over `module.modules()` so each item contributes to final outputs/metrics.
            for m in module.modules():
                # Branch on `isinstance(m, nn.Linear)` to choose the correct output computation path.
                if isinstance(m, nn.Linear):
                    # Call `nn.init.xavier_uniform_` and use its result in later steps so downstream prediction heads receive the right feature signal.
                    nn.init.xavier_uniform_(m.weight)
                    # Branch on `m.bias is not None` to choose the correct output computation path.
                    if m.bias is not None:
                        # Call `nn.init.zeros_` and use its result in later steps so downstream prediction heads receive the right feature signal.
                        nn.init.zeros_(m.bias)
        # Use a managed context to safely handle resources used during output computation.
        with torch.no_grad():
            # Call `bias.fill_` and use its result in later steps so downstream prediction heads receive the right feature signal.
            self.cls_head[-1].bias.fill_(-0.1)

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @contextmanager
    def capture_attention(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `self._capturing_attention` for subsequent steps so downstream prediction heads receive the right feature signal.
        self._capturing_attention = True
        # Set `self._captured_attention` for subsequent steps so downstream prediction heads receive the right feature signal.
        self._captured_attention = []
        # Iterate over `self.encoder.layers` so each item contributes to final outputs/metrics.
        for layer in self.encoder.layers:
            # Set `layer._attention_capture_list` for subsequent steps so downstream prediction heads receive the right feature signal.
            layer._attention_capture_list = self._captured_attention
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Execute this statement so downstream prediction heads receive the right feature signal.
            yield
        # Run cleanup that keeps subsequent output steps in a valid state.
        finally:
            # Iterate over `self.encoder.layers` so each item contributes to final outputs/metrics.
            for layer in self.encoder.layers:
                # Branch on `hasattr(layer, "_attention_capture_list")` to choose the correct output computation path.
                if hasattr(layer, "_attention_capture_list"):
                    # Execute this statement so downstream prediction heads receive the right feature signal.
                    del layer._attention_capture_list
            # Set `self._capturing_attention` for subsequent steps so downstream prediction heads receive the right feature signal.
            self._capturing_attention = False

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x: dict) -> dict:
        """Maps current inputs to this module's output tensor representation."""
        # Set `tokens` for subsequent steps so downstream prediction heads receive the right feature signal.
        tokens = x["tokens"]
        # Build `attn_mask` to gate invalid timesteps/joints from influencing outputs.
        attn_mask = x["attn_mask"]
        # Set `time_positions` for subsequent steps so downstream prediction heads receive the right feature signal.
        time_positions = x["time_positions"]
        # Set `event_type_id` for subsequent steps so downstream prediction heads receive the right feature signal.
        event_type_id = x["event_type_id"]
        # Set `token_conf` for subsequent steps so downstream prediction heads receive the right feature signal.
        token_conf = x["token_conf"]
        # Set `event_scalars` for subsequent steps so downstream prediction heads receive the right feature signal.
        event_scalars = x["event_scalars"]
        # Set `delta_t` for subsequent steps so downstream prediction heads receive the right feature signal.
        delta_t = x.get("delta_t", None)

        # Set `B, K, D` for subsequent steps so downstream prediction heads receive the right feature signal.
        B, K, D = tokens.shape
        # Branch on `delta_t is None` to choose the correct output computation path.
        if delta_t is None:
            # Set `delta_t` for subsequent steps so downstream prediction heads receive the right feature signal.
            delta_t = torch.zeros(B, K, device=tokens.device)

        # Set `conf` for subsequent steps so downstream prediction heads receive the right feature signal.
        conf = token_conf.unsqueeze(-1)
        # Compute `type_emb` as an intermediate representation used by later output layers.
        type_emb = self.type_emb(event_type_id)
        # Compute `fused` as an intermediate representation used by later output layers.
        fused = torch.cat([tokens, event_scalars, conf, type_emb], dim=-1)
        # Compute `fused` as an intermediate representation used by later output layers.
        fused = self.token_fuse(fused)

        # Positional + time-gap embedding
        # Compute `fused` as an intermediate representation used by later output layers.
        fused = fused + self.time_emb(time_positions) + self.time_gap_emb(delta_t)

        # Build `src_key_padding_mask` to gate invalid timesteps/joints from influencing outputs.
        src_key_padding_mask = ~attn_mask
        # Build `X` to gate invalid timesteps/joints from influencing outputs.
        X = self.encoder(fused, src_key_padding_mask=src_key_padding_mask)
        # Compute `X` as an intermediate representation used by later output layers.
        X = self.norm(X)

        # Build `valid` to gate invalid timesteps/joints from influencing outputs.
        valid = attn_mask.float().unsqueeze(-1)
        # Compute `X` as an intermediate representation used by later output layers.
        X = X * valid
        # Compute `z` as an intermediate representation used by later output layers.
        z = X.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        # Store raw score tensor in `logit` before probability/decision conversion.
        logit = self.cls_head(z).squeeze(-1)
        # Store raw score tensor in `prob` before probability/decision conversion.
        prob = torch.sigmoid(logit)
        # Store raw score tensor in `conf_logit` before probability/decision conversion.
        conf_logit = self.conf_head(z).squeeze(-1)
        # Store raw score tensor in `conf_score` before probability/decision conversion.
        conf_score = torch.sigmoid(conf_logit)

        # Return `{` as this function's contribution to downstream output flow.
        return {
            "z": z,
            "logit": logit,
            "prob": prob,
            "confidence_score": conf_score,
        }
