"""Model module `src/models/video/motion/behavior_transformer.py` that transforms inputs into features used for prediction."""

# Import `math` to support computations in this stage of output generation.
import math

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn


# Define class `PositionalEncoding` to package related logic in the prediction pipeline.
class PositionalEncoding(nn.Module):
    """`PositionalEncoding` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, d_model, max_len=4096):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `pe` for subsequent steps so downstream prediction heads receive the right feature signal.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        # Set `pos` for subsequent steps so downstream prediction heads receive the right feature signal.
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Set `div_term` for subsequent steps so downstream prediction heads receive the right feature signal.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        # Call `torch.sin` and use its result in later steps so downstream prediction heads receive the right feature signal.
        pe[:, 0::2] = torch.sin(pos * div_term)
        # Call `torch.cos` and use its result in later steps so downstream prediction heads receive the right feature signal.
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Call `self.register_buffer` and use its result in later steps so downstream prediction heads receive the right feature signal.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x):
        # x: [B, T, D]
        """Maps current inputs to this module's output tensor representation."""
        # Return `x + self.pe[:, : x.size(1), :]` as this function's contribution to downstream output flow.
        return x + self.pe[:, : x.size(1), :]


# Define class `BehavioralTransformer` to package related logic in the prediction pipeline.
class BehavioralTransformer(nn.Module):
    """`BehavioralTransformer` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        d_model=256,
        n_heads=4,
        n_layers=3,
        dim_ff=512,
        dropout=0.2,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Set `self.pos` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.pos = PositionalEncoding(d_model=d_model)
        # Set `self.time_mlp` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Set `enc_layer` for subsequent steps so downstream prediction heads receive the right feature signal.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Set `self.encoder` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        # Set `self.norm` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.norm = nn.LayerNorm(d_model)
        # Compute `self.head` as an intermediate representation used by later output layers.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        # Set `self.attn_pool` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, window_embeddings, window_mask=None, event_times=None, aggregation="attention"):
        # window_embeddings: [B, T, D]
        """Maps current inputs to this module's output tensor representation."""
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.pos(window_embeddings)
        # Branch on `event_times is not None` to choose the correct output computation path.
        if event_times is not None:
            # Add continuous-time embedding for event-aligned tokens.
            # Set `t` for subsequent steps so downstream prediction heads receive the right feature signal.
            t = torch.log1p(event_times.float()).unsqueeze(-1)
            # Compute `h` as an intermediate representation used by later output layers.
            h = h + self.time_mlp(t)
        # Build `key_padding_mask` to gate invalid timesteps/joints from influencing outputs.
        key_padding_mask = None
        # Branch on `window_mask is not None` to choose the correct output computation path.
        if window_mask is not None:
            # Build `key_padding_mask` to gate invalid timesteps/joints from influencing outputs.
            key_padding_mask = ~window_mask.bool()
        # Build `h` to gate invalid timesteps/joints from influencing outputs.
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        # Compute `h` as an intermediate representation used by later output layers.
        h = self.norm(h)

        # Branch on `window_mask is None` to choose the correct output computation path.
        if window_mask is None:
            # Build `window_mask` to gate invalid timesteps/joints from influencing outputs.
            window_mask = torch.ones(
                (h.size(0), h.size(1)), device=h.device, dtype=torch.bool
            )
        # Build `valid` to gate invalid timesteps/joints from influencing outputs.
        valid = window_mask.float().unsqueeze(-1)

        # Branch on `aggregation == "topk"` to choose the correct output computation path.
        if aggregation == "topk":
            # Compute `scores` as confidence values used in final prediction decisions.
            scores = torch.sigmoid(self.head(h)).squeeze(-1)
            # Build `scores` to gate invalid timesteps/joints from influencing outputs.
            scores = scores.masked_fill(~window_mask.bool(), -1.0)
            # Set `k` for subsequent steps so downstream prediction heads receive the right feature signal.
            k = max(1, min(h.shape[1] // 3, h.shape[1]))
            # Set `topk_vals, _` for subsequent steps so downstream prediction heads receive the right feature signal.
            topk_vals, _ = torch.topk(scores, k=k, dim=1)
            # Set `pooled` for subsequent steps so downstream prediction heads receive the right feature signal.
            pooled = topk_vals.mean(dim=1, keepdim=True)
            # Store raw score tensor in `logit` before probability/decision conversion.
            logit = torch.logit(torch.clamp(pooled.squeeze(-1), 1e-5, 1 - 1e-5))
            # Return `{` as this function's contribution to downstream output flow.
            return {
                "logit": logit,
                "window_scores": scores,
                "attention_weights": None,
                "pooled": pooled.squeeze(-1),
            }

        # Attention pooling.
        # Set `a` for subsequent steps so downstream prediction heads receive the right feature signal.
        a = self.attn_pool(h).squeeze(-1)
        # Build `a` to gate invalid timesteps/joints from influencing outputs.
        a = a.masked_fill(~window_mask.bool(), float("-inf"))
        # Set `w` for subsequent steps so downstream prediction heads receive the right feature signal.
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        # Set `w` for subsequent steps so downstream prediction heads receive the right feature signal.
        w = w * valid
        # Set `w` for subsequent steps so downstream prediction heads receive the right feature signal.
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        # Set `pooled` for subsequent steps so downstream prediction heads receive the right feature signal.
        pooled = (h * w).sum(dim=1)
        # Store raw score tensor in `logit` before probability/decision conversion.
        logit = self.head(pooled).squeeze(-1)
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "logit": logit,
            "window_scores": torch.sigmoid(self.head(h)).squeeze(-1),
            "attention_weights": w.squeeze(-1),
            "pooled": pooled,
        }
