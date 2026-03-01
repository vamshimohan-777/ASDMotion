# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Learnable Microkinetic Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        # Symmetric padding preserves frame count so temporal indices still match source frames.
        padding = kernel_size // 2
        # 1D temporal convolution learns motion patterns over short frame windows.
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        # GroupNorm is stable for small batches common in video detection training.
        n_groups = 16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1)
        self.norm = nn.GroupNorm(n_groups, out_ch)
        # GELU keeps weak micro-motion responses instead of hard-thresholding them.
        self.act = nn.GELU()
        # Dropout regularizes temporal features to reduce overfitting on sparse events.
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Conv -> normalize -> nonlinearity -> dropout forms one robust temporal feature extractor.
        return self.drop(self.act(self.norm(self.conv(x))))


class MicroKineticEncoder(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        d_model: int = 256,
        K_max: int = 32,
        num_event_types: int = 12,
        num_scalars: int = 8,
        conv_channels: int = 256,
        kernel_sizes: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kernel_sizes is None:
            # Multiple receptive fields capture both rapid twitches and slightly longer motions.
            kernel_sizes = [3, 5, 7]

        self.d_model = d_model
        self.K_max = K_max
        self.num_scalars = num_scalars

        self.conv_branches = nn.ModuleList([
            # Parallel temporal branches learn complementary motion cues at each scale.
            TemporalConvBlock(d_in, conv_channels, ks, dropout=dropout)
            for ks in kernel_sizes
        ])

        # Branch outputs are concatenated, so fused width scales with number of kernels.
        fused_channels = conv_channels * len(kernel_sizes)

        self.channel_fuse = nn.Sequential(
            # Compress concatenated multi-scale evidence into one shared motion descriptor.
            nn.Linear(fused_channels, conv_channels),
            nn.GELU(),
            nn.LayerNorm(conv_channels),
            nn.Dropout(dropout),
        )

        self.event_gate = nn.Sequential(
            # Gate predicts event likelihood per frame so only salient moments are tokenized.
            nn.Linear(conv_channels, conv_channels // 4),
            nn.GELU(),
            nn.Linear(conv_channels // 4, 1),
        )

        self.token_proj = nn.Sequential(
            # Projects selected event features into the transformer token space.
            nn.Linear(conv_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.scalar_head = nn.Sequential(
            # Auxiliary scalar head provides compact interpretable event statistics.
            nn.Linear(conv_channels, num_scalars),
        )

        # Event type logits let the model attach semantics to each selected motion event.
        self.type_head = nn.Linear(conv_channels, num_event_types)

        self.conf_head = nn.Sequential(
            # Confidence head calibrates token reliability for downstream masking/weighting.
            nn.Linear(conv_channels, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        # Stable initialization is critical because gate/top-k selection is sensitive to scale.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor,
                conv_features: torch.Tensor = None,
                timestamps: torch.Tensor = None,
                delta_t: torch.Tensor = None):
        # features: dense per-frame descriptors from upstream visual encoder [B, T, D].
        B, T, D = features.shape
        device = features.device

        if conv_features is not None:
            # Reuse externally-computed temporal features when provided (saves compute).
            fused = conv_features
        else:
            # Conv1d expects channel-first layout [B, D, T].
            x = features.permute(0, 2, 1)
            branch_outs = []
            for conv in self.conv_branches:
                # Each branch contributes one temporal scale of motion evidence.
                branch_outs.append(conv(x))
            # Concatenate scales along channel axis to preserve all motion hypotheses.
            multi_scale = torch.cat(branch_outs, dim=1)
            # Return to [B, T, C] for per-frame scoring and token extraction.
            multi_scale = multi_scale.permute(0, 2, 1)
            # Fuse scales into compact event-ready frame embeddings.
            fused = self.channel_fuse(multi_scale)

        # Per-frame gate logits indicate which timestamps are likely meaningful events.
        gate_logits = self.event_gate(fused).squeeze(-1)
        # Invalid/padded frames are forced out of selection.
        gate_logits = gate_logits.masked_fill(~mask.bool(), float("-inf"))
        # Sigmoid converts logits to interpretable event scores in [0, 1].
        gate_scores = torch.sigmoid(gate_logits)

        # Select up to K_max strongest candidate events per sample.
        K = min(self.K_max, T)
        topk_scores, topk_indices = torch.topk(gate_scores, K, dim=1)
        # Keep temporal order so downstream models see coherent event sequences.
        sorted_order = topk_indices.sort(dim=1).indices
        topk_indices = topk_indices.gather(1, sorted_order)
        topk_scores = topk_scores.gather(1, sorted_order)

        # Gather selected event features from fused per-frame tensor.
        idx_expand = topk_indices.unsqueeze(-1).expand(-1, -1, fused.size(-1))
        event_feats = fused.gather(1, idx_expand)

        # Convert selected event descriptors to model tokens.
        tokens = self.token_proj(event_feats)
        # Predict event-level scalar metadata useful for interpretation and fusion.
        scalars = self.scalar_head(event_feats)
        # Predict discrete event type per token.
        type_logits = self.type_head(event_feats)
        event_type_id = type_logits.argmax(dim=-1)
        # Predict confidence score used by downstream weighting or filtering.
        token_conf = self.conf_head(event_feats).squeeze(-1)

        # Build boolean attention mask aligned with selected events.
        mask_long = mask.long()
        event_mask = mask_long.gather(1, topk_indices).bool()

        if timestamps is None:
            # Fallback temporal position: use frame indices when timestamps are unavailable.
            time_positions = topk_indices.float()
        else:
            # Preserve original timestamp units when they are provided.
            time_positions = timestamps.gather(1, topk_indices)

        if delta_t is None:
            if timestamps is None:
                # No timing metadata available, so initialize with zeros.
                delta_events = torch.zeros_like(time_positions)
            else:
                # Approximate delta_t between consecutive selected events
                delta_events = torch.zeros_like(time_positions)
                if time_positions.shape[1] > 1:
                    # Temporal gap helps downstream modules reason about event cadence.
                    delta_events[:, 1:] = time_positions[:, 1:] - time_positions[:, :-1]
                    delta_events = torch.clamp(delta_events, min=0.0)
        else:
            # Use externally-computed inter-frame deltas when supplied.
            delta_events = delta_t.gather(1, topk_indices)

        # Pad to fixed K_max so batch collation and transformer input shapes stay static.
        pad_len = self.K_max - K
        if pad_len > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_len))
            event_mask = F.pad(event_mask, (0, pad_len), value=False)
            time_positions = F.pad(time_positions, (0, pad_len))
            event_type_id = F.pad(event_type_id, (0, pad_len))
            token_conf = F.pad(token_conf, (0, pad_len))
            scalars = F.pad(scalars, (0, 0, 0, pad_len))
            delta_events = F.pad(delta_events, (0, pad_len))

        # Return a complete event-token package for downstream sequence modeling.
        return {
            "tokens": tokens,
            "attn_mask": event_mask,
            "time_positions": time_positions,
            "event_type_id": event_type_id,
            "token_conf": token_conf,
            "event_scalars": scalars,
            "delta_t": delta_events,
        }

