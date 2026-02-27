"""
Learnable Microkinetic Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.video.microkinetic_encoders.event_types import NUM_EVENT_TYPES


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        n_groups = 16 if out_ch % 16 == 0 else (8 if out_ch % 8 == 0 else 1)
        self.norm = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class MicroKineticEncoder(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        d_model: int = 256,
        K_max: int = 32,
        num_event_types: int = NUM_EVENT_TYPES,
        num_scalars: int = 8,
        conv_channels: int = 256,
        kernel_sizes: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.d_model = d_model
        self.K_max = K_max
        self.num_scalars = num_scalars

        self.conv_branches = nn.ModuleList([
            TemporalConvBlock(d_in, conv_channels, ks, dropout=dropout)
            for ks in kernel_sizes
        ])

        fused_channels = conv_channels * len(kernel_sizes)

        self.channel_fuse = nn.Sequential(
            nn.Linear(fused_channels, conv_channels),
            nn.GELU(),
            nn.LayerNorm(conv_channels),
            nn.Dropout(dropout),
        )

        self.event_gate = nn.Sequential(
            nn.Linear(conv_channels, conv_channels // 4),
            nn.GELU(),
            nn.Linear(conv_channels // 4, 1),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(conv_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.scalar_head = nn.Sequential(
            nn.Linear(conv_channels, num_scalars),
        )

        self.type_head = nn.Linear(conv_channels, num_event_types)

        self.conf_head = nn.Sequential(
            nn.Linear(conv_channels, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
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
        B, T, D = features.shape
        device = features.device

        if conv_features is not None:
            fused = conv_features
        else:
            x = features.permute(0, 2, 1)
            branch_outs = []
            for conv in self.conv_branches:
                branch_outs.append(conv(x))
            multi_scale = torch.cat(branch_outs, dim=1)
            multi_scale = multi_scale.permute(0, 2, 1)
            fused = self.channel_fuse(multi_scale)

        gate_logits = self.event_gate(fused).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~mask.bool(), float("-inf"))
        gate_scores = torch.sigmoid(gate_logits)

        K = min(self.K_max, T)
        topk_scores, topk_indices = torch.topk(gate_scores, K, dim=1)
        sorted_order = topk_indices.sort(dim=1).indices
        topk_indices = topk_indices.gather(1, sorted_order)
        topk_scores = topk_scores.gather(1, sorted_order)

        idx_expand = topk_indices.unsqueeze(-1).expand(-1, -1, fused.size(-1))
        event_feats = fused.gather(1, idx_expand)

        tokens = self.token_proj(event_feats)
        scalars = self.scalar_head(event_feats)
        type_logits = self.type_head(event_feats)
        event_type_id = type_logits.argmax(dim=-1)
        token_conf = self.conf_head(event_feats).squeeze(-1)

        mask_long = mask.long()
        event_mask = mask_long.gather(1, topk_indices).bool()

        if timestamps is None:
            time_positions = topk_indices.float()
        else:
            time_positions = timestamps.gather(1, topk_indices)

        if delta_t is None:
            if timestamps is None:
                delta_events = torch.zeros_like(time_positions)
            else:
                # Approximate delta_t between consecutive selected events
                delta_events = torch.zeros_like(time_positions)
                if time_positions.shape[1] > 1:
                    delta_events[:, 1:] = time_positions[:, 1:] - time_positions[:, :-1]
                    delta_events = torch.clamp(delta_events, min=0.0)
        else:
            delta_events = delta_t.gather(1, topk_indices)

        pad_len = self.K_max - K
        if pad_len > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_len))
            event_mask = F.pad(event_mask, (0, pad_len), value=False)
            time_positions = F.pad(time_positions, (0, pad_len))
            event_type_id = F.pad(event_type_id, (0, pad_len))
            token_conf = F.pad(token_conf, (0, pad_len))
            scalars = F.pad(scalars, (0, 0, 0, pad_len))
            delta_events = F.pad(delta_events, (0, pad_len))

        return {
            "tokens": tokens,
            "attn_mask": event_mask,
            "time_positions": time_positions,
            "event_type_id": event_type_id,
            "token_conf": token_conf,
            "event_scalars": scalars,
            "delta_t": delta_events,
        }
