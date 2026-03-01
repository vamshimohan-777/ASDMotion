# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class EventTokenizer(nn.Module):
    """
    Event Tokenizer / Event Encoder

    Converts variable-length micro-kinetic events into fixed-size,
    transformer-ready event tokens with confidence and masking.
    """

    def __init__(
        self,
        d_in: int = 512,
        d_model: int = 256,
        num_event_types: int = 12,
        type_emb_dim: int = 32,
        num_scalars: int = 8,
        K_max: int = 64,
        conf_threshold: float = 0.3,
    ):
        super().__init__()

        # Fixed token budget keeps downstream transformer shape static across clips.
        self.K_max = K_max
        # Scalar feature width is reused when building empty/padded tensors.
        self.num_scalars = num_scalars
        # Output token width expected by downstream sequence model.
        self.d_model = d_model
        # Confidence threshold defines which events are attention-visible by default.
        self.conf_threshold = conf_threshold

        # Embeds discrete behavioral event classes into learnable semantic vectors.
        self.event_type_emb = nn.Embedding(num_event_types, type_emb_dim)

        # Final projection merges appearance, scalar descriptors, and event semantics.
        self.proj = nn.Linear(
            d_in + num_scalars + type_emb_dim,
            d_model
        )

    def forward(
        self,
        features: torch.Tensor,               # [T, d_in]
        energy: torch.Tensor,                 # [T]
        segments: List[Tuple[int, int]],
        frame_conf: torch.Tensor,              # [T]
        stream_conf: torch.Tensor,             # [T, 3]
        fps: int,
        event_type_id: List[int],
    ) -> Dict[str, torch.Tensor]:

        # All tensors are created on the same device as visual features.
        device = features.device
        # Convert event type list into a tensor for indexed embedding lookup.
        event_type_id = torch.tensor(event_type_id, device=device)

        # Per-event outputs collected before stacking/padding to K_max.
        tokens = []
        scalars = []
        token_conf = []
        event_type_ids = []
        attn_mask = []

        for i, (t1, t2) in enumerate(segments):
            # Skip malformed/empty segments to avoid invalid reductions.
            if t2 <= t1:
                continue

            # ---- slice event frames ----
            # Local frame descriptors for this event interval.
            seg_feat = features[t1:t2]           # [L, d_in]
            # Motion energy trace for this event interval.
            seg_energy = energy[t1:t2]            # [L]
            # Frame-level confidence for this event interval.
            seg_frame_conf = frame_conf[t1:t2]    # [L]
            # Per-stream visibility/confidence (face, pose, hands).
            seg_stream_conf = stream_conf[t1:t2]  # [L, 3]

            # ---- event embedding ----
            # Mean pooling summarizes variable-length frame features into one event vector.
            e_k = seg_feat.mean(dim=0)            # [d_in]

            # ---- scalar features (ALL tensors) ----
            # Event duration captures persistence of the behavior.
            duration_sec = torch.tensor((t2 - t1) / fps, device=device)
            # Event start latency captures when behavior appears in the clip.
            latency_sec  = torch.tensor(t1 / fps, device=device)

            # Mean motion approximates sustained movement intensity.
            mean_motion = seg_energy.mean()
            # Peak motion captures short but strong bursts.
            peak_motion = seg_energy.max()

            # Normalize energy into a distribution for temporal dispersion estimation.
            p = seg_energy / (seg_energy.sum() + 1e-6)
            # Entropy captures whether motion is concentrated or spread over the event.
            motion_entropy = -(p * torch.log(p + 1e-6)).sum()

            # Stream visibilities expose which sensing branch is reliable for the event.
            face_vis = seg_stream_conf[:, 0].mean()
            pose_vis = seg_stream_conf[:, 1].mean()
            hand_vis = seg_stream_conf[:, 2].mean()

            # Scalar bundle complements learned features with interpretable cues.
            s_k = torch.stack([
                duration_sec,
                latency_sec,
                mean_motion,
                peak_motion,
                motion_entropy,
                face_vis,
                pose_vis,
                hand_vis,
            ])  # [S]

            # Log scaling reduces dynamic range and stabilizes training on small datasets.
            s_k = torch.log1p(s_k)

            # ---- confidence ----
            # Event confidence is the mean frame confidence over the segment.
            conf_k = seg_frame_conf.mean()

            # ---- event type embedding ----
            # Add event-class semantics so identical motion can be contextually distinguished.
            g_k = self.event_type_emb(event_type_id[i])

            # ---- final token ----
            # Concatenate appearance, scalar descriptors, and type semantics.
            z_k = torch.cat([e_k, s_k, g_k], dim=0)
            # Project concatenated event descriptor to transformer token dimension.
            t_k = self.proj(z_k)                  # [d_model]

            # Store per-event outputs for later batching/padding.
            tokens.append(t_k)
            scalars.append(s_k)
            token_conf.append(conf_k)
            event_type_ids.append(event_type_id[i])
            # Attention mask keeps only sufficiently reliable events active.
            attn_mask.append(conf_k >= self.conf_threshold)

        # ---- stack & pad ----
        # Enforce upper bound on events so output tensor shape is deterministic.
        K = min(len(tokens), self.K_max)

        if K > 0:
            # Stack real events before padding.
            tokens = torch.stack(tokens[:K])
            scalars = torch.stack(scalars[:K])
            token_conf = torch.stack(token_conf[:K])
            event_type_ids = torch.stack(event_type_ids[:K])
            attn_mask = torch.tensor(attn_mask[:K], device=device, dtype=torch.bool)
        else:
            # Empty-safe tensors keep caller code simple when no events are detected.
            tokens = torch.zeros((0, self.d_model), device=device)
            scalars = torch.zeros((0, self.num_scalars), device=device)
            token_conf = torch.zeros((0,), device=device)
            event_type_ids = torch.zeros((0,), device=device, dtype=torch.long)
            attn_mask = torch.zeros((0,), device=device, dtype=torch.bool)

        # Amount of right-padding needed to reach fixed K_max length.
        pad_len = self.K_max - K

        if pad_len > 0:
            # Pad with zeros/false so downstream attention ignores synthetic slots.
            tokens = torch.cat([tokens, torch.zeros(pad_len, self.d_model, device=device)], dim=0)
            scalars = torch.cat([scalars, torch.zeros(pad_len, self.num_scalars, device=device)], dim=0)
            token_conf = torch.cat([token_conf, torch.zeros(pad_len, device=device)], dim=0)
            event_type_ids = torch.cat(
                [event_type_ids, torch.zeros(pad_len, device=device, dtype=torch.long)], dim=0
            )
            attn_mask = torch.cat(
                [attn_mask, torch.zeros(pad_len, device=device, dtype=torch.bool)], dim=0
            )

        # Standardized tokenizer output consumed by downstream temporal reasoning modules.
        return {
            "tokens": tokens,                # [K_max, d_model]
            "attn_mask": attn_mask,          # [K_max]
            "event_type_id": event_type_ids, # [K_max]
            "token_conf": token_conf,        # [K_max]
            "event_scalars": scalars,        # [K_max, S]
        }

