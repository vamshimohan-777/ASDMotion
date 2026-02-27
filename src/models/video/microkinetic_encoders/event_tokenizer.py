import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from src.models.video.microkinetic_encoders.event_types import NUM_EVENT_TYPES


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
        num_event_types: int = NUM_EVENT_TYPES,
        type_emb_dim: int = 32,
        num_scalars: int = 8,
        K_max: int = 64,
        conf_threshold: float = 0.3,
    ):
        super().__init__()

        self.K_max = K_max
        self.num_scalars = num_scalars
        self.d_model = d_model
        self.conf_threshold = conf_threshold

        # Event type embedding
        self.event_type_emb = nn.Embedding(num_event_types, type_emb_dim)

        # Final projection to transformer space
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

        device = features.device
        event_type_id = torch.tensor(event_type_id, device=device)

        tokens = []
        scalars = []
        token_conf = []
        event_type_ids = []
        attn_mask = []

        for i, (t1, t2) in enumerate(segments):
            if t2 <= t1:
                continue

            # ---- slice event frames ----
            seg_feat = features[t1:t2]           # [L, d_in]
            seg_energy = energy[t1:t2]            # [L]
            seg_frame_conf = frame_conf[t1:t2]    # [L]
            seg_stream_conf = stream_conf[t1:t2]  # [L, 3]

            # ---- event embedding ----
            e_k = seg_feat.mean(dim=0)            # [d_in]

            # ---- scalar features (ALL tensors) ----
            duration_sec = torch.tensor((t2 - t1) / fps, device=device)
            latency_sec  = torch.tensor(t1 / fps, device=device)

            mean_motion = seg_energy.mean()
            peak_motion = seg_energy.max()

            p = seg_energy / (seg_energy.sum() + 1e-6)
            motion_entropy = -(p * torch.log(p + 1e-6)).sum()

            face_vis = seg_stream_conf[:, 0].mean()
            pose_vis = seg_stream_conf[:, 1].mean()
            hand_vis = seg_stream_conf[:, 2].mean()

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

            # small-data safe normalization
            s_k = torch.log1p(s_k)

            # ---- confidence ----
            conf_k = seg_frame_conf.mean()

            # ---- event type embedding ----
            g_k = self.event_type_emb(event_type_id[i])

            # ---- final token ----
            z_k = torch.cat([e_k, s_k, g_k], dim=0)
            t_k = self.proj(z_k)                  # [d_model]

            tokens.append(t_k)
            scalars.append(s_k)
            token_conf.append(conf_k)
            event_type_ids.append(event_type_id[i])
            attn_mask.append(conf_k >= self.conf_threshold)

        # ---- stack & pad ----
        K = min(len(tokens), self.K_max)

        if K > 0:
            tokens = torch.stack(tokens[:K])
            scalars = torch.stack(scalars[:K])
            token_conf = torch.stack(token_conf[:K])
            event_type_ids = torch.stack(event_type_ids[:K])
            attn_mask = torch.tensor(attn_mask[:K], device=device, dtype=torch.bool)
        else:
            tokens = torch.zeros((0, self.d_model), device=device)
            scalars = torch.zeros((0, self.num_scalars), device=device)
            token_conf = torch.zeros((0,), device=device)
            event_type_ids = torch.zeros((0,), device=device, dtype=torch.long)
            attn_mask = torch.zeros((0,), device=device, dtype=torch.bool)

        pad_len = self.K_max - K

        if pad_len > 0:
            tokens = torch.cat([tokens, torch.zeros(pad_len, self.d_model, device=device)], dim=0)
            scalars = torch.cat([scalars, torch.zeros(pad_len, self.num_scalars, device=device)], dim=0)
            token_conf = torch.cat([token_conf, torch.zeros(pad_len, device=device)], dim=0)
            event_type_ids = torch.cat(
                [event_type_ids, torch.zeros(pad_len, device=device, dtype=torch.long)], dim=0
            )
            attn_mask = torch.cat(
                [attn_mask, torch.zeros(pad_len, device=device, dtype=torch.bool)], dim=0
            )

        return {
            "tokens": tokens,                # [K_max, d_model]
            "attn_mask": attn_mask,          # [K_max]
            "event_type_id": event_type_ids, # [K_max]
            "token_conf": token_conf,        # [K_max]
            "event_scalars": scalars,        # [K_max, S]
        }
