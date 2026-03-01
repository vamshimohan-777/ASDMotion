# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Explainability: attention maps and temporal importance.
"""

import numpy as np
import torch


def _build_inputs(batch, device):
    return {
        "face_crops": batch["face_crops"].to(device, non_blocking=True),
        "pose_maps": batch["pose_maps"].to(device, non_blocking=True),
        "motion_maps": batch["motion_maps"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "timestamps": batch["timestamps"].to(device, non_blocking=True),
        "delta_t": batch["delta_t"].to(device, non_blocking=True),
        "route_mask": batch["route_mask"].to(device, non_blocking=True),
        "qualities": {k: v.to(device, non_blocking=True) for k, v in batch["qualities"].items()},
    }


@torch.no_grad()
def extract_attention_maps(model, loader, device, n_samples=8):
    model.eval()
    all_attn = []
    count = 0

    for batch in loader:
        if count >= n_samples:
            break

        inputs = _build_inputs(batch, device)
        B = inputs["face_crops"].shape[0]
        controller = model.nas_controller

        if controller.is_discretized:
            transformers = [controller.transformer_candidates[controller._best_transformer_idx]]
        else:
            weights = F.softmax(controller.alpha_transformer, dim=0)
            best_idx = weights.argmax().item()
            transformers = [controller.transformer_candidates[best_idx]]

        for transformer in transformers:
            with transformer.capture_attention():
                dev_type = device.type if isinstance(device, torch.device) else str(device)
                with torch.amp.autocast(device_type=dev_type, enabled=dev_type.startswith("cuda")):
                    _ = model(inputs)

                if hasattr(transformer, "_captured_attention") and transformer._captured_attention:
                    attn_weights = transformer._captured_attention
                    stacked = torch.stack(attn_weights)  # [L, B, H, K, K]
                    avg = stacked.mean(dim=(0, 2))       # [B, K, K]
                    all_attn.append(avg.cpu().numpy())

        count += B

    if not all_attn:
        return None

    all_attn = np.concatenate(all_attn, axis=0)
    avg_attn = all_attn.mean(axis=0)
    return avg_attn


@torch.no_grad()
def compute_temporal_importance(model, loader, device, n_samples=16):
    """
    Approximate SHAP-like temporal importance using attention rollout.
    Returns a vector [K] with relative importance per event token.
    """
    model.eval()
    all_importance = []
    count = 0

    for batch in loader:
        if count >= n_samples:
            break

        inputs = _build_inputs(batch, device)
        B = inputs["face_crops"].shape[0]
        controller = model.nas_controller

        if controller.is_discretized:
            transformers = [controller.transformer_candidates[controller._best_transformer_idx]]
        else:
            weights = F.softmax(controller.alpha_transformer, dim=0)
            best_idx = weights.argmax().item()
            transformers = [controller.transformer_candidates[best_idx]]

        for transformer in transformers:
            with transformer.capture_attention():
                dev_type = device.type if isinstance(device, torch.device) else str(device)
                with torch.amp.autocast(device_type=dev_type, enabled=dev_type.startswith("cuda")):
                    _ = model(inputs)

                if hasattr(transformer, "_captured_attention") and transformer._captured_attention:
                    attn_weights = transformer._captured_attention
                    stacked = torch.stack(attn_weights)  # [L, B, H, K, K]
                    avg = stacked.mean(dim=(0, 2))       # [B, K, K]
                    # importance per token = mean attention received across queries
                    imp = avg.mean(axis=1)               # [B, K]
                    all_importance.append(imp.cpu().numpy())

        count += B

    if not all_importance:
        return None

    all_importance = np.concatenate(all_importance, axis=0)
    mean_imp = all_importance.mean(axis=0)
    if mean_imp.sum() > 0:
        mean_imp = mean_imp / mean_imp.sum()
    return mean_imp

