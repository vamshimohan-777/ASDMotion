# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Explainability: attention maps and temporal importance.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _build_inputs(batch, device):
    # Return the result expected by the caller.
    return {
        "face_crops": batch["face_crops"].to(device, non_blocking=True),
        "pose_maps": batch["pose_maps"].to(device, non_blocking=True),
        "motion_maps": batch["motion_maps"].to(device, non_blocking=True),
        "hand_maps": batch["motion_maps"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "timestamps": batch["timestamps"].to(device, non_blocking=True),
        "delta_t": batch["delta_t"].to(device, non_blocking=True),
        "route_mask": batch["route_mask"].to(device, non_blocking=True),
        "qualities": {k: v.to(device, non_blocking=True) for k, v in batch["qualities"].items()},
    }


@torch.no_grad()
def extract_attention_maps(model, loader, device, n_samples=8):
    # Invoke `model.eval` to advance this processing stage.
    model.eval()
    # Compute `all_attn` for the next processing step.
    all_attn = []
    # Compute `count` for the next processing step.
    count = 0

    # Iterate `batch` across `loader` to process each element.
    for batch in loader:
        # Branch behavior based on the current runtime condition.
        if count >= n_samples:
            break

        # Compute `inputs` for the next processing step.
        inputs = _build_inputs(batch, device)
        # Compute `B` for the next processing step.
        B = inputs["face_crops"].shape[0]
        # Compute `controller` for the next processing step.
        controller = model.nas_controller

        # Branch behavior based on the current runtime condition.
        if controller.is_discretized:
            # Compute `transformers` for the next processing step.
            transformers = [controller.transformer_candidates[controller._best_transformer_idx]]
        else:
            # Compute `weights` for the next processing step.
            weights = F.softmax(controller.alpha_transformer, dim=0)
            # Compute `best_idx` for the next processing step.
            best_idx = weights.argmax().item()
            # Compute `transformers` for the next processing step.
            transformers = [controller.transformer_candidates[best_idx]]

        # Iterate `transformer` across `transformers` to process each element.
        for transformer in transformers:
            # Run this block with managed resources/context cleanup.
            with transformer.capture_attention():
                # Compute `dev_type` for the next processing step.
                dev_type = device.type if isinstance(device, torch.device) else str(device)
                # Run this block with managed resources/context cleanup.
                with torch.amp.autocast(device_type=dev_type, enabled=dev_type.startswith("cuda")):
                    # Compute `_` for the next processing step.
                    _ = model(inputs)

                # Branch behavior based on the current runtime condition.
                if hasattr(transformer, "_captured_attention") and transformer._captured_attention:
                    # Compute `attn_weights` for the next processing step.
                    attn_weights = transformer._captured_attention
                    # Compute `stacked` for the next processing step.
                    stacked = torch.stack(attn_weights)  # [L, B, H, K, K]
                    # Compute `avg` for the next processing step.
                    avg = stacked.mean(dim=(0, 2))       # [B, K, K]
                    # Invoke `all_attn.append` to advance this processing stage.
                    all_attn.append(avg.cpu().numpy())

        # Update `count` in place using the latest contribution.
        count += B

    # Branch behavior based on the current runtime condition.
    if not all_attn:
        # Return the result expected by the caller.
        return None

    # Compute `all_attn` for the next processing step.
    all_attn = np.concatenate(all_attn, axis=0)
    # Compute `avg_attn` for the next processing step.
    avg_attn = all_attn.mean(axis=0)
    # Return the result expected by the caller.
    return avg_attn


@torch.no_grad()
def compute_temporal_importance(model, loader, device, n_samples=16):
    """
    Approximate SHAP-like temporal importance using attention rollout.
    Returns a vector [K] with relative importance per event token.
    """
    # Invoke `model.eval` to advance this processing stage.
    model.eval()
    # Compute `all_importance` for the next processing step.
    all_importance = []
    # Compute `count` for the next processing step.
    count = 0

    # Iterate `batch` across `loader` to process each element.
    for batch in loader:
        # Branch behavior based on the current runtime condition.
        if count >= n_samples:
            break

        # Compute `inputs` for the next processing step.
        inputs = _build_inputs(batch, device)
        # Compute `B` for the next processing step.
        B = inputs["face_crops"].shape[0]
        # Compute `controller` for the next processing step.
        controller = model.nas_controller

        # Branch behavior based on the current runtime condition.
        if controller.is_discretized:
            # Compute `transformers` for the next processing step.
            transformers = [controller.transformer_candidates[controller._best_transformer_idx]]
        else:
            # Compute `weights` for the next processing step.
            weights = F.softmax(controller.alpha_transformer, dim=0)
            # Compute `best_idx` for the next processing step.
            best_idx = weights.argmax().item()
            # Compute `transformers` for the next processing step.
            transformers = [controller.transformer_candidates[best_idx]]

        # Iterate `transformer` across `transformers` to process each element.
        for transformer in transformers:
            # Run this block with managed resources/context cleanup.
            with transformer.capture_attention():
                # Compute `dev_type` for the next processing step.
                dev_type = device.type if isinstance(device, torch.device) else str(device)
                # Run this block with managed resources/context cleanup.
                with torch.amp.autocast(device_type=dev_type, enabled=dev_type.startswith("cuda")):
                    # Compute `_` for the next processing step.
                    _ = model(inputs)

                # Branch behavior based on the current runtime condition.
                if hasattr(transformer, "_captured_attention") and transformer._captured_attention:
                    # Compute `attn_weights` for the next processing step.
                    attn_weights = transformer._captured_attention
                    # Compute `stacked` for the next processing step.
                    stacked = torch.stack(attn_weights)  # [L, B, H, K, K]
                    # Compute `avg` for the next processing step.
                    avg = stacked.mean(dim=(0, 2))       # [B, K, K]
                    # importance per token = mean attention received across queries
                    imp = avg.mean(axis=1)               # [B, K]
                    # Invoke `all_importance.append` to advance this processing stage.
                    all_importance.append(imp.cpu().numpy())

        # Update `count` in place using the latest contribution.
        count += B

    # Branch behavior based on the current runtime condition.
    if not all_importance:
        # Return the result expected by the caller.
        return None

    # Compute `all_importance` for the next processing step.
    all_importance = np.concatenate(all_importance, axis=0)
    # Compute `mean_imp` for the next processing step.
    mean_imp = all_importance.mean(axis=0)
    # Branch behavior based on the current runtime condition.
    if mean_imp.sum() > 0:
        # Compute `mean_imp` for the next processing step.
        mean_imp = mean_imp / mean_imp.sum()
    # Return the result expected by the caller.
    return mean_imp

