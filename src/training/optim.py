# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Optimizer and scheduler builders for ASD Pipeline training.

Creates separate Adam parameter groups for:
  - Model weights      (lr = model_lr)
  - NAS arch params    (lr = arch_lr)
  - Optional fusion parameters (lr = fusion_lr)

Includes cosine annealing with linear warmup scheduler.
"""

import torch
import torch.optim as optim


def build_optimizer(
    model,
    model_lr: float = 1e-4,
    arch_lr: float = 3e-4,
    fusion_lr: float = 1e-3,
    weight_decay: float = 0.001,
):
    """
    Build Adam optimizer with separate parameter groups.

    Args:
        model: ASDPipeline instance (with .model_parameters(), .arch_parameters())
        model_lr:  learning rate for main model weights
        arch_lr:   learning rate for NAS architecture parameters
        fusion_lr: learning rate for fusion module parameters (if present)

    Returns:
        torch.optim.Adam optimizer
    """
    # Collect parameter IDs for exclusion
    arch_ids = set(id(p) for p in model.arch_parameters())
    # Compute `fusion_module` for the next processing step.
    fusion_module = getattr(model, "fusion", None)
    # Compute `fusion_params` for the next processing step.
    fusion_params = []
    # Branch behavior based on the current runtime condition.
    if fusion_module is not None:
        # Compute `fusion_params` for the next processing step.
        fusion_params = [p for p in fusion_module.parameters() if p.requires_grad]
    # Compute `fusion_ids` for the next processing step.
    fusion_ids = set(id(p) for p in fusion_params)

    # Model params = everything trainable except arch + fusion
    model_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in arch_ids and id(p) not in fusion_ids
    ]

    # Compute `param_groups` for the next processing step.
    param_groups = [
        {"params": model_params, "lr": model_lr, "weight_decay": weight_decay},
        {"params": list(model.arch_parameters()), "lr": arch_lr, "weight_decay": 0.0},
    ]
    # Branch behavior based on the current runtime condition.
    if fusion_params:
        # Invoke `param_groups.append` to advance this processing stage.
        param_groups.append(
            {"params": fusion_params, "lr": fusion_lr, "weight_decay": 0.0}
        )

    # Compute `optimizer` for the next processing step.
    optimizer = optim.AdamW(param_groups)
    # Return the result expected by the caller.
    return optimizer

