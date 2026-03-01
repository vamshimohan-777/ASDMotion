"""Training module `src/training/motion_ssl.py` that optimizes model weights and output quality."""

# Import `math` to support computations in this stage of output generation.
import math
# Import `time` to support computations in this stage of output generation.
import time

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn
# Import `torch.nn.functional as F` to support computations in this stage of output generation.
import torch.nn.functional as F


# Define a reusable pipeline function whose outputs feed later steps.
def _interpolate_time(x, out_t):
    # x: [N, T, J, F]
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `n, t, j, f` for subsequent steps so gradient updates improve future predictions.
    n, t, j, f = x.shape
    # Compute `xf` as an intermediate representation used by later output layers.
    xf = x.permute(0, 2, 3, 1).reshape(n, j * f, t)
    # Set `y` for subsequent steps so gradient updates improve future predictions.
    y = F.interpolate(xf, size=out_t, mode="linear", align_corners=False)
    # Set `y` for subsequent steps so gradient updates improve future predictions.
    y = y.reshape(n, j, f, out_t).permute(0, 3, 1, 2).contiguous()
    # Return `y` as this function's contribution to downstream output flow.
    return y


# Define a reusable pipeline function whose outputs feed later steps.
def augment_motion_windows(
    x,
    joint_mask=None,
    joint_dropout=0.1,
    coord_noise_std=0.01,
    temporal_mask_ratio=0.15,
    speed_min=0.8,
    speed_max=1.2,
):
    # x: [N, W, J, 9]
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `y` for subsequent steps so gradient updates improve future predictions.
    y = x.clone()
    # Set `n, w, j, _` for subsequent steps so gradient updates improve future predictions.
    n, w, j, _ = y.shape
    # Set `device` to the execution device used for this computation path.
    device = y.device

    # Joint dropout
    # Branch on `joint_dropout > 0` to choose the correct output computation path.
    if joint_dropout > 0:
        # Set `jd` for subsequent steps so gradient updates improve future predictions.
        jd = (torch.rand((n, j), device=device) < float(joint_dropout)).float()
        # Set `jd` for subsequent steps so gradient updates improve future predictions.
        jd = jd.unsqueeze(1).unsqueeze(-1)  # [N,1,J,1]
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = y * (1.0 - jd)
        # Branch on `joint_mask is not None` to choose the correct output computation path.
        if joint_mask is not None:
            # Build `joint_mask` to gate invalid timesteps/joints from influencing outputs.
            joint_mask = joint_mask * (1.0 - jd.squeeze(-1))

    # Coordinate noise
    # Branch on `coord_noise_std > 0` to choose the correct output computation path.
    if coord_noise_std > 0:
        # Set `noise` for subsequent steps so gradient updates improve future predictions.
        noise = torch.randn_like(y[..., :3]) * float(coord_noise_std)
        # Execute this statement so gradient updates improve future predictions.
        y[..., :3] = y[..., :3] + noise

    # Temporal masking
    # Branch on `temporal_mask_ratio > 0` to choose the correct output computation path.
    if temporal_mask_ratio > 0:
        # Build `n_mask` to gate invalid timesteps/joints from influencing outputs.
        n_mask = max(1, int(w * float(temporal_mask_ratio)))
        # Iterate over `range(n)` so each item contributes to final outputs/metrics.
        for i in range(n):
            # Branch on `w <= 1` to choose the correct output computation path.
            if w <= 1:
                # Skip current loop item so it does not affect accumulated output state.
                continue
            # Build `start` to gate invalid timesteps/joints from influencing outputs.
            start = int(torch.randint(0, max(1, w - n_mask + 1), (1,), device=device).item())
            # Execute this statement so gradient updates improve future predictions.
            y[i, start : start + n_mask] = 0.0
            # Branch on `joint_mask is not None` to choose the correct output computation path.
            if joint_mask is not None:
                # Execute this statement so gradient updates improve future predictions.
                joint_mask[i, start : start + n_mask] = 0.0

    # Playback speed variation (resample to random length, then back).
    # Branch on `speed_min > 0 and speed_max > 0 and speed_max >= ...` to choose the correct output computation path.
    if speed_min > 0 and speed_max > 0 and speed_max >= speed_min:
        # Set `speed` for subsequent steps so gradient updates improve future predictions.
        speed = torch.empty((1,), device=device).uniform_(float(speed_min), float(speed_max)).item()
        # Set `target_t` for subsequent steps so gradient updates improve future predictions.
        target_t = max(4, int(round(w / max(speed, 1e-4))))
        # Compute `z` as an intermediate representation used by later output layers.
        z = _interpolate_time(y, target_t)
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = _interpolate_time(z, w)
        # Branch on `joint_mask is not None` to choose the correct output computation path.
        if joint_mask is not None:
            # Build `m` to gate invalid timesteps/joints from influencing outputs.
            m = joint_mask.unsqueeze(-1)
            # Set `m` for subsequent steps so gradient updates improve future predictions.
            m = _interpolate_time(m, target_t)
            # Set `m` for subsequent steps so gradient updates improve future predictions.
            m = _interpolate_time(m, w)
            # Build `joint_mask` to gate invalid timesteps/joints from influencing outputs.
            joint_mask = (m.squeeze(-1) > 0.5).float()

    # Return `y, joint_mask` as this function's contribution to downstream output flow.
    return y, joint_mask


# Define a reusable pipeline function whose outputs feed later steps.
def info_nce(z1, z2, temperature=0.1):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Compute `z1` as an intermediate representation used by later output layers.
    z1 = F.normalize(z1, dim=-1)
    # Compute `z2` as an intermediate representation used by later output layers.
    z2 = F.normalize(z2, dim=-1)
    # Store raw score tensor in `logits` before probability/decision conversion.
    logits = (z1 @ z2.T) / max(float(temperature), 1e-6)
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = torch.arange(z1.shape[0], device=z1.device)
    # Return `F.cross_entropy(logits, labels)` as this function's contribution to downstream output flow.
    return F.cross_entropy(logits, labels)


# Define a training routine that updates parameters and changes future outputs.
def pretrain_motion_encoder(
    model,
    loader,
    device,
    epochs=5,
    lr=1e-4,
    max_steps_per_epoch=300,
    logger=None,
):
    """
    Self-supervised pretraining objectives:
    1) Temporal contrastive
    2) Future motion prediction
    3) Motion consistency
    """
    # Call `model.train` and use its result in later steps so gradient updates improve future predictions.
    model.train()
    # Iterate over `model.motion_encoder.parameters()` so each item contributes to final outputs/metrics.
    for p in model.motion_encoder.parameters():
        # Set `p.requires_grad` for subsequent steps so gradient updates improve future predictions.
        p.requires_grad = True

    # Set `proj_dim` for subsequent steps so gradient updates improve future predictions.
    proj_dim = model.motion_encoder.embedding_dim
    # Set `predictor` to predicted labels/scores that are reported downstream.
    predictor = nn.Sequential(
        nn.Linear(proj_dim, proj_dim),
        nn.GELU(),
        nn.Linear(proj_dim, proj_dim),
    ).to(device)

    # Set `params` for subsequent steps so gradient updates improve future predictions.
    params = list(model.motion_encoder.parameters()) + list(predictor.parameters())
    # Initialize `optimizer` to control parameter updates during training.
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-4)
    # Set `scaler` for subsequent steps so gradient updates improve future predictions.
    scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

    # Iterate over `range(1, int(epochs) + 1)` so each item contributes to final outputs/metrics.
    for epoch in range(1, int(epochs) + 1):
        # Update `epoch_loss` with a loss term that drives backpropagation and output improvement.
        epoch_loss = 0.0
        # Set `n_steps` for subsequent steps so gradient updates improve future predictions.
        n_steps = 0
        # Set `started` for subsequent steps so gradient updates improve future predictions.
        started = time.time()
        # Iterate over `loader` so each item contributes to final outputs/metrics.
        for batch in loader:
            # Compute `windows` as an intermediate representation used by later output layers.
            windows = batch["motion_windows"].to(device, non_blocking=True)  # [B,S,W,J,9]
            # Build `joint_mask` to gate invalid timesteps/joints from influencing outputs.
            joint_mask = batch["joint_mask"].to(device, non_blocking=True)  # [B,S,W,J]
            # Set `b, s, w, j, f` for subsequent steps so gradient updates improve future predictions.
            b, s, w, j, f = windows.shape
            # Set `flat` for subsequent steps so gradient updates improve future predictions.
            flat = windows.reshape(b * s, w, j, f)
            # Build `flat_mask` to gate invalid timesteps/joints from influencing outputs.
            flat_mask = joint_mask.reshape(b * s, w, j)

            # Build `v1, m1` to gate invalid timesteps/joints from influencing outputs.
            v1, m1 = augment_motion_windows(flat, flat_mask.clone())
            # Build `v2, m2` to gate invalid timesteps/joints from influencing outputs.
            v2, m2 = augment_motion_windows(flat, flat_mask.clone())

            # Set `use_amp` for subsequent steps so gradient updates improve future predictions.
            use_amp = str(device).startswith("cuda")
            # Use a managed context to safely handle resources used during output computation.
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Build `z1` to gate invalid timesteps/joints from influencing outputs.
                z1 = model.motion_encoder(v1, joint_mask=m1)
                # Build `z2` to gate invalid timesteps/joints from influencing outputs.
                z2 = model.motion_encoder(v2, joint_mask=m2)
                # Update `loss_contrast` with a loss term that drives backpropagation and output improvement.
                loss_contrast = info_nce(z1, z2, temperature=0.1)
                # Update `loss_consistency` with a loss term that drives backpropagation and output improvement.
                loss_consistency = F.mse_loss(z1, z2)

                # Future prediction from sequence of windows per video.
                # Build `z_seq` to gate invalid timesteps/joints from influencing outputs.
                z_seq = model.motion_encoder(flat, joint_mask=flat_mask).reshape(b, s, -1)
                # Branch on `s > 1` to choose the correct output computation path.
                if s > 1:
                    # Set `pred` to predicted labels/scores that are reported downstream.
                    pred = predictor(z_seq[:, :-1, :])
                    # Set `tgt` for subsequent steps so gradient updates improve future predictions.
                    tgt = z_seq[:, 1:, :].detach()
                    # Update `loss_future` with a loss term that drives backpropagation and output improvement.
                    loss_future = F.smooth_l1_loss(pred, tgt)
                else:
                    # Update `loss_future` with a loss term that drives backpropagation and output improvement.
                    loss_future = z_seq.new_tensor(0.0)

                # Update `loss` with a loss term that drives backpropagation and output improvement.
                loss = loss_contrast + 0.5 * loss_future + 0.5 * loss_consistency

            # Reset gradients before next step to avoid mixing gradient signals across batches.
            optimizer.zero_grad(set_to_none=True)
            # Backpropagate current loss so gradients can update model output behavior.
            scaler.scale(loss).backward()
            # Call `scaler.unscale_` and use its result in later steps so gradient updates improve future predictions.
            scaler.unscale_(optimizer)
            # Call `torch.nn.utils.clip_grad_norm_` and use its result in later steps so gradient updates improve future predictions.
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            # Apply optimizer step so future predictions reflect this batch's gradients.
            scaler.step(optimizer)
            # Call `scaler.update` and use its result in later steps so gradient updates improve future predictions.
            scaler.update()

            # Call `float` and use its result in later steps so gradient updates improve future predictions.
            epoch_loss += float(loss.item())
            # Execute this statement so gradient updates improve future predictions.
            n_steps += 1
            # Branch on `max_steps_per_epoch and n_steps >= int(max_steps_...` to choose the correct output computation path.
            if max_steps_per_epoch and n_steps >= int(max_steps_per_epoch):
                # Stop iteration early to prevent further changes to the current output state.
                break

        # Update `avg_loss` with a loss term that drives backpropagation and output improvement.
        avg_loss = epoch_loss / max(n_steps, 1)
        # Set `elapsed` for subsequent steps so gradient updates improve future predictions.
        elapsed = time.time() - started
        # Branch on `logger is not None` to choose the correct output computation path.
        if logger is not None:
            # Log runtime values to verify that output computation is behaving as expected.
            logger.log(
                "ssl_pretrain",
                epoch=epoch,
                loss=avg_loss,
                steps=n_steps,
                elapsed_sec=round(elapsed, 2),
            )
        # Log runtime values to verify that output computation is behaving as expected.
        print(f"[SSL] epoch={epoch} loss={avg_loss:.4f} steps={n_steps} time={elapsed:.1f}s")

