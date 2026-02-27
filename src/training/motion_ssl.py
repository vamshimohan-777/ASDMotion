import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interpolate_time(x, out_t):
    # x: [N, T, J, F]
    n, t, j, f = x.shape
    xf = x.permute(0, 2, 3, 1).reshape(n, j * f, t)
    y = F.interpolate(xf, size=out_t, mode="linear", align_corners=False)
    y = y.reshape(n, j, f, out_t).permute(0, 3, 1, 2).contiguous()
    return y


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
    y = x.clone()
    n, w, j, _ = y.shape
    device = y.device

    # Joint dropout
    if joint_dropout > 0:
        jd = (torch.rand((n, j), device=device) < float(joint_dropout)).float()
        jd = jd.unsqueeze(1).unsqueeze(-1)  # [N,1,J,1]
        y = y * (1.0 - jd)
        if joint_mask is not None:
            joint_mask = joint_mask * (1.0 - jd.squeeze(-1))

    # Coordinate noise
    if coord_noise_std > 0:
        noise = torch.randn_like(y[..., :3]) * float(coord_noise_std)
        y[..., :3] = y[..., :3] + noise

    # Temporal masking
    if temporal_mask_ratio > 0:
        n_mask = max(1, int(w * float(temporal_mask_ratio)))
        for i in range(n):
            if w <= 1:
                continue
            start = int(torch.randint(0, max(1, w - n_mask + 1), (1,), device=device).item())
            y[i, start : start + n_mask] = 0.0
            if joint_mask is not None:
                joint_mask[i, start : start + n_mask] = 0.0

    # Playback speed variation (resample to random length, then back).
    if speed_min > 0 and speed_max > 0 and speed_max >= speed_min:
        speed = torch.empty((1,), device=device).uniform_(float(speed_min), float(speed_max)).item()
        target_t = max(4, int(round(w / max(speed, 1e-4))))
        z = _interpolate_time(y, target_t)
        y = _interpolate_time(z, w)
        if joint_mask is not None:
            m = joint_mask.unsqueeze(-1)
            m = _interpolate_time(m, target_t)
            m = _interpolate_time(m, w)
            joint_mask = (m.squeeze(-1) > 0.5).float()

    return y, joint_mask


def info_nce(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.T) / max(float(temperature), 1e-6)
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(logits, labels)


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
    model.train()
    for p in model.motion_encoder.parameters():
        p.requires_grad = True

    proj_dim = model.motion_encoder.embedding_dim
    predictor = nn.Sequential(
        nn.Linear(proj_dim, proj_dim),
        nn.GELU(),
        nn.Linear(proj_dim, proj_dim),
    ).to(device)

    params = list(model.motion_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        n_steps = 0
        started = time.time()
        for batch in loader:
            windows = batch["motion_windows"].to(device, non_blocking=True)  # [B,S,W,J,9]
            joint_mask = batch["joint_mask"].to(device, non_blocking=True)  # [B,S,W,J]
            b, s, w, j, f = windows.shape
            flat = windows.reshape(b * s, w, j, f)
            flat_mask = joint_mask.reshape(b * s, w, j)

            v1, m1 = augment_motion_windows(flat, flat_mask.clone())
            v2, m2 = augment_motion_windows(flat, flat_mask.clone())

            use_amp = str(device).startswith("cuda")
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z1 = model.motion_encoder(v1, joint_mask=m1)
                z2 = model.motion_encoder(v2, joint_mask=m2)
                loss_contrast = info_nce(z1, z2, temperature=0.1)
                loss_consistency = F.mse_loss(z1, z2)

                # Future prediction from sequence of windows per video.
                z_seq = model.motion_encoder(flat, joint_mask=flat_mask).reshape(b, s, -1)
                if s > 1:
                    pred = predictor(z_seq[:, :-1, :])
                    tgt = z_seq[:, 1:, :].detach()
                    loss_future = F.smooth_l1_loss(pred, tgt)
                else:
                    loss_future = z_seq.new_tensor(0.0)

                loss = loss_contrast + 0.5 * loss_future + 0.5 * loss_consistency

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            n_steps += 1
            if max_steps_per_epoch and n_steps >= int(max_steps_per_epoch):
                break

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - started
        if logger is not None:
            logger.log(
                "ssl_pretrain",
                epoch=epoch,
                loss=avg_loss,
                steps=n_steps,
                elapsed_sec=round(elapsed, 2),
            )
        print(f"[SSL] epoch={epoch} loss={avg_loss:.4f} steps={n_steps} time={elapsed:.1f}s")

