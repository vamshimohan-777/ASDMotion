"""Landmark normalization and motion-feature construction."""

from __future__ import annotations

import numpy as np


def moving_average_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    # No-op for trivial kernel sizes.
    if int(k) <= 1:
        return x
    k = int(max(1, k))
    # Use odd kernel so smoothing is centered.
    if k % 2 == 0:
        k += 1
    pad = k // 2
    # Uniform smoothing kernel.
    kernel = np.ones((k,), dtype=np.float32)
    kernel /= max(float(kernel.sum()), 1e-8)
    # Edge padding avoids shrinking the output sequence.
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def fill_missing_1d(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Copy to avoid mutating caller buffers.
    out = values.copy()
    # If nothing is valid, keep original signal.
    if valid.sum() == 0:
        return out
    idx = np.where(valid > 0.5)[0]
    first = int(idx[0])
    last = int(idx[-1])
    # Extend nearest valid values to both sequence ends.
    out[:first] = out[first]
    out[last + 1 :] = out[last]
    miss = np.where(valid < 0.5)[0]
    for m in miss:
        left = idx[idx < m]
        right = idx[idx > m]
        # Interior gaps are linearly interpolated between nearest valid samples.
        if left.size == 0 and right.size == 0:
            continue
        if left.size == 0:
            out[m] = out[right[0]]
            continue
        if right.size == 0:
            out[m] = out[left[-1]]
            continue
        l = int(left[-1])
        r = int(right[0])
        alpha = float((m - l) / max(r - l, 1))
        out[m] = (1.0 - alpha) * out[l] + alpha * out[r]
    return out


def normalize_landmarks(
    landmarks: np.ndarray,
    mask: np.ndarray,
    smooth_kernel: int = 5,
    left_hip_idx: int = 23,
    right_hip_idx: int = 24,
    left_shoulder_idx: int = 11,
    right_shoulder_idx: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    # Canonicalize dtype/layout; operate on copies for functional behavior.
    xyz = np.asarray(landmarks, dtype=np.float32).copy()
    m = np.asarray(mask, dtype=np.float32).copy()
    # Input validation keeps downstream geometry operations safe.
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"landmarks must be [T,J,3], got {tuple(xyz.shape)}")
    if m.shape != xyz.shape[:2]:
        raise ValueError(f"mask must be [T,J], got {tuple(m.shape)} for landmarks {tuple(xyz.shape)}")
    t, j, _ = xyz.shape

    # Fill missing per-joint coordinates over time.
    for jj in range(j):
        valid = m[:, jj] > 0.5
        for c in range(3):
            xyz[:, jj, c] = fill_missing_1d(xyz[:, jj, c], valid)

    # Translation normalization: center body around hip midpoint.
    hips_valid = (m[:, left_hip_idx] > 0.5) & (m[:, right_hip_idx] > 0.5)
    center = np.zeros((t, 3), dtype=np.float32)
    center[hips_valid] = 0.5 * (xyz[hips_valid, left_hip_idx] + xyz[hips_valid, right_hip_idx])
    if hips_valid.any():
        last = center[np.where(hips_valid)[0][0]].copy()
    else:
        last = np.zeros((3,), dtype=np.float32)
    for i in range(t):
        if hips_valid[i]:
            last = center[i]
        else:
            center[i] = last
    xyz = xyz - center[:, None, :]

    # Scale normalization: shoulder distance approximates body scale.
    sh_valid = (m[:, left_shoulder_idx] > 0.5) & (m[:, right_shoulder_idx] > 0.5)
    scale = np.ones((t,), dtype=np.float32)
    if sh_valid.any():
        d = np.linalg.norm(xyz[:, left_shoulder_idx] - xyz[:, right_shoulder_idx], axis=-1)
        d = np.clip(d, 1e-4, None)
        scale[sh_valid] = d[sh_valid]
        scale[~sh_valid] = float(np.median(scale[sh_valid]))
    xyz = xyz / scale[:, None, None]

    # Smooth valid points; preserve imputed/invalid positions as computed.
    for jj in range(j):
        valid = m[:, jj]
        for c in range(3):
            sm = moving_average_1d(xyz[:, jj, c], k=smooth_kernel)
            xyz[:, jj, c] = (valid * sm) + ((1.0 - valid) * xyz[:, jj, c])

    return xyz.astype(np.float32), m.astype(np.float32)


def build_motion_features(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # First- and second-order temporal derivatives.
    vel = np.zeros_like(xyz, dtype=np.float32)
    acc = np.zeros_like(xyz, dtype=np.float32)
    if xyz.shape[0] > 1:
        vel[1:] = xyz[1:] - xyz[:-1]
        acc[1:] = vel[1:] - vel[:-1]
    # Feature layout: [x,y,z, vx,vy,vz, ax,ay,az].
    feat = np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)
    # Zero-out invalid joints/frames.
    feat *= mask[..., None]
    return feat
