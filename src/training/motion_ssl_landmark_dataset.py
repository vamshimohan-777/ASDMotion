import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


class LandmarkMotionPretrainDataset(Dataset):
    """
    Self-supervised motion dataset for pre-extracted landmark tensors.

    Expected sample shape:
      [T, J, 9] where channels are [x,y,z,vx,vy,vz,ax,ay,az].
    """

    def __init__(
        self,
        source_path,
        window_length=48,
        future_offsets=(1, 2),
        samples_per_epoch=0,
        expected_joints=DEFAULT_SCHEMA.total_joints,
        cache_enabled=True,
    ):
        self.window_length = int(window_length)
        self.future_offsets = tuple(sorted({int(v) for v in future_offsets if int(v) > 0}))
        if not self.future_offsets:
            self.future_offsets = (1,)
        self.samples_per_epoch = int(max(0, samples_per_epoch))
        self.expected_joints = int(expected_joints)
        self.cache_enabled = bool(cache_enabled)
        self._cache = {} if self.cache_enabled else None

        self.files = self._collect_files(str(source_path))
        if not self.files:
            raise ValueError(f"No motion files found from source: {source_path}")

    def _collect_files(self, source_path):
        files = []
        if os.path.isdir(source_path):
            for root, _, names in os.walk(source_path):
                for name in names:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in {".npy", ".npz", ".pt", ".pth", ".skeleton"}:
                        files.append(os.path.join(root, name))
            files.sort()
            return files

        ext = os.path.splitext(source_path)[1].lower()
        if ext == ".csv":
            base_dir = os.path.dirname(os.path.abspath(source_path))
            with open(source_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ("motion_path", "path", "file", "landmark_path", "skeleton_path"):
                        value = _safe_text(row.get(key))
                        if value:
                            full = value if os.path.isabs(value) else os.path.join(base_dir, value)
                            files.append(os.path.abspath(full))
                            break
            return [p for p in files if os.path.exists(p)]

        if ext in {".txt", ".lst"}:
            base_dir = os.path.dirname(os.path.abspath(source_path))
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = _safe_text(line)
                    if p and not p.startswith("#"):
                        full = p if os.path.isabs(p) else os.path.join(base_dir, p)
                        full = os.path.abspath(full)
                        if os.path.exists(full):
                            files.append(full)
            return files

        if os.path.exists(source_path):
            return [os.path.abspath(source_path)]
        return []

    @staticmethod
    def _parse_ntu_skeleton_file(path):
        """
        Parse NTU RGB+D `.skeleton` text format into [T, J, 3].
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip() != ""]
        if not lines:
            raise ValueError(f"Empty .skeleton file: {path}")

        p = 0
        try:
            n_frames = int(float(lines[p]))
        except Exception as exc:
            raise ValueError(f"Invalid .skeleton header: {path}") from exc
        p += 1

        frames = []
        max_joints = 0
        for _ in range(max(0, n_frames)):
            if p >= len(lines):
                break
            try:
                n_bodies = int(float(lines[p]))
            except Exception:
                break
            p += 1

            best = None
            best_valid = -1
            for _ in range(max(0, n_bodies)):
                if p >= len(lines):
                    break
                p += 1  # body-info line
                if p >= len(lines):
                    break
                try:
                    n_joints = int(float(lines[p]))
                except Exception:
                    break
                p += 1

                joints = []
                for _ in range(max(0, n_joints)):
                    if p >= len(lines):
                        break
                    vals = lines[p].split()
                    p += 1
                    if len(vals) < 3:
                        joints.append([0.0, 0.0, 0.0])
                        continue
                    try:
                        x = float(vals[0])
                        y = float(vals[1])
                        z = float(vals[2])
                    except Exception:
                        x, y, z = 0.0, 0.0, 0.0
                    joints.append([x, y, z])
                if not joints:
                    continue
                arr = np.asarray(joints, dtype=np.float32)
                valid = int((np.abs(arr).sum(axis=-1) > 1e-8).sum())
                if valid > best_valid:
                    best = arr
                    best_valid = valid

            if best is None:
                best = np.zeros((25, 3), dtype=np.float32)
            max_joints = max(max_joints, int(best.shape[0]))
            frames.append(best)

        if not frames:
            raise ValueError(f"No frames parsed from .skeleton file: {path}")

        j = max(1, max_joints)
        out = []
        for arr in frames:
            if arr.shape[0] < j:
                pad = j - arr.shape[0]
                arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
            elif arr.shape[0] > j:
                arr = arr[:j]
            out.append(arr)
        return np.stack(out, axis=0).astype(np.float32)

    def __len__(self):
        if self.samples_per_epoch > 0:
            return self.samples_per_epoch
        return len(self.files)

    @staticmethod
    def _compute_motion_features(xyz):
        vel = np.zeros_like(xyz, dtype=np.float32)
        acc = np.zeros_like(xyz, dtype=np.float32)
        if xyz.shape[0] > 1:
            vel[1:] = xyz[1:] - xyz[:-1]
            acc[1:] = vel[1:] - vel[:-1]
        return np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)

    def _normalize_shape(self, arr):
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Motion tensor must have rank 3 [T,J,C], got {x.shape}")

        t, j, c = x.shape
        if c == 3:
            x = self._compute_motion_features(x)
        elif c > 9:
            x = x[..., :9]
        elif c < 9:
            pad = np.zeros((t, j, 9 - c), dtype=np.float32)
            x = np.concatenate([x, pad], axis=-1)

        if j < self.expected_joints:
            pad_j = self.expected_joints - j
            x = np.pad(x, ((0, 0), (0, pad_j), (0, 0)), mode="constant")
        elif j > self.expected_joints:
            x = x[:, : self.expected_joints, :]
        return x.astype(np.float32)

    def _load_file(self, path):
        if self._cache is not None and path in self._cache:
            return self._cache[path]

        ext = os.path.splitext(path)[1].lower()
        arr = None
        if ext == ".npy":
            loaded = np.load(path, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                obj = loaded.item()
                if isinstance(obj, dict):
                    for key in ("motion", "motion_features", "landmarks", "keypoints", "array"):
                        if key in obj:
                            arr = obj[key]
                            break
            if arr is None:
                arr = loaded
        elif ext == ".npz":
            loaded = np.load(path, allow_pickle=True)
            for key in ("motion", "motion_features", "landmarks", "keypoints", "array"):
                if key in loaded:
                    arr = loaded[key]
                    break
            if arr is None and loaded.files:
                arr = loaded[loaded.files[0]]
        elif ext in {".pt", ".pth"}:
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                loaded = torch.load(path, map_location="cpu")
            if isinstance(loaded, dict):
                for key in ("motion", "motion_features", "landmarks", "keypoints", "array"):
                    if key in loaded:
                        arr = loaded[key]
                        break
                if arr is None:
                    for value in loaded.values():
                        if torch.is_tensor(value) or isinstance(value, np.ndarray):
                            arr = value
                            break
            else:
                arr = loaded
        elif ext == ".skeleton":
            arr = self._parse_ntu_skeleton_file(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

        if torch.is_tensor(arr):
            arr = arr.detach().cpu().float().numpy()
        seq = self._normalize_shape(arr)

        if self._cache is not None:
            self._cache[path] = seq
        return seq

    def _sample_pair(self, seq):
        t = int(seq.shape[0])
        w = self.window_length
        max_offset = int(max(self.future_offsets))
        min_frames = w + max_offset
        if t < min_frames:
            pad_t = min_frames - t
            seq = np.pad(seq, ((0, pad_t), (0, 0), (0, 0)), mode="constant")
            t = min_frames

        max_start = max(0, t - w - max_offset)
        start = random.randint(0, max_start) if max_start > 0 else 0
        horizon = random.choice(self.future_offsets)
        target_start = min(start + int(horizon), max(0, t - w))

        anchor = seq[start : start + w]
        target = seq[target_start : target_start + w]
        return anchor.astype(np.float32), target.astype(np.float32), int(horizon)

    def __getitem__(self, idx):
        if self.samples_per_epoch > 0:
            file_idx = random.randint(0, len(self.files) - 1)
        else:
            file_idx = int(idx) % len(self.files)
        path = self.files[file_idx]
        seq = self._load_file(path)
        anchor, target, horizon = self._sample_pair(seq)
        return {
            "anchor_window": torch.from_numpy(anchor),
            "future_window": torch.from_numpy(target),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "video_index": torch.tensor(file_idx, dtype=torch.long),
            "path": path,
        }


def collate_landmark_ssl_batch(batch):
    return {
        "anchor_window": torch.stack([item["anchor_window"] for item in batch], dim=0),
        "future_window": torch.stack([item["future_window"] for item in batch], dim=0),
        "horizon": torch.stack([item["horizon"] for item in batch], dim=0),
        "video_index": torch.stack([item["video_index"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }


def build_landmark_ssl_dataloader(
    dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
):
    num_workers = int(max(0, num_workers))
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": collate_landmark_ssl_batch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
