"""Training module `src/training/motion_ssl_landmark_dataset.py` that optimizes model weights and output quality."""

# Import `csv` to support computations in this stage of output generation.
import csv
# Import `os` to support computations in this stage of output generation.
import os
# Import `random` to support computations in this stage of output generation.
import random

# Import `numpy as np` to support computations in this stage of output generation.
import numpy as np
# Import `torch` to support computations in this stage of output generation.
import torch
# Import symbols from `torch.utils.data` used in this stage's output computation path.
from torch.utils.data import DataLoader, Dataset

# Import symbols from `src.models.video.mediapipe_layer.landmark_schema` used in this stage's output computation path.
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA


# Define a reusable pipeline function whose outputs feed later steps.
def _safe_text(value):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `value is None` to choose the correct output computation path.
    if value is None:
        # Return `""` as this function's contribution to downstream output flow.
        return ""
    # Return `str(value).strip()` as this function's contribution to downstream output flow.
    return str(value).strip()


# Define class `LandmarkMotionPretrainDataset` to package related logic in the prediction pipeline.
class LandmarkMotionPretrainDataset(Dataset):
    """
    Self-supervised motion dataset for pre-extracted landmark tensors.

    Expected sample shape:
      [T, J, 9] where channels are [x,y,z,vx,vy,vz,ax,ay,az].
    """

    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        source_path,
        window_length=48,
        future_offsets=(1, 2),
        samples_per_epoch=0,
        expected_joints=DEFAULT_SCHEMA.total_joints,
        cache_enabled=True,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `self.window_length` as an intermediate representation used by later output layers.
        self.window_length = int(window_length)
        # Set `self.future_offsets` for subsequent steps so gradient updates improve future predictions.
        self.future_offsets = tuple(sorted({int(v) for v in future_offsets if int(v) > 0}))
        # Branch on `not self.future_offsets` to choose the correct output computation path.
        if not self.future_offsets:
            # Set `self.future_offsets` for subsequent steps so gradient updates improve future predictions.
            self.future_offsets = (1,)
        # Compute `self.samples_per_epoch` as an intermediate representation used by later output layers.
        self.samples_per_epoch = int(max(0, samples_per_epoch))
        # Compute `self.expected_joints` as an intermediate representation used by later output layers.
        self.expected_joints = int(expected_joints)
        # Compute `self.cache_enabled` as an intermediate representation used by later output layers.
        self.cache_enabled = bool(cache_enabled)
        # Compute `self._cache` as an intermediate representation used by later output layers.
        self._cache = {} if self.cache_enabled else None

        # Set `self.files` for subsequent steps so gradient updates improve future predictions.
        self.files = self._collect_files(str(source_path))
        # Branch on `not self.files` to choose the correct output computation path.
        if not self.files:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"No motion files found from source: {source_path}")

    # Define a reusable pipeline function whose outputs feed later steps.
    def _collect_files(self, source_path):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `files` for subsequent steps so gradient updates improve future predictions.
        files = []
        # Branch on `os.path.isdir(source_path)` to choose the correct output computation path.
        if os.path.isdir(source_path):
            # Iterate over `os.walk(source_path)` so each item contributes to final outputs/metrics.
            for root, _, names in os.walk(source_path):
                # Iterate over `names` so each item contributes to final outputs/metrics.
                for name in names:
                    # Compute `ext` as an intermediate representation used by later output layers.
                    ext = os.path.splitext(name)[1].lower()
                    # Branch on `ext in {".npy", ".npz", ".pt", ".pth", ".skeleton"}` to choose the correct output computation path.
                    if ext in {".npy", ".npz", ".pt", ".pth", ".skeleton"}:
                        # Call `files.append` and use its result in later steps so gradient updates improve future predictions.
                        files.append(os.path.join(root, name))
            # Call `files.sort` and use its result in later steps so gradient updates improve future predictions.
            files.sort()
            # Return `files` as this function's contribution to downstream output flow.
            return files

        # Compute `ext` as an intermediate representation used by later output layers.
        ext = os.path.splitext(source_path)[1].lower()
        # Branch on `ext == ".csv"` to choose the correct output computation path.
        if ext == ".csv":
            # Set `base_dir` for subsequent steps so gradient updates improve future predictions.
            base_dir = os.path.dirname(os.path.abspath(source_path))
            # Use a managed context to safely handle resources used during output computation.
            with open(source_path, "r", newline="", encoding="utf-8") as f:
                # Set `reader` for subsequent steps so gradient updates improve future predictions.
                reader = csv.DictReader(f)
                # Iterate over `reader` so each item contributes to final outputs/metrics.
                for row in reader:
                    # Iterate over `("motion_path", "path", "file", "land...` so each item contributes to final outputs/metrics.
                    for key in ("motion_path", "path", "file", "landmark_path", "skeleton_path"):
                        # Set `value` for subsequent steps so gradient updates improve future predictions.
                        value = _safe_text(row.get(key))
                        # Branch on `value` to choose the correct output computation path.
                        if value:
                            # Set `full` for subsequent steps so gradient updates improve future predictions.
                            full = value if os.path.isabs(value) else os.path.join(base_dir, value)
                            # Call `files.append` and use its result in later steps so gradient updates improve future predictions.
                            files.append(os.path.abspath(full))
                            # Stop iteration early to prevent further changes to the current output state.
                            break
            # Return `[p for p in files if os.path.exists(p)]` as this function's contribution to downstream output flow.
            return [p for p in files if os.path.exists(p)]

        # Branch on `ext in {".txt", ".lst"}` to choose the correct output computation path.
        if ext in {".txt", ".lst"}:
            # Set `base_dir` for subsequent steps so gradient updates improve future predictions.
            base_dir = os.path.dirname(os.path.abspath(source_path))
            # Use a managed context to safely handle resources used during output computation.
            with open(source_path, "r", encoding="utf-8") as f:
                # Iterate over `f` so each item contributes to final outputs/metrics.
                for line in f:
                    # Set `p` for subsequent steps so gradient updates improve future predictions.
                    p = _safe_text(line)
                    # Branch on `p and not p.startswith("#")` to choose the correct output computation path.
                    if p and not p.startswith("#"):
                        # Set `full` for subsequent steps so gradient updates improve future predictions.
                        full = p if os.path.isabs(p) else os.path.join(base_dir, p)
                        # Set `full` for subsequent steps so gradient updates improve future predictions.
                        full = os.path.abspath(full)
                        # Branch on `os.path.exists(full)` to choose the correct output computation path.
                        if os.path.exists(full):
                            # Call `files.append` and use its result in later steps so gradient updates improve future predictions.
                            files.append(full)
            # Return `files` as this function's contribution to downstream output flow.
            return files

        # Branch on `os.path.exists(source_path)` to choose the correct output computation path.
        if os.path.exists(source_path):
            # Return `[os.path.abspath(source_path)]` as this function's contribution to downstream output flow.
            return [os.path.abspath(source_path)]
        # Return `[]` as this function's contribution to downstream output flow.
        return []

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def _parse_ntu_skeleton_file(path):
        """
        Parse NTU RGB+D `.skeleton` text format into [T, J, 3].
        """
        # Use a managed context to safely handle resources used during output computation.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # Call `ln.strip` and use its result in later steps so gradient updates improve future predictions.
            lines = [ln.strip() for ln in f if ln.strip() != ""]
        # Branch on `not lines` to choose the correct output computation path.
        if not lines:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Empty .skeleton file: {path}")

        # Set `p` for subsequent steps so gradient updates improve future predictions.
        p = 0
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Set `n_frames` for subsequent steps so gradient updates improve future predictions.
            n_frames = int(float(lines[p]))
        # Handle exceptions and keep output behavior controlled under error conditions.
        except Exception as exc:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Invalid .skeleton header: {path}") from exc
        # Execute this statement so gradient updates improve future predictions.
        p += 1

        # Set `frames` for subsequent steps so gradient updates improve future predictions.
        frames = []
        # Compute `max_joints` as an intermediate representation used by later output layers.
        max_joints = 0
        # Iterate over `range(max(0, n_frames))` so each item contributes to final outputs/metrics.
        for _ in range(max(0, n_frames)):
            # Branch on `p >= len(lines)` to choose the correct output computation path.
            if p >= len(lines):
                # Stop iteration early to prevent further changes to the current output state.
                break
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Set `n_bodies` for subsequent steps so gradient updates improve future predictions.
                n_bodies = int(float(lines[p]))
            # Handle exceptions and keep output behavior controlled under error conditions.
            except Exception:
                # Stop iteration early to prevent further changes to the current output state.
                break
            # Execute this statement so gradient updates improve future predictions.
            p += 1

            # Set `best` for subsequent steps so gradient updates improve future predictions.
            best = None
            # Set `best_valid` for subsequent steps so gradient updates improve future predictions.
            best_valid = -1
            # Iterate over `range(max(0, n_bodies))` so each item contributes to final outputs/metrics.
            for _ in range(max(0, n_bodies)):
                # Branch on `p >= len(lines)` to choose the correct output computation path.
                if p >= len(lines):
                    # Stop iteration early to prevent further changes to the current output state.
                    break
                # Execute this statement so gradient updates improve future predictions.
                p += 1  # body-info line
                # Branch on `p >= len(lines)` to choose the correct output computation path.
                if p >= len(lines):
                    # Stop iteration early to prevent further changes to the current output state.
                    break
                # Start guarded block so failures can be handled without breaking output flow.
                try:
                    # Set `n_joints` for subsequent steps so gradient updates improve future predictions.
                    n_joints = int(float(lines[p]))
                # Handle exceptions and keep output behavior controlled under error conditions.
                except Exception:
                    # Stop iteration early to prevent further changes to the current output state.
                    break
                # Execute this statement so gradient updates improve future predictions.
                p += 1

                # Set `joints` for subsequent steps so gradient updates improve future predictions.
                joints = []
                # Iterate over `range(max(0, n_joints))` so each item contributes to final outputs/metrics.
                for _ in range(max(0, n_joints)):
                    # Branch on `p >= len(lines)` to choose the correct output computation path.
                    if p >= len(lines):
                        # Stop iteration early to prevent further changes to the current output state.
                        break
                    # Set `vals` for subsequent steps so gradient updates improve future predictions.
                    vals = lines[p].split()
                    # Execute this statement so gradient updates improve future predictions.
                    p += 1
                    # Branch on `len(vals) < 3` to choose the correct output computation path.
                    if len(vals) < 3:
                        # Call `joints.append` and use its result in later steps so gradient updates improve future predictions.
                        joints.append([0.0, 0.0, 0.0])
                        # Skip current loop item so it does not affect accumulated output state.
                        continue
                    # Start guarded block so failures can be handled without breaking output flow.
                    try:
                        # Compute `x` as an intermediate representation used by later output layers.
                        x = float(vals[0])
                        # Set `y` for subsequent steps so gradient updates improve future predictions.
                        y = float(vals[1])
                        # Compute `z` as an intermediate representation used by later output layers.
                        z = float(vals[2])
                    # Handle exceptions and keep output behavior controlled under error conditions.
                    except Exception:
                        # Compute `x, y, z` as an intermediate representation used by later output layers.
                        x, y, z = 0.0, 0.0, 0.0
                    # Call `joints.append` and use its result in later steps so gradient updates improve future predictions.
                    joints.append([x, y, z])
                # Branch on `not joints` to choose the correct output computation path.
                if not joints:
                    # Skip current loop item so it does not affect accumulated output state.
                    continue
                # Set `arr` for subsequent steps so gradient updates improve future predictions.
                arr = np.asarray(joints, dtype=np.float32)
                # Set `valid` for subsequent steps so gradient updates improve future predictions.
                valid = int((np.abs(arr).sum(axis=-1) > 1e-8).sum())
                # Branch on `valid > best_valid` to choose the correct output computation path.
                if valid > best_valid:
                    # Set `best` for subsequent steps so gradient updates improve future predictions.
                    best = arr
                    # Set `best_valid` for subsequent steps so gradient updates improve future predictions.
                    best_valid = valid

            # Branch on `best is None` to choose the correct output computation path.
            if best is None:
                # Set `best` for subsequent steps so gradient updates improve future predictions.
                best = np.zeros((25, 3), dtype=np.float32)
            # Compute `max_joints` as an intermediate representation used by later output layers.
            max_joints = max(max_joints, int(best.shape[0]))
            # Call `frames.append` and use its result in later steps so gradient updates improve future predictions.
            frames.append(best)

        # Branch on `not frames` to choose the correct output computation path.
        if not frames:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"No frames parsed from .skeleton file: {path}")

        # Set `j` for subsequent steps so gradient updates improve future predictions.
        j = max(1, max_joints)
        # Set `out` for subsequent steps so gradient updates improve future predictions.
        out = []
        # Iterate over `frames` so each item contributes to final outputs/metrics.
        for arr in frames:
            # Branch on `arr.shape[0] < j` to choose the correct output computation path.
            if arr.shape[0] < j:
                # Set `pad` for subsequent steps so gradient updates improve future predictions.
                pad = j - arr.shape[0]
                # Set `arr` for subsequent steps so gradient updates improve future predictions.
                arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
            # Use alternate condition `arr.shape[0] > j` to refine output path selection.
            elif arr.shape[0] > j:
                # Set `arr` for subsequent steps so gradient updates improve future predictions.
                arr = arr[:j]
            # Call `out.append` and use its result in later steps so gradient updates improve future predictions.
            out.append(arr)
        # Return `np.stack(out, axis=0).astype(np.float32)` as this function's contribution to downstream output flow.
        return np.stack(out, axis=0).astype(np.float32)

    # Define a reusable pipeline function whose outputs feed later steps.
    def __len__(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.samples_per_epoch > 0` to choose the correct output computation path.
        if self.samples_per_epoch > 0:
            # Return `self.samples_per_epoch` as this function's contribution to downstream output flow.
            return self.samples_per_epoch
        # Return `len(self.files)` as this function's contribution to downstream output flow.
        return len(self.files)

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def _compute_motion_features(xyz):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `vel` for subsequent steps so gradient updates improve future predictions.
        vel = np.zeros_like(xyz, dtype=np.float32)
        # Set `acc` for subsequent steps so gradient updates improve future predictions.
        acc = np.zeros_like(xyz, dtype=np.float32)
        # Branch on `xyz.shape[0] > 1` to choose the correct output computation path.
        if xyz.shape[0] > 1:
            # Execute this statement so gradient updates improve future predictions.
            vel[1:] = xyz[1:] - xyz[:-1]
            # Execute this statement so gradient updates improve future predictions.
            acc[1:] = vel[1:] - vel[:-1]
        # Return `np.concatenate([xyz, vel, acc], axis=-1).astype(np....` as this function's contribution to downstream output flow.
        return np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _normalize_shape(self, arr):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `x` as an intermediate representation used by later output layers.
        x = np.asarray(arr, dtype=np.float32)
        # Branch on `x.ndim != 3` to choose the correct output computation path.
        if x.ndim != 3:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Motion tensor must have rank 3 [T,J,C], got {x.shape}")

        # Set `t, j, c` for subsequent steps so gradient updates improve future predictions.
        t, j, c = x.shape
        # Branch on `c == 3` to choose the correct output computation path.
        if c == 3:
            # Compute `x` as an intermediate representation used by later output layers.
            x = self._compute_motion_features(x)
        # Use alternate condition `c > 9` to refine output path selection.
        elif c > 9:
            # Compute `x` as an intermediate representation used by later output layers.
            x = x[..., :9]
        # Use alternate condition `c < 9` to refine output path selection.
        elif c < 9:
            # Set `pad` for subsequent steps so gradient updates improve future predictions.
            pad = np.zeros((t, j, 9 - c), dtype=np.float32)
            # Compute `x` as an intermediate representation used by later output layers.
            x = np.concatenate([x, pad], axis=-1)

        # Branch on `j < self.expected_joints` to choose the correct output computation path.
        if j < self.expected_joints:
            # Set `pad_j` for subsequent steps so gradient updates improve future predictions.
            pad_j = self.expected_joints - j
            # Compute `x` as an intermediate representation used by later output layers.
            x = np.pad(x, ((0, 0), (0, pad_j), (0, 0)), mode="constant")
        # Use alternate condition `j > self.expected_joints` to refine output path selection.
        elif j > self.expected_joints:
            # Compute `x` as an intermediate representation used by later output layers.
            x = x[:, : self.expected_joints, :]
        # Return `x.astype(np.float32)` as this function's contribution to downstream output flow.
        return x.astype(np.float32)

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_file(self, path):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Branch on `self._cache is not None and path in self._cache` to choose the correct output computation path.
        if self._cache is not None and path in self._cache:
            # Return `self._cache[path]` as this function's contribution to downstream output flow.
            return self._cache[path]

        # Compute `ext` as an intermediate representation used by later output layers.
        ext = os.path.splitext(path)[1].lower()
        # Set `arr` for subsequent steps so gradient updates improve future predictions.
        arr = None
        # Branch on `ext == ".npy"` to choose the correct output computation path.
        if ext == ".npy":
            # Set `loaded` for subsequent steps so gradient updates improve future predictions.
            loaded = np.load(path, allow_pickle=True)
            # Branch on `isinstance(loaded, np.ndarray) and loaded.dtype =...` to choose the correct output computation path.
            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                # Set `obj` for subsequent steps so gradient updates improve future predictions.
                obj = loaded.item()
                # Branch on `isinstance(obj, dict)` to choose the correct output computation path.
                if isinstance(obj, dict):
                    # Iterate over `("motion", "motion_features", "landma...` so each item contributes to final outputs/metrics.
                    for key in ("motion", "motion_features", "landmarks", "keypoints", "array"):
                        # Branch on `key in obj` to choose the correct output computation path.
                        if key in obj:
                            # Set `arr` for subsequent steps so gradient updates improve future predictions.
                            arr = obj[key]
                            # Stop iteration early to prevent further changes to the current output state.
                            break
            # Branch on `arr is None` to choose the correct output computation path.
            if arr is None:
                # Set `arr` for subsequent steps so gradient updates improve future predictions.
                arr = loaded
        # Use alternate condition `ext == ".npz"` to refine output path selection.
        elif ext == ".npz":
            # Set `loaded` for subsequent steps so gradient updates improve future predictions.
            loaded = np.load(path, allow_pickle=True)
            # Iterate over `("motion", "motion_features", "landma...` so each item contributes to final outputs/metrics.
            for key in ("motion", "motion_features", "landmarks", "keypoints", "array"):
                # Branch on `key in loaded` to choose the correct output computation path.
                if key in loaded:
                    # Set `arr` for subsequent steps so gradient updates improve future predictions.
                    arr = loaded[key]
                    # Stop iteration early to prevent further changes to the current output state.
                    break
            # Branch on `arr is None and loaded.files` to choose the correct output computation path.
            if arr is None and loaded.files:
                # Set `arr` for subsequent steps so gradient updates improve future predictions.
                arr = loaded[loaded.files[0]]
        # Use alternate condition `ext in {".pt", ".pth"}` to refine output path selection.
        elif ext in {".pt", ".pth"}:
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Set `loaded` for subsequent steps so gradient updates improve future predictions.
                loaded = torch.load(path, map_location="cpu", weights_only=False)
            # Handle exceptions and keep output behavior controlled under error conditions.
            except TypeError:
                # Set `loaded` for subsequent steps so gradient updates improve future predictions.
                loaded = torch.load(path, map_location="cpu")
            # Branch on `isinstance(loaded, dict)` to choose the correct output computation path.
            if isinstance(loaded, dict):
                # Iterate over `("motion", "motion_features", "landma...` so each item contributes to final outputs/metrics.
                for key in ("motion", "motion_features", "landmarks", "keypoints", "array"):
                    # Branch on `key in loaded` to choose the correct output computation path.
                    if key in loaded:
                        # Set `arr` for subsequent steps so gradient updates improve future predictions.
                        arr = loaded[key]
                        # Stop iteration early to prevent further changes to the current output state.
                        break
                # Branch on `arr is None` to choose the correct output computation path.
                if arr is None:
                    # Iterate over `loaded.values()` so each item contributes to final outputs/metrics.
                    for value in loaded.values():
                        # Branch on `torch.is_tensor(value) or isinstance(value, np.nd...` to choose the correct output computation path.
                        if torch.is_tensor(value) or isinstance(value, np.ndarray):
                            # Set `arr` for subsequent steps so gradient updates improve future predictions.
                            arr = value
                            # Stop iteration early to prevent further changes to the current output state.
                            break
            else:
                # Set `arr` for subsequent steps so gradient updates improve future predictions.
                arr = loaded
        # Use alternate condition `ext == ".skeleton"` to refine output path selection.
        elif ext == ".skeleton":
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = self._parse_ntu_skeleton_file(path)
        else:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Unsupported file extension: {path}")

        # Branch on `torch.is_tensor(arr)` to choose the correct output computation path.
        if torch.is_tensor(arr):
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = arr.detach().cpu().float().numpy()
        # Set `seq` for subsequent steps so gradient updates improve future predictions.
        seq = self._normalize_shape(arr)

        # Branch on `self._cache is not None` to choose the correct output computation path.
        if self._cache is not None:
            # Compute `self._cache[path]` as an intermediate representation used by later output layers.
            self._cache[path] = seq
        # Return `seq` as this function's contribution to downstream output flow.
        return seq

    # Define a reusable pipeline function whose outputs feed later steps.
    def _sample_pair(self, seq):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `t` for subsequent steps so gradient updates improve future predictions.
        t = int(seq.shape[0])
        # Set `w` for subsequent steps so gradient updates improve future predictions.
        w = self.window_length
        # Compute `max_offset` as an intermediate representation used by later output layers.
        max_offset = int(max(self.future_offsets))
        # Set `min_frames` for subsequent steps so gradient updates improve future predictions.
        min_frames = w + max_offset
        # Branch on `t < min_frames` to choose the correct output computation path.
        if t < min_frames:
            # Set `pad_t` for subsequent steps so gradient updates improve future predictions.
            pad_t = min_frames - t
            # Set `seq` for subsequent steps so gradient updates improve future predictions.
            seq = np.pad(seq, ((0, pad_t), (0, 0), (0, 0)), mode="constant")
            # Set `t` for subsequent steps so gradient updates improve future predictions.
            t = min_frames

        # Compute `max_start` as an intermediate representation used by later output layers.
        max_start = max(0, t - w - max_offset)
        # Set `start` for subsequent steps so gradient updates improve future predictions.
        start = random.randint(0, max_start) if max_start > 0 else 0
        # Compute `horizon` as an intermediate representation used by later output layers.
        horizon = random.choice(self.future_offsets)
        # Set `target_start` for subsequent steps so gradient updates improve future predictions.
        target_start = min(start + int(horizon), max(0, t - w))

        # Compute `anchor` as an intermediate representation used by later output layers.
        anchor = seq[start : start + w]
        # Set `target` for subsequent steps so gradient updates improve future predictions.
        target = seq[target_start : target_start + w]
        # Return `anchor.astype(np.float32), target.astype(np.float32...` as this function's contribution to downstream output flow.
        return anchor.astype(np.float32), target.astype(np.float32), int(horizon)

    # Define a reusable pipeline function whose outputs feed later steps.
    def __getitem__(self, idx):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `self.samples_per_epoch > 0` to choose the correct output computation path.
        if self.samples_per_epoch > 0:
            # Compute `file_idx` as an intermediate representation used by later output layers.
            file_idx = random.randint(0, len(self.files) - 1)
        else:
            # Compute `file_idx` as an intermediate representation used by later output layers.
            file_idx = int(idx) % len(self.files)
        # Compute `path` as an intermediate representation used by later output layers.
        path = self.files[file_idx]
        # Set `seq` for subsequent steps so gradient updates improve future predictions.
        seq = self._load_file(path)
        # Compute `anchor, target, horizon` as an intermediate representation used by later output layers.
        anchor, target, horizon = self._sample_pair(seq)
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "anchor_window": torch.from_numpy(anchor),
            "future_window": torch.from_numpy(target),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "video_index": torch.tensor(file_idx, dtype=torch.long),
            "path": path,
        }


# Define batch collation so model inputs are aligned for correct output computation.
def collate_landmark_ssl_batch(batch):
    """Packs batch items into model-ready tensors, affecting what the model sees per step."""
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "anchor_window": torch.stack([item["anchor_window"] for item in batch], dim=0),
        "future_window": torch.stack([item["future_window"] for item in batch], dim=0),
        "horizon": torch.stack([item["horizon"] for item in batch], dim=0),
        "video_index": torch.stack([item["video_index"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }


# Define loading logic for config/weights that determine runtime behavior.
def build_landmark_ssl_dataloader(
    dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
):
    """Loads configuration or weights that define how subsequent computations produce outputs."""
    # Set `num_workers` for subsequent steps so gradient updates improve future predictions.
    num_workers = int(max(0, num_workers))
    # Set `kwargs` for subsequent steps so gradient updates improve future predictions.
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": collate_landmark_ssl_batch,
    }
    # Branch on `num_workers > 0` to choose the correct output computation path.
    if num_workers > 0:
        # Execute this statement so gradient updates improve future predictions.
        kwargs["persistent_workers"] = True
        # Execute this statement so gradient updates improve future predictions.
        kwargs["prefetch_factor"] = 2
    # Return `DataLoader(dataset, **kwargs)` as this function's contribution to downstream output flow.
    return DataLoader(dataset, **kwargs)
