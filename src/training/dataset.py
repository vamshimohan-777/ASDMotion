"""Training module `src/training/dataset.py` that optimizes model weights and output quality."""

# Import `csv` to support computations in this stage of output generation.
import csv
# Import `json` to support computations in this stage of output generation.
import json
# Import `os` to support computations in this stage of output generation.
import os
# Import `random` to support computations in this stage of output generation.
import random
# Import `urllib.parse` to support computations in this stage of output generation.
import urllib.parse
# Import `urllib.request` to support computations in this stage of output generation.
import urllib.request

# Import `cv2` to support computations in this stage of output generation.
import cv2
# Import `numpy as np` to support computations in this stage of output generation.
import numpy as np
# Import `torch` to support computations in this stage of output generation.
import torch
# Import symbols from `torch.utils.data` used in this stage's output computation path.
from torch.utils.data import Dataset

# Import symbols from `src.models.video.mediapipe_layer.landmark_schema` used in this stage's output computation path.
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
# Import symbols from `src.models.video.motion.features` used in this stage's output computation path.
from src.models.video.motion.features import build_motion_features, normalize_landmarks
# Import symbols from `src.pipeline.preprocess` used in this stage's output computation path.
from src.pipeline.preprocess import VideoProcessor
# Import symbols from `src.utils.video_id` used in this stage's output computation path.
from src.utils.video_id import make_video_id


# Define a reusable pipeline function whose outputs feed later steps.
def _safe_text(value):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `value is None` to choose the correct output computation path.
    if value is None:
        # Return `""` as this function's contribution to downstream output flow.
        return ""
    # Return `str(value).strip()` as this function's contribution to downstream output flow.
    return str(value).strip()


# Define a reusable pipeline function whose outputs feed later steps.
def _moving_average_1d(x, k=5):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `k <= 1` to choose the correct output computation path.
    if k <= 1:
        # Return `x` as this function's contribution to downstream output flow.
        return x
    # Set `k` for subsequent steps so gradient updates improve future predictions.
    k = int(max(1, k))
    # Branch on `k % 2 == 0` to choose the correct output computation path.
    if k % 2 == 0:
        # Execute this statement so gradient updates improve future predictions.
        k += 1
    # Set `pad` for subsequent steps so gradient updates improve future predictions.
    pad = k // 2
    # Set `kernel` for subsequent steps so gradient updates improve future predictions.
    kernel = np.ones((k,), dtype=np.float32)
    # Call `kernel.sum` and use its result in later steps so gradient updates improve future predictions.
    kernel /= kernel.sum()
    # Compute `xp` as an intermediate representation used by later output layers.
    xp = np.pad(x, (pad, pad), mode="edge")
    # Return `np.convolve(xp, kernel, mode="valid")` as this function's contribution to downstream output flow.
    return np.convolve(xp, kernel, mode="valid")


# Define a reusable pipeline function whose outputs feed later steps.
def _is_http_link(text):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `t` for subsequent steps so gradient updates improve future predictions.
    t = _safe_text(text).lower()
    # Return `t.startswith("http://") or t.startswith("https://")` as this function's contribution to downstream output flow.
    return t.startswith("http://") or t.startswith("https://")


# Define a reusable pipeline function whose outputs feed later steps.
def _fill_missing_1d(values, valid):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `out` for subsequent steps so gradient updates improve future predictions.
    out = values.copy()
    # Branch on `valid.sum() == 0` to choose the correct output computation path.
    if valid.sum() == 0:
        # Return `out` as this function's contribution to downstream output flow.
        return out

    # Compute `idx` as an intermediate representation used by later output layers.
    idx = np.where(valid > 0.5)[0]
    # Set `first` for subsequent steps so gradient updates improve future predictions.
    first = idx[0]
    # Set `last` for subsequent steps so gradient updates improve future predictions.
    last = idx[-1]
    # Execute this statement so gradient updates improve future predictions.
    out[:first] = out[first]
    # Execute this statement so gradient updates improve future predictions.
    out[last + 1:] = out[last]
    # Set `miss` for subsequent steps so gradient updates improve future predictions.
    miss = np.where(valid < 0.5)[0]
    # Iterate over `miss` so each item contributes to final outputs/metrics.
    for m in miss:
        # Set `left` for subsequent steps so gradient updates improve future predictions.
        left = idx[idx < m]
        # Compute `right` as an intermediate representation used by later output layers.
        right = idx[idx > m]
        # Branch on `left.size == 0 and right.size == 0` to choose the correct output computation path.
        if left.size == 0 and right.size == 0:
            # Skip current loop item so it does not affect accumulated output state.
            continue
        # Branch on `left.size == 0` to choose the correct output computation path.
        if left.size == 0:
            # Set `out[m]` for subsequent steps so gradient updates improve future predictions.
            out[m] = out[right[0]]
        # Use alternate condition `right.size == 0` to refine output path selection.
        elif right.size == 0:
            # Set `out[m]` for subsequent steps so gradient updates improve future predictions.
            out[m] = out[left[-1]]
        else:
            # Set `l` for subsequent steps so gradient updates improve future predictions.
            l = left[-1]
            # Set `r` for subsequent steps so gradient updates improve future predictions.
            r = right[0]
            # Compute `alpha` as an intermediate representation used by later output layers.
            alpha = (m - l) / max((r - l), 1)
            # Set `out[m]` for subsequent steps so gradient updates improve future predictions.
            out[m] = (1.0 - alpha) * out[l] + alpha * out[r]
    # Return `out` as this function's contribution to downstream output flow.
    return out


# Define class `VideoDataset` to package related logic in the prediction pipeline.
class VideoDataset(Dataset):
    """
    Landmark-first ASD dataset.

    Each item is a set of temporal windows from one video:
    - motion_windows: [S, W, J, 9]
    - joint_mask: [S, W, J]
    """

    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        csv_path,
        sequence_length=64,
        is_training=False,
        require_label=True,
        use_preprocessed=True,
        processed_root="data/processed",
        window_sizes=(32, 48, 64),
        windows_per_video=8,
        eval_windows_per_video=12,
        frame_stride=1,
        max_frames=0,
        cache_enabled=True,
        smooth_kernel=5,
        use_rgb=False,
        pose_only=False,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `self.sequence_length` as an intermediate representation used by later output layers.
        self.sequence_length = int(sequence_length)
        # Set `self.is_training` for subsequent steps so gradient updates improve future predictions.
        self.is_training = bool(is_training)
        # Set `self.require_label` for subsequent steps so gradient updates improve future predictions.
        self.require_label = bool(require_label)
        # Set `self.use_preprocessed` for subsequent steps so gradient updates improve future predictions.
        self.use_preprocessed = bool(use_preprocessed)
        # Set `self.processed_root` for subsequent steps so gradient updates improve future predictions.
        self.processed_root = str(processed_root)
        # Compute `self.window_sizes` as an intermediate representation used by later output layers.
        self.window_sizes = tuple(sorted({int(v) for v in window_sizes if int(v) > 0}))
        # Branch on `not self.window_sizes` to choose the correct output computation path.
        if not self.window_sizes:
            # Compute `self.window_sizes` as an intermediate representation used by later output layers.
            self.window_sizes = (self.sequence_length,)
        # Compute `self.windows_per_video` as an intermediate representation used by later output layers.
        self.windows_per_video = int(max(1, windows_per_video))
        # Compute `self.eval_windows_per_video` as an intermediate representation used by later output layers.
        self.eval_windows_per_video = int(max(1, eval_windows_per_video))
        # Set `self.frame_stride` for subsequent steps so gradient updates improve future predictions.
        self.frame_stride = int(frame_stride)
        # Compute `self.max_frames` as an intermediate representation used by later output layers.
        self.max_frames = int(max_frames)
        # Compute `self.cache_enabled` as an intermediate representation used by later output layers.
        self.cache_enabled = bool(cache_enabled)
        # Compute `self.smooth_kernel` as an intermediate representation used by later output layers.
        self.smooth_kernel = int(max(1, smooth_kernel))
        self.use_rgb = bool(use_rgb)
        self.pose_only = bool(pose_only)
        # Compute `self.schema` as an intermediate representation used by later output layers.
        self.schema = DEFAULT_SCHEMA
        # Set `self.processor` for subsequent steps so gradient updates improve future predictions.
        self.processor = VideoProcessor(
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
            schema=self.schema,
        )
        # Compute `self.external_cache_dir` as an intermediate representation used by later output layers.
        self.external_cache_dir = os.path.join(self.processed_root, "_external_cache")

        # Set `self.entries` for subsequent steps so gradient updates improve future predictions.
        self.entries = []
        # Use a managed context to safely handle resources used during output computation.
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            # Set `reader` for subsequent steps so gradient updates improve future predictions.
            reader = csv.DictReader(f)
            # Set `fieldnames` for subsequent steps so gradient updates improve future predictions.
            fieldnames = reader.fieldnames or []
            # Branch on `self.require_label and "subject_id" not in fieldn...` to choose the correct output computation path.
            if self.require_label and "subject_id" not in fieldnames:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError("CSV must contain subject_id for grouped validation.")
            # Branch on `self.require_label and "label" not in reader.fiel...` to choose the correct output computation path.
            if self.require_label and "label" not in reader.fieldnames:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError("CSV must contain label for supervised ASD training.")
            # Iterate over `enumerate(reader, start=1)` so each item contributes to final outputs/metrics.
            for row_idx, row in enumerate(reader, start=1):
                # Set `action_type` for subsequent steps so gradient updates improve future predictions.
                action_type = self._parse_action_type(row)
                # Set `skeleton_source` for subsequent steps so gradient updates improve future predictions.
                skeleton_source = self._parse_skeleton_source(row)
                # Set `label` for subsequent steps so gradient updates improve future predictions.
                label = self._parse_label(row)
                # Compute `video_path` as an intermediate representation used by later output layers.
                video_path = _safe_text(row.get("video_path"))
                # Branch on `not video_path` to choose the correct output computation path.
                if not video_path:
                    # Compute `video_path` as an intermediate representation used by later output layers.
                    video_path = skeleton_source or f"skeleton_sample_{row_idx:06d}"
                # Set `subject_id` for subsequent steps so gradient updates improve future predictions.
                subject_id = _safe_text(row.get("subject_id"))
                # Branch on `not subject_id` to choose the correct output computation path.
                if not subject_id:
                    # Branch on `self.require_label` to choose the correct output computation path.
                    if self.require_label:
                        # Raise explicit error to stop invalid state from producing misleading outputs.
                        raise ValueError(
                            f"Missing subject_id in row {row_idx} while require_label=true."
                        )
                    # Set `subject_id` for subsequent steps so gradient updates improve future predictions.
                    subject_id = f"subject_{row_idx:06d}"
                # Branch on `not skeleton_source and not video_path` to choose the correct output computation path.
                if not skeleton_source and not video_path:
                    # Raise explicit error to stop invalid state from producing misleading outputs.
                    raise ValueError(
                        f"Row {row_idx} must contain either video_path or skeleton source column."
                    )
                # Call `self.entries.append` and use its result in later steps so gradient updates improve future predictions.
                self.entries.append(
                    {
                        "video_path": video_path,
                        "label": label,
                        "subject_id": subject_id,
                        "action_type": action_type,
                        "skeleton_source": skeleton_source,
                    }
                )

        # Set `self.action_to_id` for subsequent steps so gradient updates improve future predictions.
        self.action_to_id = {}
        # Set `self.id_to_action` for subsequent steps so gradient updates improve future predictions.
        self.id_to_action = []
        # Iterate over `self.entries` so each item contributes to final outputs/metrics.
        for entry in self.entries:
            # Set `action` for subsequent steps so gradient updates improve future predictions.
            action = _safe_text(entry.get("action_type"))
            # Branch on `action and action not in self.action_to_id` to choose the correct output computation path.
            if action and action not in self.action_to_id:
                # Set `self.action_to_id[action]` for subsequent steps so gradient updates improve future predictions.
                self.action_to_id[action] = len(self.id_to_action)
                # Call `self.id_to_action.append` and use its result in later steps so gradient updates improve future predictions.
                self.id_to_action.append(action)
        # Set `self.num_action_classes` for subsequent steps so gradient updates improve future predictions.
        self.num_action_classes = int(len(self.id_to_action))

        # Compute `self._cache` as an intermediate representation used by later output layers.
        self._cache = {} if self.cache_enabled else None

    # Define a reusable pipeline function whose outputs feed later steps.
    def __len__(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `len(self.entries)` as this function's contribution to downstream output flow.
        return len(self.entries)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _candidate_ids(self, entry):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `video_path` as an intermediate representation used by later output layers.
        video_path = entry["video_path"]
        # Set `subject_id` for subsequent steps so gradient updates improve future predictions.
        subject_id = entry.get("subject_id")
        # Set `label` for subsequent steps so gradient updates improve future predictions.
        label = entry.get("label")
        # Set `ids` for subsequent steps so gradient updates improve future predictions.
        ids = []
        # Call `ids.append` and use its result in later steps so gradient updates improve future predictions.
        ids.append(make_video_id(video_path, subject_id=subject_id, label=label))
        # Set `prev` for subsequent steps so gradient updates improve future predictions.
        prev = make_video_id(video_path, subject_id=subject_id)
        # Branch on `prev not in ids` to choose the correct output computation path.
        if prev not in ids:
            # Call `ids.append` and use its result in later steps so gradient updates improve future predictions.
            ids.append(prev)
        # Set `legacy` for subsequent steps so gradient updates improve future predictions.
        legacy = make_video_id(video_path)
        # Branch on `legacy not in ids` to choose the correct output computation path.
        if legacy not in ids:
            # Call `ids.append` and use its result in later steps so gradient updates improve future predictions.
            ids.append(legacy)
        # Return `ids` as this function's contribution to downstream output flow.
        return ids

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def _parse_action_type(row):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Iterate over `("action_type", "action", "activity",...` so each item contributes to final outputs/metrics.
        for key in ("action_type", "action", "activity", "gesture", "motion_type"):
            # Set `value` for subsequent steps so gradient updates improve future predictions.
            value = _safe_text(row.get(key))
            # Branch on `value` to choose the correct output computation path.
            if value:
                # Return `value` as this function's contribution to downstream output flow.
                return value
        # Return `""` as this function's contribution to downstream output flow.
        return ""

    # Define a reusable pipeline function whose outputs feed later steps.
    def _parse_label(self, row):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `text` as an intermediate representation used by later output layers.
        text = _safe_text(row.get("label"))
        # Branch on `text == ""` to choose the correct output computation path.
        if text == "":
            # Branch on `self.require_label` to choose the correct output computation path.
            if self.require_label:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError("Missing label in CSV row while require_label=true.")
            # Return `0.0` as this function's contribution to downstream output flow.
            return 0.0
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Return `float(text)` as this function's contribution to downstream output flow.
            return float(text)
        # Handle exceptions and keep output behavior controlled under error conditions.
        except Exception:
            # Branch on `self.require_label` to choose the correct output computation path.
            if self.require_label:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError(f"Invalid label value: {text}")
            # Return `0.0` as this function's contribution to downstream output flow.
            return 0.0

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def _parse_skeleton_source(row):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `keys` for subsequent steps so gradient updates improve future predictions.
        keys = (
            "skeleton_path",
            "skeleton_file",
            "skeleton_link",
            "skeleton_url",
            "skeleton",
            "landmark_path",
            "landmark_file",
            "landmarks_npy",
            "skeleton_datafile",
            "skeleton_data_link",
            "link",
            "url",
            "path",
        )
        # Iterate over `keys` so each item contributes to final outputs/metrics.
        for key in keys:
            # Set `value` for subsequent steps so gradient updates improve future predictions.
            value = _safe_text(row.get(key))
            # Branch on `value` to choose the correct output computation path.
            if value:
                # Return `value` as this function's contribution to downstream output flow.
                return value
        # Return `""` as this function's contribution to downstream output flow.
        return ""

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def _parse_ntu_skeleton_file(path):
        """
        Parse common NTU RGB+D `.skeleton` text format.
        Returns landmarks [T,J,3], mask [T,J], timestamps [T].
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
        except Exception as e:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Invalid .skeleton header: {path}") from e
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
                p += 1  # skip body-info line
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
        # Set `landmarks` for subsequent steps so gradient updates improve future predictions.
        landmarks = np.stack(out, axis=0).astype(np.float32)  # [T,J,3]
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = (np.abs(landmarks).sum(axis=-1) > 1e-8).astype(np.float32)
        # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
        timestamps = np.arange(landmarks.shape[0], dtype=np.float32)
        # Return `landmarks, mask, timestamps` as this function's contribution to downstream output flow.
        return landmarks, mask, timestamps

    # Define loading logic for config/weights that determine runtime behavior.
    def _download_external_file(self, url):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
        os.makedirs(self.external_cache_dir, exist_ok=True)
        # Set `parsed` for subsequent steps so gradient updates improve future predictions.
        parsed = urllib.parse.urlparse(url)
        # Set `name` for subsequent steps so gradient updates improve future predictions.
        name = os.path.basename(parsed.path) or "skeleton_data.bin"
        # Set `safe_name` for subsequent steps so gradient updates improve future predictions.
        safe_name = name.replace("?", "_").replace("&", "_").replace("=", "_")
        # Compute `local_path` as an intermediate representation used by later output layers.
        local_path = os.path.join(self.external_cache_dir, safe_name)
        # Branch on `os.path.exists(local_path) and os.path.getsize(lo...` to choose the correct output computation path.
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            # Return `local_path` as this function's contribution to downstream output flow.
            return local_path
        # Use a managed context to safely handle resources used during output computation.
        with urllib.request.urlopen(url, timeout=30) as resp:
            # Set `data` for subsequent steps so gradient updates improve future predictions.
            data = resp.read()
        # Use a managed context to safely handle resources used during output computation.
        with open(local_path, "wb") as f:
            # Call `f.write` and use its result in later steps so gradient updates improve future predictions.
            f.write(data)
        # Return `local_path` as this function's contribution to downstream output flow.
        return local_path

    # Define a reusable pipeline function whose outputs feed later steps.
    def _normalize_skeleton_arrays(self, landmarks, mask=None, timestamps=None):
        # landmarks expected [T, J, C]
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `arr` for subsequent steps so gradient updates improve future predictions.
        arr = np.asarray(landmarks, dtype=np.float32)
        # Branch on `arr.ndim == 2` to choose the correct output computation path.
        if arr.ndim == 2:
            # [T, J*3] format
            # Set `t, d` for subsequent steps so gradient updates improve future predictions.
            t, d = arr.shape
            # Branch on `d % 3 != 0` to choose the correct output computation path.
            if d % 3 != 0:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError(f"Unsupported 2D skeleton shape: {arr.shape}")
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = arr.reshape(t, d // 3, 3)
        # Branch on `arr.ndim != 3` to choose the correct output computation path.
        if arr.ndim != 3:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Expected skeleton array rank 3, got shape={arr.shape}")

        # Set `t, j, c` for subsequent steps so gradient updates improve future predictions.
        t, j, c = arr.shape
        # Branch on `c == 2` to choose the correct output computation path.
        if c == 2:
            # Compute `z` as an intermediate representation used by later output layers.
            z = np.zeros((t, j, 1), dtype=np.float32)
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = np.concatenate([arr, z], axis=-1)
        # Use alternate condition `c >= 3` to refine output path selection.
        elif c >= 3:
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = arr[..., :3]
        else:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError(f"Unsupported channel count: {c}")

        # Set `target_j` for subsequent steps so gradient updates improve future predictions.
        target_j = int(self.schema.total_joints)
        # Branch on `j < target_j` to choose the correct output computation path.
        if j < target_j:
            # Set `pad_j` for subsequent steps so gradient updates improve future predictions.
            pad_j = target_j - j
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = np.pad(arr, ((0, 0), (0, pad_j), (0, 0)), mode="constant")
            # Branch on `mask is not None` to choose the correct output computation path.
            if mask is not None:
                # Build `mask` to gate invalid timesteps/joints from influencing outputs.
                mask = np.pad(np.asarray(mask, dtype=np.float32), ((0, 0), (0, pad_j)), mode="constant")
        # Use alternate condition `j > target_j` to refine output path selection.
        elif j > target_j:
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = arr[:, :target_j, :]
            # Branch on `mask is not None` to choose the correct output computation path.
            if mask is not None:
                # Build `mask` to gate invalid timesteps/joints from influencing outputs.
                mask = np.asarray(mask, dtype=np.float32)[:, :target_j]

        # Branch on `mask is None` to choose the correct output computation path.
        if mask is None:
            # Valid if any non-zero coordinate present.
            # Build `mask` to gate invalid timesteps/joints from influencing outputs.
            mask = (np.abs(arr).sum(axis=-1) > 1e-8).astype(np.float32)
        else:
            # Build `mask` to gate invalid timesteps/joints from influencing outputs.
            mask = np.asarray(mask, dtype=np.float32)
            # Branch on `mask.ndim != 2` to choose the correct output computation path.
            if mask.ndim != 2:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError(f"mask must be [T,J], got shape={mask.shape}")
            # Branch on `mask.shape[0] != arr.shape[0]` to choose the correct output computation path.
            if mask.shape[0] != arr.shape[0]:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError("mask/time dimension mismatch with landmarks")
            # Branch on `mask.shape[1] < target_j` to choose the correct output computation path.
            if mask.shape[1] < target_j:
                # Build `pad_j` to gate invalid timesteps/joints from influencing outputs.
                pad_j = target_j - mask.shape[1]
                # Build `mask` to gate invalid timesteps/joints from influencing outputs.
                mask = np.pad(mask, ((0, 0), (0, pad_j)), mode="constant")
            # Use alternate condition `mask.shape[1] > target_j` to refine output path selection.
            elif mask.shape[1] > target_j:
                # Build `mask` to gate invalid timesteps/joints from influencing outputs.
                mask = mask[:, :target_j]
            # Build `mask` to gate invalid timesteps/joints from influencing outputs.
            mask = (mask > 0.5).astype(np.float32)

        # Branch on `timestamps is None` to choose the correct output computation path.
        if timestamps is None:
            # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
            timestamps = np.arange(arr.shape[0], dtype=np.float32)
        else:
            # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
            timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
            # Branch on `timestamps.shape[0] != arr.shape[0]` to choose the correct output computation path.
            if timestamps.shape[0] != arr.shape[0]:
                # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
                timestamps = np.arange(arr.shape[0], dtype=np.float32)

        # Return `arr.astype(np.float32), mask.astype(np.float32), ti...` as this function's contribution to downstream output flow.
        return arr.astype(np.float32), mask.astype(np.float32), timestamps.astype(np.float32)

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_skeleton_file(self, path):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Compute `ext` as an intermediate representation used by later output layers.
        ext = os.path.splitext(path)[1].lower()
        # Set `quality` for subsequent steps so gradient updates improve future predictions.
        quality = None

        # Branch on `ext == ".npz"` to choose the correct output computation path.
        if ext == ".npz":
            # Set `data` for subsequent steps so gradient updates improve future predictions.
            data = np.load(path, allow_pickle=True)
            # Set `keys` for subsequent steps so gradient updates improve future predictions.
            keys = set(data.files)
            # Set `lmk_key` for subsequent steps so gradient updates improve future predictions.
            lmk_key = "landmarks" if "landmarks" in keys else ("keypoints" if "keypoints" in keys else None)
            # Branch on `lmk_key is None` to choose the correct output computation path.
            if lmk_key is None:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError(f"No landmarks/keypoints key in npz: {path}")
            # Set `landmarks` for subsequent steps so gradient updates improve future predictions.
            landmarks = data[lmk_key]
            # Build `mask` to gate invalid timesteps/joints from influencing outputs.
            mask = data["mask"] if "mask" in keys else None
            # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
            timestamps = data["timestamps"] if "timestamps" in keys else None
            # Build `arr, m, ts` to gate invalid timesteps/joints from influencing outputs.
            arr, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
            # Return `{"landmarks": arr, "mask": m, "timestamps": ts, "qu...` as this function's contribution to downstream output flow.
            return {"landmarks": arr, "mask": m, "timestamps": ts, "quality": quality}

        # Branch on `ext == ".npy"` to choose the correct output computation path.
        if ext == ".npy":
            # Set `arr` for subsequent steps so gradient updates improve future predictions.
            arr = np.load(path, allow_pickle=True)
            # Branch on `isinstance(arr, np.ndarray) and arr.dtype == obje...` to choose the correct output computation path.
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
                # Set `obj` for subsequent steps so gradient updates improve future predictions.
                obj = arr.item()
                # Branch on `isinstance(obj, dict)` to choose the correct output computation path.
                if isinstance(obj, dict):
                    # Set `landmarks` for subsequent steps so gradient updates improve future predictions.
                    landmarks = obj.get("landmarks", obj.get("keypoints"))
                    # Build `mask` to gate invalid timesteps/joints from influencing outputs.
                    mask = obj.get("mask")
                    # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
                    timestamps = obj.get("timestamps")
                    # Branch on `landmarks is None` to choose the correct output computation path.
                    if landmarks is None:
                        # Raise explicit error to stop invalid state from producing misleading outputs.
                        raise ValueError(f"No landmarks/keypoints in npy dict: {path}")
                    # Build `arr3, m, ts` to gate invalid timesteps/joints from influencing outputs.
                    arr3, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
                    # Return `{"landmarks": arr3, "mask": m, "timestamps": ts, "q...` as this function's contribution to downstream output flow.
                    return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}
            # Build `arr3, m, ts` to gate invalid timesteps/joints from influencing outputs.
            arr3, m, ts = self._normalize_skeleton_arrays(arr, mask=None, timestamps=None)
            # Return `{"landmarks": arr3, "mask": m, "timestamps": ts, "q...` as this function's contribution to downstream output flow.
            return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}

        # Branch on `ext == ".json"` to choose the correct output computation path.
        if ext == ".json":
            # Use a managed context to safely handle resources used during output computation.
            with open(path, "r", encoding="utf-8") as f:
                # Set `obj` for subsequent steps so gradient updates improve future predictions.
                obj = json.load(f)
            # Set `landmarks` for subsequent steps so gradient updates improve future predictions.
            landmarks = obj.get("landmarks", obj.get("keypoints"))
            # Build `mask` to gate invalid timesteps/joints from influencing outputs.
            mask = obj.get("mask")
            # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
            timestamps = obj.get("timestamps")
            # Set `quality` for subsequent steps so gradient updates improve future predictions.
            quality = obj.get("quality")
            # Branch on `landmarks is None` to choose the correct output computation path.
            if landmarks is None:
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise ValueError(f"No landmarks/keypoints in json: {path}")
            # Build `arr3, m, ts` to gate invalid timesteps/joints from influencing outputs.
            arr3, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
            # Return `{"landmarks": arr3, "mask": m, "timestamps": ts, "q...` as this function's contribution to downstream output flow.
            return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}

        # Branch on `ext == ".skeleton"` to choose the correct output computation path.
        if ext == ".skeleton":
            # Build `landmarks, mask, timestamps` to gate invalid timesteps/joints from influencing outputs.
            landmarks, mask, timestamps = self._parse_ntu_skeleton_file(path)
            # Build `arr3, m, ts` to gate invalid timesteps/joints from influencing outputs.
            arr3, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
            # Return `{"landmarks": arr3, "mask": m, "timestamps": ts, "q...` as this function's contribution to downstream output flow.
            return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}

        # Raise explicit error to stop invalid state from producing misleading outputs.
        raise ValueError(f"Unsupported skeleton file extension: {ext}")

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_from_skeleton_source(self, entry):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Set `source` for subsequent steps so gradient updates improve future predictions.
        source = _safe_text(entry.get("skeleton_source"))
        # Branch on `not source` to choose the correct output computation path.
        if not source:
            # Return `None` as this function's contribution to downstream output flow.
            return None
        # Compute `path` as an intermediate representation used by later output layers.
        path = source
        # Branch on `_is_http_link(source)` to choose the correct output computation path.
        if _is_http_link(source):
            # Compute `path` as an intermediate representation used by later output layers.
            path = self._download_external_file(source)
        # Use alternate condition `not os.path.isabs(path)` to refine output path selection.
        elif not os.path.isabs(path):
            # Resolve relative paths from repo root.
            # Compute `path` as an intermediate representation used by later output layers.
            path = os.path.abspath(path)
        # Branch on `not os.path.exists(path)` to choose the correct output computation path.
        if not os.path.exists(path):
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise FileNotFoundError(f"Skeleton source not found: {source}")
        # Return `self._load_skeleton_file(path)` as this function's contribution to downstream output flow.
        return self._load_skeleton_file(path)

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_preprocessed(self, entry):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Set `base_dir` for subsequent steps so gradient updates improve future predictions.
        base_dir = None
        # Iterate over `self._candidate_ids(entry)` so each item contributes to final outputs/metrics.
        for vid in self._candidate_ids(entry):
            # Set `candidate` for subsequent steps so gradient updates improve future predictions.
            candidate = os.path.join(self.processed_root, vid)
            # Branch on `os.path.exists(os.path.join(candidate, "landmarks...` to choose the correct output computation path.
            if os.path.exists(os.path.join(candidate, "landmarks.npy")):
                # Set `base_dir` for subsequent steps so gradient updates improve future predictions.
                base_dir = candidate
                # Stop iteration early to prevent further changes to the current output state.
                break
        # Branch on `base_dir is None` to choose the correct output computation path.
        if base_dir is None:
            # Return `None` as this function's contribution to downstream output flow.
            return None

        # Compute `landmarks_path` as an intermediate representation used by later output layers.
        landmarks_path = os.path.join(base_dir, "landmarks.npy")
        # Build `mask_path` to gate invalid timesteps/joints from influencing outputs.
        mask_path = os.path.join(base_dir, "landmark_mask.npy")
        # Compute `timestamps_path` as an intermediate representation used by later output layers.
        timestamps_path = os.path.join(base_dir, "timestamps.npy")
        # Compute `rgb_path` as an intermediate representation used by later output layers.
        rgb_path = os.path.join(base_dir, "rgb_224.npy")
        # Compute `quality_path` as an intermediate representation used by later output layers.
        quality_path = os.path.join(base_dir, "quality.json")

        # Set `landmarks` for subsequent steps so gradient updates improve future predictions.
        landmarks = np.load(landmarks_path).astype(np.float32)
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = np.load(mask_path).astype(np.float32)
        # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
        timestamps = np.load(timestamps_path).astype(np.float32)
        # Set `rgb` for subsequent steps so gradient updates improve future predictions.
        rgb = None
        if self.use_rgb and os.path.exists(rgb_path):
            rgb = np.load(rgb_path)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # Set `quality` for subsequent steps so gradient updates improve future predictions.
        quality = None
        # Branch on `os.path.exists(quality_path)` to choose the correct output computation path.
        if os.path.exists(quality_path):
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Use a managed context to safely handle resources used during output computation.
                with open(quality_path, "r", encoding="utf-8") as f:
                    # Set `quality` for subsequent steps so gradient updates improve future predictions.
                    quality = json.load(f)
            # Handle exceptions and keep output behavior controlled under error conditions.
            except Exception:
                # Set `quality` for subsequent steps so gradient updates improve future predictions.
                quality = None
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "landmarks": landmarks,
            "mask": mask,
            "timestamps": timestamps,
            "rgb": rgb,
            "quality": quality,
        }

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_from_video(self, entry):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Set `result` for subsequent steps so gradient updates improve future predictions.
        result = self.processor.process_video_file(entry["video_path"], save_rgb=self.use_rgb)
        # Set `frames` for subsequent steps so gradient updates improve future predictions.
        frames = result["frames"]
        # Branch on `not frames` to choose the correct output computation path.
        if not frames:
            # Set `T` for subsequent steps so gradient updates improve future predictions.
            T = 1
            # Set `J` for subsequent steps so gradient updates improve future predictions.
            J = self.schema.total_joints
            # Return `{` as this function's contribution to downstream output flow.
            return {
                "landmarks": np.zeros((T, J, 3), dtype=np.float32),
                "mask": np.zeros((T, J), dtype=np.float32),
                "timestamps": np.zeros((T,), dtype=np.float32),
                "rgb": None,
                "quality": None,
            }

        # Set `landmarks` for subsequent steps so gradient updates improve future predictions.
        landmarks = np.stack([f["landmarks"] for f in frames]).astype(np.float32)
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = np.stack([f["mask"] for f in frames]).astype(np.float32)
        # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
        timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.float32)
        rgb = None
        if self.use_rgb:
            if "rgb_224" in frames[0]:
                rgb = np.stack([f["rgb_224"] for f in frames]).astype(np.uint8)
            else:
                rgb = None
        # Set `quality` for subsequent steps so gradient updates improve future predictions.
        quality = [f["quality"] for f in frames]
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "landmarks": landmarks,
            "mask": mask,
            "timestamps": timestamps,
            "rgb": rgb,
            "quality": quality,
        }

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_entry_arrays(self, entry):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Set `key` for subsequent steps so gradient updates improve future predictions.
        key = None
        # Branch on `self._cache is not None` to choose the correct output computation path.
        if self._cache is not None:
            # Set `key` for subsequent steps so gradient updates improve future predictions.
            key = (
                entry["video_path"],
                entry["subject_id"],
                entry["label"],
                entry.get("skeleton_source", ""),
            )
            # Branch on `key in self._cache` to choose the correct output computation path.
            if key in self._cache:
                # Return `self._cache[key]` as this function's contribution to downstream output flow.
                return self._cache[key]

        # Set `data` for subsequent steps so gradient updates improve future predictions.
        data = None
        # Set `skeleton_source` for subsequent steps so gradient updates improve future predictions.
        skeleton_source = _safe_text(entry.get("skeleton_source"))
        # Branch on `skeleton_source` to choose the correct output computation path.
        if skeleton_source:
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Set `data` for subsequent steps so gradient updates improve future predictions.
                data = self._load_from_skeleton_source(entry)
            # Handle exceptions and keep output behavior controlled under error conditions.
            except Exception as e:
                # Log runtime values to verify that output computation is behaving as expected.
                print(f"[VideoDataset] Failed skeleton source load ({skeleton_source}): {e}")
                # Set `data` for subsequent steps so gradient updates improve future predictions.
                data = None

        # Branch on `data is None` to choose the correct output computation path.
        if data is None:
            # Set `data` for subsequent steps so gradient updates improve future predictions.
            data = self._load_preprocessed(entry) if self.use_preprocessed else None
        # Branch on `data is None` to choose the correct output computation path.
        if data is None:
            # Set `data` for subsequent steps so gradient updates improve future predictions.
            data = self._load_from_video(entry)

        # Branch on `self._cache is not None` to choose the correct output computation path.
        if self._cache is not None:
            # Compute `self._cache[key]` as an intermediate representation used by later output layers.
            self._cache[key] = data
        # Return `data` as this function's contribution to downstream output flow.
        return data

    # Define a reusable pipeline function whose outputs feed later steps.
    def _normalize_landmarks(self, xyz, mask):
        return normalize_landmarks(xyz, mask, smooth_kernel=self.smooth_kernel)

    # Execute this statement so gradient updates improve future predictions.
    @staticmethod
    def _build_motion_features(xyz, mask):
        return build_motion_features(xyz, mask)

    # Define a reusable pipeline function whose outputs feed later steps.
    def _sample_starts(self, T, window_size, n_windows):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `T <= window_size` to choose the correct output computation path.
        if T <= window_size:
            # Return `[0 for _ in range(n_windows)]` as this function's contribution to downstream output flow.
            return [0 for _ in range(n_windows)]
        # Compute `max_start` as an intermediate representation used by later output layers.
        max_start = T - window_size
        # Branch on `self.is_training` to choose the correct output computation path.
        if self.is_training:
            # Return `[random.randint(0, max_start) for _ in range(n_wind...` as this function's contribution to downstream output flow.
            return [random.randint(0, max_start) for _ in range(n_windows)]
        # Branch on `n_windows == 1` to choose the correct output computation path.
        if n_windows == 1:
            # Return `[max_start // 2]` as this function's contribution to downstream output flow.
            return [max_start // 2]
        # Return `np.linspace(0, max_start, n_windows).astype(int).to...` as this function's contribution to downstream output flow.
        return np.linspace(0, max_start, n_windows).astype(int).tolist()

    # Define a reusable pipeline function whose outputs feed later steps.
    def __getitem__(self, idx):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `entry` for subsequent steps so gradient updates improve future predictions.
        entry = self.entries[idx]
        # Set `data` for subsequent steps so gradient updates improve future predictions.
        data = self._load_entry_arrays(entry)
        # Compute `xyz` as an intermediate representation used by later output layers.
        xyz = data["landmarks"]
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = data["mask"]
        # Set `timestamps` for subsequent steps so gradient updates improve future predictions.
        timestamps = data["timestamps"]
        # Set `quality` for subsequent steps so gradient updates improve future predictions.
        quality = data.get("quality", None)
        # Set `rgb` for subsequent steps so gradient updates improve future predictions.
        rgb = data.get("rgb", None)

        # Build `xyz, mask` to gate invalid timesteps/joints from influencing outputs.
        xyz, mask = self._normalize_landmarks(xyz, mask)
        # Build `motion` to gate invalid timesteps/joints from influencing outputs.
        motion = self._build_motion_features(xyz, mask)
        if self.pose_only:
            pose_slice = self.schema.pose_slice
            motion = motion[:, pose_slice, :]
            mask = mask[:, pose_slice]

        # Branch on `self.is_training` to choose the correct output computation path.
        if self.is_training:
            # Compute `window_size` as an intermediate representation used by later output layers.
            window_size = random.choice(self.window_sizes)
            # Compute `n_windows` as an intermediate representation used by later output layers.
            n_windows = self.windows_per_video
        else:
            # Compute `window_size` as an intermediate representation used by later output layers.
            window_size = int(max(self.window_sizes))
            # Compute `n_windows` as an intermediate representation used by later output layers.
            n_windows = self.eval_windows_per_video

        # Set `starts` for subsequent steps so gradient updates improve future predictions.
        starts = self._sample_starts(motion.shape[0], window_size, n_windows)

        # Compute `windows` as an intermediate representation used by later output layers.
        windows = []
        # Build `masks` to gate invalid timesteps/joints from influencing outputs.
        masks = []
        # Set `win_timestamps` for subsequent steps so gradient updates improve future predictions.
        win_timestamps = []
        # Set `rgb_windows` for subsequent steps so gradient updates improve future predictions.
        rgb_windows = []
        # Iterate over `starts` so each item contributes to final outputs/metrics.
        for s in starts:
            # Set `e` for subsequent steps so gradient updates improve future predictions.
            e = s + window_size
            # Set `w` for subsequent steps so gradient updates improve future predictions.
            w = motion[s:e]
            # Build `m` to gate invalid timesteps/joints from influencing outputs.
            m = mask[s:e]
            # Set `ts` for subsequent steps so gradient updates improve future predictions.
            ts = timestamps[s:e]
            # Set `rw` for subsequent steps so gradient updates improve future predictions.
            rw = None
            if self.use_rgb:
                if rgb is None:
                    rw = np.zeros((w.shape[0], 224, 224, 3), dtype=np.uint8)
                else:
                    rw = rgb[s:e]

            # Branch on `w.shape[0] < window_size` to choose the correct output computation path.
            if w.shape[0] < window_size:
                # Set `pad_t` for subsequent steps so gradient updates improve future predictions.
                pad_t = window_size - w.shape[0]
                # Set `w` for subsequent steps so gradient updates improve future predictions.
                w = np.pad(w, ((0, pad_t), (0, 0), (0, 0)), mode="constant")
                # Set `m` for subsequent steps so gradient updates improve future predictions.
                m = np.pad(m, ((0, pad_t), (0, 0)), mode="constant")
                # Set `ts` for subsequent steps so gradient updates improve future predictions.
                ts = np.pad(ts, (0, pad_t), mode="edge" if ts.size > 0 else "constant")
                if self.use_rgb:
                    rw = np.pad(rw, ((0, pad_t), (0, 0), (0, 0), (0, 0)), mode="constant")

            # Call `windows.append` and use its result in later steps so gradient updates improve future predictions.
            windows.append(w.astype(np.float32))
            # Call `masks.append` and use its result in later steps so gradient updates improve future predictions.
            masks.append(m.astype(np.float32))
            # Call `win_timestamps.append` and use its result in later steps so gradient updates improve future predictions.
            win_timestamps.append(ts.astype(np.float32))
            if self.use_rgb:
                rw = rw.astype(np.float32) / 255.0
                rw = np.transpose(rw, (0, 3, 1, 2))
                rgb_windows.append(rw)

        # Compute `windows` as an intermediate representation used by later output layers.
        windows = np.stack(windows, axis=0)  # [S, W, J, 9]
        # Build `masks` to gate invalid timesteps/joints from influencing outputs.
        masks = np.stack(masks, axis=0)  # [S, W, J]
        # Set `win_timestamps` for subsequent steps so gradient updates improve future predictions.
        win_timestamps = np.stack(win_timestamps, axis=0)  # [S, W]
        if self.use_rgb:
            rgb_windows = np.stack(rgb_windows, axis=0)  # [S,W,3,224,224]

        # Per-video quality summary for optional downstream filtering.
        # Branch on `quality and isinstance(quality, list)` to choose the correct output computation path.
        if quality and isinstance(quality, list):
            # Set `face_q` for subsequent steps so gradient updates improve future predictions.
            face_q = float(np.mean([float(q.get("face_score", 0.0)) for q in quality]))
            # Set `pose_q` for subsequent steps so gradient updates improve future predictions.
            pose_q = float(np.mean([float(q.get("pose_score", 0.0)) for q in quality]))
            # Compute `hand_q` as an intermediate representation used by later output layers.
            hand_q = float(np.mean([float(q.get("hand_score", 0.0)) for q in quality]))
        else:
            if self.pose_only:
                pose_q = float(mask.mean())
                hand_q = 0.0
                face_q = 0.0
            else:
                pose_slice = self.schema.pose_slice
                l_hand_slice = self.schema.left_hand_slice
                r_hand_slice = self.schema.right_hand_slice
                face_slice = self.schema.face_slice
                pose_q = float(mask[:, pose_slice].mean())
                hand_q = float(0.5 * (mask[:, l_hand_slice].mean() + mask[:, r_hand_slice].mean()))
                face_q = float(mask[:, face_slice].mean())

        out = {
            "motion_windows": torch.from_numpy(windows),  # [S, W, J, 9]
            "joint_mask": torch.from_numpy(masks),  # [S, W, J]
            "window_timestamps": torch.from_numpy(win_timestamps),  # [S, W]
            "window_size": int(window_size),
            "label": torch.tensor(entry["label"], dtype=torch.float32),
            "action_type": _safe_text(entry.get("action_type")),
            "action_id": torch.tensor(
                self.action_to_id.get(_safe_text(entry.get("action_type")), -1),
                dtype=torch.long,
            ),
            "video_id": os.path.basename(entry["video_path"]),
            "subject_id": entry["subject_id"],
            "qualities": {
                "face_score": torch.tensor(face_q, dtype=torch.float32),
                "pose_score": torch.tensor(pose_q, dtype=torch.float32),
                "hand_score": torch.tensor(hand_q, dtype=torch.float32),
            },
        }
        if self.use_rgb:
            out["rgb_windows"] = torch.from_numpy(rgb_windows)  # [S,W,3,224,224]
        return out


# Define batch collation so model inputs are aligned for correct output computation.
def collate_motion_batch(batch):
    """Packs batch items into model-ready tensors, affecting what the model sees per step."""
    # Branch on `not batch` to choose the correct output computation path.
    if not batch:
        # Return `{}` as this function's contribution to downstream output flow.
        return {}

    # Compute `max_w` as an intermediate representation used by later output layers.
    max_w = max(int(item["motion_windows"].shape[1]) for item in batch)
    has_rgb = all(("rgb_windows" in item) for item in batch)
    # Set `s` for subsequent steps so gradient updates improve future predictions.
    s = int(batch[0]["motion_windows"].shape[0])
    # Set `j` for subsequent steps so gradient updates improve future predictions.
    j = int(batch[0]["motion_windows"].shape[2])
    # Set `f` for subsequent steps so gradient updates improve future predictions.
    f = int(batch[0]["motion_windows"].shape[3])

    # Set `motion_list` for subsequent steps so gradient updates improve future predictions.
    motion_list = []
    # Build `mask_list` to gate invalid timesteps/joints from influencing outputs.
    mask_list = []
    # Set `ts_list` for subsequent steps so gradient updates improve future predictions.
    ts_list = []
    rgb_list = []
    # Set `quality` for subsequent steps so gradient updates improve future predictions.
    quality = {"face_score": [], "pose_score": [], "hand_score": []}
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = []
    # Set `video_ids` for subsequent steps so gradient updates improve future predictions.
    video_ids = []
    # Set `subject_ids` for subsequent steps so gradient updates improve future predictions.
    subject_ids = []
    # Compute `window_sizes` as an intermediate representation used by later output layers.
    window_sizes = []
    # Set `action_types` for subsequent steps so gradient updates improve future predictions.
    action_types = []
    # Set `action_ids` for subsequent steps so gradient updates improve future predictions.
    action_ids = []

    # Iterate over `batch` so each item contributes to final outputs/metrics.
    for item in batch:
        # Set `motion` for subsequent steps so gradient updates improve future predictions.
        motion = item["motion_windows"]  # [S,W,J,F]
        # Build `joint_mask` to gate invalid timesteps/joints from influencing outputs.
        joint_mask = item["joint_mask"]  # [S,W,J]
        # Set `ts` for subsequent steps so gradient updates improve future predictions.
        ts = item["window_timestamps"]  # [S,W]
        rgb = item.get("rgb_windows", None)  # [S,W,3,224,224]
        # Set `w` for subsequent steps so gradient updates improve future predictions.
        w = int(motion.shape[1])
        # Branch on `w < max_w` to choose the correct output computation path.
        if w < max_w:
            # Set `pad_w` for subsequent steps so gradient updates improve future predictions.
            pad_w = max_w - w
            # Set `motion` for subsequent steps so gradient updates improve future predictions.
            motion = torch.cat(
                [motion, torch.zeros((s, pad_w, j, f), dtype=motion.dtype)],
                dim=1,
            )
            # Build `joint_mask` to gate invalid timesteps/joints from influencing outputs.
            joint_mask = torch.cat(
                [joint_mask, torch.zeros((s, pad_w, j), dtype=joint_mask.dtype)],
                dim=1,
            )
            # Set `ts` for subsequent steps so gradient updates improve future predictions.
            ts = torch.cat(
                [ts, torch.zeros((s, pad_w), dtype=ts.dtype)],
                dim=1,
            )
            if has_rgb and rgb is not None:
                rgb = torch.cat(
                    [
                        rgb,
                        torch.zeros((s, pad_w, 3, 224, 224), dtype=rgb.dtype),
                    ],
                    dim=1,
                )

        # Call `motion_list.append` and use its result in later steps so gradient updates improve future predictions.
        motion_list.append(motion)
        # Call `mask_list.append` and use its result in later steps so gradient updates improve future predictions.
        mask_list.append(joint_mask)
        # Call `ts_list.append` and use its result in later steps so gradient updates improve future predictions.
        ts_list.append(ts)
        if has_rgb and rgb is not None:
            rgb_list.append(rgb)
        # Call `labels.append` and use its result in later steps so gradient updates improve future predictions.
        labels.append(item["label"])
        # Call `action_types.append` and use its result in later steps so gradient updates improve future predictions.
        action_types.append(item.get("action_type", ""))
        # Call `action_ids.append` and use its result in later steps so gradient updates improve future predictions.
        action_ids.append(item.get("action_id", torch.tensor(-1, dtype=torch.long)))
        # Call `video_ids.append` and use its result in later steps so gradient updates improve future predictions.
        video_ids.append(item["video_id"])
        # Call `subject_ids.append` and use its result in later steps so gradient updates improve future predictions.
        subject_ids.append(item["subject_id"])
        # Call `window_sizes.append` and use its result in later steps so gradient updates improve future predictions.
        window_sizes.append(int(item["window_size"]))
        # Iterate over `quality.keys()` so each item contributes to final outputs/metrics.
        for k in quality.keys():
            # Call `append` and use its result in later steps so gradient updates improve future predictions.
            quality[k].append(item["qualities"][k])

    out = {
        "motion_windows": torch.stack(motion_list, dim=0),  # [B,S,W,J,F]
        "joint_mask": torch.stack(mask_list, dim=0),  # [B,S,W,J]
        "window_timestamps": torch.stack(ts_list, dim=0),  # [B,S,W]
        "label": torch.stack(labels, dim=0),
        "action_type": action_types,
        "action_id": torch.stack(action_ids, dim=0),
        "qualities": {k: torch.stack(v, dim=0) for k, v in quality.items()},
        "video_id": video_ids,
        "subject_id": subject_ids,
        "window_size": torch.tensor(window_sizes, dtype=torch.int64),
    }
    if has_rgb and rgb_list:
        out["rgb_windows"] = torch.stack(rgb_list, dim=0)  # [B,S,W,3,224,224]
    return out
