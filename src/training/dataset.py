import csv
import json
import os
import random
import urllib.parse
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
from src.pipeline.preprocess import VideoProcessor
from src.utils.video_id import make_video_id


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _moving_average_1d(x, k=5):
    if k <= 1:
        return x
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = np.ones((k,), dtype=np.float32)
    kernel /= kernel.sum()
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _is_http_link(text):
    t = _safe_text(text).lower()
    return t.startswith("http://") or t.startswith("https://")


def _fill_missing_1d(values, valid):
    out = values.copy()
    if valid.sum() == 0:
        return out

    idx = np.where(valid > 0.5)[0]
    first = idx[0]
    last = idx[-1]
    out[:first] = out[first]
    out[last + 1:] = out[last]
    miss = np.where(valid < 0.5)[0]
    for m in miss:
        left = idx[idx < m]
        right = idx[idx > m]
        if left.size == 0 and right.size == 0:
            continue
        if left.size == 0:
            out[m] = out[right[0]]
        elif right.size == 0:
            out[m] = out[left[-1]]
        else:
            l = left[-1]
            r = right[0]
            alpha = (m - l) / max((r - l), 1)
            out[m] = (1.0 - alpha) * out[l] + alpha * out[r]
    return out


class VideoDataset(Dataset):
    """
    Landmark-first ASD dataset.

    Each item is a set of temporal windows from one video:
    - motion_windows: [S, W, J, 9]
    - joint_mask: [S, W, J]
    """

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
    ):
        self.sequence_length = int(sequence_length)
        self.is_training = bool(is_training)
        self.require_label = bool(require_label)
        self.use_preprocessed = bool(use_preprocessed)
        self.processed_root = str(processed_root)
        self.window_sizes = tuple(sorted({int(v) for v in window_sizes if int(v) > 0}))
        if not self.window_sizes:
            self.window_sizes = (self.sequence_length,)
        self.windows_per_video = int(max(1, windows_per_video))
        self.eval_windows_per_video = int(max(1, eval_windows_per_video))
        self.frame_stride = int(frame_stride)
        self.max_frames = int(max_frames)
        self.cache_enabled = bool(cache_enabled)
        self.smooth_kernel = int(max(1, smooth_kernel))
        self.schema = DEFAULT_SCHEMA
        self.processor = VideoProcessor(
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
            schema=self.schema,
        )
        self.external_cache_dir = os.path.join(self.processed_root, "_external_cache")

        self.entries = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if self.require_label and "subject_id" not in fieldnames:
                raise ValueError("CSV must contain subject_id for grouped validation.")
            if self.require_label and "label" not in reader.fieldnames:
                raise ValueError("CSV must contain label for supervised ASD training.")
            for row_idx, row in enumerate(reader, start=1):
                action_type = self._parse_action_type(row)
                skeleton_source = self._parse_skeleton_source(row)
                label = self._parse_label(row)
                video_path = _safe_text(row.get("video_path"))
                if not video_path:
                    video_path = skeleton_source or f"skeleton_sample_{row_idx:06d}"
                subject_id = _safe_text(row.get("subject_id"))
                if not subject_id:
                    if self.require_label:
                        raise ValueError(
                            f"Missing subject_id in row {row_idx} while require_label=true."
                        )
                    subject_id = f"subject_{row_idx:06d}"
                if not skeleton_source and not video_path:
                    raise ValueError(
                        f"Row {row_idx} must contain either video_path or skeleton source column."
                    )
                self.entries.append(
                    {
                        "video_path": video_path,
                        "label": label,
                        "subject_id": subject_id,
                        "action_type": action_type,
                        "skeleton_source": skeleton_source,
                    }
                )

        self.action_to_id = {}
        self.id_to_action = []
        for entry in self.entries:
            action = _safe_text(entry.get("action_type"))
            if action and action not in self.action_to_id:
                self.action_to_id[action] = len(self.id_to_action)
                self.id_to_action.append(action)
        self.num_action_classes = int(len(self.id_to_action))

        self._cache = {} if self.cache_enabled else None

    def __len__(self):
        return len(self.entries)

    def _candidate_ids(self, entry):
        video_path = entry["video_path"]
        subject_id = entry.get("subject_id")
        label = entry.get("label")
        ids = []
        ids.append(make_video_id(video_path, subject_id=subject_id, label=label))
        prev = make_video_id(video_path, subject_id=subject_id)
        if prev not in ids:
            ids.append(prev)
        legacy = make_video_id(video_path)
        if legacy not in ids:
            ids.append(legacy)
        return ids

    @staticmethod
    def _parse_action_type(row):
        for key in ("action_type", "action", "activity", "gesture", "motion_type"):
            value = _safe_text(row.get(key))
            if value:
                return value
        return ""

    def _parse_label(self, row):
        text = _safe_text(row.get("label"))
        if text == "":
            if self.require_label:
                raise ValueError("Missing label in CSV row while require_label=true.")
            return 0.0
        try:
            return float(text)
        except Exception:
            if self.require_label:
                raise ValueError(f"Invalid label value: {text}")
            return 0.0

    @staticmethod
    def _parse_skeleton_source(row):
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
        for key in keys:
            value = _safe_text(row.get(key))
            if value:
                return value
        return ""

    @staticmethod
    def _parse_ntu_skeleton_file(path):
        """
        Parse common NTU RGB+D `.skeleton` text format.
        Returns landmarks [T,J,3], mask [T,J], timestamps [T].
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip() != ""]
        if not lines:
            raise ValueError(f"Empty .skeleton file: {path}")

        p = 0
        try:
            n_frames = int(float(lines[p]))
        except Exception as e:
            raise ValueError(f"Invalid .skeleton header: {path}") from e
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
                p += 1  # skip body-info line
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
        landmarks = np.stack(out, axis=0).astype(np.float32)  # [T,J,3]
        mask = (np.abs(landmarks).sum(axis=-1) > 1e-8).astype(np.float32)
        timestamps = np.arange(landmarks.shape[0], dtype=np.float32)
        return landmarks, mask, timestamps

    def _download_external_file(self, url):
        os.makedirs(self.external_cache_dir, exist_ok=True)
        parsed = urllib.parse.urlparse(url)
        name = os.path.basename(parsed.path) or "skeleton_data.bin"
        safe_name = name.replace("?", "_").replace("&", "_").replace("=", "_")
        local_path = os.path.join(self.external_cache_dir, safe_name)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path

    def _normalize_skeleton_arrays(self, landmarks, mask=None, timestamps=None):
        # landmarks expected [T, J, C]
        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.ndim == 2:
            # [T, J*3] format
            t, d = arr.shape
            if d % 3 != 0:
                raise ValueError(f"Unsupported 2D skeleton shape: {arr.shape}")
            arr = arr.reshape(t, d // 3, 3)
        if arr.ndim != 3:
            raise ValueError(f"Expected skeleton array rank 3, got shape={arr.shape}")

        t, j, c = arr.shape
        if c == 2:
            z = np.zeros((t, j, 1), dtype=np.float32)
            arr = np.concatenate([arr, z], axis=-1)
        elif c >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Unsupported channel count: {c}")

        target_j = int(self.schema.total_joints)
        if j < target_j:
            pad_j = target_j - j
            arr = np.pad(arr, ((0, 0), (0, pad_j), (0, 0)), mode="constant")
            if mask is not None:
                mask = np.pad(np.asarray(mask, dtype=np.float32), ((0, 0), (0, pad_j)), mode="constant")
        elif j > target_j:
            arr = arr[:, :target_j, :]
            if mask is not None:
                mask = np.asarray(mask, dtype=np.float32)[:, :target_j]

        if mask is None:
            # Valid if any non-zero coordinate present.
            mask = (np.abs(arr).sum(axis=-1) > 1e-8).astype(np.float32)
        else:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim != 2:
                raise ValueError(f"mask must be [T,J], got shape={mask.shape}")
            if mask.shape[0] != arr.shape[0]:
                raise ValueError("mask/time dimension mismatch with landmarks")
            if mask.shape[1] < target_j:
                pad_j = target_j - mask.shape[1]
                mask = np.pad(mask, ((0, 0), (0, pad_j)), mode="constant")
            elif mask.shape[1] > target_j:
                mask = mask[:, :target_j]
            mask = (mask > 0.5).astype(np.float32)

        if timestamps is None:
            timestamps = np.arange(arr.shape[0], dtype=np.float32)
        else:
            timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
            if timestamps.shape[0] != arr.shape[0]:
                timestamps = np.arange(arr.shape[0], dtype=np.float32)

        return arr.astype(np.float32), mask.astype(np.float32), timestamps.astype(np.float32)

    def _load_skeleton_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        quality = None

        if ext == ".npz":
            data = np.load(path, allow_pickle=True)
            keys = set(data.files)
            lmk_key = "landmarks" if "landmarks" in keys else ("keypoints" if "keypoints" in keys else None)
            if lmk_key is None:
                raise ValueError(f"No landmarks/keypoints key in npz: {path}")
            landmarks = data[lmk_key]
            mask = data["mask"] if "mask" in keys else None
            timestamps = data["timestamps"] if "timestamps" in keys else None
            arr, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
            return {"landmarks": arr, "mask": m, "timestamps": ts, "quality": quality}

        if ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
                obj = arr.item()
                if isinstance(obj, dict):
                    landmarks = obj.get("landmarks", obj.get("keypoints"))
                    mask = obj.get("mask")
                    timestamps = obj.get("timestamps")
                    if landmarks is None:
                        raise ValueError(f"No landmarks/keypoints in npy dict: {path}")
                    arr3, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
                    return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}
            arr3, m, ts = self._normalize_skeleton_arrays(arr, mask=None, timestamps=None)
            return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}

        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            landmarks = obj.get("landmarks", obj.get("keypoints"))
            mask = obj.get("mask")
            timestamps = obj.get("timestamps")
            quality = obj.get("quality")
            if landmarks is None:
                raise ValueError(f"No landmarks/keypoints in json: {path}")
            arr3, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
            return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}

        if ext == ".skeleton":
            landmarks, mask, timestamps = self._parse_ntu_skeleton_file(path)
            arr3, m, ts = self._normalize_skeleton_arrays(landmarks, mask=mask, timestamps=timestamps)
            return {"landmarks": arr3, "mask": m, "timestamps": ts, "quality": quality}

        raise ValueError(f"Unsupported skeleton file extension: {ext}")

    def _load_from_skeleton_source(self, entry):
        source = _safe_text(entry.get("skeleton_source"))
        if not source:
            return None
        path = source
        if _is_http_link(source):
            path = self._download_external_file(source)
        elif not os.path.isabs(path):
            # Resolve relative paths from repo root.
            path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Skeleton source not found: {source}")
        return self._load_skeleton_file(path)

    def _load_preprocessed(self, entry):
        base_dir = None
        for vid in self._candidate_ids(entry):
            candidate = os.path.join(self.processed_root, vid)
            if os.path.exists(os.path.join(candidate, "landmarks.npy")):
                base_dir = candidate
                break
        if base_dir is None:
            return None

        landmarks_path = os.path.join(base_dir, "landmarks.npy")
        mask_path = os.path.join(base_dir, "landmark_mask.npy")
        timestamps_path = os.path.join(base_dir, "timestamps.npy")
        quality_path = os.path.join(base_dir, "quality.json")

        landmarks = np.load(landmarks_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        timestamps = np.load(timestamps_path).astype(np.float32)
        quality = None
        if os.path.exists(quality_path):
            try:
                with open(quality_path, "r", encoding="utf-8") as f:
                    quality = json.load(f)
            except Exception:
                quality = None
        return {
            "landmarks": landmarks,
            "mask": mask,
            "timestamps": timestamps,
            "quality": quality,
        }

    def _load_from_video(self, entry):
        result = self.processor.process_video_file(entry["video_path"])
        frames = result["frames"]
        if not frames:
            T = 1
            J = self.schema.total_joints
            return {
                "landmarks": np.zeros((T, J, 3), dtype=np.float32),
                "mask": np.zeros((T, J), dtype=np.float32),
                "timestamps": np.zeros((T,), dtype=np.float32),
                "quality": None,
            }

        landmarks = np.stack([f["landmarks"] for f in frames]).astype(np.float32)
        mask = np.stack([f["mask"] for f in frames]).astype(np.float32)
        timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.float32)
        quality = [f["quality"] for f in frames]
        return {
            "landmarks": landmarks,
            "mask": mask,
            "timestamps": timestamps,
            "quality": quality,
        }

    def _load_entry_arrays(self, entry):
        key = None
        if self._cache is not None:
            key = (
                entry["video_path"],
                entry["subject_id"],
                entry["label"],
                entry.get("skeleton_source", ""),
            )
            if key in self._cache:
                return self._cache[key]

        data = None
        skeleton_source = _safe_text(entry.get("skeleton_source"))
        if skeleton_source:
            try:
                data = self._load_from_skeleton_source(entry)
            except Exception as e:
                print(f"[VideoDataset] Failed skeleton source load ({skeleton_source}): {e}")
                data = None

        if data is None:
            data = self._load_preprocessed(entry) if self.use_preprocessed else None
        if data is None:
            data = self._load_from_video(entry)

        if self._cache is not None:
            self._cache[key] = data
        return data

    def _normalize_landmarks(self, xyz, mask):
        # xyz: [T, J, 3], mask: [T, J]
        xyz = xyz.copy()
        mask = mask.copy()
        T, J, _ = xyz.shape

        # Fill missing values per joint/channel to stabilize normalization.
        for j in range(J):
            valid = mask[:, j] > 0.5
            for c in range(3):
                xyz[:, j, c] = _fill_missing_1d(xyz[:, j, c], valid)

        # Hip-centered coordinates.
        l_hip = 23
        r_hip = 24
        hips_valid = (mask[:, l_hip] > 0.5) & (mask[:, r_hip] > 0.5)
        hip_center = np.zeros((T, 3), dtype=np.float32)
        hip_center[hips_valid] = 0.5 * (xyz[hips_valid, l_hip] + xyz[hips_valid, r_hip])
        if hips_valid.any():
            last = hip_center[np.where(hips_valid)[0][0]].copy()
        else:
            last = np.zeros((3,), dtype=np.float32)
        for t in range(T):
            if hips_valid[t]:
                last = hip_center[t]
            else:
                hip_center[t] = last
        xyz = xyz - hip_center[:, None, :]

        # Shoulder-distance scale normalization.
        l_sh = 11
        r_sh = 12
        sh_valid = (mask[:, l_sh] > 0.5) & (mask[:, r_sh] > 0.5)
        scale = np.ones((T,), dtype=np.float32)
        if sh_valid.any():
            dist = np.linalg.norm(xyz[:, l_sh, :] - xyz[:, r_sh, :], axis=-1)
            dist = np.clip(dist, 1e-4, None)
            scale[sh_valid] = dist[sh_valid]
            median_scale = float(np.median(scale[sh_valid]))
            if median_scale <= 0:
                median_scale = 1.0
            scale[~sh_valid] = median_scale
        xyz = xyz / scale[:, None, None]

        # Temporal smoothing (weighted by mask).
        for j in range(J):
            valid = mask[:, j]
            for c in range(3):
                smoothed = _moving_average_1d(xyz[:, j, c], k=self.smooth_kernel)
                # Keep missing parts from filled value but blend with valid confidence.
                xyz[:, j, c] = (valid * smoothed) + ((1.0 - valid) * xyz[:, j, c])

        return xyz, mask

    @staticmethod
    def _build_motion_features(xyz, mask):
        # xyz: [T, J, 3]
        vel = np.zeros_like(xyz, dtype=np.float32)
        acc = np.zeros_like(xyz, dtype=np.float32)
        if xyz.shape[0] > 1:
            vel[1:] = xyz[1:] - xyz[:-1]
            acc[1:] = vel[1:] - vel[:-1]

        feat = np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)  # [T, J, 9]
        feat *= mask[..., None]
        return feat

    def _sample_starts(self, T, window_size, n_windows):
        if T <= window_size:
            return [0 for _ in range(n_windows)]
        max_start = T - window_size
        if self.is_training:
            return [random.randint(0, max_start) for _ in range(n_windows)]
        if n_windows == 1:
            return [max_start // 2]
        return np.linspace(0, max_start, n_windows).astype(int).tolist()

    def __getitem__(self, idx):
        entry = self.entries[idx]
        data = self._load_entry_arrays(entry)
        xyz = data["landmarks"]
        mask = data["mask"]
        timestamps = data["timestamps"]
        quality = data.get("quality", None)

        xyz, mask = self._normalize_landmarks(xyz, mask)
        motion = self._build_motion_features(xyz, mask)

        if self.is_training:
            window_size = random.choice(self.window_sizes)
            n_windows = self.windows_per_video
        else:
            window_size = int(max(self.window_sizes))
            n_windows = self.eval_windows_per_video

        starts = self._sample_starts(motion.shape[0], window_size, n_windows)

        windows = []
        masks = []
        win_timestamps = []
        for s in starts:
            e = s + window_size
            w = motion[s:e]
            m = mask[s:e]
            ts = timestamps[s:e]

            if w.shape[0] < window_size:
                pad_t = window_size - w.shape[0]
                w = np.pad(w, ((0, pad_t), (0, 0), (0, 0)), mode="constant")
                m = np.pad(m, ((0, pad_t), (0, 0)), mode="constant")
                ts = np.pad(ts, (0, pad_t), mode="edge" if ts.size > 0 else "constant")

            windows.append(w.astype(np.float32))
            masks.append(m.astype(np.float32))
            win_timestamps.append(ts.astype(np.float32))

        windows = np.stack(windows, axis=0)  # [S, W, J, 9]
        masks = np.stack(masks, axis=0)  # [S, W, J]
        win_timestamps = np.stack(win_timestamps, axis=0)  # [S, W]

        # Per-video quality summary for optional downstream filtering.
        if quality and isinstance(quality, list):
            face_q = float(np.mean([float(q.get("face_score", 0.0)) for q in quality]))
            pose_q = float(np.mean([float(q.get("pose_score", 0.0)) for q in quality]))
            hand_q = float(np.mean([float(q.get("hand_score", 0.0)) for q in quality]))
        else:
            # Fall back to modality coverage from mask.
            pose_slice = self.schema.pose_slice
            l_hand_slice = self.schema.left_hand_slice
            r_hand_slice = self.schema.right_hand_slice
            face_slice = self.schema.face_slice
            pose_q = float(mask[:, pose_slice].mean())
            hand_q = float(
                0.5 * (mask[:, l_hand_slice].mean() + mask[:, r_hand_slice].mean())
            )
            face_q = float(mask[:, face_slice].mean())

        return {
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


def collate_motion_batch(batch):
    if not batch:
        return {}

    max_w = max(int(item["motion_windows"].shape[1]) for item in batch)
    s = int(batch[0]["motion_windows"].shape[0])
    j = int(batch[0]["motion_windows"].shape[2])
    f = int(batch[0]["motion_windows"].shape[3])

    motion_list = []
    mask_list = []
    ts_list = []
    quality = {"face_score": [], "pose_score": [], "hand_score": []}
    labels = []
    video_ids = []
    subject_ids = []
    window_sizes = []
    action_types = []
    action_ids = []

    for item in batch:
        motion = item["motion_windows"]  # [S,W,J,F]
        joint_mask = item["joint_mask"]  # [S,W,J]
        ts = item["window_timestamps"]  # [S,W]
        w = int(motion.shape[1])
        if w < max_w:
            pad_w = max_w - w
            motion = torch.cat(
                [motion, torch.zeros((s, pad_w, j, f), dtype=motion.dtype)],
                dim=1,
            )
            joint_mask = torch.cat(
                [joint_mask, torch.zeros((s, pad_w, j), dtype=joint_mask.dtype)],
                dim=1,
            )
            ts = torch.cat(
                [ts, torch.zeros((s, pad_w), dtype=ts.dtype)],
                dim=1,
            )

        motion_list.append(motion)
        mask_list.append(joint_mask)
        ts_list.append(ts)
        labels.append(item["label"])
        action_types.append(item.get("action_type", ""))
        action_ids.append(item.get("action_id", torch.tensor(-1, dtype=torch.long)))
        video_ids.append(item["video_id"])
        subject_ids.append(item["subject_id"])
        window_sizes.append(int(item["window_size"]))
        for k in quality.keys():
            quality[k].append(item["qualities"][k])

    return {
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
