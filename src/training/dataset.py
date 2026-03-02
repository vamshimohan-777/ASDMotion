# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
ASD Video Dataset.

Expected CSV columns:
    video_path,label,subject_id
"""

import os
import csv
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from src.utils.video_id import make_video_id


class VideoDataset(Dataset):
    def __init__(self, csv_path, sequence_length=32, transform=None,
                 is_training=False, precompute=False, shared_cache=None,
                 frame_stride=1, max_frames=0, validate_videos=False,
                 cache_enabled=True, use_preprocessed=False,
                 processed_root="data/processed", preprocessed_only=True):
        # Compute `self.sequence_length` for the next processing step.
        self.sequence_length = sequence_length
        # Compute `self.is_training` for the next processing step.
        self.is_training = is_training
        # Compute `self.precompute` for the next processing step.
        self.precompute = precompute
        # Compute `self.frame_stride` for the next processing step.
        self.frame_stride = frame_stride
        # Compute `self.max_frames` for the next processing step.
        self.max_frames = max_frames
        # Compute `self.validate_videos` for the next processing step.
        self.validate_videos = validate_videos
        # Compute `self.cache_enabled` for the next processing step.
        self.cache_enabled = cache_enabled
        # Compute `self.use_preprocessed` for the next processing step.
        self.use_preprocessed = use_preprocessed
        # Compute `self.processed_root` for the next processing step.
        self.processed_root = processed_root
        # Compute `self.preprocessed_only` for the next processing step.
        self.preprocessed_only = preprocessed_only
        # Compute `self.transform` for the next processing step.
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Augmentation config
        self.aug_cfg = {
            "p_flip": 0.5,
            "rotation_range": 15,
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.8, 1.2),
            "saturation_range": (0.8, 1.2),
            "hue_range": (-0.1, 0.1),
            "p_affine": 0.3,
            "affine_scale": (0.9, 1.1),
            "affine_translate": 0.05,
            "p_erasing": 0.3,
            "erasing_scale": (0.02, 0.15),
            "p_temporal_mask": 0.4,
            "temporal_mask_ratio": (0.1, 0.3),
            "p_temporal_jitter": 0.5,
            "jitter_range": 2,
            "p_speed_perturb": 0.3,
            "speed_range": (0.8, 1.2),
            "p_gaussian_noise": 0.4,
            "noise_std": 0.05,
            "p_quality_noise": 0.3,
            "quality_noise_range": 0.1,
        }

        # Load CSV entries
        self.entries = []
        # Run this block with managed resources/context cleanup.
        with open(csv_path, "r", newline="") as f:
            # Compute `reader` for the next processing step.
            reader = csv.DictReader(f)
            # Branch behavior based on the current runtime condition.
            if "subject_id" not in reader.fieldnames:
                # Surface an explicit error when invariants are not met.
                raise ValueError(
                    "CSV is missing subject_id. Group-aware splitting is required to prevent "
                    "leakage. Leakage makes clinical metrics invalid."
                )
            # Iterate `row` across `reader` to process each element.
            for row in reader:
                # Compute `vpath` for the next processing step.
                vpath = row["video_path"].strip()
                # Compute `label` for the next processing step.
                label = float(row["label"])
                # Compute `subject_id` for the next processing step.
                subject_id = row["subject_id"].strip()
                # Invoke `self.entries.append` to advance this processing stage.
                self.entries.append({
                    "video_path": vpath,
                    "label": label,
                    "subject_id": subject_id,
                })

        # Compute `self._processor` for the next processing step.
        self._processor = None
        # Branch behavior based on the current runtime condition.
        if self.cache_enabled:
            # Compute `self._cache` for the next processing step.
            self._cache = shared_cache if shared_cache is not None else {}
            # Compute `self._valid_cache` for the next processing step.
            self._valid_cache = self._cache.setdefault(("__meta__", "valid"), {})
        else:
            # Compute `self._cache` for the next processing step.
            self._cache = None
            # Compute `self._valid_cache` for the next processing step.
            self._valid_cache = {}

        # Branch behavior based on the current runtime condition.
        if self.validate_videos:
            # Invoke `self._filter_invalid_entries` to advance this processing stage.
            self._filter_invalid_entries()

        # Branch behavior based on the current runtime condition.
        if self.precompute and self.cache_enabled:
            # Invoke `self._precompute_all` to advance this processing stage.
            self._precompute_all()
        # Branch behavior based on the current runtime condition.
        elif self.precompute and not self.cache_enabled:
            # Invoke `print` to advance this processing stage.
            print("[VideoDataset] cache_precompute ignored because cache_enabled=false.")

    @property
    def processor(self):
        # Branch behavior based on the current runtime condition.
        if self._processor is None:
            from src.pipeline.preprocess import VideoProcessor
            # Compute `self._processor` for the next processing step.
            self._processor = VideoProcessor(
                frame_stride=self.frame_stride,
                max_frames=self.max_frames,
            )
        # Return the result expected by the caller.
        return self._processor

    def _preprocess_video(self, video_path):
        # Branch behavior based on the current runtime condition.
        if self.use_preprocessed:
            # Compute `result` for the next processing step.
            result = self._load_preprocessed_video(video_path)
            # Branch behavior based on the current runtime condition.
            if result is not None:
                # Return the result expected by the caller.
                return result
            # Branch behavior based on the current runtime condition.
            if self.preprocessed_only:
                # Invoke `print` to advance this processing stage.
                print(f"[VideoDataset] Missing preprocessed data for {video_path}")
                # Return the result expected by the caller.
                return {"frames": [], "route": "video"}

        # Guard this block and recover cleanly from expected failures.
        try:
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Processing: {video_path}")
            # Compute `result` for the next processing step.
            result = self.processor.process_video_file(video_path)
            # Compute `frames` for the next processing step.
            frames = result.get("frames", [])
            # Compute `route` for the next processing step.
            route = result.get("route", "video")
            # Return the result expected by the caller.
            return {"frames": frames, "route": route}
        except Exception as e:
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Error processing {video_path}: {e}")
            # Return the result expected by the caller.
            return {"frames": [], "route": "video"}

    def _load_preprocessed_video(self, video_path):
        # Compute `video_id` for the next processing step.
        video_id = make_video_id(video_path)
        # Compute `base_dir` for the next processing step.
        base_dir = os.path.join(self.processed_root, video_id)
        # Compute `meta_path` for the next processing step.
        meta_path = os.path.join(base_dir, "meta.json")
        # Compute `quality_path` for the next processing step.
        quality_path = os.path.join(base_dir, "quality.json")

        # Branch behavior based on the current runtime condition.
        if not os.path.exists(meta_path):
            # Return the result expected by the caller.
            return None

        # Guard this block and recover cleanly from expected failures.
        try:
            # Run this block with managed resources/context cleanup.
            with open(meta_path, "r") as f:
                # Compute `meta` for the next processing step.
                meta = json.load(f)
        except Exception as e:
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Failed to read meta for {video_path}: {e}")
            # Return the result expected by the caller.
            return None

        # Compute `qualities` for the next processing step.
        qualities = []
        # Branch behavior based on the current runtime condition.
        if os.path.exists(quality_path):
            # Guard this block and recover cleanly from expected failures.
            try:
                # Run this block with managed resources/context cleanup.
                with open(quality_path, "r") as f:
                    # Compute `qualities` for the next processing step.
                    qualities = json.load(f)
            except Exception as e:
                # Invoke `print` to advance this processing stage.
                print(f"[VideoDataset] Failed to read quality for {video_path}: {e}")

        # Compute `frame_ids` for the next processing step.
        frame_ids = meta.get("frame_ids", [])
        # Compute `timestamps` for the next processing step.
        timestamps = meta.get("timestamps", [])
        # Compute `route` for the next processing step.
        route = meta.get("route", "video")

        # Compute `frames` for the next processing step.
        frames = []
        # Iterate `(i, frame_id)` across `enumerate(frame_ids)` to process each element.
        for i, frame_id in enumerate(frame_ids):
            # Compute `ts` for the next processing step.
            ts = timestamps[i] if i < len(timestamps) else float(frame_id)
            # Compute `q` for the next processing step.
            q = qualities[i] if i < len(qualities) else {}
            # Compute `face_path` for the next processing step.
            face_path = os.path.join(base_dir, "faces", f"{frame_id:06d}.png")
            # Compute `skeleton_path` for the next processing step.
            skeleton_path = os.path.join(base_dir, "skeletons", f"{frame_id:06d}.png")
            # Invoke `frames.append` to advance this processing stage.
            frames.append({
                "frame_id": frame_id,
                "timestamp": ts,
                "face_path": face_path,
                "skeleton_path": skeleton_path,
                "quality": q,
            })

        # Return the result expected by the caller.
        return {"frames": frames, "route": route}

    def _precompute_all(self):
        # Invoke `print` to advance this processing stage.
        print(f"[VideoDataset] Precomputing {len(self.entries)} videos...")
        # Iterate `entry` across `self.entries` to process each element.
        for entry in self.entries:
            # Compute `video_path` for the next processing step.
            video_path = entry["video_path"]
            # Branch behavior based on the current runtime condition.
            if self._cache is not None and video_path in self._cache:
                continue
            # Branch behavior based on the current runtime condition.
            if self._cache is not None:
                # Compute `self._cache[video_path]` for the next processing step.
                self._cache[video_path] = self._preprocess_video(video_path)
        # Invoke `print` to advance this processing stage.
        print("[VideoDataset] Precompute complete.")

    def _open_capture(self, video_path):
        # Compute `candidates` for the next processing step.
        candidates = []
        # Iterate `name` across `('CAP_FFMPEG', 'CAP_MSMF', 'CAP_D...` to process each element.
        for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_DSHOW"):
            # Branch behavior based on the current runtime condition.
            if hasattr(cv2, name):
                # Invoke `candidates.append` to advance this processing stage.
                candidates.append(getattr(cv2, name))
        # Invoke `candidates.append` to advance this processing stage.
        candidates.append(None)

        # Iterate `backend` across `candidates` to process each element.
        for backend in candidates:
            # Guard this block and recover cleanly from expected failures.
            try:
                # Compute `cap` for the next processing step.
                cap = cv2.VideoCapture(video_path) if backend is None else cv2.VideoCapture(video_path, backend)
            except Exception:
                continue
            # Branch behavior based on the current runtime condition.
            if cap is not None and cap.isOpened():
                # Return the result expected by the caller.
                return cap
            # Branch behavior based on the current runtime condition.
            if cap is not None:
                # Invoke `cap.release` to advance this processing stage.
                cap.release()
        # Return the result expected by the caller.
        return None

    def _is_video_valid(self, video_path):
        # Compute `cached` for the next processing step.
        cached = self._valid_cache.get(video_path)
        # Branch behavior based on the current runtime condition.
        if cached is not None:
            # Return the result expected by the caller.
            return cached

        # Branch behavior based on the current runtime condition.
        if not os.path.exists(video_path):
            # Compute `self._valid_cache[video_path]` for the next processing step.
            self._valid_cache[video_path] = False
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Missing file: {video_path}")
            # Return the result expected by the caller.
            return False

        # Compute `cap` for the next processing step.
        cap = self._open_capture(video_path)
        # Branch behavior based on the current runtime condition.
        if cap is None:
            # Compute `self._valid_cache[video_path]` for the next processing step.
            self._valid_cache[video_path] = False
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Failed to open: {video_path}")
            # Return the result expected by the caller.
            return False

        # Guard this block and recover cleanly from expected failures.
        try:
            # Guard this block and recover cleanly from expected failures.
            try:
                # Compute `(ok, frame)` for the next processing step.
                ok, frame = cap.read()
            except Exception as e:
                # Invoke `print` to advance this processing stage.
                print(f"[VideoDataset] Read failed for {video_path}: {e}")
                # Compute `(ok, frame)` for the next processing step.
                ok, frame = False, None
        finally:
            # Invoke `cap.release` to advance this processing stage.
            cap.release()

        # Compute `valid` for the next processing step.
        valid = bool(ok and frame is not None)
        # Compute `self._valid_cache[video_path]` for the next processing step.
        self._valid_cache[video_path] = valid
        # Branch behavior based on the current runtime condition.
        if not valid:
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Unreadable video: {video_path}")
        # Return the result expected by the caller.
        return valid

    def _filter_invalid_entries(self):
        # Compute `kept` for the next processing step.
        kept = []
        # Compute `bad` for the next processing step.
        bad = []
        # Iterate `entry` across `self.entries` to process each element.
        for entry in self.entries:
            # Compute `vpath` for the next processing step.
            vpath = entry["video_path"]
            # Branch behavior based on the current runtime condition.
            if self.use_preprocessed:
                # Compute `ok` for the next processing step.
                ok = self._is_preprocessed_valid(vpath)
            else:
                # Compute `ok` for the next processing step.
                ok = self._is_video_valid(vpath)
            # Branch behavior based on the current runtime condition.
            if ok:
                # Invoke `kept.append` to advance this processing stage.
                kept.append(entry)
            else:
                # Invoke `bad.append` to advance this processing stage.
                bad.append(vpath)

        # Branch behavior based on the current runtime condition.
        if bad:
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Skipping {len(bad)} unreadable videos.")
        # Compute `self.entries` for the next processing step.
        self.entries = kept

    def _is_preprocessed_valid(self, video_path):
        # Compute `cached` for the next processing step.
        cached = self._valid_cache.get(video_path)
        # Branch behavior based on the current runtime condition.
        if cached is not None:
            # Return the result expected by the caller.
            return cached

        # Compute `video_id` for the next processing step.
        video_id = make_video_id(video_path)
        # Compute `base_dir` for the next processing step.
        base_dir = os.path.join(self.processed_root, video_id)
        # Compute `meta_path` for the next processing step.
        meta_path = os.path.join(base_dir, "meta.json")
        # Branch behavior based on the current runtime condition.
        if not os.path.exists(meta_path):
            # Compute `self._valid_cache[video_path]` for the next processing step.
            self._valid_cache[video_path] = False
            # Invoke `print` to advance this processing stage.
            print(f"[VideoDataset] Missing preprocessed data: {video_path}")
            # Return the result expected by the caller.
            return False

        # Compute `self._valid_cache[video_path]` for the next processing step.
        self._valid_cache[video_path] = True
        # Return the result expected by the caller.
        return True

    def __len__(self):
        # Return the result expected by the caller.
        return len(self.entries)

    def _augment_indices(self, num_frames):
        # Compute `cfg` for the next processing step.
        cfg = self.aug_cfg
        # Compute `effective_length` for the next processing step.
        effective_length = self.sequence_length
        # Branch behavior based on the current runtime condition.
        if self.is_training and random.random() < cfg["p_speed_perturb"]:
            # Compute `speed` for the next processing step.
            speed = random.uniform(*cfg["speed_range"])
            # Compute `effective_length` for the next processing step.
            effective_length = int(self.sequence_length * speed)
            # Compute `effective_length` for the next processing step.
            effective_length = max(effective_length, 4)

        # Compute `indices` for the next processing step.
        indices = np.linspace(0, num_frames - 1, effective_length).astype(int)

        # Branch behavior based on the current runtime condition.
        if self.is_training and random.random() < cfg["p_temporal_jitter"]:
            # Compute `jitter` for the next processing step.
            jitter = np.random.randint(-cfg["jitter_range"], cfg["jitter_range"] + 1,
                                       size=len(indices))
            # Compute `indices` for the next processing step.
            indices = np.clip(indices + jitter, 0, num_frames - 1)

        # Branch behavior based on the current runtime condition.
        if effective_length != self.sequence_length:
            # Compute `resample_idx` for the next processing step.
            resample_idx = np.linspace(0, len(indices) - 1, self.sequence_length).astype(int)
            # Compute `indices` for the next processing step.
            indices = indices[resample_idx]

        # Return the result expected by the caller.
        return indices

    def _sample_spatial_aug_params(self):
        # Compute `cfg` for the next processing step.
        cfg = self.aug_cfg
        # Return the result expected by the caller.
        return {
            "do_flip": random.random() < cfg["p_flip"],
            "angle": random.uniform(-cfg["rotation_range"], cfg["rotation_range"]),
            "brightness": random.uniform(*cfg["brightness_range"]),
            "contrast": random.uniform(*cfg["contrast_range"]),
            "saturation": random.uniform(*cfg["saturation_range"]),
            "hue": random.uniform(*cfg["hue_range"]),
            "do_affine": random.random() < cfg["p_affine"],
            "scale": random.uniform(*cfg["affine_scale"]),
            "translate_x": random.uniform(-cfg["affine_translate"], cfg["affine_translate"]),
            "translate_y": random.uniform(-cfg["affine_translate"], cfg["affine_translate"]),
        }

    def _apply_spatial_aug(self, img, params, apply_color=True):
        # Branch behavior based on the current runtime condition.
        if params["do_flip"]:
            # Compute `img` for the next processing step.
            img = TF.hflip(img)
        # Compute `img` for the next processing step.
        img = TF.rotate(img, params["angle"])
        # Branch behavior based on the current runtime condition.
        if params["do_affine"]:
            # Compute `(w, h)` for the next processing step.
            w, h = img.size
            # Compute `img` for the next processing step.
            img = TF.affine(
                img,
                angle=0,
                translate=(int(params["translate_x"] * w), int(params["translate_y"] * h)),
                scale=params["scale"],
                shear=0,
            )
        # Branch behavior based on the current runtime condition.
        if apply_color:
            # Compute `img` for the next processing step.
            img = TF.adjust_brightness(img, params["brightness"])
            # Compute `img` for the next processing step.
            img = TF.adjust_contrast(img, params["contrast"])
            # Compute `img` for the next processing step.
            img = TF.adjust_saturation(img, params["saturation"])
            # Compute `img` for the next processing step.
            img = TF.adjust_hue(img, params["hue"])
        # Return the result expected by the caller.
        return img

    def _apply_tensor_aug(self, tensor):
        # Compute `cfg` for the next processing step.
        cfg = self.aug_cfg
        # Branch behavior based on the current runtime condition.
        if random.random() < cfg["p_erasing"]:
            # Compute `eraser` for the next processing step.
            eraser = transforms.RandomErasing(
                p=1.0,
                scale=cfg["erasing_scale"],
                ratio=(0.3, 3.3),
                value=0,
            )
            # Compute `tensor` for the next processing step.
            tensor = eraser(tensor)

        # Branch behavior based on the current runtime condition.
        if random.random() < cfg["p_gaussian_noise"]:
            # Compute `noise` for the next processing step.
            noise = torch.randn_like(tensor) * cfg["noise_std"]
            # Compute `tensor` for the next processing step.
            tensor = tensor + noise
        # Return the result expected by the caller.
        return tensor

    def _apply_temporal_masking(self, face_stack, pose_stack, mask):
        # Compute `cfg` for the next processing step.
        cfg = self.aug_cfg
        # Branch behavior based on the current runtime condition.
        if not self.is_training or random.random() >= cfg["p_temporal_mask"]:
            # Return the result expected by the caller.
            return face_stack, pose_stack, mask

        # Compute `T` for the next processing step.
        T = face_stack.shape[0]
        # Compute `mask_ratio` for the next processing step.
        mask_ratio = random.uniform(*cfg["temporal_mask_ratio"])
        # Compute `n_mask` for the next processing step.
        n_mask = max(1, int(T * mask_ratio))

        # Branch behavior based on the current runtime condition.
        if random.random() < 0.5:
            # Compute `start` for the next processing step.
            start = random.randint(0, T - n_mask)
            # Compute `mask_indices` for the next processing step.
            mask_indices = list(range(start, start + n_mask))
        else:
            # Compute `mask_indices` for the next processing step.
            mask_indices = sorted(random.sample(range(T), n_mask))

        # Iterate `idx` across `mask_indices` to process each element.
        for idx in mask_indices:
            # Compute `face_stack[idx]` for the next processing step.
            face_stack[idx] = 0.0
            # Compute `pose_stack[idx]` for the next processing step.
            pose_stack[idx] = 0.0
            # Compute `mask[idx]` for the next processing step.
            mask[idx] = 0.0
        # Return the result expected by the caller.
        return face_stack, pose_stack, mask

    def _apply_quality_noise(self, face_scores, pose_scores, hand_scores):
        # Compute `cfg` for the next processing step.
        cfg = self.aug_cfg
        # Branch behavior based on the current runtime condition.
        if not self.is_training or random.random() >= cfg["p_quality_noise"]:
            # Return the result expected by the caller.
            return face_scores, pose_scores, hand_scores
        # Compute `noise_range` for the next processing step.
        noise_range = cfg["quality_noise_range"]
        # Iterate `scores` across `[face_scores, pose_scores, hand_s...` to process each element.
        for scores in [face_scores, pose_scores, hand_scores]:
            # Iterate `i` across `range(len(scores))` to process each element.
            for i in range(len(scores)):
                # Compute `scores[i]` for the next processing step.
                scores[i] = max(0.0, min(1.0,
                    scores[i] + random.uniform(-noise_range, noise_range)
                ))
        # Return the result expected by the caller.
        return face_scores, pose_scores, hand_scores

    def __getitem__(self, idx):
        # Compute `entry` for the next processing step.
        entry = self.entries[idx]
        # Compute `video_path` for the next processing step.
        video_path = entry["video_path"]
        # Compute `label` for the next processing step.
        label = entry["label"]

        # Compute `cached` for the next processing step.
        cached = self._cache.get(video_path) if self._cache is not None else None
        # Branch behavior based on the current runtime condition.
        if cached is None:
            # Compute `cached` for the next processing step.
            cached = self._preprocess_video(video_path)
            # Branch behavior based on the current runtime condition.
            if self._cache is not None:
                # Compute `self._cache[video_path]` for the next processing step.
                self._cache[video_path] = cached
        # Compute `frames` for the next processing step.
        frames = cached.get("frames", [])
        # Compute `route` for the next processing step.
        route = cached.get("route", "video")

        # Compute `num_frames` for the next processing step.
        num_frames = max(len(frames), 1)

        # Branch behavior based on the current runtime condition.
        if self.is_training:
            # Compute `indices` for the next processing step.
            indices = self._augment_indices(num_frames)
        else:
            # Compute `indices` for the next processing step.
            indices = np.linspace(0, num_frames - 1, self.sequence_length).astype(int)

        # Compute `spatial_params` for the next processing step.
        spatial_params = self._sample_spatial_aug_params() if self.is_training else None

        # Compute `(face_tensors, pose_tensors)` for the next processing step.
        face_tensors, pose_tensors = [], []
        # Compute `(face_scores, pose_scores, hand_s...` for the next processing step.
        face_scores, pose_scores, hand_scores = [], [], []
        # Compute `timestamps` for the next processing step.
        timestamps = []

        # Iterate `i` across `indices` to process each element.
        for i in indices:
            # Branch behavior based on the current runtime condition.
            if i < len(frames):
                # Compute `fd` for the next processing step.
                fd = frames[i]

                # Compute `face_img` for the next processing step.
                face_img = fd.get("face_crop")
                # Branch behavior based on the current runtime condition.
                if face_img is None and fd.get("face_path"):
                    # Compute `face_img` for the next processing step.
                    face_img = cv2.imread(fd["face_path"])
                # Branch behavior based on the current runtime condition.
                if face_img is not None:
                    # Compute `img` for the next processing step.
                    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    # Branch behavior based on the current runtime condition.
                    if spatial_params:
                        # Compute `img` for the next processing step.
                        img = self._apply_spatial_aug(img, spatial_params, apply_color=True)
                    # Compute `t` for the next processing step.
                    t = self.transform(img)
                    # Branch behavior based on the current runtime condition.
                    if self.is_training:
                        # Compute `t` for the next processing step.
                        t = self._apply_tensor_aug(t)
                    # Invoke `face_tensors.append` to advance this processing stage.
                    face_tensors.append(t)
                else:
                    # Invoke `face_tensors.append` to advance this processing stage.
                    face_tensors.append(torch.zeros(3, 224, 224))

                # Compute `pose_img` for the next processing step.
                pose_img = fd.get("skeleton_img")
                # Branch behavior based on the current runtime condition.
                if pose_img is None and fd.get("skeleton_path"):
                    # Compute `pose_img` for the next processing step.
                    pose_img = cv2.imread(fd["skeleton_path"])
                # Branch behavior based on the current runtime condition.
                if pose_img is not None:
                    # Compute `img` for the next processing step.
                    img = Image.fromarray(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB))
                    # Branch behavior based on the current runtime condition.
                    if spatial_params:
                        # Compute `img` for the next processing step.
                        img = self._apply_spatial_aug(img, spatial_params, apply_color=False)
                    # Compute `t` for the next processing step.
                    t = self.transform(img)
                    # Branch behavior based on the current runtime condition.
                    if self.is_training:
                        # Compute `t` for the next processing step.
                        t = self._apply_tensor_aug(t)
                    # Invoke `pose_tensors.append` to advance this processing stage.
                    pose_tensors.append(t)
                else:
                    # Invoke `pose_tensors.append` to advance this processing stage.
                    pose_tensors.append(torch.zeros(3, 224, 224))

                # Compute `q` for the next processing step.
                q = fd.get("quality", {})
                # Invoke `face_scores.append` to advance this processing stage.
                face_scores.append(q.get("face_score", 0.5))
                # Invoke `pose_scores.append` to advance this processing stage.
                pose_scores.append(q.get("pose_score", 0.5))
                # Invoke `hand_scores.append` to advance this processing stage.
                hand_scores.append(q.get("hand_score", 0.0))
                # Invoke `timestamps.append` to advance this processing stage.
                timestamps.append(float(fd.get("timestamp", 0.0)))
            else:
                # Invoke `face_tensors.append` to advance this processing stage.
                face_tensors.append(torch.zeros(3, 224, 224))
                # Invoke `pose_tensors.append` to advance this processing stage.
                pose_tensors.append(torch.zeros(3, 224, 224))
                # Invoke `face_scores.append` to advance this processing stage.
                face_scores.append(0.0)
                # Invoke `pose_scores.append` to advance this processing stage.
                pose_scores.append(0.0)
                # Invoke `hand_scores.append` to advance this processing stage.
                hand_scores.append(0.0)
                # Invoke `timestamps.append` to advance this processing stage.
                timestamps.append(0.0)

        # Compute `face_stack` for the next processing step.
        face_stack = torch.stack(face_tensors)
        # Compute `pose_stack` for the next processing step.
        pose_stack = torch.stack(pose_tensors)

        # Motion maps (micro-motion) as abs diff between pose frames
        motion_tensors = [torch.zeros_like(pose_stack[0])]
        # Iterate `i` across `range(1, pose_stack.shape[0])` to process each element.
        for i in range(1, pose_stack.shape[0]):
            # Invoke `motion_tensors.append` to advance this processing stage.
            motion_tensors.append((pose_stack[i] - pose_stack[i - 1]).abs())
        # Compute `motion_stack` for the next processing step.
        motion_stack = torch.stack(motion_tensors)

        # Compute `mask` for the next processing step.
        mask = torch.ones(self.sequence_length)
        # Branch behavior based on the current runtime condition.
        if len(frames) < self.sequence_length:
            # Compute `mask[len(frames):]` for the next processing step.
            mask[len(frames):] = 0

        # Compute `(face_stack, pose_stack, mask)` for the next processing step.
        face_stack, pose_stack, mask = self._apply_temporal_masking(
            face_stack, pose_stack, mask
        )

        # Compute `(face_scores, pose_scores, hand_s...` for the next processing step.
        face_scores, pose_scores, hand_scores = self._apply_quality_noise(
            face_scores, pose_scores, hand_scores
        )

        # Compute `timestamps` for the next processing step.
        timestamps = np.array(timestamps, dtype=np.float32)
        # Compute `delta_t` for the next processing step.
        delta_t = np.zeros_like(timestamps)
        # Branch behavior based on the current runtime condition.
        if len(timestamps) > 1:
            # Compute `delta_t[1:]` for the next processing step.
            delta_t[1:] = timestamps[1:] - timestamps[:-1]
            # Compute `delta_t` for the next processing step.
            delta_t = np.clip(delta_t, 0.0, None)

        # Compute `use_video` for the next processing step.
        use_video = 1.0 if route == "video" else 0.0

        # Return the result expected by the caller.
        return {
            "face_crops": face_stack,
            "pose_maps": pose_stack,
            "motion_maps": motion_stack,
            "mask": mask,
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "delta_t": torch.tensor(delta_t, dtype=torch.float32),
            "qualities": {
                "face_score": torch.tensor(face_scores, dtype=torch.float32),
                "pose_score": torch.tensor(pose_scores, dtype=torch.float32),
                "hand_score": torch.tensor(hand_scores, dtype=torch.float32),
            },
            "route_mask": torch.tensor(use_video, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "video_id": os.path.basename(video_path),
            "subject_id": entry["subject_id"],
        }

