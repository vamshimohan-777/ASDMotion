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
        self.sequence_length = sequence_length
        self.is_training = is_training
        self.precompute = precompute
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.validate_videos = validate_videos
        self.cache_enabled = cache_enabled
        self.use_preprocessed = use_preprocessed
        self.processed_root = processed_root
        self.preprocessed_only = preprocessed_only
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
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "subject_id" not in reader.fieldnames:
                raise ValueError(
                    "CSV is missing subject_id. Group-aware splitting is required to prevent "
                    "leakage. Leakage makes clinical metrics invalid."
                )
            for row in reader:
                vpath = row["video_path"].strip()
                label = float(row["label"])
                subject_id = row["subject_id"].strip()
                self.entries.append({
                    "video_path": vpath,
                    "label": label,
                    "subject_id": subject_id,
                })

        self._processor = None
        if self.cache_enabled:
            self._cache = shared_cache if shared_cache is not None else {}
            self._valid_cache = self._cache.setdefault(("__meta__", "valid"), {})
        else:
            self._cache = None
            self._valid_cache = {}

        if self.validate_videos:
            self._filter_invalid_entries()

        if self.precompute and self.cache_enabled:
            self._precompute_all()
        elif self.precompute and not self.cache_enabled:
            print("[VideoDataset] cache_precompute ignored because cache_enabled=false.")

    @property
    def processor(self):
        if self._processor is None:
            from src.pipeline.preprocess import VideoProcessor
            self._processor = VideoProcessor(
                frame_stride=self.frame_stride,
                max_frames=self.max_frames,
            )
        return self._processor

    def _preprocess_video(self, video_path):
        if self.use_preprocessed:
            result = self._load_preprocessed_video(video_path)
            if result is not None:
                return result
            if self.preprocessed_only:
                print(f"[VideoDataset] Missing preprocessed data for {video_path}")
                return {"frames": [], "route": "video"}

        try:
            print(f"[VideoDataset] Processing: {video_path}")
            result = self.processor.process_video_file(video_path)
            frames = result.get("frames", [])
            route = result.get("route", "video")
            return {"frames": frames, "route": route}
        except Exception as e:
            print(f"[VideoDataset] Error processing {video_path}: {e}")
            return {"frames": [], "route": "video"}

    def _load_preprocessed_video(self, video_path):
        video_id = make_video_id(video_path)
        base_dir = os.path.join(self.processed_root, video_id)
        meta_path = os.path.join(base_dir, "meta.json")
        quality_path = os.path.join(base_dir, "quality.json")

        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[VideoDataset] Failed to read meta for {video_path}: {e}")
            return None

        qualities = []
        if os.path.exists(quality_path):
            try:
                with open(quality_path, "r") as f:
                    qualities = json.load(f)
            except Exception as e:
                print(f"[VideoDataset] Failed to read quality for {video_path}: {e}")

        frame_ids = meta.get("frame_ids", [])
        timestamps = meta.get("timestamps", [])
        route = meta.get("route", "video")

        frames = []
        for i, frame_id in enumerate(frame_ids):
            ts = timestamps[i] if i < len(timestamps) else float(frame_id)
            q = qualities[i] if i < len(qualities) else {}
            face_path = os.path.join(base_dir, "faces", f"{frame_id:06d}.png")
            skeleton_path = os.path.join(base_dir, "skeletons", f"{frame_id:06d}.png")
            frames.append({
                "frame_id": frame_id,
                "timestamp": ts,
                "face_path": face_path,
                "skeleton_path": skeleton_path,
                "quality": q,
            })

        return {"frames": frames, "route": route}

    def _precompute_all(self):
        print(f"[VideoDataset] Precomputing {len(self.entries)} videos...")
        for entry in self.entries:
            video_path = entry["video_path"]
            if self._cache is not None and video_path in self._cache:
                continue
            if self._cache is not None:
                self._cache[video_path] = self._preprocess_video(video_path)
        print("[VideoDataset] Precompute complete.")

    def _open_capture(self, video_path):
        candidates = []
        for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_DSHOW"):
            if hasattr(cv2, name):
                candidates.append(getattr(cv2, name))
        candidates.append(None)

        for backend in candidates:
            try:
                cap = cv2.VideoCapture(video_path) if backend is None else cv2.VideoCapture(video_path, backend)
            except Exception:
                continue
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return None

    def _is_video_valid(self, video_path):
        cached = self._valid_cache.get(video_path)
        if cached is not None:
            return cached

        if not os.path.exists(video_path):
            self._valid_cache[video_path] = False
            print(f"[VideoDataset] Missing file: {video_path}")
            return False

        cap = self._open_capture(video_path)
        if cap is None:
            self._valid_cache[video_path] = False
            print(f"[VideoDataset] Failed to open: {video_path}")
            return False

        try:
            try:
                ok, frame = cap.read()
            except Exception as e:
                print(f"[VideoDataset] Read failed for {video_path}: {e}")
                ok, frame = False, None
        finally:
            cap.release()

        valid = bool(ok and frame is not None)
        self._valid_cache[video_path] = valid
        if not valid:
            print(f"[VideoDataset] Unreadable video: {video_path}")
        return valid

    def _filter_invalid_entries(self):
        kept = []
        bad = []
        for entry in self.entries:
            vpath = entry["video_path"]
            if self.use_preprocessed:
                ok = self._is_preprocessed_valid(vpath)
            else:
                ok = self._is_video_valid(vpath)
            if ok:
                kept.append(entry)
            else:
                bad.append(vpath)

        if bad:
            print(f"[VideoDataset] Skipping {len(bad)} unreadable videos.")
        self.entries = kept

    def _is_preprocessed_valid(self, video_path):
        cached = self._valid_cache.get(video_path)
        if cached is not None:
            return cached

        video_id = make_video_id(video_path)
        base_dir = os.path.join(self.processed_root, video_id)
        meta_path = os.path.join(base_dir, "meta.json")
        if not os.path.exists(meta_path):
            self._valid_cache[video_path] = False
            print(f"[VideoDataset] Missing preprocessed data: {video_path}")
            return False

        self._valid_cache[video_path] = True
        return True

    def __len__(self):
        return len(self.entries)

    def _augment_indices(self, num_frames):
        cfg = self.aug_cfg
        effective_length = self.sequence_length
        if self.is_training and random.random() < cfg["p_speed_perturb"]:
            speed = random.uniform(*cfg["speed_range"])
            effective_length = int(self.sequence_length * speed)
            effective_length = max(effective_length, 4)

        indices = np.linspace(0, num_frames - 1, effective_length).astype(int)

        if self.is_training and random.random() < cfg["p_temporal_jitter"]:
            jitter = np.random.randint(-cfg["jitter_range"], cfg["jitter_range"] + 1,
                                       size=len(indices))
            indices = np.clip(indices + jitter, 0, num_frames - 1)

        if effective_length != self.sequence_length:
            resample_idx = np.linspace(0, len(indices) - 1, self.sequence_length).astype(int)
            indices = indices[resample_idx]

        return indices

    def _sample_spatial_aug_params(self):
        cfg = self.aug_cfg
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
        if params["do_flip"]:
            img = TF.hflip(img)
        img = TF.rotate(img, params["angle"])
        if params["do_affine"]:
            w, h = img.size
            img = TF.affine(
                img,
                angle=0,
                translate=(int(params["translate_x"] * w), int(params["translate_y"] * h)),
                scale=params["scale"],
                shear=0,
            )
        if apply_color:
            img = TF.adjust_brightness(img, params["brightness"])
            img = TF.adjust_contrast(img, params["contrast"])
            img = TF.adjust_saturation(img, params["saturation"])
            img = TF.adjust_hue(img, params["hue"])
        return img

    def _apply_tensor_aug(self, tensor):
        cfg = self.aug_cfg
        if random.random() < cfg["p_erasing"]:
            eraser = transforms.RandomErasing(
                p=1.0,
                scale=cfg["erasing_scale"],
                ratio=(0.3, 3.3),
                value=0,
            )
            tensor = eraser(tensor)

        if random.random() < cfg["p_gaussian_noise"]:
            noise = torch.randn_like(tensor) * cfg["noise_std"]
            tensor = tensor + noise
        return tensor

    def _apply_temporal_masking(self, face_stack, pose_stack, mask):
        cfg = self.aug_cfg
        if not self.is_training or random.random() >= cfg["p_temporal_mask"]:
            return face_stack, pose_stack, mask

        T = face_stack.shape[0]
        mask_ratio = random.uniform(*cfg["temporal_mask_ratio"])
        n_mask = max(1, int(T * mask_ratio))

        if random.random() < 0.5:
            start = random.randint(0, T - n_mask)
            mask_indices = list(range(start, start + n_mask))
        else:
            mask_indices = sorted(random.sample(range(T), n_mask))

        for idx in mask_indices:
            face_stack[idx] = 0.0
            pose_stack[idx] = 0.0
            mask[idx] = 0.0
        return face_stack, pose_stack, mask

    def _apply_quality_noise(self, face_scores, pose_scores, hand_scores):
        cfg = self.aug_cfg
        if not self.is_training or random.random() >= cfg["p_quality_noise"]:
            return face_scores, pose_scores, hand_scores
        noise_range = cfg["quality_noise_range"]
        for scores in [face_scores, pose_scores, hand_scores]:
            for i in range(len(scores)):
                scores[i] = max(0.0, min(1.0,
                    scores[i] + random.uniform(-noise_range, noise_range)
                ))
        return face_scores, pose_scores, hand_scores

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video_path = entry["video_path"]
        label = entry["label"]

        cached = self._cache.get(video_path) if self._cache is not None else None
        if cached is None:
            cached = self._preprocess_video(video_path)
            if self._cache is not None:
                self._cache[video_path] = cached
        frames = cached.get("frames", [])
        route = cached.get("route", "video")

        num_frames = max(len(frames), 1)

        if self.is_training:
            indices = self._augment_indices(num_frames)
        else:
            indices = np.linspace(0, num_frames - 1, self.sequence_length).astype(int)

        spatial_params = self._sample_spatial_aug_params() if self.is_training else None

        face_tensors, pose_tensors = [], []
        face_scores, pose_scores, hand_scores = [], [], []
        timestamps = []

        for i in indices:
            if i < len(frames):
                fd = frames[i]

                face_img = fd.get("face_crop")
                if face_img is None and fd.get("face_path"):
                    face_img = cv2.imread(fd["face_path"])
                if face_img is not None:
                    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    if spatial_params:
                        img = self._apply_spatial_aug(img, spatial_params, apply_color=True)
                    t = self.transform(img)
                    if self.is_training:
                        t = self._apply_tensor_aug(t)
                    face_tensors.append(t)
                else:
                    face_tensors.append(torch.zeros(3, 224, 224))

                pose_img = fd.get("skeleton_img")
                if pose_img is None and fd.get("skeleton_path"):
                    pose_img = cv2.imread(fd["skeleton_path"])
                if pose_img is not None:
                    img = Image.fromarray(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB))
                    if spatial_params:
                        img = self._apply_spatial_aug(img, spatial_params, apply_color=False)
                    t = self.transform(img)
                    if self.is_training:
                        t = self._apply_tensor_aug(t)
                    pose_tensors.append(t)
                else:
                    pose_tensors.append(torch.zeros(3, 224, 224))

                q = fd.get("quality", {})
                face_scores.append(q.get("face_score", 0.5))
                pose_scores.append(q.get("pose_score", 0.5))
                hand_scores.append(q.get("hand_score", 0.0))
                timestamps.append(float(fd.get("timestamp", 0.0)))
            else:
                face_tensors.append(torch.zeros(3, 224, 224))
                pose_tensors.append(torch.zeros(3, 224, 224))
                face_scores.append(0.0)
                pose_scores.append(0.0)
                hand_scores.append(0.0)
                timestamps.append(0.0)

        face_stack = torch.stack(face_tensors)
        pose_stack = torch.stack(pose_tensors)

        # Motion maps (micro-motion) as abs diff between pose frames
        motion_tensors = [torch.zeros_like(pose_stack[0])]
        for i in range(1, pose_stack.shape[0]):
            motion_tensors.append((pose_stack[i] - pose_stack[i - 1]).abs())
        motion_stack = torch.stack(motion_tensors)

        mask = torch.ones(self.sequence_length)
        if len(frames) < self.sequence_length:
            mask[len(frames):] = 0

        face_stack, pose_stack, mask = self._apply_temporal_masking(
            face_stack, pose_stack, mask
        )

        face_scores, pose_scores, hand_scores = self._apply_quality_noise(
            face_scores, pose_scores, hand_scores
        )

        timestamps = np.array(timestamps, dtype=np.float32)
        delta_t = np.zeros_like(timestamps)
        if len(timestamps) > 1:
            delta_t[1:] = timestamps[1:] - timestamps[:-1]
            delta_t = np.clip(delta_t, 0.0, None)

        use_video = 1.0 if route == "video" else 0.0

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

