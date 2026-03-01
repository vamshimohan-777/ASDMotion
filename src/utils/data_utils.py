# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms


def default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def prepare_sequence_from_frames(frames, seq_len=32, transform=None):
    if transform is None:
        transform = default_transform()

    num_frames = max(len(frames), 1)
    indices = np.linspace(0, num_frames - 1, seq_len).astype(int)

    face_tensors, pose_tensors = [], []
    face_scores, pose_scores, hand_scores = [], [], []
    timestamps = []

    for i in indices:
        if i < len(frames):
            fd = frames[i]
            # Face
            if fd.get("face_crop") is not None:
                img = Image.fromarray(cv2.cvtColor(fd["face_crop"], cv2.COLOR_BGR2RGB))
                face_tensors.append(transform(img))
            else:
                face_tensors.append(torch.zeros(3, 224, 224))

            # Pose
            if fd.get("skeleton_img") is not None:
                img = Image.fromarray(cv2.cvtColor(fd["skeleton_img"], cv2.COLOR_BGR2RGB))
                pose_tensors.append(transform(img))
            else:
                pose_tensors.append(torch.zeros(3, 224, 224))

            q = fd.get("quality", {})
            face_scores.append(float(q.get("face_score", 0.5)))
            pose_scores.append(float(q.get("pose_score", 0.5)))
            hand_scores.append(float(q.get("hand_score", 0.0)))

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

    # Motion maps: abs diff between consecutive pose frames
    motion_tensors = [torch.zeros_like(pose_stack[0])]
    for i in range(1, pose_stack.shape[0]):
        motion_tensors.append((pose_stack[i] - pose_stack[i - 1]).abs())
    motion_stack = torch.stack(motion_tensors)

    mask = torch.ones(seq_len)
    if len(frames) < seq_len:
        mask[len(frames):] = 0

    timestamps = np.array(timestamps, dtype=np.float32)
    delta_t = np.zeros_like(timestamps)
    if len(timestamps) > 1:
        delta_t[1:] = timestamps[1:] - timestamps[:-1]
        delta_t = np.clip(delta_t, 0.0, None)

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
        }
    }

