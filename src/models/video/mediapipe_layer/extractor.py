"""MediaPipe holistic landmark extractor."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA, FACE_KEYPOINTS_60

_LOCK = threading.Lock()
_HOLISTIC = None


def _get_holistic():
    global _HOLISTIC
    with _LOCK:
        if _HOLISTIC is None:
            import mediapipe as mp

            _HOLISTIC = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return _HOLISTIC


def _fill_landmarks(dst_xyz, dst_mask, start_idx, lms, expected_count):
    if lms is None:
        return 0
    points = list(lms.landmark)
    n = min(int(expected_count), len(points))
    for i in range(n):
        p = points[i]
        dst_xyz[start_idx + i, 0] = float(p.x)
        dst_xyz[start_idx + i, 1] = float(p.y)
        dst_xyz[start_idx + i, 2] = float(p.z)
        dst_mask[start_idx + i] = 1.0
    return n


def extract_holistic_landmarks(frame_bgr, schema=DEFAULT_SCHEMA):
    """
    Returns:
      xyz: [J,3] float32
      mask: [J] float32
      meta: dict with modality quality fields
    """
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR frame [H,W,3], got {tuple(frame.shape)}")

    hls = _get_holistic()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hls.process(rgb)

    j = int(schema.total_joints)
    xyz = np.zeros((j, 3), dtype=np.float32)
    mask = np.zeros((j,), dtype=np.float32)

    pose_n = _fill_landmarks(
        xyz,
        mask,
        start_idx=0,
        lms=result.pose_landmarks,
        expected_count=int(schema.pose_joints),
    )
    left_n = _fill_landmarks(
        xyz,
        mask,
        start_idx=int(schema.left_hand_slice.start),
        lms=result.left_hand_landmarks,
        expected_count=int(schema.hand_joints),
    )
    right_n = _fill_landmarks(
        xyz,
        mask,
        start_idx=int(schema.right_hand_slice.start),
        lms=result.right_hand_landmarks,
        expected_count=int(schema.hand_joints),
    )

    face_start = int(schema.face_slice.start)
    face_n = 0
    if result.face_landmarks is not None:
        pts = list(result.face_landmarks.landmark)
        for i, src_idx in enumerate(FACE_KEYPOINTS_60[: int(schema.face_joints)]):
            if 0 <= int(src_idx) < len(pts):
                p = pts[int(src_idx)]
                xyz[face_start + i, 0] = float(p.x)
                xyz[face_start + i, 1] = float(p.y)
                xyz[face_start + i, 2] = float(p.z)
                mask[face_start + i] = 1.0
                face_n += 1

    pose_score = float(pose_n / max(int(schema.pose_joints), 1))
    hand_score = float((left_n + right_n) / max(int(schema.hand_joints) * 2, 1))
    face_score = float(face_n / max(int(schema.face_joints), 1))
    overall = float((pose_score + hand_score + face_score) / 3.0)

    meta = {
        "modality_quality": {
            "pose": pose_score,
            "hands": hand_score,
            "face": face_score,
        },
        "overall_quality": overall,
    }
    return xyz, mask, meta
