import contextlib
import logging
import math
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA, FACE_KEYPOINTS_60


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")

logging.getLogger("mediapipe").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*_POSIX_C_SOURCE.*")
warnings.filterwarnings("ignore", message=".*Feedback manager.*")
warnings.filterwarnings("ignore", message=".*landmark_projection_calculator.*")


@contextlib.contextmanager
def suppress_stderr():
    old_stderr = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            if old_stderr is not None:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
        except Exception:
            pass


with suppress_stderr():
    try:
        import mediapipe as mp
    except Exception:
        mp = None


ROOT = Path(__file__).resolve().parents[4]
POSE_MODEL = os.environ.get("ASDMOTION_POSE_MODEL", str(ROOT / "assets" / "video" / "pose_landmarker_full.task"))
FACE_MODEL = os.environ.get("ASDMOTION_FACE_MODEL", str(ROOT / "assets" / "video" / "face_landmarker.task"))
HAND_MODEL = os.environ.get("ASDMOTION_HAND_MODEL", str(ROOT / "assets" / "video" / "hand_landmarker.task"))

DISABLE = os.environ.get("ASDMOTION_DISABLE_MEDIAPIPE", "0") == "1"
MODEL_COMPLEXITY = int(os.environ.get("ASDMOTION_HOLISTIC_MODEL_COMPLEXITY", "1"))
REFINE_FACE = os.environ.get("ASDMOTION_HOLISTIC_REFINE_FACE", "1") != "0"
MIN_DET_CONF = float(os.environ.get("ASDMOTION_HOLISTIC_MIN_DET_CONF", "0.5"))
MIN_TRACK_CONF = float(os.environ.get("ASDMOTION_HOLISTIC_MIN_TRACK_CONF", "0.5"))


def _is_placeholder(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return True
        if os.path.getsize(path) < 1024:
            with open(path, "rb") as f:
                head = f.read(64)
            return b"ASDMOTION_PLACEHOLDER" in head or True
    except Exception:
        return True
    return False


def _create_holistic_solution():
    if DISABLE or mp is None:
        return None
    if not hasattr(mp, "solutions"):
        return None
    if not hasattr(mp.solutions, "holistic"):
        return None
    with suppress_stderr():
        return mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=max(0, min(MODEL_COMPLEXITY, 2)),
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=bool(REFINE_FACE),
            min_detection_confidence=float(MIN_DET_CONF),
            min_tracking_confidence=float(MIN_TRACK_CONF),
        )


def _create_task_landmarkers():
    if DISABLE or mp is None:
        return None
    if not hasattr(mp, "tasks"):
        return None
    if _is_placeholder(POSE_MODEL) or _is_placeholder(FACE_MODEL):
        return None

    try:
        BaseOptions = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        running_mode = vision.RunningMode.IMAGE
    except Exception:
        return None

    lm = {"pose": None, "face": None, "hand": None}

    with suppress_stderr():
        lm["pose"] = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL),
                running_mode=running_mode,
                num_poses=1,
            )
        )
        lm["face"] = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL),
                running_mode=running_mode,
                output_face_blendshapes=False,
                num_faces=1,
            )
        )
        if not _is_placeholder(HAND_MODEL):
            try:
                lm["hand"] = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=HAND_MODEL),
                        running_mode=running_mode,
                        num_hands=2,
                    )
                )
            except Exception:
                lm["hand"] = None
    return lm


HOLISTIC = _create_holistic_solution()
TASK_LANDMARKERS = None if HOLISTIC is not None else _create_task_landmarkers()


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _landmark_confidence(lm, default=1.0):
    vis = _safe_float(getattr(lm, "visibility", default), default)
    pres = _safe_float(getattr(lm, "presence", default), default)
    return _clip01(0.5 * (vis + pres))


def _valid_xy(x, y):
    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
        return False
    return -0.25 <= float(x) <= 1.25 and -0.25 <= float(y) <= 1.25


def _fill_block(
    out_xyz,
    out_mask,
    start_idx,
    expected_count,
    landmarks,
    source_indices=None,
    has_visibility=False,
):
    if landmarks is None:
        return 0.0

    indices = list(range(expected_count)) if source_indices is None else list(source_indices)
    scores = []
    for local_idx, src_idx in enumerate(indices):
        dst = start_idx + local_idx
        if dst >= out_xyz.shape[0] or src_idx >= len(landmarks):
            continue
        lm = landmarks[src_idx]
        x = _safe_float(getattr(lm, "x", 0.0))
        y = _safe_float(getattr(lm, "y", 0.0))
        z = _safe_float(getattr(lm, "z", 0.0))
        conf = _landmark_confidence(lm, default=1.0 if not has_visibility else 0.0)
        valid = _valid_xy(x, y) and (conf > 0.05)
        if valid:
            out_xyz[dst, 0] = x
            out_xyz[dst, 1] = y
            out_xyz[dst, 2] = z
            out_mask[dst] = 1.0
            scores.append(conf)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _run_holistic(frame_bgr):
    if frame_bgr is None or mp is None:
        return None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if HOLISTIC is not None:
        with suppress_stderr():
            return HOLISTIC.process(rgb)

    if TASK_LANDMARKERS is None:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with suppress_stderr():
        pose_result = TASK_LANDMARKERS["pose"].detect(mp_image) if TASK_LANDMARKERS.get("pose") else None
        face_result = TASK_LANDMARKERS["face"].detect(mp_image) if TASK_LANDMARKERS.get("face") else None
        hand_result = TASK_LANDMARKERS["hand"].detect(mp_image) if TASK_LANDMARKERS.get("hand") else None

    pose_lms = None
    if pose_result is not None and getattr(pose_result, "pose_landmarks", None):
        pose_lms = pose_result.pose_landmarks[0]

    face_lms = None
    if face_result is not None and getattr(face_result, "face_landmarks", None):
        face_lms = face_result.face_landmarks[0]

    left_lms = None
    right_lms = None
    if hand_result is not None and getattr(hand_result, "hand_landmarks", None):
        for idx, hand in enumerate(hand_result.hand_landmarks):
            label = None
            try:
                handed = hand_result.handedness[idx]
                if handed:
                    label = str(handed[0].category_name).lower()
            except Exception:
                label = None
            if label == "left":
                left_lms = hand
            elif label == "right":
                right_lms = hand
            else:
                if left_lms is None:
                    left_lms = hand
                elif right_lms is None:
                    right_lms = hand

    return {
        "pose_landmarks": pose_lms,
        "face_landmarks": face_lms,
        "left_hand_landmarks": left_lms,
        "right_hand_landmarks": right_lms,
    }


def _get_landmark_list(results, name):
    if results is None:
        return None
    if isinstance(results, dict):
        return results.get(name)
    obj = getattr(results, name, None)
    if obj is None:
        return None
    if hasattr(obj, "landmark"):
        return obj.landmark
    return obj


def extract_holistic_landmarks(frame_bgr, schema=DEFAULT_SCHEMA, face_indices=None):
    J = int(schema.total_joints)
    out_xyz = np.zeros((J, 3), dtype=np.float32)
    out_mask = np.zeros((J,), dtype=np.float32)
    face_indices = FACE_KEYPOINTS_60 if face_indices is None else list(face_indices)

    results = _run_holistic(frame_bgr)
    pose_lms = _get_landmark_list(results, "pose_landmarks")
    left_lms = _get_landmark_list(results, "left_hand_landmarks")
    right_lms = _get_landmark_list(results, "right_hand_landmarks")
    face_lms = _get_landmark_list(results, "face_landmarks")

    pose_q = _fill_block(out_xyz, out_mask, schema.pose_slice.start, schema.pose_joints, pose_lms, has_visibility=True)
    left_q = _fill_block(out_xyz, out_mask, schema.left_hand_slice.start, schema.hand_joints, left_lms, has_visibility=False)
    right_q = _fill_block(out_xyz, out_mask, schema.right_hand_slice.start, schema.hand_joints, right_lms, has_visibility=False)
    face_q = _fill_block(
        out_xyz,
        out_mask,
        schema.face_slice.start,
        schema.face_joints,
        face_lms,
        source_indices=face_indices,
        has_visibility=False,
    )

    hands_q = 0.5 * (left_q + right_q)
    tracked_center = None
    if pose_lms is not None and len(pose_lms) > 24:
        lh = pose_lms[23]
        rh = pose_lms[24]
        if _valid_xy(lh.x, lh.y) and _valid_xy(rh.x, rh.y):
            tracked_center = (
                float(0.5 * (_safe_float(lh.x) + _safe_float(rh.x))),
                float(0.5 * (_safe_float(lh.y) + _safe_float(rh.y))),
            )
    if tracked_center is None and face_lms is not None:
        xs = []
        ys = []
        for idx in face_indices:
            if idx >= len(face_lms):
                continue
            lm = face_lms[idx]
            if _valid_xy(lm.x, lm.y):
                xs.append(float(lm.x))
                ys.append(float(lm.y))
        if xs:
            tracked_center = (float(np.mean(xs)), float(np.mean(ys)))

    n_pose = int(out_mask[schema.pose_slice].sum())
    n_hands = int(out_mask[schema.left_hand_slice].sum() + out_mask[schema.right_hand_slice].sum())
    n_face = int(out_mask[schema.face_slice].sum())
    overall_q = float((0.45 * pose_q) + (0.30 * hands_q) + (0.25 * face_q))

    return out_xyz, out_mask, {
        "tracked_center": tracked_center,
        "modality_quality": {"pose": pose_q, "hands": hands_q, "face": face_q},
        "overall_quality": _clip01(overall_q),
        "n_pose": n_pose,
        "n_hands": n_hands,
        "n_face": n_face,
    }


def extract_landmarks(frame, reference_center=None):
    del reference_center
    results = _run_holistic(frame)
    face_landmarks = _get_landmark_list(results, "face_landmarks")
    pose_landmarks = _get_landmark_list(results, "pose_landmarks")

    tracked = None
    if pose_landmarks is not None and len(pose_landmarks) > 24:
        lh = pose_landmarks[23]
        rh = pose_landmarks[24]
        if _valid_xy(lh.x, lh.y) and _valid_xy(rh.x, rh.y):
            tracked = (
                float(0.5 * (_safe_float(lh.x) + _safe_float(rh.x))),
                float(0.5 * (_safe_float(lh.y) + _safe_float(rh.y))),
            )

    return face_landmarks, pose_landmarks, {
        "tracked_center": tracked,
        "n_faces": 1 if face_landmarks is not None else 0,
        "n_poses": 1 if pose_landmarks is not None else 0,
    }

