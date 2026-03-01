# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

# -- Suppress noisy C++ / TFLite / absl warnings BEFORE any imports --
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import contextlib
import sys

import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore", message=".*_POSIX_C_SOURCE.*")
warnings.filterwarnings("ignore", message=".*Feedback manager.*")
warnings.filterwarnings("ignore", message=".*landmark_projection_calculator.*")


@contextlib.contextmanager
def suppress_stderr():
    """
    Redirects stderr (fd 2) to /dev/null to silence native C++ libraries (MediaPipe/TF).
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

# -- MediaPipe imports --
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# -------- MODEL PATHS --------
ROOT = Path(__file__).resolve().parents[4]
FACE_MODEL = os.environ.get("ASDMOTION_FACE_MODEL", str(ROOT / "assets" / "video" / "face_landmarker.task"))
POSE_MODEL = os.environ.get("ASDMOTION_POSE_MODEL", str(ROOT / "assets" / "video" / "pose_landmarker_full.task"))
DISABLE = os.environ.get("ASDMOTION_DISABLE_MEDIAPIPE", "0") == "1"


def _is_placeholder(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return True
        if os.path.getsize(path) < 1024:
            with open(path, "rb") as f:
                head = f.read(64)
            if b"ASDMOTION_PLACEHOLDER" in head:
                return True
            return True
    except Exception:
        return True
    return False


if not DISABLE:
    if _is_placeholder(FACE_MODEL) or _is_placeholder(POSE_MODEL):
        print(
            "[MediaPipe] Model files not found or placeholders detected.\n"
            "           Set ASDMOTION_FACE_MODEL/ASDMOTION_POSE_MODEL\n"
            "           or replace assets/video/*.task. Falling back to no landmarks."
        )
        DISABLE = True

# -------- LANDMARKERS --------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_landmarker = None
pose_landmarker = None

if not DISABLE:
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=False,
        num_faces=1,
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)


def extract_landmarks(frame):
    """
    Input:
        frame: BGR image (OpenCV)
    Output:
        face_landmarks: list of landmarks OR None
        pose_landmarks: list of 33 landmarks OR None
    """
    if DISABLE or face_landmarker is None or pose_landmarker is None:
        return None, None

    rgb = frame[:, :, ::-1]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    with suppress_stderr():
        face_result = face_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)

    face_landmarks = (
        face_result.face_landmarks[0]
        if face_result.face_landmarks and len(face_result.face_landmarks) > 0
        else None
    )

    pose_landmarks = (
        pose_result.pose_landmarks[0]
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0
        else None
    )

    return face_landmarks, pose_landmarks

