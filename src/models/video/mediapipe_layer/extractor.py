# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

# -- Suppress noisy C++ / TFLite / absl warnings BEFORE any imports --
import os
# Compute `os.environ['TF_CPP_MIN_LOG_LEVEL']` for the next processing step.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Compute `os.environ['GLOG_minloglevel']` for the next processing step.
os.environ["GLOG_minloglevel"] = "2"

import contextlib
import sys

import logging
# Invoke `logging.getLogger('mediapipe').se...` to advance this processing stage.
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Guard this block and recover cleanly from expected failures.
try:
    import absl.logging
    # Invoke `absl.logging.set_verbosity` to advance this processing stage.
    absl.logging.set_verbosity(absl.logging.ERROR)
    # Invoke `absl.logging.set_stderrthreshold` to advance this processing stage.
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

import warnings
# Invoke `warnings.filterwarnings` to advance this processing stage.
warnings.filterwarnings("ignore", message=".*_POSIX_C_SOURCE.*")
# Invoke `warnings.filterwarnings` to advance this processing stage.
warnings.filterwarnings("ignore", message=".*Feedback manager.*")
# Invoke `warnings.filterwarnings` to advance this processing stage.
warnings.filterwarnings("ignore", message=".*landmark_projection_calculator.*")


@contextlib.contextmanager
def suppress_stderr():
    """
    Redirects stderr (fd 2) to /dev/null to silence native C++ libraries (MediaPipe/TF).
    """
    # Guard this block and recover cleanly from expected failures.
    try:
        # Compute `devnull` for the next processing step.
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Compute `old_stderr` for the next processing step.
        old_stderr = os.dup(2)
        # Invoke `sys.stderr.flush` to advance this processing stage.
        sys.stderr.flush()
        # Invoke `os.dup2` to advance this processing stage.
        os.dup2(devnull, 2)
        # Invoke `os.close` to advance this processing stage.
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        # Guard this block and recover cleanly from expected failures.
        try:
            # Invoke `os.dup2` to advance this processing stage.
            os.dup2(old_stderr, 2)
            # Invoke `os.close` to advance this processing stage.
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
# Compute `FACE_MODEL` for the next processing step.
FACE_MODEL = os.environ.get("ASDMOTION_FACE_MODEL", str(ROOT / "assets" / "video" / "face_landmarker.task"))
# Compute `POSE_MODEL` for the next processing step.
POSE_MODEL = os.environ.get("ASDMOTION_POSE_MODEL", str(ROOT / "assets" / "video" / "pose_landmarker_full.task"))
# Compute `DISABLE` for the next processing step.
DISABLE = os.environ.get("ASDMOTION_DISABLE_MEDIAPIPE", "0") == "1"


def _is_placeholder(path: str) -> bool:
    # Guard this block and recover cleanly from expected failures.
    try:
        # Branch behavior based on the current runtime condition.
        if not os.path.exists(path):
            # Return the result expected by the caller.
            return True
        # Branch behavior based on the current runtime condition.
        if os.path.getsize(path) < 1024:
            # Run this block with managed resources/context cleanup.
            with open(path, "rb") as f:
                # Compute `head` for the next processing step.
                head = f.read(64)
            # Branch behavior based on the current runtime condition.
            if b"ASDMOTION_PLACEHOLDER" in head:
                # Return the result expected by the caller.
                return True
            # Return the result expected by the caller.
            return True
    except Exception:
        # Return the result expected by the caller.
        return True
    # Return the result expected by the caller.
    return False


# Branch behavior based on the current runtime condition.
if not DISABLE:
    # Branch behavior based on the current runtime condition.
    if _is_placeholder(FACE_MODEL) or _is_placeholder(POSE_MODEL):
        # Invoke `print` to advance this processing stage.
        print(
            "[MediaPipe] Model files not found or placeholders detected.\n"
            "           Set ASDMOTION_FACE_MODEL/ASDMOTION_POSE_MODEL\n"
            "           or replace assets/video/*.task. Falling back to no landmarks."
        )
        # Compute `DISABLE` for the next processing step.
        DISABLE = True

# -------- LANDMARKERS --------
BaseOptions = mp.tasks.BaseOptions
# Compute `FaceLandmarker` for the next processing step.
FaceLandmarker = mp.tasks.vision.FaceLandmarker
# Compute `FaceLandmarkerOptions` for the next processing step.
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# Compute `VisionRunningMode` for the next processing step.
VisionRunningMode = mp.tasks.vision.RunningMode

# Compute `face_landmarker` for the next processing step.
face_landmarker = None
# Compute `pose_landmarker` for the next processing step.
pose_landmarker = None

# Branch behavior based on the current runtime condition.
if not DISABLE:
    # Compute `face_options` for the next processing step.
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=False,
        num_faces=1,
    )
    # Compute `face_landmarker` for the next processing step.
    face_landmarker = FaceLandmarker.create_from_options(face_options)

    # Compute `PoseLandmarker` for the next processing step.
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    # Compute `PoseLandmarkerOptions` for the next processing step.
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    # Compute `pose_options` for the next processing step.
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
    # Compute `pose_landmarker` for the next processing step.
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)


def extract_landmarks(frame):
    """
    Input:
        frame: BGR image (OpenCV)
    Output:
        face_landmarks: list of landmarks OR None
        pose_landmarks: list of 33 landmarks OR None
    """
    # Branch behavior based on the current runtime condition.
    if DISABLE or face_landmarker is None or pose_landmarker is None:
        # Return the result expected by the caller.
        return None, None

    # Compute `rgb` for the next processing step.
    rgb = frame[:, :, ::-1]

    # Compute `mp_image` for the next processing step.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    # Run this block with managed resources/context cleanup.
    with suppress_stderr():
        # Compute `face_result` for the next processing step.
        face_result = face_landmarker.detect(mp_image)
        # Compute `pose_result` for the next processing step.
        pose_result = pose_landmarker.detect(mp_image)

    # Compute `face_landmarks` for the next processing step.
    face_landmarks = (
        face_result.face_landmarks[0]
        if face_result.face_landmarks and len(face_result.face_landmarks) > 0
        else None
    )

    # Compute `pose_landmarks` for the next processing step.
    pose_landmarks = (
        pose_result.pose_landmarks[0]
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0
        else None
    )

    # Return the result expected by the caller.
    return face_landmarks, pose_landmarks

