# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import cv2
import numpy as np


def aligned_face_crop(
    frame,
    face_landmarks,
    output_size=224,
    margin=0.2
):
    """
    Args:
        frame           : BGR image (OpenCV)
        face_landmarks  : list of mediapipe.tasks face landmarks
        output_size     : final crop size (square)
        margin          : extra padding around face bbox

    Returns:
        aligned face crop (output_size x output_size x 3) OR None
    """

    # Branch behavior based on the current runtime condition.
    if face_landmarks is None:
        # Return the result expected by the caller.
        return None

    # Compute `(h, w, _)` for the next processing step.
    h, w, _ = frame.shape

    # ---- extract landmark pixel coordinates ----
    xs = [int(lm.x * w) for lm in face_landmarks]
    # Compute `ys` for the next processing step.
    ys = [int(lm.y * h) for lm in face_landmarks]

    # Compute `(x_min, x_max)` for the next processing step.
    x_min, x_max = min(xs), max(xs)
    # Compute `(y_min, y_max)` for the next processing step.
    y_min, y_max = min(ys), max(ys)

    # ---- add margin ----
    box_w = x_max - x_min
    # Compute `box_h` for the next processing step.
    box_h = y_max - y_min

    # Compute `pad_x` for the next processing step.
    pad_x = int(box_w * margin)
    # Compute `pad_y` for the next processing step.
    pad_y = int(box_h * margin)

    # Compute `x_min` for the next processing step.
    x_min = max(0, x_min - pad_x)
    # Compute `y_min` for the next processing step.
    y_min = max(0, y_min - pad_y)
    # Compute `x_max` for the next processing step.
    x_max = min(w, x_max + pad_x)
    # Compute `y_max` for the next processing step.
    y_max = min(h, y_max + pad_y)

    # ---- sanity check ----
    if x_max <= x_min or y_max <= y_min:
        # Return the result expected by the caller.
        return None

    # ---- crop ----
    face_crop = frame[y_min:y_max, x_min:x_max]

    # Branch behavior based on the current runtime condition.
    if face_crop.size == 0:
        # Return the result expected by the caller.
        return None

    # ---- resize to fixed size ----
    face_crop = cv2.resize(
        face_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR
    )

    # Return the result expected by the caller.
    return face_crop

