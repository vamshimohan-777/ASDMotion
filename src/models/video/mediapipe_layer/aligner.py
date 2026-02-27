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

    if face_landmarks is None:
        return None

    h, w, _ = frame.shape

    # ---- extract landmark pixel coordinates ----
    xs = [int(lm.x * w) for lm in face_landmarks]
    ys = [int(lm.y * h) for lm in face_landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # ---- add margin ----
    box_w = x_max - x_min
    box_h = y_max - y_min

    pad_x = int(box_w * margin)
    pad_y = int(box_h * margin)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    # ---- sanity check ----
    if x_max <= x_min or y_max <= y_min:
        return None

    # ---- crop ----
    face_crop = frame[y_min:y_max, x_min:x_max]

    if face_crop.size == 0:
        return None

    # ---- resize to fixed size ----
    face_crop = cv2.resize(
        face_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR
    )

    return face_crop
