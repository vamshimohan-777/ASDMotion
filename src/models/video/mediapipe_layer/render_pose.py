# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import cv2
import numpy as np

# Official MediaPipe pose topology (manually defined, tasks-safe)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (16, 18),
    (17, 19), (18, 20),
    (19, 21), (20, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32)
]


def render_pose(pose_landmarks, image_size=(256, 256)):
    """
    Input:
        pose_landmarks: list of 33 mediapipe.tasks landmarks

    Output:
        skeleton image (H, W, 3) or None
    """

    # Branch behavior based on the current runtime condition.
    if pose_landmarks is None:
        # Return the result expected by the caller.
        return None

    # Compute `(h, w)` for the next processing step.
    h, w = image_size
    # Compute `img` for the next processing step.
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert normalized coords → pixels
    points = []
    # Iterate `lm` across `pose_landmarks` to process each element.
    for lm in pose_landmarks:
        # Compute `x` for the next processing step.
        x = int(lm.x * w)
        # Compute `y` for the next processing step.
        y = int(lm.y * h)
        # Invoke `points.append` to advance this processing stage.
        points.append((x, y))

    # Draw bones
    for a, b in POSE_CONNECTIONS:
        # Invoke `cv2.line` to advance this processing stage.
        cv2.line(img, points[a], points[b], (255, 255, 255), 2)

    # Draw joints
    for x, y in points:
        # Invoke `cv2.circle` to advance this processing stage.
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)

    # Return the result expected by the caller.
    return img

