"""
Landmark schema for multimodal ASD motion learning.

Defines deterministic index mapping for:
- Pose: 33 joints
- Left hand: 21 joints
- Right hand: 21 joints
- Face: curated 60 joints
"""

from dataclasses import dataclass


# Curated subset of MediaPipe FaceMesh indices (60 points).
FACE_KEYPOINTS_60 = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 33, 160, 158, 133,
    153, 144, 362, 385, 387, 263, 373, 380, 61, 146,
    91, 181, 84, 17, 314, 405, 321, 375, 78, 308,
]


@dataclass(frozen=True)
class LandmarkSchema:
    pose_joints: int = 33
    hand_joints: int = 21
    face_joints: int = 60

    @property
    def pose_slice(self):
        return slice(0, self.pose_joints)

    @property
    def left_hand_slice(self):
        start = self.pose_joints
        return slice(start, start + self.hand_joints)

    @property
    def right_hand_slice(self):
        start = self.pose_joints + self.hand_joints
        return slice(start, start + self.hand_joints)

    @property
    def face_slice(self):
        start = self.pose_joints + 2 * self.hand_joints
        return slice(start, start + self.face_joints)

    @property
    def total_joints(self) -> int:
        return self.pose_joints + (2 * self.hand_joints) + self.face_joints


DEFAULT_SCHEMA = LandmarkSchema()
