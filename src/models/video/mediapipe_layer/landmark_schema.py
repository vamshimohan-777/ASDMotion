"""
Landmark schema for multimodal ASD motion learning.

Defines deterministic index mapping for:
- Pose: 33 joints
- Left hand: 21 joints
- Right hand: 21 joints
- Face: curated 60 joints
"""

# Import symbols from `dataclasses` used in this stage's output computation path.
from dataclasses import dataclass


# Curated subset of MediaPipe FaceMesh indices (60 points).
# Set `FACE_KEYPOINTS_60` for subsequent steps so downstream prediction heads receive the right feature signal.
FACE_KEYPOINTS_60 = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 33, 160, 158, 133,
    153, 144, 362, 385, 387, 263, 373, 380, 61, 146,
    91, 181, 84, 17, 314, 405, 321, 375, 78, 308,
]


# Execute this statement so downstream prediction heads receive the right feature signal.
@dataclass(frozen=True)
class LandmarkSchema:
    """`LandmarkSchema` groups related operations that shape intermediate and final outputs."""
    # Execute this statement so downstream prediction heads receive the right feature signal.
    pose_joints: int = 33
    # Execute this statement so downstream prediction heads receive the right feature signal.
    hand_joints: int = 21
    # Execute this statement so downstream prediction heads receive the right feature signal.
    face_joints: int = 60

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @property
    def pose_slice(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `slice(0, self.pose_joints)` as this function's contribution to downstream output flow.
        return slice(0, self.pose_joints)

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @property
    def left_hand_slice(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `start` for subsequent steps so downstream prediction heads receive the right feature signal.
        start = self.pose_joints
        # Return `slice(start, start + self.hand_joints)` as this function's contribution to downstream output flow.
        return slice(start, start + self.hand_joints)

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @property
    def right_hand_slice(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `start` for subsequent steps so downstream prediction heads receive the right feature signal.
        start = self.pose_joints + self.hand_joints
        # Return `slice(start, start + self.hand_joints)` as this function's contribution to downstream output flow.
        return slice(start, start + self.hand_joints)

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @property
    def face_slice(self):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `start` for subsequent steps so downstream prediction heads receive the right feature signal.
        start = self.pose_joints + 2 * self.hand_joints
        # Return `slice(start, start + self.face_joints)` as this function's contribution to downstream output flow.
        return slice(start, start + self.face_joints)

    # Execute this statement so downstream prediction heads receive the right feature signal.
    @property
    def total_joints(self) -> int:
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `self.pose_joints + (2 * self.hand_joints) + self.fa...` as this function's contribution to downstream output flow.
        return self.pose_joints + (2 * self.hand_joints) + self.face_joints


# Compute `DEFAULT_SCHEMA` as an intermediate representation used by later output layers.
DEFAULT_SCHEMA = LandmarkSchema()
