# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

def compute_quality_mask(
    frame_id,
    face_landmarks,
    pose_landmarks,
    face_score=0.0,
    pose_score=0.0
):
    """
    Returns a quality dictionary for one frame.
    Scores should be normalized [0, 1].
    """

    pose_valid = 1 if pose_landmarks is not None else 0
    face_valid = 1 if face_landmarks is not None else 0

    # Ensure scores are present if valid
    if face_valid and face_score == 0.0: face_score = 0.5 # Default if not provided
    if pose_valid and pose_score == 0.0: pose_score = 0.5

    # Pose is mandatory for skeleton rendering usually, but here we track quality.
    frame_valid = 1 if (pose_valid == 1 or face_valid == 1) else 0

    return {
        "frame_id": frame_id,
        "pose_valid": pose_valid,
        "face_valid": face_valid,
        "frame_valid": frame_valid,
        "face_score": face_score,
        "pose_score": pose_score,
        # Hand scores would come from hand landmarks if we extracted them
        "hand_score": 0.0 # Placeholder
    }

