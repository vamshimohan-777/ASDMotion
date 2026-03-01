# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

EVENT_TYPE_MAP = {
    # Child responds when name is called: early social orientation cue.
    "name_call_response": 0,
    # Point-based joint attention: checks shared focus through deictic gesture following.
    "joint_attention_point": 1,
    # Gaze-based joint attention: detects shared focus using eye direction only.
    "joint_attention_gaze": 2,
    # Attempts to establish eye contact: key social engagement marker.
    "eye_contact_attempt": 3,
    # Alternating interaction turns: reflects reciprocal social timing.
    "turn_taking": 4,
    # Motor action imitation: evaluates social learning through copied actions.
    "action_imitation": 5,
    # Gesture imitation: captures communication-oriented mimicry behavior.
    "gesture_imitation": 6,
    # Repetitive hand movement: common restricted/repetitive behavior signal.
    "repetitive_hand_motion": 7,
    # Repetitive whole-body motion: broader stereotyped movement indicator.
    "repetitive_body_motion": 8,
    # Persistent object focus: can indicate reduced social orienting.
    "object_fixation": 9,
    # Matching/reflecting emotional expression: socio-emotional reciprocity cue.
    "emotional_mirroring": 10,
    # Sensory-triggered reaction pattern: tracks atypical sensory responsiveness.
    "sensory_response": 11
}

# Reverse mapping keeps inference outputs human-readable in logs and reports.
ID_TO_EVENT_TYPE = {v: k for k, v in EVENT_TYPE_MAP.items()}

# Centralized class count prevents mismatches across heads/losses/tokenizers.
NUM_EVENT_TYPES = len(EVENT_TYPE_MAP)

