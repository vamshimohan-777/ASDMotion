"""
Event taxonomy for the microkinetic encoder.

Notes:
1. Legacy IDs (0-16) are preserved for backward compatibility with older checkpoints.
2. The extended taxonomy appends fine-grained event types requested for microkinetic
   encoding and downstream evidence extraction.
"""

# Keep these first and in-order so older checkpoints remain decodable.
# Set `LEGACY_EVENT_TYPES` for subsequent steps so downstream prediction heads receive the right feature signal.
LEGACY_EVENT_TYPES = [
    "name_call_response",
    "joint_attention_point",
    "joint_attention_gaze",
    "eye_contact_attempt",
    "turn_taking",
    "action_imitation",
    "gesture_imitation",
    "repetitive_hand_motion",
    "repetitive_body_motion",
    "object_fixation",
    "emotional_mirroring",
    "sensory_response",
    "repetitive_motor_patterns",
    "postural_instability",
    "asymmetry",
    "engagement_orientation_change",
    "motion_bursts_freezes",
]

# Set `POSTURAL_CONTROL_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
POSTURAL_CONTROL_EVENTS = [
    "postural_sway_burst",
    "sustained_body_rocking",
    "forward_back_torso_oscillation",
    "lateral_torso_oscillation",
    "sudden_posture_shift",
    "gradual_posture_drift",
    "prolonged_stillness_low_motion_window",
    "sudden_freeze_abrupt_motion_drop",
    "collapse_slump_posture",
    "hyper_extended_rigid_posture",
]

# Set `REPETITIVE_MOTOR_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
REPETITIVE_MOTOR_EVENTS = [
    "hand_flapping_bilateral",
    "hand_flapping_unilateral",
    "finger_flicking",
    "wrist_oscillation_burst",
    "arm_rocking",
    "body_rocking",
    "repetitive_shoulder_shrug",
    "repetitive_elbow_flexion_extension",
    "repetitive_leg_bouncing",
    "repetitive_foot_tapping",
    "repetitive_hand_to_body_tapping",
    "high_frequency_limb_tremor",
    "low_frequency_rhythmic_rocking",
    "repetitive_hand_near_face_motion",
    "self_stimulatory_repetitive_motion_generic_rhythmic_limb_oscillation",
]

# Set `MOTION_ENERGY_KINEMATIC_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
MOTION_ENERGY_KINEMATIC_EVENTS = [
    "sudden_motion_burst",
    "gradual_motion_ramp_up",
    "abrupt_stop",
    "high_acceleration_spike",
    "high_jerk_spike_third_derivative_irregularity",
    "sustained_high_velocity_window",
    "sustained_low_velocity_window",
    "velocity_asymmetry_spike",
    "motion_variability_increase",
    "motion_variability_decrease",
]

# Set `ASYMMETRY_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
ASYMMETRY_EVENTS = [
    "persistent_unilateral_limb_dominance",
    "sudden_asymmetry_onset",
    "left_right_arm_amplitude_imbalance",
    "left_right_leg_amplitude_imbalance",
    "asymmetric_arm_elevation",
    "cross_midline_hand_crossing",
    "asymmetric_repetitive_pattern",
]

# Compute `HEAD_ORIENTATION_EVENTS` as an intermediate representation used by later output layers.
HEAD_ORIENTATION_EVENTS = [
    "rapid_head_turn",
    "sustained_head_turn",
    "head_down_posture",
    "head_tilt_asymmetry",
    "micro_head_jitter",
    "head_orientation_instability",
    "sudden_orientation_change",
    "sustained_disengaged_orientation_looking_away_posture",
]

# Set `ENGAGEMENT_INTERACTION_POST...` for subsequent steps so downstream prediction heads receive the right feature signal.
ENGAGEMENT_INTERACTION_POSTURE_EVENTS = [
    "torso_orientation_toward_stimulus",
    "torso_turning_away",
    "lean_forward_engagement",
    "lean_backward_withdrawal",
    "approach_motion_initiation",
    "retreat_motion",
    "reduced_gesture_diversity_window",
    "delayed_reach_onset",
    "incomplete_reach",
    "abrupt_gesture_termination",
]

# Set `COORDINATION_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
COORDINATION_EVENTS = [
    "limb_coordination_breakdown",
    "out_of_phase_bilateral_movement",
    "synchronized_bilateral_motion",
    "desynchronized_limb_movement",
    "arm_leg_coordination_anomaly",
    "sudden_coordination_shift",
    "cross_body_interaction_event",
    "reduced_inter_limb_coupling_window",
]

# Set `MICRO_DYNAMIC_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
MICRO_DYNAMIC_EVENTS = [
    "micro_tremor_burst",
    "irregular_oscillation_cluster",
    "sudden_frequency_shift_in_oscillation",
    "oscillation_amplitude_escalation",
    "oscillation_amplitude_collapse",
    "high_jerk_variability_window",
    "intermittent_micro_movement_cluster",
]

# Set `TEMPORAL_STRUCTURE_EVENTS` for subsequent steps so downstream prediction heads receive the right feature signal.
TEMPORAL_STRUCTURE_EVENTS = [
    "prolonged_inactivity_interval",
    "rapid_event_succession_cluster",
    "event_gap_anomaly_long_pause_after_motion",
    "event_density_spike",
    "event_density_drop",
]

# Compute `COMPOSITE_BEHAVIORAL_EVENTS` as an intermediate representation used by later output layers.
COMPOSITE_BEHAVIORAL_EVENTS = [
    "repetitive_motor_episode_multi_cycle_cluster",
    "freeze_burst_cycle",
    "rocking_pause_rocking_pattern",
    "asymmetry_correction_cycle",
    "oscillation_decay_pattern",
    "oscillation_growth_pattern",
    "repetitive_with_increasing_amplitude_pattern",
    "repetitive_with_irregular_interval_pattern",
    "motor_instability_episode",
    "sustained_motor_rigidity_episode",
]

# Set `EVENT_TYPES` for subsequent steps so downstream prediction heads receive the right feature signal.
EVENT_TYPES = (
    LEGACY_EVENT_TYPES
    + POSTURAL_CONTROL_EVENTS
    + REPETITIVE_MOTOR_EVENTS
    + MOTION_ENERGY_KINEMATIC_EVENTS
    + ASYMMETRY_EVENTS
    + HEAD_ORIENTATION_EVENTS
    + ENGAGEMENT_INTERACTION_POSTURE_EVENTS
    + COORDINATION_EVENTS
    + MICRO_DYNAMIC_EVENTS
    + TEMPORAL_STRUCTURE_EVENTS
    + COMPOSITE_BEHAVIORAL_EVENTS
)

# Branch on `len(EVENT_TYPES) != len(set(EVENT_TYPES))` to choose the correct output computation path.
if len(EVENT_TYPES) != len(set(EVENT_TYPES)):
    # Raise explicit error to stop invalid state from producing misleading outputs.
    raise ValueError("Duplicate event type names detected in EVENT_TYPES.")

# Set `EVENT_TYPE_MAP` for subsequent steps so downstream prediction heads receive the right feature signal.
EVENT_TYPE_MAP = {name: idx for idx, name in enumerate(EVENT_TYPES)}

# Reverse mapping: ID -> event name (for debugging / reports)
# Set `ID_TO_EVENT_TYPE` for subsequent steps so downstream prediction heads receive the right feature signal.
ID_TO_EVENT_TYPE = {v: k for k, v in EVENT_TYPE_MAP.items()}

# Number of defined event types
# Set `NUM_EVENT_TYPES` for subsequent steps so downstream prediction heads receive the right feature signal.
NUM_EVENT_TYPES = len(EVENT_TYPE_MAP)
