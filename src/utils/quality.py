import torch


def compute_quality_score(
    qualities: dict,
    mask: torch.Tensor,
    pose_only_if_no_face: bool = False,
    face_presence_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Compute a per-sample quality score in [0,1].

    qualities: dict with keys face_score, pose_score, hand_score, each [B, T]
    mask: [B, T] float or bool (1 for valid frames)
    """
    face = qualities.get("face_score")
    pose = qualities.get("pose_score")
    hand = qualities.get("hand_score")

    if face is None or pose is None:
        # If missing, return 1.0 for all samples to avoid forced abstain
        return torch.ones(mask.shape[0], device=mask.device)

    if hand is None:
        hand = torch.zeros_like(face)

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if face.dim() == 1:
        face = face.unsqueeze(0)
        pose = pose.unsqueeze(0)
        hand = hand.unsqueeze(0)

    if mask.dtype != torch.float32:
        mask_f = mask.float()
    else:
        mask_f = mask

    # Per-frame quality with renormalized weights:
    # prioritize pose when face is missing to match video-only signal.
    face_present = face > 0
    pose_present = pose > 0
    hand_present = hand > 0

    if pose_only_if_no_face:
        face_ratio = (face_present.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        no_face = face_ratio < float(face_presence_threshold)
        if no_face.any():
            no_face = no_face.unsqueeze(1)
            face_present = torch.where(no_face, torch.zeros_like(face_present), face_present)

    w_face = torch.full_like(face, 0.45)
    w_pose = torch.full_like(pose, 0.45)
    w_hand = torch.full_like(hand, 0.10)

    # If face missing, reweight to pose+hand
    w_face = torch.where(face_present, w_face, torch.zeros_like(w_face))
    # If pose missing, reweight to face+hand
    w_pose = torch.where(pose_present, w_pose, torch.zeros_like(w_pose))
    # If hand missing, drop its weight
    w_hand = torch.where(hand_present, w_hand, torch.zeros_like(w_hand))

    w_sum = w_face + w_pose + w_hand
    # Avoid divide-by-zero; if no signals, weights stay zero => quality 0
    w_face = torch.where(w_sum > 0, w_face / w_sum, w_face)
    w_pose = torch.where(w_sum > 0, w_pose / w_sum, w_pose)
    w_hand = torch.where(w_sum > 0, w_hand / w_sum, w_hand)

    q_frame = w_face * face + w_pose * pose + w_hand * hand

    # Average over frames that have any signal (face/pose/hand) to avoid
    # penalizing videos where some frames lack landmarks.
    frame_valid = (face_present | pose_present | hand_present).float()
    denom = (mask_f * frame_valid).sum(dim=1).clamp(min=1.0)
    q = (q_frame * mask_f * frame_valid).sum(dim=1) / denom
    return q.clamp(0.0, 1.0)
