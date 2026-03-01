# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Inference script — feed a raw video, get ASD prediction with confidence.

Usage:
    python src/pipeline/inference.py --video path/to/video.mp4 --model results/asd_best.pth
"""

# ── Suppress noisy warnings before any heavy imports ──
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import warnings
warnings.filterwarnings("ignore", message=".*_POSIX_C_SOURCE.*")
warnings.filterwarnings("ignore", message=".*Not enough SMs.*")

import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Suppress PyTorch Inductor max_autotune_gemm warning
try:
    torch._inductor.config.max_autotune_gemm = False
except (AttributeError, Exception):
    pass

from src.models.pipeline_model import ASDPipeline
from src.pipeline.preprocess import VideoProcessor


DECISION_LABELS = {1: "ASD", 0: "Non-ASD", -1: "Abstain (Uncertain)"}


def predict_video(video_path, model_path=None, device="cuda",
                   alpha=0.6, theta_high=0.70, theta_low=0.30):
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Model ──
    model = ASDPipeline(
        alpha=alpha, theta_high=theta_high, theta_low=theta_low
    ).to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print("Running with initialised (untrained) weights.")

    model.eval()

    # ── Preprocess ──
    print(f"Processing video: {video_path}")
    processor = VideoProcessor()
    result = processor.process_video_file(video_path)
    frames = result["frames"]
    route = result.get("route", "video")
    print(f"  Frames extracted: {len(frames)}  Duration: {result['duration']:.1f}s  Route: {route}")

    SEQ_LEN = 32
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    indices = np.linspace(0, max(len(frames) - 1, 0), SEQ_LEN).astype(int)
    face_tensors, pose_tensors = [], []
    face_scores, pose_scores, hand_scores = [], [], []

    for i in indices:
        if i < len(frames):
            fd = frames[i]
            if fd["face_crop"] is not None:
                img = Image.fromarray(cv2.cvtColor(fd["face_crop"], cv2.COLOR_BGR2RGB))
                face_tensors.append(transform(img))
            else:
                face_tensors.append(torch.zeros(3, 224, 224))
            if fd["skeleton_img"] is not None:
                img = Image.fromarray(cv2.cvtColor(fd["skeleton_img"], cv2.COLOR_BGR2RGB))
                pose_tensors.append(transform(img))
            else:
                pose_tensors.append(torch.zeros(3, 224, 224))
            q = fd["quality"]
            face_scores.append(q.get("face_score", 0.5))
            pose_scores.append(q.get("pose_score", 0.5))
            hand_scores.append(q.get("hand_score", 0.0))
        else:
            face_tensors.append(torch.zeros(3, 224, 224))
            pose_tensors.append(torch.zeros(3, 224, 224))
            face_scores.append(0.0)
            pose_scores.append(0.0)
            hand_scores.append(0.0)

    use_video = 1.0 if route == "video" else 0.0
    inputs = {
        "face_crops": torch.stack(face_tensors).unsqueeze(0).to(device),
        "pose_maps":  torch.stack(pose_tensors).unsqueeze(0).to(device),
        "mask":       torch.ones(1, SEQ_LEN).to(device),
        "qualities": {
            "face_score": torch.tensor(face_scores).unsqueeze(0).to(device),
            "pose_score": torch.tensor(pose_scores).unsqueeze(0).to(device),
            "hand_score": torch.tensor(hand_scores).unsqueeze(0).to(device),
        },
        "route_mask": torch.tensor([use_video], device=device),
    }

    # ── Predict ──
    with torch.no_grad():
        out = model(inputs)

    p_final    = out["p_final"].item()
    p_video    = out["p_video"].item()
    p_image    = out["p_image"].item()
    alpha_val = out["alpha"]
    if torch.is_tensor(alpha_val):
        alpha_val = alpha_val.mean().item()
    confidence = out["confidence"].item()
    decision   = out["decision"].item()

    gw = out["gating_weights"][0].mean(dim=0).tolist()

    print("\n" + "=" * 50)
    print("          ASD DETECTION RESULT")
    print("=" * 50)
    print(f"  Video Path Prob  : {p_video:.4f}")
    print(f"  Image Path Prob  : {p_image:.4f}")
    print(f"  Fusion Weight (α): {alpha_val:.3f}")
    print(f"  ─────────────────────────────")
    print(f"  Final ASD Prob   : {p_final:.4f}")
    print(f"  Confidence       : {confidence:.4f}")
    print(f"  Decision         : {DECISION_LABELS[decision]}")
    print(f"  ─────────────────────────────")
    print(f"  Gating Weights   : Face={gw[0]:.2f}  Pose={gw[1]:.2f}  Hand={gw[2]:.2f}")
    print("=" * 50)

    return {
        "p_final": p_final,
        "p_video": p_video,
        "p_image": p_image,
        "confidence": confidence,
        "decision": DECISION_LABELS[decision],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASD Inference on a single video")
    parser.add_argument("--video",      type=str, required=True)
    parser.add_argument("--model",      type=str, default="results/asd_best.pth")
    parser.add_argument("--theta_high", type=float, default=0.70)
    parser.add_argument("--theta_low",  type=float, default=0.30)
    args = parser.parse_args()

    predict_video(
        args.video, args.model,
        theta_high=args.theta_high,
        theta_low=args.theta_low,
    )

