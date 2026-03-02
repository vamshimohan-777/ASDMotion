# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.

"""
Inference script - feed a raw video, get ASD prediction with confidence.

Usage:
    python src/pipeline/inference.py --video path/to/video.mp4 --model results/asd_best.pth
"""

import os
import warnings
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.pipeline_model import ASDPipeline
from src.pipeline.preprocess import VideoProcessor


# Compute `os.environ['TF_CPP_MIN_LOG_LEVEL']` for the next processing step.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Compute `os.environ['GLOG_minloglevel']` for the next processing step.
os.environ["GLOG_minloglevel"] = "2"
# Invoke `warnings.filterwarnings` to advance this processing stage.
warnings.filterwarnings("ignore", message=".*_POSIX_C_SOURCE.*")
# Invoke `warnings.filterwarnings` to advance this processing stage.
warnings.filterwarnings("ignore", message=".*Not enough SMs.*")

# Guard this block and recover cleanly from expected failures.
try:
    # Compute `torch._inductor.config.max_autotu...` for the next processing step.
    torch._inductor.config.max_autotune_gemm = False
except (AttributeError, Exception):
    pass


# Compute `DECISION_LABELS` for the next processing step.
DECISION_LABELS = {1: "ASD", 0: "Non-ASD", -1: "Abstain (Uncertain)"}


def _adapt_state_dict_for_video_only(state_obj):
    # Branch behavior based on the current runtime condition.
    if isinstance(state_obj, dict) and "model_state" in state_obj:
        # Compute `state_dict` for the next processing step.
        state_dict = state_obj["model_state"]
    else:
        # Compute `state_dict` for the next processing step.
        state_dict = state_obj

    # Compute `out` for the next processing step.
    out = {}
    # Iterate `(k, v)` across `state_dict.items()` to process each element.
    for k, v in state_dict.items():
        # Branch behavior based on the current runtime condition.
        if (
            k.startswith("fusion.") or
            k.startswith("perception_cnn.") or
            k.startswith("static_encoder.") or
            k.startswith("image_head.")
        ):
            continue
        # Branch behavior based on the current runtime condition.
        if k.startswith("motion_cnn."):
            # Compute `out['hand_cnn.' + k[len('motion_c...` for the next processing step.
            out["hand_cnn." + k[len("motion_cnn."):]] = v
            continue
        # Compute `out[k]` for the next processing step.
        out[k] = v
    # Return the result expected by the caller.
    return out


def predict_video(video_path, model_path=None, device="cuda", alpha=0.6, theta_high=0.70, theta_low=0.30):
    # Compute `device` for the next processing step.
    device = device if torch.cuda.is_available() else "cpu"
    # Invoke `print` to advance this processing stage.
    print(f"Device: {device}")

    # Compute `model` for the next processing step.
    model = ASDPipeline(alpha=alpha, theta_high=theta_high, theta_low=theta_low).to(device)
    # Branch behavior based on the current runtime condition.
    if model_path and os.path.exists(model_path):
        # Compute `loaded` for the next processing step.
        loaded = torch.load(model_path, map_location=device)
        # Invoke `model.load_state_dict` to advance this processing stage.
        model.load_state_dict(_adapt_state_dict_for_video_only(loaded), strict=False)
        # Invoke `print` to advance this processing stage.
        print(f"Loaded weights from {model_path}")
    else:
        # Invoke `print` to advance this processing stage.
        print("Running with initialized (untrained) weights.")
    # Invoke `model.eval` to advance this processing stage.
    model.eval()

    # Invoke `print` to advance this processing stage.
    print(f"Processing video: {video_path}")
    # Compute `processor` for the next processing step.
    processor = VideoProcessor()
    # Compute `result` for the next processing step.
    result = processor.process_video_file(video_path)
    # Compute `frames` for the next processing step.
    frames = result["frames"]
    # Compute `route` for the next processing step.
    route = result.get("route", "video")
    # Invoke `print` to advance this processing stage.
    print(f"  Frames extracted: {len(frames)}  Duration: {result['duration']:.1f}s  Route: {route}")

    # Compute `seq_len` for the next processing step.
    seq_len = 32
    # Compute `transform` for the next processing step.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Compute `indices` for the next processing step.
    indices = np.linspace(0, max(len(frames) - 1, 0), seq_len).astype(int)
    # Compute `(face_tensors, pose_tensors)` for the next processing step.
    face_tensors, pose_tensors = [], []
    # Compute `(face_scores, pose_scores, hand_s...` for the next processing step.
    face_scores, pose_scores, hand_scores = [], [], []

    # Iterate `i` across `indices` to process each element.
    for i in indices:
        # Branch behavior based on the current runtime condition.
        if i < len(frames):
            # Compute `fd` for the next processing step.
            fd = frames[i]
            # Branch behavior based on the current runtime condition.
            if fd["face_crop"] is not None:
                # Compute `img` for the next processing step.
                img = Image.fromarray(cv2.cvtColor(fd["face_crop"], cv2.COLOR_BGR2RGB))
                # Invoke `face_tensors.append` to advance this processing stage.
                face_tensors.append(transform(img))
            else:
                # Invoke `face_tensors.append` to advance this processing stage.
                face_tensors.append(torch.zeros(3, 224, 224))

            # Branch behavior based on the current runtime condition.
            if fd["skeleton_img"] is not None:
                # Compute `img` for the next processing step.
                img = Image.fromarray(cv2.cvtColor(fd["skeleton_img"], cv2.COLOR_BGR2RGB))
                # Invoke `pose_tensors.append` to advance this processing stage.
                pose_tensors.append(transform(img))
            else:
                # Invoke `pose_tensors.append` to advance this processing stage.
                pose_tensors.append(torch.zeros(3, 224, 224))

            # Compute `q` for the next processing step.
            q = fd["quality"]
            # Invoke `face_scores.append` to advance this processing stage.
            face_scores.append(q.get("face_score", 0.5))
            # Invoke `pose_scores.append` to advance this processing stage.
            pose_scores.append(q.get("pose_score", 0.5))
            # Invoke `hand_scores.append` to advance this processing stage.
            hand_scores.append(q.get("hand_score", 0.0))
        else:
            # Invoke `face_tensors.append` to advance this processing stage.
            face_tensors.append(torch.zeros(3, 224, 224))
            # Invoke `pose_tensors.append` to advance this processing stage.
            pose_tensors.append(torch.zeros(3, 224, 224))
            # Invoke `face_scores.append` to advance this processing stage.
            face_scores.append(0.0)
            # Invoke `pose_scores.append` to advance this processing stage.
            pose_scores.append(0.0)
            # Invoke `hand_scores.append` to advance this processing stage.
            hand_scores.append(0.0)

    # Compute `use_video` for the next processing step.
    use_video = 1.0 if route == "video" else 0.0
    # Compute `pose_stack` for the next processing step.
    pose_stack = torch.stack(pose_tensors).unsqueeze(0).to(device)
    # Compute `inputs` for the next processing step.
    inputs = {
        "face_crops": torch.stack(face_tensors).unsqueeze(0).to(device),
        "pose_maps": pose_stack,
        "hand_maps": pose_stack,
        "mask": torch.ones(1, seq_len).to(device),
        "qualities": {
            "face_score": torch.tensor(face_scores).unsqueeze(0).to(device),
            "pose_score": torch.tensor(pose_scores).unsqueeze(0).to(device),
            "hand_score": torch.tensor(hand_scores).unsqueeze(0).to(device),
        },
        "route_mask": torch.tensor([use_video], device=device),
    }

    # Run this block with managed resources/context cleanup.
    with torch.no_grad():
        # Compute `out` for the next processing step.
        out = model(inputs)

    # Compute `p_final` for the next processing step.
    p_final = out["p_final"].item()
    # Compute `p_video` for the next processing step.
    p_video = out["p_video"].item()
    # Compute `confidence` for the next processing step.
    confidence = out["confidence"].item()
    # Compute `decision` for the next processing step.
    decision = out["decision"].item()
    # Compute `gw` for the next processing step.
    gw = out["gating_weights"][0].mean(dim=0).tolist()

    # Invoke `print` to advance this processing stage.
    print("\n" + "=" * 50)
    # Invoke `print` to advance this processing stage.
    print("          ASD DETECTION RESULT")
    # Invoke `print` to advance this processing stage.
    print("=" * 50)
    # Invoke `print` to advance this processing stage.
    print(f"  Video Prob       : {p_video:.4f}")
    # Invoke `print` to advance this processing stage.
    print("  -----------------------------")
    # Invoke `print` to advance this processing stage.
    print(f"  Final ASD Prob   : {p_final:.4f}")
    # Invoke `print` to advance this processing stage.
    print(f"  Confidence       : {confidence:.4f}")
    # Invoke `print` to advance this processing stage.
    print(f"  Decision         : {DECISION_LABELS[decision]}")
    # Invoke `print` to advance this processing stage.
    print("  -----------------------------")
    # Invoke `print` to advance this processing stage.
    print(f"  Gating Weights   : Face={gw[0]:.2f}  Pose={gw[1]:.2f}  Hand={gw[2]:.2f}")
    # Invoke `print` to advance this processing stage.
    print("=" * 50)

    # Return the result expected by the caller.
    return {
        "p_final": p_final,
        "p_video": p_video,
        "confidence": confidence,
        "decision": DECISION_LABELS[decision],
    }


# Branch behavior based on the current runtime condition.
if __name__ == "__main__":
    # Compute `parser` for the next processing step.
    parser = argparse.ArgumentParser(description="ASD Inference on a single video")
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--video", type=str, required=True)
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--model", type=str, default="results/asd_best.pth")
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--theta_high", type=float, default=0.70)
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--theta_low", type=float, default=0.30)
    # Compute `args` for the next processing step.
    args = parser.parse_args()

    # Invoke `predict_video` to advance this processing stage.
    predict_video(
        args.video,
        args.model,
        theta_high=args.theta_high,
        theta_low=args.theta_low,
    )
