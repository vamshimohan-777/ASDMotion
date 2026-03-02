# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import os
import json
import time
import cv2
import torch
import numpy as np

from src.models.pipeline_model import ASDPipeline
from src.pipeline.preprocess import VideoProcessor, load_video
from src.utils.data_utils import prepare_sequence_from_frames
from src.utils.quality import compute_quality_score
from src.utils.decision import make_decision
from src.utils.config import load_config
from src.utils.calibration import apply_temperature
from src.utils.video_id import make_video_id
from src.models.video.microkinetic_encoders.event_types import ID_TO_EVENT_TYPE


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _as_int_list(values):
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        out = [int(v) for v in values]
    else:
        out = [int(values)]
    return out if out else None


def _adapt_state_dict_for_video_only(model_state: dict) -> dict:
    # Keep video/NAS weights, remap legacy motion encoder keys to hand encoder keys,
    # and drop removed image/fusion branches.
    out = {}
    for k, v in model_state.items():
        if (
            k.startswith("fusion.") or
            k.startswith("perception_cnn.") or
            k.startswith("static_encoder.") or
            k.startswith("image_head.")
        ):
            continue
        if k.startswith("motion_cnn."):
            out["hand_cnn." + k[len("motion_cnn."):]] = v
            continue
        out[k] = v
    return out


class ASDPredictor:
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt.get("config", None)
        if cfg is None and config_path:
            cfg = load_config(config_path)
        if cfg is None:
            cfg = {}

        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        thresholds = cfg.get("thresholds", {})

        self.seq_len = cfg.get("data", {}).get("seq_len", 32)
        self.frame_stride = int(cfg.get("data", {}).get("frame_stride", 1))
        self.max_frames = int(cfg.get("data", {}).get("max_frames", 0) or 0)
        self.low_thr = thresholds.get("decision_low", 0.3)
        self.high_thr = thresholds.get("decision_high", 0.7)
        self.quality_thr = thresholds.get("quality_threshold", 0.5)
        self.model_version = cfg.get("inference", {}).get("model_version", "asd_pipeline_v1")

        model_state = ckpt.get("model_state", {})
        nas_arch = ckpt.get("nas_architecture", None)

        # Rebuild architecture from checkpoint config (or infer from state_dict if needed)
        # so loading remains compatible across model revisions.
        nas_space = model_cfg.get("nas_search_space", {})
        if not isinstance(nas_space, dict) or not nas_space:
            nas_space = cfg.get("nas", {}).get("search_space", {})
        if not isinstance(nas_space, dict):
            nas_space = {}

        trans_space = nas_space.get("transformer", {})
        if not isinstance(trans_space, dict):
            trans_space = {}

        encoder_kernel_candidates = _as_int_list(nas_space.get("encoder_kernel"))
        transformer_heads_candidates = _as_int_list(trans_space.get("n_heads"))
        transformer_layers_candidates = _as_int_list(trans_space.get("num_encoder_layers"))
        transformer_ff_candidates = _as_int_list(trans_space.get("dim_ff"))

        if isinstance(nas_arch, dict):
            trans_arch = nas_arch.get("transformer", {})
            if not isinstance(trans_arch, dict):
                trans_arch = {}
            if encoder_kernel_candidates is None and "encoder_kernel" in nas_arch:
                encoder_kernel_candidates = _as_int_list(nas_arch.get("encoder_kernel"))
            if transformer_heads_candidates is None and "n_heads" in trans_arch:
                transformer_heads_candidates = _as_int_list(trans_arch.get("n_heads"))
            if transformer_layers_candidates is None and "num_encoder_layers" in trans_arch:
                transformer_layers_candidates = _as_int_list(trans_arch.get("num_encoder_layers"))
            if transformer_ff_candidates is None and "dim_ff" in trans_arch:
                transformer_ff_candidates = _as_int_list(trans_arch.get("dim_ff"))

        cnn_backbone = str(model_cfg.get("cnn_backbone", "")).strip().lower()
        if not cnn_backbone:
            proj_w = model_state.get("face_cnn.proj.0.weight")
            if proj_w is not None and proj_w.ndim == 2:
                cnn_backbone = "resnet50" if int(proj_w.shape[1]) >= 1024 else "resnet18"
            else:
                cnn_backbone = "resnet18"

        face_use_fc_head = model_cfg.get("face_use_fc_head", None)
        if face_use_fc_head is None:
            face_use_fc_head = (
                "face_cnn.backbone.fc.weight" in model_state and
                "face_cnn.backbone.fc.bias" in model_state
            )

        num_event_types = model_cfg.get("num_event_types", None)
        if num_event_types is None:
            type_head_w = model_state.get("nas_controller.encoder.type_head.weight")
            if type_head_w is not None and type_head_w.ndim == 2:
                num_event_types = int(type_head_w.shape[0])
        if num_event_types is None:
            num_event_types = 12

        self.model = ASDPipeline(
            alpha=model_cfg.get("alpha", 0.6),
            K_max=model_cfg.get("K_max", 32),
            d_model=model_cfg.get("d_model", 256),
            dropout=model_cfg.get("dropout", 0.3),
            theta_high=self.high_thr,
            theta_low=self.low_thr,
            cnn_backbone=cnn_backbone,
            face_use_fc_head=bool(face_use_fc_head),
            num_event_types=int(num_event_types),
            encoder_kernel_candidates=encoder_kernel_candidates,
            transformer_heads_candidates=transformer_heads_candidates,
            transformer_layers_candidates=transformer_layers_candidates,
            transformer_ff_candidates=transformer_ff_candidates,
        ).to(self.device)

        if nas_arch is not None:
            self.model.apply_nas_architecture(nas_arch)

        model_state = _adapt_state_dict_for_video_only(model_state)
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()
        self.model.freeze_cnns()

        self.temperature = ckpt.get("temperature", 1.0)

        self.processor = VideoProcessor(
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
        )

        self.processed_root = cfg.get("data", {}).get("processed_root", "data/processed")

    def _predict_from_frames(self, frames: list, route: str) -> dict:
        start = time.time()

        sample = prepare_sequence_from_frames(frames, seq_len=self.seq_len)

        # add batch dimension
        inputs = {
            "face_crops": sample["face_crops"].unsqueeze(0).to(self.device),
            "pose_maps": sample["pose_maps"].unsqueeze(0).to(self.device),
            "motion_maps": sample["motion_maps"].unsqueeze(0).to(self.device),
            "hand_maps": sample["motion_maps"].unsqueeze(0).to(self.device),
            "mask": sample["mask"].unsqueeze(0).to(self.device),
            "timestamps": sample["timestamps"].unsqueeze(0).to(self.device),
            "delta_t": sample["delta_t"].unsqueeze(0).to(self.device),
            "qualities": {k: v.unsqueeze(0).to(self.device) for k, v in sample["qualities"].items()},
            "route_mask": torch.tensor([1.0 if route == "video" else 0.0], device=self.device),
        }

        with torch.no_grad():
            out = self.model(inputs)
            logit = out["logit_final"].detach().cpu().numpy().reshape(-1)[0]

        prob_raw = float(_sigmoid(logit))
        logit_cal = float(apply_temperature(torch.tensor([logit]), self.temperature).item())
        prob_cal = float(_sigmoid(logit_cal))

        q = compute_quality_score(
            {k: v.unsqueeze(0) for k, v in sample["qualities"].items()},
            sample["mask"].unsqueeze(0)
        ).item()

        decision = make_decision(prob_raw, prob_cal, q, self.quality_thr, self.low_thr, self.high_thr)
        events = self._extract_event_evidence(out, decision.decision)

        face_q = float(sample["qualities"]["face_score"].mean().item())
        pose_q = float(sample["qualities"]["pose_score"].mean().item())
        reasons = list(decision.reasons)
        reasons.append("Face quality ok" if face_q >= self.quality_thr else "Face quality low")
        reasons.append("Pose quality ok" if pose_q >= self.quality_thr else "Pose quality low")

        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "decision": decision.decision,
            "prob_raw": decision.prob_raw,
            "prob_calibrated": decision.prob_calibrated,
            "quality_score": decision.quality_score,
            "threshold_used": decision.threshold_used,
            "abstained": decision.abstained,
            "reasons": reasons,
            "events": events,
            "model_version": self.model_version,
            "inference_ms": elapsed_ms,
        }

    def _extract_event_evidence(self, model_out: dict, decision_label: str) -> list:
        label = str(decision_label).upper()
        if not (label.endswith("CHANCES OF ASD") or label == "NEEDS RECHECKING"):
            return []

        event_ids = model_out.get("event_type_id", None)
        event_mask = model_out.get("event_mask", None)
        event_conf = model_out.get("event_confidence", None)
        if event_ids is None or event_mask is None:
            return []

        event_ids = event_ids.detach().cpu()
        event_mask = event_mask.detach().cpu().bool()

        if event_ids.dim() == 1:
            event_ids = event_ids.unsqueeze(0)
        if event_mask.dim() == 1:
            event_mask = event_mask.unsqueeze(0)

        if event_conf is not None:
            event_conf = event_conf.detach().cpu().float()
            if event_conf.dim() == 1:
                event_conf = event_conf.unsqueeze(0)

        ids = event_ids[0]
        mask = event_mask[0]
        conf = event_conf[0] if event_conf is not None else torch.ones_like(ids, dtype=torch.float32)

        stats = {}
        for i in range(ids.shape[0]):
            if not bool(mask[i].item()):
                continue
            event_id = int(ids[i].item())
            name = ID_TO_EVENT_TYPE.get(event_id, f"event_{event_id}")
            score = float(conf[i].item())

            cur = stats.get(name)
            if cur is None:
                stats[name] = {
                    "event": name,
                    "count": 1,
                    "score_sum": score,
                    "max_confidence": score,
                }
            else:
                cur["count"] += 1
                cur["score_sum"] += score
                cur["max_confidence"] = max(cur["max_confidence"], score)

        events = []
        for item in stats.values():
            count = max(int(item["count"]), 1)
            mean_conf = float(item["score_sum"] / count)
            events.append({
                "event": item["event"],
                "count": count,
                "mean_confidence": round(mean_conf, 3),
                "max_confidence": round(float(item["max_confidence"]), 3),
            })

        events.sort(key=lambda x: (x["mean_confidence"], x["count"]), reverse=True)
        return events[:5]

    def _resolve_processed_dir(self, ref: str, processed_root: str = None) -> str:
        root = processed_root or self.processed_root

        # Direct directory path
        if os.path.isdir(ref):
            return ref

        # Treat as raw video path (stable id derived from path)
        candidate = os.path.join(root, make_video_id(ref))
        if os.path.isdir(candidate):
            return candidate

        # Treat as precomputed video_id
        candidate = os.path.join(root, ref)
        if os.path.isdir(candidate):
            return candidate

        # Fallback: match by uploaded filename against meta.json video_path basename
        basename = os.path.basename(ref)
        if basename and os.path.isdir(root):
            matches = []
            for entry in os.scandir(root):
                if not entry.is_dir():
                    continue
                meta_path = os.path.join(entry.path, "meta.json")
                if not os.path.exists(meta_path):
                    continue
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                if os.path.basename(str(meta.get("video_path", ""))) == basename:
                    matches.append(entry.path)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple processed entries matched filename '{basename}'. Use an exact processed folder path or video_id."
                )

        raise FileNotFoundError(
            f"Preprocessed data not found for '{ref}'. Tried: '{ref}', '{os.path.join(root, make_video_id(ref))}', '{os.path.join(root, ref)}'"
        )

    def _load_preprocessed_frames(self, ref: str, processed_root: str = None):
        base_dir = self._resolve_processed_dir(ref, processed_root=processed_root)
        meta_path = os.path.join(base_dir, "meta.json")
        quality_path = os.path.join(base_dir, "quality.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta.json in: {base_dir}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        qualities = []
        if os.path.exists(quality_path):
            with open(quality_path, "r") as f:
                qualities = json.load(f)

        frame_ids = meta.get("frame_ids", [])
        timestamps = meta.get("timestamps", [])
        route = meta.get("route", "video")

        frames = []
        for i, frame_id in enumerate(frame_ids):
            ts = timestamps[i] if i < len(timestamps) else float(frame_id)
            q = qualities[i] if i < len(qualities) else {}

            face_path = os.path.join(base_dir, "faces", f"{frame_id:06d}.png")
            skeleton_path = os.path.join(base_dir, "skeletons", f"{frame_id:06d}.png")

            face_img = cv2.imread(face_path) if os.path.exists(face_path) else None
            skeleton_img = cv2.imread(skeleton_path) if os.path.exists(skeleton_path) else None

            frames.append({
                "frame_id": frame_id,
                "timestamp": ts,
                "face_crop": face_img,
                "skeleton_img": skeleton_img,
                "quality": q,
            })

        return frames, route

    def predict_video(self, video_path: str) -> dict:
        processed = self.processor.process_video_file(video_path)
        frames = processed.get("frames", [])
        route = processed.get("route", "video")
        return self._predict_from_frames(frames, route)

    def predict_preprocessed(self, processed_ref: str, processed_root: str = None) -> dict:
        frames, route = self._load_preprocessed_frames(processed_ref, processed_root=processed_root)
        return self._predict_from_frames(frames, route)

    def predict_landmark_video(self, video_path: str) -> dict:
        frames_raw, _, _ = load_video(
            video_path,
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
        )

        frames = []
        for fd in frames_raw:
            frame_id = int(fd.get("frame_id", 0))
            timestamp = float(fd.get("timestamp", 0.0))
            image = fd.get("image", None)
            if image is None:
                continue

            frames.append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "face_crop": None,
                "skeleton_img": image,
                "quality": {
                    "frame_id": frame_id,
                    "pose_valid": 1,
                    "face_valid": 0,
                    "frame_valid": 1,
                    "face_score": 0.0,
                    "pose_score": 1.0,
                    "hand_score": 0.0,
                },
            })

        return self._predict_from_frames(frames, route="video")

