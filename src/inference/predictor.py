import json
import os
import time

import numpy as np
import torch

from src.models.pipeline_model import ASDPipeline
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
from src.pipeline.preprocess import VideoProcessor
from src.utils.calibration import apply_temperature
from src.utils.config import load_config
from src.utils.decision import make_decision
from src.utils.video_id import make_video_id


def _moving_average_1d(x, k=5):
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = np.ones((k,), dtype=np.float32)
    kernel /= kernel.sum()
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _fill_missing_1d(values, valid):
    out = values.copy()
    if valid.sum() == 0:
        return out
    idx = np.where(valid > 0.5)[0]
    first = idx[0]
    last = idx[-1]
    out[:first] = out[first]
    out[last + 1:] = out[last]
    miss = np.where(valid < 0.5)[0]
    for m in miss:
        left = idx[idx < m]
        right = idx[idx > m]
        if left.size == 0 and right.size == 0:
            continue
        if left.size == 0:
            out[m] = out[right[0]]
        elif right.size == 0:
            out[m] = out[left[-1]]
        else:
            l = left[-1]
            r = right[0]
            alpha = (m - l) / max((r - l), 1)
            out[m] = (1.0 - alpha) * out[l] + alpha * out[r]
    return out


def normalize_landmarks(landmarks, mask, schema=DEFAULT_SCHEMA, smooth_kernel=5):
    xyz = landmarks.copy()
    m = mask.copy()
    t, j, _ = xyz.shape

    for jj in range(j):
        valid = m[:, jj] > 0.5
        for c in range(3):
            xyz[:, jj, c] = _fill_missing_1d(xyz[:, jj, c], valid)

    # Hip center
    l_hip, r_hip = 23, 24
    hips_valid = (m[:, l_hip] > 0.5) & (m[:, r_hip] > 0.5)
    center = np.zeros((t, 3), dtype=np.float32)
    center[hips_valid] = 0.5 * (xyz[hips_valid, l_hip] + xyz[hips_valid, r_hip])
    if hips_valid.any():
        last = center[np.where(hips_valid)[0][0]].copy()
    else:
        last = np.zeros((3,), dtype=np.float32)
    for i in range(t):
        if hips_valid[i]:
            last = center[i]
        else:
            center[i] = last
    xyz = xyz - center[:, None, :]

    # Shoulder scale
    l_sh, r_sh = 11, 12
    sh_valid = (m[:, l_sh] > 0.5) & (m[:, r_sh] > 0.5)
    scale = np.ones((t,), dtype=np.float32)
    if sh_valid.any():
        d = np.linalg.norm(xyz[:, l_sh] - xyz[:, r_sh], axis=-1)
        d = np.clip(d, 1e-4, None)
        scale[sh_valid] = d[sh_valid]
        scale[~sh_valid] = float(np.median(scale[sh_valid]))
    xyz = xyz / scale[:, None, None]

    for jj in range(j):
        valid = m[:, jj]
        for c in range(3):
            sm = _moving_average_1d(xyz[:, jj, c], k=smooth_kernel)
            xyz[:, jj, c] = (valid * sm) + ((1.0 - valid) * xyz[:, jj, c])
    return xyz, m


def build_motion_features(xyz, mask):
    vel = np.zeros_like(xyz, dtype=np.float32)
    acc = np.zeros_like(xyz, dtype=np.float32)
    if xyz.shape[0] > 1:
        vel[1:] = xyz[1:] - xyz[:-1]
        acc[1:] = vel[1:] - vel[:-1]
    feat = np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)
    feat *= mask[..., None]
    return feat


def sliding_windows(motion, mask, timestamps, window_size=64, stride=16):
    t = motion.shape[0]
    if t <= window_size:
        starts = [0]
    else:
        starts = list(range(0, max(1, t - window_size + 1), max(1, stride)))
        if starts[-1] != (t - window_size):
            starts.append(t - window_size)

    windows = []
    masks = []
    ts_windows = []
    for s in starts:
        e = s + window_size
        w = motion[s:e]
        m = mask[s:e]
        ts = timestamps[s:e]
        if w.shape[0] < window_size:
            pad = window_size - w.shape[0]
            w = np.pad(w, ((0, pad), (0, 0), (0, 0)), mode="constant")
            m = np.pad(m, ((0, pad), (0, 0)), mode="constant")
            ts = np.pad(ts, (0, pad), mode="edge" if ts.size > 0 else "constant")
        windows.append(w.astype(np.float32))
        masks.append(m.astype(np.float32))
        ts_windows.append(ts.astype(np.float32))
    return np.stack(windows), np.stack(masks), np.stack(ts_windows), starts


class ASDPredictor:
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt.get("config", None)
        if cfg is None and config_path:
            cfg = load_config(config_path)
        if cfg is None:
            cfg = {}
        self.cfg = cfg

        model_cfg = cfg.get("model", {})
        th = cfg.get("thresholds", {})
        self.model = ASDPipeline(
            K_max=int(model_cfg.get("K_max", 16)),
            d_model=int(model_cfg.get("d_model", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            theta_high=float(th.get("decision_high", 0.7)),
            theta_low=float(th.get("decision_low", 0.3)),
        ).to(self.device)
        nas_arch = ckpt.get("nas_architecture", None)
        if nas_arch is not None:
            self.model.apply_nas_architecture(nas_arch)
        try:
            self.model.load_state_dict(ckpt["model_state"], strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the landmark-motion model. "
                "Train a new checkpoint with `python -m src.training.train --config config.yaml`."
            ) from exc
        self.model.eval()

        self.temperature = float(ckpt.get("temperature", 1.0))
        self.processor = VideoProcessor(
            frame_stride=int(cfg.get("data", {}).get("frame_stride", 1)),
            max_frames=int(cfg.get("data", {}).get("max_frames", 0)),
        )
        self.processed_root = str(cfg.get("data", {}).get("processed_root", "data/processed"))
        self.quality_thr = float(th.get("quality_threshold", 0.5))
        self.low_thr = float(th.get("decision_low", 0.3))
        self.high_thr = float(th.get("decision_high", 0.7))
        self.window_size = int(cfg.get("inference", {}).get("window_size", 64))
        self.window_stride = int(cfg.get("inference", {}).get("window_stride", 16))
        self.model_version = str(cfg.get("inference", {}).get("model_version", "asd_motion_landmark_v2"))

    def _frames_to_arrays(self, frames):
        if not frames:
            j = DEFAULT_SCHEMA.total_joints
            return (
                np.zeros((1, j, 3), dtype=np.float32),
                np.zeros((1, j), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                {"face": 0.0, "pose": 0.0, "hand": 0.0},
            )
        landmarks = np.stack([f["landmarks"] for f in frames]).astype(np.float32)
        mask = np.stack([f["mask"] for f in frames]).astype(np.float32)
        timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.float32)
        face_q = float(np.mean([f["quality"].get("face_score", 0.0) for f in frames]))
        pose_q = float(np.mean([f["quality"].get("pose_score", 0.0) for f in frames]))
        hand_q = float(np.mean([f["quality"].get("hand_score", 0.0) for f in frames]))
        return landmarks, mask, timestamps, {"face": face_q, "pose": pose_q, "hand": hand_q}

    def _predict_arrays(self, landmarks, mask, timestamps, modality_quality):
        start = time.time()
        xyz, m = normalize_landmarks(
            landmarks,
            mask,
            schema=DEFAULT_SCHEMA,
            smooth_kernel=int(self.cfg.get("data", {}).get("smooth_kernel", 5)),
        )
        motion = build_motion_features(xyz, m)
        windows, wmasks, wts, starts = sliding_windows(
            motion,
            m,
            timestamps,
            window_size=self.window_size,
            stride=self.window_stride,
        )
        inputs = {
            "motion_windows": torch.from_numpy(windows).unsqueeze(0).to(self.device),
            "joint_mask": torch.from_numpy(wmasks).unsqueeze(0).to(self.device),
            "window_timestamps": torch.from_numpy(wts).unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            out = self.model(inputs)
            logit = out["logit_final"].detach().cpu().numpy().reshape(-1)[0]

        prob_raw = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
        logit_cal = float(apply_temperature(torch.tensor([logit]), self.temperature).item())
        prob_cal = float(1.0 / (1.0 + np.exp(-np.clip(logit_cal, -40.0, 40.0))))

        quality_score = float(
            np.clip(
                0.45 * modality_quality.get("pose", 0.0)
                + 0.30 * modality_quality.get("hand", 0.0)
                + 0.25 * modality_quality.get("face", 0.0),
                0.0,
                1.0,
            )
        )

        decision = make_decision(prob_raw, prob_cal, quality_score, self.quality_thr, self.low_thr, self.high_thr)
        window_scores = out.get("window_scores")
        attention_weights = out.get("attention_weights")
        evidence = []
        event_series = []
        ev_vec = out.get("event_vector_series")
        ev_time = out.get("event_time_series")
        ev_mask = out.get("event_mask_series")
        ev_frame_idx = out.get("event_frame_index_series")
        if ev_vec is not None and ev_time is not None and ev_mask is not None:
            # Shapes: [1,S,K,D], [1,S,K], [1,S,K]
            v = ev_vec.detach().cpu().numpy()[0]
            t_ev = ev_time.detach().cpu().numpy()[0]
            m_ev = ev_mask.detach().cpu().numpy()[0].astype(bool)
            fi = None
            if ev_frame_idx is not None:
                fi = ev_frame_idx.detach().cpu().numpy()[0]
            for s_idx in range(v.shape[0]):
                for k_idx in range(v.shape[1]):
                    if not bool(m_ev[s_idx, k_idx]):
                        continue
                    row = {
                        "window_index": int(s_idx),
                        "event_rank_in_window": int(k_idx),
                        "event_time_sec": float(t_ev[s_idx, k_idx]),
                        "event_vector": v[s_idx, k_idx].astype(float).tolist(),
                    }
                    if fi is not None:
                        row["event_frame_index_in_window"] = int(fi[s_idx, k_idx])
                    event_series.append(row)

        if window_scores is not None:
            ws = window_scores.detach().cpu().numpy().reshape(-1)
            top_idx = np.argsort(-ws)[: min(5, len(ws))]
            aw = None
            if attention_weights is not None:
                aw = attention_weights.detach().cpu().numpy().reshape(-1)
            for idx in top_idx:
                ev = {
                    "window_index": int(idx),
                    "window_score": float(ws[idx]),
                    "start_frame_est": int(starts[idx]),
                    "start_time_sec": float(wts[idx][0]) if idx < len(wts) else 0.0,
                }
                if aw is not None and idx < len(aw):
                    ev["attention_weight"] = float(aw[idx])
                evidence.append(ev)

        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "decision": decision.decision,
            "prob_raw": float(decision.prob_raw),
            "prob_calibrated": float(decision.prob_calibrated),
            "quality_score": float(decision.quality_score),
            "threshold_used": float(decision.threshold_used),
            "abstained": bool(decision.abstained),
            "reasons": list(decision.reasons),
            "window_evidence": evidence,
            "event_time_series": event_series,
            "events": evidence,
            "num_windows": int(windows.shape[0]),
            "window_size": int(self.window_size),
            "window_stride": int(self.window_stride),
            "model_version": self.model_version,
            "inference_ms": elapsed_ms,
        }

    def _resolve_processed_dir(self, ref: str, processed_root: str = None) -> str:
        root = processed_root or self.processed_root
        if os.path.isdir(ref):
            return ref
        candidate = os.path.join(root, make_video_id(ref))
        if os.path.isdir(candidate):
            return candidate
        candidate = os.path.join(root, ref)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Preprocessed data not found for '{ref}'. Tried '{os.path.join(root, make_video_id(ref))}' and '{os.path.join(root, ref)}'."
        )

    def _load_preprocessed_arrays(self, ref: str, processed_root: str = None):
        base_dir = self._resolve_processed_dir(ref, processed_root=processed_root)
        landmarks_path = os.path.join(base_dir, "landmarks.npy")
        mask_path = os.path.join(base_dir, "landmark_mask.npy")
        timestamps_path = os.path.join(base_dir, "timestamps.npy")
        quality_path = os.path.join(base_dir, "quality.json")
        if not (os.path.exists(landmarks_path) and os.path.exists(mask_path) and os.path.exists(timestamps_path)):
            raise FileNotFoundError(f"Missing preprocessed arrays under: {base_dir}")

        landmarks = np.load(landmarks_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        timestamps = np.load(timestamps_path).astype(np.float32)
        modality_quality = {"face": 0.0, "pose": 0.0, "hand": 0.0}
        if os.path.exists(quality_path):
            try:
                with open(quality_path, "r", encoding="utf-8") as f:
                    quality = json.load(f)
                modality_quality["face"] = float(np.mean([q.get("face_score", 0.0) for q in quality]))
                modality_quality["pose"] = float(np.mean([q.get("pose_score", 0.0) for q in quality]))
                modality_quality["hand"] = float(np.mean([q.get("hand_score", 0.0) for q in quality]))
            except Exception:
                pass
        return landmarks, mask, timestamps, modality_quality

    def predict_video(self, video_path: str) -> dict:
        processed = self.processor.process_video_file(video_path)
        landmarks, mask, timestamps, quality = self._frames_to_arrays(processed["frames"])
        return self._predict_arrays(landmarks, mask, timestamps, quality)

    def predict_preprocessed(self, processed_ref: str, processed_root: str = None) -> dict:
        landmarks, mask, timestamps, quality = self._load_preprocessed_arrays(processed_ref, processed_root)
        return self._predict_arrays(landmarks, mask, timestamps, quality)

    def predict_landmark_video(self, video_path: str) -> dict:
        return self.predict_video(video_path)
