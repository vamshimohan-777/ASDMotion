"""Inference module `src/inference/predictor.py` that converts inputs into runtime prediction outputs."""

# Import `json` to support computations in this stage of output generation.
import json
# Import `os` to support computations in this stage of output generation.
import os
# Import `time` to support computations in this stage of output generation.
import time

# Import `numpy as np` to support computations in this stage of output generation.
import numpy as np
# Import `torch` to support computations in this stage of output generation.
import torch

# Import symbols from `src.models.pipeline_model` used in this stage's output computation path.
from src.models.pipeline_model import ASDPipeline
# Import symbols from `src.models.video.mediapipe_layer.landmark_schema` used in this stage's output computation path.
from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
# Import symbols from `src.pipeline.preprocess` used in this stage's output computation path.
from src.pipeline.preprocess import VideoProcessor
# Import symbols from `src.utils.calibration` used in this stage's output computation path.
from src.utils.calibration import apply_temperature
# Import symbols from `src.utils.config` used in this stage's output computation path.
from src.utils.config import load_config
# Import symbols from `src.utils.decision` used in this stage's output computation path.
from src.utils.decision import make_decision
# Import symbols from `src.utils.video_id` used in this stage's output computation path.
from src.utils.video_id import make_video_id


# Define a reusable pipeline function whose outputs feed later steps.
def _moving_average_1d(x, k=5):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `k <= 1` to choose the correct output computation path.
    if k <= 1:
        # Return `x` as this function's contribution to downstream output flow.
        return x
    # Branch on `k % 2 == 0` to choose the correct output computation path.
    if k % 2 == 0:
        # Execute this statement so the returned prediction payload is correct.
        k += 1
    # Set `pad` for subsequent steps so the returned prediction payload is correct.
    pad = k // 2
    # Set `kernel` for subsequent steps so the returned prediction payload is correct.
    kernel = np.ones((k,), dtype=np.float32)
    # Call `kernel.sum` and use its result in later steps so the returned prediction payload is correct.
    kernel /= kernel.sum()
    # Compute `xp` as an intermediate representation used by later output layers.
    xp = np.pad(x, (pad, pad), mode="edge")
    # Return `np.convolve(xp, kernel, mode="valid")` as this function's contribution to downstream output flow.
    return np.convolve(xp, kernel, mode="valid")


# Define a reusable pipeline function whose outputs feed later steps.
def _fill_missing_1d(values, valid):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `out` for subsequent steps so the returned prediction payload is correct.
    out = values.copy()
    # Branch on `valid.sum() == 0` to choose the correct output computation path.
    if valid.sum() == 0:
        # Return `out` as this function's contribution to downstream output flow.
        return out
    # Compute `idx` as an intermediate representation used by later output layers.
    idx = np.where(valid > 0.5)[0]
    # Set `first` for subsequent steps so the returned prediction payload is correct.
    first = idx[0]
    # Set `last` for subsequent steps so the returned prediction payload is correct.
    last = idx[-1]
    # Execute this statement so the returned prediction payload is correct.
    out[:first] = out[first]
    # Execute this statement so the returned prediction payload is correct.
    out[last + 1:] = out[last]
    # Set `miss` for subsequent steps so the returned prediction payload is correct.
    miss = np.where(valid < 0.5)[0]
    # Iterate over `miss` so each item contributes to final outputs/metrics.
    for m in miss:
        # Set `left` for subsequent steps so the returned prediction payload is correct.
        left = idx[idx < m]
        # Compute `right` as an intermediate representation used by later output layers.
        right = idx[idx > m]
        # Branch on `left.size == 0 and right.size == 0` to choose the correct output computation path.
        if left.size == 0 and right.size == 0:
            # Skip current loop item so it does not affect accumulated output state.
            continue
        # Branch on `left.size == 0` to choose the correct output computation path.
        if left.size == 0:
            # Set `out[m]` for subsequent steps so the returned prediction payload is correct.
            out[m] = out[right[0]]
        # Use alternate condition `right.size == 0` to refine output path selection.
        elif right.size == 0:
            # Set `out[m]` for subsequent steps so the returned prediction payload is correct.
            out[m] = out[left[-1]]
        else:
            # Set `l` for subsequent steps so the returned prediction payload is correct.
            l = left[-1]
            # Set `r` for subsequent steps so the returned prediction payload is correct.
            r = right[0]
            # Compute `alpha` as an intermediate representation used by later output layers.
            alpha = (m - l) / max((r - l), 1)
            # Set `out[m]` for subsequent steps so the returned prediction payload is correct.
            out[m] = (1.0 - alpha) * out[l] + alpha * out[r]
    # Return `out` as this function's contribution to downstream output flow.
    return out


# Define a reusable pipeline function whose outputs feed later steps.
def normalize_landmarks(landmarks, mask, schema=DEFAULT_SCHEMA, smooth_kernel=5):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Compute `xyz` as an intermediate representation used by later output layers.
    xyz = landmarks.copy()
    # Build `m` to gate invalid timesteps/joints from influencing outputs.
    m = mask.copy()
    # Set `t, j, _` for subsequent steps so the returned prediction payload is correct.
    t, j, _ = xyz.shape

    # Iterate over `range(j)` so each item contributes to final outputs/metrics.
    for jj in range(j):
        # Set `valid` for subsequent steps so the returned prediction payload is correct.
        valid = m[:, jj] > 0.5
        # Iterate over `range(3)` so each item contributes to final outputs/metrics.
        for c in range(3):
            # Call `_fill_missing_1d` and use its result in later steps so the returned prediction payload is correct.
            xyz[:, jj, c] = _fill_missing_1d(xyz[:, jj, c], valid)

    # Hip center
    # Compute `l_hip, r_hip` as an intermediate representation used by later output layers.
    l_hip, r_hip = 23, 24
    # Compute `hips_valid` as an intermediate representation used by later output layers.
    hips_valid = (m[:, l_hip] > 0.5) & (m[:, r_hip] > 0.5)
    # Set `center` for subsequent steps so the returned prediction payload is correct.
    center = np.zeros((t, 3), dtype=np.float32)
    # Compute `center[hips_valid]` as an intermediate representation used by later output layers.
    center[hips_valid] = 0.5 * (xyz[hips_valid, l_hip] + xyz[hips_valid, r_hip])
    # Branch on `hips_valid.any()` to choose the correct output computation path.
    if hips_valid.any():
        # Set `last` for subsequent steps so the returned prediction payload is correct.
        last = center[np.where(hips_valid)[0][0]].copy()
    else:
        # Set `last` for subsequent steps so the returned prediction payload is correct.
        last = np.zeros((3,), dtype=np.float32)
    # Iterate over `range(t)` so each item contributes to final outputs/metrics.
    for i in range(t):
        # Branch on `hips_valid[i]` to choose the correct output computation path.
        if hips_valid[i]:
            # Set `last` for subsequent steps so the returned prediction payload is correct.
            last = center[i]
        else:
            # Set `center[i]` for subsequent steps so the returned prediction payload is correct.
            center[i] = last
    # Compute `xyz` as an intermediate representation used by later output layers.
    xyz = xyz - center[:, None, :]

    # Shoulder scale
    # Compute `l_sh, r_sh` as an intermediate representation used by later output layers.
    l_sh, r_sh = 11, 12
    # Compute `sh_valid` as an intermediate representation used by later output layers.
    sh_valid = (m[:, l_sh] > 0.5) & (m[:, r_sh] > 0.5)
    # Set `scale` for subsequent steps so the returned prediction payload is correct.
    scale = np.ones((t,), dtype=np.float32)
    # Branch on `sh_valid.any()` to choose the correct output computation path.
    if sh_valid.any():
        # Set `d` for subsequent steps so the returned prediction payload is correct.
        d = np.linalg.norm(xyz[:, l_sh] - xyz[:, r_sh], axis=-1)
        # Set `d` for subsequent steps so the returned prediction payload is correct.
        d = np.clip(d, 1e-4, None)
        # Compute `scale[sh_valid]` as an intermediate representation used by later output layers.
        scale[sh_valid] = d[sh_valid]
        # Call `float` and use its result in later steps so the returned prediction payload is correct.
        scale[~sh_valid] = float(np.median(scale[sh_valid]))
    # Compute `xyz` as an intermediate representation used by later output layers.
    xyz = xyz / scale[:, None, None]

    # Iterate over `range(j)` so each item contributes to final outputs/metrics.
    for jj in range(j):
        # Set `valid` for subsequent steps so the returned prediction payload is correct.
        valid = m[:, jj]
        # Iterate over `range(3)` so each item contributes to final outputs/metrics.
        for c in range(3):
            # Set `sm` for subsequent steps so the returned prediction payload is correct.
            sm = _moving_average_1d(xyz[:, jj, c], k=smooth_kernel)
            # Call `this call` and use its result in later steps so the returned prediction payload is correct.
            xyz[:, jj, c] = (valid * sm) + ((1.0 - valid) * xyz[:, jj, c])
    # Return `xyz, m` as this function's contribution to downstream output flow.
    return xyz, m


# Define a reusable pipeline function whose outputs feed later steps.
def build_motion_features(xyz, mask):
    """Constructs components whose structure controls later training or inference outputs."""
    # Set `vel` for subsequent steps so the returned prediction payload is correct.
    vel = np.zeros_like(xyz, dtype=np.float32)
    # Set `acc` for subsequent steps so the returned prediction payload is correct.
    acc = np.zeros_like(xyz, dtype=np.float32)
    # Branch on `xyz.shape[0] > 1` to choose the correct output computation path.
    if xyz.shape[0] > 1:
        # Execute this statement so the returned prediction payload is correct.
        vel[1:] = xyz[1:] - xyz[:-1]
        # Execute this statement so the returned prediction payload is correct.
        acc[1:] = vel[1:] - vel[:-1]
    # Compute `feat` as an intermediate representation used by later output layers.
    feat = np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)
    # Build `feat *` to gate invalid timesteps/joints from influencing outputs.
    feat *= mask[..., None]
    # Return `feat` as this function's contribution to downstream output flow.
    return feat


# Define a reusable pipeline function whose outputs feed later steps.
def sliding_windows(motion, mask, timestamps, window_size=64, stride=16):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `t` for subsequent steps so the returned prediction payload is correct.
    t = motion.shape[0]
    # Branch on `t <= window_size` to choose the correct output computation path.
    if t <= window_size:
        # Set `starts` for subsequent steps so the returned prediction payload is correct.
        starts = [0]
    else:
        # Set `starts` for subsequent steps so the returned prediction payload is correct.
        starts = list(range(0, max(1, t - window_size + 1), max(1, stride)))
        # Branch on `starts[-1] != (t - window_size)` to choose the correct output computation path.
        if starts[-1] != (t - window_size):
            # Call `starts.append` and use its result in later steps so the returned prediction payload is correct.
            starts.append(t - window_size)

    # Compute `windows` as an intermediate representation used by later output layers.
    windows = []
    # Build `masks` to gate invalid timesteps/joints from influencing outputs.
    masks = []
    # Compute `ts_windows` as an intermediate representation used by later output layers.
    ts_windows = []
    # Iterate over `starts` so each item contributes to final outputs/metrics.
    for s in starts:
        # Set `e` for subsequent steps so the returned prediction payload is correct.
        e = s + window_size
        # Set `w` for subsequent steps so the returned prediction payload is correct.
        w = motion[s:e]
        # Build `m` to gate invalid timesteps/joints from influencing outputs.
        m = mask[s:e]
        # Set `ts` for subsequent steps so the returned prediction payload is correct.
        ts = timestamps[s:e]
        # Branch on `w.shape[0] < window_size` to choose the correct output computation path.
        if w.shape[0] < window_size:
            # Set `pad` for subsequent steps so the returned prediction payload is correct.
            pad = window_size - w.shape[0]
            # Set `w` for subsequent steps so the returned prediction payload is correct.
            w = np.pad(w, ((0, pad), (0, 0), (0, 0)), mode="constant")
            # Set `m` for subsequent steps so the returned prediction payload is correct.
            m = np.pad(m, ((0, pad), (0, 0)), mode="constant")
            # Set `ts` for subsequent steps so the returned prediction payload is correct.
            ts = np.pad(ts, (0, pad), mode="edge" if ts.size > 0 else "constant")
        # Call `windows.append` and use its result in later steps so the returned prediction payload is correct.
        windows.append(w.astype(np.float32))
        # Call `masks.append` and use its result in later steps so the returned prediction payload is correct.
        masks.append(m.astype(np.float32))
        # Call `ts_windows.append` and use its result in later steps so the returned prediction payload is correct.
        ts_windows.append(ts.astype(np.float32))
    # Return `np.stack(windows), np.stack(masks), np.stack(ts_win...` as this function's contribution to downstream output flow.
    return np.stack(windows), np.stack(masks), np.stack(ts_windows), starts


# Define class `ASDPredictor` to package related logic in the prediction pipeline.
class ASDPredictor:
    """`ASDPredictor` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = None):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `not os.path.exists(checkpoint_path)` to choose the correct output computation path.
        if not os.path.exists(checkpoint_path):
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # Set `self.device` to the execution device used for this computation path.
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Capture `ckpt` as model state controlling reproducible output behavior.
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # Set `cfg` for subsequent steps so the returned prediction payload is correct.
        cfg = ckpt.get("config", None)
        # Branch on `cfg is None and config_path` to choose the correct output computation path.
        if cfg is None and config_path:
            # Set `cfg` for subsequent steps so the returned prediction payload is correct.
            cfg = load_config(config_path)
        # Branch on `cfg is None` to choose the correct output computation path.
        if cfg is None:
            # Set `cfg` for subsequent steps so the returned prediction payload is correct.
            cfg = {}
        # Set `self.cfg` for subsequent steps so the returned prediction payload is correct.
        self.cfg = cfg

        # Set `model_cfg` for subsequent steps so the returned prediction payload is correct.
        model_cfg = cfg.get("model", {})
        # Compute `th` as an intermediate representation used by later output layers.
        th = cfg.get("thresholds", {})
        # Set `self.model` for subsequent steps so the returned prediction payload is correct.
        self.model = ASDPipeline(
            K_max=int(model_cfg.get("K_max", 16)),
            d_model=int(model_cfg.get("d_model", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            theta_high=float(th.get("decision_high", 0.7)),
            theta_low=float(th.get("decision_low", 0.3)),
        ).to(self.device)
        # Compute `nas_arch` as an intermediate representation used by later output layers.
        nas_arch = ckpt.get("nas_architecture", None)
        # Branch on `nas_arch is not None` to choose the correct output computation path.
        if nas_arch is not None:
            # Call `self.model.apply_nas_architecture` and use its result in later steps so the returned prediction payload is correct.
            self.model.apply_nas_architecture(nas_arch)
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Load trained weights that directly determine inference outputs.
            self.model.load_state_dict(ckpt["model_state"], strict=True)
        # Handle exceptions and keep output behavior controlled under error conditions.
        except Exception as exc:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise RuntimeError(
                "Checkpoint is incompatible with the landmark-motion model. "
                "Train a new checkpoint with `python -m src.training.train --config config.yaml`."
            ) from exc
        # Call `self.model.eval` and use its result in later steps so the returned prediction payload is correct.
        self.model.eval()

        # Set `self.temperature` for subsequent steps so the returned prediction payload is correct.
        self.temperature = float(ckpt.get("temperature", 1.0))
        # Set `self.processor` for subsequent steps so the returned prediction payload is correct.
        self.processor = VideoProcessor(
            frame_stride=int(cfg.get("data", {}).get("frame_stride", 1)),
            max_frames=int(cfg.get("data", {}).get("max_frames", 0)),
        )
        # Set `self.processed_root` for subsequent steps so the returned prediction payload is correct.
        self.processed_root = str(cfg.get("data", {}).get("processed_root", "data/processed"))
        # Compute `self.quality_thr` as an intermediate representation used by later output layers.
        self.quality_thr = float(th.get("quality_threshold", 0.5))
        # Compute `self.low_thr` as an intermediate representation used by later output layers.
        self.low_thr = float(th.get("decision_low", 0.3))
        # Compute `self.high_thr` as an intermediate representation used by later output layers.
        self.high_thr = float(th.get("decision_high", 0.7))
        # Compute `self.window_size` as an intermediate representation used by later output layers.
        self.window_size = int(cfg.get("inference", {}).get("window_size", 64))
        # Compute `self.window_stride` as an intermediate representation used by later output layers.
        self.window_stride = int(cfg.get("inference", {}).get("window_stride", 16))
        # Set `self.model_version` for subsequent steps so the returned prediction payload is correct.
        self.model_version = str(cfg.get("inference", {}).get("model_version", "asd_motion_landmark_v2"))

    # Define a reusable pipeline function whose outputs feed later steps.
    def _frames_to_arrays(self, frames):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `not frames` to choose the correct output computation path.
        if not frames:
            # Set `j` for subsequent steps so the returned prediction payload is correct.
            j = DEFAULT_SCHEMA.total_joints
            # Return `(` as this function's contribution to downstream output flow.
            return (
                np.zeros((1, j, 3), dtype=np.float32),
                np.zeros((1, j), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                {"face": 0.0, "pose": 0.0, "hand": 0.0},
            )
        # Set `landmarks` for subsequent steps so the returned prediction payload is correct.
        landmarks = np.stack([f["landmarks"] for f in frames]).astype(np.float32)
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = np.stack([f["mask"] for f in frames]).astype(np.float32)
        # Set `timestamps` for subsequent steps so the returned prediction payload is correct.
        timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.float32)
        # Set `face_q` for subsequent steps so the returned prediction payload is correct.
        face_q = float(np.mean([f["quality"].get("face_score", 0.0) for f in frames]))
        # Set `pose_q` for subsequent steps so the returned prediction payload is correct.
        pose_q = float(np.mean([f["quality"].get("pose_score", 0.0) for f in frames]))
        # Compute `hand_q` as an intermediate representation used by later output layers.
        hand_q = float(np.mean([f["quality"].get("hand_score", 0.0) for f in frames]))
        # Return `landmarks, mask, timestamps, {"face": face_q, "pose...` as this function's contribution to downstream output flow.
        return landmarks, mask, timestamps, {"face": face_q, "pose": pose_q, "hand": hand_q}

    # Define inference logic that produces the prediction returned to callers.
    def _predict_arrays(self, landmarks, mask, timestamps, modality_quality):
        """Builds inference outputs from inputs and returns values consumed by users or services."""
        # Set `start` for subsequent steps so the returned prediction payload is correct.
        start = time.time()
        # Compute `xyz, m` as an intermediate representation used by later output layers.
        xyz, m = normalize_landmarks(
            landmarks,
            mask,
            schema=DEFAULT_SCHEMA,
            smooth_kernel=int(self.cfg.get("data", {}).get("smooth_kernel", 5)),
        )
        # Set `motion` for subsequent steps so the returned prediction payload is correct.
        motion = build_motion_features(xyz, m)
        # Build `windows, wmasks, wts, starts` to gate invalid timesteps/joints from influencing outputs.
        windows, wmasks, wts, starts = sliding_windows(
            motion,
            m,
            timestamps,
            window_size=self.window_size,
            stride=self.window_stride,
        )
        # Set `inputs` for subsequent steps so the returned prediction payload is correct.
        inputs = {
            "motion_windows": torch.from_numpy(windows).unsqueeze(0).to(self.device),
            "joint_mask": torch.from_numpy(wmasks).unsqueeze(0).to(self.device),
            "window_timestamps": torch.from_numpy(wts).unsqueeze(0).to(self.device),
        }
        # Use a managed context to safely handle resources used during output computation.
        with torch.no_grad():
            # Set `out` for subsequent steps so the returned prediction payload is correct.
            out = self.model(inputs)
            # Store raw score tensor in `logit` before probability/decision conversion.
            logit = out["logit_final"].detach().cpu().numpy().reshape(-1)[0]

        # Store raw score tensor in `prob_raw` before probability/decision conversion.
        prob_raw = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
        # Store raw score tensor in `logit_cal` before probability/decision conversion.
        logit_cal = float(apply_temperature(torch.tensor([logit]), self.temperature).item())
        # Store raw score tensor in `prob_cal` before probability/decision conversion.
        prob_cal = float(1.0 / (1.0 + np.exp(-np.clip(logit_cal, -40.0, 40.0))))

        # Set `quality_score` for subsequent steps so the returned prediction payload is correct.
        quality_score = float(
            np.clip(
                0.45 * modality_quality.get("pose", 0.0)
                + 0.30 * modality_quality.get("hand", 0.0)
                + 0.25 * modality_quality.get("face", 0.0),
                0.0,
                1.0,
            )
        )

        # Set `decision` for subsequent steps so the returned prediction payload is correct.
        decision = make_decision(prob_raw, prob_cal, quality_score, self.quality_thr, self.low_thr, self.high_thr)
        # Compute `window_scores` as an intermediate representation used by later output layers.
        window_scores = out.get("window_scores")
        # Compute `attention_weights` as an intermediate representation used by later output layers.
        attention_weights = out.get("attention_weights")
        # Set `evidence` for subsequent steps so the returned prediction payload is correct.
        evidence = []
        # Set `event_series` for subsequent steps so the returned prediction payload is correct.
        event_series = []
        # Set `ev_vec` for subsequent steps so the returned prediction payload is correct.
        ev_vec = out.get("event_vector_series")
        # Set `ev_time` for subsequent steps so the returned prediction payload is correct.
        ev_time = out.get("event_time_series")
        # Build `ev_mask` to gate invalid timesteps/joints from influencing outputs.
        ev_mask = out.get("event_mask_series")
        # Compute `ev_frame_idx` as an intermediate representation used by later output layers.
        ev_frame_idx = out.get("event_frame_index_series")
        # Branch on `ev_vec is not None and ev_time is not None and ev...` to choose the correct output computation path.
        if ev_vec is not None and ev_time is not None and ev_mask is not None:
            # Shapes: [1,S,K,D], [1,S,K], [1,S,K]
            # Set `v` for subsequent steps so the returned prediction payload is correct.
            v = ev_vec.detach().cpu().numpy()[0]
            # Set `t_ev` for subsequent steps so the returned prediction payload is correct.
            t_ev = ev_time.detach().cpu().numpy()[0]
            # Build `m_ev` to gate invalid timesteps/joints from influencing outputs.
            m_ev = ev_mask.detach().cpu().numpy()[0].astype(bool)
            # Set `fi` for subsequent steps so the returned prediction payload is correct.
            fi = None
            # Branch on `ev_frame_idx is not None` to choose the correct output computation path.
            if ev_frame_idx is not None:
                # Set `fi` for subsequent steps so the returned prediction payload is correct.
                fi = ev_frame_idx.detach().cpu().numpy()[0]
            # Iterate over `range(v.shape[0])` so each item contributes to final outputs/metrics.
            for s_idx in range(v.shape[0]):
                # Iterate over `range(v.shape[1])` so each item contributes to final outputs/metrics.
                for k_idx in range(v.shape[1]):
                    # Branch on `not bool(m_ev[s_idx, k_idx])` to choose the correct output computation path.
                    if not bool(m_ev[s_idx, k_idx]):
                        # Skip current loop item so it does not affect accumulated output state.
                        continue
                    # Set `row` for subsequent steps so the returned prediction payload is correct.
                    row = {
                        "window_index": int(s_idx),
                        "event_rank_in_window": int(k_idx),
                        "event_time_sec": float(t_ev[s_idx, k_idx]),
                        "event_vector": v[s_idx, k_idx].astype(float).tolist(),
                    }
                    # Branch on `fi is not None` to choose the correct output computation path.
                    if fi is not None:
                        # Call `int` and use its result in later steps so the returned prediction payload is correct.
                        row["event_frame_index_in_window"] = int(fi[s_idx, k_idx])
                    # Call `event_series.append` and use its result in later steps so the returned prediction payload is correct.
                    event_series.append(row)

        # Branch on `window_scores is not None` to choose the correct output computation path.
        if window_scores is not None:
            # Set `ws` for subsequent steps so the returned prediction payload is correct.
            ws = window_scores.detach().cpu().numpy().reshape(-1)
            # Compute `top_idx` as an intermediate representation used by later output layers.
            top_idx = np.argsort(-ws)[: min(5, len(ws))]
            # Set `aw` for subsequent steps so the returned prediction payload is correct.
            aw = None
            # Branch on `attention_weights is not None` to choose the correct output computation path.
            if attention_weights is not None:
                # Set `aw` for subsequent steps so the returned prediction payload is correct.
                aw = attention_weights.detach().cpu().numpy().reshape(-1)
            # Iterate over `top_idx` so each item contributes to final outputs/metrics.
            for idx in top_idx:
                # Set `ev` for subsequent steps so the returned prediction payload is correct.
                ev = {
                    "window_index": int(idx),
                    "window_score": float(ws[idx]),
                    "start_frame_est": int(starts[idx]),
                    "start_time_sec": float(wts[idx][0]) if idx < len(wts) else 0.0,
                }
                # Branch on `aw is not None and idx < len(aw)` to choose the correct output computation path.
                if aw is not None and idx < len(aw):
                    # Call `float` and use its result in later steps so the returned prediction payload is correct.
                    ev["attention_weight"] = float(aw[idx])
                # Call `evidence.append` and use its result in later steps so the returned prediction payload is correct.
                evidence.append(ev)

        # Set `elapsed_ms` for subsequent steps so the returned prediction payload is correct.
        elapsed_ms = int((time.time() - start) * 1000)
        # Return `{` as this function's contribution to downstream output flow.
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

    # Define a reusable pipeline function whose outputs feed later steps.
    def _resolve_processed_dir(self, ref: str, processed_root: str = None) -> str:
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `root` for subsequent steps so the returned prediction payload is correct.
        root = processed_root or self.processed_root
        # Branch on `os.path.isdir(ref)` to choose the correct output computation path.
        if os.path.isdir(ref):
            # Return `ref` as this function's contribution to downstream output flow.
            return ref
        # Set `candidate` for subsequent steps so the returned prediction payload is correct.
        candidate = os.path.join(root, make_video_id(ref))
        # Branch on `os.path.isdir(candidate)` to choose the correct output computation path.
        if os.path.isdir(candidate):
            # Return `candidate` as this function's contribution to downstream output flow.
            return candidate
        # Set `candidate` for subsequent steps so the returned prediction payload is correct.
        candidate = os.path.join(root, ref)
        # Branch on `os.path.isdir(candidate)` to choose the correct output computation path.
        if os.path.isdir(candidate):
            # Return `candidate` as this function's contribution to downstream output flow.
            return candidate
        # Raise explicit error to stop invalid state from producing misleading outputs.
        raise FileNotFoundError(
            f"Preprocessed data not found for '{ref}'. Tried '{os.path.join(root, make_video_id(ref))}' and '{os.path.join(root, ref)}'."
        )

    # Define loading logic for config/weights that determine runtime behavior.
    def _load_preprocessed_arrays(self, ref: str, processed_root: str = None):
        """Loads configuration or weights that define how subsequent computations produce outputs."""
        # Set `base_dir` for subsequent steps so the returned prediction payload is correct.
        base_dir = self._resolve_processed_dir(ref, processed_root=processed_root)
        # Compute `landmarks_path` as an intermediate representation used by later output layers.
        landmarks_path = os.path.join(base_dir, "landmarks.npy")
        # Build `mask_path` to gate invalid timesteps/joints from influencing outputs.
        mask_path = os.path.join(base_dir, "landmark_mask.npy")
        # Compute `timestamps_path` as an intermediate representation used by later output layers.
        timestamps_path = os.path.join(base_dir, "timestamps.npy")
        # Compute `quality_path` as an intermediate representation used by later output layers.
        quality_path = os.path.join(base_dir, "quality.json")
        # Branch on `not (os.path.exists(landmarks_path) and os.path.e...` to choose the correct output computation path.
        if not (os.path.exists(landmarks_path) and os.path.exists(mask_path) and os.path.exists(timestamps_path)):
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise FileNotFoundError(f"Missing preprocessed arrays under: {base_dir}")

        # Set `landmarks` for subsequent steps so the returned prediction payload is correct.
        landmarks = np.load(landmarks_path).astype(np.float32)
        # Build `mask` to gate invalid timesteps/joints from influencing outputs.
        mask = np.load(mask_path).astype(np.float32)
        # Set `timestamps` for subsequent steps so the returned prediction payload is correct.
        timestamps = np.load(timestamps_path).astype(np.float32)
        # Set `modality_quality` for subsequent steps so the returned prediction payload is correct.
        modality_quality = {"face": 0.0, "pose": 0.0, "hand": 0.0}
        # Branch on `os.path.exists(quality_path)` to choose the correct output computation path.
        if os.path.exists(quality_path):
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Use a managed context to safely handle resources used during output computation.
                with open(quality_path, "r", encoding="utf-8") as f:
                    # Set `quality` for subsequent steps so the returned prediction payload is correct.
                    quality = json.load(f)
                # Call `float` and use its result in later steps so the returned prediction payload is correct.
                modality_quality["face"] = float(np.mean([q.get("face_score", 0.0) for q in quality]))
                # Call `float` and use its result in later steps so the returned prediction payload is correct.
                modality_quality["pose"] = float(np.mean([q.get("pose_score", 0.0) for q in quality]))
                # Call `float` and use its result in later steps so the returned prediction payload is correct.
                modality_quality["hand"] = float(np.mean([q.get("hand_score", 0.0) for q in quality]))
            # Handle exceptions and keep output behavior controlled under error conditions.
            except Exception:
                # No-op placeholder that keeps control-flow structure intact.
                pass
        # Return `landmarks, mask, timestamps, modality_quality` as this function's contribution to downstream output flow.
        return landmarks, mask, timestamps, modality_quality

    # Define inference logic that produces the prediction returned to callers.
    def predict_video(self, video_path: str) -> dict:
        """Builds inference outputs from inputs and returns values consumed by users or services."""
        # Set `processed` for subsequent steps so the returned prediction payload is correct.
        processed = self.processor.process_video_file(video_path)
        # Build `landmarks, mask, timestamps...` to gate invalid timesteps/joints from influencing outputs.
        landmarks, mask, timestamps, quality = self._frames_to_arrays(processed["frames"])
        # Return `self._predict_arrays(landmarks, mask, timestamps, q...` as this function's contribution to downstream output flow.
        return self._predict_arrays(landmarks, mask, timestamps, quality)

    # Define inference logic that produces the prediction returned to callers.
    def predict_preprocessed(self, processed_ref: str, processed_root: str = None) -> dict:
        """Builds inference outputs from inputs and returns values consumed by users or services."""
        # Build `landmarks, mask, timestamps...` to gate invalid timesteps/joints from influencing outputs.
        landmarks, mask, timestamps, quality = self._load_preprocessed_arrays(processed_ref, processed_root)
        # Return `self._predict_arrays(landmarks, mask, timestamps, q...` as this function's contribution to downstream output flow.
        return self._predict_arrays(landmarks, mask, timestamps, quality)

    # Define inference logic that produces the prediction returned to callers.
    def predict_landmark_video(self, video_path: str) -> dict:
        """Builds inference outputs from inputs and returns values consumed by users or services."""
        # Return `self.predict_video(video_path)` as this function's contribution to downstream output flow.
        return self.predict_video(video_path)
