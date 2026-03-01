"""Training module `src/training/train.py` that optimizes model weights and output quality."""

# Import `argparse` to support computations in this stage of output generation.
import argparse
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
# Import symbols from `torch.utils.data` used in this stage's output computation path.
from torch.utils.data import DataLoader, Subset

# Import symbols from `src.models.pipeline_model` used in this stage's output computation path.
from src.models.pipeline_model import ASDPipeline
# Import symbols from `src.pipeline.preprocess` used in this stage's output computation path.
from src.pipeline.preprocess import precompute_videos
# Import symbols from `src.training.checkpoints` used in this stage's output computation path.
from src.training.checkpoints import CheckpointManager
# Import symbols from `src.training.dataset` used in this stage's output computation path.
from src.training.dataset import VideoDataset, collate_motion_batch
# Import symbols from `src.training.logging_utils` used in this stage's output computation path.
from src.training.logging_utils import ExperimentLogger, export_experiment_log_pdf
# Import symbols from `src.training.losses` used in this stage's output computation path.
from src.training.losses import WeightedBCELoss, event_gate_bag_loss
# Import symbols from `src.training.motion_ssl` used in this stage's output computation path.
from src.training.motion_ssl import pretrain_motion_encoder
# Import symbols from `src.training.nas_search` used in this stage's output computation path.
from src.training.nas_search import run_micro_genetic_nas
# Import symbols from `src.utils.calibration` used in this stage's output computation path.
from src.utils.calibration import apply_temperature, fit_temperature
# Import symbols from `src.utils.config` used in this stage's output computation path.
from src.utils.config import apply_overrides, load_config
# Import symbols from `src.utils.metrics` used in this stage's output computation path.
from src.utils.metrics import (
    compute_auc,
    compute_basic_metrics,
    compute_ece,
    find_optimal_threshold,
    sensitivity_at_specificity,
)
# Import symbols from `src.utils.seed` used in this stage's output computation path.
from src.utils.seed import seed_everything, seed_worker
# Import symbols from `src.utils.splits` used in this stage's output computation path.
from src.utils.splits import check_group_overlap, make_group_kfold, make_group_stratified_split


# Define a reusable pipeline function whose outputs feed later steps.
def _write_status_file(status_file: str, payload: dict):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `not status_file` to choose the correct output computation path.
    if not status_file:
        # Return control/value to the caller for the next output-processing step.
        return
    # Set `directory` for subsequent steps so gradient updates improve future predictions.
    directory = os.path.dirname(status_file)
    # Branch on `directory` to choose the correct output computation path.
    if directory:
        # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
        os.makedirs(directory, exist_ok=True)
    # Set `tmp` for subsequent steps so gradient updates improve future predictions.
    tmp = f"{status_file}.tmp"
    # Use a managed context to safely handle resources used during output computation.
    with open(tmp, "w", encoding="utf-8") as f:
        # Call `json.dump` and use its result in later steps so gradient updates improve future predictions.
        json.dump(payload, f, indent=2)
    # Call `os.replace` and use its result in later steps so gradient updates improve future predictions.
    os.replace(tmp, status_file)


# Define a reusable pipeline function whose outputs feed later steps.
def _is_rtx_4050(device):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `not str(device).startswith("cuda") or not torch.c...` to choose the correct output computation path.
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        # Return `False` as this function's contribution to downstream output flow.
        return False
    # Start guarded block so failures can be handled without breaking output flow.
    try:
        # Set `props` for subsequent steps so gradient updates improve future predictions.
        props = torch.cuda.get_device_properties(device.index if device.index is not None else 0)
    # Handle exceptions and keep output behavior controlled under error conditions.
    except Exception:
        # Return `False` as this function's contribution to downstream output flow.
        return False
    # Return `"rtx 4050" in str(props.name).lower()` as this function's contribution to downstream output flow.
    return "rtx 4050" in str(props.name).lower()


# Define a reusable pipeline function whose outputs feed later steps.
def _auto_batch_and_workers(cfg, device):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `train_cfg` for subsequent steps so gradient updates improve future predictions.
    train_cfg = cfg.setdefault("training", {})
    # Set `data_cfg` for subsequent steps so gradient updates improve future predictions.
    data_cfg = cfg.setdefault("data", {})
    # Branch on `not _is_rtx_4050(device)` to choose the correct output computation path.
    if not _is_rtx_4050(device):
        # Return control/value to the caller for the next output-processing step.
        return
    # Branch on `"rtx4050_batch_size" in train_cfg` to choose the correct output computation path.
    if "rtx4050_batch_size" in train_cfg:
        # Call `int` and use its result in later steps so gradient updates improve future predictions.
        train_cfg["batch_size"] = int(train_cfg["rtx4050_batch_size"])
    # Branch on `"rtx4050_num_workers" in data_cfg` to choose the correct output computation path.
    if "rtx4050_num_workers" in data_cfg:
        # Call `int` and use its result in later steps so gradient updates improve future predictions.
        data_cfg["num_workers"] = int(data_cfg["rtx4050_num_workers"])


# Define a reusable pipeline function whose outputs feed later steps.
def _build_dataset(cfg, is_training):
    """Constructs components whose structure controls later training or inference outputs."""
    # Set `data_cfg` for subsequent steps so gradient updates improve future predictions.
    data_cfg = cfg.get("data", {})
    # Return `VideoDataset(` as this function's contribution to downstream output flow.
    return VideoDataset(
        csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
        sequence_length=int(data_cfg.get("seq_len", 64)),
        is_training=bool(is_training),
        use_preprocessed=bool(data_cfg.get("use_preprocessed", True)),
        processed_root=str(data_cfg.get("processed_root", "data/processed")),
        window_sizes=tuple(data_cfg.get("window_sizes", [32, 48, 64])),
        windows_per_video=int(data_cfg.get("windows_per_video", 8)),
        eval_windows_per_video=int(data_cfg.get("eval_windows_per_video", 12)),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        max_frames=int(data_cfg.get("max_frames", 0)),
        cache_enabled=bool(data_cfg.get("cache_enabled", True)),
        smooth_kernel=int(data_cfg.get("smooth_kernel", 5)),
    )


# Define loading logic for config/weights that determine runtime behavior.
def _build_loader(dataset, cfg, shuffle, generator=None):
    """Loads configuration or weights that define how subsequent computations produce outputs."""
    # Set `data_cfg` for subsequent steps so gradient updates improve future predictions.
    data_cfg = cfg.get("data", {})
    # Set `train_cfg` for subsequent steps so gradient updates improve future predictions.
    train_cfg = cfg.get("training", {})
    # Set `num_workers` for subsequent steps so gradient updates improve future predictions.
    num_workers = int(data_cfg.get("num_workers", 0))
    # Set `kwargs` for subsequent steps so gradient updates improve future predictions.
    kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 2)),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": collate_motion_batch,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    # Branch on `num_workers > 0` to choose the correct output computation path.
    if num_workers > 0:
        # Call `bool` and use its result in later steps so gradient updates improve future predictions.
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        # Call `int` and use its result in later steps so gradient updates improve future predictions.
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    # Return `DataLoader(dataset, **kwargs)` as this function's contribution to downstream output flow.
    return DataLoader(dataset, **kwargs)


# Define a reusable pipeline function whose outputs feed later steps.
def _to_inputs(batch, device):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "motion_windows": batch["motion_windows"].to(device, non_blocking=True),
        "joint_mask": batch["joint_mask"].to(device, non_blocking=True),
        "window_timestamps": batch["window_timestamps"].to(device, non_blocking=True),
    }


# Define a reusable pipeline function whose outputs feed later steps.
def _quality_score_from_batch(batch):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `q` for subsequent steps so gradient updates improve future predictions.
    q = batch["qualities"]
    # Set `face` for subsequent steps so gradient updates improve future predictions.
    face = q["face_score"].float()
    # Set `pose` for subsequent steps so gradient updates improve future predictions.
    pose = q["pose_score"].float()
    # Compute `hand` as an intermediate representation used by later output layers.
    hand = q["hand_score"].float()
    # Set `score` for subsequent steps so gradient updates improve future predictions.
    score = 0.45 * pose + 0.30 * hand + 0.25 * face
    # Return `score.clamp(0.0, 1.0)` as this function's contribution to downstream output flow.
    return score.clamp(0.0, 1.0)


# Define a training routine that updates parameters and changes future outputs.
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    clip_grad=1.0,
    gate_aux_weight=0.0,
):
    """Executes a training step/loop that updates parameters and directly changes model output behavior."""
    # Call `model.train` and use its result in later steps so gradient updates improve future predictions.
    model.train()
    # Set `total` for subsequent steps so gradient updates improve future predictions.
    total = 0.0
    # Set `n` for subsequent steps so gradient updates improve future predictions.
    n = 0
    # Iterate over `loader` so each item contributes to final outputs/metrics.
    for batch in loader:
        # Set `labels` for subsequent steps so gradient updates improve future predictions.
        labels = batch["label"].to(device, non_blocking=True)
        # Set `inputs` for subsequent steps so gradient updates improve future predictions.
        inputs = _to_inputs(batch, device)
        # Use a managed context to safely handle resources used during output computation.
        with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
            # Set `out` for subsequent steps so gradient updates improve future predictions.
            out = model(inputs)
            # Update `cls_loss` with a loss term that drives backpropagation and output improvement.
            cls_loss = criterion(out["logit_final"], labels)
            # Update `gate_aux` with a loss term that drives backpropagation and output improvement.
            gate_aux = cls_loss.new_tensor(0.0)
            # Branch on `float(gate_aux_weight) > 0.0 and out.get("frame_e...` to choose the correct output computation path.
            if float(gate_aux_weight) > 0.0 and out.get("frame_event_scores") is not None:
                # Dense supervision for frame gates to mitigate hard top-k index non-differentiability.
                # Update `gate_aux` with a loss term that drives backpropagation and output improvement.
                gate_aux = event_gate_bag_loss(
                    frame_event_scores=out["frame_event_scores"],
                    target=labels,
                    frame_valid_mask=out.get("frame_valid_mask"),
                )
            # Update `loss` with a loss term that drives backpropagation and output improvement.
            loss = cls_loss + float(gate_aux_weight) * gate_aux
        # Reset gradients before next step to avoid mixing gradient signals across batches.
        optimizer.zero_grad(set_to_none=True)
        # Backpropagate current loss so gradients can update model output behavior.
        scaler.scale(loss).backward()
        # Call `scaler.unscale_` and use its result in later steps so gradient updates improve future predictions.
        scaler.unscale_(optimizer)
        # Call `torch.nn.utils.clip_grad_norm_` and use its result in later steps so gradient updates improve future predictions.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
        # Apply optimizer step so future predictions reflect this batch's gradients.
        scaler.step(optimizer)
        # Call `scaler.update` and use its result in later steps so gradient updates improve future predictions.
        scaler.update()
        # Call `float` and use its result in later steps so gradient updates improve future predictions.
        total += float(loss.item())
        # Execute this statement so gradient updates improve future predictions.
        n += 1
    # Return `total / max(n, 1)` as this function's contribution to downstream output flow.
    return total / max(n, 1)


# Execute this statement so gradient updates improve future predictions.
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Computes validation metrics used to judge model quality and influence training decisions."""
    # Call `model.eval` and use its result in later steps so gradient updates improve future predictions.
    model.eval()
    # Set `total` for subsequent steps so gradient updates improve future predictions.
    total = 0.0
    # Set `n` for subsequent steps so gradient updates improve future predictions.
    n = 0
    # Store raw score tensor in `logits` before probability/decision conversion.
    logits = []
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = []
    # Set `qualities` for subsequent steps so gradient updates improve future predictions.
    qualities = []

    # Iterate over `loader` so each item contributes to final outputs/metrics.
    for batch in loader:
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = batch["label"].to(device, non_blocking=True)
        # Compute `x` as an intermediate representation used by later output layers.
        x = _to_inputs(batch, device)
        # Use a managed context to safely handle resources used during output computation.
        with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
            # Set `out` for subsequent steps so gradient updates improve future predictions.
            out = model(x)
            # Update `loss` with a loss term that drives backpropagation and output improvement.
            loss = criterion(out["logit_final"], y)
            # Store raw score tensor in `logit` before probability/decision conversion.
            logit = out["logit_final"]
        # Call `float` and use its result in later steps so gradient updates improve future predictions.
        total += float(loss.item())
        # Execute this statement so gradient updates improve future predictions.
        n += 1
        # Call `logits.append` and use its result in later steps so gradient updates improve future predictions.
        logits.append(logit.detach().cpu().numpy())
        # Call `labels.append` and use its result in later steps so gradient updates improve future predictions.
        labels.append(y.detach().cpu().numpy())
        # Call `qualities.append` and use its result in later steps so gradient updates improve future predictions.
        qualities.append(_quality_score_from_batch(batch).detach().cpu().numpy())

    # Branch on `logits` to choose the correct output computation path.
    if logits:
        # Store raw score tensor in `logits` before probability/decision conversion.
        logits = np.concatenate(logits).astype(float)
        # Set `labels` for subsequent steps so gradient updates improve future predictions.
        labels = np.concatenate(labels).astype(int)
        # Set `qualities` for subsequent steps so gradient updates improve future predictions.
        qualities = np.concatenate(qualities).astype(float)
        # Store raw score tensor in `probs` before probability/decision conversion.
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    else:
        # Store raw score tensor in `logits` before probability/decision conversion.
        logits = np.array([])
        # Set `labels` for subsequent steps so gradient updates improve future predictions.
        labels = np.array([])
        # Set `qualities` for subsequent steps so gradient updates improve future predictions.
        qualities = np.array([])
        # Compute `probs` as confidence values used in final prediction decisions.
        probs = np.array([])

    # Return `{` as this function's contribution to downstream output flow.
    return {
        "loss": total / max(n, 1),
        "logits": logits,
        "labels": labels,
        "probs": probs,
        "quality": qualities,
    }


# Define a reusable pipeline function whose outputs feed later steps.
def summarize_metrics(labels, probs, spec_target=0.90):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `labels.size == 0` to choose the correct output computation path.
    if labels.size == 0:
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "auc": 0.5,
            "ece": 1.0,
            "calibration_quality": 0.0,
            "sens_at_90_spec": 0.0,
            "f1_opt": 0.0,
            "acc_opt": 0.0,
            "opt_threshold": 0.5,
        }

    # Record `auc` as a metric describing current output quality.
    auc = compute_auc(labels, probs)
    # Record `ece` as a metric describing current output quality.
    ece = compute_ece(labels, probs, n_bins=10)
    # Set `cal_quality` for subsequent steps so gradient updates improve future predictions.
    cal_quality = float(np.clip(1.0 - ece, 0.0, 1.0))
    # Record `sens` as a metric describing current output quality.
    sens = sensitivity_at_specificity(
        labels, probs, target_spec=float(spec_target), min_negatives=10, allow_unstable=True
    )
    # Branch on `not np.isfinite(sens)` to choose the correct output computation path.
    if not np.isfinite(sens):
        # Record `sens` as a metric describing current output quality.
        sens = 0.0
    # Compute `thr` as an intermediate representation used by later output layers.
    thr = find_optimal_threshold(labels, probs)
    # Set `basic` for subsequent steps so gradient updates improve future predictions.
    basic = compute_basic_metrics(labels, probs, threshold=thr)
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "auc": float(auc),
        "ece": float(ece),
        "calibration_quality": cal_quality,
        "sens_at_90_spec": float(max(0.0, sens)),
        "f1_opt": float(basic["f1"]),
        "acc_opt": float(basic["accuracy"]),
        "opt_threshold": float(thr),
    }


# Define a reusable pipeline function whose outputs feed later steps.
def _selection_score(metrics):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Return `(` as this function's contribution to downstream output flow.
    return (
        0.40 * float(metrics.get("auc", 0.0))
        + 0.25 * float(metrics.get("sens_at_90_spec", 0.0))
        + 0.15 * float(metrics.get("calibration_quality", 0.0))
        + 0.10 * float(metrics.get("f1_opt", 0.0))
        + 0.10 * float(metrics.get("acc_opt", 0.0))
    )


# Define a reusable pipeline function whose outputs feed later steps.
def _build_model_from_cfg(cfg):
    """Constructs components whose structure controls later training or inference outputs."""
    # Set `model_cfg` for subsequent steps so gradient updates improve future predictions.
    model_cfg = cfg.get("model", {})
    # Compute `thresholds` as an intermediate representation used by later output layers.
    thresholds = cfg.get("thresholds", {})
    # Return `ASDPipeline(` as this function's contribution to downstream output flow.
    return ASDPipeline(
        K_max=int(model_cfg.get("K_max", 16)),
        d_model=int(model_cfg.get("d_model", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        theta_high=float(thresholds.get("decision_high", 0.7)),
        theta_low=float(thresholds.get("decision_low", 0.3)),
    )


# Define a training routine that updates parameters and changes future outputs.
def _load_pretrained_motion_encoder(model, checkpoint_path, device):
    """Executes a training step/loop that updates parameters and directly changes model output behavior."""
    # Branch on `not checkpoint_path` to choose the correct output computation path.
    if not checkpoint_path:
        # Return `False` as this function's contribution to downstream output flow.
        return False
    # Branch on `not os.path.exists(checkpoint_path)` to choose the correct output computation path.
    if not os.path.exists(checkpoint_path):
        # Log runtime values to verify that output computation is behaving as expected.
        print(f"[Train] pretrained motion encoder checkpoint not found: {checkpoint_path}")
        # Return `False` as this function's contribution to downstream output flow.
        return False

    # Capture `ckpt` as model state controlling reproducible output behavior.
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Capture `state` as model state controlling reproducible output behavior.
    state = None
    # Branch on `isinstance(ckpt, dict) and "motion_encoder" in ckpt` to choose the correct output computation path.
    if isinstance(ckpt, dict) and "motion_encoder" in ckpt:
        # Capture `state` as model state controlling reproducible output behavior.
        state = ckpt["motion_encoder"]
    # Use alternate condition `isinstance(ckpt, dict) and "model_state" in ckpt` to refine output path selection.
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        # Set `ms` for subsequent steps so gradient updates improve future predictions.
        ms = ckpt["model_state"]
        # Capture `state` as model state controlling reproducible output behavior.
        state = {}
        # Iterate over `ms.items()` so each item contributes to final outputs/metrics.
        for k, v in ms.items():
            # Branch on `str(k).startswith("motion_encoder.")` to choose the correct output computation path.
            if str(k).startswith("motion_encoder."):
                # Call `str` and use its result in later steps so gradient updates improve future predictions.
                state[str(k).replace("motion_encoder.", "", 1)] = v
    # Branch on `not state` to choose the correct output computation path.
    if not state:
        # Log runtime values to verify that output computation is behaving as expected.
        print("[Train] checkpoint does not contain motion encoder weights.")
        # Return `False` as this function's contribution to downstream output flow.
        return False

    # Compute `missing, unexpected` as an intermediate representation used by later output layers.
    missing, unexpected = model.motion_encoder.load_state_dict(state, strict=False)
    # Log runtime values to verify that output computation is behaving as expected.
    print(
        f"[Train] loaded motion encoder init from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    # Return `True` as this function's contribution to downstream output flow.
    return True


# Define a training routine that updates parameters and changes future outputs.
def train(cfg, status_file=None):
    """Executes a training step/loop that updates parameters and directly changes model output behavior."""
    # Set `seed` for subsequent steps so gradient updates improve future predictions.
    seed = int(cfg.get("seed", 42))
    # Set `generator` for subsequent steps so gradient updates improve future predictions.
    generator = seed_everything(seed, deterministic=False)
    # Set `device_name` to the execution device used for this computation path.
    device_name = cfg.get("device", "auto")
    # Branch on `device_name == "auto"` to choose the correct output computation path.
    if device_name == "auto":
        # Set `device` to the execution device used for this computation path.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Set `device` to the execution device used for this computation path.
        device = torch.device(device_name)
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[Train] device={device}")
    # Call `_auto_batch_and_workers` and use its result in later steps so gradient updates improve future predictions.
    _auto_batch_and_workers(cfg, device)

    # Set `results_dir` for subsequent steps so gradient updates improve future predictions.
    results_dir = cfg.get("reporting", {}).get("results_dir", "results")
    # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
    os.makedirs(results_dir, exist_ok=True)
    # Capture `ckpt_mgr` as model state controlling reproducible output behavior.
    ckpt_mgr = CheckpointManager(root_dir=results_dir)
    # Set `run_tag` for subsequent steps so gradient updates improve future predictions.
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    # Compute `log_path` as an intermediate representation used by later output layers.
    log_path = os.path.join(results_dir, f"experiment_log_{run_tag}.jsonl")
    # Set `logger` for subsequent steps so gradient updates improve future predictions.
    logger = ExperimentLogger(log_path)
    # Log runtime values to verify that output computation is behaving as expected.
    logger.log("train_start", config=cfg)
    # Set `started` for subsequent steps so gradient updates improve future predictions.
    started = time.time()

    # Define a reusable pipeline function whose outputs feed later steps.
    def emit_status(phase, **fields):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `payload` for subsequent steps so gradient updates improve future predictions.
        payload = {
            "phase": phase,
            "time": int(time.time()),
            "elapsed_sec": round(time.time() - started, 1),
        }
        # Call `payload.update` and use its result in later steps so gradient updates improve future predictions.
        payload.update(fields)
        # Call `_write_status_file` and use its result in later steps so gradient updates improve future predictions.
        _write_status_file(status_file, payload)

    # Set `data_cfg` for subsequent steps so gradient updates improve future predictions.
    data_cfg = cfg.get("data", {})
    # Set `train_cfg` for subsequent steps so gradient updates improve future predictions.
    train_cfg = cfg.get("training", {})
    # Set `nas_cfg` for subsequent steps so gradient updates improve future predictions.
    nas_cfg = cfg.get("nas", {})
    # Compute `gate_aux_weight` as an intermediate representation used by later output layers.
    gate_aux_weight = float(train_cfg.get("event_gate_aux_weight", 0.1))

    # Stage 1-4: optional precompute numeric landmarks.
    # Branch on `bool(data_cfg.get("preprocess_videos", False))` to choose the correct output computation path.
    if bool(data_cfg.get("preprocess_videos", False)):
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("preprocess", state="starting")
        # Call `precompute_videos` and use its result in later steps so gradient updates improve future predictions.
        precompute_videos(
            csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
            processed_root=str(data_cfg.get("processed_root", "data/processed")),
            frame_stride=int(data_cfg.get("frame_stride", 1)),
            max_frames=int(data_cfg.get("max_frames", 0)),
            overwrite=bool(data_cfg.get("preprocess_overwrite", False)),
            progress_every=int(data_cfg.get("precompute_progress_every", 10)),
            status_callback=lambda s: emit_status("preprocess", state="running", **s),
        )
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("preprocess", state="done")

    # Build datasets.
    # Set `ds_train_view` for subsequent steps so gradient updates improve future predictions.
    ds_train_view = _build_dataset(cfg, is_training=True)
    # Set `ds_eval_view` for subsequent steps so gradient updates improve future predictions.
    ds_eval_view = _build_dataset(cfg, is_training=False)
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = np.asarray([float(e["label"]) for e in ds_train_view.entries], dtype=np.float32)
    # Set `groups` for subsequent steps so gradient updates improve future predictions.
    groups = np.asarray([e["subject_id"] for e in ds_train_view.entries], dtype=object)
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[Train] samples={len(labels)} positives={int(labels.sum())} negatives={int((labels==0).sum())}")
    # Log runtime values to verify that output computation is behaving as expected.
    logger.log(
        "dataset_summary",
        samples=int(len(labels)),
        positives=int(labels.sum()),
        negatives=int((labels == 0).sum()),
    )

    # Optional init from existing motion-encoder checkpoint.
    # Set `base_model` for subsequent steps so gradient updates improve future predictions.
    base_model = _build_model_from_cfg(cfg).to(device)
    # Capture `pretrained_encoder_ckpt` as model state controlling reproducible output behavior.
    pretrained_encoder_ckpt = str(train_cfg.get("pretrained_motion_encoder_checkpoint", "")).strip()
    # Call `_load_pretrained_motion_encoder` and use its result in later steps so gradient updates improve future predictions.
    _load_pretrained_motion_encoder(base_model, pretrained_encoder_ckpt, device)

    # Stage 6: self-supervised motion pretraining.
    # Compute `ssl_epochs` as an intermediate representation used by later output layers.
    ssl_epochs = int(train_cfg.get("ssl_pretrain_epochs", 0))
    # Branch on `ssl_epochs > 0` to choose the correct output computation path.
    if ssl_epochs > 0:
        # Set `ssl_loader` for subsequent steps so gradient updates improve future predictions.
        ssl_loader = _build_loader(ds_train_view, cfg, shuffle=True, generator=generator)
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("ssl_pretrain", state="starting", epochs=ssl_epochs)
        # Call `pretrain_motion_encoder` and use its result in later steps so gradient updates improve future predictions.
        pretrain_motion_encoder(
            base_model,
            ssl_loader,
            device=device,
            epochs=ssl_epochs,
            lr=float(train_cfg.get("ssl_lr", 1e-4)),
            max_steps_per_epoch=int(train_cfg.get("ssl_steps_per_epoch", 300)),
            logger=logger,
        )
        # Call `ckpt_mgr.save_model` and use its result in later steps so gradient updates improve future predictions.
        ckpt_mgr.save_model(
            "motion_encoder_pretrained.pth",
            {"motion_encoder": base_model.motion_encoder.state_dict(), "config": cfg},
        )
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("ssl_pretrain", state="done")

    # Stage 8: micro-genetic NAS.
    # Compute `best_arch` as an intermediate representation used by later output layers.
    best_arch = None
    # Branch on `bool(nas_cfg.get("enabled", True))` to choose the correct output computation path.
    if bool(nas_cfg.get("enabled", True)):
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("nas", state="starting")
        # Log runtime values to verify that output computation is behaving as expected.
        logger.log("nas_start")
        # Set `nas_result` for subsequent steps so gradient updates improve future predictions.
        nas_result = run_micro_genetic_nas(
            cfg=cfg,
            dataset=ds_train_view,
            labels=labels,
            groups=groups,
            device=device,
            logger=logger,
        )
        # Compute `best_arch` as an intermediate representation used by later output layers.
        best_arch = nas_result["best_architecture"]
        # Call `ckpt_mgr.save_json` and use its result in later steps so gradient updates improve future predictions.
        ckpt_mgr.save_json("nas_architecture.json", nas_result)
        # Log runtime values to verify that output computation is behaving as expected.
        logger.log("nas_final", **nas_result)
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status(
            "nas",
            state="done",
            best_fitness=float(nas_result["best_fitness"]),
            best_metrics=nas_result["best_metrics"],
        )
    else:
        # Compute `best_arch` as an intermediate representation used by later output layers.
        best_arch = None
        # Log runtime values to verify that output computation is behaving as expected.
        logger.log("nas_skipped")

    # Stage 9-11: grouped CV supervised training and fine-tuning.
    # Set `cv_folds` for subsequent steps so gradient updates improve future predictions.
    cv_folds = int(train_cfg.get("cv_folds", 5))
    # Set `folds` for subsequent steps so gradient updates improve future predictions.
    folds = make_group_kfold(labels.astype(int), groups, n_splits=cv_folds, seed=seed)
    # Set `fold_summaries` for subsequent steps so gradient updates improve future predictions.
    fold_summaries = []
    # Iterate over `enumerate(folds, start=1)` so each item contributes to final outputs/metrics.
    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("train_fold", state="starting", fold=fold_idx, total_folds=len(folds))
        # Call `check_group_overlap` and use its result in later steps so gradient updates improve future predictions.
        check_group_overlap(groups[tr_idx], groups[va_idx], fold_tag=f"[Fold {fold_idx}]")

        # Set `tr_loader` for subsequent steps so gradient updates improve future predictions.
        tr_loader = _build_loader(Subset(ds_train_view, tr_idx.tolist()), cfg, shuffle=True, generator=generator)
        # Set `va_loader` for subsequent steps so gradient updates improve future predictions.
        va_loader = _build_loader(Subset(ds_eval_view, va_idx.tolist()), cfg, shuffle=False, generator=None)

        # Set `model` for subsequent steps so gradient updates improve future predictions.
        model = _build_model_from_cfg(cfg).to(device)
        # Branch on `best_arch is not None` to choose the correct output computation path.
        if best_arch is not None:
            # Call `model.apply_nas_architecture` and use its result in later steps so gradient updates improve future predictions.
            model.apply_nas_architecture(best_arch)
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Load trained weights that directly determine inference outputs.
            model.motion_encoder.load_state_dict(base_model.motion_encoder.state_dict(), strict=True)
        # Handle exceptions and keep output behavior controlled under error conditions.
        except Exception:
            # No-op placeholder that keeps control-flow structure intact.
            pass

        # Stage 7: freeze backbone for stable NAS-supervised training.
        # Call `model.freeze_motion_encoder` and use its result in later steps so gradient updates improve future predictions.
        model.freeze_motion_encoder()
        # Update `criterion` with a loss term that drives backpropagation and output improvement.
        criterion = WeightedBCELoss(
            pos_weight=WeightedBCELoss.compute_from_labels(labels[tr_idx]),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
            brier_weight=float(train_cfg.get("brier_weight", 0.0)),
        )
        # Initialize `optimizer` to control parameter updates during training.
        optimizer = torch.optim.AdamW(
            model.model_parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        # Set `scaler` for subsequent steps so gradient updates improve future predictions.
        scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

        # Compute `epochs` as an intermediate representation used by later output layers.
        epochs = int(train_cfg.get("epochs", 20))
        # Set `patience` for subsequent steps so gradient updates improve future predictions.
        patience = int(train_cfg.get("patience", 8))
        # Set `best_score` for subsequent steps so gradient updates improve future predictions.
        best_score = -1e9
        # Set `best_payload` for subsequent steps so gradient updates improve future predictions.
        best_payload = None
        # Compute `bad_epochs` as an intermediate representation used by later output layers.
        bad_epochs = 0

        # Iterate over `range(1, epochs + 1)` so each item contributes to final outputs/metrics.
        for epoch in range(1, epochs + 1):
            # Update `tr_loss` with a loss term that drives backpropagation and output improvement.
            tr_loss = train_one_epoch(
                model,
                tr_loader,
                criterion,
                optimizer,
                scaler,
                device,
                clip_grad=float(train_cfg.get("clip_grad", 1.0)),
                gate_aux_weight=gate_aux_weight,
            )
            # Set `ev` for subsequent steps so gradient updates improve future predictions.
            ev = evaluate(model, va_loader, criterion, device)
            # Set `m` for subsequent steps so gradient updates improve future predictions.
            m = summarize_metrics(ev["labels"], ev["probs"], spec_target=0.90)
            # Set `score` for subsequent steps so gradient updates improve future predictions.
            score = _selection_score(m)
            # Log runtime values to verify that output computation is behaving as expected.
            logger.log(
                "train_fold_epoch",
                fold=fold_idx,
                epoch=epoch,
                train_loss=float(tr_loss),
                val_loss=float(ev["loss"]),
                metrics=m,
                score=float(score),
            )
            # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
            emit_status(
                "train_fold",
                state="epoch",
                fold=fold_idx,
                total_folds=len(folds),
                epoch=epoch,
                total_epochs=epochs,
                train_loss=float(tr_loss),
                val_loss=float(ev["loss"]),
                score=float(score),
                metrics=m,
            )
            # Log runtime values to verify that output computation is behaving as expected.
            print(
                f"[Fold {fold_idx}] epoch={epoch}/{epochs} "
                f"train_loss={tr_loss:.4f} val_loss={ev['loss']:.4f} "
                f"auc={m['auc']:.4f} sens90={m['sens_at_90_spec']:.4f} score={score:.4f}"
            )
            # Branch on `score > best_score` to choose the correct output computation path.
            if score > best_score:
                # Set `best_score` for subsequent steps so gradient updates improve future predictions.
                best_score = score
                # Compute `bad_epochs` as an intermediate representation used by later output layers.
                bad_epochs = 0
                # Set `best_payload` for subsequent steps so gradient updates improve future predictions.
                best_payload = {
                    "model_state": model.state_dict(),
                    "best_metrics": m,
                }
            else:
                # Execute this statement so gradient updates improve future predictions.
                bad_epochs += 1
            # Branch on `bad_epochs >= patience` to choose the correct output computation path.
            if bad_epochs >= patience:
                # Stop iteration early to prevent further changes to the current output state.
                break

        # Branch on `best_payload is not None` to choose the correct output computation path.
        if best_payload is not None:
            # Load trained weights that directly determine inference outputs.
            model.load_state_dict(best_payload["model_state"])

        # Stage 11: joint fine-tuning by unfreezing upper encoder layers.
        # Compute `finetune_epochs` as an intermediate representation used by later output layers.
        finetune_epochs = int(train_cfg.get("finetune_epochs", 4))
        # Branch on `finetune_epochs > 0` to choose the correct output computation path.
        if finetune_epochs > 0:
            # Call `model.unfreeze_upper_motion_layers` and use its result in later steps so gradient updates improve future predictions.
            model.unfreeze_upper_motion_layers(num_blocks=int(train_cfg.get("finetune_unfreeze_blocks", 1)))
            # Initialize `ft_optimizer` to control parameter updates during training.
            ft_optimizer = torch.optim.AdamW(
                model.trainable_parameters(),
                lr=float(train_cfg.get("finetune_lr", 2e-5)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            )
            # Iterate over `range(1, finetune_epochs + 1)` so each item contributes to final outputs/metrics.
            for ft_epoch in range(1, finetune_epochs + 1):
                # Update `ft_loss` with a loss term that drives backpropagation and output improvement.
                ft_loss = train_one_epoch(
                    model,
                    tr_loader,
                    criterion,
                    ft_optimizer,
                    scaler,
                    device,
                    clip_grad=float(train_cfg.get("clip_grad", 1.0)),
                    gate_aux_weight=gate_aux_weight,
                )
                # Log runtime values to verify that output computation is behaving as expected.
                logger.log("finetune_fold_epoch", fold=fold_idx, epoch=ft_epoch, train_loss=float(ft_loss))

        # Set `ev` for subsequent steps so gradient updates improve future predictions.
        ev = evaluate(model, va_loader, criterion, device)
        # Set `temp` for subsequent steps so gradient updates improve future predictions.
        temp = fit_temperature(
            torch.tensor(ev["logits"], device=device, dtype=torch.float32),
            torch.tensor(ev["labels"], device=device, dtype=torch.float32),
            device=device,
        )
        # Store raw score tensor in `logits_cal` before probability/decision conversion.
        logits_cal = apply_temperature(torch.tensor(ev["logits"], dtype=torch.float32), temp).cpu().numpy()
        # Store raw score tensor in `probs_cal` before probability/decision conversion.
        probs_cal = 1.0 / (1.0 + np.exp(-np.clip(logits_cal, -40.0, 40.0)))
        # Record `metrics_cal` as a metric describing current output quality.
        metrics_cal = summarize_metrics(ev["labels"], probs_cal, spec_target=0.90)
        # Call `fold_summaries.append` and use its result in later steps so gradient updates improve future predictions.
        fold_summaries.append(metrics_cal)

        # Call `ckpt_mgr.save_model` and use its result in later steps so gradient updates improve future predictions.
        ckpt_mgr.save_model(
            f"asd_best_fold{fold_idx}.pth",
            {
                "model_state": model.state_dict(),
                "nas_architecture": best_arch,
                "temperature": float(temp),
                "config": cfg,
                "fold_metrics": metrics_cal,
            },
        )
        # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
        emit_status("train_fold", state="done", fold=fold_idx, total_folds=len(folds), metrics=metrics_cal)

    # Cross-val summary.
    # Record `aucs` as a metric describing current output quality.
    aucs = [m["auc"] for m in fold_summaries] if fold_summaries else [0.5]
    # Record `sens` as a metric describing current output quality.
    sens = [m["sens_at_90_spec"] for m in fold_summaries] if fold_summaries else [0.0]
    # Set `cv_summary` for subsequent steps so gradient updates improve future predictions.
    cv_summary = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "sens90_mean": float(np.mean(sens)),
        "sens90_std": float(np.std(sens)),
        "cv_stability": float(np.clip(1.0 - np.std(aucs), 0.0, 1.0)),
    }
    # Call `ckpt_mgr.save_json` and use its result in later steps so gradient updates improve future predictions.
    ckpt_mgr.save_json("cv_summary.json", cv_summary)
    # Log runtime values to verify that output computation is behaving as expected.
    logger.log("cv_summary", **cv_summary)

    # Final model training on full set with grouped holdout for calibration.
    # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
    emit_status("final_train", state="starting")
    # Compute `tr_idx, va_idx, _` as an intermediate representation used by later output layers.
    tr_idx, va_idx, _ = make_group_stratified_split(
        labels.astype(int),
        groups,
        val_fraction=float(train_cfg.get("final_val_fraction", 0.2)),
        seed=seed,
    )
    # Set `tr_loader` for subsequent steps so gradient updates improve future predictions.
    tr_loader = _build_loader(Subset(ds_train_view, tr_idx.tolist()), cfg, shuffle=True, generator=generator)
    # Set `va_loader` for subsequent steps so gradient updates improve future predictions.
    va_loader = _build_loader(Subset(ds_eval_view, va_idx.tolist()), cfg, shuffle=False, generator=None)

    # Set `model` for subsequent steps so gradient updates improve future predictions.
    model = _build_model_from_cfg(cfg).to(device)
    # Branch on `best_arch is not None` to choose the correct output computation path.
    if best_arch is not None:
        # Call `model.apply_nas_architecture` and use its result in later steps so gradient updates improve future predictions.
        model.apply_nas_architecture(best_arch)
    # Start guarded block so failures can be handled without breaking output flow.
    try:
        # Load trained weights that directly determine inference outputs.
        model.motion_encoder.load_state_dict(base_model.motion_encoder.state_dict(), strict=True)
    # Handle exceptions and keep output behavior controlled under error conditions.
    except Exception:
        # No-op placeholder that keeps control-flow structure intact.
        pass
    # Call `model.freeze_motion_encoder` and use its result in later steps so gradient updates improve future predictions.
    model.freeze_motion_encoder()

    # Update `criterion` with a loss term that drives backpropagation and output improvement.
    criterion = WeightedBCELoss(
        pos_weight=WeightedBCELoss.compute_from_labels(labels[tr_idx]),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        brier_weight=float(train_cfg.get("brier_weight", 0.0)),
    )
    # Initialize `optimizer` to control parameter updates during training.
    optimizer = torch.optim.AdamW(
        model.model_parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    # Set `scaler` for subsequent steps so gradient updates improve future predictions.
    scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

    # Compute `final_epochs` as an intermediate representation used by later output layers.
    final_epochs = int(train_cfg.get("final_epochs", 24))
    # Iterate over `range(1, final_epochs + 1)` so each item contributes to final outputs/metrics.
    for epoch in range(1, final_epochs + 1):
        # Update `loss` with a loss term that drives backpropagation and output improvement.
        loss = train_one_epoch(
            model,
            tr_loader,
            criterion,
            optimizer,
            scaler,
            device,
            clip_grad=float(train_cfg.get("clip_grad", 1.0)),
            gate_aux_weight=gate_aux_weight,
        )
        # Log runtime values to verify that output computation is behaving as expected.
        logger.log("final_train_epoch", epoch=epoch, train_loss=float(loss))
        # Branch on `epoch % 5 == 0 or epoch == final_epochs` to choose the correct output computation path.
        if epoch % 5 == 0 or epoch == final_epochs:
            # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
            emit_status("final_train", state="epoch", epoch=epoch, total_epochs=final_epochs, train_loss=float(loss))

    # Final gentle joint fine-tuning.
    # Call `model.unfreeze_upper_motion_layers` and use its result in later steps so gradient updates improve future predictions.
    model.unfreeze_upper_motion_layers(num_blocks=int(train_cfg.get("finetune_unfreeze_blocks", 1)))
    # Compute `ft_epochs` as an intermediate representation used by later output layers.
    ft_epochs = int(train_cfg.get("final_finetune_epochs", 4))
    # Branch on `ft_epochs > 0` to choose the correct output computation path.
    if ft_epochs > 0:
        # Set `ft_opt` for subsequent steps so gradient updates improve future predictions.
        ft_opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=float(train_cfg.get("finetune_lr", 2e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        # Iterate over `range(1, ft_epochs + 1)` so each item contributes to final outputs/metrics.
        for epoch in range(1, ft_epochs + 1):
            # Update `loss` with a loss term that drives backpropagation and output improvement.
            loss = train_one_epoch(
                model,
                tr_loader,
                criterion,
                ft_opt,
                scaler,
                device,
                clip_grad=float(train_cfg.get("clip_grad", 1.0)),
                gate_aux_weight=gate_aux_weight,
            )
            # Log runtime values to verify that output computation is behaving as expected.
            logger.log("final_finetune_epoch", epoch=epoch, train_loss=float(loss))

    # Set `ev` for subsequent steps so gradient updates improve future predictions.
    ev = evaluate(model, va_loader, criterion, device)
    # Set `temp` for subsequent steps so gradient updates improve future predictions.
    temp = fit_temperature(
        torch.tensor(ev["logits"], device=device, dtype=torch.float32),
        torch.tensor(ev["labels"], device=device, dtype=torch.float32),
        device=device,
    )
    # Store raw score tensor in `logits_cal` before probability/decision conversion.
    logits_cal = apply_temperature(torch.tensor(ev["logits"], dtype=torch.float32), temp).cpu().numpy()
    # Store raw score tensor in `probs_cal` before probability/decision conversion.
    probs_cal = 1.0 / (1.0 + np.exp(-np.clip(logits_cal, -40.0, 40.0)))
    # Record `final_metrics` as a metric describing current output quality.
    final_metrics = summarize_metrics(ev["labels"], probs_cal, spec_target=0.90)
    # Call `float` and use its result in later steps so gradient updates improve future predictions.
    final_metrics["temperature"] = float(temp)

    # Compute `final_path` as an intermediate representation used by later output layers.
    final_path = ckpt_mgr.save_model(
        "asd_pipeline_model.pth",
        {
            "model_state": model.state_dict(),
            "nas_architecture": best_arch,
            "temperature": float(temp),
            "config": cfg,
            "final_metrics": final_metrics,
            "cv_summary": cv_summary,
        },
    )
    # Call `ckpt_mgr.save_json` and use its result in later steps so gradient updates improve future predictions.
    ckpt_mgr.save_json("final_metrics.json", final_metrics)
    # Log runtime values to verify that output computation is behaving as expected.
    logger.log("final_metrics", **final_metrics)
    # Call `emit_status` and use its result in later steps so gradient updates improve future predictions.
    emit_status("done", state="completed", final_model=final_path, final_metrics=final_metrics)
    # Log runtime values to verify that output computation is behaving as expected.
    logger.log("train_done", final_model=final_path, final_metrics=final_metrics)

    # Compute `pdf_report_path` as an intermediate representation used by later output layers.
    pdf_report_path = os.path.join(results_dir, f"training_log_report_{run_tag}.pdf")
    # Call `export_experiment_log_pdf` and use its result in later steps so gradient updates improve future predictions.
    export_experiment_log_pdf(
        log_jsonl_path=log_path,
        pdf_path=pdf_report_path,
        title="ASD Training Report",
        extra_summary={
            "final_model": final_path,
            "auc_mean": cv_summary.get("auc_mean"),
            "sens90_mean": cv_summary.get("sens90_mean"),
            "final_auc": final_metrics.get("auc"),
            "final_f1_opt": final_metrics.get("f1_opt"),
        },
    )
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[Train] final model saved: {final_path}")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[Train] log jsonl: {log_path}")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[Train] log report pdf: {pdf_report_path}")


# Define a reusable pipeline function whose outputs feed later steps.
def main():
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `parser` for subsequent steps so gradient updates improve future predictions.
    parser = argparse.ArgumentParser(description="Train ASD landmark-motion pipeline")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--config", type=str, default="config.yaml")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--override", type=str, action="append", default=[])
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--csv", type=str, default=None, help="Override data.csv_path")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--status-file", type=str, default=None, help="Optional status JSON path")
    # Set `args` for subsequent steps so gradient updates improve future predictions.
    args = parser.parse_args()

    # Set `cfg` for subsequent steps so gradient updates improve future predictions.
    cfg = load_config(args.config)
    # Branch on `args.csv` to choose the correct output computation path.
    if args.csv:
        # Call `args.override.append` and use its result in later steps so gradient updates improve future predictions.
        args.override.append(f"data.csv_path={args.csv}")
    # Set `cfg` for subsequent steps so gradient updates improve future predictions.
    cfg = apply_overrides(cfg, args.override)
    # Call `train` and use its result in later steps so gradient updates improve future predictions.
    train(cfg, status_file=args.status_file)


# Branch on `__name__ == "__main__"` to choose the correct output computation path.
if __name__ == "__main__":
    # Call `main` and use its result in later steps so gradient updates improve future predictions.
    main()
