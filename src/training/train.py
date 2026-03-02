# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Training script for ASD Motion pipeline.

Usage:
  python src/training/train.py --config config.yaml
  python src/training/train.py --config config.yaml --override training.epochs=10
"""

import os
import json
import argparse
import math
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.models.pipeline_model import ASDPipeline
from src.models.video.utils.device import get_device, configure_cuda_optimizations, print_gpu_info
from src.training.dataset import VideoDataset
from src.training.losses import (
    WeightedBCELoss,
    pairwise_auc_loss,
    sens_at_spec_surrogate,
)
from src.training.optim import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.callbacks import EarlyStopping
from src.training.explainability import extract_attention_maps, compute_temporal_importance
from src.training.report import generate_training_report

from src.pipeline.preprocess import precompute_videos
from src.utils.config import load_config, apply_overrides
from src.utils.seed import seed_everything, seed_worker
from src.utils.splits import make_group_kfold, make_group_stratified_split, check_group_overlap
from src.utils.metrics import (
    sigmoid,
    compute_basic_metrics,
    compute_auc,
    find_optimal_threshold,
    compute_ece,
    sensitivity_at_specificity,
    bootstrap_ci,
)
from src.utils.calibration import fit_temperature, apply_temperature
from src.utils.quality import compute_quality_score
from src.utils.decision import make_decision


def collate_fn(batch):
    # Compute `out` for the next processing step.
    out = {}
    # Compute `out['face_crops']` for the next processing step.
    out["face_crops"] = torch.stack([b["face_crops"] for b in batch])
    # Compute `out['pose_maps']` for the next processing step.
    out["pose_maps"] = torch.stack([b["pose_maps"] for b in batch])
    # Compute `out['motion_maps']` for the next processing step.
    out["motion_maps"] = torch.stack([b["motion_maps"] for b in batch])
    # Compute `out['hand_maps']` for the next processing step.
    out["hand_maps"] = out["motion_maps"]
    # Compute `out['mask']` for the next processing step.
    out["mask"] = torch.stack([b["mask"] for b in batch])
    # Compute `out['timestamps']` for the next processing step.
    out["timestamps"] = torch.stack([b["timestamps"] for b in batch])
    # Compute `out['delta_t']` for the next processing step.
    out["delta_t"] = torch.stack([b["delta_t"] for b in batch])
    # Compute `out['route_mask']` for the next processing step.
    out["route_mask"] = torch.stack([b["route_mask"] for b in batch])
    # Compute `out['label']` for the next processing step.
    out["label"] = torch.stack([b["label"] for b in batch])
    # Compute `out['qualities']` for the next processing step.
    out["qualities"] = {
        k: torch.stack([b["qualities"][k] for b in batch])
        for k in ("face_score", "pose_score", "hand_score")
    }
    # Compute `out['video_id']` for the next processing step.
    out["video_id"] = [b["video_id"] for b in batch]
    # Compute `out['subject_id']` for the next processing step.
    out["subject_id"] = [b["subject_id"] for b in batch]
    # Return the result expected by the caller.
    return out


def _build_inputs(batch, device):
    # Return the result expected by the caller.
    return {
        "face_crops": batch["face_crops"].to(device, non_blocking=True),
        "pose_maps": batch["pose_maps"].to(device, non_blocking=True),
        "motion_maps": batch["motion_maps"].to(device, non_blocking=True),
        "hand_maps": batch["hand_maps"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "timestamps": batch["timestamps"].to(device, non_blocking=True),
        "delta_t": batch["delta_t"].to(device, non_blocking=True),
        "route_mask": batch["route_mask"].to(device, non_blocking=True),
        "qualities": {k: v.to(device, non_blocking=True) for k, v in batch["qualities"].items()},
    }


def _build_train_loader(train_ds, train_labels, cfg, generator):
    # Compute `batch_size` for the next processing step.
    batch_size = cfg["training"]["batch_size"]
    # Compute `num_workers` for the next processing step.
    num_workers = cfg["data"]["num_workers"]
    # Compute `balance` for the next processing step.
    balance = bool(cfg.get("training", {}).get("balance_batches", True))

    # Compute `sampler` for the next processing step.
    sampler = None
    # Compute `shuffle` for the next processing step.
    shuffle = True
    # Compute `y` for the next processing step.
    y = np.asarray(train_labels, dtype=int)

    # Branch behavior based on the current runtime condition.
    if balance and y.size > 1 and np.unique(y).size > 1:
        # Compute `class_counts` for the next processing step.
        class_counts = np.bincount(y, minlength=2).astype(float)
        # Compute `class_counts[class_counts == 0]` for the next processing step.
        class_counts[class_counts == 0] = 1.0
        # Compute `class_weights` for the next processing step.
        class_weights = 1.0 / class_counts
        # Compute `sample_weights` for the next processing step.
        sample_weights = class_weights[y]
        # Compute `sampler` for the next processing step.
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        # Compute `shuffle` for the next processing step.
        shuffle = False

    # Return the result expected by the caller.
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _build_criterion(cfg, pos_weight):
    # Compute `tcfg` for the next processing step.
    tcfg = cfg.get("training", {})
    # Return the result expected by the caller.
    return WeightedBCELoss(
        pos_weight=pos_weight,
        label_smoothing=float(tcfg.get("label_smoothing", 0.03)),
        logit_clip=float(tcfg.get("loss_logit_clip", 10.0)),
        brier_weight=float(tcfg.get("brier_weight", 0.1)),
    )


def _safe_metric_value(value, default=0.0):
    # Guard this block and recover cleanly from expected failures.
    try:
        # Compute `val` for the next processing step.
        val = float(value)
    except (TypeError, ValueError):
        # Return the result expected by the caller.
        return float(default)
    # Branch behavior based on the current runtime condition.
    if not math.isfinite(val):
        # Return the result expected by the caller.
        return float(default)
    # Return the result expected by the caller.
    return val


def _selection_score(metrics, cfg, apply_penalty=True):
    # Compute `tcfg` for the next processing step.
    tcfg = cfg.get("training", {})
    # Compute `auc` for the next processing step.
    auc = _safe_metric_value(metrics.get("auc"), 0.0)
    # Compute `f1` for the next processing step.
    f1 = _safe_metric_value(metrics.get("f1_opt"), 0.0)
    # Compute `sens_spec` for the next processing step.
    sens_spec = _safe_metric_value(metrics.get("sens_spec"), 0.0)

    # Compute `w_auc` for the next processing step.
    w_auc = float(tcfg.get("score_w_auc", 0.45))
    # Compute `w_f1` for the next processing step.
    w_f1 = float(tcfg.get("score_w_f1", 0.35))
    # Compute `w_ss` for the next processing step.
    w_ss = float(tcfg.get("score_w_sens_spec", 0.20))
    # Compute `score` for the next processing step.
    score = w_auc * auc + w_f1 * f1 + w_ss * sens_spec

    # Branch behavior based on the current runtime condition.
    if not apply_penalty:
        # Return the result expected by the caller.
        return float(score)

    # Compute `target_auc` for the next processing step.
    target_auc = float(tcfg.get("target_auc", 0.90))
    # Compute `target_f1` for the next processing step.
    target_f1 = float(tcfg.get("target_f1", 0.90))
    # Compute `target_ss` for the next processing step.
    target_ss = float(tcfg.get("target_sens_spec", 0.80))
    # Compute `p_auc` for the next processing step.
    p_auc = float(tcfg.get("penalty_w_auc", 0.35))
    # Compute `p_f1` for the next processing step.
    p_f1 = float(tcfg.get("penalty_w_f1", 0.35))
    # Compute `p_ss` for the next processing step.
    p_ss = float(tcfg.get("penalty_w_sens_spec", 0.55))
    # Update `score` in place using the latest contribution.
    score -= p_auc * max(0.0, target_auc - auc)
    # Update `score` in place using the latest contribution.
    score -= p_f1 * max(0.0, target_f1 - f1)
    # Update `score` in place using the latest contribution.
    score -= p_ss * max(0.0, target_ss - sens_spec)
    # Return the result expected by the caller.
    return float(score)


def _train_objective_config(cfg, spec_target):
    # Compute `tcfg` for the next processing step.
    tcfg = cfg.get("training", {})
    # Compute `surrogate_spec_target` for the next processing step.
    surrogate_spec_target = float(tcfg.get("surrogate_spec_target", spec_target))
    # Return the result expected by the caller.
    return {
        "spec_target": surrogate_spec_target,
        "auc_rank_weight": float(tcfg.get("loss_auc_rank_weight", 0.20)),
        "sens_spec_weight": float(tcfg.get("loss_sens_spec_weight", 0.35)),
        "sens_spec_margin": float(tcfg.get("sens_spec_margin", 0.03)),
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    clip_grad=0.5,
    spec_target=0.95,
    auc_rank_weight=0.0,
    sens_spec_weight=0.0,
    sens_spec_margin=0.02,
):
    # Invoke `model.train` to advance this processing stage.
    model.train()
    # Compute `(total_loss, n_batches)` for the next processing step.
    total_loss, n_batches = 0.0, 0
    # Compute `device_type` for the next processing step.
    device_type = device.type if isinstance(device, torch.device) else str(device)
    # Compute `use_amp` for the next processing step.
    use_amp = device_type.startswith("cuda")

    # Iterate `batch` across `loader` to process each element.
    for batch in loader:
        # Compute `inputs` for the next processing step.
        inputs = _build_inputs(batch, device)
        # Compute `labels` for the next processing step.
        labels = batch["label"].to(device, non_blocking=True)

        # Invoke `optimizer.zero_grad` to advance this processing stage.
        optimizer.zero_grad(set_to_none=True)
        # Run this block with managed resources/context cleanup.
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            # Compute `out` for the next processing step.
            out = model(inputs)
            # Compute `logits` for the next processing step.
            logits = out["logit_final"]
            # Compute `loss` for the next processing step.
            loss = criterion(logits, labels)

            # Branch behavior based on the current runtime condition.
            if auc_rank_weight > 0 or sens_spec_weight > 0:
                # Compute `aux` for the next processing step.
                aux = logits.new_tensor(0.0)
                # Branch behavior based on the current runtime condition.
                if auc_rank_weight > 0:
                    # Compute `aux` for the next processing step.
                    aux = aux + float(auc_rank_weight) * pairwise_auc_loss(logits, labels, temperature=1.0)
                # Branch behavior based on the current runtime condition.
                if sens_spec_weight > 0:
                    # Compute `aux` for the next processing step.
                    aux = aux + float(sens_spec_weight) * sens_at_spec_surrogate(
                        logits,
                        labels,
                        target_spec=spec_target,
                        margin=sens_spec_margin,
                        detach_threshold=True,
                    )
                # Compute `loss` for the next processing step.
                loss = loss + aux

        # Compute `loss_val` for the next processing step.
        loss_val = loss.item()
        # Branch behavior based on the current runtime condition.
        if not math.isfinite(loss_val):
            continue

        # Invoke `scaler.scale(loss).backward` to advance this processing stage.
        scaler.scale(loss).backward()
        # Invoke `scaler.unscale_` to advance this processing stage.
        scaler.unscale_(optimizer)
        # Compute `grad_norm` for the next processing step.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        # Branch behavior based on the current runtime condition.
        if not math.isfinite(float(grad_norm)):
            # Invoke `optimizer.zero_grad` to advance this processing stage.
            optimizer.zero_grad(set_to_none=True)
            # Invoke `scaler.update` to advance this processing stage.
            scaler.update()
            continue
        # Invoke `scaler.step` to advance this processing stage.
        scaler.step(optimizer)
        # Invoke `scaler.update` to advance this processing stage.
        scaler.update()

        # Update `total_loss` in place using the latest contribution.
        total_loss += loss_val
        # Update `n_batches` in place using the latest contribution.
        n_batches += 1

    # Return the result expected by the caller.
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device):
    # Invoke `model.eval` to advance this processing stage.
    model.eval()
    # Compute `(total_loss, n_batches)` for the next processing step.
    total_loss, n_batches = 0.0, 0
    # Compute `device_type` for the next processing step.
    device_type = device.type if isinstance(device, torch.device) else str(device)
    # Compute `use_amp` for the next processing step.
    use_amp = device_type.startswith("cuda")

    # Iterate `batch` across `loader` to process each element.
    for batch in loader:
        # Compute `inputs` for the next processing step.
        inputs = _build_inputs(batch, device)
        # Compute `labels` for the next processing step.
        labels = batch["label"].to(device, non_blocking=True)
        # Run this block with managed resources/context cleanup.
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            # Compute `out` for the next processing step.
            out = model(inputs)
            # Compute `loss` for the next processing step.
            loss = criterion(out["logit_final"], labels)
        # Update `total_loss` in place using the latest contribution.
        total_loss += loss.item()
        # Update `n_batches` in place using the latest contribution.
        n_batches += 1
    # Return the result expected by the caller.
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def collect_predictions(model, loader, device,
                        pose_only_if_no_face=False,
                        face_presence_threshold=0.05):
    # Invoke `model.eval` to advance this processing stage.
    model.eval()
    # Compute `logits_all` for the next processing step.
    logits_all = []
    # Compute `labels_all` for the next processing step.
    labels_all = []
    # Compute `quality_all` for the next processing step.
    quality_all = []

    # Compute `device_type` for the next processing step.
    device_type = device.type if isinstance(device, torch.device) else str(device)
    # Compute `use_amp` for the next processing step.
    use_amp = device_type.startswith("cuda")

    # Iterate `batch` across `loader` to process each element.
    for batch in loader:
        # Compute `inputs` for the next processing step.
        inputs = _build_inputs(batch, device)
        # Compute `labels` for the next processing step.
        labels = batch["label"].to(device, non_blocking=True)

        # Run this block with managed resources/context cleanup.
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            # Compute `out` for the next processing step.
            out = model(inputs)
            # Compute `logits` for the next processing step.
            logits = out["logit_final"]

        # Invoke `logits_all.append` to advance this processing stage.
        logits_all.append(logits.detach().float().cpu())
        # Invoke `labels_all.append` to advance this processing stage.
        labels_all.append(labels.detach().float().cpu())

        # Compute `q` for the next processing step.
        q = compute_quality_score(
            batch["qualities"],
            batch["mask"],
            pose_only_if_no_face=pose_only_if_no_face,
            face_presence_threshold=face_presence_threshold,
        ).detach().cpu()
        # Invoke `quality_all.append` to advance this processing stage.
        quality_all.append(q)

    # Compute `logits_all` for the next processing step.
    logits_all = torch.cat(logits_all).numpy() if logits_all else np.array([])
    # Compute `labels_all` for the next processing step.
    labels_all = torch.cat(labels_all).numpy().astype(int) if labels_all else np.array([])
    # Compute `quality_all` for the next processing step.
    quality_all = torch.cat(quality_all).numpy() if quality_all else np.array([])

    # Compute `probs` for the next processing step.
    probs = sigmoid(logits_all) if len(logits_all) else np.array([])
    # Return the result expected by the caller.
    return logits_all, labels_all, probs, quality_all


def evaluate_metrics(labels, probs, spec_target=0.95, n_bins=10, min_negatives_for_sens_spec=20):
    # Compute `n_neg` for the next processing step.
    n_neg = int((np.asarray(labels) == 0).sum())
    # Compute `auc` for the next processing step.
    auc = compute_auc(labels, probs)
    # Compute `ece` for the next processing step.
    ece = compute_ece(labels, probs, n_bins=n_bins)
    # Compute `thr_opt` for the next processing step.
    thr_opt = find_optimal_threshold(labels, probs)

    # Compute `m05` for the next processing step.
    m05 = compute_basic_metrics(labels, probs, threshold=0.5)
    # Compute `mopt` for the next processing step.
    mopt = compute_basic_metrics(labels, probs, threshold=thr_opt)

    # Compute `sens_spec` for the next processing step.
    sens_spec = sensitivity_at_specificity(
        labels,
        probs,
        target_spec=spec_target,
        min_negatives=min_negatives_for_sens_spec,
        allow_unstable=True,
    )

    # Return the result expected by the caller.
    return {
        "auc": auc,
        "ece": ece,
        "opt_threshold": thr_opt,
        "acc_05": m05["accuracy"],
        "acc_opt": mopt["accuracy"],
        "f1_05": m05["f1"],
        "f1_opt": mopt["f1"],
        "precision_opt": mopt["precision"],
        "recall_opt": mopt["recall"],
        "confusion_matrix": mopt["confusion_matrix"],
        "sens_spec": sens_spec,
        "n_negatives": n_neg,
        "sens_spec_unstable": bool(n_neg < max(int(min_negatives_for_sens_spec), 1)),
    }


def compute_abstain_rate(probs_cal, quality_scores, cfg):
    # Compute `low_thr` for the next processing step.
    low_thr = cfg["thresholds"]["decision_low"]
    # Compute `high_thr` for the next processing step.
    high_thr = cfg["thresholds"]["decision_high"]
    # Compute `q_thr` for the next processing step.
    q_thr = cfg["thresholds"]["quality_threshold"]
    # Compute `abstain` for the next processing step.
    abstain = 0
    # Iterate `(p, q)` across `zip(probs_cal, quality_scores)` to process each element.
    for p, q in zip(probs_cal, quality_scores):
        # Compute `res` for the next processing step.
        res = make_decision(float(p), float(p), float(q), q_thr, low_thr, high_thr)
        # Branch behavior based on the current runtime condition.
        if res.abstained:
            # Update `abstain` in place using the latest contribution.
            abstain += 1
    # Return the result expected by the caller.
    return abstain / max(len(probs_cal), 1)


def mutate_config(cfg, rng):
    # Compute `new_cfg` for the next processing step.
    new_cfg = json.loads(json.dumps(cfg))
    # Randomly change one field
    choices = ["encoder_kernel", "n_heads", "num_encoder_layers", "dim_ff"]
    # Compute `field` for the next processing step.
    field = str(rng.choice(choices))
    # Branch behavior based on the current runtime condition.
    if field == "encoder_kernel":
        # Compute `new_cfg['encoder_kernel']` for the next processing step.
        new_cfg["encoder_kernel"] = int(rng.choice([3, 5, 7, 11]))
    # Branch behavior based on the current runtime condition.
    elif field == "n_heads":
        # Compute `new_cfg['transformer']['n_heads']` for the next processing step.
        new_cfg["transformer"]["n_heads"] = int(rng.choice([2, 4, 8]))
    # Branch behavior based on the current runtime condition.
    elif field == "num_encoder_layers":
        # Compute `new_cfg['transformer']['num_encod...` for the next processing step.
        new_cfg["transformer"]["num_encoder_layers"] = int(rng.choice([2, 3, 4]))
    # Branch behavior based on the current runtime condition.
    elif field == "dim_ff":
        # Compute `new_cfg['transformer']['dim_ff']` for the next processing step.
        new_cfg["transformer"]["dim_ff"] = int(rng.choice([512, 1024, 2048]))
    # Return the result expected by the caller.
    return new_cfg


def _sample_from_logits(rng, choices, logits=None, temperature=1.0):
    # Branch behavior based on the current runtime condition.
    if logits is None:
        # Return the result expected by the caller.
        return int(rng.choice(choices))
    # Compute `logits` for the next processing step.
    logits = np.array(logits, dtype=float) / max(temperature, 1e-6)
    # Compute `logits` for the next processing step.
    logits = logits - logits.max()
    # Compute `probs` for the next processing step.
    probs = np.exp(logits)
    # Compute `probs` for the next processing step.
    probs = probs / probs.sum()
    # Return the result expected by the caller.
    return int(rng.choice(choices, p=probs))


def random_config(rng, choice_logits=None, temperature=1.0):
    # Return the result expected by the caller.
    return {
        "transformer": {
            "n_heads": int(_sample_from_logits(rng, [2, 4, 8],
                                           logits=None if choice_logits is None else choice_logits["n_heads"],
                                           temperature=temperature)),
            "num_encoder_layers": int(_sample_from_logits(rng, [2, 3, 4],
                                                      logits=None if choice_logits is None else choice_logits["num_layers"],
                                                      temperature=temperature)),
            "dim_ff": int(_sample_from_logits(rng, [512, 1024, 2048],
                                          logits=None if choice_logits is None else choice_logits["ff_dim"],
                                          temperature=temperature)),
        },
        "encoder_kernel": int(_sample_from_logits(rng, [3, 5, 7, 11],
                                              logits=None if choice_logits is None else choice_logits["kernel"],
                                              temperature=temperature)),
    }


def _jsonable(obj):
    # Branch behavior based on the current runtime condition.
    if isinstance(obj, dict):
        # Return the result expected by the caller.
        return {k: _jsonable(v) for k, v in obj.items()}
    # Branch behavior based on the current runtime condition.
    if isinstance(obj, (list, tuple)):
        # Return the result expected by the caller.
        return [_jsonable(v) for v in obj]
    # Branch behavior based on the current runtime condition.
    if isinstance(obj, np.integer):
        # Return the result expected by the caller.
        return int(obj)
    # Branch behavior based on the current runtime condition.
    if isinstance(obj, np.floating):
        # Return the result expected by the caller.
        return float(obj)
    # Return the result expected by the caller.
    return obj


def run_genetic_nas(cfg, dataset, labels, groups, device, generator, shared_cache,
                    frame_stride, max_frames, validate_videos, cache_enabled,
                    use_preprocessed, processed_root, preprocessed_only,
                    pose_only_if_no_face, face_presence_threshold):
    # Compute `nas_cfg` for the next processing step.
    nas_cfg = cfg["nas"]
    # Compute `seed` for the next processing step.
    seed = cfg.get("seed", 42)

    # Compute `rng` for the next processing step.
    rng = np.random.RandomState(seed)
    # Invoke `random.seed` to advance this processing stage.
    random.seed(seed)
    # Invoke `torch.manual_seed` to advance this processing stage.
    torch.manual_seed(seed)

    # Compute `(train_idx, val_idx, n_splits)` for the next processing step.
    train_idx, val_idx, n_splits = make_group_stratified_split(
        labels, groups, val_fraction=nas_cfg["val_fraction"], seed=seed
    )
    # Invoke `print` to advance this processing stage.
    print(f"[NAS] Group stratified split with {n_splits} folds. Val size={len(val_idx)}")

    # Compute `train_ds` for the next processing step.
    train_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                   is_training=True, precompute=False, shared_cache=shared_cache,
                                   frame_stride=frame_stride, max_frames=max_frames,
                                   validate_videos=validate_videos, cache_enabled=cache_enabled,
                                   use_preprocessed=use_preprocessed, processed_root=processed_root,
                                   preprocessed_only=preprocessed_only), train_idx)
    # Compute `val_ds` for the next processing step.
    val_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                 is_training=False, precompute=False, shared_cache=shared_cache,
                                 frame_stride=frame_stride, max_frames=max_frames,
                                 validate_videos=validate_videos, cache_enabled=cache_enabled,
                                 use_preprocessed=use_preprocessed, processed_root=processed_root,
                                 preprocessed_only=preprocessed_only), val_idx)

    # Compute `train_loader` for the next processing step.
    train_loader = _build_train_loader(train_ds, labels[train_idx], cfg, generator)
    # Compute `val_loader` for the next processing step.
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Compute `pop_size` for the next processing step.
    pop_size = nas_cfg["population"]
    # Compute `generations` for the next processing step.
    generations = nas_cfg["generations"]
    # Compute `elite` for the next processing step.
    elite = nas_cfg["elite"]
    # Compute `mutation_rate` for the next processing step.
    mutation_rate = nas_cfg["mutation_rate"]
    # Compute `nas_spec_target` for the next processing step.
    nas_spec_target = float(nas_cfg.get("sens_spec_target", 0.90))

    # Compute `choice_logits` for the next processing step.
    choice_logits = {
        "kernel": np.zeros(4, dtype=float),
        "n_heads": np.zeros(3, dtype=float),
        "num_layers": np.zeros(3, dtype=float),
        "ff_dim": np.zeros(3, dtype=float),
    }
    # Compute `entropy_temp` for the next processing step.
    entropy_temp = 1.0

    # Compute `population` for the next processing step.
    population = [random_config(rng, choice_logits, entropy_temp) for _ in range(pop_size)]
    # Compute `best_cfg` for the next processing step.
    best_cfg = None
    # Compute `best_score` for the next processing step.
    best_score = -1e9

    # Iterate `gen` across `range(generations)` to process each element.
    for gen in range(generations):
        # Invoke `print` to advance this processing stage.
        print(f"\n[NAS] Generation {gen + 1}/{generations}")
        # Compute `fitness` for the next processing step.
        fitness = []
        # Evaluate population
        for i, cand in enumerate(population):
            # Compute `model` for the next processing step.
            model = ASDPipeline(alpha=cfg["model"]["alpha"],
                                K_max=cfg["model"]["K_max"],
                                d_model=cfg["model"]["d_model"],
                                dropout=cfg["model"]["dropout"],
                                theta_high=cfg["thresholds"]["decision_high"],
                                theta_low=cfg["thresholds"]["decision_low"]).to(device)
            # Invoke `model.freeze_cnns` to advance this processing stage.
            model.freeze_cnns(train_projection_heads=bool(cfg["training"].get("finetune_proj_heads", True)))
            # Invoke `model.apply_nas_architecture` to advance this processing stage.
            model.apply_nas_architecture(cand)

            # Compute `optimizer` for the next processing step.
            optimizer = build_optimizer(model,
                                        model_lr=cfg["training"]["lr"],
                                        arch_lr=cfg["training"]["arch_lr"],
                                        fusion_lr=cfg["training"]["fusion_lr"],
                                        weight_decay=cfg["training"]["weight_decay"])
            # Compute `scheduler` for the next processing step.
            scheduler = build_scheduler(optimizer, nas_cfg["epochs"], warmup_epochs=1)

            # Compute `pw` for the next processing step.
            pw = WeightedBCELoss.compute_from_labels(labels[train_idx])
            # Compute `criterion` for the next processing step.
            criterion = _build_criterion(cfg, pos_weight=pw)
            # Compute `scaler` for the next processing step.
            scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
            # Compute `obj_cfg` for the next processing step.
            obj_cfg = _train_objective_config(cfg, spec_target=nas_spec_target)

            # Iterate `_` across `range(nas_cfg['epochs'])` to process each element.
            for _ in range(nas_cfg["epochs"]):
                # Invoke `scheduler.step` to advance this processing stage.
                scheduler.step()
                # Compute `_` for the next processing step.
                _ = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                    clip_grad=cfg["training"]["clip_grad"],
                                    spec_target=obj_cfg["spec_target"],
                                    auc_rank_weight=obj_cfg["auc_rank_weight"],
                                    sens_spec_weight=obj_cfg["sens_spec_weight"],
                                    sens_spec_margin=obj_cfg["sens_spec_margin"])

            # Compute `(logits, y_true, probs, _)` for the next processing step.
            logits, y_true, probs, _ = collect_predictions(
                model, val_loader, device,
                pose_only_if_no_face=pose_only_if_no_face,
                face_presence_threshold=face_presence_threshold,
            )
            # Compute `metrics` for the next processing step.
            metrics = evaluate_metrics(
                y_true,
                probs,
                spec_target=nas_spec_target,
                n_bins=cfg["reporting"]["calibration_bins"],
                min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"],
            )
            # Compute `score` for the next processing step.
            score = _selection_score(metrics, cfg, apply_penalty=False)

            # Invoke `fitness.append` to advance this processing stage.
            fitness.append((score, cand))

            # Branch behavior based on the current runtime condition.
            if score > best_score:
                # Compute `best_score` for the next processing step.
                best_score = score
                # Compute `best_cfg` for the next processing step.
                best_cfg = cand

            # Invoke `print` to advance this processing stage.
            print(
                f"  [NAS] cand {i+1}/{len(population)} "
                f"score={score:.4f} auc={metrics['auc']:.4f} "
                f"f1={metrics['f1_opt']:.4f} sens@spec={metrics['sens_spec']:.4f}"
            )

        # Invoke `fitness.sort` to advance this processing stage.
        fitness.sort(key=lambda x: x[0], reverse=True)
        # Compute `elites` for the next processing step.
        elites = [c for _, c in fitness[:elite]]

        # Patch 1: collapse check
        uniq = {json.dumps(_jsonable(c), sort_keys=True) for _, c in fitness}
        # Branch behavior based on the current runtime condition.
        if len(uniq) == 1:
            # Invoke `print` to advance this processing stage.
            print("[NAS] Collapse detected (variance 0). Adding noise and boosting entropy.")
            # Compute `entropy_boost` for the next processing step.
            entropy_boost = nas_cfg.get("collapse_entropy_boost", 0.05)
            # Compute `mutation_rate` for the next processing step.
            mutation_rate = min(0.9, mutation_rate + 0.3)
            # Compute `entropy_temp` for the next processing step.
            entropy_temp = 1.0 + entropy_boost
            # Compute `noise_std` for the next processing step.
            noise_std = nas_cfg.get("collapse_noise_std", 0.3)
            # Iterate `k` across `choice_logits` to process each element.
            for k in choice_logits:
                # Compute `choice_logits[k]` for the next processing step.
                choice_logits[k] = choice_logits[k] + rng.normal(0.0, noise_std, size=choice_logits[k].shape)
        else:
            # Compute `entropy_temp` for the next processing step.
            entropy_temp = 1.0

        # Compute `new_pop` for the next processing step.
        new_pop = list(elites)
        # Continue looping until this condition no longer holds.
        while len(new_pop) < pop_size:
            # Branch behavior based on the current runtime condition.
            if rng.rand() < mutation_rate:
                # Compute `parent` for the next processing step.
                parent = elites[int(rng.randint(0, len(elites)))]
                # Compute `child` for the next processing step.
                child = mutate_config(parent, rng)
            else:
                # Compute `child` for the next processing step.
                child = random_config(rng, choice_logits, entropy_temp)
            # Invoke `new_pop.append` to advance this processing stage.
            new_pop.append(child)
        # Compute `population` for the next processing step.
        population = new_pop

    # Invoke `print` to advance this processing stage.
    print("[NAS] Best architecture selected:")
    # Invoke `print` to advance this processing stage.
    print(best_cfg)
    # Return the result expected by the caller.
    return best_cfg


def train(cfg):
    # Compute `seed` for the next processing step.
    seed = cfg.get("seed", 42)
    # Compute `generator` for the next processing step.
    generator = seed_everything(seed, deterministic=True)

    # Compute `device_pref` for the next processing step.
    device_pref = cfg.get("device", "auto")
    # Branch behavior based on the current runtime condition.
    if device_pref == "auto":
        # Compute `device_pref` for the next processing step.
        device_pref = "cuda"
    # Compute `device` for the next processing step.
    device = torch.device(get_device(device_pref))
    # Invoke `configure_cuda_optimizations` to advance this processing stage.
    configure_cuda_optimizations()
    # Invoke `print_gpu_info` to advance this processing stage.
    print_gpu_info()

    # Compute `results_dir` for the next processing step.
    results_dir = cfg["reporting"]["results_dir"]
    # Invoke `os.makedirs` to advance this processing stage.
    os.makedirs(results_dir, exist_ok=True)

    # Compute `frame_stride` for the next processing step.
    frame_stride = int(cfg.get("data", {}).get("frame_stride", 1))
    # Compute `max_frames` for the next processing step.
    max_frames = int(cfg.get("data", {}).get("max_frames", 0) or 0)
    # Compute `validate_videos` for the next processing step.
    validate_videos = bool(cfg.get("data", {}).get("validate_videos", False))
    # Compute `preprocess_videos` for the next processing step.
    preprocess_videos = bool(cfg.get("data", {}).get("preprocess_videos", False))
    # Compute `processed_root` for the next processing step.
    processed_root = cfg.get("data", {}).get("processed_root", "data/processed")
    # Compute `preprocess_overwrite` for the next processing step.
    preprocess_overwrite = bool(cfg.get("data", {}).get("preprocess_overwrite", False))
    # Compute `use_preprocessed` for the next processing step.
    use_preprocessed = bool(cfg.get("data", {}).get("use_preprocessed", False))
    # Compute `preprocessed_only` for the next processing step.
    preprocessed_only = bool(cfg.get("data", {}).get("preprocessed_only", True))
    # Compute `pose_only_if_no_face` for the next processing step.
    pose_only_if_no_face = bool(cfg.get("data", {}).get("pose_only_quality_if_no_face", False))
    # Compute `face_presence_threshold` for the next processing step.
    face_presence_threshold = float(cfg.get("data", {}).get("face_presence_threshold", 0.05))

    # Load dataset
    cache_enabled = bool(cfg.get("data", {}).get("cache_enabled", True))
    # Compute `shared_cache` for the next processing step.
    shared_cache = {} if cache_enabled else None

    # Branch behavior based on the current runtime condition.
    if preprocess_videos:
        # Invoke `print` to advance this processing stage.
        print("[Precompute] Starting video preprocessing...")
        # Invoke `precompute_videos` to advance this processing stage.
        precompute_videos(
            cfg["data"]["csv_path"],
            processed_root=processed_root,
            frame_stride=frame_stride,
            max_frames=max_frames,
            overwrite=preprocess_overwrite,
        )
    # Compute `dataset` for the next processing step.
    dataset = VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                           is_training=False, precompute=cfg["data"]["cache_precompute"],
                           shared_cache=shared_cache, frame_stride=frame_stride,
                           max_frames=max_frames, validate_videos=validate_videos,
                           cache_enabled=cache_enabled, use_preprocessed=use_preprocessed,
                           processed_root=processed_root, preprocessed_only=preprocessed_only)

    # Compute `labels` for the next processing step.
    labels = np.array([e["label"] for e in dataset.entries], dtype=int)
    # Compute `groups` for the next processing step.
    groups = np.array([e["subject_id"] for e in dataset.entries])

    # NAS search
    best_arch = None
    # Branch behavior based on the current runtime condition.
    if cfg["nas"]["enabled"]:
        # Compute `best_arch` for the next processing step.
        best_arch = run_genetic_nas(cfg, dataset, labels, groups, device, generator, shared_cache,
                                    frame_stride, max_frames, validate_videos, cache_enabled,
                                    use_preprocessed, processed_root, preprocessed_only,
                                    pose_only_if_no_face, face_presence_threshold)
        # Run this block with managed resources/context cleanup.
        with open(os.path.join(results_dir, "nas_architecture.json"), "w") as f:
            # Invoke `json.dump` to advance this processing stage.
            json.dump(_jsonable(best_arch), f, indent=2)
    else:
        # Compute `best_arch` for the next processing step.
        best_arch = ASDPipeline.get_random_config()
        # Invoke `print` to advance this processing stage.
        print("[NAS] Disabled. Using random architecture.")

    # Compute `folds` for the next processing step.
    folds = make_group_kfold(labels, groups, n_splits=5, seed=seed)
    # Compute `fold_summaries` for the next processing step.
    fold_summaries = []
    # Compute `all_fold_labels` for the next processing step.
    all_fold_labels = []
    # Compute `all_fold_probs` for the next processing step.
    all_fold_probs = []

    # Iterate `(fold_idx, (train_idx, val_idx))` across `enumerate(folds, start=1)` to process each element.
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        # Invoke `print` to advance this processing stage.
        print("\n" + "-" * 60)
        # Invoke `print` to advance this processing stage.
        print(f"Fold {fold_idx}/5 | Train={len(train_idx)} Val={len(val_idx)}")
        # Invoke `check_group_overlap` to advance this processing stage.
        check_group_overlap(groups[train_idx], groups[val_idx], fold_tag=f" fold {fold_idx}")

        # Compute `train_ds` for the next processing step.
        train_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                       is_training=True, precompute=False, shared_cache=shared_cache,
                                       frame_stride=frame_stride, max_frames=max_frames,
                                       validate_videos=validate_videos, cache_enabled=cache_enabled,
                                       use_preprocessed=use_preprocessed, processed_root=processed_root,
                                       preprocessed_only=preprocessed_only), train_idx)
        # Compute `val_ds` for the next processing step.
        val_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                     is_training=False, precompute=False, shared_cache=shared_cache,
                                     frame_stride=frame_stride, max_frames=max_frames,
                                     validate_videos=validate_videos, cache_enabled=cache_enabled,
                                     use_preprocessed=use_preprocessed, processed_root=processed_root,
                                     preprocessed_only=preprocessed_only), val_idx)

        # Compute `train_loader` for the next processing step.
        train_loader = _build_train_loader(train_ds, labels[train_idx], cfg, generator)
        # Compute `val_loader` for the next processing step.
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=collate_fn,
        )

        # Compute `model` for the next processing step.
        model = ASDPipeline(alpha=cfg["model"]["alpha"],
                            K_max=cfg["model"]["K_max"],
                            d_model=cfg["model"]["d_model"],
                            dropout=cfg["model"]["dropout"],
                            theta_high=cfg["thresholds"]["decision_high"],
                            theta_low=cfg["thresholds"]["decision_low"]).to(device)
        # Invoke `model.freeze_cnns` to advance this processing stage.
        model.freeze_cnns(train_projection_heads=bool(cfg["training"].get("finetune_proj_heads", True)))
        # Invoke `model.apply_nas_architecture` to advance this processing stage.
        model.apply_nas_architecture(best_arch)

        # Compute `optimizer` for the next processing step.
        optimizer = build_optimizer(model,
                                    model_lr=cfg["training"]["lr"],
                                    arch_lr=cfg["training"]["arch_lr"],
                                    fusion_lr=cfg["training"]["fusion_lr"],
                                    weight_decay=cfg["training"]["weight_decay"])
        # Compute `scheduler` for the next processing step.
        scheduler = build_scheduler(optimizer, cfg["training"]["epochs"], warmup_epochs=cfg["training"]["warmup"])

        # Compute `pw` for the next processing step.
        pw = WeightedBCELoss.compute_from_labels(labels[train_idx])
        # Compute `criterion` for the next processing step.
        criterion = _build_criterion(cfg, pos_weight=pw)
        # Compute `scaler` for the next processing step.
        scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
        # Compute `obj_cfg` for the next processing step.
        obj_cfg = _train_objective_config(cfg, spec_target=cfg["reporting"]["sens_spec_dev"])

        # Compute `early` for the next processing step.
        early = EarlyStopping(patience=cfg["training"]["patience"], mode="max")

        # Compute `best_score` for the next processing step.
        best_score = -1e9
        # Compute `best_path` for the next processing step.
        best_path = os.path.join(results_dir, f"asd_best_fold{fold_idx}.pth")
        # Compute `history` for the next processing step.
        history = []

        # Iterate `epoch` across `range(1, cfg['training']['epochs'...` to process each element.
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            # Invoke `scheduler.step` to advance this processing stage.
            scheduler.step()
            # Compute `train_loss` for the next processing step.
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                         clip_grad=cfg["training"]["clip_grad"],
                                         spec_target=obj_cfg["spec_target"],
                                         auc_rank_weight=obj_cfg["auc_rank_weight"],
                                         sens_spec_weight=obj_cfg["sens_spec_weight"],
                                         sens_spec_margin=obj_cfg["sens_spec_margin"])

            # Compute `val_loss` for the next processing step.
            val_loss = compute_val_loss(model, val_loader, criterion, device)
            # Compute `(logits, y_true, probs, quality_s...` for the next processing step.
            logits, y_true, probs, quality_scores = collect_predictions(
                model, val_loader, device,
                pose_only_if_no_face=pose_only_if_no_face,
                face_presence_threshold=face_presence_threshold,
            )
            # Compute `metrics` for the next processing step.
            metrics = evaluate_metrics(y_true, probs, spec_target=cfg["reporting"]["sens_spec_dev"],
                                       n_bins=cfg["reporting"]["calibration_bins"],
                                       min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
            # Compute `sel_score` for the next processing step.
            sel_score = _selection_score(metrics, cfg)

            # Invoke `history.append` to advance this processing stage.
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "auc": metrics["auc"],
                "f1_opt": metrics["f1_opt"],
                "accuracy_05": metrics["acc_05"],
                "accuracy_opt": metrics["acc_opt"],
                "ece": metrics["ece"],
                "sens_spec": metrics["sens_spec"],
                "selection_score": sel_score,
            })

            # Invoke `print` to advance this processing stage.
            print(
                f"  Epoch {epoch}/{cfg['training']['epochs']} "
                f"loss={train_loss:.4f} auc={metrics['auc']:.3f} "
                f"f1_opt={metrics['f1_opt']:.3f} sens@spec={metrics['sens_spec']:.3f} "
                f"(n_neg={metrics['n_negatives']}{' unstable' if metrics['sens_spec_unstable'] else ''}) "
                f"score={sel_score:.3f}"
            )

            # Branch behavior based on the current runtime condition.
            if sel_score > best_score:
                # Compute `best_score` for the next processing step.
                best_score = sel_score
                # Invoke `torch.save` to advance this processing stage.
                torch.save({
                    "model_state": model.state_dict(),
                    "nas_architecture": best_arch,
                }, best_path)

            # Branch behavior based on the current runtime condition.
            if early(sel_score, epoch=epoch):
                break

        # Load best model
        if os.path.exists(best_path):
            # Compute `ckpt` for the next processing step.
            ckpt = torch.load(best_path, map_location=device)
            # Invoke `model.load_state_dict` to advance this processing stage.
            model.load_state_dict(ckpt["model_state"])

        # Final evaluation on this fold
        logits, y_true, probs, quality_scores = collect_predictions(
            model, val_loader, device,
            pose_only_if_no_face=pose_only_if_no_face,
            face_presence_threshold=face_presence_threshold,
        )
        # Branch behavior based on the current runtime condition.
        if (y_true == 0).sum() < cfg["reporting"]["min_negatives_warn"]:
            # Invoke `print` to advance this processing stage.
            print(
                f"  [Warning] Validation negatives < {cfg['reporting']['min_negatives_warn']}; "
                "sens@spec may be unstable."
            )

        # Compute `temp` for the next processing step.
        temp = fit_temperature(torch.tensor(logits).to(device), torch.tensor(y_true).to(device), device)
        # Compute `logits_cal` for the next processing step.
        logits_cal = apply_temperature(torch.tensor(logits).to(device), temp).cpu().numpy()
        # Compute `probs_cal` for the next processing step.
        probs_cal = sigmoid(logits_cal)

        # Compute `metrics` for the next processing step.
        metrics = evaluate_metrics(y_true, probs_cal, spec_target=cfg["reporting"]["sens_spec_dev"],
                                   n_bins=cfg["reporting"]["calibration_bins"],
                                   min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
        # Compute `metrics['temperature']` for the next processing step.
        metrics["temperature"] = temp
        # Compute `metrics['labels']` for the next processing step.
        metrics["labels"] = y_true.tolist()
        # Compute `metrics['probs_cal']` for the next processing step.
        metrics["probs_cal"] = probs_cal.tolist()
        # Compute `metrics['calib_bins']` for the next processing step.
        metrics["calib_bins"] = cfg["reporting"]["calibration_bins"]
        # Compute `metrics['spec_target']` for the next processing step.
        metrics["spec_target"] = cfg["reporting"]["sens_spec_dev"]
        # Compute `metrics['abstain_rate']` for the next processing step.
        metrics["abstain_rate"] = compute_abstain_rate(probs_cal, quality_scores, cfg)

        # Explainability
        attn_map = extract_attention_maps(model, val_loader, device, n_samples=32)
        # Compute `temp_importance` for the next processing step.
        temp_importance = compute_temporal_importance(model, val_loader, device, n_samples=32)

        # Compute `nas_arch` for the next processing step.
        nas_arch = best_arch
        # Invoke `generate_training_report` to advance this processing stage.
        generate_training_report(
            results_dir,
            fold_idx,
            history,
            eval_summary=metrics,
            attention_map=attn_map,
            temporal_importance=temp_importance,
            nas_architecture=nas_arch,
            ema_alpha=cfg["training"]["ema_smoothing"],
        )

        # Save fold checkpoint with calibration
        torch.save({
            "model_state": model.state_dict(),
            "nas_architecture": best_arch,
            "temperature": temp,
            "config": cfg,
        }, best_path)

        # Invoke `fold_summaries.append` to advance this processing stage.
        fold_summaries.append({
            "fold": fold_idx,
            "auc": metrics["auc"],
            "f1_opt": metrics["f1_opt"],
            "acc_opt": metrics["acc_opt"],
            "sens_spec": metrics["sens_spec"],
            "ece": metrics["ece"],
            "abstain_rate": metrics["abstain_rate"],
        })

        # Invoke `all_fold_labels.append` to advance this processing stage.
        all_fold_labels.append(np.array(y_true))
        # Invoke `all_fold_probs.append` to advance this processing stage.
        all_fold_probs.append(np.array(probs_cal))

    # Cross-validation summary
    all_labels = np.concatenate(all_fold_labels) if all_fold_labels else np.array([])
    # Compute `all_probs` for the next processing step.
    all_probs = np.concatenate(all_fold_probs) if all_fold_probs else np.array([])

    # Compute `auc_ci` for the next processing step.
    auc_ci = bootstrap_ci(all_labels, all_probs, compute_auc, n_iters=cfg["reporting"]["bootstrap_iters"], seed=seed)
    # Compute `f1_ci` for the next processing step.
    f1_ci = bootstrap_ci(all_labels, all_probs,
                         lambda y, p: compute_basic_metrics(y, p, threshold=find_optimal_threshold(y, p))["f1"],
                         n_iters=cfg["reporting"]["bootstrap_iters"], seed=seed)

    # Compute `aucs` for the next processing step.
    aucs = [f["auc"] for f in fold_summaries]
    # Compute `f1s` for the next processing step.
    f1s = [f["f1_opt"] for f in fold_summaries]

    # Compute `cv_summary` for the next processing step.
    cv_summary = {
        "AUC mean": f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
        "F1 mean": f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
        "AUC 95% CI": f"[{auc_ci[0]:.3f}, {auc_ci[2]:.3f}]",
        "F1 95% CI": f"[{f1_ci[0]:.3f}, {f1_ci[2]:.3f}]",
    }

    # Final training on full dataset; keep a small split for monitoring/calibration reports.
    print("\n[Final] Training final model on full dataset (100%)")
    # Compute `(_, final_val_idx, n_splits)` for the next processing step.
    _, final_val_idx, n_splits = make_group_stratified_split(
        labels, groups, val_fraction=cfg["training"]["final_val_fraction"], seed=seed
    )
    # Invoke `print` to advance this processing stage.
    print(
        f"[Final] Using holdout subset only for monitoring/reporting: "
        f"{len(final_val_idx)} samples ({cfg['training']['final_val_fraction']:.0%})."
    )

    # Compute `final_train_ds` for the next processing step.
    final_train_ds = VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                  is_training=True, precompute=False, shared_cache=shared_cache,
                                  frame_stride=frame_stride, max_frames=max_frames,
                                  validate_videos=validate_videos, cache_enabled=cache_enabled,
                                  use_preprocessed=use_preprocessed, processed_root=processed_root,
                                  preprocessed_only=preprocessed_only)
    # Compute `final_val_ds` for the next processing step.
    final_val_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                       is_training=False, precompute=False, shared_cache=shared_cache,
                                       frame_stride=frame_stride, max_frames=max_frames,
                                       validate_videos=validate_videos, cache_enabled=cache_enabled,
                                       use_preprocessed=use_preprocessed, processed_root=processed_root,
                                       preprocessed_only=preprocessed_only), final_val_idx)

    # Compute `final_train_loader` for the next processing step.
    final_train_loader = _build_train_loader(final_train_ds, labels, cfg, generator)
    # Compute `final_val_loader` for the next processing step.
    final_val_loader = DataLoader(
        final_val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Compute `final_model` for the next processing step.
    final_model = ASDPipeline(alpha=cfg["model"]["alpha"],
                              K_max=cfg["model"]["K_max"],
                              d_model=cfg["model"]["d_model"],
                              dropout=cfg["model"]["dropout"],
                              theta_high=cfg["thresholds"]["decision_high"],
                              theta_low=cfg["thresholds"]["decision_low"]).to(device)
    # Invoke `final_model.freeze_cnns` to advance this processing stage.
    final_model.freeze_cnns(train_projection_heads=bool(cfg["training"].get("finetune_proj_heads", True)))
    # Invoke `final_model.apply_nas_architecture` to advance this processing stage.
    final_model.apply_nas_architecture(best_arch)

    # Compute `final_optimizer` for the next processing step.
    final_optimizer = build_optimizer(final_model,
                                      model_lr=cfg["training"]["lr"],
                                      arch_lr=cfg["training"]["arch_lr"],
                                      fusion_lr=cfg["training"]["fusion_lr"],
                                      weight_decay=cfg["training"]["weight_decay"])
    # Compute `final_scheduler` for the next processing step.
    final_scheduler = build_scheduler(
        final_optimizer,
        cfg["training"]["final_epochs"],
        warmup_epochs=cfg["training"]["warmup"],
    )
    # Compute `final_pw` for the next processing step.
    final_pw = WeightedBCELoss.compute_from_labels(labels)
    # Compute `final_criterion` for the next processing step.
    final_criterion = _build_criterion(cfg, pos_weight=final_pw)
    # Compute `final_scaler` for the next processing step.
    final_scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
    # Compute `final_obj_cfg` for the next processing step.
    final_obj_cfg = _train_objective_config(cfg, spec_target=cfg["reporting"]["sens_spec_final"])

    # Compute `early_final` for the next processing step.
    early_final = EarlyStopping(patience=cfg["training"]["patience"], mode="max")
    # Compute `final_best_score` for the next processing step.
    final_best_score = -1e9
    # Compute `final_best_path` for the next processing step.
    final_best_path = os.path.join(results_dir, "asd_pipeline_model.pth")
    # Compute `final_history` for the next processing step.
    final_history = []

    # Iterate `epoch` across `range(1, cfg['training']['final_e...` to process each element.
    for epoch in range(1, cfg["training"]["final_epochs"] + 1):
        # Invoke `final_scheduler.step` to advance this processing stage.
        final_scheduler.step()
        # Compute `loss` for the next processing step.
        loss = train_one_epoch(final_model, final_train_loader, final_criterion,
                               final_optimizer, final_scaler, device,
                               clip_grad=cfg["training"]["clip_grad"],
                               spec_target=final_obj_cfg["spec_target"],
                               auc_rank_weight=final_obj_cfg["auc_rank_weight"],
                               sens_spec_weight=final_obj_cfg["sens_spec_weight"],
                               sens_spec_margin=final_obj_cfg["sens_spec_margin"])

        # Compute `val_loss` for the next processing step.
        val_loss = compute_val_loss(final_model, final_val_loader, final_criterion, device)
        # Compute `(logits, y_true, probs, quality_s...` for the next processing step.
        logits, y_true, probs, quality_scores = collect_predictions(
            final_model, final_val_loader, device,
            pose_only_if_no_face=pose_only_if_no_face,
            face_presence_threshold=face_presence_threshold,
        )
        # Compute `metrics` for the next processing step.
        metrics = evaluate_metrics(y_true, probs, spec_target=cfg["reporting"]["sens_spec_final"],
                                   n_bins=cfg["reporting"]["calibration_bins"],
                                   min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
        # Compute `sel_score` for the next processing step.
        sel_score = _selection_score(metrics, cfg)

        # Invoke `final_history.append` to advance this processing stage.
        final_history.append({
            "epoch": epoch,
            "train_loss": loss,
            "val_loss": val_loss,
            "auc": metrics["auc"],
            "f1_opt": metrics["f1_opt"],
            "accuracy_05": metrics["acc_05"],
            "accuracy_opt": metrics["acc_opt"],
            "ece": metrics["ece"],
            "sens_spec": metrics["sens_spec"],
            "selection_score": sel_score,
        })

        # Invoke `print` to advance this processing stage.
        print(
            f"  [Final] Epoch {epoch}/{cfg['training']['final_epochs']} "
            f"loss={loss:.4f} auc={metrics['auc']:.3f} "
            f"f1_opt={metrics['f1_opt']:.3f} sens@spec={metrics['sens_spec']:.3f} "
            f"(n_neg={metrics['n_negatives']}{' unstable' if metrics['sens_spec_unstable'] else ''}) "
            f"score={sel_score:.3f}"
        )

        # Branch behavior based on the current runtime condition.
        if sel_score > final_best_score:
            # Compute `final_best_score` for the next processing step.
            final_best_score = sel_score
            # Invoke `torch.save` to advance this processing stage.
            torch.save({
                "model_state": final_model.state_dict(),
                "nas_architecture": best_arch,
            }, final_best_path)

        # Branch behavior based on the current runtime condition.
        if early_final(sel_score, epoch=epoch):
            break

    # Load best final model
    if os.path.exists(final_best_path):
        # Compute `ckpt` for the next processing step.
        ckpt = torch.load(final_best_path, map_location=device)
        # Invoke `final_model.load_state_dict` to advance this processing stage.
        final_model.load_state_dict(ckpt["model_state"])

    # Final evaluation with calibration
    logits, y_true, probs, quality_scores = collect_predictions(
        final_model, final_val_loader, device,
        pose_only_if_no_face=pose_only_if_no_face,
        face_presence_threshold=face_presence_threshold,
    )
    # Compute `temp` for the next processing step.
    temp = fit_temperature(torch.tensor(logits).to(device), torch.tensor(y_true).to(device), device)
    # Compute `logits_cal` for the next processing step.
    logits_cal = apply_temperature(torch.tensor(logits).to(device), temp).cpu().numpy()
    # Compute `probs_cal` for the next processing step.
    probs_cal = sigmoid(logits_cal)

    # Compute `final_metrics` for the next processing step.
    final_metrics = evaluate_metrics(y_true, probs_cal, spec_target=cfg["reporting"]["sens_spec_final"],
                                     n_bins=cfg["reporting"]["calibration_bins"],
                                     min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
    # Compute `final_metrics['temperature']` for the next processing step.
    final_metrics["temperature"] = temp
    # Compute `final_metrics['labels']` for the next processing step.
    final_metrics["labels"] = y_true.tolist()
    # Compute `final_metrics['probs_cal']` for the next processing step.
    final_metrics["probs_cal"] = probs_cal.tolist()
    # Compute `final_metrics['calib_bins']` for the next processing step.
    final_metrics["calib_bins"] = cfg["reporting"]["calibration_bins"]
    # Compute `final_metrics['spec_target']` for the next processing step.
    final_metrics["spec_target"] = cfg["reporting"]["sens_spec_final"]
    # Compute `final_metrics['abstain_rate']` for the next processing step.
    final_metrics["abstain_rate"] = compute_abstain_rate(probs_cal, quality_scores, cfg)

    # Compute `attn_map` for the next processing step.
    attn_map = extract_attention_maps(final_model, final_val_loader, device, n_samples=32)
    # Compute `temp_importance` for the next processing step.
    temp_importance = compute_temporal_importance(final_model, final_val_loader, device, n_samples=32)

    # Invoke `generate_training_report` to advance this processing stage.
    generate_training_report(
        results_dir,
        None,
        final_history,
        eval_summary=final_metrics,
        attention_map=attn_map,
        temporal_importance=temp_importance,
        nas_architecture=best_arch,
        cv_summary=cv_summary,
        ema_alpha=cfg["training"]["ema_smoothing"],
    )

    # Save final checkpoint with calibration and config
    torch.save({
        "model_state": final_model.state_dict(),
        "nas_architecture": best_arch,
        "temperature": temp,
        "config": cfg,
    }, final_best_path)

    # Invoke `print` to advance this processing stage.
    print(f"\nFinal model saved: {final_best_path}")


def main():
    # Compute `parser` for the next processing step.
    parser = argparse.ArgumentParser(description="Train ASD Motion Pipeline")
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--config", type=str, default="config.yaml")
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--override", type=str, action="append", default=[])
    # Invoke `parser.add_argument` to advance this processing stage.
    parser.add_argument("--csv", type=str, default=None, help="Override data.csv_path")
    # Compute `args` for the next processing step.
    args = parser.parse_args()

    # Compute `cfg` for the next processing step.
    cfg = load_config(args.config)
    # Branch behavior based on the current runtime condition.
    if args.csv:
        # Invoke `args.override.append` to advance this processing stage.
        args.override.append(f"data.csv_path={args.csv}")
    # Compute `cfg` for the next processing step.
    cfg = apply_overrides(cfg, args.override)

    # Invoke `train` to advance this processing stage.
    train(cfg)


# Branch behavior based on the current runtime condition.
if __name__ == "__main__":
    # Invoke `main` to advance this processing stage.
    main()

