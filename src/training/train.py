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
    out = {}
    out["face_crops"] = torch.stack([b["face_crops"] for b in batch])
    out["pose_maps"] = torch.stack([b["pose_maps"] for b in batch])
    out["motion_maps"] = torch.stack([b["motion_maps"] for b in batch])
    out["mask"] = torch.stack([b["mask"] for b in batch])
    out["timestamps"] = torch.stack([b["timestamps"] for b in batch])
    out["delta_t"] = torch.stack([b["delta_t"] for b in batch])
    out["route_mask"] = torch.stack([b["route_mask"] for b in batch])
    out["label"] = torch.stack([b["label"] for b in batch])
    out["qualities"] = {
        k: torch.stack([b["qualities"][k] for b in batch])
        for k in ("face_score", "pose_score", "hand_score")
    }
    out["video_id"] = [b["video_id"] for b in batch]
    out["subject_id"] = [b["subject_id"] for b in batch]
    return out


def _build_inputs(batch, device):
    return {
        "face_crops": batch["face_crops"].to(device, non_blocking=True),
        "pose_maps": batch["pose_maps"].to(device, non_blocking=True),
        "motion_maps": batch["motion_maps"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "timestamps": batch["timestamps"].to(device, non_blocking=True),
        "delta_t": batch["delta_t"].to(device, non_blocking=True),
        "route_mask": batch["route_mask"].to(device, non_blocking=True),
        "qualities": {k: v.to(device, non_blocking=True) for k, v in batch["qualities"].items()},
    }


def _build_train_loader(train_ds, train_labels, cfg, generator):
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    balance = bool(cfg.get("training", {}).get("balance_batches", True))

    sampler = None
    shuffle = True
    y = np.asarray(train_labels, dtype=int)

    if balance and y.size > 1 and np.unique(y).size > 1:
        class_counts = np.bincount(y, minlength=2).astype(float)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

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
    tcfg = cfg.get("training", {})
    return WeightedBCELoss(
        pos_weight=pos_weight,
        label_smoothing=float(tcfg.get("label_smoothing", 0.03)),
        logit_clip=float(tcfg.get("loss_logit_clip", 10.0)),
        brier_weight=float(tcfg.get("brier_weight", 0.1)),
    )


def _safe_metric_value(value, default=0.0):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return val


def _selection_score(metrics, cfg, apply_penalty=True):
    tcfg = cfg.get("training", {})
    auc = _safe_metric_value(metrics.get("auc"), 0.0)
    f1 = _safe_metric_value(metrics.get("f1_opt"), 0.0)
    sens_spec = _safe_metric_value(metrics.get("sens_spec"), 0.0)

    w_auc = float(tcfg.get("score_w_auc", 0.45))
    w_f1 = float(tcfg.get("score_w_f1", 0.35))
    w_ss = float(tcfg.get("score_w_sens_spec", 0.20))
    score = w_auc * auc + w_f1 * f1 + w_ss * sens_spec

    if not apply_penalty:
        return float(score)

    target_auc = float(tcfg.get("target_auc", 0.90))
    target_f1 = float(tcfg.get("target_f1", 0.90))
    target_ss = float(tcfg.get("target_sens_spec", 0.80))
    p_auc = float(tcfg.get("penalty_w_auc", 0.35))
    p_f1 = float(tcfg.get("penalty_w_f1", 0.35))
    p_ss = float(tcfg.get("penalty_w_sens_spec", 0.55))
    score -= p_auc * max(0.0, target_auc - auc)
    score -= p_f1 * max(0.0, target_f1 - f1)
    score -= p_ss * max(0.0, target_ss - sens_spec)
    return float(score)


def _train_objective_config(cfg, spec_target):
    tcfg = cfg.get("training", {})
    surrogate_spec_target = float(tcfg.get("surrogate_spec_target", spec_target))
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
    model.train()
    total_loss, n_batches = 0.0, 0
    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_amp = device_type.startswith("cuda")

    for batch in loader:
        inputs = _build_inputs(batch, device)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            out = model(inputs)
            logits = out["logit_final"]
            loss = criterion(logits, labels)

            if auc_rank_weight > 0 or sens_spec_weight > 0:
                aux = logits.new_tensor(0.0)
                if auc_rank_weight > 0:
                    aux = aux + float(auc_rank_weight) * pairwise_auc_loss(logits, labels, temperature=1.0)
                if sens_spec_weight > 0:
                    aux = aux + float(sens_spec_weight) * sens_at_spec_surrogate(
                        logits,
                        labels,
                        target_spec=spec_target,
                        margin=sens_spec_margin,
                        detach_threshold=True,
                    )
                loss = loss + aux

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        if not math.isfinite(float(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_val
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_amp = device_type.startswith("cuda")

    for batch in loader:
        inputs = _build_inputs(batch, device)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            out = model(inputs)
            loss = criterion(out["logit_final"], labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def collect_predictions(model, loader, device,
                        pose_only_if_no_face=False,
                        face_presence_threshold=0.05):
    model.eval()
    logits_all = []
    labels_all = []
    quality_all = []

    device_type = device.type if isinstance(device, torch.device) else str(device)
    use_amp = device_type.startswith("cuda")

    for batch in loader:
        inputs = _build_inputs(batch, device)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            out = model(inputs)
            logits = out["logit_final"]

        logits_all.append(logits.detach().float().cpu())
        labels_all.append(labels.detach().float().cpu())

        q = compute_quality_score(
            batch["qualities"],
            batch["mask"],
            pose_only_if_no_face=pose_only_if_no_face,
            face_presence_threshold=face_presence_threshold,
        ).detach().cpu()
        quality_all.append(q)

    logits_all = torch.cat(logits_all).numpy() if logits_all else np.array([])
    labels_all = torch.cat(labels_all).numpy().astype(int) if labels_all else np.array([])
    quality_all = torch.cat(quality_all).numpy() if quality_all else np.array([])

    probs = sigmoid(logits_all) if len(logits_all) else np.array([])
    return logits_all, labels_all, probs, quality_all


def evaluate_metrics(labels, probs, spec_target=0.95, n_bins=10, min_negatives_for_sens_spec=20):
    n_neg = int((np.asarray(labels) == 0).sum())
    auc = compute_auc(labels, probs)
    ece = compute_ece(labels, probs, n_bins=n_bins)
    thr_opt = find_optimal_threshold(labels, probs)

    m05 = compute_basic_metrics(labels, probs, threshold=0.5)
    mopt = compute_basic_metrics(labels, probs, threshold=thr_opt)

    sens_spec = sensitivity_at_specificity(
        labels,
        probs,
        target_spec=spec_target,
        min_negatives=min_negatives_for_sens_spec,
        allow_unstable=True,
    )

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
    low_thr = cfg["thresholds"]["decision_low"]
    high_thr = cfg["thresholds"]["decision_high"]
    q_thr = cfg["thresholds"]["quality_threshold"]
    abstain = 0
    for p, q in zip(probs_cal, quality_scores):
        res = make_decision(float(p), float(p), float(q), q_thr, low_thr, high_thr)
        if res.abstained:
            abstain += 1
    return abstain / max(len(probs_cal), 1)


def mutate_config(cfg, rng):
    new_cfg = json.loads(json.dumps(cfg))
    # Randomly change one field
    choices = ["encoder_kernel", "n_heads", "num_encoder_layers", "dim_ff"]
    field = str(rng.choice(choices))
    if field == "encoder_kernel":
        new_cfg["encoder_kernel"] = int(rng.choice([3, 5, 7, 11]))
    elif field == "n_heads":
        new_cfg["transformer"]["n_heads"] = int(rng.choice([2, 4, 8]))
    elif field == "num_encoder_layers":
        new_cfg["transformer"]["num_encoder_layers"] = int(rng.choice([2, 3, 4]))
    elif field == "dim_ff":
        new_cfg["transformer"]["dim_ff"] = int(rng.choice([512, 1024, 2048]))
    return new_cfg


def _sample_from_logits(rng, choices, logits=None, temperature=1.0):
    if logits is None:
        return int(rng.choice(choices))
    logits = np.array(logits, dtype=float) / max(temperature, 1e-6)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(rng.choice(choices, p=probs))


def random_config(rng, choice_logits=None, temperature=1.0):
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
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def run_genetic_nas(cfg, dataset, labels, groups, device, generator, shared_cache,
                    frame_stride, max_frames, validate_videos, cache_enabled,
                    use_preprocessed, processed_root, preprocessed_only,
                    pose_only_if_no_face, face_presence_threshold):
    nas_cfg = cfg["nas"]
    seed = cfg.get("seed", 42)

    rng = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_idx, val_idx, n_splits = make_group_stratified_split(
        labels, groups, val_fraction=nas_cfg["val_fraction"], seed=seed
    )
    print(f"[NAS] Group stratified split with {n_splits} folds. Val size={len(val_idx)}")

    train_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                   is_training=True, precompute=False, shared_cache=shared_cache,
                                   frame_stride=frame_stride, max_frames=max_frames,
                                   validate_videos=validate_videos, cache_enabled=cache_enabled,
                                   use_preprocessed=use_preprocessed, processed_root=processed_root,
                                   preprocessed_only=preprocessed_only), train_idx)
    val_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                 is_training=False, precompute=False, shared_cache=shared_cache,
                                 frame_stride=frame_stride, max_frames=max_frames,
                                 validate_videos=validate_videos, cache_enabled=cache_enabled,
                                 use_preprocessed=use_preprocessed, processed_root=processed_root,
                                 preprocessed_only=preprocessed_only), val_idx)

    train_loader = _build_train_loader(train_ds, labels[train_idx], cfg, generator)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    pop_size = nas_cfg["population"]
    generations = nas_cfg["generations"]
    elite = nas_cfg["elite"]
    mutation_rate = nas_cfg["mutation_rate"]
    nas_spec_target = float(nas_cfg.get("sens_spec_target", 0.90))

    choice_logits = {
        "kernel": np.zeros(4, dtype=float),
        "n_heads": np.zeros(3, dtype=float),
        "num_layers": np.zeros(3, dtype=float),
        "ff_dim": np.zeros(3, dtype=float),
    }
    entropy_temp = 1.0

    population = [random_config(rng, choice_logits, entropy_temp) for _ in range(pop_size)]
    best_cfg = None
    best_score = -1e9

    for gen in range(generations):
        print(f"\n[NAS] Generation {gen + 1}/{generations}")
        fitness = []
        # Evaluate population
        for i, cand in enumerate(population):
            model = ASDPipeline(alpha=cfg["model"]["alpha"],
                                K_max=cfg["model"]["K_max"],
                                d_model=cfg["model"]["d_model"],
                                dropout=cfg["model"]["dropout"],
                                theta_high=cfg["thresholds"]["decision_high"],
                                theta_low=cfg["thresholds"]["decision_low"]).to(device)
            model.freeze_cnns(train_projection_heads=bool(cfg["training"].get("finetune_proj_heads", True)))
            model.apply_nas_architecture(cand)

            optimizer = build_optimizer(model,
                                        model_lr=cfg["training"]["lr"],
                                        arch_lr=cfg["training"]["arch_lr"],
                                        fusion_lr=cfg["training"]["fusion_lr"],
                                        weight_decay=cfg["training"]["weight_decay"])
            scheduler = build_scheduler(optimizer, nas_cfg["epochs"], warmup_epochs=1)

            pw = WeightedBCELoss.compute_from_labels(labels[train_idx])
            criterion = _build_criterion(cfg, pos_weight=pw)
            scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
            obj_cfg = _train_objective_config(cfg, spec_target=nas_spec_target)

            for _ in range(nas_cfg["epochs"]):
                scheduler.step()
                _ = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                    clip_grad=cfg["training"]["clip_grad"],
                                    spec_target=obj_cfg["spec_target"],
                                    auc_rank_weight=obj_cfg["auc_rank_weight"],
                                    sens_spec_weight=obj_cfg["sens_spec_weight"],
                                    sens_spec_margin=obj_cfg["sens_spec_margin"])

            logits, y_true, probs, _ = collect_predictions(
                model, val_loader, device,
                pose_only_if_no_face=pose_only_if_no_face,
                face_presence_threshold=face_presence_threshold,
            )
            metrics = evaluate_metrics(
                y_true,
                probs,
                spec_target=nas_spec_target,
                n_bins=cfg["reporting"]["calibration_bins"],
                min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"],
            )
            score = _selection_score(metrics, cfg, apply_penalty=False)

            fitness.append((score, cand))

            if score > best_score:
                best_score = score
                best_cfg = cand

            print(
                f"  [NAS] cand {i+1}/{len(population)} "
                f"score={score:.4f} auc={metrics['auc']:.4f} "
                f"f1={metrics['f1_opt']:.4f} sens@spec={metrics['sens_spec']:.4f}"
            )

        fitness.sort(key=lambda x: x[0], reverse=True)
        elites = [c for _, c in fitness[:elite]]

        # Patch 1: collapse check
        uniq = {json.dumps(_jsonable(c), sort_keys=True) for _, c in fitness}
        if len(uniq) == 1:
            print("[NAS] Collapse detected (variance 0). Adding noise and boosting entropy.")
            entropy_boost = nas_cfg.get("collapse_entropy_boost", 0.05)
            mutation_rate = min(0.9, mutation_rate + 0.3)
            entropy_temp = 1.0 + entropy_boost
            noise_std = nas_cfg.get("collapse_noise_std", 0.3)
            for k in choice_logits:
                choice_logits[k] = choice_logits[k] + rng.normal(0.0, noise_std, size=choice_logits[k].shape)
        else:
            entropy_temp = 1.0

        new_pop = list(elites)
        while len(new_pop) < pop_size:
            if rng.rand() < mutation_rate:
                parent = elites[int(rng.randint(0, len(elites)))]
                child = mutate_config(parent, rng)
            else:
                child = random_config(rng, choice_logits, entropy_temp)
            new_pop.append(child)
        population = new_pop

    print("[NAS] Best architecture selected:")
    print(best_cfg)
    return best_cfg


def train(cfg):
    seed = cfg.get("seed", 42)
    generator = seed_everything(seed, deterministic=True)

    device_pref = cfg.get("device", "auto")
    if device_pref == "auto":
        device_pref = "cuda"
    device = torch.device(get_device(device_pref))
    configure_cuda_optimizations()
    print_gpu_info()

    results_dir = cfg["reporting"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    frame_stride = int(cfg.get("data", {}).get("frame_stride", 1))
    max_frames = int(cfg.get("data", {}).get("max_frames", 0) or 0)
    validate_videos = bool(cfg.get("data", {}).get("validate_videos", False))
    preprocess_videos = bool(cfg.get("data", {}).get("preprocess_videos", False))
    processed_root = cfg.get("data", {}).get("processed_root", "data/processed")
    preprocess_overwrite = bool(cfg.get("data", {}).get("preprocess_overwrite", False))
    use_preprocessed = bool(cfg.get("data", {}).get("use_preprocessed", False))
    preprocessed_only = bool(cfg.get("data", {}).get("preprocessed_only", True))
    pose_only_if_no_face = bool(cfg.get("data", {}).get("pose_only_quality_if_no_face", False))
    face_presence_threshold = float(cfg.get("data", {}).get("face_presence_threshold", 0.05))

    # Load dataset
    cache_enabled = bool(cfg.get("data", {}).get("cache_enabled", True))
    shared_cache = {} if cache_enabled else None

    if preprocess_videos:
        print("[Precompute] Starting video preprocessing...")
        precompute_videos(
            cfg["data"]["csv_path"],
            processed_root=processed_root,
            frame_stride=frame_stride,
            max_frames=max_frames,
            overwrite=preprocess_overwrite,
        )
    dataset = VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                           is_training=False, precompute=cfg["data"]["cache_precompute"],
                           shared_cache=shared_cache, frame_stride=frame_stride,
                           max_frames=max_frames, validate_videos=validate_videos,
                           cache_enabled=cache_enabled, use_preprocessed=use_preprocessed,
                           processed_root=processed_root, preprocessed_only=preprocessed_only)

    labels = np.array([e["label"] for e in dataset.entries], dtype=int)
    groups = np.array([e["subject_id"] for e in dataset.entries])

    # NAS search
    best_arch = None
    if cfg["nas"]["enabled"]:
        best_arch = run_genetic_nas(cfg, dataset, labels, groups, device, generator, shared_cache,
                                    frame_stride, max_frames, validate_videos, cache_enabled,
                                    use_preprocessed, processed_root, preprocessed_only,
                                    pose_only_if_no_face, face_presence_threshold)
        with open(os.path.join(results_dir, "nas_architecture.json"), "w") as f:
            json.dump(_jsonable(best_arch), f, indent=2)
    else:
        best_arch = ASDPipeline.get_random_config()
        print("[NAS] Disabled. Using random architecture.")

    folds = make_group_kfold(labels, groups, n_splits=5, seed=seed)
    fold_summaries = []
    all_fold_labels = []
    all_fold_probs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        print("\n" + "-" * 60)
        print(f"Fold {fold_idx}/5 | Train={len(train_idx)} Val={len(val_idx)}")
        check_group_overlap(groups[train_idx], groups[val_idx], fold_tag=f" fold {fold_idx}")

        train_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                       is_training=True, precompute=False, shared_cache=shared_cache,
                                       frame_stride=frame_stride, max_frames=max_frames,
                                       validate_videos=validate_videos, cache_enabled=cache_enabled,
                                       use_preprocessed=use_preprocessed, processed_root=processed_root,
                                       preprocessed_only=preprocessed_only), train_idx)
        val_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                     is_training=False, precompute=False, shared_cache=shared_cache,
                                     frame_stride=frame_stride, max_frames=max_frames,
                                     validate_videos=validate_videos, cache_enabled=cache_enabled,
                                     use_preprocessed=use_preprocessed, processed_root=processed_root,
                                     preprocessed_only=preprocessed_only), val_idx)

        train_loader = _build_train_loader(train_ds, labels[train_idx], cfg, generator)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=collate_fn,
        )

        model = ASDPipeline(alpha=cfg["model"]["alpha"],
                            K_max=cfg["model"]["K_max"],
                            d_model=cfg["model"]["d_model"],
                            dropout=cfg["model"]["dropout"],
                            theta_high=cfg["thresholds"]["decision_high"],
                            theta_low=cfg["thresholds"]["decision_low"]).to(device)
        model.freeze_cnns(train_projection_heads=bool(cfg["training"].get("finetune_proj_heads", True)))
        model.apply_nas_architecture(best_arch)

        optimizer = build_optimizer(model,
                                    model_lr=cfg["training"]["lr"],
                                    arch_lr=cfg["training"]["arch_lr"],
                                    fusion_lr=cfg["training"]["fusion_lr"],
                                    weight_decay=cfg["training"]["weight_decay"])
        scheduler = build_scheduler(optimizer, cfg["training"]["epochs"], warmup_epochs=cfg["training"]["warmup"])

        pw = WeightedBCELoss.compute_from_labels(labels[train_idx])
        criterion = _build_criterion(cfg, pos_weight=pw)
        scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
        obj_cfg = _train_objective_config(cfg, spec_target=cfg["reporting"]["sens_spec_dev"])

        early = EarlyStopping(patience=cfg["training"]["patience"], mode="max")

        best_score = -1e9
        best_path = os.path.join(results_dir, f"asd_best_fold{fold_idx}.pth")
        history = []

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            scheduler.step()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                         clip_grad=cfg["training"]["clip_grad"],
                                         spec_target=obj_cfg["spec_target"],
                                         auc_rank_weight=obj_cfg["auc_rank_weight"],
                                         sens_spec_weight=obj_cfg["sens_spec_weight"],
                                         sens_spec_margin=obj_cfg["sens_spec_margin"])

            val_loss = compute_val_loss(model, val_loader, criterion, device)
            logits, y_true, probs, quality_scores = collect_predictions(
                model, val_loader, device,
                pose_only_if_no_face=pose_only_if_no_face,
                face_presence_threshold=face_presence_threshold,
            )
            metrics = evaluate_metrics(y_true, probs, spec_target=cfg["reporting"]["sens_spec_dev"],
                                       n_bins=cfg["reporting"]["calibration_bins"],
                                       min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
            sel_score = _selection_score(metrics, cfg)

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

            print(
                f"  Epoch {epoch}/{cfg['training']['epochs']} "
                f"loss={train_loss:.4f} auc={metrics['auc']:.3f} "
                f"f1_opt={metrics['f1_opt']:.3f} sens@spec={metrics['sens_spec']:.3f} "
                f"(n_neg={metrics['n_negatives']}{' unstable' if metrics['sens_spec_unstable'] else ''}) "
                f"score={sel_score:.3f}"
            )

            if sel_score > best_score:
                best_score = sel_score
                torch.save({
                    "model_state": model.state_dict(),
                    "nas_architecture": best_arch,
                }, best_path)

            if early(sel_score, epoch=epoch):
                break

        # Load best model
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

        # Final evaluation on this fold
        logits, y_true, probs, quality_scores = collect_predictions(
            model, val_loader, device,
            pose_only_if_no_face=pose_only_if_no_face,
            face_presence_threshold=face_presence_threshold,
        )
        if (y_true == 0).sum() < cfg["reporting"]["min_negatives_warn"]:
            print(
                f"  [Warning] Validation negatives < {cfg['reporting']['min_negatives_warn']}; "
                "sens@spec may be unstable."
            )

        temp = fit_temperature(torch.tensor(logits).to(device), torch.tensor(y_true).to(device), device)
        logits_cal = apply_temperature(torch.tensor(logits).to(device), temp).cpu().numpy()
        probs_cal = sigmoid(logits_cal)

        metrics = evaluate_metrics(y_true, probs_cal, spec_target=cfg["reporting"]["sens_spec_dev"],
                                   n_bins=cfg["reporting"]["calibration_bins"],
                                   min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
        metrics["temperature"] = temp
        metrics["labels"] = y_true.tolist()
        metrics["probs_cal"] = probs_cal.tolist()
        metrics["calib_bins"] = cfg["reporting"]["calibration_bins"]
        metrics["spec_target"] = cfg["reporting"]["sens_spec_dev"]
        metrics["abstain_rate"] = compute_abstain_rate(probs_cal, quality_scores, cfg)

        # Explainability
        attn_map = extract_attention_maps(model, val_loader, device, n_samples=32)
        temp_importance = compute_temporal_importance(model, val_loader, device, n_samples=32)

        nas_arch = best_arch
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

        fold_summaries.append({
            "fold": fold_idx,
            "auc": metrics["auc"],
            "f1_opt": metrics["f1_opt"],
            "acc_opt": metrics["acc_opt"],
            "sens_spec": metrics["sens_spec"],
            "ece": metrics["ece"],
            "abstain_rate": metrics["abstain_rate"],
        })

        all_fold_labels.append(np.array(y_true))
        all_fold_probs.append(np.array(probs_cal))

    # Cross-validation summary
    all_labels = np.concatenate(all_fold_labels) if all_fold_labels else np.array([])
    all_probs = np.concatenate(all_fold_probs) if all_fold_probs else np.array([])

    auc_ci = bootstrap_ci(all_labels, all_probs, compute_auc, n_iters=cfg["reporting"]["bootstrap_iters"], seed=seed)
    f1_ci = bootstrap_ci(all_labels, all_probs,
                         lambda y, p: compute_basic_metrics(y, p, threshold=find_optimal_threshold(y, p))["f1"],
                         n_iters=cfg["reporting"]["bootstrap_iters"], seed=seed)

    aucs = [f["auc"] for f in fold_summaries]
    f1s = [f["f1_opt"] for f in fold_summaries]

    cv_summary = {
        "AUC mean": f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
        "F1 mean": f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
        "AUC 95% CI": f"[{auc_ci[0]:.3f}, {auc_ci[2]:.3f}]",
        "F1 95% CI": f"[{f1_ci[0]:.3f}, {f1_ci[2]:.3f}]",
    }

    # Final training on full dataset; keep a small split for monitoring/calibration reports.
    print("\n[Final] Training final model on full dataset (100%)")
    _, final_val_idx, n_splits = make_group_stratified_split(
        labels, groups, val_fraction=cfg["training"]["final_val_fraction"], seed=seed
    )
    print(
        f"[Final] Using holdout subset only for monitoring/reporting: "
        f"{len(final_val_idx)} samples ({cfg['training']['final_val_fraction']:.0%})."
    )

    final_train_ds = VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                  is_training=True, precompute=False, shared_cache=shared_cache,
                                  frame_stride=frame_stride, max_frames=max_frames,
                                  validate_videos=validate_videos, cache_enabled=cache_enabled,
                                  use_preprocessed=use_preprocessed, processed_root=processed_root,
                                  preprocessed_only=preprocessed_only)
    final_val_ds = Subset(VideoDataset(cfg["data"]["csv_path"], sequence_length=cfg["data"]["seq_len"],
                                       is_training=False, precompute=False, shared_cache=shared_cache,
                                       frame_stride=frame_stride, max_frames=max_frames,
                                       validate_videos=validate_videos, cache_enabled=cache_enabled,
                                       use_preprocessed=use_preprocessed, processed_root=processed_root,
                                       preprocessed_only=preprocessed_only), final_val_idx)

    final_train_loader = _build_train_loader(final_train_ds, labels, cfg, generator)
    final_val_loader = DataLoader(
        final_val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    final_model = ASDPipeline(alpha=cfg["model"]["alpha"],
                              K_max=cfg["model"]["K_max"],
                              d_model=cfg["model"]["d_model"],
                              dropout=cfg["model"]["dropout"],
                              theta_high=cfg["thresholds"]["decision_high"],
                              theta_low=cfg["thresholds"]["decision_low"]).to(device)
    final_model.freeze_cnns(train_projection_heads=bool(cfg["training"].get("finetune_proj_heads", True)))
    final_model.apply_nas_architecture(best_arch)

    final_optimizer = build_optimizer(final_model,
                                      model_lr=cfg["training"]["lr"],
                                      arch_lr=cfg["training"]["arch_lr"],
                                      fusion_lr=cfg["training"]["fusion_lr"],
                                      weight_decay=cfg["training"]["weight_decay"])
    final_scheduler = build_scheduler(
        final_optimizer,
        cfg["training"]["final_epochs"],
        warmup_epochs=cfg["training"]["warmup"],
    )
    final_pw = WeightedBCELoss.compute_from_labels(labels)
    final_criterion = _build_criterion(cfg, pos_weight=final_pw)
    final_scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
    final_obj_cfg = _train_objective_config(cfg, spec_target=cfg["reporting"]["sens_spec_final"])

    early_final = EarlyStopping(patience=cfg["training"]["patience"], mode="max")
    final_best_score = -1e9
    final_best_path = os.path.join(results_dir, "asd_pipeline_model.pth")
    final_history = []

    for epoch in range(1, cfg["training"]["final_epochs"] + 1):
        final_scheduler.step()
        loss = train_one_epoch(final_model, final_train_loader, final_criterion,
                               final_optimizer, final_scaler, device,
                               clip_grad=cfg["training"]["clip_grad"],
                               spec_target=final_obj_cfg["spec_target"],
                               auc_rank_weight=final_obj_cfg["auc_rank_weight"],
                               sens_spec_weight=final_obj_cfg["sens_spec_weight"],
                               sens_spec_margin=final_obj_cfg["sens_spec_margin"])

        val_loss = compute_val_loss(final_model, final_val_loader, final_criterion, device)
        logits, y_true, probs, quality_scores = collect_predictions(
            final_model, final_val_loader, device,
            pose_only_if_no_face=pose_only_if_no_face,
            face_presence_threshold=face_presence_threshold,
        )
        metrics = evaluate_metrics(y_true, probs, spec_target=cfg["reporting"]["sens_spec_final"],
                                   n_bins=cfg["reporting"]["calibration_bins"],
                                   min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
        sel_score = _selection_score(metrics, cfg)

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

        print(
            f"  [Final] Epoch {epoch}/{cfg['training']['final_epochs']} "
            f"loss={loss:.4f} auc={metrics['auc']:.3f} "
            f"f1_opt={metrics['f1_opt']:.3f} sens@spec={metrics['sens_spec']:.3f} "
            f"(n_neg={metrics['n_negatives']}{' unstable' if metrics['sens_spec_unstable'] else ''}) "
            f"score={sel_score:.3f}"
        )

        if sel_score > final_best_score:
            final_best_score = sel_score
            torch.save({
                "model_state": final_model.state_dict(),
                "nas_architecture": best_arch,
            }, final_best_path)

        if early_final(sel_score, epoch=epoch):
            break

    # Load best final model
    if os.path.exists(final_best_path):
        ckpt = torch.load(final_best_path, map_location=device)
        final_model.load_state_dict(ckpt["model_state"])

    # Final evaluation with calibration
    logits, y_true, probs, quality_scores = collect_predictions(
        final_model, final_val_loader, device,
        pose_only_if_no_face=pose_only_if_no_face,
        face_presence_threshold=face_presence_threshold,
    )
    temp = fit_temperature(torch.tensor(logits).to(device), torch.tensor(y_true).to(device), device)
    logits_cal = apply_temperature(torch.tensor(logits).to(device), temp).cpu().numpy()
    probs_cal = sigmoid(logits_cal)

    final_metrics = evaluate_metrics(y_true, probs_cal, spec_target=cfg["reporting"]["sens_spec_final"],
                                     n_bins=cfg["reporting"]["calibration_bins"],
                                     min_negatives_for_sens_spec=cfg["reporting"]["min_negatives_warn"])
    final_metrics["temperature"] = temp
    final_metrics["labels"] = y_true.tolist()
    final_metrics["probs_cal"] = probs_cal.tolist()
    final_metrics["calib_bins"] = cfg["reporting"]["calibration_bins"]
    final_metrics["spec_target"] = cfg["reporting"]["sens_spec_final"]
    final_metrics["abstain_rate"] = compute_abstain_rate(probs_cal, quality_scores, cfg)

    attn_map = extract_attention_maps(final_model, final_val_loader, device, n_samples=32)
    temp_importance = compute_temporal_importance(final_model, final_val_loader, device, n_samples=32)

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

    print(f"\nFinal model saved: {final_best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ASD Motion Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, action="append", default=[])
    parser.add_argument("--csv", type=str, default=None, help="Override data.csv_path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.csv:
        args.override.append(f"data.csv_path={args.csv}")
    cfg = apply_overrides(cfg, args.override)

    train(cfg)


if __name__ == "__main__":
    main()

