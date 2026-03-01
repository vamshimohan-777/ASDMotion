"""Micro-genetic NAS loop."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.nas_controller import MicroGeneticNAS, default_search_space
from src.models.pipeline_model import ASDPipeline
from src.training.dataset import collate_motion_batch
from src.training.losses import WeightedBCELoss
from src.utils.metrics import compute_auc, compute_ece, sensitivity_at_specificity
from src.utils.splits import check_group_overlap, make_group_kfold


def _to_device(batch, device):
    x = {
        "motion_windows": batch["motion_windows"].to(device, non_blocking=True),
        "joint_mask": batch["joint_mask"].to(device, non_blocking=True),
        "window_timestamps": batch["window_timestamps"].to(device, non_blocking=True),
    }
    if "rgb_windows" in batch:
        x["rgb_windows"] = batch["rgb_windows"].to(device, non_blocking=True)
    return x


def _train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        y = batch["label"].to(device, non_blocking=True)
        x = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
            out = model(x)
            loss = criterion(out["logit_final"], y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    probs = []
    labels = []
    started = time.time()
    for batch in loader:
        y = batch["label"].to(device, non_blocking=True)
        x = _to_device(batch, device)
        p = torch.sigmoid(model(x)["logit_final"])
        probs.append(p.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    infer_time = time.time() - started
    if not probs:
        return {"auc": 0.5, "sens_at_90_spec": 0.0, "calibration_quality": 0.0}, infer_time
    probs = np.concatenate(probs).astype(float)
    labels = np.concatenate(labels).astype(int)
    auc = compute_auc(labels, probs)
    sens90 = sensitivity_at_specificity(labels, probs, target_spec=0.90, min_negatives=10, allow_unstable=True)
    if not np.isfinite(sens90):
        sens90 = 0.0
    ece = compute_ece(labels, probs, n_bins=10)
    return {
        "auc": float(auc),
        "sens_at_90_spec": float(max(0.0, sens90)),
        "calibration_quality": float(np.clip(1.0 - ece, 0.0, 1.0)),
    }, infer_time


def _efficiency_penalty(train_time, infer_time, max_mem_gb):
    t_pen = np.clip(train_time / 60.0, 0.0, 1.0)
    i_pen = np.clip(infer_time / 5.0, 0.0, 1.0)
    m_pen = np.clip(max_mem_gb / 6.0, 0.0, 1.0)
    return float(np.clip(0.45 * m_pen + 0.35 * t_pen + 0.20 * i_pen, 0.0, 1.0))


def run_micro_genetic_nas(cfg, dataset, labels, groups, device, logger=None):
    nas_cfg = cfg.get("nas", {})
    t_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    threshold_cfg = cfg.get("thresholds", {})

    pop = int(nas_cfg.get("population", 20))
    gens = int(nas_cfg.get("generations", 20))
    tour = int(nas_cfg.get("tournament_size", 3))
    mut = float(nas_cfg.get("mutation_rate", 0.15))
    crossover = bool(nas_cfg.get("crossover", True))
    elite = int(nas_cfg.get("elite", 2))
    nas_epochs = int(nas_cfg.get("epochs", 2))
    nas_folds = int(nas_cfg.get("eval_folds", 3))
    batch_size = int(t_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    use_rgb = bool(data_cfg.get("use_rgb", False))

    search_space = copy.deepcopy(nas_cfg.get("search_space")) if nas_cfg.get("search_space") else default_search_space()
    all_folds = make_group_kfold(labels, groups, n_splits=max(2, int(t_cfg.get("cv_folds", 5))), seed=int(cfg.get("seed", 42)))
    folds = all_folds[: max(1, min(len(all_folds), nas_folds))]

    def evaluate_arch(arch):
        fold_aucs, fold_sens, fold_cal = [], [], []
        fold_train_times, fold_infer_times, fold_mem = [], [], []

        original_window_sizes = tuple(getattr(dataset, "window_sizes", (48,)))
        try:
            if "window" in arch and "size" in arch["window"]:
                dataset.window_sizes = (int(arch["window"]["size"]),)

            for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
                check_group_overlap(groups[tr_idx], groups[va_idx], fold_tag=f"[NAS Fold {fold_idx}]")
                tr_ds = Subset(dataset, tr_idx.tolist())
                va_ds = Subset(dataset, va_idx.tolist())
                tr_loader = DataLoader(
                    tr_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=collate_motion_batch,
                )
                va_loader = DataLoader(
                    va_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=collate_motion_batch,
                )

                model = ASDPipeline(
                    K_max=int(model_cfg.get("K_max", 8)),
                    d_model=int(model_cfg.get("d_model", 256)),
                    dropout=float(model_cfg.get("dropout", 0.2)),
                    theta_high=float(threshold_cfg.get("decision_high", 0.7)),
                    theta_low=float(threshold_cfg.get("decision_low", 0.3)),
                    use_rgb=use_rgb,
                    rgb_pretrained=bool(model_cfg.get("rgb_pretrained", True)),
                ).to(device)
                model.apply_nas_architecture(arch)
                model.set_use_rgb(use_rgb)
                model.freeze_motion_encoder(train_event_scorer=False)
                model.unfreeze_upper_motion_layers(num_blocks=1)
                model.freeze_rgb_backbone()

                criterion = WeightedBCELoss(
                    pos_weight=WeightedBCELoss.compute_from_labels(labels[tr_idx]),
                    label_smoothing=float(t_cfg.get("label_smoothing", 0.0)),
                    brier_weight=float(t_cfg.get("brier_weight", 0.0)),
                )
                optimizer = torch.optim.AdamW(
                    model.model_parameters(),
                    lr=float(t_cfg.get("arch_lr", 3e-4)),
                    weight_decay=float(t_cfg.get("weight_decay", 1e-4)),
                )
                scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(max(1, nas_epochs)):
                    _train_epoch(model, tr_loader, optimizer, scaler, criterion, device)
                train_time = time.time() - t0

                metrics, infer_time = _evaluate(model, va_loader, device)
                max_mem_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

                fold_aucs.append(float(metrics["auc"]))
                fold_sens.append(float(metrics["sens_at_90_spec"]))
                fold_cal.append(float(metrics["calibration_quality"]))
                fold_train_times.append(float(train_time))
                fold_infer_times.append(float(infer_time))
                fold_mem.append(float(max_mem_gb))
        finally:
            dataset.window_sizes = original_window_sizes

        auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        sens_mean = float(np.mean(fold_sens)) if fold_sens else 0.0
        cal_mean = float(np.mean(fold_cal)) if fold_cal else 0.0
        stability = float(np.clip(1.0 - np.std(fold_aucs), 0.0, 1.0)) if fold_aucs else 0.0
        eff_penalty = _efficiency_penalty(
            train_time=float(np.mean(fold_train_times)) if fold_train_times else 60.0,
            infer_time=float(np.mean(fold_infer_times)) if fold_infer_times else 5.0,
            max_mem_gb=float(np.mean(fold_mem)) if fold_mem else 8.0,
        )
        return {
            "auc": auc_mean,
            "sens_at_90_spec": sens_mean,
            "calibration_quality": cal_mean,
            "cv_stability": stability,
            "efficiency_penalty": eff_penalty,
        }

    searcher = MicroGeneticNAS(
        population_size=pop,
        generations=gens,
        tournament_size=tour,
        mutation_rate=mut,
        crossover=crossover,
        elite_count=elite,
        seed=int(cfg.get("seed", 42)),
    )

    def _on_gen(info):
        best_metrics = info["best_metrics"]
        print(
            f"[NAS] gen={info['generation']} fitness={info['best_fitness']:.4f} "
            f"auc={best_metrics['auc']:.4f} sens90={best_metrics['sens_at_90_spec']:.4f}"
        )
        if logger is not None:
            logger.log(
                "nas_generation",
                generation=int(info["generation"]),
                best_fitness=float(info["best_fitness"]),
                best_auc=float(best_metrics.get("auc", 0.0)),
                best_sens90=float(best_metrics.get("sens_at_90_spec", 0.0)),
                best_calibration=float(best_metrics.get("calibration_quality", 0.0)),
                best_stability=float(best_metrics.get("cv_stability", 0.0)),
                best_efficiency_penalty=float(best_metrics.get("efficiency_penalty", 1.0)),
                best_metrics=best_metrics,
                best_architecture=info["best_architecture"],
            )

    return searcher.evolve(
        evaluate_fn=evaluate_arch,
        search_space=search_space,
        on_generation_end=_on_gen,
    )
