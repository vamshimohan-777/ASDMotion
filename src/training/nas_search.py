"""Training module `src/training/nas_search.py` that optimizes model weights and output quality."""

# Import `copy` to support computations in this stage of output generation.
import copy
# Import `time` to support computations in this stage of output generation.
import time

# Import `numpy as np` to support computations in this stage of output generation.
import numpy as np
# Import `torch` to support computations in this stage of output generation.
import torch
# Import symbols from `torch.utils.data` used in this stage's output computation path.
from torch.utils.data import DataLoader, Subset

# Import symbols from `src.models.nas_controller` used in this stage's output computation path.
from src.models.nas_controller import MicroGeneticNAS, default_search_space
# Import symbols from `src.models.pipeline_model` used in this stage's output computation path.
from src.models.pipeline_model import ASDPipeline
# Import symbols from `src.training.dataset` used in this stage's output computation path.
from src.training.dataset import collate_motion_batch
# Import symbols from `src.training.losses` used in this stage's output computation path.
from src.training.losses import WeightedBCELoss, event_gate_bag_loss
# Import symbols from `src.utils.metrics` used in this stage's output computation path.
from src.utils.metrics import compute_auc, compute_ece, sensitivity_at_specificity
# Import symbols from `src.utils.splits` used in this stage's output computation path.
from src.utils.splits import make_group_kfold, check_group_overlap


# Define a reusable pipeline function whose outputs feed later steps.
def _to_device(batch, device):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "motion_windows": batch["motion_windows"].to(device, non_blocking=True),
        "joint_mask": batch["joint_mask"].to(device, non_blocking=True),
        "window_timestamps": batch["window_timestamps"].to(device, non_blocking=True),
    }


# Define a training routine that updates parameters and changes future outputs.
def _train_epoch(model, loader, optimizer, scaler, criterion, device, gate_aux_weight=0.0):
    """Executes a training step/loop that updates parameters and directly changes model output behavior."""
    # Call `model.train` and use its result in later steps so gradient updates improve future predictions.
    model.train()
    # Set `total` for subsequent steps so gradient updates improve future predictions.
    total = 0.0
    # Set `n` for subsequent steps so gradient updates improve future predictions.
    n = 0
    # Iterate over `loader` so each item contributes to final outputs/metrics.
    for batch in loader:
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = batch["label"].to(device, non_blocking=True)
        # Compute `x` as an intermediate representation used by later output layers.
        x = _to_device(batch, device)
        # Use a managed context to safely handle resources used during output computation.
        with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
            # Set `out` for subsequent steps so gradient updates improve future predictions.
            out = model(x)
            # Update `cls_loss` with a loss term that drives backpropagation and output improvement.
            cls_loss = criterion(out["logit_final"], y)
            # Update `gate_aux` with a loss term that drives backpropagation and output improvement.
            gate_aux = cls_loss.new_tensor(0.0)
            # Branch on `float(gate_aux_weight) > 0.0 and out.get("frame_e...` to choose the correct output computation path.
            if float(gate_aux_weight) > 0.0 and out.get("frame_event_scores") is not None:
                # Keep NAS scoring aligned with final training objective.
                # Update `gate_aux` with a loss term that drives backpropagation and output improvement.
                gate_aux = event_gate_bag_loss(
                    frame_event_scores=out["frame_event_scores"],
                    target=y,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
def _evaluate(model, loader, device):
    """Computes validation metrics used to judge model quality and influence training decisions."""
    # Call `model.eval` and use its result in later steps so gradient updates improve future predictions.
    model.eval()
    # Compute `probs` as confidence values used in final prediction decisions.
    probs = []
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = []
    # Set `started` for subsequent steps so gradient updates improve future predictions.
    started = time.time()
    # Iterate over `loader` so each item contributes to final outputs/metrics.
    for batch in loader:
        # Set `y` for subsequent steps so gradient updates improve future predictions.
        y = batch["label"].to(device, non_blocking=True)
        # Compute `x` as an intermediate representation used by later output layers.
        x = _to_device(batch, device)
        # Set `out` for subsequent steps so gradient updates improve future predictions.
        out = model(x)
        # Store raw score tensor in `p` before probability/decision conversion.
        p = torch.sigmoid(out["logit_final"])
        # Call `probs.append` and use its result in later steps so gradient updates improve future predictions.
        probs.append(p.detach().cpu().numpy())
        # Call `labels.append` and use its result in later steps so gradient updates improve future predictions.
        labels.append(y.detach().cpu().numpy())
    # Set `infer_time` for subsequent steps so gradient updates improve future predictions.
    infer_time = time.time() - started
    # Branch on `not probs` to choose the correct output computation path.
    if not probs:
        # Return `{"auc": 0.5, "sens_at_90_spec": 0.0, "calibration_q...` as this function's contribution to downstream output flow.
        return {"auc": 0.5, "sens_at_90_spec": 0.0, "calibration_quality": 0.0}, infer_time
    # Compute `probs` as confidence values used in final prediction decisions.
    probs = np.concatenate(probs).astype(float)
    # Set `labels` for subsequent steps so gradient updates improve future predictions.
    labels = np.concatenate(labels).astype(int)
    # Record `auc` as a metric describing current output quality.
    auc = compute_auc(labels, probs)
    # Record `sens90` as a metric describing current output quality.
    sens90 = sensitivity_at_specificity(labels, probs, target_spec=0.90, min_negatives=10, allow_unstable=True)
    # Branch on `not np.isfinite(sens90)` to choose the correct output computation path.
    if not np.isfinite(sens90):
        # Record `sens90` as a metric describing current output quality.
        sens90 = 0.0
    # Record `ece` as a metric describing current output quality.
    ece = compute_ece(labels, probs, n_bins=10)
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "auc": float(auc),
        "sens_at_90_spec": float(max(0.0, sens90)),
        "calibration_quality": float(np.clip(1.0 - ece, 0.0, 1.0)),
    }, infer_time


# Define a reusable pipeline function whose outputs feed later steps.
def _efficiency_penalty(train_time, infer_time, max_mem_gb):
    # Normalize into [0,1] where higher means worse.
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `t_pen` for subsequent steps so gradient updates improve future predictions.
    t_pen = np.clip(train_time / 60.0, 0.0, 1.0)
    # Set `i_pen` for subsequent steps so gradient updates improve future predictions.
    i_pen = np.clip(infer_time / 5.0, 0.0, 1.0)
    # Set `m_pen` for subsequent steps so gradient updates improve future predictions.
    m_pen = np.clip(max_mem_gb / 6.0, 0.0, 1.0)
    # Return `float(np.clip(0.45 * m_pen + 0.35 * t_pen + 0.20 * ...` as this function's contribution to downstream output flow.
    return float(np.clip(0.45 * m_pen + 0.35 * t_pen + 0.20 * i_pen, 0.0, 1.0))


# Define a reusable pipeline function whose outputs feed later steps.
def run_micro_genetic_nas(cfg, dataset, labels, groups, device, logger=None):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `nas_cfg` for subsequent steps so gradient updates improve future predictions.
    nas_cfg = cfg.get("nas", {})
    # Set `t_cfg` for subsequent steps so gradient updates improve future predictions.
    t_cfg = cfg.get("training", {})
    # Set `data_cfg` for subsequent steps so gradient updates improve future predictions.
    data_cfg = cfg.get("data", {})
    # Set `model_cfg` for subsequent steps so gradient updates improve future predictions.
    model_cfg = cfg.get("model", {})
    # Compute `threshold_cfg` as an intermediate representation used by later output layers.
    threshold_cfg = cfg.get("thresholds", {})

    # Set `pop` for subsequent steps so gradient updates improve future predictions.
    pop = int(nas_cfg.get("population", 20))
    # Set `gens` for subsequent steps so gradient updates improve future predictions.
    gens = int(nas_cfg.get("generations", 20))
    # Set `tour` for subsequent steps so gradient updates improve future predictions.
    tour = int(nas_cfg.get("tournament_size", 3))
    # Set `mut` for subsequent steps so gradient updates improve future predictions.
    mut = float(nas_cfg.get("mutation_rate", 0.15))
    # Set `crossover` for subsequent steps so gradient updates improve future predictions.
    crossover = bool(nas_cfg.get("crossover", True))
    # Set `elite` for subsequent steps so gradient updates improve future predictions.
    elite = int(nas_cfg.get("elite", 2))
    # Compute `nas_epochs` as an intermediate representation used by later output layers.
    nas_epochs = int(nas_cfg.get("epochs", 2))
    # Set `nas_folds` for subsequent steps so gradient updates improve future predictions.
    nas_folds = int(nas_cfg.get("eval_folds", 3))
    # Compute `gate_aux_weight` as an intermediate representation used by later output layers.
    gate_aux_weight = float(t_cfg.get("event_gate_aux_weight", 0.1))
    # Compute `batch_size` as an intermediate representation used by later output layers.
    batch_size = int(t_cfg.get("batch_size", 2))
    # Set `num_workers` for subsequent steps so gradient updates improve future predictions.
    num_workers = int(data_cfg.get("num_workers", 0))

    # Compute `search_space` as an intermediate representation used by later output layers.
    search_space = copy.deepcopy(nas_cfg.get("search_space")) if nas_cfg.get("search_space") else default_search_space()

    # Set `all_folds` for subsequent steps so gradient updates improve future predictions.
    all_folds = make_group_kfold(labels, groups, n_splits=max(2, int(t_cfg.get("cv_folds", 5))), seed=int(cfg.get("seed", 42)))
    # Set `folds` for subsequent steps so gradient updates improve future predictions.
    folds = all_folds[: max(1, min(len(all_folds), nas_folds))]

    # Define evaluation logic used to measure prediction quality.
    def evaluate_arch(arch):
        """Computes validation metrics used to judge model quality and influence training decisions."""
        # Record `fold_aucs` as a metric describing current output quality.
        fold_aucs = []
        # Record `fold_sens` as a metric describing current output quality.
        fold_sens = []
        # Set `fold_cal` for subsequent steps so gradient updates improve future predictions.
        fold_cal = []
        # Set `fold_train_times` for subsequent steps so gradient updates improve future predictions.
        fold_train_times = []
        # Set `fold_infer_times` for subsequent steps so gradient updates improve future predictions.
        fold_infer_times = []
        # Set `fold_mem` for subsequent steps so gradient updates improve future predictions.
        fold_mem = []

        # Iterate over `enumerate(folds, start=1)` so each item contributes to final outputs/metrics.
        for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
            # Call `check_group_overlap` and use its result in later steps so gradient updates improve future predictions.
            check_group_overlap(groups[tr_idx], groups[va_idx], fold_tag=f"[NAS Fold {fold_idx}]")
            # Set `tr_ds` for subsequent steps so gradient updates improve future predictions.
            tr_ds = Subset(dataset, tr_idx.tolist())
            # Set `va_ds` for subsequent steps so gradient updates improve future predictions.
            va_ds = Subset(dataset, va_idx.tolist())
            # Set `tr_loader` for subsequent steps so gradient updates improve future predictions.
            tr_loader = DataLoader(
                tr_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_motion_batch,
            )
            # Set `va_loader` for subsequent steps so gradient updates improve future predictions.
            va_loader = DataLoader(
                va_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_motion_batch,
            )

            # Set `model` for subsequent steps so gradient updates improve future predictions.
            model = ASDPipeline(
                K_max=int(model_cfg.get("K_max", 16)),
                d_model=int(model_cfg.get("d_model", 256)),
                dropout=float(model_cfg.get("dropout", 0.2)),
                theta_high=float(threshold_cfg.get("decision_high", 0.7)),
                theta_low=float(threshold_cfg.get("decision_low", 0.3)),
            ).to(device)
            # Call `model.apply_nas_architecture` and use its result in later steps so gradient updates improve future predictions.
            model.apply_nas_architecture(arch)
            # Call `model.freeze_motion_encoder` and use its result in later steps so gradient updates improve future predictions.
            model.freeze_motion_encoder()

            # Update `criterion` with a loss term that drives backpropagation and output improvement.
            criterion = WeightedBCELoss(
                pos_weight=WeightedBCELoss.compute_from_labels(labels[tr_idx]),
                label_smoothing=float(t_cfg.get("label_smoothing", 0.0)),
                brier_weight=float(t_cfg.get("brier_weight", 0.0)),
            )
            # Initialize `optimizer` to control parameter updates during training.
            optimizer = torch.optim.AdamW(
                model.model_parameters(),
                lr=float(t_cfg.get("arch_lr", 3e-4)),
                weight_decay=float(t_cfg.get("weight_decay", 1e-4)),
            )
            # Set `scaler` for subsequent steps so gradient updates improve future predictions.
            scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

            # Branch on `torch.cuda.is_available()` to choose the correct output computation path.
            if torch.cuda.is_available():
                # Call `torch.cuda.reset_peak_memory_stats` and use its result in later steps so gradient updates improve future predictions.
                torch.cuda.reset_peak_memory_stats()

            # Set `t0` for subsequent steps so gradient updates improve future predictions.
            t0 = time.time()
            # Iterate over `range(max(1, nas_epochs))` so each item contributes to final outputs/metrics.
            for _ in range(max(1, nas_epochs)):
                # Call `_train_epoch` and use its result in later steps so gradient updates improve future predictions.
                _train_epoch(
                    model,
                    tr_loader,
                    optimizer,
                    scaler,
                    criterion,
                    device,
                    gate_aux_weight=gate_aux_weight,
                )
            # Set `train_time` for subsequent steps so gradient updates improve future predictions.
            train_time = time.time() - t0
            # Record `metrics, infer_time` as a metric describing current output quality.
            metrics, infer_time = _evaluate(model, va_loader, device)
            # Compute `max_mem_gb` as an intermediate representation used by later output layers.
            max_mem_gb = 0.0
            # Branch on `torch.cuda.is_available()` to choose the correct output computation path.
            if torch.cuda.is_available():
                # Compute `max_mem_gb` as an intermediate representation used by later output layers.
                max_mem_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))

            # Call `fold_aucs.append` and use its result in later steps so gradient updates improve future predictions.
            fold_aucs.append(float(metrics["auc"]))
            # Call `fold_sens.append` and use its result in later steps so gradient updates improve future predictions.
            fold_sens.append(float(metrics["sens_at_90_spec"]))
            # Call `fold_cal.append` and use its result in later steps so gradient updates improve future predictions.
            fold_cal.append(float(metrics["calibration_quality"]))
            # Call `fold_train_times.append` and use its result in later steps so gradient updates improve future predictions.
            fold_train_times.append(float(train_time))
            # Call `fold_infer_times.append` and use its result in later steps so gradient updates improve future predictions.
            fold_infer_times.append(float(infer_time))
            # Call `fold_mem.append` and use its result in later steps so gradient updates improve future predictions.
            fold_mem.append(float(max_mem_gb))

        # Record `auc_mean` as a metric describing current output quality.
        auc_mean = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        # Record `sens_mean` as a metric describing current output quality.
        sens_mean = float(np.mean(fold_sens)) if fold_sens else 0.0
        # Set `cal_mean` for subsequent steps so gradient updates improve future predictions.
        cal_mean = float(np.mean(fold_cal)) if fold_cal else 0.0
        # Set `stability` for subsequent steps so gradient updates improve future predictions.
        stability = float(np.clip(1.0 - np.std(fold_aucs), 0.0, 1.0)) if fold_aucs else 0.0
        # Set `eff_penalty` for subsequent steps so gradient updates improve future predictions.
        eff_penalty = _efficiency_penalty(
            train_time=float(np.mean(fold_train_times)) if fold_train_times else 60.0,
            infer_time=float(np.mean(fold_infer_times)) if fold_infer_times else 5.0,
            max_mem_gb=float(np.mean(fold_mem)) if fold_mem else 8.0,
        )

        # Return `{` as this function's contribution to downstream output flow.
        return {
            "auc": auc_mean,
            "sens_at_90_spec": sens_mean,
            "calibration_quality": cal_mean,
            "cv_stability": stability,
            "efficiency_penalty": eff_penalty,
        }

    # Compute `searcher` as an intermediate representation used by later output layers.
    searcher = MicroGeneticNAS(
        population_size=pop,
        generations=gens,
        tournament_size=tour,
        mutation_rate=mut,
        crossover=crossover,
        elite_count=elite,
        seed=int(cfg.get("seed", 42)),
    )

    # Define a reusable pipeline function whose outputs feed later steps.
    def _on_gen(info):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Record `best_metrics` as a metric describing current output quality.
        best_metrics = info["best_metrics"]
        # Log runtime values to verify that output computation is behaving as expected.
        print(
            f"[NAS] gen={info['generation']} fitness={info['best_fitness']:.4f} "
            f"auc={best_metrics['auc']:.4f} sens90={best_metrics['sens_at_90_spec']:.4f}"
        )
        # Branch on `logger is not None` to choose the correct output computation path.
        if logger is not None:
            # Log runtime values to verify that output computation is behaving as expected.
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

    # Set `result` for subsequent steps so gradient updates improve future predictions.
    result = searcher.evolve(
        evaluate_fn=evaluate_arch,
        search_space=search_space,
        on_generation_end=_on_gen,
    )
    # Return `result` as this function's contribution to downstream output flow.
    return result
