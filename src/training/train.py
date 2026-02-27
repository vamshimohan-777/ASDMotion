import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.pipeline_model import ASDPipeline
from src.pipeline.preprocess import precompute_videos
from src.training.checkpoints import CheckpointManager
from src.training.dataset import VideoDataset, collate_motion_batch
from src.training.logging_utils import ExperimentLogger, export_experiment_log_pdf
from src.training.losses import WeightedBCELoss
from src.training.motion_ssl import pretrain_motion_encoder
from src.training.nas_search import run_micro_genetic_nas
from src.utils.calibration import apply_temperature, fit_temperature
from src.utils.config import apply_overrides, load_config
from src.utils.metrics import (
    compute_auc,
    compute_basic_metrics,
    compute_ece,
    find_optimal_threshold,
    sensitivity_at_specificity,
)
from src.utils.seed import seed_everything, seed_worker
from src.utils.splits import check_group_overlap, make_group_kfold, make_group_stratified_split


def _write_status_file(status_file: str, payload: dict):
    if not status_file:
        return
    directory = os.path.dirname(status_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{status_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, status_file)


def _is_rtx_4050(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(device.index if device.index is not None else 0)
    except Exception:
        return False
    return "rtx 4050" in str(props.name).lower()


def _auto_batch_and_workers(cfg, device):
    train_cfg = cfg.setdefault("training", {})
    data_cfg = cfg.setdefault("data", {})
    if not _is_rtx_4050(device):
        return
    if "rtx4050_batch_size" in train_cfg:
        train_cfg["batch_size"] = int(train_cfg["rtx4050_batch_size"])
    if "rtx4050_num_workers" in data_cfg:
        data_cfg["num_workers"] = int(data_cfg["rtx4050_num_workers"])


def _build_dataset(cfg, is_training):
    data_cfg = cfg.get("data", {})
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


def _build_loader(dataset, cfg, shuffle, generator=None):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 2)),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": collate_motion_batch,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def _to_inputs(batch, device):
    return {
        "motion_windows": batch["motion_windows"].to(device, non_blocking=True),
        "joint_mask": batch["joint_mask"].to(device, non_blocking=True),
        "window_timestamps": batch["window_timestamps"].to(device, non_blocking=True),
    }


def _quality_score_from_batch(batch):
    q = batch["qualities"]
    face = q["face_score"].float()
    pose = q["pose_score"].float()
    hand = q["hand_score"].float()
    score = 0.45 * pose + 0.30 * hand + 0.25 * face
    return score.clamp(0.0, 1.0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, clip_grad=1.0):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        inputs = _to_inputs(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
            out = model(inputs)
            loss = criterion(out["logit_final"], labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    logits = []
    labels = []
    qualities = []

    for batch in loader:
        y = batch["label"].to(device, non_blocking=True)
        x = _to_inputs(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
            out = model(x)
            loss = criterion(out["logit_final"], y)
            logit = out["logit_final"]
        total += float(loss.item())
        n += 1
        logits.append(logit.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        qualities.append(_quality_score_from_batch(batch).detach().cpu().numpy())

    if logits:
        logits = np.concatenate(logits).astype(float)
        labels = np.concatenate(labels).astype(int)
        qualities = np.concatenate(qualities).astype(float)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    else:
        logits = np.array([])
        labels = np.array([])
        qualities = np.array([])
        probs = np.array([])

    return {
        "loss": total / max(n, 1),
        "logits": logits,
        "labels": labels,
        "probs": probs,
        "quality": qualities,
    }


def summarize_metrics(labels, probs, spec_target=0.90):
    if labels.size == 0:
        return {
            "auc": 0.5,
            "ece": 1.0,
            "calibration_quality": 0.0,
            "sens_at_90_spec": 0.0,
            "f1_opt": 0.0,
            "acc_opt": 0.0,
            "opt_threshold": 0.5,
        }

    auc = compute_auc(labels, probs)
    ece = compute_ece(labels, probs, n_bins=10)
    cal_quality = float(np.clip(1.0 - ece, 0.0, 1.0))
    sens = sensitivity_at_specificity(
        labels, probs, target_spec=float(spec_target), min_negatives=10, allow_unstable=True
    )
    if not np.isfinite(sens):
        sens = 0.0
    thr = find_optimal_threshold(labels, probs)
    basic = compute_basic_metrics(labels, probs, threshold=thr)
    return {
        "auc": float(auc),
        "ece": float(ece),
        "calibration_quality": cal_quality,
        "sens_at_90_spec": float(max(0.0, sens)),
        "f1_opt": float(basic["f1"]),
        "acc_opt": float(basic["accuracy"]),
        "opt_threshold": float(thr),
    }


def _selection_score(metrics):
    return (
        0.40 * float(metrics.get("auc", 0.0))
        + 0.25 * float(metrics.get("sens_at_90_spec", 0.0))
        + 0.15 * float(metrics.get("calibration_quality", 0.0))
        + 0.10 * float(metrics.get("f1_opt", 0.0))
        + 0.10 * float(metrics.get("acc_opt", 0.0))
    )


def _build_model_from_cfg(cfg):
    model_cfg = cfg.get("model", {})
    thresholds = cfg.get("thresholds", {})
    return ASDPipeline(
        K_max=int(model_cfg.get("K_max", 16)),
        d_model=int(model_cfg.get("d_model", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        theta_high=float(thresholds.get("decision_high", 0.7)),
        theta_low=float(thresholds.get("decision_low", 0.3)),
    )


def _load_pretrained_motion_encoder(model, checkpoint_path, device):
    if not checkpoint_path:
        return False
    if not os.path.exists(checkpoint_path):
        print(f"[Train] pretrained motion encoder checkpoint not found: {checkpoint_path}")
        return False

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = None
    if isinstance(ckpt, dict) and "motion_encoder" in ckpt:
        state = ckpt["motion_encoder"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        ms = ckpt["model_state"]
        state = {}
        for k, v in ms.items():
            if str(k).startswith("motion_encoder."):
                state[str(k).replace("motion_encoder.", "", 1)] = v
    if not state:
        print("[Train] checkpoint does not contain motion encoder weights.")
        return False

    missing, unexpected = model.motion_encoder.load_state_dict(state, strict=False)
    print(
        f"[Train] loaded motion encoder init from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return True


def train(cfg, status_file=None):
    seed = int(cfg.get("seed", 42))
    generator = seed_everything(seed, deterministic=False)
    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[Train] device={device}")
    _auto_batch_and_workers(cfg, device)

    results_dir = cfg.get("reporting", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_mgr = CheckpointManager(root_dir=results_dir)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"experiment_log_{run_tag}.jsonl")
    logger = ExperimentLogger(log_path)
    logger.log("train_start", config=cfg)
    started = time.time()

    def emit_status(phase, **fields):
        payload = {
            "phase": phase,
            "time": int(time.time()),
            "elapsed_sec": round(time.time() - started, 1),
        }
        payload.update(fields)
        _write_status_file(status_file, payload)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    nas_cfg = cfg.get("nas", {})

    # Stage 1-4: optional precompute numeric landmarks.
    if bool(data_cfg.get("preprocess_videos", False)):
        emit_status("preprocess", state="starting")
        precompute_videos(
            csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
            processed_root=str(data_cfg.get("processed_root", "data/processed")),
            frame_stride=int(data_cfg.get("frame_stride", 1)),
            max_frames=int(data_cfg.get("max_frames", 0)),
            overwrite=bool(data_cfg.get("preprocess_overwrite", False)),
            progress_every=int(data_cfg.get("precompute_progress_every", 10)),
            status_callback=lambda s: emit_status("preprocess", state="running", **s),
        )
        emit_status("preprocess", state="done")

    # Build datasets.
    ds_train_view = _build_dataset(cfg, is_training=True)
    ds_eval_view = _build_dataset(cfg, is_training=False)
    labels = np.asarray([float(e["label"]) for e in ds_train_view.entries], dtype=np.float32)
    groups = np.asarray([e["subject_id"] for e in ds_train_view.entries], dtype=object)
    print(f"[Train] samples={len(labels)} positives={int(labels.sum())} negatives={int((labels==0).sum())}")
    logger.log(
        "dataset_summary",
        samples=int(len(labels)),
        positives=int(labels.sum()),
        negatives=int((labels == 0).sum()),
    )

    # Optional init from existing motion-encoder checkpoint.
    base_model = _build_model_from_cfg(cfg).to(device)
    pretrained_encoder_ckpt = str(train_cfg.get("pretrained_motion_encoder_checkpoint", "")).strip()
    _load_pretrained_motion_encoder(base_model, pretrained_encoder_ckpt, device)

    # Stage 6: self-supervised motion pretraining.
    ssl_epochs = int(train_cfg.get("ssl_pretrain_epochs", 0))
    if ssl_epochs > 0:
        ssl_loader = _build_loader(ds_train_view, cfg, shuffle=True, generator=generator)
        emit_status("ssl_pretrain", state="starting", epochs=ssl_epochs)
        pretrain_motion_encoder(
            base_model,
            ssl_loader,
            device=device,
            epochs=ssl_epochs,
            lr=float(train_cfg.get("ssl_lr", 1e-4)),
            max_steps_per_epoch=int(train_cfg.get("ssl_steps_per_epoch", 300)),
            logger=logger,
        )
        ckpt_mgr.save_model(
            "motion_encoder_pretrained.pth",
            {"motion_encoder": base_model.motion_encoder.state_dict(), "config": cfg},
        )
        emit_status("ssl_pretrain", state="done")

    # Stage 8: micro-genetic NAS.
    best_arch = None
    if bool(nas_cfg.get("enabled", True)):
        emit_status("nas", state="starting")
        logger.log("nas_start")
        nas_result = run_micro_genetic_nas(
            cfg=cfg,
            dataset=ds_train_view,
            labels=labels,
            groups=groups,
            device=device,
            logger=logger,
        )
        best_arch = nas_result["best_architecture"]
        ckpt_mgr.save_json("nas_architecture.json", nas_result)
        logger.log("nas_final", **nas_result)
        emit_status(
            "nas",
            state="done",
            best_fitness=float(nas_result["best_fitness"]),
            best_metrics=nas_result["best_metrics"],
        )
    else:
        best_arch = None
        logger.log("nas_skipped")

    # Stage 9-11: grouped CV supervised training and fine-tuning.
    cv_folds = int(train_cfg.get("cv_folds", 5))
    folds = make_group_kfold(labels.astype(int), groups, n_splits=cv_folds, seed=seed)
    fold_summaries = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        emit_status("train_fold", state="starting", fold=fold_idx, total_folds=len(folds))
        check_group_overlap(groups[tr_idx], groups[va_idx], fold_tag=f"[Fold {fold_idx}]")

        tr_loader = _build_loader(Subset(ds_train_view, tr_idx.tolist()), cfg, shuffle=True, generator=generator)
        va_loader = _build_loader(Subset(ds_eval_view, va_idx.tolist()), cfg, shuffle=False, generator=None)

        model = _build_model_from_cfg(cfg).to(device)
        if best_arch is not None:
            model.apply_nas_architecture(best_arch)
        try:
            model.motion_encoder.load_state_dict(base_model.motion_encoder.state_dict(), strict=True)
        except Exception:
            pass

        # Stage 7: freeze backbone for stable NAS-supervised training.
        model.freeze_motion_encoder()
        criterion = WeightedBCELoss(
            pos_weight=WeightedBCELoss.compute_from_labels(labels[tr_idx]),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
            brier_weight=float(train_cfg.get("brier_weight", 0.0)),
        )
        optimizer = torch.optim.AdamW(
            model.model_parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

        epochs = int(train_cfg.get("epochs", 20))
        patience = int(train_cfg.get("patience", 8))
        best_score = -1e9
        best_payload = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            tr_loss = train_one_epoch(
                model,
                tr_loader,
                criterion,
                optimizer,
                scaler,
                device,
                clip_grad=float(train_cfg.get("clip_grad", 1.0)),
            )
            ev = evaluate(model, va_loader, criterion, device)
            m = summarize_metrics(ev["labels"], ev["probs"], spec_target=0.90)
            score = _selection_score(m)
            logger.log(
                "train_fold_epoch",
                fold=fold_idx,
                epoch=epoch,
                train_loss=float(tr_loss),
                val_loss=float(ev["loss"]),
                metrics=m,
                score=float(score),
            )
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
            print(
                f"[Fold {fold_idx}] epoch={epoch}/{epochs} "
                f"train_loss={tr_loss:.4f} val_loss={ev['loss']:.4f} "
                f"auc={m['auc']:.4f} sens90={m['sens_at_90_spec']:.4f} score={score:.4f}"
            )
            if score > best_score:
                best_score = score
                bad_epochs = 0
                best_payload = {
                    "model_state": model.state_dict(),
                    "best_metrics": m,
                }
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break

        if best_payload is not None:
            model.load_state_dict(best_payload["model_state"])

        # Stage 11: joint fine-tuning by unfreezing upper encoder layers.
        finetune_epochs = int(train_cfg.get("finetune_epochs", 4))
        if finetune_epochs > 0:
            model.unfreeze_upper_motion_layers(num_blocks=int(train_cfg.get("finetune_unfreeze_blocks", 1)))
            ft_optimizer = torch.optim.AdamW(
                model.trainable_parameters(),
                lr=float(train_cfg.get("finetune_lr", 2e-5)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            )
            for ft_epoch in range(1, finetune_epochs + 1):
                ft_loss = train_one_epoch(
                    model,
                    tr_loader,
                    criterion,
                    ft_optimizer,
                    scaler,
                    device,
                    clip_grad=float(train_cfg.get("clip_grad", 1.0)),
                )
                logger.log("finetune_fold_epoch", fold=fold_idx, epoch=ft_epoch, train_loss=float(ft_loss))

        ev = evaluate(model, va_loader, criterion, device)
        temp = fit_temperature(
            torch.tensor(ev["logits"], device=device, dtype=torch.float32),
            torch.tensor(ev["labels"], device=device, dtype=torch.float32),
            device=device,
        )
        logits_cal = apply_temperature(torch.tensor(ev["logits"], dtype=torch.float32), temp).cpu().numpy()
        probs_cal = 1.0 / (1.0 + np.exp(-np.clip(logits_cal, -40.0, 40.0)))
        metrics_cal = summarize_metrics(ev["labels"], probs_cal, spec_target=0.90)
        fold_summaries.append(metrics_cal)

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
        emit_status("train_fold", state="done", fold=fold_idx, total_folds=len(folds), metrics=metrics_cal)

    # Cross-val summary.
    aucs = [m["auc"] for m in fold_summaries] if fold_summaries else [0.5]
    sens = [m["sens_at_90_spec"] for m in fold_summaries] if fold_summaries else [0.0]
    cv_summary = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "sens90_mean": float(np.mean(sens)),
        "sens90_std": float(np.std(sens)),
        "cv_stability": float(np.clip(1.0 - np.std(aucs), 0.0, 1.0)),
    }
    ckpt_mgr.save_json("cv_summary.json", cv_summary)
    logger.log("cv_summary", **cv_summary)

    # Final model training on full set with grouped holdout for calibration.
    emit_status("final_train", state="starting")
    tr_idx, va_idx, _ = make_group_stratified_split(
        labels.astype(int),
        groups,
        val_fraction=float(train_cfg.get("final_val_fraction", 0.2)),
        seed=seed,
    )
    tr_loader = _build_loader(Subset(ds_train_view, tr_idx.tolist()), cfg, shuffle=True, generator=generator)
    va_loader = _build_loader(Subset(ds_eval_view, va_idx.tolist()), cfg, shuffle=False, generator=None)

    model = _build_model_from_cfg(cfg).to(device)
    if best_arch is not None:
        model.apply_nas_architecture(best_arch)
    try:
        model.motion_encoder.load_state_dict(base_model.motion_encoder.state_dict(), strict=True)
    except Exception:
        pass
    model.freeze_motion_encoder()

    criterion = WeightedBCELoss(
        pos_weight=WeightedBCELoss.compute_from_labels(labels[tr_idx]),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        brier_weight=float(train_cfg.get("brier_weight", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.model_parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

    final_epochs = int(train_cfg.get("final_epochs", 24))
    for epoch in range(1, final_epochs + 1):
        loss = train_one_epoch(
            model,
            tr_loader,
            criterion,
            optimizer,
            scaler,
            device,
            clip_grad=float(train_cfg.get("clip_grad", 1.0)),
        )
        logger.log("final_train_epoch", epoch=epoch, train_loss=float(loss))
        if epoch % 5 == 0 or epoch == final_epochs:
            emit_status("final_train", state="epoch", epoch=epoch, total_epochs=final_epochs, train_loss=float(loss))

    # Final gentle joint fine-tuning.
    model.unfreeze_upper_motion_layers(num_blocks=int(train_cfg.get("finetune_unfreeze_blocks", 1)))
    ft_epochs = int(train_cfg.get("final_finetune_epochs", 4))
    if ft_epochs > 0:
        ft_opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=float(train_cfg.get("finetune_lr", 2e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        for epoch in range(1, ft_epochs + 1):
            loss = train_one_epoch(
                model,
                tr_loader,
                criterion,
                ft_opt,
                scaler,
                device,
                clip_grad=float(train_cfg.get("clip_grad", 1.0)),
            )
            logger.log("final_finetune_epoch", epoch=epoch, train_loss=float(loss))

    ev = evaluate(model, va_loader, criterion, device)
    temp = fit_temperature(
        torch.tensor(ev["logits"], device=device, dtype=torch.float32),
        torch.tensor(ev["labels"], device=device, dtype=torch.float32),
        device=device,
    )
    logits_cal = apply_temperature(torch.tensor(ev["logits"], dtype=torch.float32), temp).cpu().numpy()
    probs_cal = 1.0 / (1.0 + np.exp(-np.clip(logits_cal, -40.0, 40.0)))
    final_metrics = summarize_metrics(ev["labels"], probs_cal, spec_target=0.90)
    final_metrics["temperature"] = float(temp)

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
    ckpt_mgr.save_json("final_metrics.json", final_metrics)
    logger.log("final_metrics", **final_metrics)
    emit_status("done", state="completed", final_model=final_path, final_metrics=final_metrics)
    logger.log("train_done", final_model=final_path, final_metrics=final_metrics)

    pdf_report_path = os.path.join(results_dir, f"training_log_report_{run_tag}.pdf")
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
    print(f"[Train] final model saved: {final_path}")
    print(f"[Train] log jsonl: {log_path}")
    print(f"[Train] log report pdf: {pdf_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ASD landmark-motion pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, action="append", default=[])
    parser.add_argument("--csv", type=str, default=None, help="Override data.csv_path")
    parser.add_argument("--status-file", type=str, default=None, help="Optional status JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.csv:
        args.override.append(f"data.csv_path={args.csv}")
    cfg = apply_overrides(cfg, args.override)
    train(cfg, status_file=args.status_file)


if __name__ == "__main__":
    main()
