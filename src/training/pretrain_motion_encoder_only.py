"""
Pretrain only the motion encoder (self-supervised) on generic motion data.

Usage:
  python -m src.training.pretrain_motion_encoder_only --config config.yaml --csv data/videos.csv
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.pipeline_model import ASDPipeline
from src.pipeline.preprocess import precompute_videos
from src.training.dataset import VideoDataset, collate_motion_batch
from src.training.logging_utils import ExperimentLogger, export_experiment_log_pdf
from src.training.motion_ssl import pretrain_motion_encoder
from src.utils.config import apply_overrides, load_config
from src.utils.seed import seed_everything, seed_worker


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


def _load_pretrained_motion_encoder(model, checkpoint_path, device):
    if not checkpoint_path:
        return False
    if not os.path.exists(checkpoint_path):
        print(f"[PretrainOnly] Pretrained checkpoint not found: {checkpoint_path}")
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
        print("[PretrainOnly] Checkpoint does not contain motion encoder weights.")
        return False

    missing, unexpected = model.motion_encoder.load_state_dict(state, strict=False)
    print(
        f"[PretrainOnly] Loaded motion encoder init from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})."
    )
    return True


def pretrain_motion_encoder_action_type(
    model,
    loader,
    device,
    num_classes,
    epochs=8,
    lr=1e-4,
    max_steps_per_epoch=300,
    logger=None,
):
    model.train()
    for p in model.motion_encoder.parameters():
        p.requires_grad = True

    d_emb = int(model.motion_encoder.embedding_dim)
    action_head = nn.Sequential(
        nn.Linear(d_emb, d_emb),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(d_emb, int(num_classes)),
    ).to(device)

    params = list(model.motion_encoder.parameters()) + list(action_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))
    criterion = nn.CrossEntropyLoss()

    summary = {"mode": "action_type", "epochs": int(epochs), "history": []}
    for epoch in range(1, int(epochs) + 1):
        total_loss = 0.0
        total = 0
        correct = 0
        n_steps = 0

        for batch in loader:
            motion = batch["motion_windows"].to(device, non_blocking=True)  # [B,S,W,J,F]
            joint_mask = batch["joint_mask"].to(device, non_blocking=True)  # [B,S,W,J]
            action_id = batch["action_id"].to(device, non_blocking=True).long()  # [B]

            valid = action_id >= 0
            if int(valid.sum().item()) == 0:
                continue

            b, s, w, j, f = motion.shape
            motion_flat = motion.reshape(b * s, w, j, f)
            mask_flat = joint_mask.reshape(b * s, w, j)

            with torch.amp.autocast(device_type=device.type, enabled=str(device).startswith("cuda")):
                win_emb = model.motion_encoder(motion_flat, joint_mask=mask_flat).reshape(b, s, -1)
                video_emb = win_emb.mean(dim=1)
                logits = action_head(video_emb)
                loss = criterion(logits[valid], action_id[valid])

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            preds = logits.argmax(dim=-1)
            correct += int((preds[valid] == action_id[valid]).sum().item())
            total += int(valid.sum().item())
            n_steps += 1
            if max_steps_per_epoch and n_steps >= int(max_steps_per_epoch):
                break

        avg_loss = total_loss / max(n_steps, 1)
        acc = float(correct) / float(max(total, 1))
        summary["history"].append({"epoch": epoch, "loss": avg_loss, "acc": acc, "samples": total})
        if logger is not None:
            logger.log(
                "action_pretrain_epoch",
                epoch=int(epoch),
                loss=float(avg_loss),
                accuracy=float(acc),
                samples=int(total),
                steps=int(n_steps),
            )
        print(f"[ActionPretrain] epoch={epoch} loss={avg_loss:.4f} acc={acc:.4f} samples={total}")

    return action_head, summary


def run_pretrain(cfg, status_file=None):
    seed = int(cfg.get("seed", 42))
    generator = seed_everything(seed, deterministic=False)
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device_name == "auto" and not str(device).startswith("cuda"):
        device = torch.device("cpu")
    print(f"[PretrainOnly] device={device}")

    started = time.time()

    def emit_status(state, **fields):
        payload = {
            "phase": "ssl_pretrain_only",
            "state": state,
            "time": int(time.time()),
            "elapsed_sec": round(time.time() - started, 1),
        }
        payload.update(fields)
        _write_status_file(status_file, payload)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    results_dir = cfg.get("reporting", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"pretrain_motion_log_{run_tag}.jsonl")
    logger = ExperimentLogger(log_path)
    logger.log("pretrain_start", config=cfg)

    if bool(data_cfg.get("preprocess_videos", False)):
        emit_status("preprocess_starting")
        precompute_videos(
            csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
            processed_root=str(data_cfg.get("processed_root", "data/processed")),
            frame_stride=int(data_cfg.get("frame_stride", 1)),
            max_frames=int(data_cfg.get("max_frames", 0)),
            overwrite=bool(data_cfg.get("preprocess_overwrite", False)),
            progress_every=int(data_cfg.get("precompute_progress_every", 10)),
            status_callback=lambda s: emit_status("preprocess_running", **s),
        )
        emit_status("preprocess_done")

    if _is_rtx_4050(device):
        if "rtx4050_batch_size" in train_cfg:
            train_cfg["batch_size"] = int(train_cfg["rtx4050_batch_size"])
        if "rtx4050_num_workers" in data_cfg:
            data_cfg["num_workers"] = int(data_cfg["rtx4050_num_workers"])

    dataset = VideoDataset(
        csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
        sequence_length=int(data_cfg.get("seq_len", 64)),
        is_training=True,
        require_label=False,
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

    num_workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 4)),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": collate_motion_batch,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)

    model_cfg = cfg.get("model", {})
    thresholds = cfg.get("thresholds", {})
    model = ASDPipeline(
        K_max=int(model_cfg.get("K_max", 16)),
        d_model=int(model_cfg.get("d_model", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        theta_high=float(thresholds.get("decision_high", 0.7)),
        theta_low=float(thresholds.get("decision_low", 0.3)),
    ).to(device)

    init_ckpt = str(train_cfg.get("pretrained_motion_encoder_checkpoint", "")).strip()
    _load_pretrained_motion_encoder(model, init_ckpt, device)

    ssl_epochs = int(train_cfg.get("ssl_pretrain_epochs", 8))
    ssl_lr = float(train_cfg.get("ssl_lr", 1e-4))
    ssl_steps = int(train_cfg.get("ssl_steps_per_epoch", 300))

    emit_status(
        "starting",
        csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
        samples=int(len(dataset)),
        epochs=ssl_epochs,
        lr=ssl_lr,
        action_classes=int(getattr(dataset, "num_action_classes", 0)),
    )
    logger.log(
        "pretrain_setup",
        csv_path=str(data_cfg.get("csv_path", "data/videos.csv")),
        samples=int(len(dataset)),
        epochs=int(ssl_epochs),
        lr=float(ssl_lr),
        action_classes=int(getattr(dataset, "num_action_classes", 0)),
    )
    action_classes = int(getattr(dataset, "num_action_classes", 0))
    action_mode_enabled = bool(train_cfg.get("motion_pretrain_use_action_type", True))
    action_head = None
    pretrain_summary = {}

    if action_mode_enabled and action_classes >= 2:
        action_head, pretrain_summary = pretrain_motion_encoder_action_type(
            model,
            loader,
            device=device,
            num_classes=action_classes,
            epochs=ssl_epochs,
            lr=ssl_lr,
            max_steps_per_epoch=ssl_steps,
            logger=logger,
        )
    else:
        if action_mode_enabled and action_classes < 2:
            print(
                "[PretrainOnly] action_type pretraining requested but CSV does not contain "
                ">=2 action classes. Falling back to SSL."
            )
        pretrain_motion_encoder(
            model,
            loader,
            device=device,
            epochs=ssl_epochs,
            lr=ssl_lr,
            max_steps_per_epoch=ssl_steps,
            logger=logger,
        )
        pretrain_summary = {"mode": "ssl", "epochs": ssl_epochs}

    out_path = os.path.join(results_dir, "motion_encoder_pretrained.pth")
    torch.save(
        {
            "motion_encoder": model.motion_encoder.state_dict(),
            "model_state": model.state_dict(),
            "action_to_id": getattr(dataset, "action_to_id", {}),
            "id_to_action": getattr(dataset, "id_to_action", []),
            "pretrain_summary": pretrain_summary,
            "config": cfg,
        },
        out_path,
    )

    if action_head is not None:
        action_head_path = os.path.join(results_dir, "motion_action_head.pth")
        torch.save(
            {
                "action_head_state": action_head.state_dict(),
                "action_to_id": getattr(dataset, "action_to_id", {}),
                "id_to_action": getattr(dataset, "id_to_action", []),
                "config": cfg,
            },
            action_head_path,
        )
        with open(os.path.join(results_dir, "motion_action_label_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "action_to_id": getattr(dataset, "action_to_id", {}),
                    "id_to_action": getattr(dataset, "id_to_action", []),
                },
                f,
                indent=2,
            )

    emit_status("done", output_checkpoint=out_path)
    logger.log(
        "pretrain_done",
        output_checkpoint=out_path,
        pretrain_summary=pretrain_summary,
        action_head_saved=bool(action_head is not None),
    )

    pdf_path = os.path.join(results_dir, f"pretrain_motion_report_{run_tag}.pdf")
    export_experiment_log_pdf(
        log_jsonl_path=log_path,
        pdf_path=pdf_path,
        title="Motion Encoder Pretraining Report",
        extra_summary={
            "mode": pretrain_summary.get("mode", "unknown"),
            "output_checkpoint": out_path,
            "action_classes": int(getattr(dataset, "num_action_classes", 0)),
        },
    )
    print(f"[PretrainOnly] Saved: {out_path}")
    print(f"[PretrainOnly] Log JSONL: {log_path}")
    print(f"[PretrainOnly] Report PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretrain only motion encoder")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, action="append", default=[])
    parser.add_argument("--csv", type=str, default=None, help="Override data.csv_path")
    parser.add_argument("--status-file", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.csv:
        args.override.append(f"data.csv_path={args.csv}")
    cfg = apply_overrides(cfg, args.override)
    run_pretrain(cfg, status_file=args.status_file)


if __name__ == "__main__":
    main()
