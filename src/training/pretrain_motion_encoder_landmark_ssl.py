"""
Self-supervised pretraining for multimodal landmark motion encoder.

Usage:
  python -m src.training.pretrain_motion_encoder_landmark_ssl --config config.yaml --source data/motion_ssl
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.models.video.microkinetic_encoders.motion_ssl_encoder import (
    MultiBranchMotionEncoderSSL,
    freeze_encoder,
)
from src.training.logging_utils import ExperimentLogger
from src.training.motion_ssl_landmark_augment import (
    MotionAugmentationPipeline,
    build_positive_pair,
)
from src.training.motion_ssl_landmark_dataset import (
    LandmarkMotionPretrainDataset,
    build_landmark_ssl_dataloader,
)
from src.training.motion_ssl_landmark_losses import (
    FutureMotionPredictor,
    future_motion_prediction_loss,
    temporal_contrastive_infonce,
)
from src.utils.config import apply_overrides, load_config
from src.utils.seed import seed_everything


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x):
        return self.net(x)


class MotionSSLTrainer:
    def __init__(self, cfg, source_path, results_dir):
        self.cfg = cfg
        self.results_dir = str(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        ssl_cfg = cfg.get("ssl_motion_pretrain", {})
        train_cfg = cfg.get("training", {})
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        self.device = self._build_device(cfg)
        self.epochs = int(ssl_cfg.get("epochs", train_cfg.get("ssl_pretrain_epochs", 100)))
        self.log_every = int(ssl_cfg.get("log_every", 50))
        self.loss_future_weight = float(ssl_cfg.get("loss_future_weight", 0.5))
        self.temperature = float(ssl_cfg.get("temperature", 0.1))

        self.dataset = LandmarkMotionPretrainDataset(
            source_path=source_path,
            window_length=int(ssl_cfg.get("window_length", 48)),
            future_offsets=tuple(ssl_cfg.get("future_offsets", [1, 2])),
            samples_per_epoch=int(ssl_cfg.get("samples_per_epoch", 0)),
            expected_joints=int(ssl_cfg.get("expected_joints", 135)),
            cache_enabled=bool(ssl_cfg.get("cache_enabled", True)),
        )
        batch_size = int(ssl_cfg.get("batch_size", train_cfg.get("rtx4050_batch_size", 16)))
        self.loader = build_landmark_ssl_dataloader(
            self.dataset,
            batch_size=batch_size,
            num_workers=int(ssl_cfg.get("num_workers", data_cfg.get("num_workers", 0))),
            pin_memory=bool(ssl_cfg.get("pin_memory", data_cfg.get("pin_memory", True))),
            shuffle=True,
            drop_last=True,
        )

        embedding_dim = int(ssl_cfg.get("embedding_dim", model_cfg.get("d_model", 256)))
        self.encoder = MultiBranchMotionEncoderSSL(
            in_features=9,
            branch_hidden_dim=int(ssl_cfg.get("branch_hidden_dim", 192)),
            branch_out_dim=int(ssl_cfg.get("branch_out_dim", embedding_dim)),
            embedding_dim=embedding_dim,
            kernel_sizes=tuple(ssl_cfg.get("kernel_sizes", [5, 9, 11])),
            use_dilation=bool(ssl_cfg.get("use_dilation", True)),
            dropout=float(ssl_cfg.get("dropout", 0.1)),
            joint_pool=str(ssl_cfg.get("joint_pool", "mean")),
        ).to(self.device)
        self.projection_head = ProjectionHead(
            in_dim=embedding_dim,
            hidden_dim=int(ssl_cfg.get("projection_hidden_dim", 512)),
            out_dim=int(ssl_cfg.get("projection_dim", embedding_dim)),
        ).to(self.device)
        self.future_predictor = FutureMotionPredictor(
            embedding_dim=embedding_dim,
            hidden_dim=int(ssl_cfg.get("future_hidden_dim", 512)),
            max_horizon=max(int(v) for v in tuple(ssl_cfg.get("future_offsets", [1, 2]))),
        ).to(self.device)

        self.augmenter = MotionAugmentationPipeline(
            joint_dropout_prob=float(ssl_cfg.get("joint_dropout", 0.1)),
            coordinate_noise_std=float(ssl_cfg.get("coord_noise_std", 0.005)),
            temporal_mask_ratio=float(ssl_cfg.get("temporal_mask_ratio", 0.15)),
            speed_min=float(ssl_cfg.get("speed_min", 0.9)),
            speed_max=float(ssl_cfg.get("speed_max", 1.1)),
        )

        lr = float(ssl_cfg.get("lr", train_cfg.get("ssl_lr", 1e-3)))
        wd = float(ssl_cfg.get("weight_decay", 1e-4))
        params = (
            list(self.encoder.parameters())
            + list(self.projection_head.parameters())
            + list(self.future_predictor.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        total_steps = max(1, self.epochs * max(1, len(self.loader)))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=float(ssl_cfg.get("min_lr", 1e-6)),
        )
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=str(self.device).startswith("cuda"),
        )

    @staticmethod
    def _build_device(cfg):
        requested = str(cfg.get("device", "auto")).lower().strip()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _save_checkpoint(self, run_dir, name, epoch, metrics):
        path = os.path.join(run_dir, name)
        torch.save(
            {
                "epoch": int(epoch),
                "metrics": metrics,
                "encoder": self.encoder.state_dict(),
                "projection_head": self.projection_head.state_dict(),
                "future_predictor": self.future_predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.cfg,
            },
            path,
        )
        return path

    def train(self):
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.results_dir, f"motion_ssl_{run_tag}")
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train_log.jsonl")
        logger = ExperimentLogger(log_path)

        logger.log(
            "ssl_landmark_setup",
            device=str(self.device),
            epochs=int(self.epochs),
            samples=int(len(self.dataset)),
            batches_per_epoch=int(len(self.loader)),
        )

        best_loss = float("inf")
        global_step = 0
        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.projection_head.train()
            self.future_predictor.train()

            epoch_total = 0.0
            epoch_ctr = 0.0
            epoch_future = 0.0
            steps = 0
            started = time.time()

            for batch in self.loader:
                anchor = batch["anchor_window"].to(self.device, non_blocking=True)
                future = batch["future_window"].to(self.device, non_blocking=True)
                horizon = batch["horizon"].to(self.device, non_blocking=True)

                aug_1, aug_2 = build_positive_pair(anchor, self.augmenter)
                use_amp = str(self.device).startswith("cuda")
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    emb_1 = self.encoder(aug_1)
                    emb_2 = self.encoder(aug_2)
                    proj_1 = self.projection_head(emb_1)
                    proj_2 = self.projection_head(emb_2)
                    loss_ctr = temporal_contrastive_infonce(
                        proj_1,
                        proj_2,
                        temperature=self.temperature,
                    )

                    anchor_emb = self.encoder(anchor)
                    future_emb = self.encoder(future)
                    future_pred = self.future_predictor(anchor_emb, horizon)
                    loss_future = future_motion_prediction_loss(future_pred, future_emb)
                    loss = loss_ctr + (self.loss_future_weight * loss_future)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                global_step += 1
                steps += 1
                epoch_total += float(loss.item())
                epoch_ctr += float(loss_ctr.item())
                epoch_future += float(loss_future.item())

                if self.log_every > 0 and (global_step % self.log_every) == 0:
                    logger.log(
                        "ssl_landmark_step",
                        epoch=int(epoch),
                        step=int(global_step),
                        loss=float(loss.item()),
                        loss_contrastive=float(loss_ctr.item()),
                        loss_future=float(loss_future.item()),
                        lr=float(self.optimizer.param_groups[0]["lr"]),
                    )

            metrics = {
                "epoch": int(epoch),
                "loss": epoch_total / max(1, steps),
                "loss_contrastive": epoch_ctr / max(1, steps),
                "loss_future": epoch_future / max(1, steps),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "steps": int(steps),
                "elapsed_sec": round(time.time() - started, 2),
            }
            logger.log("ssl_landmark_epoch", **metrics)
            print(
                f"[MotionSSL] epoch={epoch}/{self.epochs} "
                f"loss={metrics['loss']:.4f} ctr={metrics['loss_contrastive']:.4f} "
                f"future={metrics['loss_future']:.4f} lr={metrics['lr']:.6f}"
            )

            self._save_checkpoint(run_dir, "checkpoint_latest.pth", epoch=epoch, metrics=metrics)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self._save_checkpoint(run_dir, "checkpoint_best.pth", epoch=epoch, metrics=metrics)
                torch.save(
                    {
                        "motion_encoder": self.encoder.state_dict(),
                        "embedding_dim": int(self.encoder.embedding_dim),
                        "epoch": int(epoch),
                        "loss": float(metrics["loss"]),
                        "config": self.cfg,
                    },
                    os.path.join(run_dir, "motion_encoder_best.pth"),
                )

        freeze_encoder(self.encoder)
        frozen_path = os.path.join(run_dir, "motion_encoder_frozen.pth")
        torch.save(
            {
                "motion_encoder": self.encoder.state_dict(),
                "embedding_dim": int(self.encoder.embedding_dim),
                "frozen": True,
                "config": self.cfg,
            },
            frozen_path,
        )
        logger.log(
            "ssl_landmark_done",
            best_loss=float(best_loss),
            frozen_encoder=frozen_path,
        )
        print(f"[MotionSSL] best_loss={best_loss:.6f}")
        print(f"[MotionSSL] frozen_encoder={frozen_path}")
        print(f"[MotionSSL] log={log_path}")
        return run_dir


def _resolve_source_path(cfg, arg_source):
    if arg_source:
        return arg_source
    ssl_cfg = cfg.get("ssl_motion_pretrain", {})
    data_cfg = cfg.get("data", {})
    for value in (
        ssl_cfg.get("source_path"),
        data_cfg.get("motion_ssl_source"),
        data_cfg.get("processed_root"),
    ):
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return ""


def main():
    parser = argparse.ArgumentParser(description="Pretrain multimodal landmark motion encoder (self-supervised)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, action="append", default=[])
    parser.add_argument("--source", type=str, default=None, help="Directory/file/csv/txt of [T,J,9] tensors")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    seed_everything(int(cfg.get("seed", 42)), deterministic=False)

    source_path = _resolve_source_path(cfg, args.source)
    if not source_path:
        raise ValueError("Source path is required. Pass --source or set ssl_motion_pretrain.source_path.")
    results_dir = args.results_dir or cfg.get("reporting", {}).get("results_dir", "results")

    trainer = MotionSSLTrainer(cfg=cfg, source_path=source_path, results_dir=results_dir)
    run_dir = trainer.train()

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir, "source_path": source_path}, f, indent=2)


if __name__ == "__main__":
    main()
