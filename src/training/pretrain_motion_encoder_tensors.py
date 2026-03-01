"""
Self-supervised pretraining of motion encoder from preprocessed tensors.

Expected input tensors per file: [T, J, 9]
"""

# Import `argparse` to support computations in this stage of output generation.
import argparse
# Import `os` to support computations in this stage of output generation.
import os
# Import `time` to support computations in this stage of output generation.
import time

# Import `torch` to support computations in this stage of output generation.
import torch
# Import `torch.nn as nn` to support computations in this stage of output generation.
import torch.nn as nn

# Import symbols from `src.models.video.microkinetic_encoders.motion_ssl_encoder` used in this stage's output computation path.
from src.models.video.microkinetic_encoders.motion_ssl_encoder import MultiBranchMotionEncoderSSL
# Import symbols from `src.training.logging_utils` used in this stage's output computation path.
from src.training.logging_utils import ExperimentLogger
# Import symbols from `src.training.motion_ssl_landmark_augment` used in this stage's output computation path.
from src.training.motion_ssl_landmark_augment import MotionAugmentationPipeline, build_positive_pair
# Import symbols from `src.training.motion_ssl_landmark_dataset` used in this stage's output computation path.
from src.training.motion_ssl_landmark_dataset import (
    LandmarkMotionPretrainDataset,
    build_landmark_ssl_dataloader,
)
# Import symbols from `src.training.motion_ssl_landmark_losses` used in this stage's output computation path.
from src.training.motion_ssl_landmark_losses import (
    FutureMotionPredictor,
    future_motion_prediction_loss,
    temporal_contrastive_infonce,
)
# Import symbols from `src.utils.config` used in this stage's output computation path.
from src.utils.config import apply_overrides, load_config
# Import symbols from `src.utils.seed` used in this stage's output computation path.
from src.utils.seed import seed_everything


# Define class `ProjectionHead` to package related logic in the prediction pipeline.
class ProjectionHead(nn.Module):
    """`ProjectionHead` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, in_dim, hidden_dim, out_dim):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Call `super` and use its result in later steps so gradient updates improve future predictions.
        super().__init__()
        # Set `self.net` for subsequent steps so gradient updates improve future predictions.
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    # Define the forward mapping from current inputs to tensors used for final prediction.
    def forward(self, x):
        """Maps current inputs to this module's output tensor representation."""
        # Return `self.net(x)` as this function's contribution to downstream output flow.
        return self.net(x)


# Define a reusable pipeline function whose outputs feed later steps.
def _build_device(device_name):
    """Constructs components whose structure controls later training or inference outputs."""
    # Set `dn` for subsequent steps so gradient updates improve future predictions.
    dn = str(device_name).strip().lower()
    # Branch on `dn == "auto"` to choose the correct output computation path.
    if dn == "auto":
        # Return `torch.device("cuda" if torch.cuda.is_available() el...` as this function's contribution to downstream output flow.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Return `torch.device(device_name)` as this function's contribution to downstream output flow.
    return torch.device(device_name)


# Define a training routine that updates parameters and changes future outputs.
def train(cfg, source_path, results_dir, batch_size_override=None):
    """Executes a training step/loop that updates parameters and directly changes model output behavior."""
    # Set `ssl_cfg` for subsequent steps so gradient updates improve future predictions.
    ssl_cfg = cfg.get("ssl_motion_pretrain", {})
    # Set `train_cfg` for subsequent steps so gradient updates improve future predictions.
    train_cfg = cfg.get("training", {})
    # Set `model_cfg` for subsequent steps so gradient updates improve future predictions.
    model_cfg = cfg.get("model", {})
    # Set `data_cfg` for subsequent steps so gradient updates improve future predictions.
    data_cfg = cfg.get("data", {})

    # Set `device` to the execution device used for this computation path.
    device = _build_device(cfg.get("device", "auto"))
    # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
    os.makedirs(results_dir, exist_ok=True)
    # Set `run_tag` for subsequent steps so gradient updates improve future predictions.
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    # Set `run_dir` for subsequent steps so gradient updates improve future predictions.
    run_dir = os.path.join(results_dir, f"motion_encoder_tensors_{run_tag}")
    # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
    os.makedirs(run_dir, exist_ok=True)

    # Compute `log_path` as an intermediate representation used by later output layers.
    log_path = os.path.join(run_dir, "train_log.jsonl")
    # Set `logger` for subsequent steps so gradient updates improve future predictions.
    logger = ExperimentLogger(log_path)

    # Compute `batch_size` as an intermediate representation used by later output layers.
    batch_size = (
        int(batch_size_override)
        # Branch on `batch_size_override is not None` to choose the correct output computation path.
        if batch_size_override is not None
        else int(ssl_cfg.get("batch_size", train_cfg.get("batch_size", 4)))
    )
    # Set `num_workers` for subsequent steps so gradient updates improve future predictions.
    num_workers = int(ssl_cfg.get("num_workers", data_cfg.get("num_workers", 0)))

    # Set `dataset` for subsequent steps so gradient updates improve future predictions.
    dataset = LandmarkMotionPretrainDataset(
        source_path=source_path,
        window_length=int(ssl_cfg.get("window_length", 48)),
        future_offsets=tuple(ssl_cfg.get("future_offsets", [1, 2])),
        samples_per_epoch=int(ssl_cfg.get("samples_per_epoch", 0)),
        expected_joints=int(ssl_cfg.get("expected_joints", 135)),
        cache_enabled=bool(ssl_cfg.get("cache_enabled", False)),
    )
    # Set `loader` for subsequent steps so gradient updates improve future predictions.
    loader = build_landmark_ssl_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(ssl_cfg.get("pin_memory", data_cfg.get("pin_memory", True))),
        shuffle=True,
        drop_last=True,
    )

    # Compute `emb_dim` as an intermediate representation used by later output layers.
    emb_dim = int(ssl_cfg.get("embedding_dim", model_cfg.get("d_model", 256)))
    # Set `encoder` for subsequent steps so gradient updates improve future predictions.
    encoder = MultiBranchMotionEncoderSSL(
        in_features=9,
        branch_hidden_dim=int(ssl_cfg.get("branch_hidden_dim", 192)),
        branch_out_dim=int(ssl_cfg.get("branch_out_dim", emb_dim)),
        embedding_dim=emb_dim,
        kernel_sizes=tuple(ssl_cfg.get("kernel_sizes", [5, 9, 11])),
        use_dilation=bool(ssl_cfg.get("use_dilation", True)),
        dropout=float(ssl_cfg.get("dropout", 0.1)),
        joint_pool=str(ssl_cfg.get("joint_pool", "mean")),
    ).to(device)
    # Set `proj` for subsequent steps so gradient updates improve future predictions.
    proj = ProjectionHead(
        in_dim=emb_dim,
        hidden_dim=int(ssl_cfg.get("projection_hidden_dim", 512)),
        out_dim=int(ssl_cfg.get("projection_dim", emb_dim)),
    ).to(device)
    # Set `future_pred` to predicted labels/scores that are reported downstream.
    future_pred = FutureMotionPredictor(
        embedding_dim=emb_dim,
        hidden_dim=int(ssl_cfg.get("future_hidden_dim", 512)),
        max_horizon=max(1, int(max(ssl_cfg.get("future_offsets", [1, 2])))),
    ).to(device)
    # Set `augmenter` for subsequent steps so gradient updates improve future predictions.
    augmenter = MotionAugmentationPipeline(
        joint_dropout_prob=float(ssl_cfg.get("joint_dropout", 0.1)),
        coordinate_noise_std=float(ssl_cfg.get("coord_noise_std", 0.005)),
        temporal_mask_ratio=float(ssl_cfg.get("temporal_mask_ratio", 0.15)),
        speed_min=float(ssl_cfg.get("speed_min", 0.9)),
        speed_max=float(ssl_cfg.get("speed_max", 1.1)),
    )

    # Set `lr` for subsequent steps so gradient updates improve future predictions.
    lr = float(ssl_cfg.get("lr", train_cfg.get("ssl_lr", 1e-3)))
    # Set `wd` for subsequent steps so gradient updates improve future predictions.
    wd = float(ssl_cfg.get("weight_decay", 1e-4))
    # Set `params` for subsequent steps so gradient updates improve future predictions.
    params = list(encoder.parameters()) + list(proj.parameters()) + list(future_pred.parameters())
    # Initialize `optimizer` to control parameter updates during training.
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    # Set `total_steps` for subsequent steps so gradient updates improve future predictions.
    total_steps = max(1, int(ssl_cfg.get("epochs", 50)) * max(1, len(loader)))
    # Initialize `scheduler` to adjust learning rate and stabilize output convergence.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=float(ssl_cfg.get("min_lr", 1e-6)),
    )
    # Set `scaler` for subsequent steps so gradient updates improve future predictions.
    scaler = torch.amp.GradScaler(device=str(device), enabled=str(device).startswith("cuda"))

    # Compute `epochs` as an intermediate representation used by later output layers.
    epochs = int(ssl_cfg.get("epochs", 50))
    # Update `future_w` with a loss term that drives backpropagation and output improvement.
    future_w = float(ssl_cfg.get("loss_future_weight", 0.5))
    # Set `temp` for subsequent steps so gradient updates improve future predictions.
    temp = float(ssl_cfg.get("temperature", 0.1))

    # Log runtime values to verify that output computation is behaving as expected.
    logger.log(
        "tensor_ssl_setup",
        source_path=str(source_path),
        device=str(device),
        dataset_size=int(len(dataset)),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        epochs=int(epochs),
        batches_per_epoch=int(len(loader)),
    )

    # Update `best_loss` with a loss term that drives backpropagation and output improvement.
    best_loss = float("inf")
    # Iterate over `range(1, epochs + 1)` so each item contributes to final outputs/metrics.
    for epoch in range(1, epochs + 1):
        # Call `encoder.train` and use its result in later steps so gradient updates improve future predictions.
        encoder.train()
        # Call `proj.train` and use its result in later steps so gradient updates improve future predictions.
        proj.train()
        # Call `future_pred.train` and use its result in later steps so gradient updates improve future predictions.
        future_pred.train()
        # Update `sum_loss` with a loss term that drives backpropagation and output improvement.
        sum_loss = 0.0
        # Set `sum_ctr` for subsequent steps so gradient updates improve future predictions.
        sum_ctr = 0.0
        # Set `sum_fut` for subsequent steps so gradient updates improve future predictions.
        sum_fut = 0.0
        # Set `n_steps` for subsequent steps so gradient updates improve future predictions.
        n_steps = 0
        # Set `t0` for subsequent steps so gradient updates improve future predictions.
        t0 = time.time()

        # Iterate over `loader` so each item contributes to final outputs/metrics.
        for batch in loader:
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Compute `anchor` as an intermediate representation used by later output layers.
                anchor = batch["anchor_window"].to(device, non_blocking=True)
                # Set `future` for subsequent steps so gradient updates improve future predictions.
                future = batch["future_window"].to(device, non_blocking=True)
                # Compute `horizon` as an intermediate representation used by later output layers.
                horizon = batch["horizon"].to(device, non_blocking=True)

                # Set `v1, v2` for subsequent steps so gradient updates improve future predictions.
                v1, v2 = build_positive_pair(anchor, augmenter)
                # Set `use_amp` for subsequent steps so gradient updates improve future predictions.
                use_amp = str(device).startswith("cuda")
                # Use a managed context to safely handle resources used during output computation.
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    # Compute `z1` as an intermediate representation used by later output layers.
                    z1 = proj(encoder(v1))
                    # Compute `z2` as an intermediate representation used by later output layers.
                    z2 = proj(encoder(v2))
                    # Update `loss_ctr` with a loss term that drives backpropagation and output improvement.
                    loss_ctr = temporal_contrastive_infonce(z1, z2, temperature=temp)

                    # Compute `a_emb` as an intermediate representation used by later output layers.
                    a_emb = encoder(anchor)
                    # Compute `f_emb` as an intermediate representation used by later output layers.
                    f_emb = encoder(future)
                    # Set `pred` to predicted labels/scores that are reported downstream.
                    pred = future_pred(a_emb, horizon)
                    # Update `loss_fut` with a loss term that drives backpropagation and output improvement.
                    loss_fut = future_motion_prediction_loss(pred, f_emb)

                    # Update `loss` with a loss term that drives backpropagation and output improvement.
                    loss = loss_ctr + future_w * loss_fut

                # Reset gradients before next step to avoid mixing gradient signals across batches.
                optimizer.zero_grad(set_to_none=True)
                # Backpropagate current loss so gradients can update model output behavior.
                scaler.scale(loss).backward()
                # Call `scaler.unscale_` and use its result in later steps so gradient updates improve future predictions.
                scaler.unscale_(optimizer)
                # Call `torch.nn.utils.clip_grad_norm_` and use its result in later steps so gradient updates improve future predictions.
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                # Apply optimizer step so future predictions reflect this batch's gradients.
                scaler.step(optimizer)
                # Call `scaler.update` and use its result in later steps so gradient updates improve future predictions.
                scaler.update()
                # Call `scheduler.step` and use its result in later steps so gradient updates improve future predictions.
                scheduler.step()

                # Execute this statement so gradient updates improve future predictions.
                n_steps += 1
                # Call `float` and use its result in later steps so gradient updates improve future predictions.
                sum_loss += float(loss.item())
                # Call `float` and use its result in later steps so gradient updates improve future predictions.
                sum_ctr += float(loss_ctr.item())
                # Call `float` and use its result in later steps so gradient updates improve future predictions.
                sum_fut += float(loss_fut.item())
            # Handle exceptions and keep output behavior controlled under error conditions.
            except torch.OutOfMemoryError as exc:
                # Branch on `torch.cuda.is_available()` to choose the correct output computation path.
                if torch.cuda.is_available():
                    # Call `torch.cuda.empty_cache` and use its result in later steps so gradient updates improve future predictions.
                    torch.cuda.empty_cache()
                # Raise explicit error to stop invalid state from producing misleading outputs.
                raise RuntimeError(
                    "CUDA OOM during tensor pretraining. Reduce batch size "
                    "(try --batch-size 1 or 2), lower ssl_motion_pretrain.branch_hidden_dim, "
                    "and keep ssl_motion_pretrain.expected_joints=25 for NTU."
                ) from exc

        # Update `epoch_loss` with a loss term that drives backpropagation and output improvement.
        epoch_loss = sum_loss / max(n_steps, 1)
        # Compute `epoch_ctr` as an intermediate representation used by later output layers.
        epoch_ctr = sum_ctr / max(n_steps, 1)
        # Compute `epoch_fut` as an intermediate representation used by later output layers.
        epoch_fut = sum_fut / max(n_steps, 1)
        # Set `elapsed` for subsequent steps so gradient updates improve future predictions.
        elapsed = time.time() - t0

        # Log runtime values to verify that output computation is behaving as expected.
        print(
            f"[TensorSSL] epoch={epoch}/{epochs} "
            f"loss={epoch_loss:.4f} ctr={epoch_ctr:.4f} fut={epoch_fut:.4f} "
            f"bs={batch_size} steps={n_steps} time={elapsed:.1f}s"
        )
        # Log runtime values to verify that output computation is behaving as expected.
        logger.log(
            "tensor_ssl_epoch",
            epoch=int(epoch),
            loss=float(epoch_loss),
            loss_contrastive=float(epoch_ctr),
            loss_future=float(epoch_fut),
            batch_size=int(batch_size),
            steps=int(n_steps),
            lr=float(optimizer.param_groups[0]["lr"]),
            elapsed_sec=round(elapsed, 2),
        )

        # Compute `latest_path` as an intermediate representation used by later output layers.
        latest_path = os.path.join(run_dir, "checkpoint_latest.pth")
        # Call `torch.save` and use its result in later steps so gradient updates improve future predictions.
        torch.save(
            {
                "epoch": int(epoch),
                "encoder": encoder.state_dict(),
                "projection_head": proj.state_dict(),
                "future_predictor": future_pred.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            },
            latest_path,
        )
        # Branch on `epoch_loss < best_loss` to choose the correct output computation path.
        if epoch_loss < best_loss:
            # Update `best_loss` with a loss term that drives backpropagation and output improvement.
            best_loss = epoch_loss
            # Call `torch.save` and use its result in later steps so gradient updates improve future predictions.
            torch.save(
                {
                    "epoch": int(epoch),
                    "motion_encoder": encoder.state_dict(),
                    "embedding_dim": int(encoder.embedding_dim),
                    "loss": float(best_loss),
                    "config": cfg,
                },
                os.path.join(run_dir, "motion_encoder_best.pth"),
            )

    # Log runtime values to verify that output computation is behaving as expected.
    logger.log("tensor_ssl_done", best_loss=float(best_loss), run_dir=run_dir)
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[TensorSSL] done. best_loss={best_loss:.6f}")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"[TensorSSL] run_dir={run_dir}")


# Define a reusable pipeline function whose outputs feed later steps.
def main():
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `parser` for subsequent steps so gradient updates improve future predictions.
    parser = argparse.ArgumentParser(description="Pretrain motion encoder from preprocessed tensors")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--config", type=str, default="config.yaml")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--source", type=str, required=True, help="Tensor source: folder/csv/list/file")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--results-dir", type=str, default="results")
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--batch-size", type=int, default=None)
    # Call `parser.add_argument` and use its result in later steps so gradient updates improve future predictions.
    parser.add_argument("--override", type=str, action="append", default=[])
    # Set `args` for subsequent steps so gradient updates improve future predictions.
    args = parser.parse_args()

    # Set `cfg` for subsequent steps so gradient updates improve future predictions.
    cfg = load_config(args.config)
    # Set `cfg` for subsequent steps so gradient updates improve future predictions.
    cfg = apply_overrides(cfg, args.override)
    # Call `seed_everything` and use its result in later steps so gradient updates improve future predictions.
    seed_everything(int(cfg.get("seed", 42)), deterministic=False)
    # Call `train` and use its result in later steps so gradient updates improve future predictions.
    train(
        cfg=cfg,
        source_path=args.source,
        results_dir=args.results_dir,
        batch_size_override=args.batch_size,
    )


# Branch on `__name__ == "__main__"` to choose the correct output computation path.
if __name__ == "__main__":
    # Call `main` and use its result in later steps so gradient updates improve future predictions.
    main()
