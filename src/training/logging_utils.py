"""Training module `src/training/logging_utils.py` that optimizes model weights and output quality."""

# Import `json` to support computations in this stage of output generation.
import json
# Import `os` to support computations in this stage of output generation.
import os
# Import `time` to support computations in this stage of output generation.
import time

# Import `matplotlib.pyplot as plt` to support computations in this stage of output generation.
import matplotlib.pyplot as plt
# Import `numpy as np` to support computations in this stage of output generation.
import numpy as np
# Import symbols from `matplotlib.backends.backend_pdf` used in this stage's output computation path.
from matplotlib.backends.backend_pdf import PdfPages


# Define class `ExperimentLogger` to package related logic in the prediction pipeline.
class ExperimentLogger:
    """`ExperimentLogger` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, out_path):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `self.out_path` as an intermediate representation used by later output layers.
        self.out_path = str(out_path)
        # Set `directory` for subsequent steps so gradient updates improve future predictions.
        directory = os.path.dirname(self.out_path)
        # Branch on `directory` to choose the correct output computation path.
        if directory:
            # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
            os.makedirs(directory, exist_ok=True)

    # Define a reusable pipeline function whose outputs feed later steps.
    def log(self, stage, **fields):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `payload` for subsequent steps so gradient updates improve future predictions.
        payload = {
            "time": int(time.time()),
            "stage": str(stage),
        }
        # Call `payload.update` and use its result in later steps so gradient updates improve future predictions.
        payload.update(fields)
        # Use a managed context to safely handle resources used during output computation.
        with open(self.out_path, "a", encoding="utf-8") as f:
            # Call `f.write` and use its result in later steps so gradient updates improve future predictions.
            f.write(json.dumps(payload) + "\n")


# Define a reusable pipeline function whose outputs feed later steps.
def _read_jsonl(path):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `rows` for subsequent steps so gradient updates improve future predictions.
    rows = []
    # Branch on `not path or not os.path.exists(path)` to choose the correct output computation path.
    if not path or not os.path.exists(path):
        # Return `rows` as this function's contribution to downstream output flow.
        return rows
    # Use a managed context to safely handle resources used during output computation.
    with open(path, "r", encoding="utf-8") as f:
        # Iterate over `f` so each item contributes to final outputs/metrics.
        for line in f:
            # Set `line` for subsequent steps so gradient updates improve future predictions.
            line = line.strip()
            # Branch on `not line` to choose the correct output computation path.
            if not line:
                # Skip current loop item so it does not affect accumulated output state.
                continue
            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Call `rows.append` and use its result in later steps so gradient updates improve future predictions.
                rows.append(json.loads(line))
            # Handle exceptions and keep output behavior controlled under error conditions.
            except Exception:
                # Skip current loop item so it does not affect accumulated output state.
                continue
    # Return `rows` as this function's contribution to downstream output flow.
    return rows


# Define a reusable pipeline function whose outputs feed later steps.
def _group_by_stage(rows):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `out` for subsequent steps so gradient updates improve future predictions.
    out = {}
    # Iterate over `rows` so each item contributes to final outputs/metrics.
    for r in rows:
        # Set `stage` for subsequent steps so gradient updates improve future predictions.
        stage = str(r.get("stage", "unknown"))
        # Call `out.setdefault` and use its result in later steps so gradient updates improve future predictions.
        out.setdefault(stage, []).append(r)
    # Return `out` as this function's contribution to downstream output flow.
    return out


# Define a reusable pipeline function whose outputs feed later steps.
def _extract_series(rows, x_key, y_key):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Compute `xs` as an intermediate representation used by later output layers.
    xs = []
    # Set `ys` for subsequent steps so gradient updates improve future predictions.
    ys = []
    # Iterate over `rows` so each item contributes to final outputs/metrics.
    for r in rows:
        # Branch on `x_key not in r or y_key not in r` to choose the correct output computation path.
        if x_key not in r or y_key not in r:
            # Skip current loop item so it does not affect accumulated output state.
            continue
        # Start guarded block so failures can be handled without breaking output flow.
        try:
            # Compute `x` as an intermediate representation used by later output layers.
            x = float(r[x_key])
            # Set `y` for subsequent steps so gradient updates improve future predictions.
            y = float(r[y_key])
            # Branch on `np.isfinite(x) and np.isfinite(y)` to choose the correct output computation path.
            if np.isfinite(x) and np.isfinite(y):
                # Call `xs.append` and use its result in later steps so gradient updates improve future predictions.
                xs.append(x)
                # Call `ys.append` and use its result in later steps so gradient updates improve future predictions.
                ys.append(y)
        # Handle exceptions and keep output behavior controlled under error conditions.
        except Exception:
            # Skip current loop item so it does not affect accumulated output state.
            continue
    # Return `np.asarray(xs, dtype=float), np.asarray(ys, dtype=f...` as this function's contribution to downstream output flow.
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# Define a reusable pipeline function whose outputs feed later steps.
def export_experiment_log_pdf(log_jsonl_path, pdf_path, title="Training Log Report", extra_summary=None):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `rows` for subsequent steps so gradient updates improve future predictions.
    rows = _read_jsonl(log_jsonl_path)
    # Set `stages` for subsequent steps so gradient updates improve future predictions.
    stages = _group_by_stage(rows)
    # Call `os.makedirs` and use its result in later steps so gradient updates improve future predictions.
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    # Use a managed context to safely handle resources used during output computation.
    with PdfPages(pdf_path) as pdf:
        # Cover / summary page
        # Set `fig` for subsequent steps so gradient updates improve future predictions.
        fig = plt.figure(figsize=(11.69, 8.27))
        # Compute `ax` as an intermediate representation used by later output layers.
        ax = fig.add_axes([0, 0, 1, 1])
        # Call `ax.axis` and use its result in later steps so gradient updates improve future predictions.
        ax.axis("off")
        # Call `ax.text` and use its result in later steps so gradient updates improve future predictions.
        ax.text(0.04, 0.95, title, fontsize=18, fontweight="bold", va="top")
        # Call `ax.text` and use its result in later steps so gradient updates improve future predictions.
        ax.text(0.04, 0.91, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        # Call `ax.text` and use its result in later steps so gradient updates improve future predictions.
        ax.text(0.04, 0.88, f"Log file: {os.path.basename(log_jsonl_path)}", fontsize=10)
        # Call `ax.text` and use its result in later steps so gradient updates improve future predictions.
        ax.text(0.04, 0.85, f"Total log rows: {len(rows)}", fontsize=10)

        # Set `stage_rows` for subsequent steps so gradient updates improve future predictions.
        stage_rows = [[k, str(len(v))] for k, v in sorted(stages.items(), key=lambda kv: kv[0])]
        # Branch on `stage_rows` to choose the correct output computation path.
        if stage_rows:
            # Set `table` for subsequent steps so gradient updates improve future predictions.
            table = ax.table(
                cellText=stage_rows,
                colLabels=["Stage", "Rows"],
                cellLoc="left",
                colLoc="left",
                bbox=[0.04, 0.35, 0.45, 0.45],
            )
            # Call `table.auto_set_font_size` and use its result in later steps so gradient updates improve future predictions.
            table.auto_set_font_size(False)
            # Call `table.set_fontsize` and use its result in later steps so gradient updates improve future predictions.
            table.set_fontsize(10)
            # Call `table.scale` and use its result in later steps so gradient updates improve future predictions.
            table.scale(1.0, 1.2)

        # Branch on `extra_summary` to choose the correct output computation path.
        if extra_summary:
            # Set `summary_rows` for subsequent steps so gradient updates improve future predictions.
            summary_rows = [[str(k), str(v)] for k, v in extra_summary.items()]
            # Set `table2` for subsequent steps so gradient updates improve future predictions.
            table2 = ax.table(
                cellText=summary_rows,
                colLabels=["Summary", "Value"],
                cellLoc="left",
                colLoc="left",
                bbox=[0.52, 0.35, 0.44, 0.45],
            )
            # Call `table2.auto_set_font_size` and use its result in later steps so gradient updates improve future predictions.
            table2.auto_set_font_size(False)
            # Call `table2.set_fontsize` and use its result in later steps so gradient updates improve future predictions.
            table2.set_fontsize(10)
            # Call `table2.scale` and use its result in later steps so gradient updates improve future predictions.
            table2.scale(1.0, 1.2)
        # Call `pdf.savefig` and use its result in later steps so gradient updates improve future predictions.
        pdf.savefig(fig, bbox_inches="tight")
        # Call `plt.close` and use its result in later steps so gradient updates improve future predictions.
        plt.close(fig)

        # SSL pretrain curves
        # Set `ssl_rows` for subsequent steps so gradient updates improve future predictions.
        ssl_rows = stages.get("ssl_pretrain", [])
        # Set `action_rows` for subsequent steps so gradient updates improve future predictions.
        action_rows = stages.get("action_pretrain_epoch", [])
        # Branch on `ssl_rows or action_rows` to choose the correct output computation path.
        if ssl_rows or action_rows:
            # Compute `fig, axes` as an intermediate representation used by later output layers.
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.8))
            # Call `fig.suptitle` and use its result in later steps so gradient updates improve future predictions.
            fig.suptitle("Motion Encoder Pretraining", fontsize=14, fontweight="bold")

            # Branch on `ssl_rows` to choose the correct output computation path.
            if ssl_rows:
                # Update `x, y` with a loss term that drives backpropagation and output improvement.
                x, y = _extract_series(ssl_rows, "epoch", "loss")
                # Branch on `x.size` to choose the correct output computation path.
                if x.size:
                    # Call `plot` and use its result in later steps so gradient updates improve future predictions.
                    axes[0].plot(x, y, marker="o", linewidth=2)
                # Call `set_title` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_title("SSL Loss")
                # Call `set_xlabel` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_xlabel("Epoch")
                # Call `set_ylabel` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_ylabel("Loss")
            else:
                # Call `text` and use its result in later steps so gradient updates improve future predictions.
                axes[0].text(0.5, 0.5, "No SSL rows", ha="center", va="center")
                # Call `set_axis_off` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_axis_off()

            # Branch on `action_rows` to choose the correct output computation path.
            if action_rows:
                # Update `x1, y1` with a loss term that drives backpropagation and output improvement.
                x1, y1 = _extract_series(action_rows, "epoch", "loss")
                # Compute `x2, y2` as an intermediate representation used by later output layers.
                x2, y2 = _extract_series(action_rows, "epoch", "accuracy")
                # Branch on `x1.size` to choose the correct output computation path.
                if x1.size:
                    # Call `plot` and use its result in later steps so gradient updates improve future predictions.
                    axes[1].plot(x1, y1, marker="o", linewidth=2, label="Loss")
                # Branch on `x2.size` to choose the correct output computation path.
                if x2.size:
                    # Compute `ax2` as an intermediate representation used by later output layers.
                    ax2 = axes[1].twinx()
                    # Call `ax2.plot` and use its result in later steps so gradient updates improve future predictions.
                    ax2.plot(x2, y2, color="tab:orange", marker="s", linewidth=1.7, label="Accuracy")
                    # Call `ax2.set_ylabel` and use its result in later steps so gradient updates improve future predictions.
                    ax2.set_ylabel("Accuracy")
                    # Call `ax2.set_ylim` and use its result in later steps so gradient updates improve future predictions.
                    ax2.set_ylim(0.0, 1.05)
                # Call `set_title` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_title("Action-Type Pretraining")
                # Call `set_xlabel` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_xlabel("Epoch")
                # Call `set_ylabel` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_ylabel("Loss")
            else:
                # Call `text` and use its result in later steps so gradient updates improve future predictions.
                axes[1].text(0.5, 0.5, "No action-type rows", ha="center", va="center")
                # Call `set_axis_off` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_axis_off()

            # Call `fig.tight_layout` and use its result in later steps so gradient updates improve future predictions.
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            # Call `pdf.savefig` and use its result in later steps so gradient updates improve future predictions.
            pdf.savefig(fig)
            # Call `plt.close` and use its result in later steps so gradient updates improve future predictions.
            plt.close(fig)

        # NAS curves
        # Set `nas_rows` for subsequent steps so gradient updates improve future predictions.
        nas_rows = stages.get("nas_generation", [])
        # Branch on `nas_rows` to choose the correct output computation path.
        if nas_rows:
            # Compute `fig, ax` as an intermediate representation used by later output layers.
            fig, ax = plt.subplots(figsize=(11.69, 4.8))
            # Compute `x, y` as an intermediate representation used by later output layers.
            x, y = _extract_series(nas_rows, "generation", "best_fitness")
            # Branch on `x.size` to choose the correct output computation path.
            if x.size:
                # Call `ax.plot` and use its result in later steps so gradient updates improve future predictions.
                ax.plot(x, y, marker="o", linewidth=2, label="Best fitness")
            # Compute `x_auc, y_auc` as an intermediate representation used by later output layers.
            x_auc, y_auc = _extract_series(nas_rows, "generation", "best_auc")
            # Branch on `x_auc.size` to choose the correct output computation path.
            if x_auc.size:
                # Call `ax.plot` and use its result in later steps so gradient updates improve future predictions.
                ax.plot(x_auc, y_auc, marker="s", linewidth=1.6, label="Best AUC")
            # Call `ax.set_title` and use its result in later steps so gradient updates improve future predictions.
            ax.set_title("NAS Evolution")
            # Call `ax.set_xlabel` and use its result in later steps so gradient updates improve future predictions.
            ax.set_xlabel("Generation")
            # Call `ax.set_ylabel` and use its result in later steps so gradient updates improve future predictions.
            ax.set_ylabel("Score")
            # Call `ax.legend` and use its result in later steps so gradient updates improve future predictions.
            ax.legend(loc="best")
            # Call `fig.tight_layout` and use its result in later steps so gradient updates improve future predictions.
            fig.tight_layout()
            # Call `pdf.savefig` and use its result in later steps so gradient updates improve future predictions.
            pdf.savefig(fig)
            # Call `plt.close` and use its result in later steps so gradient updates improve future predictions.
            plt.close(fig)

        # Fold training curves
        # Set `fold_rows` for subsequent steps so gradient updates improve future predictions.
        fold_rows = stages.get("train_fold_epoch", [])
        # Branch on `fold_rows` to choose the correct output computation path.
        if fold_rows:
            # Group by fold.
            # Set `per_fold` for subsequent steps so gradient updates improve future predictions.
            per_fold = {}
            # Iterate over `fold_rows` so each item contributes to final outputs/metrics.
            for r in fold_rows:
                # Set `fold` for subsequent steps so gradient updates improve future predictions.
                fold = int(r.get("fold", 0))
                # Call `per_fold.setdefault` and use its result in later steps so gradient updates improve future predictions.
                per_fold.setdefault(fold, []).append(r)
            # Compute `fig, axes` as an intermediate representation used by later output layers.
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            # Call `fig.suptitle` and use its result in later steps so gradient updates improve future predictions.
            fig.suptitle("Cross-Validation Training", fontsize=14, fontweight="bold")

            # Update `ax_loss` with a loss term that drives backpropagation and output improvement.
            ax_loss = axes[0, 0]
            # Compute `ax_auc` as an intermediate representation used by later output layers.
            ax_auc = axes[0, 1]
            # Compute `ax_score` as an intermediate representation used by later output layers.
            ax_score = axes[1, 0]
            # Compute `ax_f1` as an intermediate representation used by later output layers.
            ax_f1 = axes[1, 1]
            # Iterate over `sorted(per_fold.items(), key=lambda k...` so each item contributes to final outputs/metrics.
            for fold, items in sorted(per_fold.items(), key=lambda kv: kv[0]):
                # Set `items` for subsequent steps so gradient updates improve future predictions.
                items = sorted(items, key=lambda r: float(r.get("epoch", 0)))
                # Compute `x` as an intermediate representation used by later output layers.
                x = np.asarray([float(r.get("epoch", 0)) for r in items], dtype=float)
                # Update `tr` with a loss term that drives backpropagation and output improvement.
                tr = np.asarray([float(r.get("train_loss", np.nan)) for r in items], dtype=float)
                # Update `vl` with a loss term that drives backpropagation and output improvement.
                vl = np.asarray([float(r.get("val_loss", np.nan)) for r in items], dtype=float)
                # Set `sc` for subsequent steps so gradient updates improve future predictions.
                sc = np.asarray([float(r.get("score", np.nan)) for r in items], dtype=float)
                # Record `auc` as a metric describing current output quality.
                auc = np.asarray([float(r.get("metrics", {}).get("auc", np.nan)) for r in items], dtype=float)
                # Set `f1` for subsequent steps so gradient updates improve future predictions.
                f1 = np.asarray([float(r.get("metrics", {}).get("f1_opt", np.nan)) for r in items], dtype=float)
                # Call `ax_loss.plot` and use its result in later steps so gradient updates improve future predictions.
                ax_loss.plot(x, tr, linewidth=1.4, label=f"Fold {fold} train")
                # Call `ax_loss.plot` and use its result in later steps so gradient updates improve future predictions.
                ax_loss.plot(x, vl, linewidth=1.4, linestyle="--", label=f"Fold {fold} val")
                # Call `ax_auc.plot` and use its result in later steps so gradient updates improve future predictions.
                ax_auc.plot(x, auc, linewidth=1.8, label=f"Fold {fold}")
                # Call `ax_score.plot` and use its result in later steps so gradient updates improve future predictions.
                ax_score.plot(x, sc, linewidth=1.8, label=f"Fold {fold}")
                # Call `ax_f1.plot` and use its result in later steps so gradient updates improve future predictions.
                ax_f1.plot(x, f1, linewidth=1.8, label=f"Fold {fold}")

            # Call `ax_loss.set_title` and use its result in later steps so gradient updates improve future predictions.
            ax_loss.set_title("Loss")
            # Call `ax_loss.set_xlabel` and use its result in later steps so gradient updates improve future predictions.
            ax_loss.set_xlabel("Epoch")
            # Call `ax_loss.set_ylabel` and use its result in later steps so gradient updates improve future predictions.
            ax_loss.set_ylabel("Loss")
            # Call `ax_loss.legend` and use its result in later steps so gradient updates improve future predictions.
            ax_loss.legend(loc="best", fontsize=8)

            # Call `ax_auc.set_title` and use its result in later steps so gradient updates improve future predictions.
            ax_auc.set_title("AUC")
            # Call `ax_auc.set_xlabel` and use its result in later steps so gradient updates improve future predictions.
            ax_auc.set_xlabel("Epoch")
            # Call `ax_auc.set_ylabel` and use its result in later steps so gradient updates improve future predictions.
            ax_auc.set_ylabel("AUC")
            # Call `ax_auc.set_ylim` and use its result in later steps so gradient updates improve future predictions.
            ax_auc.set_ylim(0.0, 1.05)
            # Call `ax_auc.legend` and use its result in later steps so gradient updates improve future predictions.
            ax_auc.legend(loc="best", fontsize=8)

            # Call `ax_score.set_title` and use its result in later steps so gradient updates improve future predictions.
            ax_score.set_title("Selection Score")
            # Call `ax_score.set_xlabel` and use its result in later steps so gradient updates improve future predictions.
            ax_score.set_xlabel("Epoch")
            # Call `ax_score.set_ylabel` and use its result in later steps so gradient updates improve future predictions.
            ax_score.set_ylabel("Score")
            # Call `ax_score.legend` and use its result in later steps so gradient updates improve future predictions.
            ax_score.legend(loc="best", fontsize=8)

            # Call `ax_f1.set_title` and use its result in later steps so gradient updates improve future predictions.
            ax_f1.set_title("F1 (opt threshold)")
            # Call `ax_f1.set_xlabel` and use its result in later steps so gradient updates improve future predictions.
            ax_f1.set_xlabel("Epoch")
            # Call `ax_f1.set_ylabel` and use its result in later steps so gradient updates improve future predictions.
            ax_f1.set_ylabel("F1")
            # Call `ax_f1.set_ylim` and use its result in later steps so gradient updates improve future predictions.
            ax_f1.set_ylim(0.0, 1.05)
            # Call `ax_f1.legend` and use its result in later steps so gradient updates improve future predictions.
            ax_f1.legend(loc="best", fontsize=8)

            # Call `fig.tight_layout` and use its result in later steps so gradient updates improve future predictions.
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            # Call `pdf.savefig` and use its result in later steps so gradient updates improve future predictions.
            pdf.savefig(fig)
            # Call `plt.close` and use its result in later steps so gradient updates improve future predictions.
            plt.close(fig)

        # Final training curves
        # Set `final_rows` for subsequent steps so gradient updates improve future predictions.
        final_rows = stages.get("final_train_epoch", [])
        # Set `finetune_rows` for subsequent steps so gradient updates improve future predictions.
        finetune_rows = stages.get("final_finetune_epoch", [])
        # Branch on `final_rows or finetune_rows` to choose the correct output computation path.
        if final_rows or finetune_rows:
            # Compute `fig, axes` as an intermediate representation used by later output layers.
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.8))
            # Call `fig.suptitle` and use its result in later steps so gradient updates improve future predictions.
            fig.suptitle("Final Training", fontsize=14, fontweight="bold")

            # Branch on `final_rows` to choose the correct output computation path.
            if final_rows:
                # Update `x, y` with a loss term that drives backpropagation and output improvement.
                x, y = _extract_series(final_rows, "epoch", "train_loss")
                # Branch on `x.size` to choose the correct output computation path.
                if x.size:
                    # Call `plot` and use its result in later steps so gradient updates improve future predictions.
                    axes[0].plot(x, y, marker="o", linewidth=2)
                # Call `set_title` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_title("Final Train Loss")
                # Call `set_xlabel` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_xlabel("Epoch")
                # Call `set_ylabel` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_ylabel("Loss")
            else:
                # Call `text` and use its result in later steps so gradient updates improve future predictions.
                axes[0].text(0.5, 0.5, "No final-train rows", ha="center", va="center")
                # Call `set_axis_off` and use its result in later steps so gradient updates improve future predictions.
                axes[0].set_axis_off()

            # Branch on `finetune_rows` to choose the correct output computation path.
            if finetune_rows:
                # Update `x, y` with a loss term that drives backpropagation and output improvement.
                x, y = _extract_series(finetune_rows, "epoch", "train_loss")
                # Branch on `x.size` to choose the correct output computation path.
                if x.size:
                    # Call `plot` and use its result in later steps so gradient updates improve future predictions.
                    axes[1].plot(x, y, marker="o", linewidth=2, color="tab:orange")
                # Call `set_title` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_title("Final Finetune Loss")
                # Call `set_xlabel` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_xlabel("Epoch")
                # Call `set_ylabel` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_ylabel("Loss")
            else:
                # Call `text` and use its result in later steps so gradient updates improve future predictions.
                axes[1].text(0.5, 0.5, "No final-finetune rows", ha="center", va="center")
                # Call `set_axis_off` and use its result in later steps so gradient updates improve future predictions.
                axes[1].set_axis_off()

            # Call `fig.tight_layout` and use its result in later steps so gradient updates improve future predictions.
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            # Call `pdf.savefig` and use its result in later steps so gradient updates improve future predictions.
            pdf.savefig(fig)
            # Call `plt.close` and use its result in later steps so gradient updates improve future predictions.
            plt.close(fig)

    # Return `pdf_path` as this function's contribution to downstream output flow.
    return pdf_path
