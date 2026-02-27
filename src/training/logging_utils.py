import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class ExperimentLogger:
    def __init__(self, out_path):
        self.out_path = str(out_path)
        directory = os.path.dirname(self.out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def log(self, stage, **fields):
        payload = {
            "time": int(time.time()),
            "stage": str(stage),
        }
        payload.update(fields)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def _read_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _group_by_stage(rows):
    out = {}
    for r in rows:
        stage = str(r.get("stage", "unknown"))
        out.setdefault(stage, []).append(r)
    return out


def _extract_series(rows, x_key, y_key):
    xs = []
    ys = []
    for r in rows:
        if x_key not in r or y_key not in r:
            continue
        try:
            x = float(r[x_key])
            y = float(r[y_key])
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        except Exception:
            continue
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def export_experiment_log_pdf(log_jsonl_path, pdf_path, title="Training Log Report", extra_summary=None):
    rows = _read_jsonl(log_jsonl_path)
    stages = _group_by_stage(rows)
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # Cover / summary page
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.04, 0.95, title, fontsize=18, fontweight="bold", va="top")
        ax.text(0.04, 0.91, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        ax.text(0.04, 0.88, f"Log file: {os.path.basename(log_jsonl_path)}", fontsize=10)
        ax.text(0.04, 0.85, f"Total log rows: {len(rows)}", fontsize=10)

        stage_rows = [[k, str(len(v))] for k, v in sorted(stages.items(), key=lambda kv: kv[0])]
        if stage_rows:
            table = ax.table(
                cellText=stage_rows,
                colLabels=["Stage", "Rows"],
                cellLoc="left",
                colLoc="left",
                bbox=[0.04, 0.35, 0.45, 0.45],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.2)

        if extra_summary:
            summary_rows = [[str(k), str(v)] for k, v in extra_summary.items()]
            table2 = ax.table(
                cellText=summary_rows,
                colLabels=["Summary", "Value"],
                cellLoc="left",
                colLoc="left",
                bbox=[0.52, 0.35, 0.44, 0.45],
            )
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1.0, 1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # SSL pretrain curves
        ssl_rows = stages.get("ssl_pretrain", [])
        action_rows = stages.get("action_pretrain_epoch", [])
        if ssl_rows or action_rows:
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.8))
            fig.suptitle("Motion Encoder Pretraining", fontsize=14, fontweight="bold")

            if ssl_rows:
                x, y = _extract_series(ssl_rows, "epoch", "loss")
                if x.size:
                    axes[0].plot(x, y, marker="o", linewidth=2)
                axes[0].set_title("SSL Loss")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
            else:
                axes[0].text(0.5, 0.5, "No SSL rows", ha="center", va="center")
                axes[0].set_axis_off()

            if action_rows:
                x1, y1 = _extract_series(action_rows, "epoch", "loss")
                x2, y2 = _extract_series(action_rows, "epoch", "accuracy")
                if x1.size:
                    axes[1].plot(x1, y1, marker="o", linewidth=2, label="Loss")
                if x2.size:
                    ax2 = axes[1].twinx()
                    ax2.plot(x2, y2, color="tab:orange", marker="s", linewidth=1.7, label="Accuracy")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_ylim(0.0, 1.05)
                axes[1].set_title("Action-Type Pretraining")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
            else:
                axes[1].text(0.5, 0.5, "No action-type rows", ha="center", va="center")
                axes[1].set_axis_off()

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        # NAS curves
        nas_rows = stages.get("nas_generation", [])
        if nas_rows:
            fig, ax = plt.subplots(figsize=(11.69, 4.8))
            x, y = _extract_series(nas_rows, "generation", "best_fitness")
            if x.size:
                ax.plot(x, y, marker="o", linewidth=2, label="Best fitness")
            x_auc, y_auc = _extract_series(nas_rows, "generation", "best_auc")
            if x_auc.size:
                ax.plot(x_auc, y_auc, marker="s", linewidth=1.6, label="Best AUC")
            ax.set_title("NAS Evolution")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Score")
            ax.legend(loc="best")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Fold training curves
        fold_rows = stages.get("train_fold_epoch", [])
        if fold_rows:
            # Group by fold.
            per_fold = {}
            for r in fold_rows:
                fold = int(r.get("fold", 0))
                per_fold.setdefault(fold, []).append(r)
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            fig.suptitle("Cross-Validation Training", fontsize=14, fontweight="bold")

            ax_loss = axes[0, 0]
            ax_auc = axes[0, 1]
            ax_score = axes[1, 0]
            ax_f1 = axes[1, 1]
            for fold, items in sorted(per_fold.items(), key=lambda kv: kv[0]):
                items = sorted(items, key=lambda r: float(r.get("epoch", 0)))
                x = np.asarray([float(r.get("epoch", 0)) for r in items], dtype=float)
                tr = np.asarray([float(r.get("train_loss", np.nan)) for r in items], dtype=float)
                vl = np.asarray([float(r.get("val_loss", np.nan)) for r in items], dtype=float)
                sc = np.asarray([float(r.get("score", np.nan)) for r in items], dtype=float)
                auc = np.asarray([float(r.get("metrics", {}).get("auc", np.nan)) for r in items], dtype=float)
                f1 = np.asarray([float(r.get("metrics", {}).get("f1_opt", np.nan)) for r in items], dtype=float)
                ax_loss.plot(x, tr, linewidth=1.4, label=f"Fold {fold} train")
                ax_loss.plot(x, vl, linewidth=1.4, linestyle="--", label=f"Fold {fold} val")
                ax_auc.plot(x, auc, linewidth=1.8, label=f"Fold {fold}")
                ax_score.plot(x, sc, linewidth=1.8, label=f"Fold {fold}")
                ax_f1.plot(x, f1, linewidth=1.8, label=f"Fold {fold}")

            ax_loss.set_title("Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend(loc="best", fontsize=8)

            ax_auc.set_title("AUC")
            ax_auc.set_xlabel("Epoch")
            ax_auc.set_ylabel("AUC")
            ax_auc.set_ylim(0.0, 1.05)
            ax_auc.legend(loc="best", fontsize=8)

            ax_score.set_title("Selection Score")
            ax_score.set_xlabel("Epoch")
            ax_score.set_ylabel("Score")
            ax_score.legend(loc="best", fontsize=8)

            ax_f1.set_title("F1 (opt threshold)")
            ax_f1.set_xlabel("Epoch")
            ax_f1.set_ylabel("F1")
            ax_f1.set_ylim(0.0, 1.05)
            ax_f1.legend(loc="best", fontsize=8)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        # Final training curves
        final_rows = stages.get("final_train_epoch", [])
        finetune_rows = stages.get("final_finetune_epoch", [])
        if final_rows or finetune_rows:
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.8))
            fig.suptitle("Final Training", fontsize=14, fontweight="bold")

            if final_rows:
                x, y = _extract_series(final_rows, "epoch", "train_loss")
                if x.size:
                    axes[0].plot(x, y, marker="o", linewidth=2)
                axes[0].set_title("Final Train Loss")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
            else:
                axes[0].text(0.5, 0.5, "No final-train rows", ha="center", va="center")
                axes[0].set_axis_off()

            if finetune_rows:
                x, y = _extract_series(finetune_rows, "epoch", "train_loss")
                if x.size:
                    axes[1].plot(x, y, marker="o", linewidth=2, color="tab:orange")
                axes[1].set_title("Final Finetune Loss")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
            else:
                axes[1].text(0.5, 0.5, "No final-finetune rows", ha="center", va="center")
                axes[1].set_axis_off()

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path
