# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

"""
Training report generation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from src.utils.metrics import (
    roc_pr_curves,
    compute_calibration_curve,
)
from src.utils.plotting import ema_smooth


def _plot_confusion_matrix(cm, title):
    # Compute `(fig, ax)` for the next processing step.
    fig, ax = plt.subplots(figsize=(5, 4))
    # Invoke `ax.imshow` to advance this processing stage.
    ax.imshow(cm, cmap="Blues")
    # Invoke `ax.set_title` to advance this processing stage.
    ax.set_title(title)
    # Invoke `ax.set_xlabel` to advance this processing stage.
    ax.set_xlabel("Predicted")
    # Invoke `ax.set_ylabel` to advance this processing stage.
    ax.set_ylabel("Actual")
    # Invoke `ax.set_xticks` to advance this processing stage.
    ax.set_xticks([0, 1])
    # Invoke `ax.set_yticks` to advance this processing stage.
    ax.set_yticks([0, 1])
    # Invoke `ax.set_xticklabels` to advance this processing stage.
    ax.set_xticklabels(["0", "1"])
    # Invoke `ax.set_yticklabels` to advance this processing stage.
    ax.set_yticklabels(["0", "1"])
    # Iterate `i` across `range(2)` to process each element.
    for i in range(2):
        # Iterate `j` across `range(2)` to process each element.
        for j in range(2):
            # Invoke `ax.text` to advance this processing stage.
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    # Invoke `fig.tight_layout` to advance this processing stage.
    fig.tight_layout()
    # Return the result expected by the caller.
    return fig


def generate_training_report(
    output_dir,
    fold_idx,
    history,
    eval_summary,
    attention_map=None,
    temporal_importance=None,
    nas_architecture=None,
    cv_summary=None,
    ema_alpha=0.2,
):
    # Invoke `os.makedirs` to advance this processing stage.
    os.makedirs(output_dir, exist_ok=True)
    # Compute `report_path` for the next processing step.
    report_path = os.path.join(
        output_dir,
        f"training_report_fold{fold_idx}.pdf" if fold_idx else "training_report_final.pdf",
    )

    # Compute `title_suffix` for the next processing step.
    title_suffix = f"Fold {fold_idx}" if fold_idx else "Final Model"

    # Compute `df` for the next processing step.
    df = pd.DataFrame(history) if history else pd.DataFrame()

    # Run this block with managed resources/context cleanup.
    with PdfPages(report_path) as pdf:
        # Summary page
        plt.figure(figsize=(12, 6))
        # Invoke `plt.axis` to advance this processing stage.
        plt.axis("off")
        # Invoke `plt.title` to advance this processing stage.
        plt.title(f"Training Summary - {title_suffix}", fontsize=14, fontweight="bold", y=1.05)

        # Compute `summary_rows` for the next processing step.
        summary_rows = []
        # Branch behavior based on the current runtime condition.
        if eval_summary:
            # Invoke `summary_rows.extend` to advance this processing stage.
            summary_rows.extend([
                ["AUC", f"{eval_summary.get('auc', 0):.3f}"],
                ["F1 (opt)", f"{eval_summary.get('f1_opt', 0):.3f}"],
                ["Accuracy@0.5", f"{eval_summary.get('acc_05', 0):.3f}"],
                ["Accuracy@opt", f"{eval_summary.get('acc_opt', 0):.3f}"],
                ["Sens@Spec", f"{eval_summary.get('sens_spec', 0):.3f}"],
                ["ECE", f"{eval_summary.get('ece', 0):.3f}"],
                ["Abstain rate", f"{eval_summary.get('abstain_rate', 0):.3f}"],
                ["Opt threshold", f"{eval_summary.get('opt_threshold', 0):.3f}"],
                ["Spec target", f"{eval_summary.get('spec_target', 0):.2f}"],
                ["Temp", f"{eval_summary.get('temperature', 1.0):.3f}"],
            ])
        else:
            # Invoke `summary_rows.append` to advance this processing stage.
            summary_rows.append(["Summary", "N/A"])

        # Compute `table` for the next processing step.
        table = plt.table(cellText=summary_rows, colLabels=["Metric", "Value"],
                          cellLoc="center", loc="center")
        # Invoke `table.auto_set_font_size` to advance this processing stage.
        table.auto_set_font_size(False)
        # Invoke `table.set_fontsize` to advance this processing stage.
        table.set_fontsize(10)
        # Invoke `table.scale` to advance this processing stage.
        table.scale(1, 1.5)
        # Invoke `pdf.savefig` to advance this processing stage.
        pdf.savefig(bbox_inches="tight")
        # Invoke `plt.close` to advance this processing stage.
        plt.close()

        # Loss curves
        if not df.empty and "train_loss" in df.columns and "val_loss" in df.columns:
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(10, 6))
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot(df["epoch"], ema_smooth(df["train_loss"].tolist(), ema_alpha), label="Train Loss")
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot(df["epoch"], ema_smooth(df["val_loss"].tolist(), ema_alpha), label="Val Loss")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"Loss Curves - {title_suffix}")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("Epoch")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("Loss")
            # Invoke `plt.grid` to advance this processing stage.
            plt.grid(True, linestyle="--", alpha=0.5)
            # Invoke `plt.legend` to advance this processing stage.
            plt.legend()
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

        # Metric curves
        if not df.empty and "auc" in df.columns:
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(10, 6))
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot(df["epoch"], ema_smooth(df["auc"].tolist(), ema_alpha), label="AUC")
            # Branch behavior based on the current runtime condition.
            if "f1_opt" in df.columns:
                # Invoke `plt.plot` to advance this processing stage.
                plt.plot(df["epoch"], ema_smooth(df["f1_opt"].tolist(), ema_alpha), label="F1 (opt)")
            # Branch behavior based on the current runtime condition.
            if "accuracy_05" in df.columns:
                # Invoke `plt.plot` to advance this processing stage.
                plt.plot(df["epoch"], ema_smooth(df["accuracy_05"].tolist(), ema_alpha), label="Acc@0.5")
            # Branch behavior based on the current runtime condition.
            if "accuracy_opt" in df.columns:
                # Invoke `plt.plot` to advance this processing stage.
                plt.plot(df["epoch"], ema_smooth(df["accuracy_opt"].tolist(), ema_alpha), label="Acc@opt")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"Metrics - {title_suffix}")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("Epoch")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("Score")
            # Invoke `plt.ylim` to advance this processing stage.
            plt.ylim(0, 1.05)
            # Invoke `plt.grid` to advance this processing stage.
            plt.grid(True, linestyle="--", alpha=0.5)
            # Invoke `plt.legend` to advance this processing stage.
            plt.legend()
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

        # ROC / PR / Calibration / Confusion Matrix
        if eval_summary and "labels" in eval_summary and "probs_cal" in eval_summary:
            # Compute `labels` for the next processing step.
            labels = np.array(eval_summary["labels"], dtype=int)
            # Compute `probs` for the next processing step.
            probs = np.array(eval_summary["probs_cal"], dtype=float)

            # Compute `(fpr, tpr, precision, recall)` for the next processing step.
            fpr, tpr, precision, recall = roc_pr_curves(labels, probs)
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(6, 6))
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot(fpr, tpr, label="ROC")
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot([0, 1], [0, 1], "--", color="gray")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("FPR")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("TPR")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"ROC - {title_suffix}")
            # Invoke `plt.grid` to advance this processing stage.
            plt.grid(True, linestyle="--", alpha=0.5)
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(6, 6))
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot(recall, precision, label="PR")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("Recall")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("Precision")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"PR Curve - {title_suffix}")
            # Invoke `plt.grid` to advance this processing stage.
            plt.grid(True, linestyle="--", alpha=0.5)
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

            # Compute `(bin_centers, accs, confs)` for the next processing step.
            bin_centers, accs, confs = compute_calibration_curve(labels, probs, n_bins=eval_summary.get("calib_bins", 10))
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(6, 6))
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot([0, 1], [0, 1], "--", color="gray")
            # Invoke `plt.scatter` to advance this processing stage.
            plt.scatter(confs, accs, s=40, c="blue")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("Confidence")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("Accuracy")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"Calibration - {title_suffix}")
            # Invoke `plt.grid` to advance this processing stage.
            plt.grid(True, linestyle="--", alpha=0.5)
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

            # Compute `cm` for the next processing step.
            cm = eval_summary.get("confusion_matrix")
            # Branch behavior based on the current runtime condition.
            if cm is not None:
                # Compute `fig` for the next processing step.
                fig = _plot_confusion_matrix(cm, f"Confusion Matrix (opt) - {title_suffix}")
                # Invoke `pdf.savefig` to advance this processing stage.
                pdf.savefig(fig)
                # Invoke `plt.close` to advance this processing stage.
                plt.close(fig)

        # Attention map
        if attention_map is not None:
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(8, 6))
            # Invoke `plt.imshow` to advance this processing stage.
            plt.imshow(attention_map, cmap="viridis", aspect="auto")
            # Invoke `plt.colorbar` to advance this processing stage.
            plt.colorbar(label="Attention")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"Attention Map - {title_suffix}")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("Key Token")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("Query Token")
            # Invoke `plt.tight_layout` to advance this processing stage.
            plt.tight_layout()
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

        # Temporal importance
        if temporal_importance is not None:
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(8, 4))
            # Invoke `plt.plot` to advance this processing stage.
            plt.plot(temporal_importance, marker="o", linewidth=1.5)
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"Temporal Importance (approx) - {title_suffix}")
            # Invoke `plt.xlabel` to advance this processing stage.
            plt.xlabel("Event Token")
            # Invoke `plt.ylabel` to advance this processing stage.
            plt.ylabel("Importance")
            # Invoke `plt.grid` to advance this processing stage.
            plt.grid(True, linestyle="--", alpha=0.5)
            # Invoke `plt.tight_layout` to advance this processing stage.
            plt.tight_layout()
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

        # NAS architecture
        if nas_architecture is not None:
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(10, 6))
            # Invoke `plt.axis` to advance this processing stage.
            plt.axis("off")
            # Invoke `plt.title` to advance this processing stage.
            plt.title(f"NAS Architecture - {title_suffix}", fontsize=14, fontweight="bold", y=0.9)
            # Compute `k_size` for the next processing step.
            k_size = nas_architecture.get("encoder_kernel", "N/A")
            # Compute `trans` for the next processing step.
            trans = nas_architecture.get("transformer", {})
            # Compute `n_heads` for the next processing step.
            n_heads = trans.get("n_heads", "N/A")
            # Compute `n_layers` for the next processing step.
            n_layers = trans.get("num_encoder_layers", "N/A")
            # Compute `ff_dim` for the next processing step.
            ff_dim = trans.get("dim_ff", "N/A")
            # Compute `text_str` for the next processing step.
            text_str = (
                f"MicroKinetic Encoder:\n"
                f"  - Kernel Size: {k_size}\n\n"
                f"Temporal Transformer:\n"
                f"  - Heads: {n_heads}\n"
                f"  - Layers: {n_layers}\n"
                f"  - FF Dim: {ff_dim}"
            )
            # Invoke `plt.text` to advance this processing stage.
            plt.text(0.1, 0.6, text_str, fontsize=12, family="monospace", va="top")
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig()
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

        # CV summary (final report)
        if cv_summary is not None:
            # Invoke `plt.figure` to advance this processing stage.
            plt.figure(figsize=(12, 6))
            # Invoke `plt.axis` to advance this processing stage.
            plt.axis("off")
            # Invoke `plt.title` to advance this processing stage.
            plt.title("Cross-Validation Summary", fontsize=14, fontweight="bold", y=1.05)
            # Compute `rows` for the next processing step.
            rows = []
            # Iterate `(k, v)` across `cv_summary.items()` to process each element.
            for k, v in cv_summary.items():
                # Invoke `rows.append` to advance this processing stage.
                rows.append([k, v])
            # Compute `table` for the next processing step.
            table = plt.table(cellText=rows, colLabels=["Metric", "Value"],
                              cellLoc="center", loc="center")
            # Invoke `table.auto_set_font_size` to advance this processing stage.
            table.auto_set_font_size(False)
            # Invoke `table.set_fontsize` to advance this processing stage.
            table.set_fontsize(10)
            # Invoke `table.scale` to advance this processing stage.
            table.scale(1, 1.5)
            # Invoke `pdf.savefig` to advance this processing stage.
            pdf.savefig(bbox_inches="tight")
            # Invoke `plt.close` to advance this processing stage.
            plt.close()

    # Invoke `print` to advance this processing stage.
    print(f"Report generated: {report_path}")

