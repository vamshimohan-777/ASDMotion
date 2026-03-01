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
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
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
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(
        output_dir,
        f"training_report_fold{fold_idx}.pdf" if fold_idx else "training_report_final.pdf",
    )

    title_suffix = f"Fold {fold_idx}" if fold_idx else "Final Model"

    df = pd.DataFrame(history) if history else pd.DataFrame()

    with PdfPages(report_path) as pdf:
        # Summary page
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        plt.title(f"Training Summary - {title_suffix}", fontsize=14, fontweight="bold", y=1.05)

        summary_rows = []
        if eval_summary:
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
            summary_rows.append(["Summary", "N/A"])

        table = plt.table(cellText=summary_rows, colLabels=["Metric", "Value"],
                          cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        pdf.savefig(bbox_inches="tight")
        plt.close()

        # Loss curves
        if not df.empty and "train_loss" in df.columns and "val_loss" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], ema_smooth(df["train_loss"].tolist(), ema_alpha), label="Train Loss")
            plt.plot(df["epoch"], ema_smooth(df["val_loss"].tolist(), ema_alpha), label="Val Loss")
            plt.title(f"Loss Curves - {title_suffix}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            pdf.savefig()
            plt.close()

        # Metric curves
        if not df.empty and "auc" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], ema_smooth(df["auc"].tolist(), ema_alpha), label="AUC")
            if "f1_opt" in df.columns:
                plt.plot(df["epoch"], ema_smooth(df["f1_opt"].tolist(), ema_alpha), label="F1 (opt)")
            if "accuracy_05" in df.columns:
                plt.plot(df["epoch"], ema_smooth(df["accuracy_05"].tolist(), ema_alpha), label="Acc@0.5")
            if "accuracy_opt" in df.columns:
                plt.plot(df["epoch"], ema_smooth(df["accuracy_opt"].tolist(), ema_alpha), label="Acc@opt")
            plt.title(f"Metrics - {title_suffix}")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            pdf.savefig()
            plt.close()

        # ROC / PR / Calibration / Confusion Matrix
        if eval_summary and "labels" in eval_summary and "probs_cal" in eval_summary:
            labels = np.array(eval_summary["labels"], dtype=int)
            probs = np.array(eval_summary["probs_cal"], dtype=float)

            fpr, tpr, precision, recall = roc_pr_curves(labels, probs)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label="ROC")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC - {title_suffix}")
            plt.grid(True, linestyle="--", alpha=0.5)
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(recall, precision, label="PR")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve - {title_suffix}")
            plt.grid(True, linestyle="--", alpha=0.5)
            pdf.savefig()
            plt.close()

            bin_centers, accs, confs = compute_calibration_curve(labels, probs, n_bins=eval_summary.get("calib_bins", 10))
            plt.figure(figsize=(6, 6))
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.scatter(confs, accs, s=40, c="blue")
            plt.xlabel("Confidence")
            plt.ylabel("Accuracy")
            plt.title(f"Calibration - {title_suffix}")
            plt.grid(True, linestyle="--", alpha=0.5)
            pdf.savefig()
            plt.close()

            cm = eval_summary.get("confusion_matrix")
            if cm is not None:
                fig = _plot_confusion_matrix(cm, f"Confusion Matrix (opt) - {title_suffix}")
                pdf.savefig(fig)
                plt.close(fig)

        # Attention map
        if attention_map is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(attention_map, cmap="viridis", aspect="auto")
            plt.colorbar(label="Attention")
            plt.title(f"Attention Map - {title_suffix}")
            plt.xlabel("Key Token")
            plt.ylabel("Query Token")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Temporal importance
        if temporal_importance is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(temporal_importance, marker="o", linewidth=1.5)
            plt.title(f"Temporal Importance (approx) - {title_suffix}")
            plt.xlabel("Event Token")
            plt.ylabel("Importance")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # NAS architecture
        if nas_architecture is not None:
            plt.figure(figsize=(10, 6))
            plt.axis("off")
            plt.title(f"NAS Architecture - {title_suffix}", fontsize=14, fontweight="bold", y=0.9)
            k_size = nas_architecture.get("encoder_kernel", "N/A")
            trans = nas_architecture.get("transformer", {})
            n_heads = trans.get("n_heads", "N/A")
            n_layers = trans.get("num_encoder_layers", "N/A")
            ff_dim = trans.get("dim_ff", "N/A")
            text_str = (
                f"MicroKinetic Encoder:\n"
                f"  - Kernel Size: {k_size}\n\n"
                f"Temporal Transformer:\n"
                f"  - Heads: {n_heads}\n"
                f"  - Layers: {n_layers}\n"
                f"  - FF Dim: {ff_dim}"
            )
            plt.text(0.1, 0.6, text_str, fontsize=12, family="monospace", va="top")
            pdf.savefig()
            plt.close()

        # CV summary (final report)
        if cv_summary is not None:
            plt.figure(figsize=(12, 6))
            plt.axis("off")
            plt.title("Cross-Validation Summary", fontsize=14, fontweight="bold", y=1.05)
            rows = []
            for k, v in cv_summary.items():
                rows.append([k, v])
            table = plt.table(cellText=rows, colLabels=["Metric", "Value"],
                              cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            pdf.savefig(bbox_inches="tight")
            plt.close()

    print(f"Report generated: {report_path}")

