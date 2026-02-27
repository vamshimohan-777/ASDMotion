"""
Training report generation with clinically readable layout and diagnostics.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import pandas as pd

from src.utils.metrics import (
    roc_pr_curves,
    compute_calibration_curve,
)
from src.utils.plotting import ema_smooth


COLOR_PRIMARY = "#103b63"
COLOR_ACCENT = "#1f6aa5"
COLOR_BORDER = "#c9d6e3"
COLOR_SOFT = "#eef4fb"
COLOR_TEXT = "#1b2a3a"
COLOR_MUTED = "#4f6477"


def _fmt(value, digits=3):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{digits}f}"


def _safe_array(values, dtype=float):
    try:
        arr = np.asarray(values, dtype=dtype)
    except Exception:
        return np.array([], dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _style_table(table, header_color=COLOR_SOFT, body_colors=("#ffffff", "#f8fbff"), fontsize=9):
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLOR_BORDER)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=COLOR_PRIMARY)
        else:
            cell.set_facecolor(body_colors[(r - 1) % len(body_colors)])
            cell.set_text_props(color=COLOR_TEXT)


def _plot_confusion_matrix(cm, title):
    cm = np.asarray(cm, dtype=float)
    if cm.shape != (2, 2):
        return None

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.zeros_like(cm, dtype=float)
    np.divide(cm, np.maximum(row_sums, 1.0), out=cm_pct, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            pct = cm_pct[i, j]
            cnt = int(round(cm[i, j]))
            text_color = "white" if pct >= 0.45 else COLOR_TEXT
            ax.text(j, i, f"{cnt}\n({pct * 100:.1f}%)", ha="center", va="center", fontsize=10, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion", fontsize=9)
    fig.tight_layout()
    return fig


def _apply_style():
    original = plt.rcParams.copy()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLOR_BORDER,
        "axes.labelcolor": COLOR_TEXT,
        "axes.titlecolor": COLOR_PRIMARY,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "font.family": "DejaVu Sans",
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": COLOR_BORDER,
        "savefig.facecolor": "white",
    })
    return original


def _cover_page(pdf, title_suffix, eval_summary, cv_summary):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.add_patch(Rectangle((0.03, 0.90), 0.94, 0.07, transform=ax.transAxes, color=COLOR_PRIMARY))
    ax.text(0.05, 0.935, f"ASDMotion Training Report - {title_suffix}", transform=ax.transAxes,
            color="white", fontsize=17, fontweight="bold", va="center")
    ax.text(0.05, 0.892, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            transform=ax.transAxes, color=COLOR_MUTED, fontsize=10)

    labels = _safe_array(eval_summary.get("labels", []), dtype=int) if eval_summary else np.array([], dtype=int)
    n_total = int(labels.size)
    n_pos = int((labels == 1).sum()) if n_total else 0
    n_neg = int((labels == 0).sum()) if n_total else 0
    pos_rate = (n_pos / n_total) if n_total else float("nan")

    summary_rows = [
        ["AUC", _fmt(eval_summary.get("auc")) if eval_summary else "N/A"],
        ["F1 (optimal threshold)", _fmt(eval_summary.get("f1_opt")) if eval_summary else "N/A"],
        ["Accuracy @ 0.5", _fmt(eval_summary.get("acc_05")) if eval_summary else "N/A"],
        ["Accuracy @ optimal threshold", _fmt(eval_summary.get("acc_opt")) if eval_summary else "N/A"],
        ["Sensitivity @ target specificity", _fmt(eval_summary.get("sens_spec")) if eval_summary else "N/A"],
        ["Expected calibration error", _fmt(eval_summary.get("ece")) if eval_summary else "N/A"],
        ["Optimal threshold", _fmt(eval_summary.get("opt_threshold")) if eval_summary else "N/A"],
        ["Temperature", _fmt(eval_summary.get("temperature"), digits=4) if eval_summary else "N/A"],
    ]

    meta_rows = [
        ["Samples in evaluation split", str(n_total)],
        ["Positive samples", str(n_pos)],
        ["Negative samples", str(n_neg)],
        ["Positive prevalence", _fmt(pos_rate)],
        ["Specificity target", _fmt(eval_summary.get("spec_target"), digits=2) if eval_summary else "N/A"],
        ["Calibration bins", str(eval_summary.get("calib_bins", "N/A")) if eval_summary else "N/A"],
        ["Sensitivity@specificity unstable", "Yes" if eval_summary and eval_summary.get("sens_spec_unstable") else "No"],
    ]

    table_left = ax.table(
        cellText=summary_rows,
        colLabels=["Key Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.05, 0.26, 0.43, 0.58],
    )
    _style_table(table_left, fontsize=9)

    table_right = ax.table(
        cellText=meta_rows,
        colLabels=["Run Metadata", "Value"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.52, 0.42, 0.43, 0.42],
    )
    _style_table(table_right, fontsize=9)

    notes = [
        "Report intent: summarize model development diagnostics and calibration behavior.",
        "This document supports QA/research workflows and is not a clinical diagnosis report.",
    ]
    if eval_summary and eval_summary.get("sens_spec_unstable"):
        notes.append("Warning: sensitivity@specificity may be unstable due to limited negatives.")

    note_y = 0.30
    ax.text(0.52, note_y + 0.06, "Interpretation Notes", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=COLOR_PRIMARY)
    for i, text in enumerate(notes):
        ax.text(0.53, note_y + 0.03 - i * 0.035, f"- {text}", transform=ax.transAxes,
                fontsize=9.2, color=COLOR_TEXT)

    if cv_summary:
        cv_preview = list(cv_summary.items())[:4]
        cv_text = " | ".join([f"{k}: {v}" for k, v in cv_preview])
        ax.text(0.05, 0.20, f"Cross-validation preview: {cv_text}",
                transform=ax.transAxes, fontsize=9.5, color=COLOR_MUTED)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_training_dynamics(pdf, df, title_suffix, ema_alpha):
    if df.empty:
        return

    epochs = _safe_array(df["epoch"], dtype=float) if "epoch" in df.columns else np.arange(1, len(df) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"Training Dynamics - {title_suffix}", fontsize=14, fontweight="bold", color=COLOR_PRIMARY)

    ax = axes[0, 0]
    has_loss = ("train_loss" in df.columns) and ("val_loss" in df.columns)
    if has_loss:
        train_loss = ema_smooth(_safe_array(df["train_loss"]).tolist(), ema_alpha)
        val_loss = ema_smooth(_safe_array(df["val_loss"]).tolist(), ema_alpha)
        ax.plot(epochs, train_loss, label="Train Loss", color="#b23b3b", linewidth=2)
        ax.plot(epochs, val_loss, label="Validation Loss", color="#165c91", linewidth=2)
    ax.set_title("Loss Curves", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")

    ax = axes[0, 1]
    metric_plotted = False
    for key, label, color in [
        ("auc", "AUC", "#165c91"),
        ("f1_opt", "F1 (opt)", "#ce7f17"),
        ("sens_spec", "Sens@Spec", "#26734d"),
    ]:
        if key in df.columns:
            values = ema_smooth(_safe_array(df[key]).tolist(), ema_alpha)
            ax.plot(epochs, values, label=label, linewidth=2, color=color)
            metric_plotted = True
    ax.set_title("Primary Scores", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    if metric_plotted:
        ax.legend(loc="lower right")

    ax = axes[1, 0]
    acc_plotted = False
    for key, label, color in [
        ("accuracy_05", "Acc@0.5", "#7a4eab"),
        ("accuracy_opt", "Acc@opt", "#3a8f8f"),
        ("ece", "ECE", "#7f2f2f"),
    ]:
        if key in df.columns:
            values = ema_smooth(_safe_array(df[key]).tolist(), ema_alpha)
            ax.plot(epochs, values, label=label, linewidth=2, color=color)
            acc_plotted = True
    ax.set_title("Accuracy and Calibration Error", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.05)
    if acc_plotted:
        ax.legend(loc="best")

    ax = axes[1, 1]
    if "selection_score" in df.columns:
        selection = ema_smooth(_safe_array(df["selection_score"]).tolist(), ema_alpha)
        ax.plot(epochs, selection, color="#2f455a", linewidth=2, label="Selection Score")
    if "val_loss" in df.columns:
        val_loss_raw = _safe_array(df["val_loss"])
        if val_loss_raw.size:
            ax2 = ax.twinx()
            ax2.plot(epochs, val_loss_raw, color="#c94d58", linewidth=1.4, alpha=0.55, label="Val Loss (raw)")
            ax2.set_ylabel("Val Loss", color="#c94d58")
            ax2.tick_params(axis="y", colors="#c94d58")
    ax.set_title("Model Selection Signal", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Selection Score")
    ax.legend(loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _plot_eval_diagnostics(pdf, eval_summary, title_suffix):
    if not eval_summary or "labels" not in eval_summary or "probs_cal" not in eval_summary:
        return

    labels = _safe_array(eval_summary.get("labels", []), dtype=int)
    probs = _safe_array(eval_summary.get("probs_cal", []), dtype=float)
    if labels.size == 0 or probs.size == 0 or labels.size != probs.size:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"Evaluation Diagnostics - {title_suffix}", fontsize=14, fontweight="bold", color=COLOR_PRIMARY)

    ax_roc = axes[0, 0]
    ax_pr = axes[0, 1]
    ax_cal = axes[1, 0]
    ax_cm = axes[1, 1]

    if np.unique(labels).size >= 2:
        fpr, tpr, precision, recall = roc_pr_curves(labels, probs)
        auc = float(eval_summary.get("auc", np.nan))

        ax_roc.plot(fpr, tpr, color=COLOR_ACCENT, linewidth=2, label=f"ROC (AUC={_fmt(auc)})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
        ax_roc.fill_between(fpr, 0, tpr, color=COLOR_ACCENT, alpha=0.12)
        ax_roc.set_title("Receiver Operating Characteristic", fontsize=11, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")

        order = np.argsort(recall)
        pr_auc = _trapz(precision[order], recall[order]) if recall.size > 1 else np.nan
        ax_pr.plot(recall, precision, color="#d07f19", linewidth=2, label=f"PR (AUC~{_fmt(pr_auc)})")
        ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0.0, 1.0)
        ax_pr.set_ylim(0.0, 1.05)
        ax_pr.legend(loc="lower left")
    else:
        ax_roc.text(0.5, 0.5, "ROC unavailable\n(single-class labels)", ha="center", va="center", fontsize=11)
        ax_pr.text(0.5, 0.5, "PR unavailable\n(single-class labels)", ha="center", va="center", fontsize=11)
        ax_roc.set_axis_off()
        ax_pr.set_axis_off()

    calib_bins = int(eval_summary.get("calib_bins", 10))
    centers, accs, confs = compute_calibration_curve(labels, probs, n_bins=calib_bins)
    valid = np.isfinite(accs) & np.isfinite(confs)
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Ideal")
    if valid.any():
        ax_cal.plot(confs[valid], accs[valid], marker="o", color="#26734d", linewidth=1.8, label="Observed")
        for c, a in zip(confs[valid], accs[valid]):
            ax_cal.vlines(c, min(c, a), max(c, a), color="#8ea9bf", alpha=0.45, linewidth=1)
    ax_cal.set_title("Calibration Reliability", fontsize=11, fontweight="bold")
    ax_cal.set_xlabel("Mean Confidence")
    ax_cal.set_ylabel("Empirical Accuracy")
    ax_cal.set_xlim(0.0, 1.0)
    ax_cal.set_ylim(0.0, 1.0)
    ax_cal.legend(loc="upper left")

    cm = eval_summary.get("confusion_matrix", None)
    if cm is not None:
        cm = np.asarray(cm, dtype=float)
        if cm.shape == (2, 2):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct = np.zeros_like(cm, dtype=float)
            np.divide(cm, np.maximum(row_sums, 1.0), out=cm_pct, where=row_sums > 0)
            im = ax_cm.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["0", "1"])
            ax_cm.set_yticklabels(["0", "1"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Confusion Matrix (row-normalized)", fontsize=11, fontweight="bold")
            for i in range(2):
                for j in range(2):
                    pct = cm_pct[i, j]
                    cnt = int(round(cm[i, j]))
                    color = "white" if pct >= 0.45 else COLOR_TEXT
                    ax_cm.text(j, i, f"{cnt}\n({pct * 100:.1f}%)", ha="center", va="center", fontsize=9.5, color=color)
            fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        else:
            ax_cm.text(0.5, 0.5, "Confusion matrix shape unsupported", ha="center", va="center")
            ax_cm.set_axis_off()
    else:
        ax_cm.text(0.5, 0.5, "Confusion matrix unavailable", ha="center", va="center")
        ax_cm.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _plot_probability_diagnostics(pdf, eval_summary, title_suffix):
    if not eval_summary or "labels" not in eval_summary or "probs_cal" not in eval_summary:
        return

    labels = _safe_array(eval_summary.get("labels", []), dtype=int)
    probs = _safe_array(eval_summary.get("probs_cal", []), dtype=float)
    if labels.size == 0 or probs.size == 0 or labels.size != probs.size:
        return

    binary_thr = 0.50

    fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.8))
    fig.suptitle(f"Probability Distribution and Binary Decisions - {title_suffix}",
                 fontsize=13, fontweight="bold", color=COLOR_PRIMARY)

    ax_hist = axes[0]
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    bins = np.linspace(0, 1, 16)
    if neg.size:
        ax_hist.hist(neg, bins=bins, alpha=0.60, color="#3f78ad", label=f"Negative class ({neg.size})")
    if pos.size:
        ax_hist.hist(pos, bins=bins, alpha=0.60, color="#d27c2c", label=f"Positive class ({pos.size})")
    ax_hist.axvline(binary_thr, color="#3f3f3f", linestyle="--", linewidth=1.5, label="Binary thr (0.50)")
    ax_hist.set_title("Calibrated Probability Histogram", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Calibrated probability")
    ax_hist.set_ylabel("Sample count")
    ax_hist.legend(loc="upper center", fontsize=8.4)

    ax_zone = axes[1]
    z_neg = int((probs < binary_thr).sum())
    z_pos = int((probs >= binary_thr).sum())
    zone_names = ["Negative", "Positive"]
    zone_counts = [z_neg, z_pos]
    colors = ["#3f78ad", "#c6513a"]
    bars = ax_zone.bar(zone_names, zone_counts, color=colors, alpha=0.88)
    ax_zone.set_title("Count by Binary Decision", fontsize=11, fontweight="bold")
    ax_zone.set_ylabel("Sample count")
    for bar, val in zip(bars, zone_counts):
        ax_zone.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(val),
                     ha="center", va="bottom", fontsize=9, color=COLOR_TEXT)
    ax_zone.set_ylim(0, max(zone_counts + [1]) * 1.20)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def _plot_attention_page(pdf, attention_map, title_suffix):
    if attention_map is None:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(attention_map, cmap="viridis", aspect="auto")
    ax.set_title(f"Attention Map - {title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Key Token")
    ax.set_ylabel("Query Token")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_temporal_importance_page(pdf, temporal_importance, title_suffix):
    if temporal_importance is None:
        return
    values = _safe_array(temporal_importance, dtype=float)
    if values.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(values.size)
    ax.plot(xs, values, marker="o", linewidth=1.8, color=COLOR_ACCENT)
    ax.fill_between(xs, 0, values, alpha=0.18, color=COLOR_ACCENT)
    ax.set_title(f"Temporal Importance (approx.) - {title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Event token index")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_nas_page(pdf, nas_architecture, title_suffix):
    if nas_architecture is None:
        return

    fig = plt.figure(figsize=(11.0, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.add_patch(Rectangle((0.05, 0.86), 0.90, 0.09, transform=ax.transAxes, color=COLOR_SOFT, ec=COLOR_BORDER))
    ax.text(0.07, 0.905, f"Neural Architecture Summary - {title_suffix}",
            transform=ax.transAxes, fontsize=15, fontweight="bold", color=COLOR_PRIMARY, va="center")

    trans = nas_architecture.get("transformer", {})
    rows = [
        ["Microkinetic encoder kernel", str(nas_architecture.get("encoder_kernel", "N/A"))],
        ["Transformer heads", str(trans.get("n_heads", "N/A"))],
        ["Transformer encoder layers", str(trans.get("num_encoder_layers", "N/A"))],
        ["Transformer feed-forward dim", str(trans.get("dim_ff", "N/A"))],
    ]
    table = ax.table(cellText=rows, colLabels=["Architecture parameter", "Selected value"],
                     cellLoc="left", colLoc="left", bbox=[0.07, 0.48, 0.86, 0.30])
    _style_table(table, fontsize=11)

    ax.text(0.07, 0.38, "Interpretation", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=COLOR_PRIMARY)
    notes = [
        "The NAS search selected these parameters under group-aware validation.",
        "Performance curves and calibration plots should be interpreted jointly with this architecture choice.",
    ]
    for i, line in enumerate(notes):
        ax.text(0.08, 0.33 - i * 0.05, f"- {line}", transform=ax.transAxes, fontsize=10, color=COLOR_TEXT)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_cv_summary_page(pdf, cv_summary):
    if cv_summary is None:
        return

    rows = [[str(k), str(v)] for k, v in cv_summary.items()]
    if not rows:
        return

    fig = plt.figure(figsize=(11.69, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(Rectangle((0.04, 0.88), 0.92, 0.09, transform=ax.transAxes, color=COLOR_SOFT, ec=COLOR_BORDER))
    ax.text(0.06, 0.925, "Cross-Validation Aggregate Summary", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=COLOR_PRIMARY, va="center")

    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], cellLoc="left", colLoc="left",
                     bbox=[0.06, 0.22, 0.88, 0.58])
    _style_table(table, fontsize=10)

    ax.text(0.06, 0.15,
            "Confidence intervals are bootstrap-based and reflect variability across grouped resampling.",
            transform=ax.transAxes, fontsize=9.5, color=COLOR_MUTED)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_glossary_page(pdf):
    fig = plt.figure(figsize=(11.69, 6.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.add_patch(Rectangle((0.04, 0.88), 0.92, 0.09, transform=ax.transAxes, color=COLOR_SOFT, ec=COLOR_BORDER))
    ax.text(0.06, 0.925, "Metric Glossary", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=COLOR_PRIMARY, va="center")

    glossary = [
        ("AUC", "Area under ROC curve; higher indicates better ranking quality."),
        ("F1 (optimal)", "F1 score at threshold maximizing precision-recall harmonic mean."),
        ("Sens@Spec", "Sensitivity at the configured specificity target."),
        ("ECE", "Expected calibration error between predicted confidence and empirical correctness."),
        ("Temperature", "Post-hoc calibration scaling parameter applied to logits."),
    ]

    rows = [[k, v] for k, v in glossary]
    table = ax.table(cellText=rows, colLabels=["Metric", "Meaning"], cellLoc="left", colLoc="left",
                     bbox=[0.06, 0.16, 0.88, 0.66])
    _style_table(table, fontsize=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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

    original_rc = _apply_style()
    try:
        with PdfPages(report_path) as pdf:
            _cover_page(pdf, title_suffix, eval_summary, cv_summary)
            _plot_training_dynamics(pdf, df, title_suffix, ema_alpha=ema_alpha)
            _plot_eval_diagnostics(pdf, eval_summary, title_suffix)
            _plot_probability_diagnostics(pdf, eval_summary, title_suffix)

            # Keep a dedicated confusion-matrix page for quick printing/reference.
            cm = eval_summary.get("confusion_matrix") if eval_summary else None
            if cm is not None:
                fig_cm = _plot_confusion_matrix(cm, f"Confusion Matrix (opt threshold) - {title_suffix}")
                if fig_cm is not None:
                    pdf.savefig(fig_cm)
                    plt.close(fig_cm)

            _plot_attention_page(pdf, attention_map, title_suffix)
            _plot_temporal_importance_page(pdf, temporal_importance, title_suffix)
            _plot_nas_page(pdf, nas_architecture, title_suffix)
            _plot_cv_summary_page(pdf, cv_summary)
            _plot_metric_glossary_page(pdf)
    finally:
        plt.rcParams.update(original_rc)

    print(f"Report generated: {report_path}")
