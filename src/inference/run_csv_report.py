"""Inference module `src/inference/run_csv_report.py` that converts inputs into runtime prediction outputs."""

# Import `argparse` to support computations in this stage of output generation.
import argparse
# Import `csv` to support computations in this stage of output generation.
import csv
# Import `json` to support computations in this stage of output generation.
import json
# Import `os` to support computations in this stage of output generation.
import os
# Import symbols from `datetime` used in this stage's output computation path.
from datetime import datetime


# Define a reusable pipeline function whose outputs feed later steps.
def _parse_binary_label(value):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `value is None` to choose the correct output computation path.
    if value is None:
        # Return `None` as this function's contribution to downstream output flow.
        return None
    # Compute `text` as an intermediate representation used by later output layers.
    text = str(value).strip().lower()
    # Branch on `text == ""` to choose the correct output computation path.
    if text == "":
        # Return `None` as this function's contribution to downstream output flow.
        return None
    # Branch on `text in {"1", "1.0", "true", "yes", "asd", "posit...` to choose the correct output computation path.
    if text in {"1", "1.0", "true", "yes", "asd", "positive", "pos"}:
        # Return `1` as this function's contribution to downstream output flow.
        return 1
    # Branch on `text in {"0", "0.0", "false", "no", "non-asd", "n...` to choose the correct output computation path.
    if text in {"0", "0.0", "false", "no", "non-asd", "negative", "neg"}:
        # Return `0` as this function's contribution to downstream output flow.
        return 0
    # Start guarded block so failures can be handled without breaking output flow.
    try:
        # Set `num` for subsequent steps so the returned prediction payload is correct.
        num = int(float(text))
        # Return `num if num in (0, 1) else None` as this function's contribution to downstream output flow.
        return num if num in (0, 1) else None
    # Handle exceptions and keep output behavior controlled under error conditions.
    except (TypeError, ValueError):
        # Return `None` as this function's contribution to downstream output flow.
        return None


# Define a reusable pipeline function whose outputs feed later steps.
def _parse_bool_like(value):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `value is None` to choose the correct output computation path.
    if value is None:
        # Return `False` as this function's contribution to downstream output flow.
        return False
    # Compute `text` as an intermediate representation used by later output layers.
    text = str(value).strip().lower()
    # Return `text in {"1", "true", "yes", "y", "landmark"}` as this function's contribution to downstream output flow.
    return text in {"1", "true", "yes", "y", "landmark"}


# Define a reusable pipeline function whose outputs feed later steps.
def _ratio(numerator, denominator):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `denominator <= 0` to choose the correct output computation path.
    if denominator <= 0:
        # Return `None` as this function's contribution to downstream output flow.
        return None
    # Return `float(numerator) / float(denominator)` as this function's contribution to downstream output flow.
    return float(numerator) / float(denominator)


# Define a reusable pipeline function whose outputs feed later steps.
def _safe_float(value):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Start guarded block so failures can be handled without breaking output flow.
    try:
        # Set `out` for subsequent steps so the returned prediction payload is correct.
        out = float(value)
        # Branch on `out != out: # NaN check` to choose the correct output computation path.
        if out != out:  # NaN check
            # Return `None` as this function's contribution to downstream output flow.
            return None
        # Return `out` as this function's contribution to downstream output flow.
        return out
    # Handle exceptions and keep output behavior controlled under error conditions.
    except (TypeError, ValueError):
        # Return `None` as this function's contribution to downstream output flow.
        return None


# Define a reusable pipeline function whose outputs feed later steps.
def _now_iso():
    """Executes this routine and returns values used by later pipeline output steps."""
    # Return `datetime.now().isoformat(timespec="seconds")` as this function's contribution to downstream output flow.
    return datetime.now().isoformat(timespec="seconds")


# Define a reusable pipeline function whose outputs feed later steps.
def _format_metric(value, digits=4):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Branch on `value is None` to choose the correct output computation path.
    if value is None:
        # Return `"N/A"` as this function's contribution to downstream output flow.
        return "N/A"
    # Start guarded block so failures can be handled without breaking output flow.
    try:
        # Set `v` for subsequent steps so the returned prediction payload is correct.
        v = float(value)
        # Branch on `v != v` to choose the correct output computation path.
        if v != v:
            # Return `"N/A"` as this function's contribution to downstream output flow.
            return "N/A"
        # Return `f"{v:.{digits}f}"` as this function's contribution to downstream output flow.
        return f"{v:.{digits}f}"
    # Handle exceptions and keep output behavior controlled under error conditions.
    except (TypeError, ValueError):
        # Return `"N/A"` as this function's contribution to downstream output flow.
        return "N/A"


# Define a reusable pipeline function whose outputs feed later steps.
def _compute_auc(labels, scores):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `n` for subsequent steps so the returned prediction payload is correct.
    n = len(labels)
    # Branch on `n == 0 or len(scores) != n` to choose the correct output computation path.
    if n == 0 or len(scores) != n:
        # Return `None` as this function's contribution to downstream output flow.
        return None

    # Call `sum` and use its result in later steps so the returned prediction payload is correct.
    n_pos = sum(1 for y in labels if y == 1)
    # Call `sum` and use its result in later steps so the returned prediction payload is correct.
    n_neg = sum(1 for y in labels if y == 0)
    # Branch on `n_pos == 0 or n_neg == 0` to choose the correct output computation path.
    if n_pos == 0 or n_neg == 0:
        # Return `None` as this function's contribution to downstream output flow.
        return None

    # Compute `indexed` as an intermediate representation used by later output layers.
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    # Set `ranks` for subsequent steps so the returned prediction payload is correct.
    ranks = [0.0] * n

    # Set `i` for subsequent steps so the returned prediction payload is correct.
    i = 0
    # Repeat computation while condition holds, affecting convergence and final outputs.
    # Repeat while `i < n` so iterative updates converge to stable outputs.
    while i < n:
        # Set `j` for subsequent steps so the returned prediction payload is correct.
        j = i
        # Repeat computation while condition holds, affecting convergence and final outputs.
        # Repeat while `j + 1 < n and indexed[j + 1][1] == indexed[i][1]` so iterative updates converge to stable outputs.
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            # Execute this statement so the returned prediction payload is correct.
            j += 1
        # Set `avg_rank` for subsequent steps so the returned prediction payload is correct.
        avg_rank = (i + j + 2) / 2.0
        # Iterate over `range(i, j + 1)` so each item contributes to final outputs/metrics.
        for k in range(i, j + 1):
            # Compute `original_idx` as an intermediate representation used by later output layers.
            original_idx = indexed[k][0]
            # Compute `ranks[original_idx]` as an intermediate representation used by later output layers.
            ranks[original_idx] = avg_rank
        # Set `i` for subsequent steps so the returned prediction payload is correct.
        i = j + 1

    # Set `sum_pos_ranks` for subsequent steps so the returned prediction payload is correct.
    sum_pos_ranks = 0.0
    # Iterate over `enumerate(labels)` so each item contributes to final outputs/metrics.
    for idx, y in enumerate(labels):
        # Branch on `y == 1` to choose the correct output computation path.
        if y == 1:
            # Execute this statement so the returned prediction payload is correct.
            sum_pos_ranks += ranks[idx]

    # Record `auc` as a metric describing current output quality.
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    # Return `float(auc)` as this function's contribution to downstream output flow.
    return float(auc)


# Define a reusable pipeline function whose outputs feed later steps.
def _compute_binary_metrics(rows):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `eval_rows` for subsequent steps so the returned prediction payload is correct.
    eval_rows = [
        r for r in rows
        # Branch on `r.get("true_label") in (0, 1) and r.get("prob_cal...` to choose the correct output computation path.
        if r.get("true_label") in (0, 1) and r.get("prob_calibrated") is not None
    ]
    # Branch on `not eval_rows` to choose the correct output computation path.
    if not eval_rows:
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "n_evaluated": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    # Set `labels` for subsequent steps so the returned prediction payload is correct.
    labels = [int(r["true_label"]) for r in eval_rows]
    # Compute `probs` as confidence values used in final prediction decisions.
    probs = [float(r["prob_calibrated"]) for r in eval_rows]
    # Execute this statement so the returned prediction payload is correct.
    preds = [1 if p >= 0.5 else 0 for p in probs]

    # Call `sum` and use its result in later steps so the returned prediction payload is correct.
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    # Call `sum` and use its result in later steps so the returned prediction payload is correct.
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    # Call `sum` and use its result in later steps so the returned prediction payload is correct.
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    # Call `sum` and use its result in later steps so the returned prediction payload is correct.
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    # Set `precision` for subsequent steps so the returned prediction payload is correct.
    precision = _ratio(tp, tp + fp)
    # Set `recall` for subsequent steps so the returned prediction payload is correct.
    recall = _ratio(tp, tp + fn)
    # Set `f1` for subsequent steps so the returned prediction payload is correct.
    f1 = None
    # Branch on `precision is not None and recall is not None and ...` to choose the correct output computation path.
    if precision is not None and recall is not None and (precision + recall) > 0:
        # Set `f1` for subsequent steps so the returned prediction payload is correct.
        f1 = 2.0 * precision * recall / (precision + recall)

    # Return `{` as this function's contribution to downstream output flow.
    return {
        "n_evaluated": len(eval_rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _ratio(tp + tn, len(eval_rows)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": _compute_auc(labels, probs),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


# Define a reusable pipeline function whose outputs feed later steps.
def _write_markdown_report(path, summary, failed_rows):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `lines` for subsequent steps so the returned prediction payload is correct.
    lines = []
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("# CSV Inference Report")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Generated: `{summary['generated_at']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Input CSV: `{summary['input_csv']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Checkpoint: `{summary['checkpoint']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Config: `{summary['config']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Device: `{summary['device']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("## Counts")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Total rows: `{summary['counts']['total_rows']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Succeeded: `{summary['counts']['succeeded']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Failed: `{summary['counts']['failed']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Missing video files: `{summary['counts']['missing_files']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("")

    # Set `m` for subsequent steps so the returned prediction payload is correct.
    m = summary["metrics_at_0_5_threshold"]
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("## Binary Metrics (if labels are available)")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Evaluated rows: `{m['n_evaluated']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- TP: `{m['tp']}`, TN: `{m['tn']}`, FP: `{m['fp']}`, FN: `{m['fn']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Accuracy: `{m['accuracy']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Precision: `{m['precision']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Recall: `{m['recall']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- F1: `{m['f1']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- AUC: `{m['auc']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append(f"- Confusion matrix [[TN, FP], [FN, TP]]: `{m['confusion_matrix']}`")
    # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
    lines.append("")

    # Branch on `failed_rows` to choose the correct output computation path.
    if failed_rows:
        # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
        lines.append("## Failed Rows")
        # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
        lines.append("")
        # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
        lines.append("| row_index | video_path | status | error |")
        # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
        lines.append("|---:|---|---|---|")
        # Iterate over `failed_rows[:50]` so each item contributes to final outputs/metrics.
        for r in failed_rows[:50]:
            # Compute `video_path` as an intermediate representation used by later output layers.
            video_path = str(r.get("video_path", "")).replace("|", "\\|")
            # Set `status` for subsequent steps so the returned prediction payload is correct.
            status = str(r.get("status", "")).replace("|", "\\|")
            # Set `error` for subsequent steps so the returned prediction payload is correct.
            error = str(r.get("error", "")).replace("|", "\\|")
            # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
            lines.append(f"| {r.get('row_index', '')} | {video_path} | {status} | {error} |")
        # Call `lines.append` and use its result in later steps so the returned prediction payload is correct.
        lines.append("")

    # Use a managed context to safely handle resources used during output computation.
    with open(path, "w", encoding="utf-8") as f:
        # Call `f.write` and use its result in later steps so the returned prediction payload is correct.
        f.write("\n".join(lines))


# Define a reusable pipeline function whose outputs feed later steps.
def _write_pdf_report(path, summary):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Import `numpy as np` to support computations in this stage of output generation.
    import numpy as np
    # Import `matplotlib.pyplot as plt` to support computations in this stage of output generation.
    import matplotlib.pyplot as plt
    # Import symbols from `matplotlib.backends.backend_pdf` used in this stage's output computation path.
    from matplotlib.backends.backend_pdf import PdfPages

    # Set `m` for subsequent steps so the returned prediction payload is correct.
    m = summary["metrics_at_0_5_threshold"]
    # Set `cm` for subsequent steps so the returned prediction payload is correct.
    cm = np.asarray(m.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=float)
    # Call `int` and use its result in later steps so the returned prediction payload is correct.
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    # Call `int` and use its result in later steps so the returned prediction payload is correct.
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    # Call `int` and use its result in later steps so the returned prediction payload is correct.
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    # Call `int` and use its result in later steps so the returned prediction payload is correct.
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    # Use a managed context to safely handle resources used during output computation.
    with PdfPages(path) as pdf:
        # Set `fig` for subsequent steps so the returned prediction payload is correct.
        fig = plt.figure(figsize=(11.69, 8.27))
        # Compute `ax` as an intermediate representation used by later output layers.
        ax = fig.add_axes([0, 0, 1, 1])
        # Call `ax.axis` and use its result in later steps so the returned prediction payload is correct.
        ax.axis("off")
        # Call `ax.text` and use its result in later steps so the returned prediction payload is correct.
        ax.text(0.05, 0.95, "ASDMotion CSV Inference Report", fontsize=18, weight="bold", va="top")
        # Call `ax.text` and use its result in later steps so the returned prediction payload is correct.
        ax.text(0.05, 0.91, f"Generated: {summary['generated_at']}", fontsize=10)
        # Call `ax.text` and use its result in later steps so the returned prediction payload is correct.
        ax.text(0.05, 0.88, f"Input CSV: {summary['input_csv']}", fontsize=10)
        # Call `ax.text` and use its result in later steps so the returned prediction payload is correct.
        ax.text(0.05, 0.85, f"Checkpoint: {summary['checkpoint']}", fontsize=10)

        # Record `metrics_rows` as a metric describing current output quality.
        metrics_rows = [
            ["AUC", _format_metric(m.get("auc"))],
            ["F1 @ 0.5", _format_metric(m.get("f1"))],
            ["Accuracy @ 0.5", _format_metric(m.get("accuracy"))],
            ["Precision @ 0.5", _format_metric(m.get("precision"))],
            ["Recall @ 0.5", _format_metric(m.get("recall"))],
            ["Evaluated rows", str(m.get("n_evaluated", 0))],
        ]
        # Set `table` for subsequent steps so the returned prediction payload is correct.
        table = ax.table(
            cellText=metrics_rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.05, 0.47, 0.45, 0.32],
        )
        # Call `table.auto_set_font_size` and use its result in later steps so the returned prediction payload is correct.
        table.auto_set_font_size(False)
        # Call `table.set_fontsize` and use its result in later steps so the returned prediction payload is correct.
        table.set_fontsize(10)
        # Call `table.scale` and use its result in later steps so the returned prediction payload is correct.
        table.scale(1.0, 1.2)

        # Set `counts_rows` for subsequent steps so the returned prediction payload is correct.
        counts_rows = [
            ["Total rows", str(summary["counts"]["total_rows"])],
            ["Succeeded", str(summary["counts"]["succeeded"])],
            ["Failed", str(summary["counts"]["failed"])],
            ["Missing files", str(summary["counts"]["missing_files"])],
            ["TN, FP, FN, TP", f"{tn}, {fp}, {fn}, {tp}"],
        ]
        # Set `counts_table` for subsequent steps so the returned prediction payload is correct.
        counts_table = ax.table(
            cellText=counts_rows,
            colLabels=["Count", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.53, 0.47, 0.42, 0.32],
        )
        # Call `counts_table.auto_set_font_size` and use its result in later steps so the returned prediction payload is correct.
        counts_table.auto_set_font_size(False)
        # Call `counts_table.set_fontsize` and use its result in later steps so the returned prediction payload is correct.
        counts_table.set_fontsize(10)
        # Call `counts_table.scale` and use its result in later steps so the returned prediction payload is correct.
        counts_table.scale(1.0, 1.2)

        # Call `ax.text` and use its result in later steps so the returned prediction payload is correct.
        ax.text(
            0.05,
            0.40,
            "Confusion matrix uses threshold 0.5 on calibrated probability.",
            fontsize=10,
        )
        # Call `ax.text` and use its result in later steps so the returned prediction payload is correct.
        ax.text(
            0.05,
            0.36,
            "Matrix layout: [[TN, FP], [FN, TP]]",
            fontsize=10,
        )

        # Call `pdf.savefig` and use its result in later steps so the returned prediction payload is correct.
        pdf.savefig(fig, bbox_inches="tight")
        # Call `plt.close` and use its result in later steps so the returned prediction payload is correct.
        plt.close(fig)

        # Compute `fig_cm, ax_cm` as an intermediate representation used by later output layers.
        fig_cm, ax_cm = plt.subplots(figsize=(7.5, 6.2))
        # Branch on `cm.shape == (2, 2)` to choose the correct output computation path.
        if cm.shape == (2, 2):
            # Set `row_sums` for subsequent steps so the returned prediction payload is correct.
            row_sums = cm.sum(axis=1, keepdims=True)
            # Set `cm_pct` for subsequent steps so the returned prediction payload is correct.
            cm_pct = np.zeros_like(cm, dtype=float)
            # Call `np.divide` and use its result in later steps so the returned prediction payload is correct.
            np.divide(cm, np.maximum(row_sums, 1.0), out=cm_pct, where=row_sums > 0)
            # Set `im` for subsequent steps so the returned prediction payload is correct.
            im = ax_cm.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
            # Call `ax_cm.set_title` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_title("Confusion Matrix (row-normalized)", fontsize=13, weight="bold")
            # Call `ax_cm.set_xlabel` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_xlabel("Predicted label")
            # Call `ax_cm.set_ylabel` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_ylabel("True label")
            # Call `ax_cm.set_xticks` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_xticks([0, 1])
            # Call `ax_cm.set_yticks` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_yticks([0, 1])
            # Call `ax_cm.set_xticklabels` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_xticklabels(["0", "1"])
            # Call `ax_cm.set_yticklabels` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.set_yticklabels(["0", "1"])
            # Iterate over `range(2)` so each item contributes to final outputs/metrics.
            for i in range(2):
                # Iterate over `range(2)` so each item contributes to final outputs/metrics.
                for j in range(2):
                    # Set `cnt` for subsequent steps so the returned prediction payload is correct.
                    cnt = int(cm[i, j])
                    # Set `pct` for subsequent steps so the returned prediction payload is correct.
                    pct = cm_pct[i, j] * 100.0
                    # Execute this statement so the returned prediction payload is correct.
                    color = "white" if cm_pct[i, j] >= 0.45 else "black"
                    # Call `ax_cm.text` and use its result in later steps so the returned prediction payload is correct.
                    ax_cm.text(j, i, f"{cnt}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11)
            # Call `fig_cm.colorbar` and use its result in later steps so the returned prediction payload is correct.
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row fraction")
        else:
            # Call `ax_cm.axis` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.axis("off")
            # Call `ax_cm.text` and use its result in later steps so the returned prediction payload is correct.
            ax_cm.text(0.5, 0.5, "Confusion matrix unavailable", ha="center", va="center", fontsize=12)
        # Call `fig_cm.tight_layout` and use its result in later steps so the returned prediction payload is correct.
        fig_cm.tight_layout()
        # Call `pdf.savefig` and use its result in later steps so the returned prediction payload is correct.
        pdf.savefig(fig_cm)
        # Call `plt.close` and use its result in later steps so the returned prediction payload is correct.
        plt.close(fig_cm)


# Define a reusable pipeline function whose outputs feed later steps.
def main():
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `parser` for subsequent steps so the returned prediction payload is correct.
    parser = argparse.ArgumentParser(description="Run ASD inference for each row in a CSV and generate a report.")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--csv", type=str, default="data/videos.csv", help="Input CSV with at least video_path column.")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--checkpoint", type=str, default="results/asd_pipeline_model.pth")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--config", type=str, default=None)
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--device", type=str, default=None)
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--output-dir", type=str, default="results")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--output-prefix", type=str, default="inference_csv_report")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    # Call `parser.add_argument` and use its result in later steps so the returned prediction payload is correct.
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="If set, stop immediately when a video file is missing.",
    )
    # Set `args` for subsequent steps so the returned prediction payload is correct.
    args = parser.parse_args()

    # Import symbols from `src.inference.predictor` used in this stage's output computation path.
    from src.inference.predictor import ASDPredictor

    # Branch on `not os.path.exists(args.csv)` to choose the correct output computation path.
    if not os.path.exists(args.csv):
        # Raise explicit error to stop invalid state from producing misleading outputs.
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    # Branch on `not os.path.exists(args.checkpoint)` to choose the correct output computation path.
    if not os.path.exists(args.checkpoint):
        # Raise explicit error to stop invalid state from producing misleading outputs.
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Call `os.makedirs` and use its result in later steps so the returned prediction payload is correct.
    os.makedirs(args.output_dir, exist_ok=True)

    # Set `predictor` to predicted labels/scores that are reported downstream.
    predictor = ASDPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    # Set `rows` for subsequent steps so the returned prediction payload is correct.
    rows = []
    # Use a managed context to safely handle resources used during output computation.
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        # Set `reader` for subsequent steps so the returned prediction payload is correct.
        reader = csv.DictReader(f)
        # Branch on `not reader.fieldnames or "video_path" not in read...` to choose the correct output computation path.
        if not reader.fieldnames or "video_path" not in reader.fieldnames:
            # Raise explicit error to stop invalid state from producing misleading outputs.
            raise ValueError("CSV must contain a 'video_path' column.")

        # Iterate over `enumerate(reader, start=1)` so each item contributes to final outputs/metrics.
        for idx, row in enumerate(reader, start=1):
            # Branch on `args.max_rows > 0 and len(rows) >= args.max_rows` to choose the correct output computation path.
            if args.max_rows > 0 and len(rows) >= args.max_rows:
                # Stop iteration early to prevent further changes to the current output state.
                break

            # Compute `video_path` as an intermediate representation used by later output layers.
            video_path = str(row.get("video_path", "")).strip()
            # Set `subject_id` for subsequent steps so the returned prediction payload is correct.
            subject_id = str(row.get("subject_id", "")).strip()
            # Set `label_raw` for subsequent steps so the returned prediction payload is correct.
            label_raw = row.get("label")
            # Set `true_label` for subsequent steps so the returned prediction payload is correct.
            true_label = _parse_binary_label(label_raw)
            # Call `_parse_bool_like` and use its result in later steps so the returned prediction payload is correct.
            is_landmark_video = _parse_bool_like(row.get("is_landmark_video")) or str(row.get("input_type", "")).strip().lower() == "landmark"

            # Set `out` for subsequent steps so the returned prediction payload is correct.
            out = {
                "row_index": idx,
                "video_path": video_path,
                "subject_id": subject_id,
                "label_raw": "" if label_raw is None else str(label_raw),
                "true_label": true_label,
                "is_landmark_video": int(is_landmark_video),
                "status": "",
                "error": "",
                "decision": "",
                "prob_raw": None,
                "prob_calibrated": None,
                "quality_score": None,
                "threshold_used": None,
                "abstained": None,
                "pred_label_05": None,
                "correct_05": None,
                "inference_ms": None,
                "reasons": "",
                "events": "",
            }

            # Branch on `not video_path` to choose the correct output computation path.
            if not video_path:
                # Execute this statement so the returned prediction payload is correct.
                out["status"] = "error"
                # Execute this statement so the returned prediction payload is correct.
                out["error"] = "missing video_path"
                # Call `rows.append` and use its result in later steps so the returned prediction payload is correct.
                rows.append(out)
                # Skip current loop item so it does not affect accumulated output state.
                continue

            # Branch on `not os.path.exists(video_path)` to choose the correct output computation path.
            if not os.path.exists(video_path):
                # Execute this statement so the returned prediction payload is correct.
                out["status"] = "missing_file"
                # Execute this statement so the returned prediction payload is correct.
                out["error"] = "video file not found"
                # Call `rows.append` and use its result in later steps so the returned prediction payload is correct.
                rows.append(out)
                # Branch on `args.fail_on_missing` to choose the correct output computation path.
                if args.fail_on_missing:
                    # Stop iteration early to prevent further changes to the current output state.
                    break
                # Skip current loop item so it does not affect accumulated output state.
                continue

            # Start guarded block so failures can be handled without breaking output flow.
            try:
                # Branch on `is_landmark_video` to choose the correct output computation path.
                if is_landmark_video:
                    # Set `pred` to predicted labels/scores that are reported downstream.
                    pred = predictor.predict_landmark_video(video_path)
                else:
                    # Set `pred` to predicted labels/scores that are reported downstream.
                    pred = predictor.predict_video(video_path)
            # Handle exceptions and keep output behavior controlled under error conditions.
            except Exception as exc:
                # Execute this statement so the returned prediction payload is correct.
                out["status"] = "error"
                # Call `str` and use its result in later steps so the returned prediction payload is correct.
                out["error"] = str(exc)
                # Call `rows.append` and use its result in later steps so the returned prediction payload is correct.
                rows.append(out)
                # Skip current loop item so it does not affect accumulated output state.
                continue

            # Execute this statement so the returned prediction payload is correct.
            out["status"] = "ok"
            # Call `str` and use its result in later steps so the returned prediction payload is correct.
            out["decision"] = str(pred.get("decision", ""))
            # Call `_safe_float` and use its result in later steps so the returned prediction payload is correct.
            out["prob_raw"] = _safe_float(pred.get("prob_raw"))
            # Call `_safe_float` and use its result in later steps so the returned prediction payload is correct.
            out["prob_calibrated"] = _safe_float(pred.get("prob_calibrated"))
            # Call `_safe_float` and use its result in later steps so the returned prediction payload is correct.
            out["quality_score"] = _safe_float(pred.get("quality_score"))
            # Call `_safe_float` and use its result in later steps so the returned prediction payload is correct.
            out["threshold_used"] = _safe_float(pred.get("threshold_used"))
            # Call `bool` and use its result in later steps so the returned prediction payload is correct.
            out["abstained"] = bool(pred.get("abstained", False))
            # Call `int` and use its result in later steps so the returned prediction payload is correct.
            out["inference_ms"] = int(pred.get("inference_ms", 0))

            # Set `reasons` for subsequent steps so the returned prediction payload is correct.
            reasons = pred.get("reasons", [])
            # Branch on `isinstance(reasons, list)` to choose the correct output computation path.
            if isinstance(reasons, list):
                # Call `join` and use its result in later steps so the returned prediction payload is correct.
                out["reasons"] = " | ".join(str(x) for x in reasons)

            # Set `events` for subsequent steps so the returned prediction payload is correct.
            events = pred.get("events", [])
            # Branch on `(not events) and isinstance(pred.get("window_evid...` to choose the correct output computation path.
            if (not events) and isinstance(pred.get("window_evidence"), list):
                # Set `events` for subsequent steps so the returned prediction payload is correct.
                events = pred.get("window_evidence", [])
            # Branch on `isinstance(events, list)` to choose the correct output computation path.
            if isinstance(events, list):
                # Set `event_parts` for subsequent steps so the returned prediction payload is correct.
                event_parts = []
                # Iterate over `events[:5]` so each item contributes to final outputs/metrics.
                for e in events[:5]:
                    # Branch on `not isinstance(e, dict)` to choose the correct output computation path.
                    if not isinstance(e, dict):
                        # Skip current loop item so it does not affect accumulated output state.
                        continue
                    # Branch on `"event" in e` to choose the correct output computation path.
                    if "event" in e:
                        # Set `name` for subsequent steps so the returned prediction payload is correct.
                        name = str(e.get("event", "event"))
                        # Set `count` for subsequent steps so the returned prediction payload is correct.
                        count = e.get("count", "")
                        # Set `conf` for subsequent steps so the returned prediction payload is correct.
                        conf = e.get("mean_confidence", "")
                        # Call `event_parts.append` and use its result in later steps so the returned prediction payload is correct.
                        event_parts.append(f"{name}(count={count},mean_conf={conf})")
                    else:
                        # Compute `idx` as an intermediate representation used by later output layers.
                        idx = e.get("window_index", "")
                        # Set `score` for subsequent steps so the returned prediction payload is correct.
                        score = e.get("window_score", "")
                        # Set `start_s` for subsequent steps so the returned prediction payload is correct.
                        start_s = e.get("start_time_sec", "")
                        # Call `event_parts.append` and use its result in later steps so the returned prediction payload is correct.
                        event_parts.append(
                            f"window_{idx}(score={score},start_sec={start_s})"
                        )
                # Call `join` and use its result in later steps so the returned prediction payload is correct.
                out["events"] = " | ".join(event_parts)

            # Branch on `out["prob_calibrated"] is not None` to choose the correct output computation path.
            if out["prob_calibrated"] is not None:
                # Execute this statement so the returned prediction payload is correct.
                out["pred_label_05"] = 1 if out["prob_calibrated"] >= 0.5 else 0
                # Branch on `out["true_label"] in (0, 1)` to choose the correct output computation path.
                if out["true_label"] in (0, 1):
                    # Call `int` and use its result in later steps so the returned prediction payload is correct.
                    out["correct_05"] = int(out["pred_label_05"] == out["true_label"])

            # Call `rows.append` and use its result in later steps so the returned prediction payload is correct.
            rows.append(out)
            # Log runtime values to verify that output computation is behaving as expected.
            print(f"[{idx}] ok: {video_path}")

    # Set `timestamp` for subsequent steps so the returned prediction payload is correct.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set `base` for subsequent steps so the returned prediction payload is correct.
    base = f"{args.output_prefix}_{timestamp}"
    # Compute `detailed_csv_path` as an intermediate representation used by later output layers.
    detailed_csv_path = os.path.join(args.output_dir, f"{base}.csv")
    # Compute `summary_json_path` as an intermediate representation used by later output layers.
    summary_json_path = os.path.join(args.output_dir, f"{base}.summary.json")
    # Compute `summary_md_path` as an intermediate representation used by later output layers.
    summary_md_path = os.path.join(args.output_dir, f"{base}.md")
    # Compute `summary_pdf_path` as an intermediate representation used by later output layers.
    summary_pdf_path = os.path.join(args.output_dir, f"{base}.pdf")

    # Set `fieldnames` for subsequent steps so the returned prediction payload is correct.
    fieldnames = [
        "row_index",
        "video_path",
        "subject_id",
        "label_raw",
        "true_label",
        "is_landmark_video",
        "status",
        "error",
        "decision",
        "prob_raw",
        "prob_calibrated",
        "quality_score",
        "threshold_used",
        "abstained",
        "pred_label_05",
        "correct_05",
        "inference_ms",
        "reasons",
        "events",
    ]

    # Use a managed context to safely handle resources used during output computation.
    with open(detailed_csv_path, "w", encoding="utf-8", newline="") as f:
        # Set `writer` for subsequent steps so the returned prediction payload is correct.
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Call `writer.writeheader` and use its result in later steps so the returned prediction payload is correct.
        writer.writeheader()
        # Call `writer.writerows` and use its result in later steps so the returned prediction payload is correct.
        writer.writerows(rows)

    # Call `r.get` and use its result in later steps so the returned prediction payload is correct.
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    # Call `r.get` and use its result in later steps so the returned prediction payload is correct.
    missing_rows = [r for r in rows if r.get("status") == "missing_file"]
    # Call `r.get` and use its result in later steps so the returned prediction payload is correct.
    failed_rows = [r for r in rows if r.get("status") != "ok"]

    # Set `summary` for subsequent steps so the returned prediction payload is correct.
    summary = {
        "generated_at": _now_iso(),
        "input_csv": args.csv,
        "checkpoint": args.checkpoint,
        "config": args.config if args.config else "embedded_or_default",
        "device": args.device if args.device else "auto",
        "counts": {
            "total_rows": len(rows),
            "succeeded": len(ok_rows),
            "failed": len(failed_rows),
            "missing_files": len(missing_rows),
        },
        "metrics_at_0_5_threshold": _compute_binary_metrics(ok_rows),
        "outputs": {
            "detailed_csv": detailed_csv_path,
            "summary_json": summary_json_path,
            "summary_markdown": summary_md_path,
            "summary_pdf": summary_pdf_path,
        },
    }

    # Use a managed context to safely handle resources used during output computation.
    with open(summary_json_path, "w", encoding="utf-8") as f:
        # Call `json.dump` and use its result in later steps so the returned prediction payload is correct.
        json.dump(summary, f, indent=2)

    # Call `_write_markdown_report` and use its result in later steps so the returned prediction payload is correct.
    _write_markdown_report(summary_md_path, summary, failed_rows)
    # Call `_write_pdf_report` and use its result in later steps so the returned prediction payload is correct.
    _write_pdf_report(summary_pdf_path, summary)

    # Log runtime values to verify that output computation is behaving as expected.
    print("Report generation complete.")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"Detailed CSV: {detailed_csv_path}")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"Summary JSON: {summary_json_path}")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"Summary MD: {summary_md_path}")
    # Log runtime values to verify that output computation is behaving as expected.
    print(f"Summary PDF: {summary_pdf_path}")


# Branch on `__name__ == "__main__"` to choose the correct output computation path.
if __name__ == "__main__":
    # Call `main` and use its result in later steps so the returned prediction payload is correct.
    main()
