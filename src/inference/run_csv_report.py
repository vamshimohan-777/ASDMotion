import argparse
import csv
import json
import os
from datetime import datetime


def _parse_binary_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "1.0", "true", "yes", "asd", "positive", "pos"}:
        return 1
    if text in {"0", "0.0", "false", "no", "non-asd", "negative", "neg"}:
        return 0
    try:
        num = int(float(text))
        return num if num in (0, 1) else None
    except (TypeError, ValueError):
        return None


def _parse_bool_like(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "landmark"}


def _ratio(numerator, denominator):
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _safe_float(value):
    try:
        out = float(value)
        if out != out:  # NaN check
            return None
        return out
    except (TypeError, ValueError):
        return None


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _format_metric(value, digits=4):
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if v != v:
            return "N/A"
        return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _compute_auc(labels, scores):
    n = len(labels)
    if n == 0 or len(scores) != n:
        return None

    n_pos = sum(1 for y in labels if y == 1)
    n_neg = sum(1 for y in labels if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None

    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1

    sum_pos_ranks = 0.0
    for idx, y in enumerate(labels):
        if y == 1:
            sum_pos_ranks += ranks[idx]

    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _compute_binary_metrics(rows):
    eval_rows = [
        r for r in rows
        if r.get("true_label") in (0, 1) and r.get("prob_calibrated") is not None
    ]
    if not eval_rows:
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

    labels = [int(r["true_label"]) for r in eval_rows]
    probs = [float(r["prob_calibrated"]) for r in eval_rows]
    preds = [1 if p >= 0.5 else 0 for p in probs]

    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

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


def _write_markdown_report(path, summary, failed_rows):
    lines = []
    lines.append("# CSV Inference Report")
    lines.append("")
    lines.append(f"- Generated: `{summary['generated_at']}`")
    lines.append(f"- Input CSV: `{summary['input_csv']}`")
    lines.append(f"- Checkpoint: `{summary['checkpoint']}`")
    lines.append(f"- Config: `{summary['config']}`")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Total rows: `{summary['counts']['total_rows']}`")
    lines.append(f"- Succeeded: `{summary['counts']['succeeded']}`")
    lines.append(f"- Failed: `{summary['counts']['failed']}`")
    lines.append(f"- Missing video files: `{summary['counts']['missing_files']}`")
    lines.append("")

    m = summary["metrics_at_0_5_threshold"]
    lines.append("## Binary Metrics (if labels are available)")
    lines.append("")
    lines.append(f"- Evaluated rows: `{m['n_evaluated']}`")
    lines.append(f"- TP: `{m['tp']}`, TN: `{m['tn']}`, FP: `{m['fp']}`, FN: `{m['fn']}`")
    lines.append(f"- Accuracy: `{m['accuracy']}`")
    lines.append(f"- Precision: `{m['precision']}`")
    lines.append(f"- Recall: `{m['recall']}`")
    lines.append(f"- F1: `{m['f1']}`")
    lines.append(f"- AUC: `{m['auc']}`")
    lines.append(f"- Confusion matrix [[TN, FP], [FN, TP]]: `{m['confusion_matrix']}`")
    lines.append("")

    if failed_rows:
        lines.append("## Failed Rows")
        lines.append("")
        lines.append("| row_index | video_path | status | error |")
        lines.append("|---:|---|---|---|")
        for r in failed_rows[:50]:
            video_path = str(r.get("video_path", "")).replace("|", "\\|")
            status = str(r.get("status", "")).replace("|", "\\|")
            error = str(r.get("error", "")).replace("|", "\\|")
            lines.append(f"| {r.get('row_index', '')} | {video_path} | {status} | {error} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_pdf_report(path, summary):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    m = summary["metrics_at_0_5_threshold"]
    cm = np.asarray(m.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=float)
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.05, 0.95, "ASDMotion CSV Inference Report", fontsize=18, weight="bold", va="top")
        ax.text(0.05, 0.91, f"Generated: {summary['generated_at']}", fontsize=10)
        ax.text(0.05, 0.88, f"Input CSV: {summary['input_csv']}", fontsize=10)
        ax.text(0.05, 0.85, f"Checkpoint: {summary['checkpoint']}", fontsize=10)

        metrics_rows = [
            ["AUC", _format_metric(m.get("auc"))],
            ["F1 @ 0.5", _format_metric(m.get("f1"))],
            ["Accuracy @ 0.5", _format_metric(m.get("accuracy"))],
            ["Precision @ 0.5", _format_metric(m.get("precision"))],
            ["Recall @ 0.5", _format_metric(m.get("recall"))],
            ["Evaluated rows", str(m.get("n_evaluated", 0))],
        ]
        table = ax.table(
            cellText=metrics_rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.05, 0.47, 0.45, 0.32],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.2)

        counts_rows = [
            ["Total rows", str(summary["counts"]["total_rows"])],
            ["Succeeded", str(summary["counts"]["succeeded"])],
            ["Failed", str(summary["counts"]["failed"])],
            ["Missing files", str(summary["counts"]["missing_files"])],
            ["TN, FP, FN, TP", f"{tn}, {fp}, {fn}, {tp}"],
        ]
        counts_table = ax.table(
            cellText=counts_rows,
            colLabels=["Count", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.53, 0.47, 0.42, 0.32],
        )
        counts_table.auto_set_font_size(False)
        counts_table.set_fontsize(10)
        counts_table.scale(1.0, 1.2)

        ax.text(
            0.05,
            0.40,
            "Confusion matrix uses threshold 0.5 on calibrated probability.",
            fontsize=10,
        )
        ax.text(
            0.05,
            0.36,
            "Matrix layout: [[TN, FP], [FN, TP]]",
            fontsize=10,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig_cm, ax_cm = plt.subplots(figsize=(7.5, 6.2))
        if cm.shape == (2, 2):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct = np.zeros_like(cm, dtype=float)
            np.divide(cm, np.maximum(row_sums, 1.0), out=cm_pct, where=row_sums > 0)
            im = ax_cm.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
            ax_cm.set_title("Confusion Matrix (row-normalized)", fontsize=13, weight="bold")
            ax_cm.set_xlabel("Predicted label")
            ax_cm.set_ylabel("True label")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["0", "1"])
            ax_cm.set_yticklabels(["0", "1"])
            for i in range(2):
                for j in range(2):
                    cnt = int(cm[i, j])
                    pct = cm_pct[i, j] * 100.0
                    color = "white" if cm_pct[i, j] >= 0.45 else "black"
                    ax_cm.text(j, i, f"{cnt}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11)
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row fraction")
        else:
            ax_cm.axis("off")
            ax_cm.text(0.5, 0.5, "Confusion matrix unavailable", ha="center", va="center", fontsize=12)
        fig_cm.tight_layout()
        pdf.savefig(fig_cm)
        plt.close(fig_cm)


def main():
    parser = argparse.ArgumentParser(description="Run ASD inference for each row in a CSV and generate a report.")
    parser.add_argument("--csv", type=str, default="data/videos.csv", help="Input CSV with at least video_path column.")
    parser.add_argument("--checkpoint", type=str, default="results/asd_pipeline_model.pth")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--output-prefix", type=str, default="inference_csv_report")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="If set, stop immediately when a video file is missing.",
    )
    args = parser.parse_args()

    from src.inference.predictor import ASDPredictor

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    predictor = ASDPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    rows = []
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "video_path" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'video_path' column.")

        for idx, row in enumerate(reader, start=1):
            if args.max_rows > 0 and len(rows) >= args.max_rows:
                break

            video_path = str(row.get("video_path", "")).strip()
            subject_id = str(row.get("subject_id", "")).strip()
            label_raw = row.get("label")
            true_label = _parse_binary_label(label_raw)
            is_landmark_video = _parse_bool_like(row.get("is_landmark_video")) or str(row.get("input_type", "")).strip().lower() == "landmark"

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

            if not video_path:
                out["status"] = "error"
                out["error"] = "missing video_path"
                rows.append(out)
                continue

            if not os.path.exists(video_path):
                out["status"] = "missing_file"
                out["error"] = "video file not found"
                rows.append(out)
                if args.fail_on_missing:
                    break
                continue

            try:
                if is_landmark_video:
                    pred = predictor.predict_landmark_video(video_path)
                else:
                    pred = predictor.predict_video(video_path)
            except Exception as exc:
                out["status"] = "error"
                out["error"] = str(exc)
                rows.append(out)
                continue

            out["status"] = "ok"
            out["decision"] = str(pred.get("decision", ""))
            out["prob_raw"] = _safe_float(pred.get("prob_raw"))
            out["prob_calibrated"] = _safe_float(pred.get("prob_calibrated"))
            out["quality_score"] = _safe_float(pred.get("quality_score"))
            out["threshold_used"] = _safe_float(pred.get("threshold_used"))
            out["abstained"] = bool(pred.get("abstained", False))
            out["inference_ms"] = int(pred.get("inference_ms", 0))

            reasons = pred.get("reasons", [])
            if isinstance(reasons, list):
                out["reasons"] = " | ".join(str(x) for x in reasons)

            events = pred.get("events", [])
            if (not events) and isinstance(pred.get("window_evidence"), list):
                events = pred.get("window_evidence", [])
            if isinstance(events, list):
                event_parts = []
                for e in events[:5]:
                    if not isinstance(e, dict):
                        continue
                    if "event" in e:
                        name = str(e.get("event", "event"))
                        count = e.get("count", "")
                        conf = e.get("mean_confidence", "")
                        event_parts.append(f"{name}(count={count},mean_conf={conf})")
                    else:
                        idx = e.get("window_index", "")
                        score = e.get("window_score", "")
                        start_s = e.get("start_time_sec", "")
                        event_parts.append(
                            f"window_{idx}(score={score},start_sec={start_s})"
                        )
                out["events"] = " | ".join(event_parts)

            if out["prob_calibrated"] is not None:
                out["pred_label_05"] = 1 if out["prob_calibrated"] >= 0.5 else 0
                if out["true_label"] in (0, 1):
                    out["correct_05"] = int(out["pred_label_05"] == out["true_label"])

            rows.append(out)
            print(f"[{idx}] ok: {video_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.output_prefix}_{timestamp}"
    detailed_csv_path = os.path.join(args.output_dir, f"{base}.csv")
    summary_json_path = os.path.join(args.output_dir, f"{base}.summary.json")
    summary_md_path = os.path.join(args.output_dir, f"{base}.md")
    summary_pdf_path = os.path.join(args.output_dir, f"{base}.pdf")

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

    with open(detailed_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    missing_rows = [r for r in rows if r.get("status") == "missing_file"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]

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

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_markdown_report(summary_md_path, summary, failed_rows)
    _write_pdf_report(summary_pdf_path, summary)

    print("Report generation complete.")
    print(f"Detailed CSV: {detailed_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary MD: {summary_md_path}")
    print(f"Summary PDF: {summary_pdf_path}")


if __name__ == "__main__":
    main()
