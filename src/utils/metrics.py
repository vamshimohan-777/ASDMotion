import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def find_optimal_threshold(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1))
    thr = float(thresholds[best_idx])
    return max(0.05, min(0.95, thr))


def compute_basic_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }


def compute_auc(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.5


def sensitivity_at_specificity(labels, probs, target_spec=0.95, min_negatives=20, allow_unstable=True):
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)

    if len(np.unique(labels)) < 2:
        return float("nan")

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    if n_neg < max(int(min_negatives), 1) and not allow_unstable:
        return float("nan")

    fpr, tpr, _ = roc_curve(labels, probs)
    if fpr.size == 0:
        return float("nan")

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    fpr_u, idx = np.unique(fpr, return_index=True)
    tpr_u = tpr[idx]
    tpr_u = np.maximum.accumulate(tpr_u)

    target_fpr = float(np.clip(1.0 - target_spec, 0.0, 1.0))

    if target_fpr <= fpr_u[0]:
        return float(tpr_u[0])
    if target_fpr >= fpr_u[-1]:
        return float(tpr_u[-1])

    return float(np.interp(target_fpr, fpr_u, tpr_u))


def compute_ece(labels, probs, n_bins=10):
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, 1.0)

    total = len(labels)
    if total == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == 0:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs > lo) & (probs <= hi)

        n_in = int(in_bin.sum())
        if n_in == 0:
            continue

        acc = np.mean(labels[in_bin])
        conf = np.mean(probs[in_bin])
        ece += (n_in / total) * abs(acc - conf)

    return float(ece)


def compute_calibration_curve(labels, probs, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    accs = []
    confs = []
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == 0:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs > lo) & (probs <= hi)
        if in_bin.sum() == 0:
            accs.append(np.nan)
            confs.append(np.nan)
        else:
            accs.append(np.mean(labels[in_bin]))
            confs.append(np.mean(probs[in_bin]))
    return np.array(bin_centers), np.array(accs), np.array(confs)


def roc_pr_curves(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    return fpr, tpr, precision, recall


def bootstrap_ci(labels, probs, metric_fn, n_iters=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n == 0:
        return (0.0, 0.0, 0.0)
    stats = []
    for _ in range(n_iters):
        idx = rng.randint(0, n, size=n)
        try:
            stats.append(metric_fn(labels[idx], probs[idx]))
        except Exception:
            continue
    if len(stats) == 0:
        return (0.0, 0.0, 0.0)
    stats = np.sort(stats)
    low = float(np.percentile(stats, 2.5))
    mid = float(np.percentile(stats, 50))
    high = float(np.percentile(stats, 97.5))
    return low, mid, high
