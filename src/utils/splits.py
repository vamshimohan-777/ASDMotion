# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def ensure_subject_id(groups):
    if groups is None:
        raise ValueError(
            "CSV is missing subject_id. Group-aware splitting is required to prevent "
            "leakage. Leakage makes clinical metrics invalid."
        )


def make_group_kfold(labels, groups, n_splits=5, seed=42):
    ensure_subject_id(groups)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(labels)), labels, groups))


def make_group_stratified_split(labels, groups, val_fraction=0.2, seed=42):
    ensure_subject_id(groups)
    if val_fraction <= 0 or val_fraction >= 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    n_splits = max(2, int(round(1.0 / val_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Use the first split for determinism
    train_idx, val_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    return train_idx, val_idx, n_splits


def check_group_overlap(train_groups, val_groups, fold_tag=""):
    train_set = set(train_groups)
    val_set = set(val_groups)
    overlap = train_set.intersection(val_set)
    print(f"  [LeakageCheck]{fold_tag} subject overlap count: {len(overlap)}")
    if overlap:
        print(f"  [LeakageCheck]{fold_tag} OVERLAP SUBJECTS: {sorted(list(overlap))}")
        raise RuntimeError("Subject leakage detected between train and validation.")

