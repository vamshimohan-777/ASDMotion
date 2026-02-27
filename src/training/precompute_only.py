"""
Precompute-only entrypoint for ASD Motion video preprocessing.

Usage:
  python -m src.training.precompute_only --config config.yaml
  python -m src.training.precompute_only --config config.yaml --csv data/videos_train_mixed.csv
"""

import argparse
import csv
import json
import os
import time

from src.pipeline.preprocess import precompute_videos
from src.utils.config import apply_overrides, load_config


def _write_status_file(status_file: str, payload: dict):
    if not status_file:
        return
    directory = os.path.dirname(status_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{status_file}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, status_file)


def _summarize_csv_rows(csv_path: str):
    total = 0
    landmark = 0
    normal = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            is_landmark = str(row.get("is_landmark_video", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            if is_landmark:
                landmark += 1
            else:
                normal += 1
    return {"total": total, "normal": normal, "landmark": landmark}


def _summarize_processed_root(processed_root: str):
    total = 0
    landmark = 0
    normal = 0
    if not os.path.isdir(processed_root):
        return {"total": 0, "normal": 0, "landmark": 0}

    for name in os.listdir(processed_root):
        base_dir = os.path.join(processed_root, name)
        if not os.path.isdir(base_dir):
            continue
        meta_path = os.path.join(base_dir, "meta.json")
        if not os.path.exists(meta_path):
            continue
        total += 1
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if bool(meta.get("is_landmark_video", False)):
                landmark += 1
            else:
                normal += 1
        except Exception:
            # Keep summary resilient to partially-written or invalid metadata.
            pass
    return {"total": total, "normal": normal, "landmark": landmark}


def run_precompute(cfg, status_file=None):
    data_cfg = cfg.get("data", {})
    csv_path = str(data_cfg.get("csv_path", "data/videos.csv"))
    processed_root = str(data_cfg.get("processed_root", "data/processed"))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    max_frames = int(data_cfg.get("max_frames", 0) or 0)
    overwrite = bool(data_cfg.get("preprocess_overwrite", False))
    progress_every = int(data_cfg.get("precompute_progress_every", 10))

    started = time.time()
    csv_summary = _summarize_csv_rows(csv_path)
    print(
        f"[PrecomputeOnly] CSV rows: total={csv_summary['total']} "
        f"normal={csv_summary['normal']} landmark={csv_summary['landmark']}"
    )

    def emit_status(state, **fields):
        payload = {
            "phase": "precompute",
            "state": state,
            "unix_time": int(time.time()),
            "elapsed_sec": round(time.time() - started, 1),
        }
        payload.update(fields)
        _write_status_file(status_file, payload)

    print("[PrecomputeOnly] Starting preprocessing...")
    emit_status(
        "starting",
        csv_path=csv_path,
        processed_root=processed_root,
        overwrite=overwrite,
        frame_stride=frame_stride,
        max_frames=max_frames,
        csv_total=csv_summary["total"],
        csv_normal=csv_summary["normal"],
        csv_landmark=csv_summary["landmark"],
    )

    summary = precompute_videos(
        csv_path=csv_path,
        processed_root=processed_root,
        frame_stride=frame_stride,
        max_frames=max_frames,
        overwrite=overwrite,
        progress_every=progress_every,
        status_callback=lambda s: emit_status(
            "running",
            **{k: v for k, v in s.items() if k != "phase"},
        ),
    )

    emit_status("done", **summary)
    processed_summary = _summarize_processed_root(processed_root)
    print("[PrecomputeOnly] Done:", summary)
    print(
        f"[PrecomputeOnly] Processed folders with meta: total={processed_summary['total']} "
        f"normal={processed_summary['normal']} landmark={processed_summary['landmark']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute ASD Motion video data only")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--override", type=str, action="append", default=[])
    parser.add_argument("--csv", type=str, default=None, help="Override data.csv_path")
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Optional JSON heartbeat path for status monitoring.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.csv:
        args.override.append(f"data.csv_path={args.csv}")
    cfg = apply_overrides(cfg, args.override)

    run_precompute(cfg, status_file=args.status_file)


if __name__ == "__main__":
    main()
