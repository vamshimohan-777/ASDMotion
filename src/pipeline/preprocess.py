import argparse
import csv
import importlib
import json
import math
import os
import time

import cv2
import numpy as np

from src.models.video.mediapipe_layer.landmark_schema import DEFAULT_SCHEMA
from src.pipeline.router import route_video
from src.utils.video_id import make_video_id


TMIN = 2.0
VIDEO_READ_TIMEOUT_SEC = float(os.environ.get("ASDMOTION_VIDEO_READ_TIMEOUT_SEC", "180"))


def _safe_float(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _parse_bool(value, default=False):
    text = _safe_text(value).lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _parse_target_center_from_row(row):
    if row is None:
        return None

    x = _safe_float(row.get("target_x"))
    y = _safe_float(row.get("target_y"))
    if x is None or y is None:
        x = _safe_float(row.get("target_cx"))
        y = _safe_float(row.get("target_cy"))
    if x is None or y is None:
        x = _safe_float(row.get("focus_x"))
        y = _safe_float(row.get("focus_y"))

    if x is None or y is None:
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return (float(x), float(y))


def _parse_is_landmark_video_from_row(row):
    if row is None:
        return False
    for key in ("is_landmark_video", "landmark_video", "is_landmark"):
        if key in row and _safe_text(row.get(key)) != "":
            return _parse_bool(row.get(key), default=False)
    return False


def _open_capture(video_path):
    candidates = []
    for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_DSHOW"):
        if hasattr(cv2, name):
            candidates.append(getattr(cv2, name))
    candidates.append(None)

    for backend in candidates:
        try:
            cap = cv2.VideoCapture(video_path) if backend is None else cv2.VideoCapture(video_path, backend)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None


def _load_landmark_extractor():
    # Import lazily so training/inference can run when preprocessing extras are removed.
    try:
        module = importlib.import_module("src.models.video.mediapipe_layer.extractor")
        extract_holistic_landmarks = getattr(module, "extract_holistic_landmarks")
    except Exception as exc:
        raise RuntimeError(
            "Preprocessing extractor is unavailable (src.models.video.mediapipe_layer.extractor). "
            "Either restore the extractor module or keep data.use_preprocessed=true."
        ) from exc
    return extract_holistic_landmarks


def load_video(video_path, frame_stride=1, max_frames=0):
    if not os.path.exists(video_path):
        print(f"[Video] Missing file: {video_path}")
        return [], 0.0, 0.0

    cap = _open_capture(video_path)
    if cap is None:
        print(f"[Video] Failed to open: {video_path}")
        return [], 0.0, 0.0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps and fps > 0 else 0.0
        stride = max(1, int(frame_stride))
        if max_frames and frame_count > 0:
            stride = max(stride, int(math.ceil(frame_count / float(max_frames))))

        frames = []
        frame_id = 0
        kept = 0
        started = time.time()
        while True:
            if VIDEO_READ_TIMEOUT_SEC > 0 and (time.time() - started) > VIDEO_READ_TIMEOUT_SEC:
                print(f"[Video] Timeout, stopping early: {video_path}")
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_id % stride == 0:
                ts = (frame_id / fps) if fps and fps > 0 else float(frame_id)
                frames.append({"frame_id": frame_id, "timestamp": ts, "image": frame})
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            frame_id += 1
    finally:
        cap.release()

    return frames, float(fps or 0.0), float(duration)


class VideoProcessor:
    def __init__(self, t_min=2.0, frame_stride=1, max_frames=0, schema=DEFAULT_SCHEMA):
        self.t_min = float(t_min)
        self.frame_stride = int(frame_stride)
        self.max_frames = int(max_frames)
        self.schema = schema

    def process_video_file(self, video_path, target_center=None, is_landmark_video=False, save_rgb=False):
        # target_center/is_landmark_video are kept for backward-compatible API.
        del target_center, is_landmark_video
        extract_holistic_landmarks = _load_landmark_extractor()

        frames_raw, fps, duration = load_video(
            video_path,
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
        )
        route = route_video(duration, self.t_min, video_path=video_path)

        processed = []
        for fd in frames_raw:
            xyz, mask, meta = extract_holistic_landmarks(fd["image"], schema=self.schema)
            q = meta["modality_quality"]
            quality = {
                "frame_id": int(fd["frame_id"]),
                "pose_score": float(q.get("pose", 0.0)),
                "hand_score": float(q.get("hands", 0.0)),
                "face_score": float(q.get("face", 0.0)),
                "overall_score": float(meta.get("overall_quality", 0.0)),
            }
            row = {
                "frame_id": int(fd["frame_id"]),
                "timestamp": float(fd["timestamp"]),
                "landmarks": xyz,
                "mask": mask,
                "quality": quality,
            }
            if bool(save_rgb):
                rgb = cv2.cvtColor(fd["image"], cv2.COLOR_BGR2RGB)
                row["rgb_224"] = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            processed.append(row)

        return {
            "fps": float(fps),
            "duration": float(duration),
            "route": route,
            "frames": processed,
            "schema": {
                "pose_joints": int(self.schema.pose_joints),
                "hand_joints": int(self.schema.hand_joints),
                "face_joints": int(self.schema.face_joints),
                "total_joints": int(self.schema.total_joints),
            },
        }


def _patch_existing_meta(meta_path, subject_id=None, label=None):
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    changed = False
    if subject_id is not None and _safe_text(meta.get("subject_id")) != _safe_text(subject_id):
        meta["subject_id"] = _safe_text(subject_id)
        changed = True
    if label is not None and _safe_text(meta.get("label")) != _safe_text(label):
        meta["label"] = _safe_text(label)
        changed = True

    if changed:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return changed


def process_video_to_disk(
    video_path,
    processed_root,
    processor=None,
    subject_id=None,
    label=None,
    target_center=None,
    is_landmark_video=False,
    overwrite=False,
    save_rgb=False,
):
    del target_center, is_landmark_video

    if not os.path.exists(video_path):
        print(f"[Video] Missing file: {video_path}")
        return {"ok": False, "reason": "missing"}

    video_id = make_video_id(video_path, subject_id=subject_id, label=label)
    out_dir = os.path.join(processed_root, video_id)
    meta_path = os.path.join(out_dir, "meta.json")
    if not overwrite and os.path.exists(meta_path):
        patched = _patch_existing_meta(meta_path, subject_id=subject_id, label=label)
        return {"ok": True, "video_id": video_id, "skipped": True, "meta_updated": bool(patched)}

    os.makedirs(out_dir, exist_ok=True)
    proc = processor or VideoProcessor(t_min=TMIN)
    result = proc.process_video_file(video_path, save_rgb=bool(save_rgb))
    frames = result["frames"]

    if not frames:
        return {"ok": False, "video_id": video_id, "reason": "no_frames"}

    landmarks = np.stack([f["landmarks"] for f in frames]).astype(np.float32)  # [T, J, 3]
    masks = np.stack([f["mask"] for f in frames]).astype(np.float32)  # [T, J]
    frame_ids = np.asarray([f["frame_id"] for f in frames], dtype=np.int32)
    timestamps = np.asarray([f["timestamp"] for f in frames], dtype=np.float32)
    quality = [f["quality"] for f in frames]

    np.save(os.path.join(out_dir, "landmarks.npy"), landmarks)
    np.save(os.path.join(out_dir, "landmark_mask.npy"), masks)
    np.save(os.path.join(out_dir, "frame_ids.npy"), frame_ids)
    np.save(os.path.join(out_dir, "timestamps.npy"), timestamps)
    if bool(save_rgb):
        rgb = np.stack([f["rgb_224"] for f in frames]).astype(np.uint8)
        np.save(os.path.join(out_dir, "rgb_224.npy"), rgb)

    with open(os.path.join(out_dir, "quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    meta = {
        "video_id": video_id,
        "video_path": video_path,
        "subject_id": _safe_text(subject_id),
        "label": _safe_text(label),
        "fps": float(result["fps"]),
        "duration": float(result["duration"]),
        "route": result["route"],
        "frame_stride": int(proc.frame_stride),
        "max_frames": int(proc.max_frames),
        "schema": result["schema"],
        "num_frames": int(landmarks.shape[0]),
        "saved_rgb_224": bool(save_rgb),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"ok": True, "video_id": video_id, "frames": int(landmarks.shape[0])}


def precompute_videos(
    csv_path,
    processed_root,
    frame_stride=1,
    max_frames=0,
    t_min=TMIN,
    overwrite=False,
    progress_every=10,
    status_callback=None,
    skip_duplicates=True,
    save_rgb=False,
):
    os.makedirs(processed_root, exist_ok=True)
    processor = VideoProcessor(t_min=t_min, frame_stride=frame_stride, max_frames=max_frames)

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "video_path": row["video_path"].strip(),
                    "subject_id": _safe_text(row.get("subject_id")),
                    "label": _safe_text(row.get("label")),
                    "target_center": _parse_target_center_from_row(row),
                    "is_landmark_video": _parse_is_landmark_video_from_row(row),
                }
            )

    deduped = []
    duplicates = 0
    if skip_duplicates:
        seen = set()
        for row in rows:
            key = (
                row["video_path"],
                row["subject_id"],
                row["label"],
                row["target_center"],
                row["is_landmark_video"],
            )
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            deduped.append(row)
    else:
        deduped = rows

    total = len(deduped)
    ok = 0
    skipped = 0
    failed = 0
    meta_updated = 0
    started = time.time()
    progress_every = max(1, int(progress_every or 1))

    print(
        f"[Precompute] Queue: total_rows={len(rows)} unique_jobs={total} "
        f"duplicates_skipped={duplicates} overwrite={bool(overwrite)}"
    )

    for idx, row in enumerate(deduped, start=1):
        res = process_video_to_disk(
            row["video_path"],
            processed_root,
            processor=processor,
            subject_id=row["subject_id"],
            label=row["label"],
            target_center=row["target_center"],
            is_landmark_video=row["is_landmark_video"],
            overwrite=overwrite,
            save_rgb=bool(save_rgb),
        )
        if res.get("ok") and res.get("skipped"):
            skipped += 1
            if res.get("meta_updated"):
                meta_updated += 1
        elif res.get("ok"):
            ok += 1
        else:
            failed += 1

        elapsed = time.time() - started
        rate = (idx / elapsed) if elapsed > 0 else 0.0
        eta_sec = ((total - idx) / rate) if rate > 0 else None
        if status_callback is not None:
            status_callback(
                {
                    "phase": "precompute",
                    "current_index": idx,
                    "total": total,
                    "ok": ok,
                    "skipped": skipped,
                    "failed": failed,
                    "meta_updated": meta_updated,
                    "duplicates_skipped": duplicates,
                    "current_video": row["video_path"],
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": None if eta_sec is None else round(eta_sec, 1),
                    "jobs_per_sec": round(rate, 4),
                }
            )

        if idx == 1 or idx % progress_every == 0 or idx == total:
            eta_text = "unknown" if eta_sec is None else f"{eta_sec/60.0:.1f}m"
            print(
                f"[Precompute] {idx}/{total} ok={ok} skipped={skipped} failed={failed} "
                f"meta_updated={meta_updated} dup={duplicates} rate={rate:.3f}/s eta={eta_text} "
                f"video={os.path.basename(row['video_path'])}"
            )

    print(
        f"[Precompute] Done. total={total} ok={ok} skipped={skipped} "
        f"failed={failed} meta_updated={meta_updated} duplicates_skipped={duplicates}"
    )
    return {
        "total": total,
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "meta_updated": meta_updated,
        "duplicates_skipped": duplicates,
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute landmark tensors to disk")
    parser.add_argument("--csv", type=str, default=os.environ.get("ASDMOTION_CSV", "data/videos.csv"))
    parser.add_argument(
        "--processed-root",
        type=str,
        default=os.environ.get("ASDMOTION_PROCESSED_ROOT", "data/processed"),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--t-min", type=float, default=float(TMIN))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-rgb", action="store_true", help="Store resized RGB frames as rgb_224.npy")
    args = parser.parse_args()

    precompute_videos(
        csv_path=args.csv,
        processed_root=args.processed_root,
        frame_stride=int(args.frame_stride),
        max_frames=int(args.max_frames),
        t_min=float(args.t_min),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
        save_rgb=bool(args.save_rgb),
    )


if __name__ == "__main__":
    main()
