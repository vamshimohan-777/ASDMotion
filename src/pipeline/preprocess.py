# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import csv
import os
import cv2
import json
import math

from src.pipeline.router import route_video
from src.models.video.mediapipe_layer.extractor import extract_landmarks
from src.models.video.mediapipe_layer.render_pose import render_pose
from src.models.video.mediapipe_layer.aligner import aligned_face_crop
from src.models.video.mediapipe_layer.quality import compute_quality_mask
from src.utils.video_id import make_video_id


CSV_PATH = os.environ.get("ASDMOTION_CSV", "data/videos.csv")
TMIN = 2.0

SKELETON_OUT_ROOT = "data/processed/skeletons"
FACE_OUT_ROOT = "data/processed/faces"
QUALITY_OUT_ROOT = "data/processed/quality"


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
        duration = frame_count / fps if fps > 0 else 0.0

        stride = max(1, int(frame_stride))
        if max_frames and frame_count > 0:
            stride = max(stride, int(math.ceil(frame_count / float(max_frames))))

        frames = []
        frame_id = 0
        kept = 0

        while True:
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"[Video] Read failed for {video_path}: {e}")
                break
            if not ret or frame is None:
                break

            if frame_id % stride == 0:
                timestamp = (frame_id / fps) if fps > 0 else float(frame_id)
                frames.append({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "image": frame
                })
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            frame_id += 1
    finally:
        cap.release()

    return frames, fps, duration


class VideoProcessor:
    def __init__(self, t_min=2.0, frame_stride=1, max_frames=0):
        self.t_min = t_min
        self.frame_stride = frame_stride
        self.max_frames = max_frames

    def process_video_file(self, video_path):
        if not os.path.exists(video_path):
            print(f"[Video] Missing file: {video_path}")
            return {"fps": 0.0, "duration": 0.0, "route": "image", "frames": []}

        cap = _open_capture(video_path)
        if cap is None:
            print(f"[Video] Failed to open: {video_path}")
            return {"fps": 0.0, "duration": 0.0, "route": "image", "frames": []}

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            route = route_video(duration, self.t_min, video_path=video_path)

            stride = max(1, int(self.frame_stride))
            if self.max_frames and frame_count > 0:
                stride = max(stride, int(math.ceil(frame_count / float(self.max_frames))))

            processed_frames = []
            frame_id = 0
            kept = 0

            while True:
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    print(f"[Video] Read failed for {video_path}: {e}")
                    break

                if not ret or frame is None:
                    break

                if frame_id % stride == 0:
                    timestamp = (frame_id / fps) if fps > 0 else float(frame_id)

                    face_landmarks, pose_landmarks = extract_landmarks(frame)

                    quality = compute_quality_mask(
                        frame_id=frame_id,
                        face_landmarks=face_landmarks,
                        pose_landmarks=pose_landmarks
                    )

                    skeleton_img = None
                    if pose_landmarks is not None:
                        skeleton_img = render_pose(pose_landmarks)

                    face_crop = None
                    if face_landmarks is not None:
                        face_crop = aligned_face_crop(frame, face_landmarks)

                    processed_frames.append({
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "skeleton_img": skeleton_img,
                        "face_crop": face_crop,
                        "quality": quality
                    })

                    kept += 1
                    if self.max_frames and kept >= self.max_frames:
                        break

                frame_id += 1
        finally:
            cap.release()

        return {
            "fps": fps,
            "duration": duration,
            "route": route,
            "frames": processed_frames
        }


def main():
    processor = VideoProcessor(t_min=TMIN)

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_path = row["video_path"]
            video_id = os.path.basename(os.path.dirname(video_path))

            print(f"\nProcessing video: {video_path}")

            result = processor.process_video_file(video_path)

            print(f"FPS: {result['fps']}")
            print(f"Total frames: {len(result['frames'])}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Route: {result['route']}")

            skeleton_dir = os.path.join(SKELETON_OUT_ROOT, video_id)
            face_dir = os.path.join(FACE_OUT_ROOT, video_id)
            quality_dir = os.path.join(QUALITY_OUT_ROOT, video_id)

            os.makedirs(skeleton_dir, exist_ok=True)
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(quality_dir, exist_ok=True)

            quality_list = []
            skeleton_saved = 0
            face_saved = 0

            for p_frame in result['frames']:
                frame_id = p_frame['frame_id']
                quality_list.append(p_frame['quality'])

                if p_frame['skeleton_img'] is not None:
                    out_path = os.path.join(skeleton_dir, f"{frame_id:06d}.png")
                    cv2.imwrite(out_path, p_frame['skeleton_img'])
                    skeleton_saved += 1

                if p_frame['face_crop'] is not None:
                    out_path = os.path.join(face_dir, f"{frame_id:06d}.png")
                    cv2.imwrite(out_path, p_frame['face_crop'])
                    face_saved += 1

            quality_path = os.path.join(quality_dir, "quality.json")
            with open(quality_path, "w") as f:
                json.dump(quality_list, f, indent=2)

            print(f"Skeleton frames saved: {skeleton_saved}")
            print(f"Face crops saved: {face_saved}")
            print(f"Quality entries saved: {len(quality_list)}")


def process_video_to_disk(
    video_path,
    processed_root,
    processor=None,
    overwrite=False,
):
    if not os.path.exists(video_path):
        print(f"[Video] Missing file: {video_path}")
        return {"ok": False, "reason": "missing"}

    video_id = make_video_id(video_path)
    out_dir = os.path.join(processed_root, video_id)
    meta_path = os.path.join(out_dir, "meta.json")
    if not overwrite and os.path.exists(meta_path):
        return {"ok": True, "video_id": video_id, "skipped": True}

    os.makedirs(out_dir, exist_ok=True)
    face_dir = os.path.join(out_dir, "faces")
    skeleton_dir = os.path.join(out_dir, "skeletons")
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)

    proc = processor or VideoProcessor(t_min=TMIN)

    cap = _open_capture(video_path)
    if cap is None:
        print(f"[Video] Failed to open: {video_path}")
        return {"ok": False, "reason": "open_failed"}

    frame_ids = []
    timestamps = []
    quality_list = []
    saved_faces = 0
    saved_skeletons = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        route = route_video(duration, proc.t_min, video_path=video_path)

        stride = max(1, int(proc.frame_stride))
        if proc.max_frames and frame_count > 0:
            stride = max(stride, int(math.ceil(frame_count / float(proc.max_frames))))

        frame_id = 0
        kept = 0

        while True:
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"[Video] Read failed for {video_path}: {e}")
                break

            if not ret or frame is None:
                break

            if frame_id % stride == 0:
                timestamp = (frame_id / fps) if fps > 0 else float(frame_id)

                face_landmarks, pose_landmarks = extract_landmarks(frame)

                quality = compute_quality_mask(
                    frame_id=frame_id,
                    face_landmarks=face_landmarks,
                    pose_landmarks=pose_landmarks
                )

                skeleton_img = None
                if pose_landmarks is not None:
                    skeleton_img = render_pose(pose_landmarks)

                face_crop = None
                if face_landmarks is not None:
                    face_crop = aligned_face_crop(frame, face_landmarks)

                if skeleton_img is not None:
                    out_path = os.path.join(skeleton_dir, f"{frame_id:06d}.png")
                    cv2.imwrite(out_path, skeleton_img)
                    saved_skeletons += 1

                if face_crop is not None:
                    out_path = os.path.join(face_dir, f"{frame_id:06d}.png")
                    cv2.imwrite(out_path, face_crop)
                    saved_faces += 1

                frame_ids.append(frame_id)
                timestamps.append(timestamp)
                quality_list.append(quality)

                kept += 1
                if proc.max_frames and kept >= proc.max_frames:
                    break

            frame_id += 1
    finally:
        cap.release()

    meta = {
        "video_id": video_id,
        "video_path": video_path,
        "fps": fps,
        "duration": duration,
        "route": route,
        "frame_stride": stride,
        "max_frames": proc.max_frames,
        "frame_ids": frame_ids,
        "timestamps": timestamps,
        "saved_faces": saved_faces,
        "saved_skeletons": saved_skeletons,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    quality_path = os.path.join(out_dir, "quality.json")
    with open(quality_path, "w") as f:
        json.dump(quality_list, f, indent=2)

    return {
        "ok": True,
        "video_id": video_id,
        "frames": len(frame_ids),
        "saved_faces": saved_faces,
        "saved_skeletons": saved_skeletons,
    }


def precompute_videos(
    csv_path,
    processed_root,
    frame_stride=1,
    max_frames=0,
    t_min=TMIN,
    overwrite=False,
):
    os.makedirs(processed_root, exist_ok=True)
    processor = VideoProcessor(t_min=t_min, frame_stride=frame_stride, max_frames=max_frames)

    total = 0
    ok = 0
    skipped = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row["video_path"].strip()
            total += 1
            res = process_video_to_disk(
                video_path,
                processed_root,
                processor=processor,
                overwrite=overwrite,
            )
            if res.get("ok") and res.get("skipped"):
                skipped += 1
            elif res.get("ok"):
                ok += 1

    print(f"[Precompute] Done. total={total} ok={ok} skipped={skipped}")
    return {"total": total, "ok": ok, "skipped": skipped}


if __name__ == "__main__":
    main()

