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


# Compute `CSV_PATH` for the next processing step.
CSV_PATH = os.environ.get("ASDMOTION_CSV", "data/videos.csv")
# Compute `TMIN` for the next processing step.
TMIN = 2.0

# Compute `SKELETON_OUT_ROOT` for the next processing step.
SKELETON_OUT_ROOT = "data/processed/skeletons"
# Compute `FACE_OUT_ROOT` for the next processing step.
FACE_OUT_ROOT = "data/processed/faces"
# Compute `QUALITY_OUT_ROOT` for the next processing step.
QUALITY_OUT_ROOT = "data/processed/quality"


def _open_capture(video_path):
    # Compute `candidates` for the next processing step.
    candidates = []
    # Iterate `name` across `('CAP_FFMPEG', 'CAP_MSMF', 'CAP_D...` to process each element.
    for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_DSHOW"):
        # Branch behavior based on the current runtime condition.
        if hasattr(cv2, name):
            # Invoke `candidates.append` to advance this processing stage.
            candidates.append(getattr(cv2, name))
    # Invoke `candidates.append` to advance this processing stage.
    candidates.append(None)

    # Iterate `backend` across `candidates` to process each element.
    for backend in candidates:
        # Guard this block and recover cleanly from expected failures.
        try:
            # Compute `cap` for the next processing step.
            cap = cv2.VideoCapture(video_path) if backend is None else cv2.VideoCapture(video_path, backend)
        except Exception:
            continue
        # Branch behavior based on the current runtime condition.
        if cap is not None and cap.isOpened():
            # Return the result expected by the caller.
            return cap
        # Branch behavior based on the current runtime condition.
        if cap is not None:
            # Invoke `cap.release` to advance this processing stage.
            cap.release()
    # Return the result expected by the caller.
    return None


def load_video(video_path, frame_stride=1, max_frames=0):
    # Branch behavior based on the current runtime condition.
    if not os.path.exists(video_path):
        # Invoke `print` to advance this processing stage.
        print(f"[Video] Missing file: {video_path}")
        # Return the result expected by the caller.
        return [], 0.0, 0.0

    # Compute `cap` for the next processing step.
    cap = _open_capture(video_path)
    # Branch behavior based on the current runtime condition.
    if cap is None:
        # Invoke `print` to advance this processing stage.
        print(f"[Video] Failed to open: {video_path}")
        # Return the result expected by the caller.
        return [], 0.0, 0.0

    # Guard this block and recover cleanly from expected failures.
    try:
        # Compute `fps` for the next processing step.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Compute `frame_count` for the next processing step.
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute `duration` for the next processing step.
        duration = frame_count / fps if fps > 0 else 0.0

        # Compute `stride` for the next processing step.
        stride = max(1, int(frame_stride))
        # Branch behavior based on the current runtime condition.
        if max_frames and frame_count > 0:
            # Compute `stride` for the next processing step.
            stride = max(stride, int(math.ceil(frame_count / float(max_frames))))

        # Compute `frames` for the next processing step.
        frames = []
        # Compute `frame_id` for the next processing step.
        frame_id = 0
        # Compute `kept` for the next processing step.
        kept = 0

        # Continue looping until this condition no longer holds.
        while True:
            # Guard this block and recover cleanly from expected failures.
            try:
                # Compute `(ret, frame)` for the next processing step.
                ret, frame = cap.read()
            except Exception as e:
                # Invoke `print` to advance this processing stage.
                print(f"[Video] Read failed for {video_path}: {e}")
                break
            # Branch behavior based on the current runtime condition.
            if not ret or frame is None:
                break

            # Branch behavior based on the current runtime condition.
            if frame_id % stride == 0:
                # Compute `timestamp` for the next processing step.
                timestamp = (frame_id / fps) if fps > 0 else float(frame_id)
                # Invoke `frames.append` to advance this processing stage.
                frames.append({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "image": frame
                })
                # Update `kept` in place using the latest contribution.
                kept += 1
                # Branch behavior based on the current runtime condition.
                if max_frames and kept >= max_frames:
                    break
            # Update `frame_id` in place using the latest contribution.
            frame_id += 1
    finally:
        # Invoke `cap.release` to advance this processing stage.
        cap.release()

    # Return the result expected by the caller.
    return frames, fps, duration


class VideoProcessor:
    def __init__(self, t_min=2.0, frame_stride=1, max_frames=0):
        # Compute `self.t_min` for the next processing step.
        self.t_min = t_min
        # Compute `self.frame_stride` for the next processing step.
        self.frame_stride = frame_stride
        # Compute `self.max_frames` for the next processing step.
        self.max_frames = max_frames

    def process_video_file(self, video_path):
        # Branch behavior based on the current runtime condition.
        if not os.path.exists(video_path):
            # Invoke `print` to advance this processing stage.
            print(f"[Video] Missing file: {video_path}")
            # Return the result expected by the caller.
            return {"fps": 0.0, "duration": 0.0, "route": "image", "frames": []}

        # Compute `cap` for the next processing step.
        cap = _open_capture(video_path)
        # Branch behavior based on the current runtime condition.
        if cap is None:
            # Invoke `print` to advance this processing stage.
            print(f"[Video] Failed to open: {video_path}")
            # Return the result expected by the caller.
            return {"fps": 0.0, "duration": 0.0, "route": "image", "frames": []}

        # Guard this block and recover cleanly from expected failures.
        try:
            # Compute `fps` for the next processing step.
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Compute `frame_count` for the next processing step.
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Compute `duration` for the next processing step.
            duration = frame_count / fps if fps > 0 else 0.0
            # Compute `route` for the next processing step.
            route = route_video(duration, self.t_min, video_path=video_path)

            # Compute `stride` for the next processing step.
            stride = max(1, int(self.frame_stride))
            # Branch behavior based on the current runtime condition.
            if self.max_frames and frame_count > 0:
                # Compute `stride` for the next processing step.
                stride = max(stride, int(math.ceil(frame_count / float(self.max_frames))))

            # Compute `processed_frames` for the next processing step.
            processed_frames = []
            # Compute `frame_id` for the next processing step.
            frame_id = 0
            # Compute `kept` for the next processing step.
            kept = 0

            # Continue looping until this condition no longer holds.
            while True:
                # Guard this block and recover cleanly from expected failures.
                try:
                    # Compute `(ret, frame)` for the next processing step.
                    ret, frame = cap.read()
                except Exception as e:
                    # Invoke `print` to advance this processing stage.
                    print(f"[Video] Read failed for {video_path}: {e}")
                    break

                # Branch behavior based on the current runtime condition.
                if not ret or frame is None:
                    break

                # Branch behavior based on the current runtime condition.
                if frame_id % stride == 0:
                    # Compute `timestamp` for the next processing step.
                    timestamp = (frame_id / fps) if fps > 0 else float(frame_id)

                    # Compute `(face_landmarks, pose_landmarks)` for the next processing step.
                    face_landmarks, pose_landmarks = extract_landmarks(frame)

                    # Compute `quality` for the next processing step.
                    quality = compute_quality_mask(
                        frame_id=frame_id,
                        face_landmarks=face_landmarks,
                        pose_landmarks=pose_landmarks
                    )

                    # Compute `skeleton_img` for the next processing step.
                    skeleton_img = None
                    # Branch behavior based on the current runtime condition.
                    if pose_landmarks is not None:
                        # Compute `skeleton_img` for the next processing step.
                        skeleton_img = render_pose(pose_landmarks)

                    # Compute `face_crop` for the next processing step.
                    face_crop = None
                    # Branch behavior based on the current runtime condition.
                    if face_landmarks is not None:
                        # Compute `face_crop` for the next processing step.
                        face_crop = aligned_face_crop(frame, face_landmarks)

                    # Invoke `processed_frames.append` to advance this processing stage.
                    processed_frames.append({
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "skeleton_img": skeleton_img,
                        "face_crop": face_crop,
                        "quality": quality
                    })

                    # Update `kept` in place using the latest contribution.
                    kept += 1
                    # Branch behavior based on the current runtime condition.
                    if self.max_frames and kept >= self.max_frames:
                        break

                # Update `frame_id` in place using the latest contribution.
                frame_id += 1
        finally:
            # Invoke `cap.release` to advance this processing stage.
            cap.release()

        # Return the result expected by the caller.
        return {
            "fps": fps,
            "duration": duration,
            "route": route,
            "frames": processed_frames
        }


def main():
    # Compute `processor` for the next processing step.
    processor = VideoProcessor(t_min=TMIN)

    # Run this block with managed resources/context cleanup.
    with open(CSV_PATH, newline="") as f:
        # Compute `reader` for the next processing step.
        reader = csv.DictReader(f)

        # Iterate `row` across `reader` to process each element.
        for row in reader:
            # Compute `video_path` for the next processing step.
            video_path = row["video_path"]
            # Compute `video_id` for the next processing step.
            video_id = os.path.basename(os.path.dirname(video_path))

            # Invoke `print` to advance this processing stage.
            print(f"\nProcessing video: {video_path}")

            # Compute `result` for the next processing step.
            result = processor.process_video_file(video_path)

            # Invoke `print` to advance this processing stage.
            print(f"FPS: {result['fps']}")
            # Invoke `print` to advance this processing stage.
            print(f"Total frames: {len(result['frames'])}")
            # Invoke `print` to advance this processing stage.
            print(f"Duration: {result['duration']:.2f}s")
            # Invoke `print` to advance this processing stage.
            print(f"Route: {result['route']}")

            # Compute `skeleton_dir` for the next processing step.
            skeleton_dir = os.path.join(SKELETON_OUT_ROOT, video_id)
            # Compute `face_dir` for the next processing step.
            face_dir = os.path.join(FACE_OUT_ROOT, video_id)
            # Compute `quality_dir` for the next processing step.
            quality_dir = os.path.join(QUALITY_OUT_ROOT, video_id)

            # Invoke `os.makedirs` to advance this processing stage.
            os.makedirs(skeleton_dir, exist_ok=True)
            # Invoke `os.makedirs` to advance this processing stage.
            os.makedirs(face_dir, exist_ok=True)
            # Invoke `os.makedirs` to advance this processing stage.
            os.makedirs(quality_dir, exist_ok=True)

            # Compute `quality_list` for the next processing step.
            quality_list = []
            # Compute `skeleton_saved` for the next processing step.
            skeleton_saved = 0
            # Compute `face_saved` for the next processing step.
            face_saved = 0

            # Iterate `p_frame` across `result['frames']` to process each element.
            for p_frame in result['frames']:
                # Compute `frame_id` for the next processing step.
                frame_id = p_frame['frame_id']
                # Invoke `quality_list.append` to advance this processing stage.
                quality_list.append(p_frame['quality'])

                # Branch behavior based on the current runtime condition.
                if p_frame['skeleton_img'] is not None:
                    # Compute `out_path` for the next processing step.
                    out_path = os.path.join(skeleton_dir, f"{frame_id:06d}.png")
                    # Invoke `cv2.imwrite` to advance this processing stage.
                    cv2.imwrite(out_path, p_frame['skeleton_img'])
                    # Update `skeleton_saved` in place using the latest contribution.
                    skeleton_saved += 1

                # Branch behavior based on the current runtime condition.
                if p_frame['face_crop'] is not None:
                    # Compute `out_path` for the next processing step.
                    out_path = os.path.join(face_dir, f"{frame_id:06d}.png")
                    # Invoke `cv2.imwrite` to advance this processing stage.
                    cv2.imwrite(out_path, p_frame['face_crop'])
                    # Update `face_saved` in place using the latest contribution.
                    face_saved += 1

            # Compute `quality_path` for the next processing step.
            quality_path = os.path.join(quality_dir, "quality.json")
            # Run this block with managed resources/context cleanup.
            with open(quality_path, "w") as f:
                # Invoke `json.dump` to advance this processing stage.
                json.dump(quality_list, f, indent=2)

            # Invoke `print` to advance this processing stage.
            print(f"Skeleton frames saved: {skeleton_saved}")
            # Invoke `print` to advance this processing stage.
            print(f"Face crops saved: {face_saved}")
            # Invoke `print` to advance this processing stage.
            print(f"Quality entries saved: {len(quality_list)}")


def process_video_to_disk(
    video_path,
    processed_root,
    processor=None,
    overwrite=False,
):
    # Branch behavior based on the current runtime condition.
    if not os.path.exists(video_path):
        # Invoke `print` to advance this processing stage.
        print(f"[Video] Missing file: {video_path}")
        # Return the result expected by the caller.
        return {"ok": False, "reason": "missing"}

    # Compute `video_id` for the next processing step.
    video_id = make_video_id(video_path)
    # Compute `out_dir` for the next processing step.
    out_dir = os.path.join(processed_root, video_id)
    # Compute `meta_path` for the next processing step.
    meta_path = os.path.join(out_dir, "meta.json")
    # Branch behavior based on the current runtime condition.
    if not overwrite and os.path.exists(meta_path):
        # Return the result expected by the caller.
        return {"ok": True, "video_id": video_id, "skipped": True}

    # Invoke `os.makedirs` to advance this processing stage.
    os.makedirs(out_dir, exist_ok=True)
    # Compute `face_dir` for the next processing step.
    face_dir = os.path.join(out_dir, "faces")
    # Compute `skeleton_dir` for the next processing step.
    skeleton_dir = os.path.join(out_dir, "skeletons")
    # Invoke `os.makedirs` to advance this processing stage.
    os.makedirs(face_dir, exist_ok=True)
    # Invoke `os.makedirs` to advance this processing stage.
    os.makedirs(skeleton_dir, exist_ok=True)

    # Compute `proc` for the next processing step.
    proc = processor or VideoProcessor(t_min=TMIN)

    # Compute `cap` for the next processing step.
    cap = _open_capture(video_path)
    # Branch behavior based on the current runtime condition.
    if cap is None:
        # Invoke `print` to advance this processing stage.
        print(f"[Video] Failed to open: {video_path}")
        # Return the result expected by the caller.
        return {"ok": False, "reason": "open_failed"}

    # Compute `frame_ids` for the next processing step.
    frame_ids = []
    # Compute `timestamps` for the next processing step.
    timestamps = []
    # Compute `quality_list` for the next processing step.
    quality_list = []
    # Compute `saved_faces` for the next processing step.
    saved_faces = 0
    # Compute `saved_skeletons` for the next processing step.
    saved_skeletons = 0

    # Guard this block and recover cleanly from expected failures.
    try:
        # Compute `fps` for the next processing step.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Compute `frame_count` for the next processing step.
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute `duration` for the next processing step.
        duration = frame_count / fps if fps > 0 else 0.0
        # Compute `route` for the next processing step.
        route = route_video(duration, proc.t_min, video_path=video_path)

        # Compute `stride` for the next processing step.
        stride = max(1, int(proc.frame_stride))
        # Branch behavior based on the current runtime condition.
        if proc.max_frames and frame_count > 0:
            # Compute `stride` for the next processing step.
            stride = max(stride, int(math.ceil(frame_count / float(proc.max_frames))))

        # Compute `frame_id` for the next processing step.
        frame_id = 0
        # Compute `kept` for the next processing step.
        kept = 0

        # Continue looping until this condition no longer holds.
        while True:
            # Guard this block and recover cleanly from expected failures.
            try:
                # Compute `(ret, frame)` for the next processing step.
                ret, frame = cap.read()
            except Exception as e:
                # Invoke `print` to advance this processing stage.
                print(f"[Video] Read failed for {video_path}: {e}")
                break

            # Branch behavior based on the current runtime condition.
            if not ret or frame is None:
                break

            # Branch behavior based on the current runtime condition.
            if frame_id % stride == 0:
                # Compute `timestamp` for the next processing step.
                timestamp = (frame_id / fps) if fps > 0 else float(frame_id)

                # Compute `(face_landmarks, pose_landmarks)` for the next processing step.
                face_landmarks, pose_landmarks = extract_landmarks(frame)

                # Compute `quality` for the next processing step.
                quality = compute_quality_mask(
                    frame_id=frame_id,
                    face_landmarks=face_landmarks,
                    pose_landmarks=pose_landmarks
                )

                # Compute `skeleton_img` for the next processing step.
                skeleton_img = None
                # Branch behavior based on the current runtime condition.
                if pose_landmarks is not None:
                    # Compute `skeleton_img` for the next processing step.
                    skeleton_img = render_pose(pose_landmarks)

                # Compute `face_crop` for the next processing step.
                face_crop = None
                # Branch behavior based on the current runtime condition.
                if face_landmarks is not None:
                    # Compute `face_crop` for the next processing step.
                    face_crop = aligned_face_crop(frame, face_landmarks)

                # Branch behavior based on the current runtime condition.
                if skeleton_img is not None:
                    # Compute `out_path` for the next processing step.
                    out_path = os.path.join(skeleton_dir, f"{frame_id:06d}.png")
                    # Invoke `cv2.imwrite` to advance this processing stage.
                    cv2.imwrite(out_path, skeleton_img)
                    # Update `saved_skeletons` in place using the latest contribution.
                    saved_skeletons += 1

                # Branch behavior based on the current runtime condition.
                if face_crop is not None:
                    # Compute `out_path` for the next processing step.
                    out_path = os.path.join(face_dir, f"{frame_id:06d}.png")
                    # Invoke `cv2.imwrite` to advance this processing stage.
                    cv2.imwrite(out_path, face_crop)
                    # Update `saved_faces` in place using the latest contribution.
                    saved_faces += 1

                # Invoke `frame_ids.append` to advance this processing stage.
                frame_ids.append(frame_id)
                # Invoke `timestamps.append` to advance this processing stage.
                timestamps.append(timestamp)
                # Invoke `quality_list.append` to advance this processing stage.
                quality_list.append(quality)

                # Update `kept` in place using the latest contribution.
                kept += 1
                # Branch behavior based on the current runtime condition.
                if proc.max_frames and kept >= proc.max_frames:
                    break

            # Update `frame_id` in place using the latest contribution.
            frame_id += 1
    finally:
        # Invoke `cap.release` to advance this processing stage.
        cap.release()

    # Compute `meta` for the next processing step.
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

    # Run this block with managed resources/context cleanup.
    with open(meta_path, "w") as f:
        # Invoke `json.dump` to advance this processing stage.
        json.dump(meta, f, indent=2)

    # Compute `quality_path` for the next processing step.
    quality_path = os.path.join(out_dir, "quality.json")
    # Run this block with managed resources/context cleanup.
    with open(quality_path, "w") as f:
        # Invoke `json.dump` to advance this processing stage.
        json.dump(quality_list, f, indent=2)

    # Return the result expected by the caller.
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
    # Invoke `os.makedirs` to advance this processing stage.
    os.makedirs(processed_root, exist_ok=True)
    # Compute `processor` for the next processing step.
    processor = VideoProcessor(t_min=t_min, frame_stride=frame_stride, max_frames=max_frames)

    # Compute `total` for the next processing step.
    total = 0
    # Compute `ok` for the next processing step.
    ok = 0
    # Compute `skipped` for the next processing step.
    skipped = 0

    # Run this block with managed resources/context cleanup.
    with open(csv_path, "r", newline="") as f:
        # Compute `reader` for the next processing step.
        reader = csv.DictReader(f)
        # Iterate `row` across `reader` to process each element.
        for row in reader:
            # Compute `video_path` for the next processing step.
            video_path = row["video_path"].strip()
            # Update `total` in place using the latest contribution.
            total += 1
            # Compute `res` for the next processing step.
            res = process_video_to_disk(
                video_path,
                processed_root,
                processor=processor,
                overwrite=overwrite,
            )
            # Branch behavior based on the current runtime condition.
            if res.get("ok") and res.get("skipped"):
                # Update `skipped` in place using the latest contribution.
                skipped += 1
            # Branch behavior based on the current runtime condition.
            elif res.get("ok"):
                # Update `ok` in place using the latest contribution.
                ok += 1

    # Invoke `print` to advance this processing stage.
    print(f"[Precompute] Done. total={total} ok={ok} skipped={skipped}")
    # Return the result expected by the caller.
    return {"total": total, "ok": ok, "skipped": skipped}


# Branch behavior based on the current runtime condition.
if __name__ == "__main__":
    # Invoke `main` to advance this processing stage.
    main()

