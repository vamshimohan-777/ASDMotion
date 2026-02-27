"""
Convert NTU `.skeleton` files into preprocessed motion tensors.

Output tensors are float32 with shape [T, J, 9]:
  [x, y, z, vx, vy, vz, ax, ay, az]
"""

import argparse
import csv
import os

import numpy as np


def parse_ntu_skeleton_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if not lines:
        raise ValueError(f"Empty .skeleton file: {path}")

    p = 0
    try:
        n_frames = int(float(lines[p]))
    except Exception as exc:
        raise ValueError(f"Invalid .skeleton header: {path}") from exc
    p += 1

    frames = []
    max_joints = 0
    for _ in range(max(0, n_frames)):
        if p >= len(lines):
            break
        try:
            n_bodies = int(float(lines[p]))
        except Exception:
            break
        p += 1

        best = None
        best_valid = -1
        for _ in range(max(0, n_bodies)):
            if p >= len(lines):
                break
            p += 1  # body-info line
            if p >= len(lines):
                break
            try:
                n_joints = int(float(lines[p]))
            except Exception:
                break
            p += 1

            joints = []
            for _ in range(max(0, n_joints)):
                if p >= len(lines):
                    break
                vals = lines[p].split()
                p += 1
                if len(vals) < 3:
                    joints.append([0.0, 0.0, 0.0])
                    continue
                try:
                    x = float(vals[0])
                    y = float(vals[1])
                    z = float(vals[2])
                except Exception:
                    x, y, z = 0.0, 0.0, 0.0
                joints.append([x, y, z])

            if not joints:
                continue
            arr = np.asarray(joints, dtype=np.float32)
            valid = int((np.abs(arr).sum(axis=-1) > 1e-8).sum())
            if valid > best_valid:
                best = arr
                best_valid = valid

        if best is None:
            best = np.zeros((25, 3), dtype=np.float32)
        max_joints = max(max_joints, int(best.shape[0]))
        frames.append(best)

    if not frames:
        raise ValueError(f"No frames parsed from .skeleton file: {path}")

    j = max(1, max_joints)
    out = []
    for arr in frames:
        if arr.shape[0] < j:
            pad = j - arr.shape[0]
            arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
        elif arr.shape[0] > j:
            arr = arr[:j]
        out.append(arr)
    return np.stack(out, axis=0).astype(np.float32)  # [T,J,3]


def to_fixed_joint_count(xyz, expected_joints):
    t, j, c = xyz.shape
    if c != 3:
        raise ValueError(f"Expected xyz channels=3, got {c}")
    if j < expected_joints:
        xyz = np.pad(xyz, ((0, 0), (0, expected_joints - j), (0, 0)), mode="constant")
    elif j > expected_joints:
        xyz = xyz[:, :expected_joints, :]
    return xyz


def root_center_and_scale_normalize(xyz):
    """
    NTU normalization:
    1) Root-center using joint 0 (spine base).
    2) Scale normalize using shoulder distance (j4-j8) with robust fallback.
    """
    xyz = xyz.astype(np.float32)
    t, j, _ = xyz.shape
    valid = (np.abs(xyz).sum(axis=-1) > 1e-8).astype(np.float32)  # [T, J]

    root_idx = 0
    root_track = xyz[:, root_idx, :].copy()  # [T,3]
    root_valid = valid[:, root_idx] > 0.5

    if root_valid.any():
        first = int(np.where(root_valid)[0][0])
        last = int(np.where(root_valid)[0][-1])
        root_track[:first] = root_track[first]
        root_track[last + 1 :] = root_track[last]
        miss = np.where(~root_valid)[0]
        for m in miss:
            left = np.where(root_valid[:m])[0]
            right = np.where(root_valid[m + 1 :])[0]
            if left.size > 0 and right.size > 0:
                l = int(left[-1])
                r = int(m + 1 + right[0])
                alpha = float(m - l) / float(max(r - l, 1))
                root_track[m] = (1.0 - alpha) * root_track[l] + alpha * root_track[r]
            elif left.size > 0:
                root_track[m] = root_track[int(left[-1])]
            elif right.size > 0:
                root_track[m] = root_track[int(m + 1 + right[0])]
    else:
        root_track[:] = 0.0

    centered = xyz - root_track[:, None, :]

    shoulder_l = 4
    shoulder_r = 8
    spine_shoulder = 20
    scales = np.zeros((t,), dtype=np.float32)

    if j > max(shoulder_l, shoulder_r):
        both_sh = (valid[:, shoulder_l] > 0.5) & (valid[:, shoulder_r] > 0.5)
        if both_sh.any():
            scales[both_sh] = np.linalg.norm(
                centered[both_sh, shoulder_l, :] - centered[both_sh, shoulder_r, :],
                axis=-1,
            )

    if j > spine_shoulder:
        fallback = (scales <= 1e-8) & (valid[:, root_idx] > 0.5) & (valid[:, spine_shoulder] > 0.5)
        if fallback.any():
            scales[fallback] = np.linalg.norm(
                centered[fallback, spine_shoulder, :] - centered[fallback, root_idx, :],
                axis=-1,
            )

    good = scales > 1e-8
    if good.any():
        seq_scale = float(np.median(scales[good]))
    else:
        seq_scale = 1.0
    seq_scale = max(seq_scale, 1e-4)

    centered = centered / seq_scale
    centered *= valid[..., None]
    return centered.astype(np.float32)


def build_motion_features(xyz):
    xyz = np.nan_to_num(xyz.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vel = np.zeros_like(xyz, dtype=np.float32)
    acc = np.zeros_like(xyz, dtype=np.float32)
    if xyz.shape[0] > 1:
        vel[1:] = xyz[1:] - xyz[:-1]
        acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([xyz, vel, acc], axis=-1).astype(np.float32)


def find_skeleton_files(input_dir):
    files = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() == ".skeleton":
                files.append(os.path.join(root, name))
    files.sort()
    return files


def convert_one_file(in_path, out_path, expected_joints):
    xyz = parse_ntu_skeleton_file(in_path)
    xyz = to_fixed_joint_count(xyz, expected_joints=expected_joints)
    xyz = root_center_and_scale_normalize(xyz)
    motion = build_motion_features(xyz)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, motion)
    return int(motion.shape[0]), int(motion.shape[1])


def run_conversion(input_dir, output_dir, manifest_path, expected_joints=25, overwrite=False):
    files = find_skeleton_files(input_dir)
    if not files:
        raise ValueError(f"No .skeleton files found in: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    converted = 0
    skipped = 0
    failed = 0

    for i, in_path in enumerate(files, start=1):
        rel = os.path.relpath(in_path, input_dir)
        rel_npy = os.path.splitext(rel)[0] + ".npy"
        out_path = os.path.join(output_dir, rel_npy)

        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            rows.append(
                {
                    "motion_path": os.path.abspath(out_path),
                    "source_path": os.path.abspath(in_path),
                    "num_frames": "",
                    "num_joints": "",
                    "status": "skipped_exists",
                }
            )
            continue

        try:
            n_frames, n_joints = convert_one_file(in_path, out_path, expected_joints=expected_joints)
            converted += 1
            rows.append(
                {
                    "motion_path": os.path.abspath(out_path),
                    "source_path": os.path.abspath(in_path),
                    "num_frames": n_frames,
                    "num_joints": n_joints,
                    "status": "ok",
                }
            )
        except Exception as exc:
            failed += 1
            rows.append(
                {
                    "motion_path": "",
                    "source_path": os.path.abspath(in_path),
                    "num_frames": "",
                    "num_joints": "",
                    "status": f"error: {exc}",
                }
            )

        if (i % 50) == 0 or i == len(files):
            print(
                f"[Convert] {i}/{len(files)} processed "
                f"(converted={converted}, skipped={skipped}, failed={failed})"
            )

    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["motion_path", "source_path", "num_frames", "num_joints", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Convert] input_dir={input_dir}")
    print(f"[Convert] output_dir={output_dir}")
    print(f"[Convert] manifest={manifest_path}")
    print(f"[Convert] converted={converted} skipped={skipped} failed={failed}")


def main():
    parser = argparse.ArgumentParser(description="Convert NTU .skeleton files to [T,J,9] motion tensors")
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--manifest", default=None, type=str)
    parser.add_argument("--expected-joints", default=25, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")

    manifest_path = args.manifest
    if not manifest_path:
        manifest_path = os.path.join(output_dir, "motion_tensors_manifest.csv")
    manifest_path = os.path.abspath(manifest_path)

    run_conversion(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        expected_joints=int(args.expected_joints),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
