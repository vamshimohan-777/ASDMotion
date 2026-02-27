"""
Merge normal + landmark CSVs for dual-input training.

Output columns:
  video_path,label,subject_id,is_landmark_video
"""

import argparse
import csv
import os
import re


def _label_to_binary(label_value):
    text = str(label_value).strip()
    if text == "":
        return 0
    try:
        num = float(text)
        return 1 if abs(num - 1.0) < 1e-6 else 0
    except Exception:
        return 1 if text.lower() in {"1", "true", "yes", "y", "asd"} else 0


def _subject_hint_from_path(video_path):
    text = str(video_path or "")
    m = re.search(r"subj[_-]?(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"subject_{m.group(1)}"
    return ""


def _subject_source_key(row):
    subject_id = str(row.get("subject_id", "")).strip()
    if subject_id:
        return subject_id
    hint = _subject_hint_from_path(row.get("video_path", ""))
    if hint:
        return hint
    return "__unknown__"


def _read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def merge_csvs(normal_csv, landmark_csv, output_csv, asd_start=1000, nonasd_start=2000):
    normal_total = 0
    normal_asd = 0
    normal_nonasd = 0
    landmark_total = 0
    landmark_asd = 0
    landmark_nonasd = 0
    landmark_subjects_asd = 0
    landmark_subjects_nonasd = 0
    written = 0
    landmark_subject_map = {}

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["video_path", "label", "subject_id", "is_landmark_video"],
        )
        writer.writeheader()

        for row in _read_rows(normal_csv):
            vpath = str(row.get("video_path", "")).strip()
            if not vpath:
                continue
            label_bin = _label_to_binary(row.get("label"))
            subject_id = str(row.get("subject_id", "")).strip()
            writer.writerow(
                {
                    "video_path": vpath,
                    "label": str(label_bin),
                    "subject_id": subject_id,
                    "is_landmark_video": "0",
                }
            )
            written += 1
            normal_total += 1
            if label_bin == 1:
                normal_asd += 1
            else:
                normal_nonasd += 1

        asd_id = int(asd_start)
        nonasd_id = int(nonasd_start)
        for row in _read_rows(landmark_csv):
            vpath = str(row.get("video_path", "")).strip()
            if not vpath:
                continue
            label_bin = _label_to_binary(row.get("label"))
            src_subject = _subject_source_key(row)
            map_key = (label_bin, src_subject)
            assigned = landmark_subject_map.get(map_key)
            if assigned is None:
                if label_bin == 1:
                    assigned = f"subject_{asd_id}"
                    asd_id += 1
                    landmark_subjects_asd += 1
                else:
                    assigned = f"subject_{nonasd_id}"
                    nonasd_id += 1
                    landmark_subjects_nonasd += 1
                landmark_subject_map[map_key] = assigned

            if label_bin == 1:
                landmark_asd += 1
            else:
                landmark_nonasd += 1

            writer.writerow(
                {
                    "video_path": vpath,
                    "label": str(label_bin),
                    "subject_id": assigned,
                    "is_landmark_video": "1",
                }
            )
            written += 1
            landmark_total += 1

    return {
        "written": written,
        "normal_total": normal_total,
        "normal_asd": normal_asd,
        "normal_nonasd": normal_nonasd,
        "landmark_total": landmark_total,
        "landmark_asd": landmark_asd,
        "landmark_nonasd": landmark_nonasd,
        "landmark_subjects_asd": landmark_subjects_asd,
        "landmark_subjects_nonasd": landmark_subjects_nonasd,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge normal and landmark CSV files")
    parser.add_argument("--normal-csv", required=True)
    parser.add_argument("--landmark-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--asd-start", type=int, default=1000)
    parser.add_argument("--nonasd-start", type=int, default=2000)
    args = parser.parse_args()

    summary = merge_csvs(
        normal_csv=args.normal_csv,
        landmark_csv=args.landmark_csv,
        output_csv=args.output_csv,
        asd_start=args.asd_start,
        nonasd_start=args.nonasd_start,
    )

    print(
        f"[MergeCSV] normal={summary['normal_total']} "
        f"(asd={summary['normal_asd']} nonasd={summary['normal_nonasd']})"
    )
    print(
        f"[MergeCSV] landmark={summary['landmark_total']} "
        f"(asd={summary['landmark_asd']} nonasd={summary['landmark_nonasd']})"
    )
    print(
        f"[MergeCSV] landmark subjects: asd={summary['landmark_subjects_asd']} "
        f"nonasd={summary['landmark_subjects_nonasd']}"
    )
    if summary["landmark_total"] > 0 and (
        summary["landmark_asd"] == 0 or summary["landmark_nonasd"] == 0
    ):
        print(
            "[MergeCSV] WARNING: landmark CSV has only one class. "
            "Expected both ASD and non-ASD for balanced training."
        )
    print(f"[MergeCSV] wrote={summary['written']} -> {args.output_csv}")


if __name__ == "__main__":
    main()
