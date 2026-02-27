import hashlib
import os
import re


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _clean_token(text: str) -> str:
    token = _SAFE_RE.sub("_", str(text)).strip("_")
    return token or "x"


def _canonical_label(label) -> str:
    text = str(label).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    return text


def make_video_id(video_path: str, subject_id: str = None, label: str = None) -> str:
    """
    Stable, filesystem-safe video id derived from path + short hash.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    base = _SAFE_RE.sub("_", base).strip("_") or "video"
    key = str(video_path)
    subject_token = ""
    label_token = ""
    label_key = _canonical_label(label) if label is not None else ""
    if label_key:
        label_token = f"label_{_clean_token(label_key)}"
        key = f"{label_key}|{key}"
    if subject_id is not None and str(subject_id).strip() != "":
        subject_token = _clean_token(subject_id)
        key = f"{subject_id}|{video_path}"
        if label_token:
            key = f"{label_key}|{subject_id}|{video_path}"

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    prefix_parts = []
    if label_token:
        prefix_parts.append(label_token)
    if subject_token:
        prefix_parts.append(subject_token)
    if prefix_parts:
        return f"{'__'.join(prefix_parts)}__{base}_{digest}"
    return f"{base}_{digest}"
