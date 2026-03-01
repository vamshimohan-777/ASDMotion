# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

import hashlib
import os
import re


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def make_video_id(video_path: str) -> str:
    """
    Stable, filesystem-safe video id derived from path + short hash.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    base = _SAFE_RE.sub("_", base).strip("_") or "video"
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"

