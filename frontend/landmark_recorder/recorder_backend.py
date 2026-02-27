import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

app = FastAPI(title="ASDMotion Landmark Recorder Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_output_filename(name: str | None, fallback_ext: str = ".webm") -> str:
    text = (name or "").strip()
    if not text:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return f"landmark_capture_{stamp}{fallback_ext}"

    base = os.path.basename(text)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if not base:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return f"landmark_capture_{stamp}{fallback_ext}"

    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    allowed_ext = {".mp4", ".avi", ".webm"}
    if ext not in allowed_ext:
        ext = fallback_ext
    if not stem:
        stem = "landmark_capture"
    return f"{stem}{ext}"


@app.get("/health")
def health():
    return {"status": "ok", "service": "landmark_recorder_backend"}


@app.post("/recordings/save")
async def save_recording(file: UploadFile = File(...)):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to prepare data folder: {e}"})

    suffix = os.path.splitext(file.filename or "")[-1].lower() or ".webm"
    safe_name = _sanitize_output_filename(file.filename, fallback_ext=suffix)
    out_path = DATA_DIR / safe_name

    # Avoid overwrite by adding numeric suffix.
    if out_path.exists():
        stem = out_path.stem
        ext = out_path.suffix
        idx = 1
        while True:
            candidate = DATA_DIR / f"{stem}_{idx}{ext}"
            if not candidate.exists():
                out_path = candidate
                break
            idx += 1

    try:
        data = await file.read()
        with open(out_path, "wb") as f:
            f.write(data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to save recording: {e}"})

    rel_path = out_path.relative_to(APP_DIR).as_posix()
    return {
        "ok": True,
        "file_name": out_path.name,
        "saved_path": str(out_path),
        "saved_path_rel": rel_path,
    }


if APP_DIR.exists():
    app.mount("/", StaticFiles(directory=str(APP_DIR), html=True), name="recorder_frontend")
