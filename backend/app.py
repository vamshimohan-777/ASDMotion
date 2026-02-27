import os
import time
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.inference.predictor import ASDPredictor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asd_backend")

app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTOR = None


class ProcessedPredictRequest(BaseModel):
    processed_ref: str
    processed_root: str | None = None


@app.on_event("startup")
def _load_model():
    global PREDICTOR
    checkpoint = os.environ.get("ASD_CHECKPOINT", "results/asd_pipeline_model.pth")
    config_path = os.environ.get("ASD_CONFIG", None)
    device = os.environ.get("ASD_DEVICE", None)
    try:
        PREDICTOR = ASDPredictor(checkpoint, config_path=config_path, device=device)
        logger.info("Model loaded")
    except Exception as exc:
        PREDICTOR = None
        logger.exception("Model failed to load: %s", exc)


@app.get("/health")
def health():
    return {"status": "ok" if PREDICTOR is not None else "degraded", "model_loaded": PREDICTOR is not None}


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if PREDICTOR is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    start = time.time()
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = PREDICTOR.predict_video(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    elapsed_ms = int((time.time() - start) * 1000)
    result["inference_ms"] = elapsed_ms

    logger.info("Predict: %s", result)
    return result


@app.post("/predict_processed")
def predict_processed(payload: ProcessedPredictRequest):
    if PREDICTOR is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    start = time.time()
    try:
        result = PREDICTOR.predict_preprocessed(
            processed_ref=payload.processed_ref,
            processed_root=payload.processed_root,
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    elapsed_ms = int((time.time() - start) * 1000)
    result["inference_ms"] = elapsed_ms

    logger.info("Predict (processed): %s", result)
    return result


@app.post("/predict_processed_file")
async def predict_processed_file(
    file: UploadFile = File(...),
):
    if PREDICTOR is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    start = time.time()
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = PREDICTOR.predict_landmark_video(tmp_path)
    except FileNotFoundError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    elapsed_ms = int((time.time() - start) * 1000)
    result["inference_ms"] = elapsed_ms

    logger.info("Predict (processed file): %s", result)
    return result


# Serve frontend and static assets from the same origin as the API.
# This makes cross-network deployment simpler: one host:port for UI + model.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
