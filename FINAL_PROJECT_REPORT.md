# ASDMotion Final Project Report
Version: 1.0  


## 1. Executive Summary
ASDMotion is an end-to-end ASD behavioral screening prototype that:
- Ingests standard videos or landmarks-only videos.
- Extracts face/pose/motion evidence from frames.
- Performs multimodal temporal reasoning with NAS-selected architecture.
- Produces calibrated risk outputs with quality gating.
- Returns clinically oriented labels and event evidence.
- Serves a 3-stage web UI (Upload -> Processing -> Report) and downloadable PDF reports.

Current decision outputs are:
- `NEGATIVE`
- `LOW CHANCES OF ASD`
- `MEDIUM CHANCES OF ASD`
- `HIGH CHANCES OF ASD`
- `NEEDS RECHECKING`
- `LOW QUALITY VIDEO/FALSE VIDEO UPLOAD`

## 2. Scope and Objectives
### 2.1 Core objectives
- Group-leakage-safe training and evaluation using subject-level split strategy.
- Multistream model combining:
  - face appearance stream
  - pose/skeleton stream
  - motion-difference stream
  - static image evidence stream
- Clinically useful output formatting with reasons, event evidence, follow-up text, and PDF export.
- Deployable on LAN and public internet (Docker or no-Docker fallback).

### 2.2 Intended usage
- Decision-support screening prototype.
- Not a standalone diagnostic system.

## 3. High-Level System Architecture
## 3.1 Layered architecture
- Data layer:
  - `data/videos.csv` with `video_path,label,subject_id`
  - optional preprocessed artifacts under `data/processed`
- Preprocessing layer:
  - MediaPipe-based landmark extraction and quality scoring
  - face crops + skeleton renderings + quality JSON
- Modeling layer:
  - CNN encoders -> stream gating -> microkinetic encoder -> temporal transformer
  - static image path + evidence fusion
  - post-hoc temperature calibration + rule-based decision mapping
- Service layer:
  - FastAPI endpoints for raw and landmarks-only inference
  - static frontend hosted from backend
- Presentation/reporting layer:
  - three-screen UI workflow
  - professional screening PDF generation (client-side)
  - professional training report PDF generation (Matplotlib)
- Deployment layer:
  - local run, Docker compose, Caddy reverse proxy, cloudflared/localtunnel tunnels

## 3.2 Runtime blueprint (request flow)
### Standard video mode
1. User uploads file in frontend (`/`).
2. Frontend sends multipart POST to `POST /predict_file`.
3. Backend stores temporary file, calls predictor.
4. Predictor preprocesses frames via `VideoProcessor.process_video_file`.
5. Model inference returns logits/probabilities/events.
6. Decision engine maps outputs to clinical label.
7. Frontend renders report + allows PDF download.

### Landmarks-only mode
1. User selects "Landmarks-Only Video".
2. Frontend sends multipart POST to `POST /predict_processed_file`.
3. Predictor reads uploaded video frames directly as skeleton-like input.
4. Model runs with face disabled in quality map (pose-dominant path).
5. Decision + evidence returned and rendered.

## 4. Model Architecture Details
## 4.1 Input tensor structure
Primary inputs to model (`ASDPipeline`):
- `face_crops`: `[B, T, 3, 224, 224]`
- `pose_maps`: `[B, T, 3, 224, 224]`
- `motion_maps`: `[B, T, 3, 224, 224]` (abs diff between consecutive pose frames)
- `mask`: `[B, T]`
- `timestamps`: `[B, T]`
- `delta_t`: `[B, T]`
- `qualities`: face/pose/hand quality scores
- `route_mask`: video-vs-image routing scalar

## 4.2 CNN backbones
Current configurable backbone (`model.cnn_backbone`):
- `resnet18` or `resnet50`

Current config default:
- `resnet50` (`config.yaml`)

Used in:
- Face encoder
- Pose encoder
- Motion encoder
- Perception (static image) encoder

All streams project to 256-dim features after backbone.

## 4.3 Multistream gating
Face/pose/hand quality signals are converted into softmax gating weights, then applied to stream features before concatenation:
- Output token dim into NAS controller: 768 (`3 x 256`)

## 4.4 Microkinetic + NAS controller
NAS search space includes:
- encoder kernel: `[3, 5, 7, 11]`
- transformer heads: `[2, 4, 8]`
- transformer layers: `[2, 3, 4]`
- transformer FF dim: `[512, 1024, 2048]`

Mechanism:
- Gumbel-softmax during search/training for architecture weights.
- Discretization to final architecture after search.
- Best architecture saved to checkpoint and JSON.

## 4.5 Temporal reasoning
Temporal transformer uses:
- sinusoidal time-position embedding
- learned time-gap (`delta_t`) embedding
- event token features + scalar attributes + type embeddings + token confidence

Outputs include:
- video-path logit/probability
- confidence score
- event type IDs and event confidences for explainability.

## 4.6 Static image path and fusion
Image path:
- midpoint frame -> perception CNN -> static evidence encoder -> image logit

Fusion (`EvidenceFusion`):
- combines video and image probabilities with learnable alpha
- converts fused probability back to stable logit for BCE training
- supports route masking
- computes confidence and threshold-based tri-state decision in fusion module

## 4.7 Decision engine and clinical labels
Final backend decision labels come from `src/utils/decision.py` using:
- quality threshold (`quality_threshold`)
- low/high decision thresholds (`decision_low`, `decision_high`)
- calibrated probability

Rules:
- Quality below threshold -> `LOW QUALITY VIDEO/FALSE VIDEO UPLOAD`
- Above high threshold -> staged positive:
  - `LOW CHANCES OF ASD`
  - `MEDIUM CHANCES OF ASD`
  - `HIGH CHANCES OF ASD`
- Below low threshold -> `NEGATIVE`
- Between thresholds -> `NEEDS RECHECKING`

## 4.8 Event evidence extraction
Event evidence is included for:
- Positive staged labels
- `NEEDS RECHECKING`

Backend aggregates event tokens into top events:
- count
- mean confidence
- max confidence

## 5. Data and Training Pipeline
## 5.1 Dataset requirements
Mandatory CSV columns:
- `video_path`
- `label`
- `subject_id`

`subject_id` is required to enforce group-aware splitting and prevent leakage.

## 5.2 Preprocessing pipeline
Per frame:
- read frame with OpenCV
- extract face + pose landmarks (MediaPipe)
- render skeleton map
- aligned face crop
- compute quality mask scores

Outputs can be precomputed on disk:
- `data/processed/<video_id>/faces/*.png`
- `data/processed/<video_id>/skeletons/*.png`
- `data/processed/<video_id>/quality.json`
- `data/processed/<video_id>/meta.json`

## 5.3 Train-time data handling
- sequence sampling length: configurable (`seq_len`, default 32)
- augmentations:
  - spatial (flip, rotation, affine)
  - photometric jitter
  - random erasing
  - gaussian noise
  - temporal masking/jitter/speed perturbation
  - quality noise
- optional cache and precompute support

## 5.4 Split/evaluation strategy
- NAS validation uses group-stratified split.
- Main development uses group K-fold splits.
- Final model stage trains on full dataset with a holdout subset used only for monitoring/report/calibration.

## 5.5 Calibration and reporting
- Temperature scaling applied to logits.
- Metrics:
  - AUC, F1, accuracy, sensitivity@specificity, ECE, abstain rate
- Outputs:
  - `results/asd_best_fold{i}.pth`
  - `results/training_report_fold{i}.pdf`
  - `results/asd_pipeline_model.pth`
  - `results/training_report_final.pdf`

## 6. Inference and API Blueprint
## 6.1 Backend service
FastAPI app in `backend/app.py`
- startup loads predictor/checkpoint
- serves frontend static files from `frontend/`

### Endpoints
- `GET /health`
  - response: `{"status":"ok"}`
- `POST /predict_file`
  - input: multipart video file
  - mode: standard/raw processing path
- `POST /predict_processed_file`
  - input: multipart video file
  - mode: landmarks-only video input
- `POST /predict_processed`
  - input JSON:
    - `processed_ref`
    - optional `processed_root`
  - mode: inference from preprocessed disk artifacts

### Response fields (core)
- `decision`
- `prob_raw`
- `prob_calibrated`
- `quality_score`
- `threshold_used`
- `abstained`
- `reasons[]`
- `events[]`
- `model_version`
- `inference_ms`

## 7. Frontend and Clinical Reporting
## 7.1 Web UI flow
Three-page workflow:
1. Upload / case intake
2. Processing
3. Clinical report display

Upload options:
- Standard Video
- Landmarks-Only Video

## 7.2 Report contents (UI)
- decision badge
- interpretation text
- calibrated/raw probabilities
- quality score + abstention status
- rationale list
- behavioral event evidence list
- follow-up recommendation
- clinical-use disclaimer

## 7.3 Downloaded PDF report
Frontend generates professional multi-section PDF with:
- branded header + decision badge
- case metadata
- quantitative measures
- rationale and event evidence table
- follow-up section
- clinical-use notice
- signature/date lines and page footer

## 8. Video Requirements and Operational Input Standards
## 8.1 Supported upload type
- Frontend accepts `video/*`
- Backend relies on OpenCV decode support (codec/container availability matters)

Recommended practical formats:
- `.mp4` (H.264)
- `.mov` (supported codec)

## 8.2 Content expectations
- Subject should be visible enough for landmarks.
- Avoid severe blur, darkness, heavy occlusion.
- Stable framing improves quality score and event reliability.

## 8.3 Frame processing assumptions
- Frames are resized to 224x224 for CNN paths.
- Sequence length fixed to config (`seq_len`).
- Motion maps derived from pose frame differences.

## 8.4 Route behavior
- Routing function currently uses file extension rule:
  - `.gif` -> image route
  - other video files -> video route

## 8.5 Landmarks-only upload mode
- Use processed/augmented videos containing landmarks/skeleton render only.
- Same video container handling as regular upload.
- Face signal is intentionally absent/low; pose signal dominates quality logic.

## 8.6 Standardized recording action script (recommended)
Target duration: 90 seconds

1. 0-10s: Baseline still
- Subject faces camera with neutral expression.
- Keep both hands visible.

2. 10-25s: Gaze tracking
- Prompt: "Look center, left, center, right, center, up, center, down."
- Hold each direction for about 2 seconds.

3. 25-40s: Facial expression set
- Prompt: "Neutral, smile, neutral, raise eyebrows, neutral, puff cheeks, neutral."
- Hold each expression for about 2 seconds.

4. 40-60s: Head movement
- Prompt: "Turn head left, center, right, center, nod yes 2 times, shake no 2 times."

5. 60-80s: Hand and gesture movement
- Prompt: "Raise both hands to shoulder level, wave right hand, wave left hand, clap 3 times, touch nose, hands down."

6. 80-90s: Closing still
- Subject returns to neutral and looks at camera with minimal movement.

Capture rules:
- Keep face and upper body visible throughout.
- Use stable framing (no camera panning or zooming).
- Avoid severe blur, darkness, and heavy occlusion.
- Prefer `.mp4` (H.264) or supported `.mov`.

## 9. Deployment Blueprint
## 9.1 Local development
- `run_backend.bat` starts uvicorn on `0.0.0.0:8000`
- frontend served from backend at `http://localhost:8000/`

## 9.2 LAN deployment
- Open `http://<SERVER_IP>:8000/` from devices on same network
- Allow inbound firewall rule for port 8000

## 9.3 Docker deployment
Core services:
- `app` (FastAPI + model)
- `caddy` (reverse proxy + HTTPS)
- optional tunnel profiles:
  - `cloudflared`
  - `localtunnel`

Launch modes via script:
- `deploy_internet.bat cloudflared`
- `deploy_internet.bat localtunnel`
- `deploy_internet.bat direct`

## 9.4 No-Docker internet fallback (Windows)
For environments that cannot run Linux containers:
- `deploy_internet_nodocker.bat cloudflared`
- `deploy_internet_nodocker.bat localtunnel`

This starts host backend + tunnel process.

## 9.5 Reverse proxy security headers
Caddy configuration includes:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` restrictions
- request body size limit: `500MB`

## 10. Environment and Dependencies
Key runtime dependencies:
- PyTorch / Torchvision
- OpenCV
- MediaPipe
- FastAPI + Uvicorn
- scikit-learn, numpy, pandas, matplotlib

External model files required for full preprocessing:
- `assets/video/face_landmarker.task`
- `assets/video/pose_landmarker_full.task`

## 11. Security, Clinical, and Production Considerations
## 11.1 Current state
- API is open by default (no auth/rate limiting in backend).
- Tunnel modes expose service publicly.
- Reports include disclaimers but are still decision-support artifacts.

## 11.2 Recommended hardening before broad external use
- Add authentication and role-based access.
- Add request limits/rate limiting.
- Add TLS-only ingress in production.
- Add audit logging for report generation and predictions.
- Add PHI handling policy and retention controls.

## 12. Known Limitations
- Performance gain from `resnet50` is not guaranteed; higher compute/memory cost.
- Quality score heavily depends on successful landmark extraction.
- Route logic currently extension-driven for GIF/non-GIF.
- Decision labels are threshold-based; sensitivity to threshold tuning remains.
- Model is a prototype screening aid, not a certified clinical diagnostic tool.

## 13. Recommended Next Steps
1. Benchmark `resnet18` vs `resnet50` on the same held-out cohort and compare:
   - AUC, F1, Sens@Spec, calibration, inference latency.
2. Add authentication + rate limiting in backend/proxy for public deployment.
3. Add dataset quality QA script (codec/readability/landmark success rate).
4. Add structured experiment tracking for architecture, thresholds, and reports.
5. Define clinical validation protocol with external test cohort.

## 14. Project Artifacts Checklist
- Model checkpoints:
  - `results/asd_best_fold{i}.pth`
  - `results/asd_pipeline_model.pth`
- Training reports:
  - `results/training_report_fold{i}.pdf`
  - `results/training_report_final.pdf`
- NAS architecture export:
  - `results/nas_architecture.json` (when NAS enabled)
- Deployment files:
  - `docker-compose.yml`
  - `Dockerfile`
  - `deploy/Caddyfile`
  - `deploy_internet.bat`
  - `deploy_internet_nodocker.bat`
