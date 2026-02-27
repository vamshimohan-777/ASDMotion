# ASDMotion Landmark Recorder (Self-Contained)

This folder is self-contained. All runtime dependencies are local under `vendor/`.

## Included
- `index.html`
- `app.js`
- `style.css`
- `requirements.txt` (recorder backend only)
- `install_recorder_deps.bat`
- `run_recorder_backend.bat`
- `vendor/mediapipe/...` (camera utils + holistic + assets)
- `vendor/ffmpeg/...` (ffmpeg wasm runtime for MP4/AVI conversion)

## Run
1. Install recorder backend dependencies (local to this folder):
   - `frontend\landmark_recorder\install_recorder_deps.bat`
2. Start recorder backend:
   - `frontend\landmark_recorder\run_recorder_backend.bat`
3. Open recorder:
   - `http://localhost:8010/`

## Notes
- Browser `file://` mode blocks some local wasm/model fetches by CORS policy.
  The app handles this by automatically using CDN URLs for those specific runtime assets only when opened as `file://`.
- On `http/https`, it uses local `vendor` files.
- Main control buttons: `Enable Camera`, `Start/Stop Recording`
- Green participant box (ROI): keep participant fully inside this box; adjust `x/y/width/height` if needed. Landmarks outside box are ignored.
- Data Collection Guide option is built in (actions, camera position, and lighting recommendations).
- Recording target FPS: 60 (or highest supported by camera/browser)
- `Enable Camera` acts as a toggle and can disable camera.
- Recorder saves through backend endpoint `/recordings/save` from `recorder_backend.py`.
- Files are written to this folder: `frontend/landmark_recorder/data/`
- If backend is not reachable, it falls back to normal browser download.
- Landmarks drawn per frame:
  - pose
  - face (468 points)
  - left hand (21 points)
  - right hand (21 points)
- Output: prefers MP4, then AVI fallback, then WEBM fallback.
