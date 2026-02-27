const permissionBtn = document.getElementById("permissionBtn");
const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const roiEnabledEl = document.getElementById("roiEnabled");
const roiXEl = document.getElementById("roiX");
const roiYEl = document.getElementById("roiY");
const roiWEl = document.getElementById("roiW");
const roiHEl = document.getElementById("roiH");
const roiTextEl = document.getElementById("roiText");
const roiOverlayEl = document.getElementById("roiOverlay");

const cameraVideo = document.getElementById("cameraVideo");
const landmarkCanvas = document.getElementById("landmarkCanvas");
const canvasCtx = landmarkCanvas.getContext("2d", { alpha: false, desynchronized: true });
const TARGET_RECORD_FPS = 60;
const MAX_CAMERA_FPS = 120;
const TARGET_CAMERA_WIDTH = 640;
const TARGET_CAMERA_HEIGHT = 480;

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [15, 17], [16, 18],
  [17, 19], [18, 20],
  [19, 21], [20, 22],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [24, 26],
  [25, 27], [26, 28],
  [27, 29], [28, 30],
  [29, 31], [30, 32],
];

let holistic = null;
let cameraStream = null;
let renderLoopId = null;
let inferenceBusy = false;
let cameraStarted = false;
let mediaRecorder = null;
let recorderMimeType = "";
let recordedChunks = [];
let ffmpeg = null;
let ffmpegLoaded = false;
let achievedCameraFps = null;
const DEFAULT_BACKEND_BASE = "http://127.0.0.1:8010";
const roiState = {
  enabled: true,
  x: 0.15,
  y: 0.10,
  w: 0.70,
  h: 0.80,
};

function setStatus(text) {
  statusEl.textContent = text;
}

function formatError(err) {
  if (!err) {
    return "Unknown error.";
  }
  const name = typeof err.name === "string" ? err.name : "";
  const msg = typeof err.message === "string" ? err.message : "";
  const combined = `${name}${name && msg ? ": " : ""}${msg}`.trim();
  if (combined) {
    return combined;
  }
  if (typeof err.toString === "function") {
    const text = String(err.toString()).trim();
    if (text && text !== "[object Object]") {
      return text;
    }
  }
  return "Unknown error.";
}

function cameraGuidanceFromError(err) {
  const name = (err && err.name) ? String(err.name) : "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera blocked. Allow camera permission for this page and retry.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera device found.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Camera is busy in another app/browser tab.";
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return "Requested FPS/resolution unsupported by camera.";
  }
  if (name === "SecurityError") {
    return "Browser blocked camera for this context. Use Chrome/Edge and allow permissions.";
  }
  if (name === "TypeError") {
    return "Camera API unavailable in this browser/context.";
  }
  return formatError(err);
}

function drawBlackFrame() {
  canvasCtx.fillStyle = "#000000";
  canvasCtx.fillRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function updateRoiText() {
  if (!roiTextEl) {
    return;
  }
  if (!roiState.enabled) {
    roiTextEl.textContent = "Green box disabled (full frame used).";
    return;
  }
  const x = Math.round(roiState.x * 100);
  const y = Math.round(roiState.y * 100);
  const w = Math.round(roiState.w * 100);
  const h = Math.round(roiState.h * 100);
  roiTextEl.textContent = `Keep participant inside green box: x=${x}%, y=${y}%, w=${w}%, h=${h}%`;
}

function updateRoiOverlay() {
  if (!roiOverlayEl) {
    return;
  }
  if (!roiState.enabled) {
    roiOverlayEl.classList.add("hidden");
    return;
  }
  roiOverlayEl.classList.remove("hidden");
  roiOverlayEl.style.left = `${roiState.x * 100}%`;
  roiOverlayEl.style.top = `${roiState.y * 100}%`;
  roiOverlayEl.style.width = `${roiState.w * 100}%`;
  roiOverlayEl.style.height = `${roiState.h * 100}%`;
}

function normalizeRoiState() {
  roiState.x = clamp01(roiState.x);
  roiState.y = clamp01(roiState.y);
  roiState.w = clamp01(roiState.w);
  roiState.h = clamp01(roiState.h);

  const minSize = 0.10;
  if (roiState.w < minSize) {
    roiState.w = minSize;
  }
  if (roiState.h < minSize) {
    roiState.h = minSize;
  }
  if (roiState.x + roiState.w > 1) {
    roiState.w = Math.max(minSize, 1 - roiState.x);
  }
  if (roiState.y + roiState.h > 1) {
    roiState.h = Math.max(minSize, 1 - roiState.y);
  }
}

function syncRoiInputsFromState() {
  if (roiEnabledEl) {
    roiEnabledEl.checked = roiState.enabled;
  }
  if (roiXEl) {
    roiXEl.value = String(Math.round(roiState.x * 100));
  }
  if (roiYEl) {
    roiYEl.value = String(Math.round(roiState.y * 100));
  }
  if (roiWEl) {
    roiWEl.value = String(Math.round(roiState.w * 100));
  }
  if (roiHEl) {
    roiHEl.value = String(Math.round(roiState.h * 100));
  }
  updateRoiText();
  updateRoiOverlay();
}

function syncRoiStateFromInputs() {
  roiState.enabled = Boolean(roiEnabledEl && roiEnabledEl.checked);
  roiState.x = Number(roiXEl ? roiXEl.value : 15) / 100;
  roiState.y = Number(roiYEl ? roiYEl.value : 10) / 100;
  roiState.w = Number(roiWEl ? roiWEl.value : 70) / 100;
  roiState.h = Number(roiHEl ? roiHEl.value : 80) / 100;
  normalizeRoiState();
  syncRoiInputsFromState();
}

function mapToCaptureRegion(lm) {
  if (!lm) {
    return null;
  }
  let x = Number(lm.x);
  let y = Number(lm.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  if (roiState.enabled) {
    const xMin = roiState.x;
    const yMin = roiState.y;
    const xMax = roiState.x + roiState.w;
    const yMax = roiState.y + roiState.h;
    if (x < xMin || x > xMax || y < yMin || y > yMax) {
      return null;
    }
    x = (x - xMin) / roiState.w;
    y = (y - yMin) / roiState.h;
  }

  x = clamp01(x);
  y = clamp01(y);
  return { x, y };
}

function toCanvasPoints(landmarks, maxCount = 0) {
  if (!Array.isArray(landmarks)) {
    return [];
  }
  const w = landmarkCanvas.width;
  const h = landmarkCanvas.height;
  const limit = maxCount > 0 ? Math.min(maxCount, landmarks.length) : landmarks.length;
  const out = [];
  for (let i = 0; i < limit; i += 1) {
    const mapped = mapToCaptureRegion(landmarks[i]);
    if (!mapped) {
      out.push(null);
      continue;
    }
    out.push({
      x: mapped.x * w,
      y: mapped.y * h,
    });
  }
  return out;
}

function drawConnections(points, connections, color = "#ffffff", width = 2) {
  canvasCtx.strokeStyle = color;
  canvasCtx.lineWidth = width;
  canvasCtx.lineCap = "round";
  for (const [a, b] of connections) {
    const p1 = points[a];
    const p2 = points[b];
    if (!p1 || !p2) {
      continue;
    }
    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x, p1.y);
    canvasCtx.lineTo(p2.x, p2.y);
    canvasCtx.stroke();
  }
}

function drawPoints(points, radius = 2, color = "#ffffff") {
  canvasCtx.fillStyle = color;
  for (const p of points) {
    if (!p) {
      continue;
    }
    canvasCtx.beginPath();
    canvasCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    canvasCtx.fill();
  }
}

function drawLandmarkFrame(results) {
  drawBlackFrame();

  const posePoints = toCanvasPoints(results.poseLandmarks);
  const facePoints = toCanvasPoints(results.faceLandmarks, 468);
  const leftHandPoints = toCanvasPoints(results.leftHandLandmarks, 21);
  const rightHandPoints = toCanvasPoints(results.rightHandLandmarks, 21);

  drawConnections(posePoints, POSE_CONNECTIONS, "#ffffff", 2);
  drawPoints(posePoints, 2.5, "#ffffff");

  // Face: 468 points
  drawPoints(facePoints, 0.9, "#d9d9d9");
  // Hands: 21 points each
  drawPoints(leftHandPoints, 1.6, "#ffffff");
  drawPoints(rightHandPoints, 1.6, "#ffffff");
}

function pickMimeType() {
  const candidates = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

function isFileProtocol() {
  return window.location.protocol === "file:";
}

function getHolisticAssetBase() {
  if (isFileProtocol()) {
    return "https://cdn.jsdelivr.net/npm/@mediapipe/holistic";
  }
  return "./vendor/mediapipe/holistic";
}

function getFfmpegCoreUrls() {
  if (isFileProtocol()) {
    return {
      coreURL: "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/umd/ffmpeg-core.js",
      wasmURL: "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/umd/ffmpeg-core.wasm",
    };
  }
  return {
    coreURL: "./vendor/ffmpeg/core/dist/umd/ffmpeg-core.js",
    wasmURL: "./vendor/ffmpeg/core/dist/umd/ffmpeg-core.wasm",
  };
}

function buildBaseName() {
  return `landmark_capture_${new Date().toISOString().replace(/[:.]/g, "-")}`;
}

function getBackendBaseUrl() {
  const configured = typeof window.ASD_RECORDER_API_BASE === "string"
    ? window.ASD_RECORDER_API_BASE.trim()
    : "";
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return "";
  }
  return DEFAULT_BACKEND_BASE;
}

function getBackendSaveUrl() {
  const base = getBackendBaseUrl();
  return base ? `${base}/recordings/save` : "/recordings/save";
}

async function saveBlobViaBackend(blob, fileName, label) {
  const fileForUpload = new File([blob], fileName, {
    type: blob.type || "application/octet-stream",
  });
  const formData = new FormData();
  formData.append("file", fileForUpload);

  const response = await fetch(getBackendSaveUrl(), {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Backend save failed (${response.status})`);
  }

  const rel = payload.saved_path_rel || payload.saved_path || fileName;
  setStatus(`Saved ${label} to ${rel}`);
}

function downloadBlob(blob, fileName, label) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus(`Saved ${label}: ${fileName}`);
}

async function persistOutputBlob(blob, fileName, label) {
  try {
    await saveBlobViaBackend(blob, fileName, label);
    return;
  } catch (err) {
    console.warn("Backend save failed, falling back to download.", err);
  }
  downloadBlob(blob, fileName, label);
}

async function ensureFfmpegLoaded() {
  if (ffmpegLoaded) {
    return;
  }
  if (!window.FFmpegWASM) {
    throw new Error("FFmpeg libraries not available.");
  }

  const { FFmpeg } = window.FFmpegWASM;
  ffmpeg = new FFmpeg();
  const urls = getFfmpegCoreUrls();
  await ffmpeg.load(urls);
  ffmpegLoaded = true;
}

async function convertBlob(inputBlob, targetExt) {
  await ensureFfmpegLoaded();
  const inputName = "input.webm";
  const outputName = targetExt === "mp4" ? "output.mp4" : "output.avi";

  const inputBytes = new Uint8Array(await inputBlob.arrayBuffer());
  await ffmpeg.writeFile(inputName, inputBytes);
  if (targetExt === "mp4") {
    await ffmpeg.exec([
      "-i", inputName,
      "-r", String(TARGET_RECORD_FPS),
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      outputName,
    ]);
  } else {
    await ffmpeg.exec([
      "-i", inputName,
      "-r", String(TARGET_RECORD_FPS),
      outputName,
    ]);
  }

  const data = await ffmpeg.readFile(outputName);
  return new Blob(
    [data],
    { type: targetExt === "mp4" ? "video/mp4" : "video/x-msvideo" },
  );
}

async function ensureHolistic() {
  if (holistic) {
    return;
  }
  if (typeof Holistic === "undefined") {
    throw new Error("MediaPipe Holistic not loaded.");
  }

  holistic = new Holistic({
    locateFile: (file) => `${getHolisticAssetBase()}/${file}`,
  });
  holistic.setOptions({
    // Lower complexity + no smoothing reduces lag for high-FPS capture.
    modelComplexity: 0,
    smoothLandmarks: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  holistic.onResults((results) => {
    drawLandmarkFrame(results);
  });
}

function startInferenceLoop() {
  if (renderLoopId !== null) {
    cancelAnimationFrame(renderLoopId);
    renderLoopId = null;
  }

  const tick = async () => {
    if (!cameraStarted || !holistic) {
      return;
    }

    // Keep latency low by processing the newest frame only.
    if (!inferenceBusy) {
      inferenceBusy = true;
      try {
        await holistic.send({ image: cameraVideo });
      } catch (err) {
        console.error(err);
      } finally {
        inferenceBusy = false;
      }
    }

    renderLoopId = window.requestAnimationFrame(() => {
      tick();
    });
  };

  renderLoopId = window.requestAnimationFrame(() => {
    tick();
  });
}

async function startCamera() {
  if (cameraStarted) {
    await disableCamera();
    return;
  }
  await ensureHolistic();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Camera API unavailable. Use Chrome/Edge with camera permission.");
  }
  if (!isSecureContext && !isFileProtocol()) {
    throw new Error("Insecure context. Open via https/localhost or file://.");
  }

  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: TARGET_CAMERA_WIDTH },
      height: { ideal: TARGET_CAMERA_HEIGHT },
      frameRate: { ideal: TARGET_RECORD_FPS, max: MAX_CAMERA_FPS },
    },
  });
  cameraVideo.srcObject = cameraStream;
  await cameraVideo.play();

  // Try to force high FPS constraints after stream starts.
  achievedCameraFps = null;
  try {
    const track = cameraStream && typeof cameraStream.getVideoTracks === "function"
      ? cameraStream.getVideoTracks()[0]
      : null;
    if (track && typeof track.applyConstraints === "function") {
      await track.applyConstraints({
        frameRate: { ideal: TARGET_RECORD_FPS, max: MAX_CAMERA_FPS },
      });
    }
    if (track && typeof track.getSettings === "function") {
      const settings = track.getSettings();
      if (settings && Number.isFinite(settings.frameRate)) {
        achievedCameraFps = Number(settings.frameRate);
      }
    }
  } catch (err) {
    console.warn("Camera FPS constraint could not be fully applied:", err);
  }

  inferenceBusy = false;
  cameraStarted = true;
  permissionBtn.textContent = "Disable Camera";
  recordBtn.disabled = false;
  startInferenceLoop();
  if (achievedCameraFps && achievedCameraFps > 0) {
    setStatus(`Camera enabled (${achievedCameraFps.toFixed(1)} FPS).`);
  } else {
    setStatus(`Camera enabled (target ${TARGET_RECORD_FPS} FPS).`);
  }
}

async function disableCamera() {
  if (!cameraStarted) {
    return;
  }
  if (isRecording()) {
    stopRecording();
  }
  try {
    if (renderLoopId !== null) {
      cancelAnimationFrame(renderLoopId);
      renderLoopId = null;
    }
  } catch (err) {
    console.error(err);
  }
  inferenceBusy = false;
  stopCameraStreamOnExit();
  cameraVideo.srcObject = null;
  cameraStream = null;
  cameraStarted = false;
  permissionBtn.textContent = "Enable Camera";
  recordBtn.disabled = true;
  recordBtn.textContent = "Start Recording";
  drawBlackFrame();
  setStatus("Camera disabled.");
}

function isRecording() {
  return mediaRecorder && mediaRecorder.state === "recording";
}

async function startRecording() {
  if (!cameraStarted) {
    setStatus("Enable camera first.");
    return;
  }
  if (isRecording()) {
    return;
  }

  recorderMimeType = pickMimeType();
  if (!recorderMimeType) {
    throw new Error("No supported recorder format in this browser.");
  }

  const stream = landmarkCanvas.captureStream(TARGET_RECORD_FPS);
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, {
    mimeType: recorderMimeType,
    videoBitsPerSecond: 3_000_000,
  });

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const baseName = buildBaseName();
    const rawBlob = new Blob(recordedChunks, { type: recorderMimeType });

    if (recorderMimeType.includes("mp4")) {
      await persistOutputBlob(rawBlob, `${baseName}.mp4`, "MP4");
      return;
    }

    try {
      setStatus("Converting to MP4...");
      const mp4Blob = await convertBlob(rawBlob, "mp4");
      await persistOutputBlob(mp4Blob, `${baseName}.mp4`, "MP4");
      return;
    } catch (mp4Err) {
      console.error(mp4Err);
    }

    try {
      setStatus("MP4 conversion failed. Converting to AVI...");
      const aviBlob = await convertBlob(rawBlob, "avi");
      await persistOutputBlob(aviBlob, `${baseName}.avi`, "AVI");
      return;
    } catch (aviErr) {
      console.error(aviErr);
    }

    await persistOutputBlob(rawBlob, `${baseName}.webm`, "WEBM fallback");
  };

  mediaRecorder.start(250);
  recordBtn.textContent = "Stop Recording";
  setStatus(`Recording landmarks at ${TARGET_RECORD_FPS} FPS...`);
}

function stopRecording() {
  if (!isRecording()) {
    return;
  }
  mediaRecorder.stop();
  recordBtn.textContent = "Start Recording";
  setStatus("Stopping...");
}

function stopCameraStreamOnExit() {
  try {
    const stream = cameraVideo.srcObject;
    if (stream && typeof stream.getTracks === "function") {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }
  } catch (err) {
    console.error(err);
  }
}

permissionBtn.addEventListener("click", async () => {
  try {
    setStatus(cameraStarted ? "Disabling camera..." : "Requesting camera permission...");
    await startCamera();
  } catch (err) {
    console.error(err);
    setStatus(`Camera permission failed: ${cameraGuidanceFromError(err)}`);
  }
});

for (const el of [roiEnabledEl, roiXEl, roiYEl, roiWEl, roiHEl]) {
  if (!el) {
    continue;
  }
  el.addEventListener("input", () => {
    syncRoiStateFromInputs();
  });
}

window.addEventListener("resize", () => {
  updateRoiOverlay();
});

recordBtn.addEventListener("click", async () => {
  try {
    if (isRecording()) {
      stopRecording();
    } else {
      await startRecording();
    }
  } catch (err) {
    console.error(err);
    recordBtn.textContent = "Start Recording";
    setStatus(`Recording failed: ${formatError(err)}`);
  }
});

window.addEventListener("beforeunload", () => {
  stopCameraStreamOnExit();
});

drawBlackFrame();
syncRoiInputsFromState();
setStatus("Idle");
