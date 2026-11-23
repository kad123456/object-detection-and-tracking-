import { YOLO } from "@ultralytics/yolo";
import "./style.css";

const MODEL_URL = "/models/yolov8n.onnx";
const $ = (id) => document.getElementById(id);

const video = $("source-video");
const canvas = $("tracking-canvas");
const ctx = canvas.getContext("2d", { alpha: false });
const fileInput = $("file-input");
const runButton = $("run-button");
const stopButton = $("stop-button");
const statusMessage = $("status-message");

let model = null;
let sourceKind = null;
let objectUrl = null;
let cameraStream = null;
let running = false;
let busy = false;
let frameNumber = 0;
let startedAt = 0;
let sampledFrames = 0;
let inferenceTotal = 0;
let lastResult = null;
let lastTracks = [];
let recording = null;
let recordingStream = null;
let recordedChunks = [];
let csvRows = [];
let trails = new Map();
let lastCenters = new Map();
let countedIds = new Set();
let crossingLine = null;
let crossings = 0;

function setStatus(message, error = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", error);
}

function classEntries() {
  const names = model?.names ?? {};
  return Array.isArray(names) ? names.map((name, id) => [id, name]) : Object.entries(names);
}

function resolveClassFilter() {
  const raw = $("class-filter").value.trim();
  if (!raw) return undefined;
  const names = classEntries();
  const byName = new Map(names.map(([id, name]) => [String(name).toLowerCase(), Number(id)]));
  const ids = [];
  for (const token of raw.split(/[ ,]+/).filter(Boolean)) {
    let id;
    if (/^\d+$/.test(token)) {
      id = Number(token);
      if (!names.some(([candidate]) => Number(candidate) === id)) {
        throw new Error(`Unknown class ID ${id}.`);
      }
    } else {
      id = byName.get(token.toLowerCase());
      if (id === undefined) throw new Error(`Unknown class “${token}”. Check the model's class names.`);
    }
    if (!ids.includes(id)) ids.push(id);
  }
  return ids;
}

function readCrossingLine() {
  const raw = $("crossing-line").value.trim();
  if (!raw) return null;
  const values = raw.split(/[ ,]+/).map(Number);
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("Line crossing must be four numbers: x1,y1,x2,y2.");
  }
  return [[values[0], values[1]], [values[2], values[3]]];
}

async function loadModel() {
  if (model) return model;
  setStatus("Loading YOLOv8n model… first load is about 13 MB.");
  $("device-badge").textContent = "LOADING MODEL";
  model = await YOLO.load(MODEL_URL, { device: "auto" });
  $("device-badge").textContent = model.device === "webgpu" ? "WEBGPU" : "CPU / WASM";
  return model;
}

function setSourceName(name) {
  $("source-name").textContent = name;
  $("empty-state").hidden = true;
  $("preview-stage").classList.add("has-source");
  runButton.disabled = false;
}

function setSourceControlsDisabled(disabled) {
  fileInput.disabled = disabled;
  $("camera-button").disabled = disabled;
  $("drop-zone").classList.toggle("controls-disabled", disabled);
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = null;
  }
  if (video.srcObject) video.srcObject = null;
}

function clearSource() {
  stopCamera();
  video.pause();
  video.removeAttribute("src");
  video.load();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  sourceKind = null;
  running = false;
  lastResult = null;
  lastTracks = [];
  trails.clear();
  lastCenters.clear();
  countedIds.clear();
  crossings = 0;
  $("crossing-count").textContent = "0";
  $("track-count").textContent = "0";
  $("fps-value").textContent = "—";
  $("inference-time").textContent = "—";
  $("preview-stage").classList.remove("has-source");
  $("empty-state").hidden = false;
  runButton.disabled = true;
  $("export-row").hidden = true;
  setStatus("Select a video or start your webcam.");
}

function selectVideoFile(file) {
  if (busy) return;
  if (!file || !file.type.startsWith("video/")) {
    setStatus("Choose a supported video file.", true);
    return;
  }
  clearSource();
  objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  sourceKind = "file";
  video.onloadedmetadata = () => {
    setSourceName(`${file.name} · ${video.videoWidth}×${video.videoHeight}`);
    fitCanvas(video.videoWidth, video.videoHeight);
    video.currentTime = 0;
    setStatus("Ready. Start tracking to process this clip.");
  };
  video.onerror = () => setStatus("The selected video could not be opened by this browser.", true);
}

function fitCanvas(width, height) {
  canvas.width = Math.max(1, width);
  canvas.height = Math.max(1, height);
}

async function startCamera() {
  try {
    clearSource();
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    video.srcObject = cameraStream;
    sourceKind = "camera";
    await new Promise((resolve) => { video.onloadedmetadata = resolve; });
    await video.play();
    fitCanvas(video.videoWidth, video.videoHeight);
    setSourceName(`Camera · ${video.videoWidth}×${video.videoHeight}`);
    setStatus("Camera ready. Start tracking when you’re ready.");
  } catch (error) {
    stopCamera();
    setStatus(error?.name === "NotAllowedError" ? "Allow camera access in your browser, then try again." : `Could not start camera: ${error.message}`, true);
  }
}

function lineSegmentsIntersect(a, b, c, d) {
  const cross = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const on = (p, q, r) => r[0] >= Math.min(p[0], q[0]) - 1e-8 && r[0] <= Math.max(p[0], q[0]) + 1e-8 && r[1] >= Math.min(p[1], q[1]) - 1e-8 && r[1] <= Math.max(p[1], q[1]) + 1e-8;
  const abC = cross(a, b, c), abD = cross(a, b, d), cdA = cross(c, d, a), cdB = cross(c, d, b);
  if (abC * abD < 0 && cdA * cdB < 0) return true;
  return (Math.abs(abC) < 1e-8 && on(a, b, c)) || (Math.abs(abD) < 1e-8 && on(a, b, d)) || (Math.abs(cdA) < 1e-8 && on(c, d, a)) || (Math.abs(cdB) < 1e-8 && on(c, d, b));
}

function updateCrossings(boxes) {
  if (!crossingLine) return;
  for (const box of boxes) {
    if (box.id === undefined || box.id === null) continue;
    const id = Number(box.id);
    const center = [(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2];
    const previous = lastCenters.get(id);
    if (previous && !countedIds.has(id) && lineSegmentsIntersect(previous, center, crossingLine[0], crossingLine[1])) {
      countedIds.add(id);
      crossings += 1;
    }
    lastCenters.set(id, center);
  }
  $("crossing-count").textContent = String(crossings);
}

function recordRows(boxes) {
  for (const box of boxes) {
    if (box.id === undefined || box.id === null) continue;
    csvRows.push([
      frameNumber,
      box.id,
      box.name,
      Number(box.conf).toFixed(6),
      Number(box.x1).toFixed(3),
      Number(box.y1).toFixed(3),
      Number(box.x2).toFixed(3),
      Number(box.y2).toFixed(3),
    ]);
  }
}

function pushTrails(boxes) {
  for (const box of boxes) {
    if (box.id === undefined || box.id === null) continue;
    const id = Number(box.id);
    const points = trails.get(id) ?? [];
    points.push([(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2]);
    if (points.length > 24) points.shift();
    trails.set(id, points);
  }
}

function colorForId(id) {
  return `hsl(${(Number(id) * 67 + 99) % 360} 78% 62%)`;
}

function drawFrame(image, boxes, fps, inferenceMs) {
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const threshold = Number($("confidence").value);
  const visible = (boxes ?? []).filter((box) => Number(box.conf) >= threshold);
  const scaleX = canvas.width / (lastResult?.width || image.width || canvas.width);
  const scaleY = canvas.height / (lastResult?.height || image.height || canvas.height);

  for (const box of visible) {
    const color = colorForId(box.id ?? box.cls);
    const x1 = box.x1 * scaleX, y1 = box.y1 * scaleY, x2 = box.x2 * scaleX, y2 = box.y2 * scaleY;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, canvas.width / 800);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (box.id !== undefined && box.id !== null) {
      const points = trails.get(Number(box.id)) ?? [];
      if (points.length > 1) {
        ctx.beginPath();
        points.forEach((point, index) => index === 0 ? ctx.moveTo(point[0] * scaleX, point[1] * scaleY) : ctx.lineTo(point[0] * scaleX, point[1] * scaleY));
        ctx.stroke();
      }
    }

    const label = `${box.name} ${Number(box.conf).toFixed(2)}${box.id === undefined || box.id === null ? "" : ` ID:${box.id}`}`;
    ctx.font = `600 ${Math.max(12, canvas.width / 90)}px "DM Sans", sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const labelY = Math.max(0, y1 - 23);
    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, textWidth + 12, 22);
    ctx.fillStyle = "#111711";
    ctx.fillText(label, x1 + 6, labelY + 15);
  }

  if (crossingLine) {
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = Math.max(2, canvas.width / 700);
    ctx.beginPath();
    ctx.moveTo(crossingLine[0][0] * scaleX, crossingLine[0][1] * scaleY);
    ctx.lineTo(crossingLine[1][0] * scaleX, crossingLine[1][1] * scaleY);
    ctx.stroke();
  }

  const fontSize = Math.max(13, canvas.width / 75);
  ctx.font = `600 ${fontSize}px "DM Mono", monospace`;
  ctx.fillStyle = "#b7f36b";
  ctx.fillText(`FPS ${fps.toFixed(1)}`, 14, fontSize + 14);
  if (crossingLine) {
    ctx.fillStyle = "#ffd166";
    ctx.fillText(`CROSSINGS ${crossings}`, 14, fontSize * 2 + 22);
  }
  $("track-count").textContent = String(visible.filter((box) => box.id !== undefined && box.id !== null).length);
  $("fps-value").textContent = fps.toFixed(1);
  $("inference-time").textContent = inferenceMs > 0 ? `${inferenceMs.toFixed(0)} ms` : "—";
}

async function advanceFileFrame() {
  if (sourceKind !== "file" || !running || video.ended) return;
  await video.play();
  if (!running || video.ended) return;

  await new Promise((resolve) => {
    let callbackId;
    let timerId;
    const finish = () => {
      video.pause();
      video.removeEventListener("ended", finish);
      if (callbackId !== undefined) video.cancelVideoFrameCallback?.(callbackId);
      if (timerId !== undefined) clearTimeout(timerId);
      resolve();
    };
    video.addEventListener("ended", finish, { once: true });
    if (video.requestVideoFrameCallback) {
      callbackId = video.requestVideoFrameCallback(finish);
    } else {
      timerId = setTimeout(finish, 1000 / 30);
    }
  });
}

function csvEscape(value) {
  const string = String(value ?? "");
  return /[",\r\n]/.test(string) ? `"${string.replaceAll('"', '""')}"` : string;
}

function downloadCsv() {
  const header = ["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"];
  const csv = [header, ...csvRows].map((row) => row.map(csvEscape).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = $("csv-download");
  if (link.dataset.url) URL.revokeObjectURL(link.dataset.url);
  link.dataset.url = url;
  link.href = url;
}

function finalizeRecording() {
  if (!recording || recording.state === "inactive") return;
  recording.stop();
  recording = null;
}

function beginRecording() {
  if (!canvas.captureStream || !window.MediaRecorder) {
    setStatus("This browser cannot record the annotated canvas. CSV export will still be available.", true);
    return;
  }
  recordedChunks = [];
  recordingStream = canvas.captureStream(30);
  const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((type) => MediaRecorder.isTypeSupported(type));
  const activeRecorder = new MediaRecorder(recordingStream, mime ? { mimeType: mime } : undefined);
  recording = activeRecorder;
  activeRecorder.ondataavailable = (event) => { if (event.data.size) recordedChunks.push(event.data); };
  activeRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: activeRecorder.mimeType || "video/webm" });
    const link = $("video-download");
    if (link.dataset.url) URL.revokeObjectURL(link.dataset.url);
    link.dataset.url = URL.createObjectURL(blob);
    link.href = link.dataset.url;
    link.hidden = false;
    $("export-row").hidden = false;
    for (const track of recordingStream?.getTracks() ?? []) track.stop();
    recordingStream = null;
  };
  activeRecorder.start(1000);
}

async function stopTracking({ ended = false } = {}) {
  if (!running) return;
  running = false;
  video.pause();
  stopButton.hidden = true;
  runButton.hidden = false;
  runButton.disabled = false;
  busy = false;
  setSourceControlsDisabled(false);
  $("preview-title").classList.remove("is-live");
  $("preview-label").textContent = "PREVIEW";
  finalizeRecording();
  downloadCsv();
  $("export-row").hidden = false;
  if (!recording) $("video-download").hidden = recordedChunks.length === 0;
  setStatus(ended ? "Video complete. Downloads are ready." : "Tracking stopped. Downloads are ready.");
  if (sourceKind === "camera") {
    stopCamera();
    sourceKind = null;
    $("source-name").textContent = "Camera stopped · choose webcam to run again";
    runButton.disabled = true;
  }
}

async function processFrame() {
  if (!running) return;
  if (sourceKind === "file") video.pause();
  const frame = await createImageBitmap(video);
  const frameStarted = performance.now();
  let inferenceMs = 0;
  try {
    if (frameNumber % Number($("frame-skip").value) === 0) {
      const options = {
        tracker: `${$("tracker-select").value}.yaml`,
        conf: 0.1,
      };
      const classes = resolveClassFilter();
      if (classes) options.classes = classes;
      const inferenceStarted = performance.now();
      lastResult = await model.track(frame, options);
      inferenceMs = performance.now() - inferenceStarted;
      inferenceTotal += inferenceMs;
      sampledFrames += 1;
      lastTracks = lastResult.boxes ?? [];
      const threshold = Number($("confidence").value);
      const csvTracks = lastTracks.filter((box) => Number(box.conf) >= threshold && box.id !== undefined && box.id !== null);
      pushTrails(csvTracks);
      updateCrossings(csvTracks);
      recordRows(csvTracks);
    }
    const elapsed = (performance.now() - startedAt) / 1000;
    const fps = elapsed > 0 ? (frameNumber + 1) / elapsed : 0;
    drawFrame(frame, lastTracks, fps, inferenceMs || (sampledFrames ? inferenceTotal / sampledFrames : 0));
    frameNumber += 1;
  } catch (error) {
    running = false;
    busy = false;
    video.pause();
    finalizeRecording();
    downloadCsv();
    $("export-row").hidden = false;
    stopButton.hidden = true;
    runButton.hidden = false;
    runButton.disabled = false;
    setSourceControlsDisabled(false);
    $("preview-title").classList.remove("is-live");
    $("preview-label").textContent = "PREVIEW";
    setStatus(error.message || "Tracking failed.", true);
  } finally {
    frame.close();
  }

  if (!running) return;
  if (sourceKind === "file" && video.ended) {
    await stopTracking({ ended: true });
    return;
  }
  if (sourceKind === "file") {
    try {
      await advanceFileFrame();
    } catch (error) {
      await stopTracking();
      setStatus(`Video playback stopped: ${error.message}`, true);
      return;
    }
    if (!running) return;
  }
  requestAnimationFrame(processFrame);
}

async function startTracking() {
  if (!sourceKind) return;
  if (busy) return;
  busy = true;
  runButton.disabled = true;
  setSourceControlsDisabled(true);
  try {
    await loadModel();
    resolveClassFilter();
    crossingLine = readCrossingLine();
    model.resetTracker();
    frameNumber = 0;
    sampledFrames = 0;
    inferenceTotal = 0;
    csvRows = [];
    lastResult = null;
    lastTracks = [];
    trails.clear();
    lastCenters.clear();
    countedIds.clear();
    crossings = 0;
    $("crossing-count").textContent = "0";
    $("export-row").hidden = true;
    if (sourceKind === "file") video.currentTime = 0;
    await video.play();
    fitCanvas(video.videoWidth, video.videoHeight);
    running = true;
    startedAt = performance.now();
    $("preview-title").classList.add("is-live");
    $("preview-label").textContent = "TRACKING";
    runButton.hidden = true;
    stopButton.hidden = false;
    beginRecording();
    setStatus("Tracking is running. Stop to finalize your video and CSV.");
    requestAnimationFrame(processFrame);
  } catch (error) {
    busy = false;
    setSourceControlsDisabled(false);
    setStatus(error.message || "Could not start tracking.", true);
    runButton.disabled = false;
  }
}

$("confidence").addEventListener("input", () => { $("confidence-value").value = Number($("confidence").value).toFixed(2); });
fileInput.addEventListener("change", (event) => selectVideoFile(event.target.files?.[0]));
$("camera-button").addEventListener("click", startCamera);
runButton.addEventListener("click", startTracking);
stopButton.addEventListener("click", () => stopTracking());
video.addEventListener("ended", () => { if (running) stopTracking({ ended: true }); });

const dropZone = $("drop-zone");
for (const name of ["dragenter", "dragover"]) {
  dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("drag-active"); });
}
for (const name of ["dragleave", "drop"]) {
  dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("drag-active"); });
}
dropZone.addEventListener("drop", (event) => selectVideoFile(event.dataTransfer?.files?.[0]));
