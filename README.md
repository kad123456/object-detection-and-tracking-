# Object detection and tracking

A small Python project for detecting objects in a video file, webcam, or RTSP/HTTP stream and following them across frames with persistent IDs. It uses Ultralytics YOLO for detections, Ultralytics' built-in ByteTrack by default (or BoT-SORT), and OpenCV for video input, annotation, and output.

## Live web app on Vercel

The repository also includes a browser app for Vercel. It runs YOLOv8n through Ultralytics' browser inference package, using WebGPU when available and CPU/WASM otherwise. Video files and webcam frames are processed in the browser; the app exports an annotated WebM video and a tracking CSV. No upload API is used.

```powershell
Set-Location web
npm install
npm run dev
```

For a production build, run `npm run build` from `web/`. Vercel uses `web/` as its project root and `web/vercel.json` to build the Vite app and serve `web/dist/` as a static deployment. The ONNX model is included at `web/public/models/yolov8n.onnx`. The browser package and model are subject to Ultralytics' [AGPL-3.0 license](https://github.com/ultralytics/inference/blob/main/web/LICENSE).

The static app supports video files and webcam input. RTSP sources remain available through the Python CLI below, since browsers cannot read arbitrary RTSP streams directly.

## Requirements and install

- Python 3.10 or newer
- Windows, macOS, or Linux
- A CUDA-enabled PyTorch build is optional; the program selects CUDA when `torch.cuda.is_available()` is true and otherwise uses CPU.

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`. For a specific CUDA build, use the install command from the [PyTorch selector](https://pytorch.org/get-started/locally/) while keeping the pinned PyTorch/torchvision versions compatible. The default `yolov8n.pt` weights download on the first run if they are not already available locally.

## Run

Sample run on the included public demo clip:

```powershell
python -m src.main --source samples/solutions_ci_demo.mp4 --model yolov8n.pt --tracker bytetrack --conf 0.4 --save --no-show --output-dir outputs/sample-run
```

The short demo clip is from [Ultralytics Assets](https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4).

Run on your own video, webcam, or stream:

```powershell
python -m src.main --source path\to\video.mp4 --model yolov8n.pt --tracker bytetrack --conf 0.4 --classes person car --save --show
```

Other sources:

```powershell
python -m src.main --source 0 --tracker botsort --conf 0.4 --save --show
python -m src.main --source rtsp://user:password@camera-address/stream --imgsz 640 --frame-skip 2
```

The CLI reads defaults from `configs/config.yaml`; command-line values override them. `--no-save` disables video and CSV output, while `--no-show` disables the display. `--frame-skip 2` runs detection and tracking on every second frame and carries the most recent annotations across the intervening frames. Use `--classes person car` to filter by model class names (numeric class IDs are also accepted). Press **q** in the display window or **Ctrl+C** in the terminal to stop; output files are finalized on exit.

Optional line crossing uses a finite segment in pixel coordinates:

```powershell
python -m src.main --source path\to\video.mp4 --line-crossing 320 0 320 720
```

The program writes `<source>_tracked.mp4` and `<source>_tracks.csv` under `outputs/` by default. Webcam output uses `webcam_<index>` and live streams use `stream`. The CSV columns are `frame, track_id, class, confidence, x1, y1, x2, y2`; frame numbers start at zero and refer to the original input. The display and annotated video include a live FPS estimate, track labels, short motion trails, and the optional crossing count. At completion, average FPS and mean inference, tracking, and drawing latency are logged.

## Configuration

Edit `configs/config.yaml` to change defaults without adding command-line flags:

```yaml
model: yolov8n.pt
tracker: bytetrack
conf: 0.4
imgsz: 640
classes: []
frame_skip: 1
save: true
show: false
output_dir: outputs
trail_length: 24
line_crossing: null
```

`line_crossing` accepts four pixel coordinates, for example `[320, 0, 320, 720]`.

## Tests

```powershell
python -m pytest -q
```

The tests cover drawing helpers, line-crossing counts, CSV formatting, and an end-to-end smoke run against a generated short clip. The smoke test injects deterministic detector/tracker stubs so it does not need network access or model weights.

## Notes

- Video is processed as a stream; frames are not accumulated in memory.
- Inference uses FP16 when a CUDA device is available and FP32 on CPU.
- `--show` needs a display environment. A remote or headless machine can run with `--no-show`.
- The tracker update runs only on frames selected by `--frame-skip`; larger skips can reduce ID continuity and make displayed boxes stale between inference frames.
- The reported FPS includes the complete frame-processing loop. Stage latencies are averages over the frames that ran the corresponding stage.
- Live capture uses OpenCV's built-in stream handling. A separate capture thread is not enabled by default; no live source was available here to establish that threading improves FPS for this workload.
