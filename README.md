# TrackVision

Real-time object detection and multi-object tracking using YOLOv8 and DeepSORT.

---

## What it does

TrackVision processes live webcam feeds or local video files frame by frame, detects objects using YOLOv8, assigns persistent tracking IDs via DeepSORT, and renders annotated output with bounding boxes, labels, velocity vectors, and track trails. A FastAPI server exposes the stream and detections as an API for downstream integration.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU mode supported)
- FFmpeg installed for video writing

```
pip install -r requirements.txt
```

`requirements.txt` covers: `ultralytics`, `deep-sort-realtime`, `opencv-python`, `fastapi`, `uvicorn`, `torch`, `torchvision`, `numpy`, `pyyaml`, `scipy`.

---

## Project Structure

```
trackvision/
├── main.py                  # Entry point
├── config.yaml              # All runtime configuration
├── requirements.txt
├── core/
│   ├── input.py             # Frame generator (webcam + file)
│   ├── preprocessor.py      # Resize, normalize
│   ├── detector.py          # YOLOv8 and Faster R-CNN backends
│   ├── tracker.py           # DeepSORT and SORT backends
│   ├── renderer.py          # OpenCV overlay drawing
│   └── state.py             # In-memory track state store
├── api/
│   ├── server.py            # FastAPI app
│   └── routes.py            # /stream, /detections, /stats, /config
├── output/
│   ├── csv_logger.py        # Per-frame detection CSV writer
│   └── video_writer.py      # MP4 H.264 output
└── runs/                    # Generated output files
```

---

## Quickstart

**Run on webcam (default):**
```bash
python main.py
```

**Run on a video file:**
```bash
python main.py --source path/to/video.mp4
```

**Headless mode with API only (no display window):**
```bash
python main.py --headless
```

**Save annotated output video:**
```bash
python main.py --source input.mp4 --save-video
```

**Use a custom config file:**
```bash
python main.py --config my_config.yaml
```

---

## Configuration

All settings live in `config.yaml`. Key options:

```yaml
input:
  source: 0               # 0 = default webcam; or a file path
  width: 1280
  height: 720

detector:
  backend: yolov8         # yolov8 | faster_rcnn
  model_size: small       # nano | small | medium
  confidence: 0.45
  nms_iou: 0.50
  classes: []             # [] = all COCO classes; e.g. [0, 2] = person, car

tracker:
  backend: deepsort       # deepsort | sort
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  reid_model: osnet_x0_25

visualization:
  trail_length: 20
  draw_velocity: true
  headless: false

output:
  save_video: false
  save_csv: true
  output_dir: ./runs/

api:
  enabled: true
  port: 8080
```

---

## API Endpoints

When `api.enabled: true`, the server starts on the configured port alongside inference.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/stream` | MJPEG stream of annotated video, viewable in any browser |
| GET | `/detections/latest` | JSON of all active tracks in the most recent frame |
| GET | `/detections/history?track_id=N` | Full position history of a specific track |
| POST | `/config` | Hot-reload configuration without restarting the process |
| GET | `/stats` | Session summary: unique IDs seen, class breakdown, avg FPS |
| GET | `/health` | Uptime check |

**Example response from `/detections/latest`:**
```json
{
  "frame": 412,
  "timestamp": "2026-06-06T14:23:11.204Z",
  "tracks": [
    {
      "id": 7,
      "class": "person",
      "confidence": 0.87,
      "bbox": [312, 148, 420, 390],
      "velocity": [1.2, 0.4],
      "age": 38,
      "state": "confirmed"
    }
  ]
}
```

Open the live stream in a browser:
```
http://localhost:8080/stream
```

---

## Output Files

All output is written to `./runs/` by default (configurable via `output.output_dir`).

`detections_<timestamp>.csv` — one row per detection per frame:
```
frame_id, timestamp, track_id, class, x1, y1, x2, y2, confidence
412, 2026-06-06T14:23:11.204Z, 7, person, 312, 148, 420, 390, 0.87
```

`session_<timestamp>.json` — written at end of run:
```json
{
  "duration_seconds": 142,
  "total_frames": 3550,
  "avg_fps": 24.9,
  "peak_fps": 31.2,
  "unique_track_ids": 14,
  "class_counts": { "person": 11, "car": 3 }
}
```

`output_<timestamp>.mp4` — annotated video, only written when `--save-video` is passed.

---

## Switching Backends

**Use SORT instead of DeepSORT** (faster, no Re-ID, suitable for CPU-only):
```yaml
tracker:
  backend: sort
```

**Use Faster R-CNN instead of YOLOv8** (higher accuracy, slower):
```yaml
detector:
  backend: faster_rcnn
```

Both swaps require only a config change. No code modifications needed.

---

## Bounding Box Colors

Colors are assigned per track ID using a deterministic hash: `hue = (track_id × 47) mod 360` at fixed saturation and value. The same track ID always renders the same color. There is no random color assignment.

---

## Performance Reference

| Setup | Hardware | Target FPS |
|---|---|---|
| YOLOv8n + SORT | CPU (i7 12th gen) | ≥ 28 |
| YOLOv8s + DeepSORT | GPU (RTX 3060) | ≥ 45 |
| YOLOv8m + DeepSORT | GPU (RTX 3060) | ≥ 28 |
| YOLOv8n + SORT | Raspberry Pi 5 | ≥ 8 |

Run `python benchmark.py` to measure actual throughput on your hardware across a 500-frame window.

---

## Keyboard Shortcuts (OpenCV Window)

| Key | Action |
|---|---|
| `q` | Quit |
| `p` | Pause / resume |
| `s` | Save screenshot of current frame |
| `h` | Toggle track trail visibility |
| `v` | Toggle velocity vector visibility |
| `i` | Toggle HUD overlay |

---

## Common Issues

**Camera not found:**
Check your device index. Try `source: 1` or `source: 2` in config if `0` fails. On Linux, list devices with `ls /dev/video*`.

**Low FPS on CPU:**
Switch to `model_size: nano` and `tracker: backend: sort`. Disable video saving and API if not needed.

**CUDA out of memory:**
Reduce input resolution (`width: 640`, `height: 360`) or switch to `model_size: nano`.

**Track IDs reset between runs:**
Expected behavior. IDs are session-scoped integers starting from 1. Cross-session identity is not supported in v1.0.

---

## License

MIT License. See `LICENSE` for details. Model weights are subject to their respective upstream licenses — YOLOv8 weights are licensed under AGPL-3.0 by Ultralytics.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Deep SORT Realtime](https://github.com/levan92/deep_sort_realtime)
- [OSNet Re-ID](https://github.com/KaiyangZhou/deep-person-reid)
- [OpenCV](https://opencv.org)
- MOT17 benchmark for tracking evaluation
- 
