# Vehicle Detection & Speed Estimation Project

## Project Overview

This is a **YOLOv8-based vehicle detection and real-time speed estimation system** with a GUI interface. The system uses optical flow + homography for speed calculation and works with both live camera feeds and video files.

**Core Technologies:**
- YOLOv8 object detection (9 vehicle classes)
- Lucas-Kanade optical flow for vehicle tracking
- Homography for pixel-to-world coordinate transformation
- PyTorch with GPU acceleration (GTX 3050 / 4GB VRAM)
- Tkinter GUI with real-time vehicle monitoring

---

## Architecture

### Entry Points

| Script | Purpose | Command |
|--------|---------|---------|
| `train_yolov8.py` | Train detection model from scratch | `python train_yolov8.py` |
| `calibrate.py` | Interactive road calibration tool | `python calibrate.py` (run FIRST before speed estimation) |
| `speed_estimator.py` | Batch video processing with speed detection | `python speed_estimator.py` |
| `vehicle_speed_v2.py` | GUI app with live camera/video + real-time monitoring | `python vehicle_speed_v2.py` |

### Data & Model Organization

```
runs/vehicle_detection_v1/       → Best trained model
  weights/best.pt               → Production model (used in vehicle_speed_v2.py)
  args.yaml, results.csv        → Training metadata

train/, valid/, test/           → Dataset (9 classes, 9991 images)
  images/, labels/              → Roboflow format (YOLO .txt format)

*.pt files                       → Pre-trained model checkpoints
```

### Key Components

1. **Detection Pipeline**: YOLOv8 inference on video frames
2. **Tracking**: Optical flow to track feature points within bounding boxes
3. **Speed Calculation**: Homography matrix maps pixel displacement → real-world meters
4. **Distance Estimation**: Vehicle bottom-center projected to road surface
5. **GUI Dashboard**: Vehicle table, warning overlay, live feed canvas

---

## Git Workflow & Conventions

### Commit Message Guidelines

Use descriptive, feature-focused commits to track changes across training iterations:

```
feat: [model] description (experiments/model-updates)
- Train YOLOv8s with batch=8, 100 epochs
- Update model weights in runs/vehicle_detection_v1/

fix: [speed] description (calibration/detection logic)
- Fix homography matrix scaling in speed_estimator.py

data: [dataset] description (data preparation)
- Add 50 new frames for truck class
- Re-balance training/validation split

docs: description
- Update calibration instructions in calibrate.py

ci: description (automation, model versioning)
- Add model checkpoint versioning
```

### Branches

- `main` — Production-ready models and stable code
- `experiments/*` — Model training iterations (e.g., `experiments/yolov8m-high-speed`)
- `features/*` — New features (GUI updates, speed algorithm improvements)

### Large Files & Models

- Git LFS is **not configured** — model weights (`.pt`) are currently committed directly
- **Recommendation**: Exclude large files in `.gitignore` once LFS is set up:
  ```
  *.pt filter=lfs diff=lfs merge=lfs -text
  runs/**/*.pt filter=lfs diff=lfs merge=lfs -text
  ```

---

## Build & Run Commands

### Setup (One-time)

```powershell
# Install dependencies
pip install ultralytics torch torchvision opencv-python Pillow

# Verify GPU setup
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Training a New Model

```powershell
# Run training with default config (GTX 3050 optimized)
python train_yolov8.py

# Monitor GPU memory
nvidia-smi -l 1
```

**Key Config in `train_yolov8.py`:**
- `model`: "yolov8s.pt" (4GB VRAM, batch=8) or "yolov8m.pt" (needs batch=4)
- `epochs`: 100, `patience`: 20
- Early stopping auto-triggers on no mAP gain

### Calibration (First-time per camera)

```powershell
# Opens your video, click 4 road points, measure real distances
python calibrate.py

# Copy printed SRC_POINTS & DST_POINTS into:
# - speed_estimator.py (for batch processing)
# - vehicle_speed_v2.py (for GUI)
```

### GUI Application (Main Interface)

```powershell
python vehicle_speed_v2.py
```

**Features:**
- Source selector (camera index or video file)
- Real-time vehicle detection & speed display
- Warning overlay (overspeed vehicles, closest-vehicle distance)
- Vehicle data table (refreshes every frame)
- Calibration manager

### Batch Video Processing

```powershell
python speed_estimator.py
# Outputs annotated video to: runs/speed_output.mp4
```

---

## Critical Paths & Environment

### Hardcoded Paths (Update Before Running)

These scripts assume Windows paths — update for your environment:

| Script | Variable | Default |
|--------|----------|---------|
| `train_yolov8.py` | `data_yaml` | `C:/Final project/data.yaml` |
| `calibrate.py` | `VIDEO_SRC` | `C:\Users\Nadhu\Downloads\...` |
| `speed_estimator.py` | `WEIGHTS` | `C:\Final project\runs\vehicle_detection_v1\weights\best.pt` |
| `speed_estimator.py` | `VIDEO_SRC` | `C:\Users\Nadhu\Downloads\...` |
| `vehicle_speed_v2.py` | `DEFAULT_MODEL` | `C:\Final project\runs\vehicle_detection_v1\weights\best.pt` |
| `vehicle_speed_v2.py` | `DEFAULT_VIDEO` | `C:\Users\Nadhu\Downloads\...` |

### GPU Memory Optimization

```python
# Already set in train_yolov8.py:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
```

**Troubleshooting:**
- **OOM Error**: Reduce `batch` size (8 → 4) or use "yolov8n.pt" instead of "s"
- **Slow training**: Ensure GPU is being used (`nvidia-smi`)

---

## Common Pitfalls & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| Speed shows 0 for all vehicles | Calibration points not set | Run `calibrate.py` first, copy SRC/DST_POINTS |
| Poor detection accuracy | Using wrong model weights | Ensure `DEFAULT_MODEL` points to `best.pt` from latest run |
| Video not found | Hardcoded path outdated | Update `VIDEO_SRC` path in script |
| Camera won't connect | Camera already in use or index wrong | Try index 0, 1, 2 in GUI source selector |
| CUDA out of memory | Batch size too large | Reduce in `train_yolov8.py`: batch=4 |
| Homography matrix error | Image too small or no features detected | Use higher-quality video, ensure wide road view |

---

## Dependencies

**Core Libraries** (auto-installed by `pip install`):
- `ultralytics` — YOLOv8 framework
- `torch`, `torchvision` — PyTorch (GPU or CPU)
- `opencv-python` — Video I/O, image processing
- `Pillow` — Image handling in GUI
- `numpy` — Numerical operations

**Hardware Requirements:**
- GPU: NVIDIA with CUDA support (4GB+ VRAM recommended)
- CPU fallback: Extremely slow (~30 min per training epoch)

---

## Development Workflow

### When Adding a New Feature

1. Create feature branch: `git checkout -b features/my-feature`
2. Update relevant script(s)
3. Test locally (e.g., `python vehicle_speed_v2.py`)
4. Commit with descriptive message: `feat: [component] description`
5. Test on GPU before merging to main
6. Merge to main when stable

### When Retraining the Model

1. Update training config in `train_yolov8.py` (epochs, batch, etc.)
2. Commit config changes: `data: augment training dataset` or `feat: [model] experiment with yolov8m`
3. Run `python train_yolov8.py`
4. Review `runs/vehicle_detection_v1/results.csv` for metrics
5. If mAP improves significantly, commit new weights: `ci: update best.pt (mAP +0.05)`
6. Test in GUI before marking as production-ready

### Model Versioning Pattern

```
runs/vehicle_detection_v1/      → Base version
runs/vehicle_detection_v1_val/  → Validation run (temporary)
runs/vehicle_detection_v12/     → Iteration 12 (if experimenting)
  → Commit pattern: "ci: [model-v12] description"
```

---

## Useful Commands for AI Agents

```powershell
# Check which model weights exist
Get-ChildItem -Recurse -Filter "best.pt" c:\Final\ project\runs\

# List dataset structure
ls train/images | Measure-Object -Line
ls valid/images | Measure-Object -Line

# Extract latest training metrics
Get-Content runs/vehicle_detection_v1/results.csv | Select-Object -Last 3

# Check git branches with uncommitted changes
git status
git branch -v
```

---

## Next Steps for Customization

- [ ] Create `.gitignore` to exclude `runs/**/*.pt`, video files, cache directories
- [ ] Set up Git LFS for model weights (`.pt` files)
- [ ] Add GitHub Actions workflow for automated model testing
- [ ] Create CONTRIBUTING.md with dataset labeling guidelines
- [ ] Add pytest suite for speed calculation accuracy tests
