"""
YOLOv8 Vehicle Detection — Real-Time Video Display
Works with: video files (.mp4, .avi, .mov) and webcam
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import time

# ─────────────────────────────────────────
#  CONFIG — Change these to your paths
# ─────────────────────────────────────────
WEIGHTS   = r"C:\Final project\runs\vehicle_detection_v1\weights\best.pt"
VIDEO_SRC = r"C:\Users\Nadhu\Downloads\stock-footage-car-driving-on-the-road-in-leh-ladakh-india-october-th.mp4"  # ← your video file
                                                             # use 0 for webcam

CONF      = 0.3     # confidence threshold (lower = more detections)
IOU       = 0.45     # NMS overlap threshold
IMGSZ     = 640      # inference image size
SAVE_OUT  = False     # save output video to disk
OUT_PATH  = r"C:\Final project\runs\output_detected.mp4"

# Class colors (BGR) — one per class index
# 0:Tree weeler  1:bicycle  2:bus  3:car  4:minitruck
# 5:motorbike    6:riksha   7:truck  8:van
CLASS_COLORS = {
    0: (255, 180,  50),   # Tree weeler — blue-ish
    1: ( 50, 220, 255),   # bicycle     — yellow
    2: ( 50,  50, 255),   # bus         — red
    3: ( 50, 200,  50),   # car         — green
    4: (255, 100, 200),   # minitruck   — pink
    5: (  0, 165, 255),   # motorbike   — orange
    6: (180,  50, 255),   # riksha      — purple
    7: (  0,  80, 255),   # truck       — deep red
    8: (255, 220,   0),   # van         — cyan
}


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
def load_model(weights: str) -> YOLO:
    print(f"Loading model: {weights}")
    model = YOLO(weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on   : {device.upper()}")
    model.to(device)
    return model


# ─────────────────────────────────────────
#  DRAW BOXES ON FRAME
# ─────────────────────────────────────────
def draw_boxes(frame, result, names):
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{names[cls_id]}  {conf:.2f}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

        # Label text
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────
#  OVERLAY — FPS + COUNT
# ─────────────────────────────────────────
def draw_overlay(frame, fps, counts, names):
    h, w = frame.shape[:2]

    # FPS top-left
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Detection count panel — top-right
    if counts:
        panel_x = w - 200
        y_start = 10
        cv2.rectangle(frame, (panel_x - 8, y_start),
                      (w - 8, y_start + len(counts) * 22 + 10),
                      (20, 20, 20), -1)
        for i, (cls_id, cnt) in enumerate(sorted(counts.items())):
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            text  = f"{names[cls_id]}: {cnt}"
            cv2.putText(frame, text, (panel_x, y_start + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────
#  MAIN VIDEO LOOP
# ─────────────────────────────────────────
def run_video(source=VIDEO_SRC, weights=WEIGHTS):
    model  = load_model(weights)
    names  = model.names          # {0: 'Tree weeler', 1: 'bicycle', ...}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nSource  : {source}")
    print(f"Size    : {width}×{height}  |  FPS: {fps_src:.1f}  |  Frames: {total}")
    print("\nControls:")
    print("  Q or ESC → quit")
    print("  SPACE    → pause / resume")
    print("  S        → save current frame as screenshot\n")

    # Video writer
    writer = None
    if SAVE_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_PATH, fourcc, fps_src, (width, height))
        print(f"Saving output to: {OUT_PATH}")

    paused     = False
    frame_num  = 0
    fps_display = 0.0
    t_prev     = time.time()

    while True:
        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # Q or ESC
            break
        if key == ord(" "):                # SPACE
            paused = not paused
            print("Paused" if paused else "Resumed")
        if key == ord("s"):                # S — screenshot
            ss_path = f"screenshot_{frame_num:05d}.jpg"
            cv2.imwrite(ss_path, frame)
            print(f"Screenshot saved: {ss_path}")

        if paused:
            cv2.imshow("Vehicle Detection", frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_num += 1

        # ── Inference ──
        results = model.predict(
            source  = frame,
            imgsz   = IMGSZ,
            conf    = CONF,
            iou     = IOU,
            device  = 0 if torch.cuda.is_available() else "cpu",
            verbose = False,
        )

        # ── Count detections per class ──
        counts = {}
        for box in results[0].boxes:
            c = int(box.cls[0])
            counts[c] = counts.get(c, 0) + 1

        # ── Draw ──
        frame = draw_boxes(frame, results[0], names)

        # FPS calculation (rolling)
        t_now        = time.time()
        fps_display  = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev       = t_now

        frame = draw_overlay(frame, fps_display, counts, names)

        # ── Progress bar (bottom of frame) ──
        if total > 0:
            progress = int(width * frame_num / total)
            cv2.rectangle(frame, (0, height - 5), (progress, height), (0, 200, 0), -1)

        # ── Show + save ──
        cv2.imshow("Vehicle Detection  [Q=quit  SPACE=pause  S=screenshot]", frame)
        if writer:
            writer.write(frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_num} frames.")
    if SAVE_OUT:
        print(f"Output saved: {OUT_PATH}")


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: run on a video file ──
    run_video(
        source  = VIDEO_SRC,
        weights = WEIGHTS,
    )

    # ── Option B: run on webcam ──
    # run_video(source=0, weights=WEIGHTS)

    # ── Option C: run on a folder of images ──
    # run_video(source=r"C:\Final project\test\images", weights=WEIGHTS)
