"""
Vehicle Speed Estimator — Optical Flow + Homography
Integrates with your existing YOLOv8 vehicle detection model.

Pipeline:
  1. YOLOv8 detects vehicles → bounding boxes
  2. Lucas-Kanade optical flow tracks feature points inside each box
  3. Homography maps pixel displacement → real-world metres
  4. Speed = displacement (m) / time (s) → km/h
  5. Distance = mapped position of vehicle bottom-centre

CALIBRATION (do this once before running):
  - Place 4 markers on the road whose real positions you know
  - Measure actual distances between them with a tape measure
  - Enter pixel and real-world coords in CALIBRATION_POINTS below
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# ─────────────────────────────────────────
#  PATHS — update these
# ─────────────────────────────────────────
WEIGHTS   = r"C:\Final project\runs\vehicle_detection_v1\weights\best.pt"
VIDEO_SRC = r"C:\Users\Nadhu\Downloads\stock-footage-car-driving-on-the-road-in-leh-ladakh-india-october-th.mp4"
OUT_PATH  = r"C:\Final project\runs\speed_output.mp4"
SAVE_OUT  = True

# ─────────────────────────────────────────
#  CALIBRATION — MUST SET FOR YOUR CAMERA
# ─────────────────────────────────────────
# Step 1: pause your video at the start, pick 4 road points
# Step 2: note their pixel coordinates (x, y) from the frame
# Step 3: measure the real-world distances between those points
# Step 4: fill in both arrays below

# Pixel coordinates (x, y) of 4 road points — from your video frame
# Example: corners of a road lane marking you can measure in real life
SRC_POINTS = np.float32([
    [150, 400],   # point A — bottom-left of your reference area
    [490, 400],   # point B — bottom-right
    [420, 260],   # point C — top-right
    [220, 260],   # point D — top-left
])

# Real-world coordinates (x, y) in METRES corresponding to above points
# Measure these with a tape measure on the actual road
DST_POINTS = np.float32([
    [0,   0],     # point A
    [3.5, 0],     # point B — 3.5 m to the right of A (one lane width)
    [3.5, 15],    # point C — 15 m ahead and 3.5 m right
    [0,   15],    # point D — 15 m ahead
])

# ─────────────────────────────────────────
#  DETECTION & TRACKING CONFIG
# ─────────────────────────────────────────
CONF          = 0.35
IOU           = 0.45
IMGSZ         = 640
SPEED_WINDOW  = 8        # average speed over this many frames (smoothing)
MAX_TRACK_AGE = 30       # drop a track if unseen for this many frames
MIN_FLOW_PTS  = 4        # minimum flow points needed to trust speed reading
DIST_ALERT_M  = 10.0     # draw warning if vehicle is closer than this (metres)

CLASS_COLORS = {
    0: (255, 180,  50),
    1: ( 50, 220, 255),
    2: ( 50,  50, 255),
    3: ( 50, 200,  50),
    4: (255, 100, 200),
    5: (  0, 165, 255),
    6: (180,  50, 255),
    7: (  0,  80, 255),
    8: (255, 220,   0),
}

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize   = (21, 21),
    maxLevel  = 3,
    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)

# Shi-Tomasi corner detection parameters (feature points to track)
FEATURE_PARAMS = dict(
    maxCorners   = 12,    # max points per vehicle box
    qualityLevel = 0.2,
    minDistance  = 8,
    blockSize    = 7,
)


# ─────────────────────────────────────────
#  TRACK STATE
# ─────────────────────────────────────────
class VehicleTrack:
    def __init__(self, track_id, cls_id):
        self.id          = track_id
        self.cls_id      = cls_id
        self.speeds      = []          # rolling speed history (km/h)
        self.distances   = []          # rolling distance history (m)
        self.flow_points = None        # current LK feature points (Nx1x2)
        self.age         = 0           # frames since last detection
        self.prev_box    = None        # last known bounding box

    def update_speed(self, kmh):
        self.speeds.append(kmh)
        if len(self.speeds) > SPEED_WINDOW:
            self.speeds.pop(0)

    def update_distance(self, metres):
        self.distances.append(metres)
        if len(self.distances) > SPEED_WINDOW:
            self.distances.pop(0)

    @property
    def smooth_speed(self):
        return np.mean(self.speeds) if self.speeds else 0.0

    @property
    def smooth_distance(self):
        return np.mean(self.distances) if self.distances else 0.0


# ─────────────────────────────────────────
#  HOMOGRAPHY HELPER
# ─────────────────────────────────────────
def build_homography():
    H, _ = cv2.findHomography(SRC_POINTS, DST_POINTS)
    if H is None:
        raise ValueError("Failed to compute homography matrix. Check calibration points.")
    return H

def pixel_to_world(H, px, py):
    """Map a single pixel point to real-world (x, y) in metres."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0, 0, 0]), float(world[0, 0, 1])

def distance_to_camera(H, px, py):
    """Euclidean distance from camera origin to point in metres."""
    wx, wy = pixel_to_world(H, px, py)
    return float(np.hypot(wx, wy))


# ─────────────────────────────────────────
#  OPTICAL FLOW SPEED
# ─────────────────────────────────────────
def extract_features(gray, box):
    """Find good feature points inside a bounding box."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(gray.shape[1] - 1, x2)
    y2 = min(gray.shape[0] - 1, y2)
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None
    roi = gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, mask=None, **FEATURE_PARAMS)
    if pts is None:
        return None
    pts[:, 0, 0] += x1
    pts[:, 0, 1] += y1
    return pts

def compute_flow_speed(H, prev_gray, curr_gray, pts, dt):
    """
    Track pts from prev_gray to curr_gray using Lucas-Kanade.
    Returns: (new_points, speed_kmh, displacement_pixels)
    """
    if pts is None or len(pts) == 0 or dt < 1e-6:
        return None, 0.0, 0.0

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **LK_PARAMS
    )

    if new_pts is None or status is None:
        return None, 0.0, 0.0

    good_old = pts[status.flatten() == 1]
    good_new = new_pts[status.flatten() == 1]

    if len(good_old) < MIN_FLOW_PTS:
        if len(good_new) > 0:
            try:
                return good_new.reshape(-1, 1, 2), 0.0, 0.0
            except:
                return None, 0.0, 0.0
        return None, 0.0, 0.0

    # Pixel displacement per point
    displacements = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
    # Use median to reject outlier movements
    median_dx = float(np.median(displacements[:, 0]))
    median_dy = float(np.median(displacements[:, 1]))

    # Map reference point + displaced reference point to world coords
    ref_x = float(np.mean(good_old[:, 0]))
    ref_y = float(np.mean(good_old[:, 1]))
    wx0, wy0 = pixel_to_world(H, ref_x, ref_y)
    wx1, wy1 = pixel_to_world(H, ref_x + median_dx, ref_y + median_dy)

    real_disp_m = np.hypot(wx1 - wx0, wy1 - wy0)
    speed_ms    = real_disp_m / dt
    speed_kmh   = speed_ms * 3.6

    px_disp = np.hypot(median_dx, median_dy)
    return good_new.reshape(-1, 1, 2), speed_kmh, px_disp


# ─────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────
def draw_vehicle(frame, box, track, names):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(track.cls_id, (200, 200, 200))
    is_close = track.smooth_distance < DIST_ALERT_M and track.smooth_distance > 0

    border_color = (0, 0, 255) if is_close else color
    thickness    = 3           if is_close else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

    name  = names[track.cls_id]
    speed = track.smooth_speed
    dist  = track.smooth_distance

    lines = [
        f"{name}  #{track.id}",
        f"{speed:.1f} km/h",
        f"{dist:.1f} m",
    ]
    if is_close:
        lines.append("! TOO CLOSE")

    lh = 20
    bw = 130
    bh = lh * len(lines) + 8
    cv2.rectangle(frame, (x1, y1 - bh), (x1 + bw, y1), border_color, -1)
    for i, line in enumerate(lines):
        col = (0, 0, 0)
        if "TOO CLOSE" in line:
            col = (255, 255, 255)
        cv2.putText(frame, line, (x1 + 4, y1 - bh + 4 + (i + 1) * lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

def draw_flow_points(frame, track):
    if track.flow_points is None:
        return
    for pt in track.flow_points:
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

def draw_homography_overlay(frame):
    """Draw the calibration quadrilateral on the frame."""
    pts = SRC_POINTS.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 200, 255), 1)
    for i, (px, py) in enumerate(SRC_POINTS):
        cv2.circle(frame, (int(px), int(py)), 5, (0, 200, 255), -1)
        cv2.putText(frame, chr(65 + i), (int(px) + 6, int(py) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

def draw_hud(frame, tracks, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Vehicles: {len(tracks)}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    speeds = [t.smooth_speed for t in tracks.values() if t.smooth_speed > 0.5]
    if speeds:
        cv2.putText(frame, f"Fastest: {max(speeds):.1f} km/h", (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 1, cv2.LINE_AA)


# ─────────────────────────────────────────
#  MATCH DETECTIONS TO TRACKS  (simple IoU)
# ─────────────────────────────────────────
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0

def match_detections(tracks, detections, iou_thresh=0.35):
    """
    detections: list of (box, cls_id)
    Returns: {det_idx: track_id}, unmatched_det_indices
    """
    matched   = {}
    used_trks = set()
    track_ids = list(tracks.keys())

    for di, (dbox, _) in enumerate(detections):
        best_iou, best_tid = 0, None
        for tid in track_ids:
            if tid in used_trks:
                continue
            if tracks[tid].prev_box is None:
                continue
            s = iou(dbox, tracks[tid].prev_box)
            if s > best_iou:
                best_iou, best_tid = s, tid
        if best_iou >= iou_thresh:
            matched[di]   = best_tid
            used_trks.add(best_tid)

    unmatched = [di for di in range(len(detections)) if di not in matched]
    return matched, unmatched


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def run():
    print("Loading model...")
    model  = YOLO(WEIGHTS)
    names  = model.names
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {'GPU' if device == 0 else 'CPU'}")

    H_mat = build_homography()
    print("Homography matrix built.")

    cap = cv2.VideoCapture(VIDEO_SRC)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_SRC}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps_src:.1f} fps")

    writer = None
    if SAVE_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_PATH, fourcc, fps_src, (width, height))

    tracks     = {}
    next_id    = 0
    prev_gray  = None
    t_prev     = time.time()
    fps_disp   = 0.0
    show_homo  = True

    print("\nControls:  Q/ESC=quit  SPACE=pause  H=toggle homography overlay  S=screenshot\n")

    paused = False
    frame  = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('h'):
            show_homo = not show_homo
        if key == ord('s'):
            cv2.imwrite("speed_screenshot.jpg", frame)
            print("Screenshot saved.")

        if paused:
            cv2.imshow("Vehicle Speed & Distance", frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        t_now   = time.time()
        dt      = max(t_now - t_prev, 1e-6)
        t_prev  = t_now
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 1. Detect ──────────────────────────────────────────
        results = model.predict(
            source  = frame,
            imgsz   = IMGSZ,
            conf    = CONF,
            iou     = IOU,
            device  = device,
            verbose = False,
        )

        detections = []
        for box in results[0].boxes:
            b      = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            detections.append((b, cls_id))

        # ── 2. Match detections to existing tracks ─────────────
        matched, unmatched = match_detections(tracks, detections)

        # Update matched tracks
        for di, tid in matched.items():
            dbox, cls_id = detections[di]
            tracks[tid].prev_box = dbox
            tracks[tid].age      = 0
            tracks[tid].cls_id   = cls_id

        # Create new tracks for unmatched detections
        for di in unmatched:
            dbox, cls_id = detections[di]
            tracks[next_id] = VehicleTrack(next_id, cls_id)
            tracks[next_id].prev_box = dbox
            next_id += 1

        # Age out tracks not seen recently
        dead = [tid for tid, t in tracks.items() if t.age > MAX_TRACK_AGE]
        for tid in dead:
            del tracks[tid]
        for t in tracks.values():
            t.age += 1

        # ── 3. Optical Flow per track ──────────────────────────
        if prev_gray is not None:
            for tid, track in tracks.items():
                box = track.prev_box
                if box is None:
                    continue

                # If no existing flow points, extract new ones
                if track.flow_points is None or len(track.flow_points) < MIN_FLOW_PTS:
                    track.flow_points = extract_features(prev_gray, box)

                if track.flow_points is not None:
                    new_pts, speed_kmh, _ = compute_flow_speed(
                        H_mat, prev_gray, curr_gray, track.flow_points, dt
                    )
                    track.flow_points = new_pts

                    if speed_kmh > 0.1:
                        track.update_speed(speed_kmh)

                # Distance from camera (vehicle bottom-centre)
                bx1, by1, bx2, by2 = box
                foot_x = (bx1 + bx2) / 2
                foot_y = by2
                dist_m = distance_to_camera(H_mat, foot_x, foot_y)
                if 0 < dist_m < 200:
                    track.update_distance(dist_m)

        # ── 4. Draw ────────────────────────────────────────────
        if show_homo:
            draw_homography_overlay(frame)

        for tid, track in tracks.items():
            if track.prev_box is None:
                continue
            draw_vehicle(frame, track.prev_box, track, names)
            draw_flow_points(frame, track)

        draw_hud(frame, tracks, fps_disp)

        cv2.putText(frame, "H=toggle overlay | SPACE=pause | Q=quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow("Vehicle Speed & Distance", frame)
        if writer:
            writer.write(frame)

        prev_gray = curr_gray

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")
    if SAVE_OUT:
        print(f"Output saved: {OUT_PATH}")


if __name__ == "__main__":
    run()