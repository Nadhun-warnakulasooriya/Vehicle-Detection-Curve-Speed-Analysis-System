"""
Calibration Helper — pick your 4 road points interactively
Run this FIRST, before speed_estimator.py

Usage:
  python calibrate.py

Instructions:
  1. The first frame of your video opens
  2. Click 4 points on the road in this order:
        A — bottom-left of your reference rectangle
        B — bottom-right
        C — top-right
        D — top-left
  3. Enter the real-world distances when prompted
  4. Copy the printed SRC_POINTS and DST_POINTS into speed_estimator.py
"""

import cv2
import numpy as np

VIDEO_SRC = r"C:\Users\Nadhu\Downloads\stock-footage-car-driving-on-the-road-in-leh-ladakh-india-october-th.mp4"

clicks = []
frame_orig = None

def on_click(event, x, y, flags, param):
    global frame_orig, clicks
    if frame_orig is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))
        label = chr(65 + len(clicks) - 1)
        cv2.circle(frame_orig, (x, y), 6, (0, 200, 255), -1)
        cv2.putText(frame_orig, label, (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if len(clicks) > 1:
            cv2.line(frame_orig, clicks[-2], clicks[-1], (0, 200, 255), 1)
        if len(clicks) == 4:
            cv2.line(frame_orig, clicks[-1], clicks[0], (0, 200, 255), 1)
        cv2.imshow("Calibration", frame_orig)
        print(f"  Point {label}: pixel ({x}, {y})")


def main():
    global frame_orig

    cap = cv2.VideoCapture(VIDEO_SRC)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Cannot open {VIDEO_SRC}")
        return

    frame_orig = frame.copy()

    print("=" * 52)
    print("  Calibration Helper")
    print("=" * 52)
    print("Click 4 road points in order: A (bottom-left)")
    print("  → B (bottom-right) → C (top-right) → D (top-left)")
    print("Press Q when done.\n")

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)
    cv2.imshow("Calibration", frame_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicks) < 4:
        print("Need exactly 4 points. Please re-run.")
        return

    print("\nNow enter the REAL-WORLD distances (in metres).")
    print("Measure these on the actual road with a tape measure.\n")

    real_pts = []
    coords = [
        ("A", "0.0", "0.0"),
        ("B", "width of your road reference (e.g. 3.5 for one lane)", "0.0"),
        ("C", "same width as B", "length of your road reference (e.g. 15.0)"),
        ("D", "0.0", "same length as C"),
    ]

    for i, (label, x_hint, y_hint) in enumerate(coords):
        print(f"Point {label} — pixel {clicks[i]}")
        rx = float(input(f"  Real X (metres) [{x_hint}]: ") or x_hint.split()[0])
        ry = float(input(f"  Real Y (metres) [{y_hint}]: ") or y_hint.split()[0])
        real_pts.append((rx, ry))

    print("\n" + "=" * 52)
    print("  Copy these into speed_estimator.py")
    print("=" * 52)
    print("\nSRC_POINTS = np.float32([")
    for p in clicks:
        print(f"    [{p[0]}, {p[1]}],")
    print("])\n")
    print("DST_POINTS = np.float32([")
    for p in real_pts:
        print(f"    [{p[0]}, {p[1]}],")
    print("])")

    H, _ = cv2.findHomography(
        np.float32(clicks), np.float32(real_pts)
    )
    print(f"\nHomography matrix computed successfully.")
    print("Calibration complete!")


if __name__ == "__main__":
    main()
