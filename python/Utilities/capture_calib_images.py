"""
capture_calib_images.py — RealSense live preview + calibration image capture.

Place this in your project root (next to gui.py / Main.py).

Usage:
    python capture_calib_images.py

Controls:
    SPACE  — save current frame to Calib_Images/
    Q/ESC  — quit

Hold the checkerboard at different angles, distances, and positions.
Aim for 15–25 images covering the full field of view.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

# ── Settings (match your vision_capture_worker.py) ────────────────────────────
WIDTH   = 1920
HEIGHT  = 1080
FPS     = 30
WARMUP  = 5

# ROI crop (match Vision_Main.py)
ROI_CROP_LEFT   = 230
ROI_CROP_RIGHT  = 80
ROI_CROP_TOP    = 0
ROI_CROP_BOTTOM = 0

OUTPUT_DIR = Path(__file__).resolve().parent / "Calib_Images"


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(cfg)

    # Apply same sensor settings as your capture worker
    try:
        profile = pipeline.get_active_profile()
        color_sensor = profile.get_device().first_color_sensor()
        _try_set(color_sensor, rs.option.enable_auto_exposure,      0.0)
        _try_set(color_sensor, rs.option.enable_auto_white_balance, 0.0)
        _try_set(color_sensor, rs.option.exposure,                  50.0)
        _try_set(color_sensor, rs.option.gain,                      64.0)
        _try_set(color_sensor, rs.option.white_balance,             4500.0)
    except Exception as e:
        print(f"Warning: could not set sensor options: {e}")

    # Warmup
    for _ in range(WARMUP):
        pipeline.wait_for_frames()

    count = len(list(OUTPUT_DIR.glob("calib_*.png")))
    print(f"\n  Calibration image capture")
    print(f"  Saving to: {OUTPUT_DIR}")
    print(f"  Existing images: {count}")
    print(f"\n  SPACE = capture  |  Q / ESC = quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # ROI crop (same as Vision_Main.py)
            h_full, w_full = img.shape[:2]
            x0 = ROI_CROP_LEFT
            x1 = w_full - ROI_CROP_RIGHT
            y0 = ROI_CROP_TOP
            y1 = h_full - ROI_CROP_BOTTOM
            img = img[y0:y1, x0:x1].copy()

            # Downscale for preview window (half size)
            ph, pw = img.shape[:2]
            preview = cv2.resize(img, (pw // 2, ph // 2))

            # Overlay status text
            cv2.putText(
                preview,
                f"Images captured: {count}  |  SPACE=capture  Q=quit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            cv2.imshow("Calibration Capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break

            if key == ord(" "):  # SPACE
                count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"calib_{count:03d}_{ts}.png"
                path = OUTPUT_DIR / fname
                cv2.imwrite(str(path), img)
                print(f"  ✅  Saved {fname}  ({count} total)")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"\n  Done — {count} images in {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()