"""
compare_undistort.py — Capture one frame, show original vs undistorted.

Place this in your project root (next to camera_calibration.npz).

Usage:
    python compare_undistort.py

Controls:
    SPACE  — capture a new frame
    Q/ESC  — quit
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── Settings (match your pipeline) ───────────────────────────────────────────
WIDTH   = 1920
HEIGHT  = 1080
FPS     = 30
WARMUP  = 5

# ROI crop (match Vision_Main.py)
ROI_CROP_LEFT   = 230
ROI_CROP_RIGHT  = 80
ROI_CROP_TOP    = 0
ROI_CROP_BOTTOM = 0

CALIB_FILE = Path(__file__).resolve().parent / "camera_calibration.npz"


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def _get_factory_intrinsics(pipeline) -> tuple:
    """Read the factory-burned intrinsics from the RealSense device.
    Returns (camera_matrix, dist_coeffs) in OpenCV format."""
    profile = pipeline.get_active_profile()
    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intr.fx,  0,        intr.ppx],
        [0,        intr.fy,  intr.ppy],
        [0,        0,        1.0     ],
    ], dtype=np.float64)

    # RealSense provides Brown-Conrady coefficients in order:
    # [k1, k2, p1, p2, k3] — same layout OpenCV expects
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)

    return camera_matrix, dist_coeffs, intr.model


def capture_frame(pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color frame")
    img = np.asanyarray(color_frame.get_data()).copy()

    # ROI crop
    h, w = img.shape[:2]
    img = img[ROI_CROP_TOP : h - ROI_CROP_BOTTOM,
              ROI_CROP_LEFT : w - ROI_CROP_RIGHT].copy()
    return img


def main() -> None:
    # ── Load calibration ──────────────────────────────────────────────────
    if not CALIB_FILE.exists():
        print(f"Calibration file not found: {CALIB_FILE}")
        sys.exit(1)

    data = np.load(str(CALIB_FILE))
    camera_matrix     = data["camera_matrix"]
    dist_coeffs       = data["dist_coeffs"]
    new_camera_matrix = data["new_camera_matrix"]

    print(f"  Loaded calibration from {CALIB_FILE.name}")
    print(f"  RMS error was: {float(data['rms_error']):.4f} px\n")

    # ── Start camera ──────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(cfg)

    try:
        profile = pipeline.get_active_profile()
        color_sensor = profile.get_device().first_color_sensor()
        _try_set(color_sensor, rs.option.enable_auto_exposure,      0.0)
        _try_set(color_sensor, rs.option.enable_auto_white_balance, 0.0)
        _try_set(color_sensor, rs.option.exposure,                  50.0)
        _try_set(color_sensor, rs.option.gain,                      64.0)
        _try_set(color_sensor, rs.option.white_balance,             4500.0)
    except Exception as e:
        print(f"Warning: sensor settings: {e}")

    # ── Read factory intrinsics ───────────────────────────────────────────
    fact_matrix, fact_dist, fact_model = _get_factory_intrinsics(pipeline)
    print(f"  Factory intrinsics (model: {fact_model}):")
    print(f"    fx={fact_matrix[0,0]:.2f}  fy={fact_matrix[1,1]:.2f}  "
          f"cx={fact_matrix[0,2]:.2f}  cy={fact_matrix[1,2]:.2f}")
    print(f"    dist: {fact_dist}\n")

    # Optimal matrix for factory undistort
    img_size = (WIDTH - ROI_CROP_LEFT - ROI_CROP_RIGHT,
                HEIGHT - ROI_CROP_TOP - ROI_CROP_BOTTOM)
    fact_new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        fact_matrix, fact_dist, img_size, alpha=1, newImgSize=img_size,
    )

    for _ in range(WARMUP):
        pipeline.wait_for_frames()

    print("  SPACE = new capture  |  Q / ESC = quit\n")

    try:
        while True:
            img = capture_frame(pipeline)

            # Undistort with checkerboard calibration
            undist_checker = cv2.undistort(
                img, camera_matrix, dist_coeffs, None, new_camera_matrix)

            # Undistort with factory intrinsics
            undist_factory = cv2.undistort(
                img, fact_matrix, fact_dist, None, fact_new_matrix)

            # Labels
            img_label = img.copy()
            chk_label = undist_checker.copy()
            fac_label = undist_factory.copy()

            cv2.putText(img_label, "ORIGINAL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(chk_label, "CHECKERBOARD CALIB", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(fac_label, "FACTORY INTRINSICS", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

            # 3-way side by side
            combined = np.hstack([img_label, chk_label, fac_label])

            # Scale down to fit screen
            ch, cw = combined.shape[:2]
            scale = min(1920 / cw, 1080 / ch, 1.0)
            if scale < 1.0:
                combined = cv2.resize(combined, (int(cw * scale), int(ch * scale)))

            cv2.imshow("Original  |  Checkerboard  |  Factory", combined)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()