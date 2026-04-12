"""
calibrate_camera.py — Compute camera matrix & distortion coefficients.

Place this in your project root (next to capture_calib_images.py).

Usage:
    python calibrate_camera.py

Reads all images from Calib_Images/, detects checkerboard corners,
and saves the calibration result to camera_calibration.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ── Checkerboard geometry ─────────────────────────────────────────────────────
# 11×8 squares  →  10×7 inner corners
COLS = 10       # inner corners horizontal
ROWS = 7        # inner corners vertical
SQUARE_SIZE_MM = 25.0

IMAGES_DIR  = Path(__file__).resolve().parent / "Calib_Images"
OUTPUT_FILE = Path(__file__).resolve().parent / "camera_calibration.npz"


def main() -> None:
    images = sorted(IMAGES_DIR.glob("calib_*.png"))
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    print(f"\n  Calibration")
    print(f"  Board: {COLS+1}x{ROWS+1} squares → {COLS}x{ROWS} inner corners")
    print(f"  Square size: {SQUARE_SIZE_MM} mm")
    print(f"  Images found: {len(images)}\n")

    # 3D object points for one board (z=0 plane)
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_SIZE_MM

    obj_points = []   # 3D points per image
    img_points = []   # 2D detected corners per image
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, (COLS, ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        status = "✅" if found else "—"
        print(f"  [{i+1:3d}/{len(images)}] {path.name}  {status}", end="\r")

        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_sub)

    print()  # clear \r line
    print(f"\n  Board detected in {len(obj_points)} / {len(images)} images")

    if len(obj_points) < 10:
        print("  ⚠  Too few valid detections (need at least 10). Check board/images.")
        sys.exit(1)

    # ── Calibrate ─────────────────────────────────────────────────────────────
    print("  Running initial calibration …")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None,
    )
    print(f"  Initial RMS: {ret:.4f} px  ({len(obj_points)} images)")

    # ── Iterative outlier rejection ───────────────────────────────────────────
    # Compute per-image reprojection error, drop worst 15%, recalibrate.
    # Repeat until RMS stabilises or too few images remain.
    DROP_FRACTION = 0.15
    MIN_IMAGES    = 30

    for iteration in range(10):
        errors = []
        for j in range(len(obj_points)):
            projected, _ = cv2.projectPoints(
                obj_points[j], rvecs[j], tvecs[j], camera_matrix, dist_coeffs,
            )
            err = cv2.norm(img_points[j], projected, cv2.NORM_L2) / len(projected)
            errors.append(err)

        errors = np.array(errors)
        threshold = np.percentile(errors, (1 - DROP_FRACTION) * 100)
        keep = errors <= threshold

        n_drop = int(np.sum(~keep))
        n_keep = int(np.sum(keep))

        if n_keep < MIN_IMAGES:
            print(f"  Round {iteration+1}: would drop to {n_keep} images, stopping.")
            break

        obj_points = [o for o, k in zip(obj_points, keep) if k]
        img_points = [p for p, k in zip(img_points, keep) if k]

        prev_rms = ret
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None,
        )
        print(f"  Round {iteration+1}: dropped {n_drop} worst, kept {n_keep} → RMS {ret:.4f} px")

        if prev_rms - ret < 0.005:
            print(f"  Converged (improvement < 0.005).")
            break

    print(f"\n  RMS reprojection error: {ret:.4f} px")
    print(f"  (< 0.5 = excellent, < 1.0 = good, > 1.0 = check images)\n")

    print("  Camera matrix:")
    print(f"    fx = {camera_matrix[0, 0]:.2f}")
    print(f"    fy = {camera_matrix[1, 1]:.2f}")
    print(f"    cx = {camera_matrix[0, 2]:.2f}")
    print(f"    cy = {camera_matrix[1, 2]:.2f}")

    print(f"\n  Distortion coefficients:")
    print(f"    {dist_coeffs.ravel()}")

    # ── Optimal new matrix (crops black edges after undistort) ─────────────
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, alpha=1, newImgSize=img_size,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        str(OUTPUT_FILE),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        new_camera_matrix=new_camera_matrix,
        roi=np.array(roi),
        img_size=np.array(img_size),
        rms_error=np.array(ret),
    )

    print(f"\n  ✅  Saved to {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()