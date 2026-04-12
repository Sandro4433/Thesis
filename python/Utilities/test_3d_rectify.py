"""
test_3d_rectify.py — Virtual top-down camera using RealSense depth + color.

Captures aligned color + depth, deprojects every pixel to 3D,
then reprojects from a virtual camera positioned straight down.

Place this in your project root.

Usage:
    python test_3d_rectify.py

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

# ── Settings ─────────────────────────────────────────────────────────────────
WIDTH   = 1920
HEIGHT  = 1080
FPS     = 30
WARMUP  = 10   # more warmup for depth to stabilise

# ROI crop (match Vision_Main.py)
ROI_CROP_LEFT   = 230
ROI_CROP_RIGHT  = 80
ROI_CROP_TOP    = 0
ROI_CROP_BOTTOM = 0


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def create_virtual_topdown(
    color_img: np.ndarray,
    depth_frame: rs.depth_frame,
    color_intr: rs.intrinsics,
    depth_scale: float,
) -> np.ndarray:
    """
    Deproject every pixel to 3D using depth, then reproject from a
    virtual camera positioned directly above the scene center,
    looking straight down.
    """
    h, w = color_img.shape[:2]

    # ── Step 1: build 3D point cloud from depth ──────────────────────────
    # Create pixel coordinate grids
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us = us.astype(np.float32).ravel()
    vs = vs.astype(np.float32).ravel()

    # Get depth values for every pixel (in meters)
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depths_m = depth_image[ROI_CROP_TOP:h + ROI_CROP_TOP,
                           ROI_CROP_LEFT:w + ROI_CROP_LEFT].ravel() * depth_scale

    # Deproject to 3D (vectorised)
    fx = color_intr.fx
    fy = color_intr.fy
    # Adjust principal point for ROI crop
    ppx = color_intr.ppx - ROI_CROP_LEFT
    ppy = color_intr.ppy - ROI_CROP_TOP

    x3d = (us - ppx) * depths_m / fx
    y3d = (vs - ppy) * depths_m / fy
    z3d = depths_m

    # ── Step 2: define virtual top-down camera ───────────────────────────
    # Find the median depth (table distance) — this becomes our virtual
    # camera height. Use median to be robust to outliers.
    valid = depths_m > 0.1  # ignore zero / too-close depth
    if valid.sum() < 1000:
        print("  Warning: too few valid depth pixels")
        return color_img

    median_z = np.median(depths_m[valid])

    # Virtual camera: same position but looking perfectly straight down.
    # We keep the same focal length and image size.
    # The virtual camera center is at the center of the 3D point cloud
    # (median x, median y), at height = median_z.
    center_x = np.median(x3d[valid])
    center_y = np.median(y3d[valid])

    # ── Step 3: reproject from virtual camera ────────────────────────────
    # Virtual camera is at (center_x, center_y, 0) looking along +Z.
    # Every 3D point (x, y, z) gets projected as:
    #   u' = fx * (x - center_x) / z + ppx_virtual
    #   v' = fy * (y - center_y) / z + ppy_virtual
    # where ppx_virtual, ppy_virtual center the output image.

    ppx_v = w / 2.0
    ppy_v = h / 2.0

    # Only reproject pixels with valid depth
    out_img = np.zeros_like(color_img)

    # Vectorised reprojection
    with np.errstate(divide='ignore', invalid='ignore'):
        u_new = fx * (x3d - center_x) / z3d + ppx_v
        v_new = fy * (y3d - center_y) / z3d + ppy_v

    # Round to integer pixel coords
    u_new_i = np.round(u_new).astype(np.int32)
    v_new_i = np.round(v_new).astype(np.int32)

    # Filter valid projections
    mask = valid & (u_new_i >= 0) & (u_new_i < w) & (v_new_i >= 0) & (v_new_i < h)

    # Flatten color image
    colors = color_img.reshape(-1, 3)

    # Z-buffer: for overlapping projections, keep the one closest to camera
    zbuf = np.full((h, w), np.inf, dtype=np.float32)

    u_m = u_new_i[mask]
    v_m = v_new_i[mask]
    z_m = z3d[mask]
    c_m = colors[mask]

    # Sort by depth (far to near) so nearer pixels overwrite farther ones
    order = np.argsort(-z_m)
    u_m = u_m[order]
    v_m = v_m[order]
    z_m = z_m[order]
    c_m = c_m[order]

    out_img[v_m, u_m] = c_m
    zbuf[v_m, u_m] = z_m

    # ── Step 4: fill small holes with dilation ───────────────────────────
    # The reprojection creates small gaps between pixels.
    hole_mask = (out_img.sum(axis=2) == 0).astype(np.uint8)
    # Only fill small holes (1-2 px), not large missing regions
    kernel = np.ones((3, 3), np.uint8)
    filled = cv2.dilate(out_img, kernel, iterations=1)
    out_img = np.where(hole_mask[:, :, None] > 0, filled, out_img)

    return out_img


def main() -> None:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, WIDTH if WIDTH <= 1280 else 1280,
                      HEIGHT if HEIGHT <= 720 else 720, rs.format.z16, FPS)
    profile = pipeline.start(cfg)

    # Enable align (depth → color)
    align = rs.align(rs.stream.color)

    # Depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"  Depth scale: {depth_scale:.6f} m/unit")

    # Color sensor settings
    try:
        color_sensor = profile.get_device().first_color_sensor()
        _try_set(color_sensor, rs.option.enable_auto_exposure,      0.0)
        _try_set(color_sensor, rs.option.enable_auto_white_balance, 0.0)
        _try_set(color_sensor, rs.option.exposure,                  50.0)
        _try_set(color_sensor, rs.option.gain,                      64.0)
        _try_set(color_sensor, rs.option.white_balance,             4500.0)
    except Exception as e:
        print(f"Warning: sensor settings: {e}")

    # Get color intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    color_intr = color_profile.as_video_stream_profile().get_intrinsics()
    print(f"  Color intrinsics: fx={color_intr.fx:.1f} fy={color_intr.fy:.1f} "
          f"ppx={color_intr.ppx:.1f} ppy={color_intr.ppy:.1f}")

    # Warmup (depth needs more frames to stabilise)
    for _ in range(WARMUP):
        pipeline.wait_for_frames()

    print(f"\n  3D rectification test")
    print(f"  SPACE = new capture  |  Q / ESC = quit\n")

    try:
        while True:
            # Capture aligned frames
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data()).copy()

            # ROI crop (color only — depth is accessed by original coords)
            h, w = color_img.shape[:2]
            color_cropped = color_img[ROI_CROP_TOP : h - ROI_CROP_BOTTOM,
                                      ROI_CROP_LEFT : w - ROI_CROP_RIGHT].copy()

            # 3D rectification
            rectified = create_virtual_topdown(
                color_cropped, depth_frame, color_intr, depth_scale)

            # Labels
            orig = color_cropped.copy()
            cv2.putText(orig, "ORIGINAL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(rectified, "3D TOP-DOWN", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            combined = np.hstack([orig, rectified])
            ch, cw = combined.shape[:2]
            scale = min(1920 / cw, 1080 / ch, 1.0)
            if scale < 1.0:
                combined = cv2.resize(combined, (int(cw * scale), int(ch * scale)))

            cv2.imshow("Original  |  3D Top-Down", combined)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()