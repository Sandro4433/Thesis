"""
test_perspective_warp.py — Test top-down perspective correction using ChArUco board.

Captures a frame, detects the ChArUco board, computes a homography to
create a perfect top-down view of the workspace, and shows the result.

Place this in your project root.

Usage:
    python test_perspective_warp.py

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

# Board config (match config.py)
BOARD_COLS     = 11
BOARD_ROWS     = 8
SQUARE_SIZE_M  = 0.020
MARKER_SIZE_M  = 0.015


def _try_set(sensor: rs.sensor, option: rs.option, value: float) -> None:
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
    except Exception:
        pass


def _get_factory_intrinsics(pipeline) -> tuple:
    profile = pipeline.get_active_profile()
    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1.0     ],
    ], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    return camera_matrix, dist_coeffs


def id_to_square_map(cols: int, rows: int) -> dict:
    m = {}
    k = 0
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                m[k] = (i, j)
                k += 1
    return m


def marker_obj_corners(i: int, j: int, s: float, m: float) -> np.ndarray:
    x0, y0 = i * s, j * s
    off = (s - m) * 0.5
    return np.array([
        (x0 + off,     y0 + off),
        (x0 + off + m, y0 + off),
        (x0 + off + m, y0 + off + m),
        (x0 + off,     y0 + off + m),
    ], dtype=np.float32)


def detect_board(gray: np.ndarray) -> tuple:
    """Detect ChArUco markers and compute board→image homography."""
    id2ij = id_to_square_map(BOARD_COLS, BOARD_ROWS)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    aruco = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = aruco.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, 0

    idsf = ids.flatten().astype(int)
    brd_pts, img_pts = [], []
    for c, mid in zip(corners, idsf):
        if mid not in id2ij:
            continue
        i, j = id2ij[mid]
        brd_pts.append(marker_obj_corners(i, j, SQUARE_SIZE_M, MARKER_SIZE_M))
        img_pts.append(c.reshape(4, 2).astype(np.float32))

    if len(brd_pts) < 2:
        return None, 0

    brd_pts = np.vstack(brd_pts)
    img_pts = np.vstack(img_pts)

    H, mask = cv2.findHomography(brd_pts, img_pts, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def compute_topdown_warp(H: np.ndarray, img_shape: tuple) -> tuple:
    """
    Compute a warp matrix that maps the camera image to a top-down view.

    Uses the board homography H (board_coords → image_pixels) to find
    where the four board corners land in the image, then computes a
    perspective transform that maps those image points back to an
    axis-aligned rectangle, scaled to preserve resolution.

    Returns (M_warp, output_size).
    """
    h_img, w_img = img_shape[:2]

    board_w = BOARD_COLS * SQUARE_SIZE_M
    board_h = BOARD_ROWS * SQUARE_SIZE_M

    # Board corners in board coordinates (meters)
    board_corners_b = np.array([
        [0.0,     0.0],
        [board_w, 0.0],
        [board_w, board_h],
        [0.0,     board_h],
    ], dtype=np.float32)

    # Where they appear in the image
    board_corners_img = cv2.perspectiveTransform(
        board_corners_b.reshape(-1, 1, 2), H).reshape(-1, 2)

    # Compute pixels-per-meter from the board as seen in the image.
    # Use the longer board edge for best accuracy.
    edge_top    = np.linalg.norm(board_corners_img[1] - board_corners_img[0])
    edge_left   = np.linalg.norm(board_corners_img[3] - board_corners_img[0])
    ppm_x = edge_top / board_w
    ppm_y = edge_left / board_h
    ppm = (ppm_x + ppm_y) / 2.0  # average pixels per meter

    # Destination: board occupies the same pixel area, but axis-aligned.
    # Place the board at the same image center to keep surrounding workspace.
    board_center_img = board_corners_img.mean(axis=0)

    dst_board_w = board_w * ppm
    dst_board_h = board_h * ppm

    dst_corners = np.array([
        [board_center_img[0] - dst_board_w/2, board_center_img[1] - dst_board_h/2],
        [board_center_img[0] + dst_board_w/2, board_center_img[1] - dst_board_h/2],
        [board_center_img[0] + dst_board_w/2, board_center_img[1] + dst_board_h/2],
        [board_center_img[0] - dst_board_w/2, board_center_img[1] + dst_board_h/2],
    ], dtype=np.float32)

    M_warp = cv2.getPerspectiveTransform(board_corners_img, dst_corners)

    return M_warp, (w_img, h_img)


def main() -> None:
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

    # Factory intrinsics for undistortion
    cam_matrix, dist_coeffs = _get_factory_intrinsics(pipeline)

    for _ in range(WARMUP):
        pipeline.wait_for_frames()

    print("\n  Perspective warp test")
    print("  SPACE = new capture  |  Q / ESC = quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data()).copy()

            # 1) Factory undistort
            h, w = img.shape[:2]
            new_mat, _ = cv2.getOptimalNewCameraMatrix(
                cam_matrix, dist_coeffs, (w, h), alpha=0)
            img = cv2.undistort(img, cam_matrix, dist_coeffs, None, new_mat)

            # 2) ROI crop
            h, w = img.shape[:2]
            img = img[ROI_CROP_TOP : h - ROI_CROP_BOTTOM,
                      ROI_CROP_LEFT : w - ROI_CROP_RIGHT].copy()

            # 3) Detect board
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, inliers = detect_board(gray)

            if H is None or inliers < 20:
                print("  Board not detected — showing original only")
                preview = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                cv2.putText(preview, "NO BOARD DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Perspective Warp Test", preview)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            print(f"  Board detected: {inliers} inliers")

            # 4) Compute and apply perspective warp
            M_warp, out_size = compute_topdown_warp(H, img.shape)
            warped = cv2.warpPerspective(img, M_warp, out_size,
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))

            # Labels
            orig = img.copy()
            cv2.putText(orig, "ORIGINAL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(warped, "TOP-DOWN WARP", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            combined = np.hstack([orig, warped])
            ch, cw = combined.shape[:2]
            scale = min(1920 / cw, 1080 / ch, 1.0)
            if scale < 1.0:
                combined = cv2.resize(combined, (int(cw * scale), int(ch * scale)))

            cv2.imshow("Perspective Warp Test", combined)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()