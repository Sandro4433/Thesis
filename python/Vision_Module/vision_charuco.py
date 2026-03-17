# vision_charuco.py
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from Vision_Module.geometry import id_to_square_map, marker_obj_corners, project, choose_origin_index, choose_axis_endpoints, OriginAxes
from Vision_Module.geometry import Corner, Direction



def detect_board_homography(
    gray: np.ndarray,
    cols: int,
    rows: int,
    square_size_m: float,
    marker_size_m: float,
    ransac_reproj_thresh_px: float = 3.0,
) -> Tuple[Optional[np.ndarray], int, float, float]:
    """
    Returns (H, inliers_count, board_w, board_h)
    """
    id2ij = id_to_square_map(cols, rows)
    board_w = cols * square_size_m
    board_h = rows * square_size_m

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    aruco = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = aruco.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, 0, board_w, board_h

    idsf = ids.flatten().astype(int)

    brd_pts, img_pts = [], []
    for c, mid in zip(corners, idsf):
        if mid not in id2ij:
            continue
        i, j = id2ij[mid]
        brd_pts.append(marker_obj_corners(i, j, square_size_m, marker_size_m))
        img_pts.append(c.reshape(4, 2).astype(np.float32))

    if len(brd_pts) < 2:
        return None, 0, board_w, board_h

    brd_pts = np.vstack(brd_pts)
    img_pts = np.vstack(img_pts)

    H, mask = cv2.findHomography(brd_pts, img_pts, cv2.RANSAC, ransac_reproj_thresh_px)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers, board_w, board_h


def choose_origin_and_axes(
    H: np.ndarray,
    board_w: float,
    board_h: float,
    axis_len: float,
    origin: Corner = "top-right",
    x_dir: Direction = "left",
    y_dir: Direction = "down",
) -> OriginAxes:
    H_inv = np.linalg.inv(H)

    board_corners_b = np.array(
        [
            [0.0, 0.0],
            [board_w, 0.0],
            [0.0, board_h],
            [board_w, board_h],
        ],
        dtype=np.float32,
    )

    board_corners_i = project(H, board_corners_b)

    origin_idx = choose_origin_index(board_corners_i, origin)
    o_i = board_corners_i[origin_idx]
    o_b = board_corners_b[origin_idx]

    ux_unit_b, uy_unit_b, x_end_i, y_end_i = choose_axis_endpoints(
        H=H,
        o_b=o_b,
        o_i=o_i,
        axis_len=axis_len,
        x_dir=x_dir,
        y_dir=y_dir,
    )

    return OriginAxes(
        o_b=o_b.astype(np.float32),
        o_i=o_i.astype(np.float32),
        ux_unit_b=ux_unit_b.astype(np.float32),
        uy_unit_b=uy_unit_b.astype(np.float32),
        H=H.astype(np.float64),
        H_inv=H_inv.astype(np.float64),
        x_end_i=x_end_i.astype(np.float32),
        y_end_i=y_end_i.astype(np.float32),
    )


def draw_origin_and_axes(img: np.ndarray, origin_axes: OriginAxes) -> None:
    o = tuple(np.round(origin_axes.o_i).astype(int))
    px = tuple(np.round(origin_axes.x_end_i).astype(int))
    py = tuple(np.round(origin_axes.y_end_i).astype(int))

    cv2.circle(img, o, 6, (255, 255, 255), -1)
    cv2.arrowedLine(img, o, px, (0, 0, 255), 4, tipLength=0.25)
    cv2.arrowedLine(img, o, py, (0, 255, 0), 4, tipLength=0.25)

    cv2.putText(img, "X", (px[0] + 10, px[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(img, "Y", (py[0] + 30, py[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)