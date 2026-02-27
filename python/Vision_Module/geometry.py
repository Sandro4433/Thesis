# geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import cv2
import numpy as np


def id_to_square_map(cols: int = 11, rows: int = 8) -> Dict[int, Tuple[int, int]]:
    """
    IDs assigned row-major over black squares, with top-left square black (ID 0).
    """
    m: Dict[int, Tuple[int, int]] = {}
    k = 0
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                m[k] = (i, j)
                k += 1
    return m


def marker_obj_corners(i: int, j: int, s: float, m: float) -> np.ndarray:
    """
    Returns 4 corners of the marker square in board coordinates (meters).
    """
    x0, y0 = i * s, j * s
    off = (s - m) * 0.5
    return np.array(
        [
            (x0 + off, y0 + off),
            (x0 + off + m, y0 + off),
            (x0 + off + m, y0 + off + m),
            (x0 + off, y0 + off + m),
        ],
        dtype=np.float32,
    )


def project(H: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts2d, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def wrap_deg_180(a_deg: float) -> float:
    return (a_deg + 180.0) % 360.0 - 180.0


def wrap_deg_90(a_deg: float) -> float:
    # Two-finger symmetry: angle and angle+180 are equivalent -> smallest signed rotation
    return (a_deg + 90.0) % 180.0 - 90.0


def rot2d(dx: float, dy: float, c: float, s: float) -> Tuple[float, float]:
    """
    Rotate (dx, dy) by angle with cos=c, sin=s.
    """
    rx = dx * c - dy * s
    ry = dx * s + dy * c
    return rx, ry


Direction = Literal["left", "right", "up", "down"]
Corner = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


@dataclass(frozen=True)
class OriginAxes:
    """
    Board-origin and axes in board coordinates.
    - o_b: origin point in board coords (meters)
    - o_i: origin point in image coords (pixels)
    - ux_unit_b: unit direction of workspace X in board coords (meters/meter)
    - uy_unit_b: unit direction of workspace Y in board coords (meters/meter)
    - H: board->image homography
    - H_inv: image->board homography
    - x_end_i / y_end_i: endpoints for drawing axes in image
    """
    o_b: np.ndarray
    o_i: np.ndarray
    ux_unit_b: np.ndarray
    uy_unit_b: np.ndarray
    H: np.ndarray
    H_inv: np.ndarray
    x_end_i: np.ndarray
    y_end_i: np.ndarray


def choose_origin_index(board_corners_i: np.ndarray, origin: Corner) -> int:
    # board_corners_i are image points for:
    # [ (0,0), (w,0), (0,h), (w,h) ] in board coords.
    if origin == "top-left":
        # small x, small y -> minimize x+y
        return int(np.argmin(board_corners_i[:, 0] + board_corners_i[:, 1]))
    if origin == "top-right":
        # large x, small y -> maximize (x - y)
        return int(np.argmax(board_corners_i[:, 0] - board_corners_i[:, 1]))
    if origin == "bottom-left":
        # small x, large y -> maximize (y - x)
        return int(np.argmax(board_corners_i[:, 1] - board_corners_i[:, 0]))
    if origin == "bottom-right":
        # large x, large y -> maximize x+y
        return int(np.argmax(board_corners_i[:, 0] + board_corners_i[:, 1]))
    raise ValueError(f"Unknown origin: {origin}")


def choose_axis_endpoints(
    H: np.ndarray,
    o_b: np.ndarray,
    o_i: np.ndarray,
    axis_len: float,
    x_dir: Direction,
    y_dir: Direction,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Choose which of +/-X_b or +/-Y_b becomes workspace X and workspace Y based on desired image directions.
    Returns: ux_unit_b, uy_unit_b, x_end_i, y_end_i
    """
    candidates = [
        ("+x", np.array([axis_len, 0.0], np.float32)),
        ("-x", np.array([-axis_len, 0.0], np.float32)),
        ("+y", np.array([0.0, axis_len], np.float32)),
        ("-y", np.array([0.0, -axis_len], np.float32)),
    ]

    dirs = []
    for name, delta_b in candidates:
        p_b = o_b + delta_b
        p_i = project(H, np.array([p_b], np.float32))[0]
        v_i = p_i - o_i
        dirs.append((name, p_i, v_i, delta_b))

    def score_for_direction(v_i: np.ndarray, desired: Direction) -> float:
        # Higher is better
        dx, dy = float(v_i[0]), float(v_i[1])
        if desired == "left":
            return -dx
        if desired == "right":
            return dx
        if desired == "up":
            return -dy
        if desired == "down":
            return dy
        raise ValueError(desired)

    x_choice = max(dirs, key=lambda t: score_for_direction(t[2], x_dir))
    # avoid using same exact candidate for y
    dirs_for_y = [d for d in dirs if d[0] != x_choice[0]]
    y_choice = max(dirs_for_y, key=lambda t: score_for_direction(t[2], y_dir))

    ux_unit_b = x_choice[3] / axis_len
    uy_unit_b = y_choice[3] / axis_len
    x_end_i = x_choice[1]
    y_end_i = y_choice[1]
    return ux_unit_b, uy_unit_b, x_end_i, y_end_i