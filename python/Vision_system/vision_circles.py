# vision_circles.py
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def _make_color_mask_bgr(
    bgr: np.ndarray,
    ref_bgr: Tuple[int, int, int],
    tol_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """
    Per-channel tolerance box in BGR space:
      |B-Bref|<=tolB, |G-Gref|<=tolG, |R-Rref|<=tolR
    """
    ref = np.array(ref_bgr, dtype=np.int16)
    tol = np.array(tol_bgr, dtype=np.int16)

    img = bgr.astype(np.int16)
    diff = np.abs(img - ref[None, None, :])

    mask = (
        (diff[:, :, 0] <= tol[0]) &
        (diff[:, :, 1] <= tol[1]) &
        (diff[:, :, 2] <= tol[2])
    ).astype(np.uint8) * 255

    # Morph cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    return mask


def _to_board_m(cx_px: float, cy_px: float, H_inv: np.ndarray) -> Tuple[float, float]:
    pts_i = np.array([[[cx_px, cy_px]]], dtype=np.float32)
    c_b = cv2.perspectiveTransform(pts_i, H_inv).reshape(2,)
    return float(c_b[0]), float(c_b[1])


def _component_is_circular(component_mask: np.ndarray) -> Tuple[bool, float, float]:
    """
    Returns: (ok, circularity, fill_ratio)

    circularity = 4*pi*A / P^2  (1.0 is perfect)
    fill_ratio  = A / (pi*r^2)  using minEnclosingCircle
    """
    cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, 0.0, 0.0

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 10.0:
        return False, 0.0, 0.0

    peri = float(cv2.arcLength(c, True))
    if peri <= 1e-6:
        return False, 0.0, 0.0

    circularity = (4.0 * np.pi * area) / (peri * peri)

    (_, _), r = cv2.minEnclosingCircle(c)
    if r <= 1e-6:
        return False, circularity, 0.0

    circle_area = np.pi * float(r) * float(r)
    fill_ratio = area / circle_area if circle_area > 1e-9 else 0.0

    return True, circularity, fill_ratio


def detect_color_cluster_parts_on_board(
    bgr: np.ndarray,
    H_inv: np.ndarray,
    # reference colors in RGB (as provided by you)
    ref_rgb: Dict[str, Tuple[int, int, int]],
    # per-channel tolerance in RGB space (converted internally to BGR)
    tol_rgb: Tuple[int, int, int] = (60, 60, 60),
    # cluster size threshold (px)
    min_area_px: int = 1500,
    # circularity test thresholds
    circularity_min: float = 0.75,
    fill_ratio_min: float = 0.65,
) -> List[Dict[str, float]]:
    """
    Detect parts by dense color clusters + shape test (circular).

    Returns list of dicts:
      {
        "color": "Blue"|"Red"|"Green",
        "cx_px","cy_px",
        "area_px",
        "cx_b_m","cy_b_m",
        "circularity","fill_ratio"
      }
    """
    detections: List[Dict[str, float]] = []
    tol_bgr = _rgb_to_bgr(tol_rgb)

    for color_name, rgb in ref_rgb.items():
        ref_bgr = _rgb_to_bgr(rgb)
        mask = _make_color_mask_bgr(bgr, ref_bgr=ref_bgr, tol_bgr=tol_bgr)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label_id in range(1, num):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < int(min_area_px):
                continue

            # isolate component and test circularity
            comp_mask = (labels == label_id).astype(np.uint8) * 255
            ok, circ, fill = _component_is_circular(comp_mask)
            if not ok:
                continue
            if circ < float(circularity_min) or fill < float(fill_ratio_min):
                continue

            cx, cy = centroids[label_id]
            cx_px = float(cx)
            cy_px = float(cy)

            cx_b_m, cy_b_m = _to_board_m(cx_px, cy_px, H_inv)

            detections.append(
                {
                    "color": str(color_name),
                    "cx_px": cx_px,
                    "cy_px": cy_px,
                    "area_px": float(area),
                    "cx_b_m": cx_b_m,
                    "cy_b_m": cy_b_m,
                    "circularity": float(circ),
                    "fill_ratio": float(fill),
                }
            )

    # Stable ordering: by color then left->right then top->bottom
    detections.sort(key=lambda d: (d["color"], d["cx_px"], d["cy_px"]))
    return detections