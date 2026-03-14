# vision_circles.py (additions: debug plot for Green mask + optional overlay)

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def _make_color_mask_bgr(
    bgr: np.ndarray,
    ref_bgr: Tuple[int, int, int],
    tol_bgr: Tuple[int, int, int],
    # NEW: make morph params tunable (useful for Green)
    morph_kernel: int = 7,
    open_iter: int = 1,
    close_iter: int = 2,
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)

    return mask


def _to_board_m(cx_px: float, cy_px: float, H_inv: np.ndarray) -> Tuple[float, float]:
    pts_i = np.array([[[cx_px, cy_px]]], dtype=np.float32)
    c_b = cv2.perspectiveTransform(pts_i, H_inv).reshape(2,)
    return float(c_b[0]), float(c_b[1])


def _component_is_circular(component_mask: np.ndarray) -> Tuple[bool, float, float, float]:
    """Returns (is_circular, circularity, fill_ratio, radius_px)."""
    cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, 0.0, 0.0, 0.0

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 10.0:
        return False, 0.0, 0.0, 0.0

    peri = float(cv2.arcLength(c, True))
    if peri <= 1e-6:
        return False, 0.0, 0.0, 0.0

    circularity = (4.0 * np.pi * area) / (peri * peri)

    (_, _), r = cv2.minEnclosingCircle(c)
    if r <= 1e-6:
        return False, circularity, 0.0, 0.0

    circle_area = np.pi * float(r) * float(r)
    fill_ratio = area / circle_area if circle_area > 1e-9 else 0.0

    return True, circularity, fill_ratio, float(r)


def detect_color_cluster_parts_on_board(
    bgr: np.ndarray,
    H_inv: np.ndarray,
    # reference colors in RGB (as provided by you)
    ref_rgb: Dict[str, Tuple[int, int, int]],
    # fallback per-channel tolerance in RGB space
    tol_rgb: Tuple[int, int, int] = (60, 60, 60),
    # NEW: per-color tolerance override (RGB space)
    tol_rgb_by_color: Optional[Dict[str, Tuple[int, int, int]]] = None,
    # cluster size threshold (px)
    min_area_px: int = 1500,
    # circularity test thresholds
    circularity_min: float = 0.75,
    fill_ratio_min: float = 0.65,
    # NEW: allow per-color morph overrides (optional)
    morph_by_color: Optional[Dict[str, Dict[str, int]]] = None,
    # Morph defaults
    morph_kernel: int = 7,
    open_iter: int = 1,
    close_iter: int = 2,
    # Debug
    debug_mask_color: Optional[str] = None,
    debug_show_mask: bool = False,
    debug_show_overlay: bool = False,
) -> List[Dict[str, float]]:
    """
    Detect parts by dense color clusters + shape test (circular).
    Per-color tuning:
      - tol_rgb_by_color: {"Blue":(60,60,60), "Green":(30,30,30), ...}
      - morph_by_color: {"Blue":{"morph_kernel":5,"open_iter":0,"close_iter":2}, ...}
    """
    detections: List[Dict[str, float]] = []

    for color_name, rgb in ref_rgb.items():
        # pick tolerance for this color
        use_tol_rgb = tol_rgb
        if tol_rgb_by_color and color_name in tol_rgb_by_color:
            use_tol_rgb = tol_rgb_by_color[color_name]

        # pick morph params for this color (optional)
        mk = morph_kernel
        oi = open_iter
        ci = close_iter
        if morph_by_color and color_name in morph_by_color:
            m = morph_by_color[color_name]
            mk = int(m.get("morph_kernel", mk))
            oi = int(m.get("open_iter", oi))
            ci = int(m.get("close_iter", ci))

        tol_bgr = _rgb_to_bgr(use_tol_rgb)
        ref_bgr = _rgb_to_bgr(rgb)

        mask = _make_color_mask_bgr(
            bgr,
            ref_bgr=ref_bgr,
            tol_bgr=tol_bgr,
            morph_kernel=mk,
            open_iter=oi,
            close_iter=ci,
        )

        # Debug show
        if debug_show_mask and (debug_mask_color is None or str(color_name) == str(debug_mask_color)):
            cv2.imshow(f"Mask_{color_name}", mask)
            cv2.waitKey(1)

        if debug_show_overlay and (debug_mask_color is None or str(color_name) == str(debug_mask_color)):
            overlay = bgr.copy()
            overlay[mask > 0, 1] = 255
            vis = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0.0)
            cv2.imshow(f"Overlay_{color_name}", vis)
            cv2.waitKey(1)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label_id in range(1, num):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < int(min_area_px):
                continue

            comp_mask = (labels == label_id).astype(np.uint8) * 255
            ok, circ, fill, radius_px = _component_is_circular(comp_mask)
            if not ok:
                continue
            if circ < float(circularity_min) or fill < float(fill_ratio_min):
                continue

            cx, cy = centroids[label_id]
            cx_px = float(cx)
            cy_px = float(cy)
            cx_b_m, cy_b_m = _to_board_m(cx_px, cy_px, H_inv)

            # Convert pixel radius → physical diameter (mm) via the homography.
            # Project a point on the circle edge into board coords and measure
            # its distance from the centre; this naturally handles perspective.
            edge_b_m, _ = _to_board_m(cx_px + radius_px, cy_px, H_inv)
            diameter_mm = abs(edge_b_m - cx_b_m) * 2.0 * 1000.0

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
                    "diameter_mm": float(diameter_mm),
                }
            )

    detections.sort(key=lambda d: (d["color"], d["cx_px"], d["cy_px"]))
    return detections