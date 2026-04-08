"""
vision_circles.py — Part detection via HSV saturation gating + hue classification.

APPROACH:
  The workspace has black containers/kits, grey metal rails, a Charuco board,
  and colored cylindrical parts (red, blue, green).  The parts are the ONLY
  objects in the scene with high color saturation.

  Pipeline:
    1. Convert to HSV.
    2. Gate on saturation (S > threshold) to isolate all "colorful" pixels.
       This kills black surfaces, grey rails, white reflections in one step.
    3. Classify each saturated pixel by hue into red / blue / green bins.
    4. For each color, find connected components and filter by:
       - minimum area (rejects small noise clusters)
       - circularity + fill ratio (rejects non-circular blobs)
    5. Final color label uses mean BGR from the original image for accuracy.

  Why this is better than BGR tolerance boxes:
    - Hue is largely invariant to brightness — dark corners, shadows, and
      vignetting don't cause missed detections.
    - No per-color reference RGB values to hand-tune.
    - No CLAHE or saturation boost needed as pre-processing.
    - Saturation gate is a single threshold that separates "colored" from
      "not colored" — works for any number of colors.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── HSV hue classification bins ──────────────────────────────────────────────
# OpenCV HSV: H = 0..179 (maps 0-360 to 0-179), S = 0..255, V = 0..255.
#
# Red wraps around 0/179, so it gets two ranges.
# These bins are wide with gaps — for 3 very distinct colors, tight
# boundaries aren't needed and would only cause edge-case failures.

DEFAULT_HUE_BINS: Dict[str, List[Tuple[int, int]]] = {
    "Red":   [(0, 12), (160, 179)],
    "Green": [(30, 90)],
    "Blue":  [(90, 140)],
}


def _to_board_m(cx_px: float, cy_px: float, H_inv: np.ndarray) -> Tuple[float, float]:
    pts_i = np.array([[[cx_px, cy_px]]], dtype=np.float32)
    c_b = cv2.perspectiveTransform(pts_i, H_inv).reshape(2,)
    return float(c_b[0]), float(c_b[1])


def _component_is_circular(component_mask: np.ndarray) -> Tuple[bool, float, float, float]:
    """Returns (is_valid, circularity, fill_ratio, radius_px)."""
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
    # ── Saturation gate ──────────────────────────────────────────────
    # Minimum S value (0-255) for a pixel to be considered "colored".
    # Black surfaces: S ≈ 0-30.  Grey metal: S ≈ 0-20.
    # Colored parts: S ≈ 80+.   Safe default: 40-50.
    saturation_min: int = 50,
    # Minimum V value to reject very dark pixels with noisy hue.
    value_min: int = 30,
    # ── Hue bins ─────────────────────────────────────────────────────
    hue_bins: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    # ── Shape filters ────────────────────────────────────────────────
    min_area_px: int = 900,
    circularity_min: float = 0.30,
    fill_ratio_min: float = 0.30,
    # ── Morphology ───────────────────────────────────────────────────
    morph_kernel: int = 7,
    open_iter: int = 1,
    close_iter: int = 2,
    morph_by_color: Optional[Dict[str, Dict[str, int]]] = None,
    # ── Debug ────────────────────────────────────────────────────────
    debug_mask_color: Optional[str] = None,
    debug_show_mask: bool = False,
    debug_show_overlay: bool = False,
    # ── Legacy params (accepted for call-site compatibility) ─────────
    ref_rgb: Optional[Dict] = None,
    tol_rgb: Optional[Tuple] = None,
    tol_rgb_by_color: Optional[Dict] = None,
    saturation_boost: float = 1.0,
    clahe_clip_limit: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Detect colored circular parts via HSV saturation gating + hue bins.

    Returns list of dicts: color, cx_px, cy_px, area_px, cx_b_m, cy_b_m,
    circularity, fill_ratio, diameter_mm.
    """
    if hue_bins is None:
        hue_bins = DEFAULT_HUE_BINS

    # ── Step 1: HSV + saturation/value gate ──────────────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    saturated = (s_ch >= saturation_min) & (v_ch >= value_min)

    detections: List[Dict[str, float]] = []

    # ── Step 2-3: per-color hue mask → components → shape filter ─────
    for color_name, hue_ranges in hue_bins.items():
        # Hue mask
        hue_mask = np.zeros(h_ch.shape, dtype=np.uint8)
        for h_lo, h_hi in hue_ranges:
            hue_mask |= ((h_ch >= h_lo) & (h_ch <= h_hi)).astype(np.uint8)

        # Combine: saturated AND correct hue
        mask = (saturated & (hue_mask > 0)).astype(np.uint8) * 255

        # Morphology
        mk, oi, ci = morph_kernel, open_iter, close_iter
        if morph_by_color and color_name in morph_by_color:
            m = morph_by_color[color_name]
            mk = int(m.get("morph_kernel", mk))
            oi = int(m.get("open_iter", oi))
            ci = int(m.get("close_iter", ci))

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        if oi > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=oi)
        if ci > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=ci)

        # Debug
        if debug_show_mask and (debug_mask_color is None or color_name == debug_mask_color):
            cv2.imshow(f"Mask_{color_name}", mask)
            cv2.waitKey(1)

        if debug_show_overlay and (debug_mask_color is None or color_name == debug_mask_color):
            overlay = bgr.copy()
            overlay[mask > 0, 1] = 255
            vis = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0.0)
            cv2.imshow(f"Overlay_{color_name}", vis)
            cv2.waitKey(1)

        # Connected components
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label_id in range(1, num):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue

            comp_mask = (labels == label_id).astype(np.uint8) * 255
            ok, circ, fill, radius_px = _component_is_circular(comp_mask)
            if not ok:
                continue
            if circ < circularity_min or fill < fill_ratio_min:
                continue

            cx, cy = centroids[label_id]
            cx_px, cy_px = float(cx), float(cy)
            cx_b_m, cy_b_m = _to_board_m(cx_px, cy_px, H_inv)

            edge_b_m, _ = _to_board_m(cx_px + radius_px, cy_px, H_inv)
            diameter_mm = abs(edge_b_m - cx_b_m) * 2.0 * 1000.0

            # ── Step 4: Classify by mean BGR on original image ───────
            blob_pixels = bgr[comp_mask > 0]
            mean_b, mean_g, mean_r = blob_pixels.mean(axis=0)

            # Reject white/grey blobs (all channels roughly equal)
            mean_spread = max(mean_b, mean_g, mean_r) - min(mean_b, mean_g, mean_r)
            if mean_spread < 25:
                continue

            if mean_b >= mean_g and mean_b >= mean_r:
                classified_color = "Blue"
            elif mean_g >= mean_r:
                classified_color = "Green"
            else:
                classified_color = "Red"

            detections.append({
                "color":       classified_color,
                "cx_px":       cx_px,
                "cy_px":       cy_px,
                "area_px":     float(area),
                "cx_b_m":      cx_b_m,
                "cy_b_m":      cy_b_m,
                "circularity": float(circ),
                "fill_ratio":  float(fill),
                "diameter_mm": float(diameter_mm),
            })

    detections.sort(key=lambda d: (d["color"], d["cx_px"], d["cy_px"]))
    return detections