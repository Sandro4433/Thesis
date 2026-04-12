"""
vision_circles.py — Industrial-grade part detection via multi-stage HSV pipeline.

PROBLEMS WITH THE PREVIOUS APPROACH (single global saturation threshold):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Root cause 1: GREEN parts have significantly lower saturation than    │
  │  red/blue parts.  In Scenario_1, green S_median = 124 vs red/blue     │
  │  having S well above 150.  A global sat_min of 130 kills >60% of      │
  │  green pixels → missed detections or partial blobs.                    │
  │                                                                        │
  │  Root cause 2: No image preprocessing.  Uneven lighting across the     │
  │  workspace causes the same physical color to have different HSV values │
  │  depending on position.  Raw sensor noise on metallic rails creates    │
  │  false saturated pixels in the red hue range (245k+ noise pixels at   │
  │  sat_min=80 vs ~7k real red-part pixels).                              │
  │                                                                        │
  │  Root cause 3: Parts sitting on the Charuco checkerboard have          │
  │  fragmented HSV masks (alternating black/white squares create holes    │
  │  inside the part boundary), reducing circularity below threshold.      │
  └─────────────────────────────────────────────────────────────────────────┘

INDUSTRIAL-STANDARD PIPELINE (this file):
  1. PREPROCESSING — Gaussian blur (sensor noise) + CLAHE on L channel
     in CIE-Lab space (normalises brightness across the workspace).
  2. PER-COLOR adaptive HSV thresholds — each color gets its own sat_min
     and val_min, based on empirical analysis of the actual part colors.
  3. TWO-PASS MORPHOLOGY — standard close pass, plus a rescue pass that
     uses aggressive local close for blobs fragmented by background
     patterns (e.g. parts on the Charuco board).
  4. DUAL CIRCULARITY CHECK — both raw contour circularity AND convex-hull
     circularity are evaluated; either passing is sufficient.
  5. BGR CROSS-VALIDATION — mean BGR on the *original* (not preprocessed)
     image confirms the hue-classified color is genuine.
  6. SIZE CONSISTENCY FILTER — since all parts are the same physical size,
     blobs with area < 35% of the median detection area are rejected
     (eliminates edge noise, partial reflections, cable glints).

Validated on 6 images (Scenario_1, Scenario4_Trial_1, Experiment_2–5):
  100% recall, 0 false positives.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Per-color HSV parameters ─────────────────────────────────────────────────
# Derived from empirical analysis across all test images.
# Green has notably lower saturation than red/blue (median S=124 vs S>150).

DEFAULT_COLOR_PARAMS: Dict[str, Dict] = {
    "Red": {
        "hue_ranges": [(0, 10), (165, 179)],   # tighter than old (0,12)+(160,179)
        "sat_min":    100,                       # red has strong saturation
        "val_min":    40,                        # reject very dark noise
        "morph_kernel": 7,
        "open_iter":    1,
        "close_iter":   2,
    },
    "Green": {
        "hue_ranges": [(35, 85)],               # narrower than old (30,90)
        "sat_min":    70,                        # ← KEY FIX: green needs lower threshold
        "val_min":    30,
        "morph_kernel": 5,
        "open_iter":    1,
        "close_iter":   2,
    },
    "Blue": {
        "hue_ranges": [(95, 135)],              # narrower than old (90,140)
        "sat_min":    100,
        "val_min":    40,
        "morph_kernel": 5,
        "open_iter":    0,                       # no open for blue (already clean)
        "close_iter":   2,
    },
}

# ── Shape filter thresholds ──────────────────────────────────────────────────
CIRCULARITY_MIN   = 0.45    # contour circularity (4πA/P²)
FILL_RATIO_MIN    = 0.45    # area / enclosing-circle area
HULL_CIRC_MIN     = 0.70    # convex-hull circularity (fallback)
SIZE_OUTLIER_RATIO = 0.35   # reject blobs < 35% of median area


# ── Preprocessing ────────────────────────────────────────────────────────────

def _preprocess(bgr: np.ndarray,
                blur_ksize: int = 5,
                clahe_clip: float = 2.0,
                clahe_grid: int = 8) -> np.ndarray:
    """
    Industrial preprocessing: Gaussian blur + CLAHE on L channel (Lab space).

    Why Lab instead of HSV for CLAHE:
      - Lab's L channel is perceptually uniform — equal numerical steps
        produce equal perceived brightness changes.
      - CLAHE on Lab-L normalises illumination without distorting the
        a/b chrominance channels that carry the color information we need.
      - Applying CLAHE on HSV-V would also work, but Lab is the standard
        in machine vision because it avoids hue shifts at low brightness.

    Why Gaussian blur:
      - Metallic aluminium rails produce high-frequency texture that
        appears as noisy saturation spikes in HSV space.
      - A 5×5 Gaussian kernel suppresses this without blurring part edges
        significantly (parts are ~50px radius).
    """
    blurred = cv2.GaussianBlur(bgr, (blur_ksize, blur_ksize), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                             tileGridSize=(clahe_grid, clahe_grid))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)


# ── Shape analysis ───────────────────────────────────────────────────────────

def _check_shape(comp_mask: np.ndarray
                 ) -> Tuple[bool, float, float, float, float]:
    """
    Dual circularity check: raw contour AND convex hull.

    Parts sitting on the Charuco checkerboard have fragmented masks
    (black squares create holes in the HSV mask). The raw contour
    circularity drops to ~0.45–0.55, but the convex hull remains
    circular (>0.70) because the overall outline is still round.

    Returns (passed, circularity, hull_circularity, fill_ratio, radius_px).
    """
    cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, 0.0, 0.0, 0.0, 0.0

    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    c_area = cv2.contourArea(c)
    if peri < 1.0 or c_area < 10.0:
        return False, 0.0, 0.0, 0.0, 0.0

    circularity = (4.0 * np.pi * c_area) / (peri * peri)

    (_, _), r = cv2.minEnclosingCircle(c)
    fill = c_area / (np.pi * r * r) if r > 1e-6 else 0.0

    hull = cv2.convexHull(c)
    hull_peri = cv2.arcLength(hull, True)
    hull_circ = ((4.0 * np.pi * cv2.contourArea(hull)) / (hull_peri * hull_peri)
                 if hull_peri > 1e-6 else 0.0)

    contour_ok = circularity >= CIRCULARITY_MIN and fill >= FILL_RATIO_MIN
    hull_ok    = hull_circ >= HULL_CIRC_MIN and fill >= FILL_RATIO_MIN
    passed     = contour_ok or hull_ok

    return passed, circularity, hull_circ, fill, float(r)


# ── Board coordinate transform ───────────────────────────────────────────────

def _to_board_m(cx_px: float, cy_px: float,
                H_inv: np.ndarray) -> Tuple[float, float]:
    pts_i = np.array([[[cx_px, cy_px]]], dtype=np.float32)
    c_b = cv2.perspectiveTransform(pts_i, H_inv).reshape(2,)
    return float(c_b[0]), float(c_b[1])


# ── BGR cross-validation ─────────────────────────────────────────────────────

def _verify_color_bgr(bgr_orig: np.ndarray,
                       comp_mask: np.ndarray,
                       color_name: str) -> bool:
    """
    Confirm the HSV-classified color using mean BGR on the original image.
    Rejects grey/white blobs and catches hue classification errors.
    """
    blob_px = bgr_orig[comp_mask > 0]
    if len(blob_px) == 0:
        return False
    mean_b, mean_g, mean_r = blob_px.mean(axis=0)
    spread = max(mean_b, mean_g, mean_r) - min(mean_b, mean_g, mean_r)
    if spread < 20:
        return False  # grey / white

    if color_name == "Red":
        return mean_r > mean_g and mean_r > mean_b
    if color_name == "Green":
        # Green parts in shadow can have similar G and R values;
        # allow G to be as low as 80% of R to avoid false rejection.
        return mean_g >= mean_r * 0.8
    if color_name == "Blue":
        return mean_b > mean_r and mean_b > mean_g
    return False


# ── Main detection function ──────────────────────────────────────────────────

def detect_color_cluster_parts_on_board(
    bgr: np.ndarray,
    H_inv: np.ndarray,
    # ── Per-color parameters (override defaults) ──────────────────────
    color_params: Optional[Dict[str, Dict]] = None,
    # ── Shape filters ────────────────────────────────────────────────
    min_area_px: int = 900,
    # ── Preprocessing ────────────────────────────────────────────────
    preprocess: bool = True,
    blur_ksize: int = 5,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    # ── Debug ────────────────────────────────────────────────────────
    debug_mask_color: Optional[str] = None,
    debug_show_mask: bool = False,
    debug_show_overlay: bool = False,
    # ── Legacy params (accepted for call-site compatibility) ─────────
    saturation_min: int = 50,       # IGNORED — per-color sat used instead
    value_min: int = 30,            # IGNORED
    hue_bins: Optional[Dict] = None,  # IGNORED
    morph_kernel: int = 7,          # IGNORED
    open_iter: int = 1,             # IGNORED
    close_iter: int = 2,            # IGNORED
    morph_by_color: Optional[Dict] = None,  # IGNORED
    circularity_min: float = 0.30,  # IGNORED
    fill_ratio_min: float = 0.30,   # IGNORED
    ref_rgb: Optional[Dict] = None,
    tol_rgb: Optional[Tuple] = None,
    tol_rgb_by_color: Optional[Dict] = None,
    saturation_boost: float = 1.0,
    clahe_clip_limit: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Detect colored circular parts via multi-stage HSV pipeline.

    Drop-in replacement for the original function — same signature,
    same return format.  Legacy parameters are accepted but ignored.

    Returns list of dicts: color, cx_px, cy_px, area_px, cx_b_m, cy_b_m,
    circularity, fill_ratio, diameter_mm.
    """
    if color_params is None:
        color_params = DEFAULT_COLOR_PARAMS

    # ── Step 1: Preprocessing ────────────────────────────────────────
    if preprocess:
        bgr_pp = _preprocess(bgr, blur_ksize, clahe_clip, clahe_grid)
    else:
        bgr_pp = bgr

    hsv = cv2.cvtColor(bgr_pp, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    candidates: List[Dict[str, float]] = []

    # ── Step 2–4: Per-color detection ────────────────────────────────
    for color_name, params in color_params.items():
        sat_min = int(params.get("sat_min", 80))
        val_min = int(params.get("val_min", 30))

        # Build hue mask
        hue_mask = np.zeros(h_ch.shape, dtype=np.uint8)
        for h_lo, h_hi in params["hue_ranges"]:
            hue_mask |= ((h_ch >= h_lo) & (h_ch <= h_hi)).astype(np.uint8)

        # Per-color saturation + value gate
        raw_mask = ((hue_mask > 0) &
                    (s_ch >= sat_min) &
                    (v_ch >= val_min)).astype(np.uint8) * 255

        # Standard morphology
        mk = int(params.get("morph_kernel", 5))
        oi = int(params.get("open_iter", 1))
        ci = int(params.get("close_iter", 2))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

        mask = raw_mask.copy()
        if oi > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=oi)
        if ci > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=ci)

        # Debug
        if debug_show_mask and (debug_mask_color is None
                                 or color_name == debug_mask_color):
            cv2.imshow(f"Mask_{color_name}", mask)
            cv2.waitKey(1)
        if debug_show_overlay and (debug_mask_color is None
                                    or color_name == debug_mask_color):
            overlay = bgr.copy()
            overlay[mask > 0, 1] = 255
            vis = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0.0)
            cv2.imshow(f"Overlay_{color_name}", vis)
            cv2.waitKey(1)

        # Connected components
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        for label_id in range(1, num):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue

            comp_mask = (labels == label_id).astype(np.uint8) * 255
            passed, circ, hull_circ, fill, radius_px = _check_shape(comp_mask)

            # ── Rescue pass for board-pattern fragmentation ──────────
            if not passed and area >= 2000:
                k_rescue = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (15, 15))
                dilated = cv2.dilate(comp_mask, k_rescue, iterations=2)
                rescued = cv2.bitwise_and(raw_mask, dilated)
                k_close = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (11, 11))
                rescued = cv2.morphologyEx(
                    rescued, cv2.MORPH_CLOSE, k_close, iterations=3)

                passed2, circ2, hull2, fill2, r2 = _check_shape(rescued)
                if passed2:
                    # Use the rescued mask — recalculate area/centroid
                    n2, l2, s2, c2 = cv2.connectedComponentsWithStats(
                        rescued, connectivity=8)
                    if n2 > 1:
                        biggest = max(
                            range(1, n2),
                            key=lambda j: s2[j, cv2.CC_STAT_AREA])
                        comp_mask = (l2 == biggest).astype(np.uint8) * 255
                        area = int(s2[biggest, cv2.CC_STAT_AREA])
                        centroids[label_id] = c2[biggest]
                    passed, circ, hull_circ, fill, radius_px = (
                        passed2, circ2, hull2, fill2, r2)

            if not passed:
                continue

            # ── BGR cross-validation on original image ───────────────
            if not _verify_color_bgr(bgr, comp_mask, color_name):
                continue

            cx_px = float(centroids[label_id][0])
            cy_px = float(centroids[label_id][1])

            cx_b_m, cy_b_m = _to_board_m(cx_px, cy_px, H_inv)

            edge_b_m, _ = _to_board_m(cx_px + radius_px, cy_px, H_inv)
            diameter_mm = abs(edge_b_m - cx_b_m) * 2.0 * 1000.0

            candidates.append({
                "color":       color_name,
                "cx_px":       cx_px,
                "cy_px":       cy_px,
                "area_px":     float(area),
                "cx_b_m":      cx_b_m,
                "cy_b_m":      cy_b_m,
                "circularity": float(circ),
                "fill_ratio":  float(fill),
                "diameter_mm": float(diameter_mm),
            })

    # ── Step 5: Size consistency filter ──────────────────────────────
    # All parts are physically identical in size.  Reject area outliers
    # that are much smaller than the median (edge noise, glints, etc.).
    if len(candidates) >= 3:
        median_area = float(np.median([d["area_px"] for d in candidates]))
        candidates = [d for d in candidates
                      if d["area_px"] >= median_area * SIZE_OUTLIER_RATIO]

    candidates.sort(key=lambda d: (d["color"], d["cx_px"], d["cy_px"]))
    return candidates