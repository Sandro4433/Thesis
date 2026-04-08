# vision_circles.py
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
    morph_kernel: int = 7,
    open_iter: int = 1,
    close_iter: int = 2,
    min_channel_spread: int = 0,
) -> np.ndarray:
    """
    Per-channel tolerance box in BGR space:
      |B-Bref|<=tolB, |G-Gref|<=tolG, |R-Rref|<=tolR

    Optional min_channel_spread: requires max(B,G,R) - min(B,G,R) >= this value.
    Use this to reject white/grey pixels that happen to fall inside the
    tolerance box.  A grey-green part (e.g. R=50,G=85,B=50) has spread=35;
    a white reflection (R=200,G=200,B=200) has spread=0.
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

    if min_channel_spread > 0:
        ch_max = img.max(axis=2)
        ch_min = img.min(axis=2)
        spread_ok = ((ch_max - ch_min) >= min_channel_spread).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, spread_ok)

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


def _boost_saturation(bgr: np.ndarray, factor: float) -> np.ndarray:
    """
    Convert BGR → HSV, multiply the S channel by factor (clamped to 255),
    convert back to BGR.  Values > 1.0 make colours more vivid; 1.0 = no change.
    Applied only to the mask-building step so detection is easier on
    desaturated parts, while the final mean BGR classification still uses
    the original image for accurate colour labelling.
    """
    if factor == 1.0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _normalize_lighting(bgr: np.ndarray, clip_limit: float = 2.0,
                         tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to even out
    lighting across the frame.  Works on the L channel in LAB space so hue and
    saturation are preserved while brightness variations (vignetting, uneven
    illumination, shadows at frame edges) are corrected.

    This ensures parts at the bottom/edges of the image — which often appear
    darker due to camera lighting falloff — have similar channel values to
    parts in the center, making them pass the same BGR tolerance box.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def detect_color_cluster_parts_on_board(
    bgr: np.ndarray,
    H_inv: np.ndarray,
    # Reference colors in RGB
    ref_rgb: Dict[str, Tuple[int, int, int]],
    # Fallback per-channel tolerance in RGB space
    tol_rgb: Tuple[int, int, int] = (60, 60, 60),
    # Per-color tolerance override (RGB space)
    tol_rgb_by_color: Optional[Dict[str, Tuple[int, int, int]]] = None,
    # Per-color minimum channel spread override (rejects white/grey pixels).
    # max(R,G,B) - min(R,G,B) must be >= this value for a pixel to pass.

    # Saturation boost applied to the image before building tolerance masks.
    # 1.0 = no change, 2.0 = double saturation.  Helps desaturated/grey-tinted
    # parts pass the tolerance box.  The mean BGR classification at the end
    # always uses the original unmodified image so labels stay accurate.
    saturation_boost: float = 1.0,
    # CLAHE lighting normalization.  Evens out brightness across the frame so
    # parts in darker areas (edges, bottom) pass the same tolerance box as
    # parts in the center.  Applied before saturation boost. 
    # Set to 0.0 to disable, typical value 2.0-3.0.
    clahe_clip_limit: float = 2.0,
    # Cluster size threshold (px2)
    min_area_px: int = 1500,
    # Circularity test thresholds
    circularity_min: float = 0.75,
    fill_ratio_min: float = 0.65,
    # Per-color morph overrides
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
    Detect parts by dense color clusters + circular shape test.

    For each color in ref_rgb, build a per-channel tolerance mask around
    the reference BGR value, clean it up with morphology, then check each
    connected component for circularity and fill ratio.

    The final color label is determined by the mean BGR across all pixels
    inside each blob (on the original image), so desaturated parts are
    classified correctly regardless of which mask found them.

    Tuning per color:
      ref_rgb             -- central RGB reference for each color name
      tol_rgb_by_color    -- per-color tolerance box; falls back to tol_rgb
   
      saturation_boost    -- pre-boost saturation before mask building (e.g. 1.5)
      morph_by_color      -- per-color morph params; falls back to defaults
    """
    # Build the pre-processed image used only for mask building.
    # 1. CLAHE normalizes lighting so edge/bottom parts aren't darker.
    # 2. Saturation boost makes desaturated parts more vivid.
    # Classification always uses the original image so labels stay accurate.
    bgr_for_mask = bgr
    if clahe_clip_limit > 0:
        bgr_for_mask = _normalize_lighting(bgr_for_mask, clip_limit=clahe_clip_limit)
    bgr_for_mask = _boost_saturation(bgr_for_mask, saturation_boost)

    detections: List[Dict[str, float]] = []

    for color_name, rgb in ref_rgb.items():
        use_tol_rgb = tol_rgb
        if tol_rgb_by_color and color_name in tol_rgb_by_color:
            use_tol_rgb = tol_rgb_by_color[color_name]

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

      

        # Build mask on the pre-processed image (CLAHE + saturation boost)
        mask = _make_color_mask_bgr(
            bgr_for_mask,
            ref_bgr=ref_bgr,
            tol_bgr=tol_bgr,
            morph_kernel=mk,
            open_iter=oi,
            close_iter=ci,
            
        )

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

            edge_b_m, _ = _to_board_m(cx_px + radius_px, cy_px, H_inv)
            diameter_mm = abs(edge_b_m - cx_b_m) * 2.0 * 1000.0

            # ── classify by mean BGR over all blob pixels (original image) ────
            # Always use the original unboosted image so the label reflects
            # the actual part colour, not the artificially enhanced one.
            blob_pixels = bgr[comp_mask > 0]        # shape (N, 3), BGR order
            mean_b, mean_g, mean_r = blob_pixels.mean(axis=0)

            # Reject white reflections and near-grey blobs by requiring a
            # minimum spread between the highest and lowest mean channel.
            # White reflections: all channels ~equal → spread near 0.
            # Real parts — even grey-green (50,85,50) — have spread >= 30.
            # This is more discriminating than a brightness threshold because
            # a bright legitimate part and a white reflection can have similar
            # mean brightness, but a white reflection will never have a clearly
            # dominant channel.
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