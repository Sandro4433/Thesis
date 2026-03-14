# pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from Vision_Module.geometry import project, wrap_deg_180, wrap_deg_90, rot2d, OriginAxes
from Vision_Module.vision_circles import detect_color_cluster_parts_on_board


def _tag_center_to_board_m(r: Any, H_inv: np.ndarray) -> np.ndarray:
    c_i = np.array([[r.center]], dtype=np.float32)
    return cv2.perspectiveTransform(c_i, H_inv).reshape(2,)


def _tag_corners_to_board_m(r: Any, H_inv: np.ndarray) -> np.ndarray:
    tag_corners_i = r.corners.astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(tag_corners_i, H_inv).reshape(-1, 2)


def _tag_axis_vector_board(r: Any, H_inv: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns (v_b_unit, norm) where v_b_unit points "down" the tag (top-mid -> bottom-mid)
    in board coordinates.
    Assumes corners order TL,TR,BR,BL from pupil_apriltags.
    """
    tag_corners_b = _tag_corners_to_board_m(r, H_inv)

    top_mid_b = 0.5 * (tag_corners_b[0] + tag_corners_b[1])
    bot_mid_b = 0.5 * (tag_corners_b[3] + tag_corners_b[2])
    v_b = bot_mid_b - top_mid_b

    n = float(np.linalg.norm(v_b))
    if n <= 1e-9:
        return np.zeros(2, dtype=np.float32), n
    return (v_b / n).astype(np.float32), n


def _tag_pose_in_workspace_mm(
    c_b: np.ndarray,
    o_b: np.ndarray,
    ux_unit_b: np.ndarray,
    uy_unit_b: np.ndarray,
) -> Tuple[float, float]:
    d_b = c_b - o_b
    tag_x_mm = float(d_b.dot(ux_unit_b)) * 1000.0
    tag_y_mm = float(d_b.dot(uy_unit_b)) * 1000.0
    return tag_x_mm, tag_y_mm


def compute_tag_targets_and_annotate(
    img_vis: np.ndarray,
    img_raw: np.ndarray,
    detections: List[Any],
    H: np.ndarray,
    origin_axes: OriginAxes,
    kit_points: List[Dict[str, float]],
    container_points: List[Dict[str, float]],
    kit_ids: Set[int],
    container_ids: Set[int],
    tag_axis_draw_len: float,
    part_size_classes: List[tuple] = None,
) -> Dict[int, List[Dict[str, float]]]:
    """
    Produces:
      tag_targets[tag_id] = list of {name_suffix, x_mm, y_mm, orientation_deg}
    Draws only on img_vis. Uses img_raw for detection.
    Also detects colored parts via dense color clusters + circular shape test.
    """
    H_inv = origin_axes.H_inv
    o_b = origin_axes.o_b
    ux_unit_b = origin_axes.ux_unit_b
    uy_unit_b = origin_axes.uy_unit_b

    tag_targets: Dict[int, List[Dict[str, float]]] = {}

    # ----------------------------
    # AprilTag loop
    # ----------------------------
    for r in detections:
        tag_id = int(r.tag_id)

        # Draw tag border (VIS only)
        pts_i_int = r.corners.astype(int)
        for i in range(4):
            cv2.line(
                img_vis,
                tuple(pts_i_int[i]),
                tuple(pts_i_int[(i + 1) % 4]),
                (255, 0, 0),
                3,
            )

        label_xy = (int(pts_i_int[0][0]), int(pts_i_int[0][1] - 10))

        # Label by group (VIS only)
        if tag_id in kit_ids:
            cv2.putText(img_vis, f"Kit (ID: {tag_id})", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif tag_id in container_ids:
            cv2.putText(img_vis, f"Container (ID: {tag_id})", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(img_vis, f"ID: {tag_id}", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Tag center in board coords
        c_b = _tag_center_to_board_m(r, H_inv)
        tag_x_mm, tag_y_mm = _tag_pose_in_workspace_mm(c_b, o_b, ux_unit_b, uy_unit_b)

        # Tag axis vector in board coords
        v_b_unit, n = _tag_axis_vector_board(r, H_inv)

        # Degenerate axis
        if n <= 1e-9:
            if tag_id in kit_ids or tag_id in container_ids:
                point_set = kit_points if tag_id in kit_ids else container_points
                tag_targets[tag_id] = []
                for kp in point_set:
                    dx_mm = float(kp["dx_mm"])
                    dy_mm = float(kp["dy_mm"])
                    obj_x_mm = tag_x_mm + dx_mm
                    obj_y_mm = tag_y_mm + dy_mm
                    tag_targets[tag_id].append(
                        {"name_suffix": str(kp["name"]), "x_mm": obj_x_mm, "y_mm": obj_y_mm, "orientation_deg": 0.0}
                    )
            else:
                tag_targets[tag_id] = [{"name_suffix": "Pos_0", "x_mm": tag_x_mm, "y_mm": tag_y_mm, "orientation_deg": 0.0}]
            continue

        # Express tag axis in workspace (X,Y) basis
        vx = float(v_b_unit.dot(ux_unit_b))
        vy = float(v_b_unit.dot(uy_unit_b))

        theta_x_deg = float(np.degrees(np.arctan2(vy, vx)))
        theta_x_deg = wrap_deg_180(theta_x_deg)
        tag_rot_deg_to_x = wrap_deg_90(theta_x_deg)

        # Draw tag axis in image (VIS only)
        end_b = c_b + v_b_unit * float(tag_axis_draw_len)
        end_i = project(H, np.array([end_b], dtype=np.float32))[0]
        p0 = tuple(np.round(r.center).astype(int))
        p1 = tuple(np.round(end_i).astype(int))
        cv2.arrowedLine(img_vis, p0, p1, (255, 255, 0), 3, tipLength=0.25)

        # Unknown tag: save only center
        if tag_id not in kit_ids and tag_id not in container_ids:
            tag_targets[tag_id] = [{"name_suffix": "Pos_0", "x_mm": tag_x_mm, "y_mm": tag_y_mm, "orientation_deg": tag_rot_deg_to_x}]
            continue

        # Choose point-set based on tag group
        if tag_id in kit_ids:
            object_label = "Kit"
            point_set = kit_points
            draw_color = (0, 255, 255)
        else:
            object_label = "Container"
            point_set = container_points
            draw_color = (0, 200, 255)

        tag_targets[tag_id] = []

        # For rotating local offsets: use tag angle relative to workspace X-axis
        theta_x_rad = float(np.arctan2(vy, vx))
        cth = float(np.cos(theta_x_rad))
        sth = float(np.sin(theta_x_rad))

        for kp in point_set:
            dx_mm = float(kp["dx_mm"])
            dy_mm = float(kp["dy_mm"])
            off_deg = float(kp["grip_off_deg"])

            # Rotate local point into workspace and translate
            rx_mm, ry_mm = rot2d(dx_mm, dy_mm, c=cth, s=sth)
            obj_x_mm = tag_x_mm + rx_mm
            obj_y_mm = tag_y_mm + ry_mm

            # Grip angle at this point relative to X-axis (deg), then smallest two-finger rotation
            grip_theta_x_deg = wrap_deg_180(theta_x_deg + off_deg)
            grip_rot_deg_to_x = wrap_deg_90(grip_theta_x_deg)

            # Draw point (VIS only)
            obj_x_m = obj_x_mm / 1000.0
            obj_y_m = obj_y_mm / 1000.0
            p_b = o_b + ux_unit_b * obj_x_m + uy_unit_b * obj_y_m
            p_i = project(H, np.array([p_b], dtype=np.float32))[0]
            pi = tuple(np.round(p_i).astype(int))

            cv2.circle(img_vis, pi, 6, draw_color, -1)
            cv2.putText(
                img_vis,
                f"{object_label}_{kp['name']}",
                (pi[0] + 8, pi[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                draw_color,
                2,
            )

            tag_targets[tag_id].append(
                {"name_suffix": str(kp["name"]), "x_mm": obj_x_mm, "y_mm": obj_y_mm, "orientation_deg": grip_rot_deg_to_x}
            )

    # ----------------------------
    # Color-cluster part detection (RAW only), circularity-validated, then draw on VIS
    # ----------------------------
    ref_rgb = {
        "Blue": (40, 80, 130),
        "Red": (130, 40, 40),
        "Green": (50, 80, 50),
    }

    # Tuning knobs:
    # - tol_rgb: widen if you miss parts; tighten if you get background detections
    # - min_area_px: cluster-size threshold (increase to reject noise)
    # - circularity_min/fill_ratio_min: how "circle-like" the cluster must be
    part_dets = detect_color_cluster_parts_on_board(
        bgr=img_raw,
        H_inv=H_inv,
        ref_rgb=ref_rgb,

        # fallback (used if color not in dict)
        tol_rgb=(45, 45, 45),

        # per-color tolerance overrides (start values)
        tol_rgb_by_color={
            "Blue":  (45, 45, 45),   # widen to recover matte/dark blue
            "Red":   (45, 45, 45),
            "Green": (30, 30, 30),   # tighter to avoid highlights
        },

        # optional per-color morph tweaks (useful if blue gets fragmented)
        morph_by_color={
            "Blue":  {"morph_kernel": 5, "open_iter": 0, "close_iter": 2},
            "Green": {"morph_kernel": 7, "open_iter": 1, "close_iter": 2},
            "Red":   {"morph_kernel": 7, "open_iter": 1, "close_iter": 2},
        },

        min_area_px=1000,
        circularity_min=0.35,
        fill_ratio_min=0.45,

        # debug (optional)
        debug_mask_color="Blue",
        debug_show_mask=False,
        debug_show_overlay=False,
    )

    # Single global counter — parts are numbered in detection order (spatially
    # sorted by x then y position within each colour group).
    # Color is stored as a separate field in the tag_targets entry so it
    # no longer needs to be embedded in the name.
    part_counter: int = 0

    for d in part_dets:
        color = str(d["color"])
        if color not in {"Blue", "Red", "Green"}:
            continue

        part_counter += 1
        name_suffix = f"Part_{part_counter}"

        # Convert board center (meters) -> workspace mm using your axis basis
        c_b = np.array([d["cx_b_m"], d["cy_b_m"]], dtype=np.float32)
        delta_b = c_b - o_b
        x_mm = float(delta_b.dot(ux_unit_b)) * 1000.0
        y_mm = float(delta_b.dot(uy_unit_b)) * 1000.0

        # Visualization (VIS only)
        diameter_mm = float(d.get("diameter_mm", 0.0))
        size_label = "unknown"
        if part_size_classes:
            for label, lo, hi in part_size_classes:
                if lo <= diameter_mm < hi:
                    size_label = label
                    break

        center = (int(round(d["cx_px"])), int(round(d["cy_px"])))
        cv2.circle(img_vis, center, 10, (255, 255, 255), 2)
        cv2.putText(
            img_vis,
            name_suffix,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img_vis,
            f"{size_label}  ({diameter_mm:.1f}mm)",
            (center[0] + 10, center[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        tag_targets.setdefault(-1000, []).append(
            {"name_suffix": name_suffix, "x_mm": x_mm, "y_mm": y_mm,
             "orientation_deg": 0.0, "color": color,
             "diameter_mm": float(d.get("diameter_mm", 0.0))}
        )

    return tag_targets


def targets_to_robot_entries(
    tag_targets: Dict[int, List[Dict[str, float]]],
    charuco_origin_in_robot_m: Dict[str, float],
    z_robot: float,
    camera_quat: List[float],
    kit_ids: Set[int],
    container_ids: Set[int],
    part_size_classes: List[tuple] = None,
) -> List[Dict[str, Any]]:
    """
    Outputs objects WITHOUT groupname:
      - Slot: name="Kit_<tagid>_Pos_1" or "Container_<tagid>_Pos_6"
      - Part: name="Part_<k>"  (color stored separately in the Color field)

    Schema (human-readable, pretty on disk):
      Slot:
        { name, pos, quat, orientation, Role, child_part }
      Part:
        { name, pos, quat, orientation, Color, Size, diameter_mm, Role }

    part_size_classes: ordered list of (label, min_mm, max_mm) tuples.
      The first range that contains the measured diameter wins.
      Parts with no diameter info or no matching range receive Size=None.
    """
    def _classify_size(diameter_mm: float) -> Optional[str]:
        if not part_size_classes or diameter_mm <= 0.0:
            return None
        for label, lo, hi in part_size_classes:
            if lo <= diameter_mm < hi:
                return label
        return None

    new_entries: List[Dict[str, Any]] = []

    for tag_id, targets in tag_targets.items():
        tag_id_int = int(tag_id)

        for t in targets:
            x_charuco_m = float(t["x_mm"]) / 1000.0
            y_charuco_m = float(t["y_mm"]) / 1000.0

            x_robot = float(charuco_origin_in_robot_m["x"] + x_charuco_m)
            y_robot = float(charuco_origin_in_robot_m["y"] + y_charuco_m)

            name_suffix = str(t["name_suffix"])
            is_part = name_suffix.startswith("Part_")

            if is_part:
                part_name = name_suffix
                diameter_mm = float(t.get("diameter_mm", 0.0))
                # Color is stored explicitly in the tag_targets entry — it is no
                # longer embedded in the part name.
                color = str(t.get("color") or "Unknown")
                entry = {
                    "name": part_name,
                    "pos": [x_robot, y_robot, float(z_robot)],
                    "quat": camera_quat,
                    "orientation": None,
                    "Color": color,
                    "Size": _classify_size(diameter_mm),
                    "diameter_mm": round(diameter_mm, 1),
                    "Role": None,
                }
                new_entries.append(entry)
                continue

            # Slot
            if tag_id_int in kit_ids:
                prefix = "Kit"
            elif tag_id_int in container_ids:
                prefix = "Container"
            else:
                prefix = "Unknown"

            slot_name = f"{prefix}_{tag_id_int}_{name_suffix}"

            entry = {
                "name": slot_name,
                "pos": [x_robot, y_robot, float(z_robot)],
                "quat": camera_quat,
                "orientation": float(t["orientation_deg"]),
                "Role": None,
                "child_part": None,                   # will be set if a part is assigned
            }
            new_entries.append(entry)

    return new_entries