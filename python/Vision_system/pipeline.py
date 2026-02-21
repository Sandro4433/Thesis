# pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from geometry import project, wrap_deg_180, wrap_deg_90, rot2d, OriginAxes


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
    img: np.ndarray,
    detections: List[Any],
    H: np.ndarray,
    origin_axes: OriginAxes,
    kit_points: List[Dict[str, float]],
    container_points: List[Dict[str, float]],
    tag_axis_draw_len: float,
) -> Dict[int, List[Dict[str, float]]]:
    """
    Produces:
      tag_targets[tag_id] = list of {name_suffix, x_mm, y_mm, orientation_deg}
    Also draws detections and target points on img.
    """
    H_inv = origin_axes.H_inv
    o_b = origin_axes.o_b
    ux_unit_b = origin_axes.ux_unit_b
    uy_unit_b = origin_axes.uy_unit_b

    tag_targets: Dict[int, List[Dict[str, float]]] = {}

    for r in detections:
        # Draw tag border
        pts_i_int = r.corners.astype(int)
        for i in range(4):
            cv2.line(img, tuple(pts_i_int[i]), tuple(pts_i_int[(i + 1) % 4]), (255, 0, 0), 3)

        label_xy = (int(pts_i_int[0][0]), int(pts_i_int[0][1] - 10))

        if r.tag_id == 0:
            cv2.putText(img, "Kit (ID: 0)", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif r.tag_id == 1:
            cv2.putText(img, "Container (ID: 1)", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(img, f"ID: {r.tag_id}", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Tag center in board coords
        c_b = _tag_center_to_board_m(r, H_inv)
        tag_x_mm, tag_y_mm = _tag_pose_in_workspace_mm(c_b, o_b, ux_unit_b, uy_unit_b)

        # Tag axis vector in board coords
        v_b_unit, n = _tag_axis_vector_board(r, H_inv)

        cv2.putText(
            img,
            f"X={tag_x_mm:.1f}mm  Y={tag_y_mm:.1f}mm",
            (label_xy[0], label_xy[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        if n <= 1e-9:
            # degenerate; save center only for unknown tags
            tag_targets[r.tag_id] = [
                {"name_suffix": "Pos_0", "x_mm": tag_x_mm, "y_mm": tag_y_mm, "orientation_deg": 0.0}
            ]
            continue

        # Express tag axis in workspace (X,Y) basis
        vx = float(v_b_unit.dot(ux_unit_b))
        vy = float(v_b_unit.dot(uy_unit_b))

        theta_x_deg = float(np.degrees(np.arctan2(vy, vx)))
        theta_x_deg = wrap_deg_180(theta_x_deg)
        tag_rot_deg_to_x = wrap_deg_90(theta_x_deg)

        cv2.putText(
            img,
            f"rot={tag_rot_deg_to_x:.1f} deg (to X)",
            (label_xy[0], label_xy[1] + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        # Draw tag axis in image
        end_b = c_b + v_b_unit * float(tag_axis_draw_len)
        end_i = project(H, np.array([end_b], dtype=np.float32))[0]
        p0 = tuple(np.round(r.center).astype(int))
        p1 = tuple(np.round(end_i).astype(int))
        cv2.arrowedLine(img, p0, p1, (255, 255, 0), 3, tipLength=0.25)

        # Unknown tag: save only center
        if r.tag_id not in (0, 1):
            tag_targets[r.tag_id] = [
                {"name_suffix": "Pos_0", "x_mm": tag_x_mm, "y_mm": tag_y_mm, "orientation_deg": tag_rot_deg_to_x}
            ]
            continue

        # Choose point-set based on tag id
        if r.tag_id == 0:
            object_label = "Kit"
            point_set = kit_points
            draw_color = (0, 255, 255)
        else:
            object_label = "Container"
            point_set = container_points
            draw_color = (0, 200, 255)

        tag_targets[r.tag_id] = []

        # For rotating local offsets: use tag angle relative to workspace X-axis
        theta_x_rad = float(np.arctan2(vy, vx))
        cth = float(np.cos(theta_x_rad))
        sth = float(np.sin(theta_x_rad))

        for kp in point_set:
            dx_mm = float(kp["dx_mm"])
            dy_mm = float(kp["dy_mm"])
            off_deg = float(kp["grip_off_deg"])

            rx_mm, ry_mm = rot2d(dx_mm, dy_mm, c=cth, s=sth)
            obj_x_mm = tag_x_mm + rx_mm
            obj_y_mm = tag_y_mm + ry_mm

            grip_theta_x_deg = wrap_deg_180(theta_x_deg + off_deg)
            grip_rot_deg_to_x = wrap_deg_90(grip_theta_x_deg)

            # Draw point
            obj_x_m = obj_x_mm / 1000.0
            obj_y_m = obj_y_mm / 1000.0
            p_b = o_b + ux_unit_b * obj_x_m + uy_unit_b * obj_y_m
            p_i = project(H, np.array([p_b], dtype=np.float32))[0]
            pi = tuple(np.round(p_i).astype(int))

            cv2.circle(img, pi, 6, draw_color, -1)
            cv2.putText(
                img,
                f'{object_label}_{kp["name"]} ({obj_x_mm:.0f},{obj_y_mm:.0f})mm',
                (pi[0] + 8, pi[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                draw_color,
                2,
            )
            cv2.putText(
                img,
                f"rot={grip_rot_deg_to_x:.1f} deg (to X)",
                (pi[0] + 8, pi[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                draw_color,
                2,
            )

            tag_targets[r.tag_id].append(
                {"name_suffix": str(kp["name"]), "x_mm": obj_x_mm, "y_mm": obj_y_mm, "orientation_deg": grip_rot_deg_to_x}
            )

    return tag_targets


def targets_to_robot_entries(
    tag_targets: Dict[int, List[Dict[str, float]]],
    charuco_origin_in_robot_m: Dict[str, float],
    z_robot: float,
    camera_quat: List[float],
) -> List[Dict[str, Any]]:
    new_entries: List[Dict[str, Any]] = []

    for tag_id, targets in tag_targets.items():
        for t in targets:
            x_charuco_m = float(t["x_mm"]) / 1000.0
            y_charuco_m = float(t["y_mm"]) / 1000.0

            x_robot = float(charuco_origin_in_robot_m["x"] + x_charuco_m)
            y_robot = float(charuco_origin_in_robot_m["y"] + y_charuco_m)

            entry = {
                "name": f"April_Tag_{tag_id}_{t['name_suffix']}",
                "pos": [x_robot, y_robot, float(z_robot)],
                "quat": camera_quat,
                "orientation": float(t["orientation_deg"]),  # degrees
            }
            new_entries.append(entry)

    return new_entries