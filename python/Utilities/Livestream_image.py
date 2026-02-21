import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector


def id_to_square_map(cols=11, rows=8):
    m = {}
    k = 0
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                m[k] = (i, j)
                k += 1
    return m


def marker_obj_corners(i, j, s, m):
    x0, y0 = i * s, j * s
    off = (s - m) * 0.5
    return np.array(
        [
            (x0 + off,     y0 + off),
            (x0 + off + m, y0 + off),
            (x0 + off + m, y0 + off + m),
            (x0 + off,     y0 + off + m),
        ],
        dtype=np.float32,
    )


def project(H, pts2d):
    pts = np.asarray(pts2d, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def wrap_deg_180(a_deg: float) -> float:
    return (a_deg + 180.0) % 360.0 - 180.0


def wrap_deg_90(a_deg: float) -> float:
    return (a_deg + 90.0) % 180.0 - 90.0


def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    # Board config (cali.io): Rows=8, Columns=11
    cols, rows = 11, 8
    square = 0.020  # meters
    marker = 0.015  # meters
    id2ij = id_to_square_map(cols, rows)

    board_w = cols * square
    board_h = rows * square

    axis_len = square * 3.0      # meters (for choosing axes)
    tag_axis_draw_len = 0.04     # meters (4 cm) for drawing tag axis

    # Kit (AprilTag ID==0) local points (mm) + allowed grip offsets relative to tag orientation (deg)
    kit_points = [
        {"name": "Pos_1", "dx_mm": 65.0,  "dy_mm": 30.0,  "grip_off_deg": 30.0},
        {"name": "Pos_2", "dx_mm": 65.0,  "dy_mm": -30.0, "grip_off_deg": -30.0},
        {"name": "Pos_3", "dx_mm": 120.0, "dy_mm": 0.0,   "grip_off_deg": -80.0},
    ]

    # ----------------------------
    # ArUco (Charuco board) detector
    # ----------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    aruco = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ----------------------------
    # AprilTag detector
    # ----------------------------
    april = Detector(
        families="tag25h9",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )

    # ----------------------------
    # RealSense live stream
    # ----------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    pipeline.start(config)
    try:
        # warmup
        for _ in range(10):
            pipeline.wait_for_frames()

        cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ----------------------------
            # Find board homography each frame
            # ----------------------------
            H = None
            H_inv = None
            o_b = None
            ux_unit_b = None
            uy_unit_b = None
            inliers = 0

            corners, ids, _ = aruco.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                idsf = ids.flatten().astype(int)

                brd_pts, img_pts = [], []
                for c, mid in zip(corners, idsf):
                    if mid not in id2ij:
                        continue
                    i, j = id2ij[mid]
                    brd_pts.append(marker_obj_corners(i, j, square, marker))
                    img_pts.append(c.reshape(4, 2).astype(np.float32))

                if len(brd_pts) >= 2:
                    brd_pts = np.vstack(brd_pts)
                    img_pts = np.vstack(img_pts)
                    H, mask = cv2.findHomography(brd_pts, img_pts, cv2.RANSAC, 3.0)
                    if mask is not None:
                        inliers = int(mask.sum())

            board_ok = (H is not None and inliers >= 20)
            if board_ok:
                H_inv = np.linalg.inv(H)

                # ORIGIN = top-left corner in IMAGE.
                # AXES: X down, Y right (in IMAGE).
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
                origin_idx = int(np.argmin(board_corners_i[:, 0] + board_corners_i[:, 1]))
                o_i = board_corners_i[origin_idx]
                o_b = board_corners_b[origin_idx]

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
                    v = p_i - o_i
                    dirs.append((name, p_i, v, delta_b))

                x_choice = max(dirs, key=lambda t: t[2][1])  # most "down" in image
                dirs_for_y = [d for d in dirs if d[0] != x_choice[0]]
                y_choice = max(dirs_for_y, key=lambda t: t[2][0])  # most "right" in image

                ux_unit_b = x_choice[3] / axis_len
                uy_unit_b = y_choice[3] / axis_len

                # Draw origin + axes
                o = tuple(np.round(o_i).astype(int))
                px = tuple(np.round(x_choice[1]).astype(int))
                py = tuple(np.round(y_choice[1]).astype(int))
                cv2.circle(img, o, 6, (255, 255, 255), -1)
                cv2.arrowedLine(img, o, px, (0, 0, 255), 4, tipLength=0.25)
                cv2.arrowedLine(img, o, py, (0, 255, 0), 4, tipLength=0.25)
                cv2.putText(img, "X", (px[0] + 10, px[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(img, "Y", (py[0] + 10, py[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.putText(img, "Charuco board NOT detected", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # ----------------------------
            # AprilTags + visualization (only if board ok)
            # ----------------------------
            if board_ok:
                detections = april.detect(gray)

                for r in detections:
                    pts_i_int = r.corners.astype(int)
                    for i in range(4):
                        cv2.line(img, tuple(pts_i_int[i]), tuple(pts_i_int[(i + 1) % 4]), (255, 0, 0), 3)

                    label_xy = (pts_i_int[0][0], pts_i_int[0][1] - 10)
                    if r.tag_id == 0:
                        cv2.putText(img, "Kit (ID: 0)", label_xy,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        cv2.putText(img, f"ID: {r.tag_id}", label_xy,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Center image -> board (meters)
                    c_i = np.array([[r.center]], dtype=np.float32)
                    c_b = cv2.perspectiveTransform(c_i, H_inv).reshape(2,)

                    # Tag position in Charuco coords (mm)
                    d_b = c_b - o_b
                    tag_x_mm = float(d_b.dot(ux_unit_b)) * 1000.0
                    tag_y_mm = float(d_b.dot(uy_unit_b)) * 1000.0

                    # Tag corners image -> board (meters)
                    tag_corners_i = r.corners.astype(np.float32).reshape(-1, 1, 2)
                    tag_corners_b = cv2.perspectiveTransform(tag_corners_i, H_inv).reshape(-1, 2)

                    # "Up" direction (flipped): top-mid -> bottom-mid (assumes TL,TR,BR,BL)
                    top_mid_b = 0.5 * (tag_corners_b[0] + tag_corners_b[1])
                    bot_mid_b = 0.5 * (tag_corners_b[3] + tag_corners_b[2])
                    v_b = bot_mid_b - top_mid_b

                    n = float(np.linalg.norm(v_b))
                    if n <= 1e-9:
                        continue
                    v_b_unit = v_b / n

                    # Express tag axis in Charuco (X,Y) basis
                    vx = float(v_b_unit.dot(ux_unit_b))  # along X
                    vy = float(v_b_unit.dot(uy_unit_b))  # along Y

                    # Tag angle relative to Y-axis (deg)
                    theta_y_deg = float(np.degrees(np.arctan2(vx, vy)))
                    theta_y_deg = wrap_deg_180(-theta_y_deg)

                    # Smallest two-finger rotation relative to Y-axis (deg)
                    tag_rot_deg_to_y = wrap_deg_90(theta_y_deg)

                    # Draw tag axis in image
                    end_b = c_b + v_b_unit * tag_axis_draw_len
                    end_i = project(H, np.array([end_b], dtype=np.float32))[0]
                    p0 = tuple(np.round(r.center).astype(int))
                    p1 = tuple(np.round(end_i).astype(int))
                    cv2.arrowedLine(img, p0, p1, (255, 255, 0), 3, tipLength=0.25)

                    # Overlay tag pose info
                    cv2.putText(img, f"X={tag_x_mm:.1f}mm  Y={tag_y_mm:.1f}mm",
                                (label_xy[0], label_xy[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.putText(img, f"rot={tag_rot_deg_to_y:.1f} deg (to Y)",
                                (label_xy[0], label_xy[1] + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    # KIT: draw the 3 sub-positions and their grip rotations
                    if r.tag_id == 0:
                        theta_x_rad = float(np.arctan2(vy, vx))
                        cth = float(np.cos(theta_x_rad))
                        sth = float(np.sin(theta_x_rad))

                        for kp in kit_points:
                            dx_mm = float(kp["dx_mm"])
                            dy_mm = float(kp["dy_mm"])
                            off_deg = float(kp["grip_off_deg"])

                            rx_mm = dx_mm * cth - dy_mm * sth
                            ry_mm = dx_mm * sth + dy_mm * cth

                            kit_x_mm = tag_x_mm + rx_mm
                            kit_y_mm = tag_y_mm + ry_mm

                            grip_theta_y_deg = wrap_deg_180(theta_y_deg + off_deg)
                            grip_rot_deg_to_y = wrap_deg_90(grip_theta_y_deg)

                            # Draw kit point
                            kit_x_m = kit_x_mm / 1000.0
                            kit_y_m = kit_y_mm / 1000.0
                            p_b = o_b + ux_unit_b * kit_x_m + uy_unit_b * kit_y_m
                            p_i = project(H, np.array([p_b], dtype=np.float32))[0]
                            pi = tuple(np.round(p_i).astype(int))

                            cv2.circle(img, pi, 6, (0, 255, 255), -1)
                            cv2.putText(
                                img,
                                f'{kp["name"]} ({kit_x_mm:.0f},{kit_y_mm:.0f})mm',
                                (pi[0] + 8, pi[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 255, 255),
                                2,
                            )
                            cv2.putText(
                                img,
                                f"rot={grip_rot_deg_to_y:.1f} deg (to Y)",
                                (pi[0] + 8, pi[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 255, 255),
                                2,
                            )

            cv2.imshow("Live", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
