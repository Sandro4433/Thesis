import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector


def id_to_square_map(cols=11, rows=8):
    # IDs assigned row-major over black squares, with top-left square black (ID 0)
    m = {}
    k = 0
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                m[k] = (i, j)
                k += 1
    return m


def marker_obj_corners(i, j, s, m):
    # Marker corners in board coords (x right, y down), order: TL,TR,BR,BL
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


def main():
    # ----------------------------
    # RealSense RGB setup (AUTO – no manual exposure/gain)
    # ----------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    print("Active COLOR:", color_vsp.width(), "x", color_vsp.height())

    # ----------------------------
    # Board (cali.io): Rows=8, Columns=11
    # ----------------------------
    cols, rows = 11, 8
    square = 0.020
    marker = 0.015
    id2ij = id_to_square_map(cols, rows)

    # ----------------------------
    # ArUco detection
    # ----------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    aruco = cv2.aruco.ArucoDetector(aruco_dict, params)

    # ----------------------------
    # AprilTag detection
    # ----------------------------
    april = Detector(
        families="tag25h9",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )

    # ----------------------------
    # Axes (SWAPPED X/Y)
    # ----------------------------
    axis_len = square * 3.0

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # --- Board markers ---
            corners, ids, _ = aruco.detectMarkers(gray)
            H = None
            inliers = 0

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

                if H is not None and inliers >= 20:
                    o_b = np.array([0.0, 0.0], np.float32)
                    x_b = o_b + np.array([0.0, axis_len], np.float32)   # X down (swapped)
                    y_b = o_b + np.array([axis_len, 0.0], np.float32)   # Y right (swapped)

                    pts_b = np.array([[o_b], [x_b], [y_b]], dtype=np.float32)  # (3,1,2)
                    pts_i = cv2.perspectiveTransform(pts_b, H).reshape(-1, 2)

                    o = tuple(np.round(pts_i[0]).astype(int))
                    px = tuple(np.round(pts_i[1]).astype(int))
                    py = tuple(np.round(pts_i[2]).astype(int))

                    cv2.circle(img, o, 6, (255, 255, 255), -1)
                    cv2.arrowedLine(img, o, px, (0, 0, 255), 4, tipLength=0.25)  # X
                    cv2.arrowedLine(img, o, py, (0, 255, 0), 4, tipLength=0.25)  # Y
                    cv2.putText(img, "X", (px[0] + 10, px[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.putText(img, "Y", (py[0] + 10, py[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # --- AprilTags ---
            for r in april.detect(gray):
                pts = r.corners.astype(int)
                for i in range(4):
                    cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (255, 0, 0), 3)
                cv2.putText(
                    img,
                    f"ID: {r.tag_id}",
                    (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

            cv2.imshow("Live", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
