import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector


def main():
    # ----------------------------
    # RealSense RGB setup
    # ----------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    print("Active COLOR:", color_vsp.width(), "x", color_vsp.height())

    # ----------------------------
    # ChArUco board setup
    # ----------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # You did not specify board size (squaresX, squaresY).
    # Adjust if needed.
    squaresX = 8
    squaresY = 11

    squareLength = 0.020  # 20 mm
    markerLength = 0.015  # 15 mm

    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength,
        markerLength,
        aruco_dict
    )

    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict)

    # ----------------------------
    # AprilTag detector setup
    # ----------------------------
    april_detector = Detector(
        families="tag25h9",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # ============================
            # --- ChArUco detection ---
            # ============================
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)

                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )

                if retval and charuco_ids is not None and len(charuco_ids) > 4:
                    # Bounding box around all detected charuco corners
                    pts = charuco_corners.reshape(-1, 2)
                    x_min, y_min = np.min(pts, axis=0).astype(int)
                    x_max, y_max = np.max(pts, axis=0).astype(int)

                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 3)
                    cv2.putText(image, "ChArUco Board",
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

            # ============================
            # --- AprilTag detection ---
            # ============================
            results = april_detector.detect(gray)

            for r in results:
                pts = r.corners.astype(int)

                # Draw bounding box
                for i in range(4):
                    cv2.line(image,
                             tuple(pts[i]),
                             tuple(pts[(i + 1) % 4]),
                             (255, 0, 0), 3)

                # Tag ID text
                tag_id = r.tag_id
                cv2.putText(image,
                            f"ID: {tag_id}",
                            (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 0, 0), 2)

            # Resize for display only
            cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
            cv2.imshow("Live", image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
