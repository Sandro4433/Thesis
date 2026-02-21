# main.py
import cv2

from config import (
    POSITIONS_PATH,
    CHARUCO_ORIGIN_IN_ROBOT_M,
    CAMERA_HOME,
    KIT_POINTS,
    CONTAINER_POINTS,
    BOARD_COLS,
    BOARD_ROWS,
    SQUARE_SIZE_M,
    MARKER_SIZE_M,
    AXIS_LEN_M,
    TAG_AXIS_DRAW_LEN_M,
    Z_ROBOT_M,
    REALSENSE_WIDTH,
    REALSENSE_HEIGHT,
    REALSENSE_FPS,
    REALSENSE_WARMUP_FRAMES,
)
from vision_realsense import capture_color_frame
from vision_charuco import (
    detect_board_homography,
    choose_origin_and_axes,
    draw_origin_and_axes,
)
from vision_apriltag import create_detector, detect_tags
from pipeline import compute_tag_targets_and_annotate, targets_to_robot_entries
from io_jsonl import upsert_jsonl_by_name


def main() -> None:
    # ----------------------------
    # CAPTURE ONE IMAGE (RealSense)
    # ----------------------------
    img = capture_color_frame(
        width=REALSENSE_WIDTH,
        height=REALSENSE_HEIGHT,
        fps=REALSENSE_FPS,
        warmup_frames=REALSENSE_WARMUP_FRAMES,
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # Find board homography
    # ----------------------------
    H, inliers, board_w, board_h = detect_board_homography(
        gray=gray,
        cols=BOARD_COLS,
        rows=BOARD_ROWS,
        square_size_m=SQUARE_SIZE_M,
        marker_size_m=MARKER_SIZE_M,
        ransac_reproj_thresh_px=3.0,
    )

    if H is None or inliers < 20:
        print("ERROR: Charuco board not reliably detected (homography invalid). Nothing will be saved.")
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ORIGIN = top-right corner in IMAGE.
    # AXES: X left, Y down (in IMAGE).
    origin_axes = choose_origin_and_axes(
        H=H,
        board_w=board_w,
        board_h=board_h,
        axis_len=AXIS_LEN_M,
        origin="top-right",
        x_dir="left",
        y_dir="down",
    )
    draw_origin_and_axes(img, origin_axes)

    # ----------------------------
    # AprilTag detection
    # ----------------------------
    detector = create_detector(
        families="tag25h9",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    detections = detect_tags(detector, gray)

    # ----------------------------
    # Compute targets + annotate image
    # ----------------------------
    tag_targets = compute_tag_targets_and_annotate(
        img=img,
        detections=detections,
        H=H,
        origin_axes=origin_axes,
        kit_points=KIT_POINTS,
        container_points=CONTAINER_POINTS,
        tag_axis_draw_len=TAG_AXIS_DRAW_LEN_M,
    )

    # ----------------------------
    # SAVE ALL TARGETS TO JSONL (replace by name)
    # ----------------------------
    new_entries = targets_to_robot_entries(
        tag_targets=tag_targets,
        charuco_origin_in_robot_m=CHARUCO_ORIGIN_IN_ROBOT_M,
        z_robot=Z_ROBOT_M,
        camera_quat=CAMERA_HOME["quat"],
    )

    inserted, overwritten = upsert_jsonl_by_name(POSITIONS_PATH, new_entries)
    print(
        f"Wrote {len(new_entries)} entries to: {POSITIONS_PATH} "
        f"(inserted={inserted}, overwritten={overwritten})"
    )

    # ----------------------------
    # SHOW RESULT
    # ----------------------------
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()