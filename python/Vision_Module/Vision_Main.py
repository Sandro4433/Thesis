# Vision_Main.py
from pathlib import Path
import sys

# Allow running this file directly (by path) as well as importing it from python/Main.py
PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../python
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2

from Vision_Module.config import (
    CONTAINER_TAG_IDS,
    KIT_TAG_IDS,
    POSITIONS_PATH,
    LLM_INPUT_PATH,
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

from Vision_Module.vision_realsense import capture_color_frame
from Vision_Module.vision_charuco import detect_board_homography, choose_origin_and_axes, draw_origin_and_axes
from Vision_Module.vision_apriltag import create_detector, detect_tags
from Vision_Module.pipeline import compute_tag_targets_and_annotate, targets_to_robot_entries
from Vision_Module.assign_parts import assign_parts_to_slots
from Vision_Module.workspace_state import entries_to_state, save_json_snapshot, save_llm_snapshot

def main() -> None:
    img_raw = capture_color_frame(
        width=REALSENSE_WIDTH,
        height=REALSENSE_HEIGHT,
        fps=REALSENSE_FPS,
        warmup_frames=REALSENSE_WARMUP_FRAMES,
    )
    img_vis = img_raw.copy()
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

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
        cv2.imshow("Result", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    origin_axes = choose_origin_and_axes(
        H=H,
        board_w=board_w,
        board_h=board_h,
        axis_len=AXIS_LEN_M,
        origin="top-right",
        x_dir="left",
        y_dir="down",
    )
    draw_origin_and_axes(img_vis, origin_axes)

    detector = create_detector(
        families="tag25h9",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    detections = detect_tags(detector, gray)
    del detector

    tag_targets = compute_tag_targets_and_annotate(
        img_vis=img_vis,
        img_raw=img_raw,
        detections=detections,
        H=H,
        origin_axes=origin_axes,
        kit_points=KIT_POINTS,
        container_points=CONTAINER_POINTS,
        kit_ids=KIT_TAG_IDS,
        container_ids=CONTAINER_TAG_IDS,
        tag_axis_draw_len=TAG_AXIS_DRAW_LEN_M,
    )

    new_entries = targets_to_robot_entries(
        tag_targets=tag_targets,
        charuco_origin_in_robot_m=CHARUCO_ORIGIN_IN_ROBOT_M,
        z_robot=Z_ROBOT_M,
        camera_quat=CAMERA_HOME["quat"],
        kit_ids=KIT_TAG_IDS,
        container_ids=CONTAINER_TAG_IDS,
    )

    final_entries = assign_parts_to_slots(
        new_entries,
        xy_threshold_m=0.02,  # tune (meters)
    )

    state = entries_to_state(final_entries)
    save_json_snapshot(POSITIONS_PATH, state, pretty=True)
    print(f"Wrote JSON snapshot to: {POSITIONS_PATH}")
    save_llm_snapshot(LLM_INPUT_PATH, state, compact_keys=True, drop_nulls=True, pretty=False)
    print(f"Wrote LLM input snapshot to: {LLM_INPUT_PATH}")

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()