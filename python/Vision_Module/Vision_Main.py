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

# =============================================================================
# IMAGE SOURCE TOGGLE
# Set USE_CAMERA = True  → capture live from RealSense
# Set USE_CAMERA = False → load a PNG from the Images folder
# =============================================================================
USE_CAMERA = False

# Only used when USE_CAMERA = False.
# Path is relative to this file's directory (Vision_Module/Images/).
TEST_IMAGE_NAME = "Experiment_2.png"
# =============================================================================


# Path to the Images folder (sits next to this file)
IMAGES_DIR = Path(__file__).resolve().parent / "Images"


def load_test_image(filename: str) -> "np.ndarray":
    img_path = IMAGES_DIR / filename
    if not img_path.exists():
        available = [p.name for p in sorted(IMAGES_DIR.glob("*.png"))]
        raise FileNotFoundError(
            f"Test image not found: {img_path}\n"
            f"Available images in {IMAGES_DIR}:\n  " + "\n  ".join(available or ["(none)"])
        )
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cv2.imread failed to load: {img_path}")
    print(f"Loaded test image: {img_path}  ({img.shape[1]}x{img.shape[0]})")
    return img


def main() -> None:
    # ------------------------------------------------------------------
    # Acquire raw image
    # ------------------------------------------------------------------
    if USE_CAMERA:
        print("Capturing from RealSense camera...")
        img_raw = capture_color_frame(
            width=REALSENSE_WIDTH,
            height=REALSENSE_HEIGHT,
            fps=REALSENSE_FPS,
            warmup_frames=REALSENSE_WARMUP_FRAMES,
        )
    else:
        print(f"Using test image: {TEST_IMAGE_NAME}")
        img_raw = load_test_image(TEST_IMAGE_NAME)

    img_vis = img_raw.copy()
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Board homography
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # AprilTag detection
    # ------------------------------------------------------------------
    detector = create_detector(
        families="tag25h9",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    detections = detect_tags(detector, gray)

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

    del detector  # safe to release after compute_tag_targets_and_annotate is done

    # ------------------------------------------------------------------
    # Robot entries + part assignment
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Save snapshots
    # ------------------------------------------------------------------
    state = entries_to_state(final_entries)
    save_json_snapshot(POSITIONS_PATH, state, pretty=True)
    print(f"Wrote JSON snapshot to: {POSITIONS_PATH}")
    save_llm_snapshot(LLM_INPUT_PATH, state, pretty=True)
    print(f"Wrote LLM input snapshot to: {LLM_INPUT_PATH}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    source_label = "CAMERA" if USE_CAMERA else f"IMAGE: {TEST_IMAGE_NAME}"
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("Result", f"Result [{source_label}]")
    cv2.imshow("Result", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()