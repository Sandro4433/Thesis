# Vision_Main.py
from pathlib import Path
import json
import sys

# Allow running this file directly (by path) as well as importing it from python/Main.py
PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../python
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np

from Vision_Module.config import (
    CONTAINER_TAG_IDS,
    KIT_TAG_IDS,
    CONFIGURATION_PATH,
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

from Vision_Module.vision_charuco import detect_board_homography, choose_origin_and_axes, draw_origin_and_axes
from Vision_Module.pipeline import compute_tag_targets_and_annotate, targets_to_robot_entries, annotate_parts
from Vision_Module.assign_parts import assign_parts_to_slots
from Vision_Module.workspace_state import entries_to_state, save_json_snapshot, save_llm_snapshot

# ── Safety guard ──────────────────────────────────────────────────────────────
# pyrealsense2 and libapriltag corrupt each other's heap when loaded in the
# same process.  Capture is deliberately done in a subprocess worker so that
# pyrealsense2 never enters this process.  If something in the import chain
# above has pulled it in anyway, fail loudly here rather than with a silent
# malloc abort later.
if "pyrealsense2" in sys.modules:
    _offenders = [
        name for name, mod in sys.modules.items()
        if name != "pyrealsense2"
        and getattr(mod, "__file__", None)
        and "pyrealsense2" in str(getattr(mod, "__file__", ""))
    ]
    raise ImportError(
        "pyrealsense2 was imported into the main Vision process — this will "
        "cause a malloc heap-corruption crash when libapriltag is loaded.\n"
        "Find which module imported it and guard it behind  "
        "  if not USE_CAMERA  or move it into vision_capture_worker.py.\n"
        f"Loaded sys.modules keys containing 'realsense': "
        f"{[k for k in sys.modules if 'realsense' in k.lower()]}"
    )
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# IMAGE SOURCE TOGGLE
# Set USE_CAMERA = True  → capture live from RealSense (via subprocess worker)
# Set USE_CAMERA = False → load a PNG from the Images folder
# =============================================================================
USE_CAMERA = False

# Only used when USE_CAMERA = False.
# Path is relative to this file's directory (Vision_Module/Images/).
TEST_IMAGE_NAME = "Scenario_1.png"
# =============================================================================


# Path to the Images folder (sits next to this file)
IMAGES_DIR = Path(__file__).resolve().parent / "Images"

# Worker script lives next to this file
_WORKER = Path(__file__).resolve().parent / "vision_capture_worker.py"


def _capture_via_subprocess() -> "np.ndarray":
    """
    Spawn a clean Python process that imports ONLY pyrealsense2 + cv2 to
    capture one frame and write it to a temp file.  This avoids the
    heap-corruption crash caused by pyrealsense2 + libapriltag sharing a
    process.

    The worker also saves factory intrinsics alongside the image.
    If available, the frame is undistorted before returning.
    """
    import subprocess
    import tempfile
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    tmp_path = tmp.name

    cmd = [
        sys.executable,
        str(_WORKER),
        tmp_path,
        str(REALSENSE_WIDTH),
        str(REALSENSE_HEIGHT),
        str(REALSENSE_FPS),
        str(REALSENSE_WARMUP_FRAMES),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Camera worker failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )

    img = cv2.imread(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if img is None:
        raise RuntimeError(f"cv2.imread could not load temp frame: {tmp_path}")

    print(f"Captured frame via subprocess: {img.shape[1]}x{img.shape[0]}")

    # ── Undistort using factory intrinsics (if worker saved them) ─────────
    intr_path = tmp_path.rsplit(".", 1)[0] + "_intrinsics.npz"
    intr_file = Path(intr_path)
    if intr_file.exists():
        try:
            data = np.load(str(intr_file))
            camera_matrix = data["camera_matrix"]
            dist_coeffs   = data["dist_coeffs"]

            h, w = img.shape[:2]
            new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h),
            )
            img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_matrix)
            print(f"Applied factory lens undistortion.")
        except Exception as e:
            print(f"Warning: undistortion failed, using raw frame: {e}")
        finally:
            intr_file.unlink(missing_ok=True)

    return img


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
        print("Capturing from RealSense camera (subprocess worker)...")
        img_raw = _capture_via_subprocess()
    else:
        print(f"Using test image: {TEST_IMAGE_NAME}")
        img_raw = load_test_image(TEST_IMAGE_NAME)

    # ------------------------------------------------------------------
    # ROI crop — remove non-workspace edges (walls, cables, robot arm)
    # Crop is applied once here so every downstream stage (Charuco,
    # AprilTags, part detection, display) operates on the clean frame.
    # Set any value to 0 to disable that side.
    # ------------------------------------------------------------------
    ROI_CROP_LEFT   = 230
    ROI_CROP_RIGHT  = 80
    ROI_CROP_TOP    = 0
    ROI_CROP_BOTTOM = 0

    h_full, w_full = img_raw.shape[:2]
    x0 = ROI_CROP_LEFT
    x1 = w_full - ROI_CROP_RIGHT
    y0 = ROI_CROP_TOP
    y1 = h_full - ROI_CROP_BOTTOM
    img_raw = img_raw[y0:y1, x0:x1].copy()
    print(f"ROI crop: {w_full}x{h_full} → {img_raw.shape[1]}x{img_raw.shape[0]}  "
          f"(L={ROI_CROP_LEFT} R={ROI_CROP_RIGHT} T={ROI_CROP_TOP} B={ROI_CROP_BOTTOM})")

    img_vis = img_raw.copy()
    # Force a contiguous C-layout array before passing to any native library.
    # cv2.cvtColor does not guarantee contiguous output, and pupil_apriltags
    # will intermittently segfault if the array is not contiguous in memory.
    gray = np.ascontiguousarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY))

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

    # ── AprilTag detection ────────────────────────────────────────────────────
    # Second checkpoint: catch any late import of pyrealsense2.
    assert "pyrealsense2" not in sys.modules, (
        "pyrealsense2 was imported after startup — check any code that ran "
        "between program launch and this point (capture helper, config, etc.)."
    )

    # pupil_apriltags intermittently segfaults on the first call when the
    # process memory is in a certain state.  Running detection in a subprocess
    # isolates the crash so Vision_Main itself survives.  We retry once with a
    # fresh subprocess if the first attempt fails.
    import subprocess as _sub
    import tempfile as _tmp
    import pickle as _pickle

    _detect_script = Path(__file__).resolve().parent / "_apriltag_worker.py"

    def _run_apriltag_subprocess(gray_arr: "np.ndarray") -> list:
        """
        Serialise gray array → temp file, run apriltag in a clean subprocess,
        return detections list.  Raises RuntimeError on failure.
        """
        with _tmp.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in:
            _pickle.dump(gray_arr, f_in)
            in_path = f_in.name
        with _tmp.NamedTemporaryFile(suffix=".pkl", delete=False) as f_out:
            out_path = f_out.name

        try:
            result = _sub.run(
                [sys.executable, str(_detect_script), in_path, out_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"AprilTag worker failed (exit {result.returncode}):\n"
                    f"  stdout: {result.stdout.strip()}\n"
                    f"  stderr: {result.stderr.strip()}"
                )
            with open(out_path, "rb") as f:
                detections = _pickle.load(f)
            return detections
        finally:
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    for _attempt in range(2):
        try:
            detections = _run_apriltag_subprocess(gray)
            break
        except Exception as _e:
            if _attempt == 0:
                print(f"  ⚠  AprilTag attempt 1 failed ({_e}) — retrying …")
            else:
                print(f"❌  AprilTag detection failed after 2 attempts: {_e}")
                return

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
        draw_parts=False,           # parts are drawn below after saving base image
    )

    # Save base image (tags + slots annotated, NO part labels) for re-annotation
    # after config updates.  This image serves as a clean starting point when
    # part IDs change during an "Update Config" session.
    _file_exchange = Path(__file__).resolve().parents[1] / "File_Exchange"
    _file_exchange.mkdir(parents=True, exist_ok=True)
    _base_image_path = _file_exchange / "latest_image_base.png"
    cv2.imwrite(str(_base_image_path), img_vis)

    # Build part annotation list from tag_targets and draw onto img_vis
    _part_annotations = []
    for p in tag_targets.get(-1000, []):
        _part_annotations.append({
            "name": p["name_suffix"],
            "cx_px": p["cx_px"],
            "cy_px": p["cy_px"],
            "color": p.get("color", "Unknown"),
            "diameter_mm": float(p.get("diameter_mm", 0.0)),
        })

    annotate_parts(img_vis, _part_annotations)

    # Save pixel map for re-annotation after config updates
    _pixel_map_path = _file_exchange / "latest_pixel_map.json"
    with open(str(_pixel_map_path), "w", encoding="utf-8") as f:
        json.dump(_part_annotations, f, indent=2, ensure_ascii=False)

    del detections  # free before part detection

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
    save_json_snapshot(CONFIGURATION_PATH, state, pretty=True)
    print(f"Wrote JSON snapshot to: {CONFIGURATION_PATH}")
    save_llm_snapshot(LLM_INPUT_PATH, state, pretty=True)
    print(f"Wrote LLM input snapshot to: {LLM_INPUT_PATH}")

    # ------------------------------------------------------------------
    # Save annotated image
    # ------------------------------------------------------------------
    _latest_image_path = Path(__file__).resolve().parents[1] / "File_Exchange" / "latest_image.png"
    cv2.imwrite(str(_latest_image_path), img_vis)
    print(f"Saved annotated image to: {_latest_image_path}")

    # ------------------------------------------------------------------
    # Display  (skipped when ROBOT_GUI_MODE=1 — the GUI shows the image)
    # ------------------------------------------------------------------
    import os as _os
    if _os.environ.get("ROBOT_GUI_MODE") != "1":
        source_label = "CAMERA" if USE_CAMERA else f"IMAGE: {TEST_IMAGE_NAME}"
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.setWindowTitle("Result", f"Result [{source_label}]")
        cv2.imshow("Result", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()