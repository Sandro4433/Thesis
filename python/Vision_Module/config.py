# config.py
from typing import Dict, List, Any

# Output (stable paths; independent of current working directory)
from robot_configurator.core.paths import CONFIGURATION_PATH as CONFIGURATION_JSON, FILE_EXCHANGE_DIR as _FE_DIR
LLM_INPUT_JSON = _FE_DIR / "llm_input.json"

CONFIGURATION_PATH = str(CONFIGURATION_JSON.resolve())
LLM_INPUT_PATH     = str(LLM_INPUT_JSON.resolve())

# Define the Charuco origin in the ROBOT BASE FRAME. (METERS)
CHARUCO_ORIGIN_IN_ROBOT_M: Dict[str, float] = {
    "x": 0.157,  # +157 mm
    "y": 0.301,  # 301 mm
}

CAMERA_HOME: Dict[str, Any] = {
    "name": "Camera_Home",
    "pos": [0.25468104952011544, 0.4446658106363947, 0.7958241558793229],
    "quat": [-0.9214052431423866, -0.3884316547590146, -0.011368659081744474, 0.001995264788503215],
    "joints": [
        1.0283937304877353,
        0.1866175160754216,
        0.0338391137322888,
        -0.8281534481959998,
        -0.019907020211219786,
        1.0351364941067163,
        0.2533314509864326,
    ],
}


# Tag groupings (which AprilTag IDs belong to which object type)
KIT_TAG_IDS = {0, 2, 4}
CONTAINER_TAG_IDS = {1, 3, 5}

# Mapping from AprilTag ID to sequential numbering (1-3)
# Kits: AprilTag 0 → Kit 1, AprilTag 2 → Kit 2, AprilTag 4 → Kit 3
KIT_TAG_ID_TO_NUMBER: Dict[int, int] = {
    0: 1,
    2: 2,
    4: 3,
}

# Containers: AprilTag 1 → Container 1, AprilTag 3 → Container 2, AprilTag 5 → Container 3
CONTAINER_TAG_ID_TO_NUMBER: Dict[int, int] = {
    1: 1,
    3: 2,
    5: 3,
}

# Kit (AprilTag ID==0) local points (mm) + grip offsets relative to tag orientation (deg)
KIT_POINTS: List[Dict[str, float]] = [
    {"name": "Pos_1", "dx_mm": 65.0, "dy_mm": 30.0, "grip_off_deg": 30.0},
    {"name": "Pos_2", "dx_mm": 65.0, "dy_mm": -30.0, "grip_off_deg": -30.0},
    {"name": "Pos_3", "dx_mm": 120.0, "dy_mm": 0.0, "grip_off_deg": -90.0},
]

# Container (AprilTag ID==1) local points (mm) + grip offsets relative to tag orientation (deg)
CONTAINER_POINTS: List[Dict[str, float]] = [
    {"name": "Pos_1", "dx_mm": 65.0, "dy_mm": 0.0, "grip_off_deg": 0.0},
    {"name": "Pos_2", "dx_mm": 35.0, "dy_mm": 55.0, "grip_off_deg": 60.0},
    {"name": "Pos_3", "dx_mm": -35.0, "dy_mm": 55.0, "grip_off_deg": 120.0},
    {"name": "Pos_4", "dx_mm": -65.0, "dy_mm": 0.0, "grip_off_deg": 0.0},
    {"name": "Pos_5", "dx_mm": -35.0, "dy_mm": -55.0, "grip_off_deg": -120.0},
    {"name": "Pos_6", "dx_mm": 35.0, "dy_mm": -55.0, "grip_off_deg": -60.0},
]

# Board config (cali.io): Rows=8, Columns=11
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015

# Drawing / axes
AXIS_LEN_M = SQUARE_SIZE_M * 3.0       # meters
TAG_AXIS_DRAW_LEN_M = 0.04            # meters

# Robot output
Z_ROBOT_M = 0.2

# ── Pick offset correction ────────────────────────────────────────────────────
# X: linear left/right correction.  0 at image center, ±mm at edges.
#   Set to 0 to disable.
# Y: varies with vertical position in the image.
#   BOTTOM: mm offset at the bottom of the image
#   TOP:    mm offset at the top of the image
#   Interpolates linearly between them.  Set both to 0 to disable.
PERSPECTIVE_X_OFFSET_MM = 5.0
Y_OFFSET_BOTTOM_MM      = 3.0
Y_OFFSET_TOP_MM          = 15.0

# RealSense capture
REALSENSE_WIDTH = 1920
REALSENSE_HEIGHT = 1080
REALSENSE_FPS = 30
REALSENSE_WARMUP_FRAMES = 5

# ── Planner toggle ────────────────────────────────────────────────────────────
# Set USE_PDDL_PLANNER = True  to use Fast Downward for sequence generation.
# Set USE_PDDL_PLANNER = False to use the LLM dialogue for sequence generation.
#
# NOTE: Workspace reconfiguration (attribute changes, role assignment, etc.)
# always goes through the LLM regardless of this setting.
# Only the SEQUENCE PLANNING step (Scenarios 4–6) is affected.
USE_PDDL_PLANNER: bool = True

# Path to the Fast Downward executable.
# After cloning and building: https://github.com/aibasel/downward
# Set this to the full path of the fast-downward script, e.g.:
#   FAST_DOWNWARD_PATH = "/home/user/downward/fast-downward.py"
#
# If set, Fast Downward is used with PDDL 2.1 action costs (enforces priority
# ordering as a hard constraint). If empty, pyperplan is used as a fallback.
# Resolved relative to this file: python/downward/fast-downward.py
# Override with the DOWNWARD_PATH environment variable if your layout differs.
import os as _os
FAST_DOWNWARD_PATH: str = _os.environ.get(
    "DOWNWARD_PATH",
    str(Path(__file__).resolve().parents[1] / "downward" / "fast-downward.py"),
)
del _os