"""
core/config.py — Application settings loaded from environment variables.

All tuneable parameters live here.  Source code never contains secrets or
magic strings — import from this module instead.

Usage
-----
    from Core.config import settings

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(model=settings.model, ...)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels above this file:
# src/robot_configurator/core/config.py → project root).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=True)


class Settings:
    """
    Immutable settings object.  All values are read once at import time so
    that misconfiguration is caught early rather than mid-session.
    """

    # ── LLM ────────────────────────────────────────────────────────────────
    openai_api_key: str
    model: str
    max_tool_rounds: int

    # ── Hardware / mode toggles ────────────────────────────────────────────
    use_pddl_planner: bool
    """True → Fast Downward; False → LLM dialogue for sequence generation."""
    use_camera: bool
    """True → capture live from RealSense; False → load test image from disk."""
    test_image_name: str
    """Filename inside Vision_Module/Images/ used when use_camera is False."""

    # ── MoveIt / robot hardware ────────────────────────────────────────────
    arm_group: str
    """MoveIt planning group name for the arm (e.g. 'panda_arm')."""
    hand_group: str
    """MoveIt planning group name for the hand (e.g. 'panda_hand')."""
    finger_joint_1: str
    """URDF name of the first gripper finger joint."""
    finger_joint_2: str
    """URDF name of the second gripper finger joint."""

    # ── Physical thresholds ────────────────────────────────────────────────
    position_match_threshold_m: float
    """XY distance threshold (metres) for auto-matching parts across scans."""

    # ── Paths ──────────────────────────────────────────────────────────────
    project_root: Path
    configuration_path: Path
    sequence_path: Path
    changes_path: Path
    memory_dir: Path
    workspace_dir: Path
    downward_path: Path

    def __init__(self) -> None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set.  "
                "Copy .env.example to .env and add your key."
            )
        self.openai_api_key = key

        self.model = os.environ.get("RC_MODEL", "gpt-4.1")
        self.max_tool_rounds = int(os.environ.get("RC_MAX_TOOL_ROUNDS", "3"))
        self.use_pddl_planner = os.environ.get(
            "RC_USE_PDDL_PLANNER", "true"
        ).lower() in ("1", "true", "yes")
        self.use_camera = os.environ.get(
            "RC_USE_CAMERA", "false"
        ).lower() in ("1", "true", "yes")
        self.test_image_name = os.environ.get("RC_TEST_IMAGE", "Scenario_1.png")
        self.arm_group      = os.environ.get("RC_ARM_GROUP",      "panda_arm")
        self.hand_group     = os.environ.get("RC_HAND_GROUP",     "panda_hand")
        self.finger_joint_1 = os.environ.get("RC_FINGER_JOINT_1", "panda_finger_joint1")
        self.finger_joint_2 = os.environ.get("RC_FINGER_JOINT_2", "panda_finger_joint2")
        self.position_match_threshold_m = float(
            os.environ.get("RC_POSITION_MATCH_THRESHOLD_M", "0.040")
        )

        root = Path(os.environ.get("RC_PROJECT_ROOT", str(_PROJECT_ROOT)))
        self.project_root = root
        self.configuration_path = root / os.environ.get(
            "RC_CONFIGURATION_PATH", "configuration.json"
        )
        self.sequence_path = root / os.environ.get(
            "RC_SEQUENCE_PATH", "sequence.json"
        )
        self.changes_path = root / os.environ.get(
            "RC_CHANGES_PATH", "workspace/changes.json"
        )
        self.memory_dir = root / os.environ.get("RC_MEMORY_DIR", "Memory")
        self.workspace_dir = root / os.environ.get(
            "RC_WORKSPACE_DIR", "workspace"
        )
        self.downward_path = root / os.environ.get(
            "DOWNWARD_PATH", "downward/fast-downward.py"
        )


# Module-level singleton — import this everywhere.
settings = Settings()