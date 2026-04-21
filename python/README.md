# Robot Configurator

LLM-driven workspace configurator and motion planner for the Franka Emika Panda robot.  
Operators describe workspace changes in natural language; the system translates instructions
into validated configurations and pick-and-place sequences executed via MoveIt.

This is the software artefact for the Master's Thesis:
> Gabl, S. (2026). *Natural-language-driven workspace configuration for robot manipulation*. Master's Thesis, University of Applied Sciences.

---

## System Requirements

This project runs on **Ubuntu 22.04** and requires the following to be installed before proceeding:

| Dependency | Version | Install guide |
|---|---|---|
| ROS 1 Noetic | 1.16 | [wiki.ros.org/noetic/Installation/Ubuntu](http://wiki.ros.org/noetic/Installation/Ubuntu) |
| MoveIt 1 | 1.1.x | `sudo apt install ros-noetic-moveit` |
| franka_ros | 0.9.x | [frankaemika.github.io/docs/installation_linux](https://frankaemika.github.io/docs/installation_linux.html) |
| libfranka | 0.9.x | Installed as part of franka_ros |
| pilz_industrial_motion_planner | — | `sudo apt install ros-noetic-pilz-industrial-motion-planner` |
| Intel RealSense SDK | 2.x | [github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md) |
| Python | ≥ 3.9 | Ships with Ubuntu 22.04 |

> **Hardware:** Franka Emika Panda robot arm + Intel RealSense D435 camera.  
> The system has been tested on Ubuntu 22.04 + ROS Noetic + libfranka 0.9.2.

---

## Architecture

```
python/
├── Main.py                          Entry point — launches the GUI
│
├── Communication_Module/            LLM conversation layer (OpenAI)
│   ├── api_main.py                  Main conversation loop
│   ├── prompts.py                   System prompts
│   ├── ambiguity_detection.py       Detects ambiguous user instructions
│   ├── block_parsing.py             Parses structured LLM output blocks
│   ├── capacity_tools.py            Deterministic capacity checks
│   ├── change_management.py         Tracks and validates proposed changes
│   ├── scene_helpers.py             Scene description utilities
│   ├── user_intent.py               Classifies user intent
│   └── tests/                       Unit tests for this module
│
├── Configuration_Module/            Applies validated changes to scene state
│   ├── apply_config_changes.py      Merges LLM changes into configuration
│   ├── apply_sequence_changes.py    Applies sequence-level changes
│   ├── update_scene.py              Full scene update pipeline
│   └── tests/                       Unit tests for this module
│
├── Planning_Module/                 Motion sequence planning
│   ├── pddl_planner.py              PDDL 2.1 planner (Fast Downward backend)
│   └── planner_interface.py         Unified planner entry point
│
├── Orchestration/                   Pipeline coordination and UI
│   ├── gui.py                       Tkinter GUI
│   ├── session_handler.py           Orchestrates all modules end-to-end
│   └── run_execute.py               Robot execution subprocess entry point
│
├── Vision_Module/                   Camera pipeline
│   ├── Vision_Main.py               Entry point for vision subprocess
│   ├── pipeline.py                  Full detection pipeline
│   ├── vision_circles.py            Colour-based part detection
│   ├── vision_charuco.py            ChArUco board detection
│   ├── vision_apriltag.py           AprilTag detection
│   ├── workspace_state.py           Converts detections to scene state
│   └── ...
│
├── Execution_Module/                ROS / MoveIt robot motion control
│   ├── Robot_Main.py                Robot configuration and entry point
│   ├── robot.py                     MoveIt robot interface
│   ├── sequence_executor.py         Executes pick-and-place sequences
│   ├── pick_and_place.py            Low-level pick and place primitives
│   └── move_camera_home.py          Moves robot to camera home position
│
├── Core/                            Shared settings and path constants
│   ├── config.py                    All settings loaded from environment variables
│   └── paths.py                     Path constants and atomic file I/O helpers
│
├── workspace/                       Runtime data exchanged between modules
│   ├── configuration.json           Active scene state
│   ├── sequence.json                Current pick-and-place sequence
│   ├── positions.json               Detected part positions
│   └── ...
│
├── Memory/                          Timestamped configuration snapshots
├── downward/                        Fast Downward PDDL planner (git submodule)
└── .env                             Local secrets and path overrides (git-ignored)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/robot-configurator.git
cd robot-configurator/python
```

### 2. Initialise Fast Downward (PDDL planner)

```bash
git submodule update --init --recursive
cd downward
python build.py
cd ..
```

> Fast Downward requires a C++17 compiler, CMake ≥ 3.16, and Python ≥ 3.6.  
> See `downward/BUILD.md` for details.

### 3. Install Python dependencies

```bash
pip install openai python-dotenv numpy opencv-python pupil-apriltags --break-system-packages
```

Install the RealSense Python bindings separately:

```bash
pip install pyrealsense2 --break-system-packages
```

> If `pip install pyrealsense2` fails, install via the Intel RealSense SDK instead:  
> [github.com/IntelRealSense/librealsense/tree/master/wrappers/python](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)

### 4. Set your OpenAI API key and paths

```bash
cp env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 5. Source ROS

```bash
source /opt/ros/noetic/setup.bash
source ~/ws_franka/devel/setup.bash   # your catkin workspace
```

---

## Running the system

### Start the GUI

```bash
python3 Main.py
```

This launches the Tkinter GUI. All other modules (Vision, Communication, Configuration, Planning, Execution) are started from within the GUI as needed.

---

## Configuration

All tuneable parameters are set through environment variables in your `.env` file.  
See `env.example` for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key. |
| `RC_MODEL` | `gpt-4.1` | LLM model used for all API calls. |
| `RC_MAX_TOOL_ROUNDS` | `3` | Max tool-call rounds per LLM turn. |
| `RC_POSITION_MATCH_THRESHOLD_M` | `0.040` | XY auto-match threshold in metres. |
| `RC_CONFIGURATION_PATH` | `workspace/configuration.json` | Active scene state file. |
| `RC_SEQUENCE_PATH` | `workspace/sequence.json` | Active sequence file. |
| `RC_CHANGES_PATH` | `workspace/changes.json` | Pending changes file. |
| `RC_MEMORY_DIR` | `Memory` | Directory for timestamped snapshots. |
| `DOWNWARD_PATH` | `downward/fast-downward.py` | Path to Fast Downward entry point. |
| `ROS_WS_PATH` | — | Absolute path to your catkin `devel/setup.bash`. Required for robot execution. |

Toggle between PDDL-based and LLM-based sequence planning in `Vision_Module/config.py`:

```python
USE_PDDL_PLANNER = True   # use Fast Downward (recommended)
USE_PDDL_PLANNER = False  # use LLM dialogue
```

---

## Adapting to a different robot

All robot-specific names are in one place: `Execution_Module/Robot_Main.py`.

```python
ROBOT_CONFIG = {
    "arm_group":      "panda_arm",
    "hand_group":     "panda_hand",
    "finger_joint_1": "panda_finger_joint1",
    "finger_joint_2": "panda_finger_joint2",
}
```

Change these four strings to match your robot's MoveIt configuration. Motion profiles (velocity/acceleration scaling) are in `Execution_Module/robot.py` and may also need tuning for a different manipulator.

---

## Running tests

Tests live inside each module alongside the code they test.

```bash
pytest Communication_Module/tests/
pytest Configuration_Module/tests/
pytest -v   # verbose output
```

---

## Citing

If you use this software in academic work, please cite:

> Gabl, S. (2026). *Natural-language-driven workspace configuration for robot manipulation*. Master's Thesis, University of Applied Sciences.

## License

MIT — see `LICENSE`.
