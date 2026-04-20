# Robot Configurator

LLM-driven workspace configurator and motion planner for the Franka Emika Panda robot.  
Operators describe workspace changes in natural language; the system translates instructions
into validated configurations and pick-and-place sequences executed via MoveIt.

This is the software artefact for the Master's Thesis:
> Gabl, S. (2026). *Natural-language-driven workspace configuration for robot manipulation*. Master's Thesis, University of Applied Sciences.

---

## System Requirements

This project runs on **Ubuntu 22.04** and requires the following to be installed on the system before proceeding:

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
├── Main.py                    Entry point (CLI)
├── gui.py                     Tkinter GUI entry point
├── session_handler.py         Pipeline orchestrator
├── run_execute.py             Robot execution subprocess
├── planner_interface.py       PDDL / LLM planner selector
├── paths.py                   Legacy path shim (compatibility)
│
├── src/robot_configurator/    Installable Python package
│   ├── core/
│   │   ├── config.py          Settings loaded from environment variables
│   │   └── paths.py           Path constants + atomic file I/O helpers
│   ├── communication/
│   │   ├── api_main.py        LLM conversation loop (OpenAI)
│   │   ├── prompts.py         System prompts
│   │   ├── capacity_tools.py  Deterministic capacity checks
│   │   └── ...
│   └── configuration/
│       ├── apply_config_changes.py
│       ├── apply_sequence_changes.py
│       └── update_scene.py
│
├── Vision_Module/             Camera, AprilTag & ChArUco detection, RealSense
├── Execution_Module/          ROS/MoveIt robot motion control
├── Utilities/                 Camera calibration & diagnostic scripts
├── workspace/             Runtime JSON exchange between modules
└── Memory/                    Timestamped configuration snapshots
```

The Fast Downward PDDL planner is used as a **git submodule** at `downward/`.

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

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the Python package

```bash
pip install -e ".[dev]"
```

This installs `robot_configurator` in editable mode along with:
`openai`, `python-dotenv`, `numpy`, `opencv-python`, `pupil-apriltags`

Install the RealSense Python bindings separately (not on PyPI for all platforms):

```bash
pip install pyrealsense2
```

> If `pip install pyrealsense2` fails, install via the Intel RealSense SDK instead:  
> [github.com/IntelRealSense/librealsense/tree/master/wrappers/python](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)

### 5. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 6. Source ROS

```bash
source /opt/ros/noetic/setup.bash
source ~/ws_moveit/devel/setup.bash   # your catkin workspace
```

---

## Running the system

### GUI (recommended)

```bash
python gui.py
```

### CLI

```bash
python Main.py
```

### Execution only (move robot using existing sequence.json)

```bash
python run_execute.py
```

---

## Configuration

All tuneable parameters are set through environment variables in your `.env` file.  
See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key. |
| `RC_MODEL` | `gpt-4.1` | LLM model for all API calls. |
| `RC_MAX_TOOL_ROUNDS` | `3` | Max tool-call rounds per LLM turn. |
| `RC_POSITION_MATCH_THRESHOLD_M` | `0.040` | XY auto-match threshold (metres). |
| `DOWNWARD_PATH` | `downward/fast-downward.py` | Path to Fast Downward entry point. |

Toggle between LLM-based and PDDL-based sequence planning in `Vision_Module/config.py`:

```python
USE_PDDL_PLANNER = True   # use Fast Downward
USE_PDDL_PLANNER = False  # use LLM dialogue
```

---


## Adapting to a different robot

All robot-specific names are in one place: `Execution_Module/Robot_Main.py`.

```python
ROBOT_CONFIG = {
    "arm_group":      "panda_arm",        # MoveIt planning group for the arm
    "hand_group":     "panda_hand",       # MoveIt planning group for the gripper
    "finger_joint_1": "panda_finger_joint1",  # URDF name of gripper finger 1
    "finger_joint_2": "panda_finger_joint2",  # URDF name of gripper finger 2
}
```

Change these four strings to match your robot's MoveIt configuration.  
The motion profiles (PTP/LIN, velocity/acceleration scaling) are in `robot.py`
and may also need tuning for a different manipulator.

## Running tests

```bash
pytest              # all unit tests
pytest -v           # verbose
pytest --cov        # with coverage report
```

---

## Branching strategy

| Branch | Purpose |
|---|---|
| `main` | Stable releases |
| `refactored-cleanup` | Package restructuring |
| `Priority_New` | Priority-handling experiments |

---

## Citing

If you use this software in academic work, please cite:

> Gabl, S. (2026). *Natural-language-driven workspace configuration for robot manipulation*. Master's Thesis, University of Applied Sciences.

## License

MIT — see `LICENSE`.
