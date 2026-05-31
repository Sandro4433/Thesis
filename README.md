# Conversational Task Configuration for Modular Pick-and-Place Robotics

LLM-driven, vision-grounded workspace configurator and motion planner for the Franka Emika Panda robot.
Non-expert operators describe workspace changes in natural language; the system grounds those
instructions in camera-based perception, resolves ambiguity through dialogue, validates the result
against the current workspace state, and produces verified pick-and-place sequences executed via MoveIt.

This is the software artefact for the Master's Thesis:

> Gabl, S. (2026). *Conversational Task Configuration for Modular Pick-and-Place Robotics: Resolving Instruction Ambiguity through Vision-Grounded Dialogue*. Master of Science with specialization in Robotics, University West, Department of Engineering Science, Trollhättan, Sweden.

## Demo

[![Demo video](https://img.youtube.com/vi/ELS43Rx1s2k/0.jpg)](https://www.youtube.com/watch?v=ELS43Rx1s2k)

## Overview

High-mix, low-volume manufacturing requires frequent robot reconfiguration, but the specialist
knowledge this normally demands limits flexibility on the shop floor. This project investigates
whether non-expert operators can perform semantic, task-level reconfiguration of a collaborative
pick-and-place system through natural language dialogue grounded in vision-based perception.

The integrated framework consists of:

- a **vision module** for object detection and localisation (ChArUco board calibration, AprilTag detection, colour-based part detection),
- a multi-turn **LLM dialogue module** with ambiguity detection and constraint checking,
- a **configuration module** that maintains a symbolic world state across sessions,
- a **PDDL planner** (Fast Downward backend) that produces verified motion sequences, and
- an **execution module** that drives the Franka Emika Panda via ROS and MoveIt.

The system was evaluated across four scenarios of increasing complexity, from single-attribute
baseline configuration to a full end-to-end production changeover, using deliberately ambiguous
phrasing and constraint-violating inputs. Across 45 configuration sessions it produced correct
configurations in all cases, 40 of them without operator intervention beyond answering the system's
own clarifications. In a separate set of 14 constraint-violation sub-trials, 12 violations were
detected.

The source code lives in [`python/`](python/). See [`python/README.md`](python/README.md) for the
full installation and usage guide. The thesis PDF is included in this repository as
[`Master_Thesis_Final.pdf`](Master_Thesis_Final.pdf).

## System requirements

Runs on **Ubuntu 22.04**. The following must be installed before proceeding:

| Dependency | Version | Install guide |
|---|---|---|
| ROS 1 Noetic | 1.16 | [wiki.ros.org/noetic/Installation/Ubuntu](http://wiki.ros.org/noetic/Installation/Ubuntu) |
| MoveIt 1 | 1.1.x | `sudo apt install ros-noetic-moveit` |
| franka_ros | 0.9.x | [frankaemika.github.io/docs/installation_linux](https://frankaemika.github.io/docs/installation_linux.html) |
| libfranka | 0.9.x | Installed as part of franka_ros |
| pilz_industrial_motion_planner | — | `sudo apt install ros-noetic-pilz-industrial-motion-planner` |
| Intel RealSense SDK | 2.x | [github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md) |
| Python | >= 3.9 | Ships with Ubuntu 22.04 |

> **Hardware:** Franka Emika Panda robot arm + Intel RealSense D435 camera.
> Tested on Ubuntu 22.04 + ROS Noetic + libfranka 0.9.2.

## Architecture

```
python/
├── Main.py                          Entry point, launches the GUI
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
│   ├── pddl_planner.py              PDDL planner (Fast Downward backend)
│   └── planner_interface.py         Unified planner entry point
│
├── Orchestration/                   Pipeline coordination and UI
│   ├── gui.py                       Tkinter GUI
│   ├── session_handler.py           Orchestrates all modules end-to-end
│   └── run_execute.py               Robot execution subprocess entry point
│
├── Vision_Module/                   Camera pipeline
│   ├── Vision_Main.py               Entry point for the vision subprocess
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
│   ├── config.py                    Settings loaded from environment variables
│   └── paths.py                     Path constants and atomic file I/O helpers
│
├── workspace/                       Runtime data exchanged between modules
├── Memory/                          Timestamped configuration snapshots
└── downward/                        Fast Downward PDDL planner (git submodule)
```

## Installation

```bash
git clone https://github.com/Sandro4433/Thesis.git
cd Thesis/python

# Fast Downward (PDDL planner) submodule
git submodule update --init --recursive
cd downward
python build.py
cd ..

# Python dependencies
pip install openai python-dotenv numpy opencv-python pupil-apriltags --break-system-packages
pip install pyrealsense2 --break-system-packages

# Configuration
cp env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Source ROS
source /opt/ros/noetic/setup.bash
source ~/ws_franka/devel/setup.bash   # your catkin workspace
```

> If `pip install pyrealsense2` fails, install via the Intel RealSense SDK instead:
> [github.com/IntelRealSense/librealsense/tree/master/wrappers/python](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)

## Running the system

```bash
python3 Main.py
```

This launches the Tkinter GUI. All other modules (Vision, Communication, Configuration, Planning,
Execution) are started from the GUI as needed.

## Configuration

All tuneable parameters are set through environment variables in your `.env` file.
See `python/env.example` for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key. |
| `RC_MODEL` | `gpt-4.1` | LLM model used for all API calls. |
| `RC_MAX_TOOL_ROUNDS` | `3` | Max tool-call rounds per LLM turn. |
| `RC_USE_CAMERA` | `false` | Live RealSense capture (`true`) or a test image (`false`). |
| `RC_USE_PDDL_PLANNER` | `false` | Use the Fast Downward PDDL planner (`true`) or LLM dialogue (`false`) for sequences. |
| `RC_POSITION_MATCH_THRESHOLD_M` | `0.02` | XY auto-match threshold in metres. |
| `DOWNWARD_PATH` | `downward/fast-downward.py` | Path to the Fast Downward entry point. |
| `ROS_WS_PATH` | — | Absolute path to your catkin `devel/setup.bash`. Required for robot execution. |

## Adapting to a different robot

Robot-specific names are set via environment variables (see `env.example`) and default to the
Franka Panda:

```
RC_ARM_GROUP=panda_arm
RC_HAND_GROUP=panda_hand
RC_FINGER_JOINT_1=panda_finger_joint1
RC_FINGER_JOINT_2=panda_finger_joint2
```

Change these to match your robot's MoveIt configuration. Motion profiles (velocity/acceleration
scaling) are in `Execution_Module/robot.py` and may also need tuning for a different manipulator.

## Running tests

Tests live inside each module alongside the code they test.

```bash
pytest Communication_Module/tests/
pytest Configuration_Module/tests/
pytest -v
```

## Citing

If you use this work academically, please cite the thesis:

> Gabl, S. (2026). *Conversational Task Configuration for Modular Pick-and-Place Robotics: Resolving Instruction Ambiguity through Vision-Grounded Dialogue*. Master of Science with specialization in Robotics, University West, Department of Engineering Science, Trollhättan, Sweden.

And the software:

> Gabl, S. (2026). *Robot Configurator* (software). GitHub. https://github.com/Sandro4433/Thesis

## License

MIT, see [`LICENSE`](LICENSE).
