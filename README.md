# Master Thesis — LLM-Driven Workspace Configurator for Franka Robot

**Author:** Sandro Gabl  
**Year:** 2026

Natural-language-driven workspace configuration and motion planning for the
Franka Emika robot arm. An operator describes workspace changes in plain language;
the system translates instructions into validated configurations and pick-and-place
sequences via a PDDL planner or LLM dialogue.

## Repository layout

```
├── python/          Main system — LLM orchestration, vision, execution
│   ├── src/         Installable Python package (robot_configurator)
│   ├── Vision_Module/    Camera, AprilTag & ChArUco detection
│   ├── Execution_Module/ ROS-based robot motion
│   ├── Utilities/        Calibration & diagnostic scripts
│   └── README.md    Full installation & usage guide
│
└── Experiments/     Thesis evaluation data
    ├── Scenario_1…4/    Session transcripts + before/after photos
    └── README.md    Experiment descriptions
```

> **Note:** `build/`, `devel/`, and `logs/` are ROS catkin build artifacts
> (from the `franka_ros` workspace) and are excluded from version control
> via `.gitignore`.

## Quick start

See [`python/README.md`](python/README.md) for full installation instructions.
