# Franka Emika – Automated Assembly Task Controller

This project controls a Franka Emika Panda robot using **ROS**, **MoveIt**, and a **connected PLC**.
It processes vision-based assembly codes, generates task sequences, and executes complex manipulation actions such as assembling, disassembling, rotating, discarding, and placing parts into buffers or onto an AGV.

---

## Features

- Automated reading of assembly codes from a PLC
- Task sequence generation based on object color, type, and position
- Robust execution of assembly workflows with:
  - `Assemble`
  - `Disassemble`
  - `Rotate`
  - `Discard`
  - `PlaceOn_AGV`
- Integrated error recovery for Franka
- JSON-based persistent storage of:
  - taught robot points
  - buffer states
- MoveIt-based motion planning:
  - PTP (joint-based and pose-based)
  - LIN (Cartesian linear motion)

---

## Requirements

### Software
- Ubuntu 20.04
- ROS Noetic
- MoveIt
- Franka ROS Interface (FRI)
- Python 3.8+
- NumPy
- JSON
- `PLC_Connection_Franka` module

### Robot
- Franka Emika Panda
- Pilz Industrial Motion Planner enabled

---

## Installation

Clone repository and install dependencies:

```bash
sudo apt install ros-noetic-moveit
pip3 install numpy
```

---

## Usage

Run the main robot controller:

```bash
python3 franka_controller.py
```

Make sure the PLC IP/AMS addresses inside the script match your setup.

---

## File Structure

- `punkte5.jsonl` — stored robot poses and joint configurations
- `buffer_state.json` — persistent buffer fill levels
- `PLC_Connection_Franka.py` — PLC communication interface
- `franka_controller.py` — main program

---

## Error Recovery

If Franka detects a collision or stops unexpectedly, the script automatically calls:

```bash
/franka_control/error_recovery
```

This clears the robot errors and allows execution to continue.

---

## License

MIT License
