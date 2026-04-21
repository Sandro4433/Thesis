"""Shared pytest fixtures for robot_configurator tests."""
from __future__ import annotations

import pytest


@pytest.fixture
def minimal_scene() -> dict:
    """A minimal but structurally valid slim_scene dict."""
    return {
        "workspace": {"operation_mode": "sorting", "batch_size": None},
        "receptacle_xy": {
            "Container_1": [0.5, 0.3],
            "Container_2": [0.2, 0.3],
            "Kit_1": [0.5, 0.6],
        },
        "capacity": {
            "Container_1": {
                "total_slots": 3, "occupied": 2, "empty": 1,
                "parts_by_color": {"red": 1, "blue": 1}, "role": "input",
            },
            "Container_2": {
                "total_slots": 3, "occupied": 0, "empty": 3,
                "parts_by_color": {}, "role": "output",
            },
            "Kit_1": {
                "total_slots": 3, "occupied": 0, "empty": 3,
                "parts_by_color": {}, "role": "output",
            },
        },
        "slots": {
            "Container_1_Pos_1": {
                "xy": [0.55, 0.28], "role": "input",
                "child_part": {"name": "Part_1", "color": "red", "fragility": "normal"},
            },
            "Container_1_Pos_2": {
                "xy": [0.50, 0.28], "role": "input",
                "child_part": {"name": "Part_2", "color": "blue", "fragility": "normal"},
            },
            "Container_1_Pos_3": {"xy": [0.45, 0.28], "role": "input", "child_part": None},
            "Container_2_Pos_1": {"xy": [0.25, 0.28], "role": "output", "child_part": None},
            "Container_2_Pos_2": {"xy": [0.20, 0.28], "role": "output", "child_part": None},
            "Container_2_Pos_3": {"xy": [0.15, 0.28], "role": "output", "child_part": None},
            "Kit_1_Pos_1": {"xy": [0.55, 0.62], "role": "output", "child_part": None},
            "Kit_1_Pos_2": {"xy": [0.50, 0.62], "role": "output", "child_part": None},
            "Kit_1_Pos_3": {"xy": [0.45, 0.62], "role": "output", "child_part": None},
        },
        "parts": {},
        "priority": [],
        "kit_recipe": [],
        "part_compatibility": [],
    }


@pytest.fixture
def minimal_pddl_state() -> dict:
    """A minimal PDDL-friendly configuration.json dict."""
    return {
        "workspace": {"operation_mode": "sorting", "batch_size": None},
        "objects": {
            "kits": ["Kit_1"],
            "containers": ["Container_1", "Container_2"],
            "slots": [
                "Container_1_Pos_1", "Container_1_Pos_2",
                "Container_2_Pos_1", "Container_2_Pos_2",
                "Kit_1_Pos_1", "Kit_1_Pos_2",
            ],
            "parts": ["Part_1", "Part_2"],
        },
        "slot_belongs_to": {
            "Container_1_Pos_1": "Container_1",
            "Container_1_Pos_2": "Container_1",
            "Container_2_Pos_1": "Container_2",
            "Container_2_Pos_2": "Container_2",
            "Kit_1_Pos_1": "Kit_1",
            "Kit_1_Pos_2": "Kit_1",
        },
        "predicates": {
            "at": [
                {"part": "Part_1", "slot": "Container_1_Pos_1"},
                {"part": "Part_2", "slot": "Container_1_Pos_2"},
            ],
            "slot_empty": ["Container_2_Pos_1", "Container_2_Pos_2", "Kit_1_Pos_1", "Kit_1_Pos_2"],
            "color": [
                {"part": "Part_1", "color": "red"},
                {"part": "Part_2", "color": "blue"},
            ],
            "role": [
                {"object": "Container_1", "role": "input"},
                {"object": "Container_2", "role": "output"},
            ],
            "priority": [],
            "kit_recipe": [],
            "part_compatibility": [],
            "fragility": [],
        },
        "metric": {
            "Part_1": {"pos": [0.55, 0.28, 0.01], "quat": [0, 0, 0, 1]},
            "Part_2": {"pos": [0.50, 0.28, 0.01], "quat": [0, 0, 0, 1]},
            "Container_1_Pos_1": {"pos": [0.55, 0.28, 0.0]},
            "Container_1_Pos_2": {"pos": [0.50, 0.28, 0.0]},
            "Container_2_Pos_1": {"pos": [0.25, 0.28, 0.0]},
            "Container_2_Pos_2": {"pos": [0.20, 0.28, 0.0]},
            "Kit_1_Pos_1": {"pos": [0.55, 0.62, 0.0]},
            "Kit_1_Pos_2": {"pos": [0.50, 0.62, 0.0]},
        },
    }
