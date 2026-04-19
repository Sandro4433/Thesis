"""Tests for scene_helpers.slim_scene."""
from __future__ import annotations

import pytest

from robot_configurator.communication.scene_helpers import slim_scene


class TestSlimScene:
    def test_returns_expected_keys(self, minimal_pddl_state):
        result = slim_scene(minimal_pddl_state)
        for key in ("workspace", "receptacle_xy", "capacity", "slots", "parts"):
            assert key in result

    def test_slot_has_child_part(self, minimal_pddl_state):
        result = slim_scene(minimal_pddl_state)
        slot = result["slots"]["Container_1_Pos_1"]
        assert slot["child_part"] is not None
        assert slot["child_part"]["name"] == "Part_1"
        assert slot["child_part"]["color"] == "red"

    def test_empty_slot_has_none_child(self, minimal_pddl_state):
        result = slim_scene(minimal_pddl_state)
        slot = result["slots"]["Container_2_Pos_1"]
        assert slot["child_part"] is None

    def test_receptacle_xy_computed_as_centroid(self, minimal_pddl_state):
        result = slim_scene(minimal_pddl_state)
        # Container_1 has two slots at [0.55, 0.28] and [0.50, 0.28]
        xy = result["receptacle_xy"]["Container_1"]
        assert abs(xy[0] - 0.525) < 0.01
        assert abs(xy[1] - 0.28) < 0.01

    def test_capacity_counts(self, minimal_pddl_state):
        result = slim_scene(minimal_pddl_state)
        cap = result["capacity"]["Container_1"]
        assert cap["total_slots"] == 2
        assert cap["occupied"] == 2
        assert cap["empty"] == 0

    def test_role_propagated_to_slot(self, minimal_pddl_state):
        result = slim_scene(minimal_pddl_state)
        assert result["slots"]["Container_1_Pos_1"]["role"] == "input"
        assert result["slots"]["Container_2_Pos_1"]["role"] == "output"

    def test_standalone_parts_in_parts_view(self, minimal_pddl_state):
        # Add a standalone part
        state = dict(minimal_pddl_state)
        state["objects"] = dict(state["objects"])
        state["objects"]["parts"] = ["Part_1", "Part_2", "Part_3"]
        state["predicates"] = dict(state["predicates"])
        state["predicates"]["color"] = state["predicates"]["color"] + [
            {"part": "Part_3", "color": "green"}
        ]
        state["metric"] = dict(state["metric"])
        state["metric"]["Part_3"] = {"pos": [0.3, 0.4, 0.01]}

        result = slim_scene(state)
        assert "Part_3" in result["parts"]
        assert result["parts"]["Part_3"]["color"] == "green"

    def test_empty_state_does_not_raise(self):
        result = slim_scene({})
        assert result["slots"] == {}
        assert result["parts"] == {}
