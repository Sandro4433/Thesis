"""Tests for apply_config_changes.apply_changes."""
from __future__ import annotations

import pytest

from Configuration_Module.apply_config_changes import apply_changes


class TestApplyChanges:
    def test_role_change_receptacle(self, minimal_pddl_state):
        changes = {"Container_2": {"role": "output"}}
        result = apply_changes(minimal_pddl_state, changes)
        roles = {e["object"]: e["role"] for e in result["predicates"]["role"]}
        assert roles["Container_2"] == "output"

    def test_role_upserts_existing(self, minimal_pddl_state):
        changes = {"Container_1": {"role": "output"}}
        result = apply_changes(minimal_pddl_state, changes)
        roles = {e["object"]: e["role"] for e in result["predicates"]["role"]}
        assert roles["Container_1"] == "output"

    def test_workspace_update(self, minimal_pddl_state):
        changes = {"workspace": {"operation_mode": "kitting", "batch_size": 3}}
        result = apply_changes(minimal_pddl_state, changes)
        assert result["workspace"]["operation_mode"] == "kitting"
        assert result["workspace"]["batch_size"] == 3

    def test_priority_replaces(self, minimal_pddl_state):
        changes = {"priority": [{"color": "red", "order": 1}]}
        result = apply_changes(minimal_pddl_state, changes)
        assert result["predicates"]["priority"] == [{"color": "red", "order": 1}]

    def test_kit_recipe_replaces(self, minimal_pddl_state):
        changes = {"kit_recipe": [{"color": "blue", "quantity": 2}]}
        result = apply_changes(minimal_pddl_state, changes)
        assert result["predicates"]["kit_recipe"] == [{"color": "blue", "quantity": 2}]

    def test_part_compatibility_replaces(self, minimal_pddl_state):
        changes = {"part_compatibility": [{"part_color": "red", "allowed_in": ["Kit_1"]}]}
        result = apply_changes(minimal_pddl_state, changes)
        assert len(result["predicates"]["part_compatibility"]) == 1

    def test_color_change_part(self, minimal_pddl_state):
        changes = {"Part_1": {"color": "green"}}
        result = apply_changes(minimal_pddl_state, changes)
        colors = {e["part"]: e["color"] for e in result["predicates"]["color"]}
        assert colors["Part_1"] == "green"

    def test_fragility_set_fragile(self, minimal_pddl_state):
        changes = {"Part_1": {"fragility": "fragile"}}
        result = apply_changes(minimal_pddl_state, changes)
        frags = {e["part"]: e["fragility"] for e in result["predicates"]["fragility"]}
        assert frags["Part_1"] == "fragile"

    def test_fragility_set_normal_removes_entry(self, minimal_pddl_state):
        # First set it fragile, then reset to normal
        state = apply_changes(minimal_pddl_state, {"Part_1": {"fragility": "fragile"}})
        result = apply_changes(state, {"Part_1": {"fragility": "normal"}})
        frags = [e["part"] for e in result["predicates"]["fragility"]]
        assert "Part_1" not in frags

    def test_does_not_mutate_input(self, minimal_pddl_state):
        import copy
        original = copy.deepcopy(minimal_pddl_state)
        apply_changes(minimal_pddl_state, {"Container_1": {"role": "output"}})
        assert minimal_pddl_state == original

    def test_unknown_key_skipped_without_crash(self, minimal_pddl_state):
        changes = {"NonExistent_99": {"role": "input"}}
        result = apply_changes(minimal_pddl_state, changes)
        # Should not raise, unknown key is logged and skipped
        assert result is not None

    def test_null_priority_sets_empty_list(self, minimal_pddl_state):
        changes = {"priority": None}
        result = apply_changes(minimal_pddl_state, changes)
        assert result["predicates"]["priority"] == []
