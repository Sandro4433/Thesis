"""Tests for change_management — merge, conflict detection, priority ambiguity."""
from __future__ import annotations

import pytest

from Communication_Module.change_management import (
    detect_conflicts,
    detect_priority_ambiguity,
    format_priority_ambiguities,
    merge_changes,
    resolve_conflicts,
)


class TestMergeChanges:
    def test_simple_merge(self):
        acc = {"Container_1": {"role": "input"}}
        new = {"Container_2": {"role": "output"}}
        result = merge_changes(acc, new)
        assert result["Container_1"]["role"] == "input"
        assert result["Container_2"]["role"] == "output"

    def test_workspace_merge_keys(self):
        acc = {"workspace": {"operation_mode": "sorting"}}
        new = {"workspace": {"batch_size": 2}}
        result = merge_changes(acc, new)
        assert result["workspace"]["operation_mode"] == "sorting"
        assert result["workspace"]["batch_size"] == 2

    def test_priority_replaces(self):
        acc = {"priority": [{"color": "red", "order": 1}]}
        new = {"priority": [{"color": "blue", "order": 1}]}
        result = merge_changes(acc, new)
        assert result["priority"] == [{"color": "blue", "order": 1}]

    def test_kit_recipe_replaces(self):
        acc = {"kit_recipe": [{"color": "red", "quantity": 1}]}
        new = {"kit_recipe": [{"color": "blue", "quantity": 2}]}
        result = merge_changes(acc, new)
        assert len(result["kit_recipe"]) == 1
        assert result["kit_recipe"][0]["color"] == "blue"

    def test_part_compatibility_accumulates(self):
        acc = {"part_compatibility": [{"part_color": "red", "allowed_in": ["Kit_1"]}]}
        new = {"part_compatibility": [{"part_color": "blue", "allowed_in": ["Kit_2"]}]}
        result = merge_changes(acc, new)
        assert len(result["part_compatibility"]) == 2

    def test_part_compatibility_empty_list_deletes(self):
        acc = {"part_compatibility": [{"part_color": "red", "allowed_in": ["Kit_1"]}]}
        new = {"part_compatibility": []}
        result = merge_changes(acc, new)
        assert result["part_compatibility"] == []

    def test_none_priority_normalised_to_empty_list(self):
        acc = {}
        new = {"priority": None}
        result = merge_changes(acc, new)
        assert result["priority"] == []

    def test_does_not_mutate_inputs(self):
        acc = {"Container_1": {"role": "input"}}
        new = {"Container_1": {"role": "output"}}
        original_acc = {"Container_1": {"role": "input"}}
        merge_changes(acc, new)
        assert acc == original_acc


class TestDetectConflicts:
    def test_no_conflict(self):
        acc = {"Container_1": {"role": "input"}}
        new = {"Container_2": {"role": "output"}}
        assert detect_conflicts(acc, new) == []

    def test_workspace_conflict(self):
        acc = {"workspace": {"operation_mode": "sorting"}}
        new = {"workspace": {"operation_mode": "kitting"}}
        conflicts = detect_conflicts(acc, new)
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "workspace"
        assert conflicts[0]["key"] == "operation_mode"

    def test_receptacle_role_conflict(self):
        acc = {"Container_1": {"role": "input"}}
        new = {"Container_1": {"role": "output"}}
        conflicts = detect_conflicts(acc, new)
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "receptacle"

    def test_part_color_conflict(self):
        acc = {"Part_1": {"color": "red"}}
        new = {"Part_1": {"color": "blue"}}
        conflicts = detect_conflicts(acc, new)
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "part"

    def test_none_workspace_value_not_a_conflict(self):
        """None → value is a fresh set, not a conflict."""
        acc = {"workspace": {"operation_mode": None}}
        new = {"workspace": {"operation_mode": "kitting"}}
        assert detect_conflicts(acc, new) == []


class TestResolveConflicts:
    def test_keep_new(self):
        acc = {"Container_1": {"role": "input"}}
        new = {"Container_1": {"role": "output"}}
        conflicts = detect_conflicts(acc, new)
        result = resolve_conflicts(acc, new, conflicts, keep_new=True)
        assert result["Container_1"]["role"] == "output"

    def test_keep_old(self):
        acc = {"Container_1": {"role": "input"}}
        new = {"Container_1": {"role": "output"}}
        conflicts = detect_conflicts(acc, new)
        result = resolve_conflicts(acc, new, conflicts, keep_new=False)
        assert result["Container_1"]["role"] == "input"


class TestDetectPriorityAmbiguity:
    def test_no_ambiguity(self):
        plist = [
            {"kit": "Kit_1", "order": 1},
            {"kit": "Kit_2", "order": 2},
        ]
        result = detect_priority_ambiguity(plist, {})
        assert result == []

    def test_duplicate_fill_order(self):
        plist = [
            {"kit": "Kit_1", "order": 1},
            {"kit": "Kit_2", "order": 1},
        ]
        result = detect_priority_ambiguity(plist, {})
        assert len(result) == 1
        assert "Kit_1" in result[0]
        assert "Kit_2" in result[0]

    def test_format_ambiguities_empty(self):
        assert format_priority_ambiguities([]) == ""

    def test_format_ambiguities_nonempty(self):
        output = format_priority_ambiguities(["Kit_1 and Kit_2 conflict"])
        assert "conflict" in output
