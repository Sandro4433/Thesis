"""Tests for apply_sequence_changes.apply_sequence."""
from __future__ import annotations

import pytest

from robot_configurator.configuration.apply_sequence_changes import apply_sequence


class TestApplySequence:
    def test_part_moves_from_slot_to_slot(self, minimal_pddl_state):
        sequence = [["Part_1", "Container_2_Pos_1"]]
        result = apply_sequence(minimal_pddl_state, sequence)
        at = {e["part"]: e["slot"] for e in result["predicates"]["at"]}
        assert at["Part_1"] == "Container_2_Pos_1"

    def test_source_slot_becomes_empty(self, minimal_pddl_state):
        sequence = [["Part_1", "Container_2_Pos_1"]]
        result = apply_sequence(minimal_pddl_state, sequence)
        assert "Container_1_Pos_1" in result["predicates"]["slot_empty"]

    def test_destination_slot_no_longer_empty(self, minimal_pddl_state):
        sequence = [["Part_1", "Container_2_Pos_1"]]
        result = apply_sequence(minimal_pddl_state, sequence)
        assert "Container_2_Pos_1" not in result["predicates"]["slot_empty"]

    def test_metric_updated_to_destination(self, minimal_pddl_state):
        sequence = [["Part_1", "Container_2_Pos_1"]]
        result = apply_sequence(minimal_pddl_state, sequence)
        dest_pos = minimal_pddl_state["metric"]["Container_2_Pos_1"]["pos"]
        assert result["metric"]["Part_1"]["pos"] == dest_pos

    def test_two_step_sequence(self, minimal_pddl_state):
        sequence = [
            ["Part_1", "Container_2_Pos_1"],
            ["Part_2", "Container_2_Pos_2"],
        ]
        result = apply_sequence(minimal_pddl_state, sequence)
        at = {e["part"]: e["slot"] for e in result["predicates"]["at"]}
        assert at["Part_1"] == "Container_2_Pos_1"
        assert at["Part_2"] == "Container_2_Pos_2"

    def test_does_not_mutate_input(self, minimal_pddl_state):
        import copy
        original = copy.deepcopy(minimal_pddl_state)
        apply_sequence(minimal_pddl_state, [["Part_1", "Container_2_Pos_1"]])
        assert minimal_pddl_state == original

    def test_malformed_entry_skipped(self, minimal_pddl_state):
        """Entries with fewer than 2 elements must be skipped without raising."""
        sequence = [["Part_1"], ["Part_2", "Container_2_Pos_1"]]
        result = apply_sequence(minimal_pddl_state, sequence)
        at = {e["part"]: e["slot"] for e in result["predicates"]["at"]}
        assert at["Part_2"] == "Container_2_Pos_1"
        # Part_1 should remain at its original slot
        assert at["Part_1"] == "Container_1_Pos_1"

    def test_unknown_destination_skipped(self, minimal_pddl_state):
        """Unknown destination slot should be skipped without raising."""
        sequence = [["Part_1", "NonExistent_Pos_99"]]
        result = apply_sequence(minimal_pddl_state, sequence)
        at = {e["part"]: e["slot"] for e in result["predicates"]["at"]}
        # Part_1 should remain at its original slot
        assert at["Part_1"] == "Container_1_Pos_1"

    def test_empty_sequence_returns_same_state(self, minimal_pddl_state):
        result = apply_sequence(minimal_pddl_state, [])
        at_before = {e["part"]: e["slot"] for e in minimal_pddl_state["predicates"]["at"]}
        at_after = {e["part"]: e["slot"] for e in result["predicates"]["at"]}
        assert at_before == at_after
