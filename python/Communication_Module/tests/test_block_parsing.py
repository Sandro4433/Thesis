"""Tests for block_parsing — the most safety-critical module in the codebase."""
from __future__ import annotations

import json
import pytest

from Communication_Module.block_parsing import (
    extract_changes_block,
    extract_mapping_block,
    extract_sequence_block,
)


# ── sequence block ─────────────────────────────────────────────────────────────

class TestExtractSequenceBlock:
    def test_canonical_two_element(self):
        text = '```sequence\n[["Part_1", "Kit_1_Pos_1"]]\n```'
        result = extract_sequence_block(text)
        assert result == [["Part_1", "Kit_1_Pos_1"]]

    def test_multiple_steps(self):
        text = '```sequence\n[["Part_1", "Kit_1_Pos_1"], ["Part_2", "Kit_1_Pos_2"]]\n```'
        result = extract_sequence_block(text)
        assert len(result) == 2
        assert result[1] == ["Part_2", "Kit_1_Pos_2"]

    def test_three_element_strips_gripper_width(self):
        text = '```sequence\n[["Part_1", "Kit_1_Pos_1", 0.04]]\n```'
        result = extract_sequence_block(text)
        assert result == [["Part_1", "Kit_1_Pos_1"]]

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="No.*sequence.*block"):
            extract_sequence_block("no block here")

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            extract_sequence_block("```sequence\n[]\n```")

    def test_non_string_names_raise(self):
        with pytest.raises(ValueError):
            extract_sequence_block('```sequence\n[[1, "Kit_1_Pos_1"]]\n```')

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            extract_sequence_block('```sequence\n[["", "Kit_1_Pos_1"]]\n```')

    def test_case_insensitive_fence(self):
        text = "```SEQUENCE\n[[\"Part_1\", \"Slot_1\"]]\n```"
        result = extract_sequence_block(text)
        assert result == [["Part_1", "Slot_1"]]


# ── changes block ──────────────────────────────────────────────────────────────

class TestExtractChangesBlock:
    def test_role_change(self):
        text = '```changes\n{"Container_1": {"role": "input"}}\n```\nConfirm?'
        result = extract_changes_block(text)
        assert result == {"Container_1": {"role": "input"}}

    def test_workspace_change(self):
        text = '```changes\n{"workspace": {"operation_mode": "kitting", "batch_size": 2}}\n```'
        result = extract_changes_block(text)
        assert result["workspace"]["operation_mode"] == "kitting"

    def test_color_change(self):
        text = '```changes\n{"Part_1": {"color": "Red"}}\n```'
        result = extract_changes_block(text)
        assert result["Part_1"]["color"] == "Red"

    def test_fragility_change(self):
        text = '```changes\n{"Part_1": {"fragility": "fragile"}}\n```'
        result = extract_changes_block(text)
        assert result["Part_1"]["fragility"] == "fragile"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            extract_changes_block('```changes\n{"Container_1": {"role": "invalid"}}\n```')

    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="color"):
            extract_changes_block('```changes\n{"Part_1": {"color": "purple"}}\n```')

    def test_unknown_attribute_raises(self):
        with pytest.raises(ValueError, match="unknown attribute"):
            extract_changes_block('```changes\n{"Part_1": {"weight": 0.5}}\n```')

    def test_null_role_allowed(self):
        text = '```changes\n{"Container_1": {"role": null}}\n```'
        result = extract_changes_block(text)
        assert result["Container_1"]["role"] is None

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="No.*changes.*block"):
            extract_changes_block("no block here")

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            extract_changes_block("```changes\n{}\n```")

    def test_fallback_before_confirm(self):
        """Fallback regex: block without closing fence before 'Confirm'."""
        text = '```changes\n{"Container_2": {"role": "output"}}\nConfirm?'
        result = extract_changes_block(text)
        assert result["Container_2"]["role"] == "output"

    def test_priority_list_allowed(self):
        text = '```changes\n{"priority": [{"color": "red", "order": 1}]}\n```'
        result = extract_changes_block(text)
        assert result["priority"][0]["order"] == 1

    def test_null_priority_allowed(self):
        text = '```changes\n{"priority": null}\n```'
        result = extract_changes_block(text)
        assert result["priority"] is None

    def test_empty_part_compatibility_allowed(self):
        text = '```changes\n{"part_compatibility": []}\n```'
        result = extract_changes_block(text)
        assert result["part_compatibility"] == []


# ── mapping block ──────────────────────────────────────────────────────────────

class TestExtractMappingBlock:
    def test_canonical(self):
        text = '```mapping\n{"Part_5": "Part_3"}\n```'
        result = extract_mapping_block(text)
        assert result == {"Part_5": "Part_3"}

    def test_new_value(self):
        text = '```mapping\n{"Part_7": "new"}\n```'
        result = extract_mapping_block(text)
        assert result["Part_7"] == "new"

    def test_empty_mapping_allowed(self):
        text = '```mapping\n{}\n```'
        result = extract_mapping_block(text)
        assert result == {}

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Keys must be Part_"):
            extract_mapping_block('```mapping\n{"Container_1": "Part_1"}\n```')

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Values must be old Part_"):
            extract_mapping_block('```mapping\n{"Part_1": "blah"}\n```')

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="No.*mapping.*block"):
            extract_mapping_block("no block here")

    def test_fallback_label_outside_fences(self):
        """Fallback 1: label on its own line, JSON inside separate fences."""
        text = "\nmapping\n```\n{\"Part_5\": \"Part_3\"}\n```"
        result = extract_mapping_block(text)
        assert result == {"Part_5": "Part_3"}
