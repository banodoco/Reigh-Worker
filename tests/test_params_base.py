"""Tests for source/core/params/base.py."""

import pytest

from source.core.params.base import ParamGroup


class TestFlattenParams:
    def test_empty(self):
        assert ParamGroup.flatten_params({}) == {}

    def test_top_level_only(self):
        params = {"prompt": "hello", "steps": 10}
        result = ParamGroup.flatten_params(params)
        assert result == {"prompt": "hello", "steps": 10}

    def test_orchestrator_details_merge(self):
        params = {
            "orchestrator_details": {"resolution": "960x544", "steps": 5},
            "prompt": "test",
        }
        result = ParamGroup.flatten_params(params)
        assert result["prompt"] == "test"
        assert result["resolution"] == "960x544"
        assert result["steps"] == 5

    def test_full_orchestrator_payload_merge(self):
        params = {
            "full_orchestrator_payload": {"seed": 42, "model": "vace"},
            "prompt": "test",
        }
        result = ParamGroup.flatten_params(params)
        assert result["seed"] == 42
        assert result["model"] == "vace"

    def test_three_level_precedence(self):
        """top_level > orchestrator_details > full_orchestrator_payload."""
        params = {
            "full_orchestrator_payload": {"steps": 1, "seed": 100, "model": "a"},
            "orchestrator_details": {"steps": 2, "seed": 200},
            "steps": 3,
        }
        result = ParamGroup.flatten_params(params)
        assert result["steps"] == 3      # top-level wins
        assert result["seed"] == 200     # details wins over payload
        assert result["model"] == "a"    # only in payload

    def test_non_dict_nested_dropped(self):
        """If orchestrator_details is not a dict, it's excluded from flattened result."""
        params = {"orchestrator_details": "not_a_dict", "prompt": "test"}
        result = ParamGroup.flatten_params(params)
        # Non-dict orchestrator_details is neither merged nor kept as top-level
        assert "orchestrator_details" not in result
        assert result["prompt"] == "test"


class TestParseList:
    def test_none_returns_empty(self):
        assert ParamGroup._parse_list(None) == []

    def test_list_passthrough(self):
        assert ParamGroup._parse_list(["a", "b"]) == ["a", "b"]

    def test_comma_separated(self):
        assert ParamGroup._parse_list("a, b, c") == ["a", "b", "c"]

    def test_custom_separator(self):
        assert ParamGroup._parse_list("a;b;c", separator=";") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert ParamGroup._parse_list("  a , b  ") == ["a", "b"]

    def test_filters_empty(self):
        assert ParamGroup._parse_list("a,,b,") == ["a", "b"]

    def test_non_string_wrapping(self):
        assert ParamGroup._parse_list(42) == ["42"]


class TestGetFirstOf:
    def test_first_key(self):
        params = {"a": 1, "b": 2}
        assert ParamGroup._get_first_of(params, "a", "b") == 1

    def test_second_key(self):
        params = {"b": 2}
        assert ParamGroup._get_first_of(params, "a", "b") == 2

    def test_none_skipped(self):
        params = {"a": None, "b": 2}
        assert ParamGroup._get_first_of(params, "a", "b") == 2

    def test_no_keys_returns_default(self):
        assert ParamGroup._get_first_of({}, "a", "b") is None

    def test_custom_default(self):
        assert ParamGroup._get_first_of({}, "a", default=99) == 99
