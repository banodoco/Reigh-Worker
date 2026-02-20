"""Tests for source/core/log/safe.py."""

import json
from unittest.mock import patch, MagicMock

import pytest

from source.core.log.safe import (
    LOG_MAX_STRING_REPR,
    LOG_MAX_OBJECT_OUTPUT,
    LOG_MAX_COLLECTION_ITEMS,
    LOG_MAX_NESTING_DEPTH,
    LOG_MAX_JSON_OUTPUT,
    LOG_MAX_DEBUG_MESSAGE,
    LOG_LARGE_DICT_KEYS,
    safe_repr,
    safe_dict_repr,
    safe_log_params,
    safe_json_repr,
    safe_log_change,
    SafeComponentLogger,
    headless_logger_safe,
    queue_logger_safe,
)


class TestSafeRepr:
    """Tests for the safe_repr function."""

    def test_simple_string(self):
        result = safe_repr("hello")
        assert "hello" in result

    def test_simple_int(self):
        result = safe_repr(42)
        assert "42" in result

    def test_none(self):
        result = safe_repr(None)
        assert "None" in result

    def test_large_list_truncated(self):
        big_list = list(range(1000))
        result = safe_repr(big_list)
        # Should be truncated, not contain all 1000 numbers
        assert len(result) <= LOG_MAX_OBJECT_OUTPUT + 10  # small margin for "...}"

    def test_custom_max_length(self):
        big_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = safe_repr(big_dict, max_length=50)
        assert len(result) <= 60  # 50 + "...}"

    def test_nested_dict_truncated(self):
        nested = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        result = safe_repr(nested)
        # Should be truncated at nesting depth
        assert len(result) < 500

    def test_repr_failure_returns_error_string(self):
        """Objects whose repr raises should return a safe fallback."""
        class BadRepr:
            def __repr__(self):
                raise ValueError("repr exploded")
        result = safe_repr(BadRepr())
        assert "<repr failed:" in result


class TestSafeDictRepr:
    """Tests for the safe_dict_repr function."""

    def test_simple_dict(self):
        result = safe_dict_repr({"a": 1, "b": 2})
        assert "'a': 1" in result
        assert "'b': 2" in result

    def test_non_dict_falls_back_to_safe_repr(self):
        result = safe_dict_repr("not a dict")
        assert isinstance(result, str)

    def test_large_dict_keys_collapsed(self):
        """Known problematic keys should be collapsed to '<dict with N keys>'."""
        d = {"orchestrator_payload": {"k1": "v1", "k2": "v2", "k3": "v3"}}
        result = safe_dict_repr(d)
        assert "<dict with 3 keys>" in result

    def test_max_items_limits_output(self):
        d = {f"key_{i}": i for i in range(20)}
        result = safe_dict_repr(d, max_items=3)
        assert "...17 more" in result

    def test_max_length_truncates(self):
        d = {f"key_{i}": "x" * 50 for i in range(10)}
        result = safe_dict_repr(d, max_length=100)
        assert len(result) <= 110  # 100 + "...}"

    def test_empty_dict(self):
        result = safe_dict_repr({})
        assert result == "{}"

    def test_large_collection_value_uses_reprlib(self):
        d = {"data": list(range(200))}
        result = safe_dict_repr(d)
        # The list value should be truncated
        assert "..." in result

    def test_long_string_value_truncated(self):
        d = {"key": "x" * 200}
        result = safe_dict_repr(d)
        assert "..." in result


class TestSafeLogParams:
    """Tests for safe_log_params."""

    def test_basic_usage(self):
        result = safe_log_params({"seed": 123})
        assert "parameters:" in result
        assert "seed" in result

    def test_custom_param_name(self):
        result = safe_log_params({"x": 1}, param_name="config")
        assert "config:" in result

    def test_large_params_truncated(self):
        params = {f"p{i}": "v" * 100 for i in range(50)}
        result = safe_log_params(params)
        # Should not be excessively long
        assert len(result) < 2000


class TestSafeJsonRepr:
    """Tests for safe_json_repr."""

    def test_simple_types(self):
        assert safe_json_repr("hello") == '"hello"'
        assert safe_json_repr(42) == "42"
        assert safe_json_repr(True) == "true"
        assert safe_json_repr(None) == "null"

    def test_small_dict(self):
        result = safe_json_repr({"a": 1})
        parsed = json.loads(result)
        assert parsed == {"a": 1}

    def test_large_output_truncated(self):
        big = {"key": list(range(500))}
        result = safe_json_repr(big, max_length=100)
        assert len(result) <= 110  # 100 + "...}"

    def test_non_serializable_fallback(self):
        """Non-JSON-serializable objects should fall back to safe_repr."""
        class Custom:
            pass
        result = safe_json_repr(Custom())
        assert isinstance(result, str)

    def test_default_max_length(self):
        big = {"data": list(range(10000))}
        result = safe_json_repr(big)
        assert len(result) <= LOG_MAX_JSON_OUTPUT + 10


class TestSafeLogChange:
    """Tests for safe_log_change."""

    def test_simple_values(self):
        result = safe_log_change("seed", 123, 456)
        assert "seed:" in result
        assert "123" in result
        assert "456" in result
        # Unicode arrow
        assert "\u2192" in result

    def test_dict_old_value(self):
        result = safe_log_change("config", {"a": 1, "b": 2}, "new")
        assert "<dict with 2 keys>" in result

    def test_dict_new_value(self):
        result = safe_log_change("config", "old", {"x": 1})
        assert "<dict with 1 keys>" in result

    def test_not_set_special_case(self):
        result = safe_log_change("param", "NOT_SET", "value")
        assert "NOT_SET" in result

    def test_custom_max_length(self):
        result = safe_log_change("p", "a" * 500, "b" * 500, max_length=50)
        # Both values should be truncated
        assert len(result) < 300


class TestSafeComponentLogger:
    """Tests for the SafeComponentLogger class."""

    def test_inherits_component_logger(self):
        from source.core.log.core import ComponentLogger
        logger = SafeComponentLogger("TEST")
        assert isinstance(logger, ComponentLogger)

    def test_debug_truncates_long_messages(self):
        logger = SafeComponentLogger("TEST")
        long_msg = "x" * (LOG_MAX_DEBUG_MESSAGE + 1000)
        with patch("source.core.log.core._original_debug") as mock_debug:
            with patch("source.core.log.core._debug_mode", True):
                logger.debug(long_msg)
                called_msg = mock_debug.call_args[0][1]
                assert len(called_msg) <= LOG_MAX_DEBUG_MESSAGE + 50  # truncation message

    def test_debug_short_message_unchanged(self):
        logger = SafeComponentLogger("TEST")
        with patch("source.core.log.core._original_debug") as mock_debug:
            with patch("source.core.log.core._debug_mode", True):
                logger.debug("short msg")
                called_msg = mock_debug.call_args[0][1]
                assert called_msg == "short msg"

    def test_safe_debug_dict(self):
        logger = SafeComponentLogger("TEST")
        with patch("source.core.log.core._original_debug") as mock_debug:
            with patch("source.core.log.core._debug_mode", True):
                logger.safe_debug_dict("Params", {"seed": 42})
                called_msg = mock_debug.call_args[0][1]
                assert "Params:" in called_msg
                assert "seed" in called_msg

    def test_safe_debug_change(self):
        logger = SafeComponentLogger("TEST")
        with patch("source.core.log.core._original_debug") as mock_debug:
            with patch("source.core.log.core._debug_mode", True):
                logger.safe_debug_change("seed", 100, 200)
                called_msg = mock_debug.call_args[0][1]
                assert "seed:" in called_msg
                assert "100" in called_msg
                assert "200" in called_msg


class TestPreConfiguredSafeLoggers:
    """Tests for pre-configured safe logger instances."""

    def test_headless_logger_safe_exists(self):
        assert headless_logger_safe.component == "HEADLESS"
        assert isinstance(headless_logger_safe, SafeComponentLogger)

    def test_queue_logger_safe_exists(self):
        assert queue_logger_safe.component == "QUEUE"
        assert isinstance(queue_logger_safe, SafeComponentLogger)


class TestConstants:
    """Tests that constants have reasonable values."""

    def test_max_string_repr_positive(self):
        assert LOG_MAX_STRING_REPR > 0

    def test_max_object_output_positive(self):
        assert LOG_MAX_OBJECT_OUTPUT > 0

    def test_max_collection_items_positive(self):
        assert LOG_MAX_COLLECTION_ITEMS > 0

    def test_max_nesting_depth_positive(self):
        assert LOG_MAX_NESTING_DEPTH > 0

    def test_large_dict_keys_is_set(self):
        assert isinstance(LOG_LARGE_DICT_KEYS, set)
        assert "orchestrator_payload" in LOG_LARGE_DICT_KEYS
