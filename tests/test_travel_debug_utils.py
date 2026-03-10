"""Tests for source/task_handlers/travel/debug_utils.py."""

import os
from unittest import mock

import pytest

from source.task_handlers.travel.debug_utils import log_ram_usage


class TestLogRamUsage:
    """Tests for log_ram_usage."""

    def test_returns_dict(self):
        """log_ram_usage should always return a dict."""
        result = log_ram_usage("test_label")
        assert isinstance(result, dict)

    def test_has_available_key(self):
        """Result always has an 'available' key."""
        result = log_ram_usage("test_label")
        assert "available" in result

    def test_with_psutil_available(self):
        """If psutil is available, result should have RAM metrics."""
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False

        result = log_ram_usage("test_label")

        if psutil_available:
            assert result["available"] is True
            assert "process_rss_mb" in result
            assert "process_rss_gb" in result
            assert "system_total_gb" in result
            assert "system_available_gb" in result
            assert "system_used_percent" in result
            assert result["process_rss_mb"] > 0
            assert result["system_total_gb"] > 0
        else:
            assert result["available"] is False

    def test_with_psutil_unavailable(self):
        """When _PSUTIL_AVAILABLE is False, returns unavailable."""
        import source.task_handlers.travel.debug_utils as mod
        original = mod._PSUTIL_AVAILABLE
        try:
            mod._PSUTIL_AVAILABLE = False
            result = log_ram_usage("test_label")
            assert result == {"available": False}
        finally:
            mod._PSUTIL_AVAILABLE = original

    def test_custom_logger(self):
        """Accepts a custom logger parameter."""
        mock_logger = mock.MagicMock()
        result = log_ram_usage("test_label", logger=mock_logger)
        assert isinstance(result, dict)

    def test_custom_task_id(self):
        """Accepts a task_id parameter."""
        result = log_ram_usage("test_label", task_id="task-123")
        assert isinstance(result, dict)
