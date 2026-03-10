"""Tests for source/core/log/core.py."""

import sys
import threading
from io import StringIO
from unittest.mock import patch, MagicMock, mock_open

import pytest

import source.core.log.core as log_core


@pytest.fixture(autouse=True)
def reset_log_state():
    """Reset module-level globals between tests."""
    original_debug = log_core._debug_mode
    original_file = log_core._log_file
    original_lock = log_core._log_file_lock
    original_interceptor = log_core._log_interceptor
    yield
    log_core._debug_mode = original_debug
    log_core._log_file = original_file
    log_core._log_file_lock = original_lock
    log_core._log_interceptor = original_interceptor


class TestDebugMode:
    """Tests for debug mode enable/disable/query."""

    def test_debug_disabled_by_default(self):
        log_core._debug_mode = False
        assert log_core.is_debug_enabled() is False

    def test_enable_debug_mode(self):
        log_core._debug_mode = False
        log_core.enable_debug_mode()
        assert log_core.is_debug_enabled() is True

    def test_disable_debug_mode(self):
        log_core.enable_debug_mode()
        log_core.disable_debug_mode()
        assert log_core.is_debug_enabled() is False


class TestFormatMessage:
    """Tests for internal _format_message helper."""

    def test_format_without_task_id(self):
        msg = log_core._format_message("INFO", "TEST", "hello world")
        # Should contain the parts but with a timestamp
        assert "INFO" in msg
        assert "TEST" in msg
        assert "hello world" in msg
        assert "Task" not in msg

    def test_format_with_task_id(self):
        msg = log_core._format_message("ERROR", "COMP", "oops", task_id="abc-123")
        assert "ERROR" in msg
        assert "COMP" in msg
        assert "[Task abc-123]" in msg
        assert "oops" in msg

    def test_format_contains_timestamp(self):
        msg = log_core._format_message("INFO", "X", "y")
        # Timestamp format is [HH:MM:SS]
        assert msg.startswith("[")
        assert "]" in msg


class TestLoggingFunctions:
    """Tests for essential, success, warning, error, critical, debug, progress, status."""

    def test_essential_prints_to_stdout(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core._original_essential("COMP", "test message")
        captured = capsys.readouterr()
        assert "INFO" in captured.out
        assert "COMP" in captured.out
        assert "test message" in captured.out

    def test_error_prints_to_stderr(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core._original_error("COMP", "error msg")
        captured = capsys.readouterr()
        assert "error msg" in captured.err

    def test_critical_prints_to_stderr(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core.critical("COMP", "critical msg")
        captured = capsys.readouterr()
        assert "CRITICAL" in captured.err
        assert "critical msg" in captured.err

    def test_debug_suppressed_when_disabled(self, capsys):
        log_core._debug_mode = False
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core._original_debug("COMP", "hidden debug")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_debug_shown_when_enabled(self, capsys):
        log_core._debug_mode = True
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core._original_debug("COMP", "visible debug")
        captured = capsys.readouterr()
        assert "visible debug" in captured.out

    def test_success_prints(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core._original_success("COMP", "done")
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_progress_prints(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core.progress("COMP", "in progress")
        captured = capsys.readouterr()
        assert "in progress" in captured.out

    def test_status_prints(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core.status("COMP", "status update")
        captured = capsys.readouterr()
        assert "status update" in captured.out


class TestSetLogFile:
    """Tests for set_log_file and _write_to_log_file."""

    def test_set_log_file_creates_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        log_core._log_interceptor = None
        log_core.set_log_file(str(log_path))
        assert log_core._log_file is not None
        assert log_core._log_file_lock is not None
        # Clean up
        log_core._log_file.close()

    def test_write_to_log_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        log_core._log_interceptor = None
        log_core.set_log_file(str(log_path))
        log_core._write_to_log_file("test message")
        log_core._log_file.close()
        content = log_path.read_text()
        assert "test message" in content

    def test_set_log_file_oserror_handled(self, capsys):
        """When the path is invalid, error should be logged, not raised."""
        log_core._log_interceptor = None
        # Use a path that's definitely invalid
        with patch("builtins.open", side_effect=OSError("disk full")):
            log_core.set_log_file("/fake/path/test.log")
        # Should not have set the log file
        # The error function would print to stderr
        captured = capsys.readouterr()
        # The error message prints to stderr via the error() function
        assert log_core._log_file is None or "disk full" in captured.err

    def test_write_to_log_file_when_disabled(self):
        """Writing when no log file set should be a no-op."""
        log_core._log_file = None
        log_core._log_file_lock = None
        # Should not raise
        log_core._write_to_log_file("test")


class TestComponentLogger:
    """Tests for the ComponentLogger class."""

    def test_component_name_stored(self):
        logger = log_core.ComponentLogger("MY_COMPONENT")
        assert logger.component == "MY_COMPONENT"

    def test_essential_delegates(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        logger = log_core.ComponentLogger("TEST")
        logger.essential("hello")
        captured = capsys.readouterr()
        assert "TEST" in captured.out
        assert "hello" in captured.out

    def test_info_aliases_essential(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        logger = log_core.ComponentLogger("TEST")
        logger.info("info message")
        captured = capsys.readouterr()
        assert "info message" in captured.out

    def test_error_delegates(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        logger = log_core.ComponentLogger("TEST")
        logger.error("oops")
        captured = capsys.readouterr()
        assert "oops" in captured.err

    def test_debug_respects_debug_mode(self, capsys):
        log_core._log_file = None
        log_core._log_interceptor = None
        log_core._debug_mode = False
        logger = log_core.ComponentLogger("TEST")
        logger.debug("hidden")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestPreConfiguredLoggers:
    """Tests for pre-configured logger instances."""

    def test_headless_logger_exists(self):
        assert log_core.headless_logger.component == "HEADLESS"

    def test_queue_logger_exists(self):
        assert log_core.queue_logger.component == "QUEUE"

    def test_orchestrator_logger_exists(self):
        assert log_core.orchestrator_logger.component == "ORCHESTRATOR"

    def test_travel_logger_exists(self):
        assert log_core.travel_logger.component == "TRAVEL"

    def test_generation_logger_exists(self):
        assert log_core.generation_logger.component == "GENERATION"

    def test_model_logger_exists(self):
        assert log_core.model_logger.component == "MODEL"

    def test_task_logger_exists(self):
        assert log_core.task_logger.component == "TASK"


class TestInterceptor:
    """Tests for log interceptor integration."""

    def test_set_log_interceptor(self):
        mock_interceptor = MagicMock()
        log_core.set_log_interceptor(mock_interceptor)
        assert log_core._log_interceptor is mock_interceptor

    def test_set_log_interceptor_none(self):
        log_core._log_interceptor = MagicMock()
        log_core.set_log_interceptor(None)
        assert log_core._log_interceptor is None

    def test_set_current_task_context_delegates(self):
        mock_interceptor = MagicMock()
        log_core._log_interceptor = mock_interceptor
        log_core.set_current_task_context("task-123")
        mock_interceptor.set_current_task.assert_called_once_with("task-123")

    def test_set_current_task_context_no_interceptor(self):
        log_core._log_interceptor = None
        # Should not raise
        log_core.set_current_task_context("task-123")

    def test_set_current_task_context_handles_error(self, capsys):
        mock_interceptor = MagicMock()
        mock_interceptor.set_current_task.side_effect = ValueError("bad task")
        log_core._log_interceptor = mock_interceptor
        log_core.set_current_task_context("task-bad")
        captured = capsys.readouterr()
        assert "Failed to set task context" in captured.err

    def test_intercept_log_called_on_essential(self):
        mock_interceptor = MagicMock()
        log_core._log_interceptor = mock_interceptor
        log_core._log_file = None
        log_core.essential("COMP", "msg")
        mock_interceptor.capture_log.assert_called_once_with("INFO", "COMP: msg", None)

    def test_intercept_log_called_on_warning(self):
        mock_interceptor = MagicMock()
        log_core._log_interceptor = mock_interceptor
        log_core._log_file = None
        log_core.warning("COMP", "warn msg")
        mock_interceptor.capture_log.assert_called_once_with("WARNING", "COMP: warn msg", None)

    def test_intercept_log_called_on_error(self):
        mock_interceptor = MagicMock()
        log_core._log_interceptor = mock_interceptor
        log_core._log_file = None
        log_core.error("COMP", "err msg")
        mock_interceptor.capture_log.assert_called_once_with("ERROR", "COMP: err msg", None)

    def test_debug_intercept_only_when_debug_enabled(self):
        mock_interceptor = MagicMock()
        log_core._log_interceptor = mock_interceptor
        log_core._log_file = None

        log_core._debug_mode = False
        log_core.debug("COMP", "hidden")
        mock_interceptor.capture_log.assert_not_called()

        log_core._debug_mode = True
        log_core.debug("COMP", "shown")
        mock_interceptor.capture_log.assert_called_once_with("DEBUG", "COMP: shown", None)
