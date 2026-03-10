"""
Tests for source/core/log/timing.py — LogTimer context manager.

Covers:
  - Successful completion (logs start + success)
  - Exception path (logs start + error)
  - Debug level variant
  - Duration calculation

Run with: python -m pytest tests/test_log_timing.py -v
"""

import datetime
from unittest.mock import patch, MagicMock

_MOD = "source.core.log.timing"


class TestLogTimerSuccess:
    """LogTimer should log start and completed messages on success."""

    @patch(f"{_MOD}.success")
    @patch(f"{_MOD}.essential")
    @patch(f"{_MOD}.datetime")
    def test_success_essential_level(self, mock_dt, mock_essential, mock_success):
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 12, 0, 5)
        mock_dt.datetime.now.side_effect = [t0, t1]

        with LogTimer("COMP", "encode video", task_id="t1"):
            pass

        mock_essential.assert_called_once()
        assert "Starting encode video" in mock_essential.call_args[0][1]

        mock_success.assert_called_once()
        msg = mock_success.call_args[0][1]
        assert "encode video completed" in msg
        assert "5.0s" in msg

    @patch(f"{_MOD}.debug")
    @patch(f"{_MOD}.datetime")
    def test_success_debug_level(self, mock_dt, mock_debug):
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 12, 0, 2)
        mock_dt.datetime.now.side_effect = [t0, t1]

        with LogTimer("C", "minor op", level="debug"):
            pass

        # Should have been called twice: once on enter, once on exit
        assert mock_debug.call_count == 2
        assert "Starting minor op" in mock_debug.call_args_list[0][0][1]
        assert "minor op completed" in mock_debug.call_args_list[1][0][1]
        assert "2.0s" in mock_debug.call_args_list[1][0][1]


class TestLogTimerError:
    """LogTimer should log error when exception occurs inside the block."""

    @patch(f"{_MOD}.error")
    @patch(f"{_MOD}.essential")
    @patch(f"{_MOD}.datetime")
    def test_exception_logs_error(self, mock_dt, mock_essential, mock_error):
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 12, 0, 3)
        mock_dt.datetime.now.side_effect = [t0, t1]

        try:
            with LogTimer("X", "risky op", task_id="t2"):
                raise ValueError("boom")
        except ValueError:
            pass

        mock_error.assert_called_once()
        msg = mock_error.call_args[0][1]
        assert "risky op failed" in msg
        assert "3.0s" in msg
        assert "boom" in msg

    @patch(f"{_MOD}.error")
    @patch(f"{_MOD}.essential")
    @patch(f"{_MOD}.datetime")
    def test_exception_propagates(self, mock_dt, mock_essential, mock_error):
        """The context manager does NOT suppress exceptions."""
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 12, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 12, 0, 1)
        mock_dt.datetime.now.side_effect = [t0, t1]

        raised = False
        try:
            with LogTimer("X", "op"):
                raise RuntimeError("fail")
        except RuntimeError:
            raised = True

        assert raised, "LogTimer should not swallow exceptions"


class TestLogTimerTaskId:
    """LogTimer should forward task_id to all logging calls."""

    @patch(f"{_MOD}.success")
    @patch(f"{_MOD}.essential")
    @patch(f"{_MOD}.datetime")
    def test_task_id_forwarded(self, mock_dt, mock_essential, mock_success):
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 0, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 0, 0, 1)
        mock_dt.datetime.now.side_effect = [t0, t1]

        with LogTimer("C", "work", task_id="my-task-42"):
            pass

        # essential(component, msg, task_id) — task_id is the third arg
        assert mock_essential.call_args[0][2] == "my-task-42"
        assert mock_success.call_args[0][2] == "my-task-42"

    @patch(f"{_MOD}.success")
    @patch(f"{_MOD}.essential")
    @patch(f"{_MOD}.datetime")
    def test_no_task_id(self, mock_dt, mock_essential, mock_success):
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 0, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 0, 0, 1)
        mock_dt.datetime.now.side_effect = [t0, t1]

        with LogTimer("C", "work"):
            pass

        # task_id defaults to None
        assert mock_essential.call_args[0][2] is None


class TestLogTimerReturnsSelf:
    """LogTimer.__enter__ should return self."""

    @patch(f"{_MOD}.essential")
    @patch(f"{_MOD}.success")
    @patch(f"{_MOD}.datetime")
    def test_enter_returns_self(self, mock_dt, mock_success, mock_essential):
        from source.core.log.timing import LogTimer

        t0 = datetime.datetime(2025, 1, 1, 0, 0, 0)
        t1 = datetime.datetime(2025, 1, 1, 0, 0, 0)
        mock_dt.datetime.now.side_effect = [t0, t1]

        with LogTimer("C", "op") as timer:
            assert isinstance(timer, LogTimer)
            assert timer.component == "C"
            assert timer.operation == "op"
