"""Tests for source/core/log/database.py."""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from source.core.log.database import LogBuffer, WorkerDatabaseLogHandler, CustomLogInterceptor


class TestLogBuffer:
    """Tests for the LogBuffer thread-safe log collection."""

    def test_add_single_entry(self):
        buf = LogBuffer(max_size=100)
        result = buf.add("INFO", "test message")
        assert result == []  # Not full yet
        assert buf.total_logs == 1

    def test_add_with_task_id(self):
        buf = LogBuffer(max_size=100)
        buf.add("ERROR", "error msg", task_id="task-42")
        logs = buf.flush()
        assert len(logs) == 1
        assert logs[0]["task_id"] == "task-42"
        assert logs[0]["level"] == "ERROR"
        assert logs[0]["message"] == "error msg"

    def test_add_with_metadata(self):
        buf = LogBuffer(max_size=100)
        buf.add("DEBUG", "debug msg", metadata={"module": "test"})
        logs = buf.flush()
        assert logs[0]["metadata"] == {"module": "test"}

    def test_add_default_metadata_empty_dict(self):
        buf = LogBuffer(max_size=100)
        buf.add("INFO", "msg")
        logs = buf.flush()
        assert logs[0]["metadata"] == {}

    def test_auto_flush_on_max_size(self):
        buf = LogBuffer(max_size=3)
        buf.add("INFO", "msg1")
        buf.add("INFO", "msg2")
        result = buf.add("INFO", "msg3")  # Triggers auto-flush
        assert len(result) == 3
        assert buf.total_flushes == 1

    def test_flush_returns_and_clears(self):
        buf = LogBuffer(max_size=100)
        buf.add("INFO", "a")
        buf.add("INFO", "b")
        logs = buf.flush()
        assert len(logs) == 2
        # Buffer should be empty now
        logs2 = buf.flush()
        assert len(logs2) == 0

    def test_flush_empty_buffer_no_flush_count(self):
        buf = LogBuffer(max_size=100)
        buf.flush()
        assert buf.total_flushes == 0

    def test_get_stats(self):
        buf = LogBuffer(max_size=100)
        buf.add("INFO", "a")
        buf.add("INFO", "b")
        stats = buf.get_stats()
        assert stats["current_buffer_size"] == 2
        assert stats["total_logs_buffered"] == 2
        assert stats["total_flushes"] == 0

    def test_get_stats_after_flush(self):
        buf = LogBuffer(max_size=100)
        buf.add("INFO", "a")
        buf.flush()
        stats = buf.get_stats()
        assert stats["current_buffer_size"] == 0
        assert stats["total_logs_buffered"] == 1
        assert stats["total_flushes"] == 1

    def test_shared_queue_receives_entries(self):
        mock_queue = MagicMock()
        buf = LogBuffer(max_size=100, shared_queue=mock_queue)
        buf.add("INFO", "queued msg")
        mock_queue.put_nowait.assert_called_once()
        entry = mock_queue.put_nowait.call_args[0][0]
        assert entry["message"] == "queued msg"

    def test_shared_queue_error_ignored(self):
        mock_queue = MagicMock()
        mock_queue.put_nowait.side_effect = OSError("queue broken")
        buf = LogBuffer(max_size=100, shared_queue=mock_queue)
        # Should not raise
        buf.add("INFO", "msg")
        assert buf.total_logs == 1

    def test_shared_queue_value_error_ignored(self):
        mock_queue = MagicMock()
        mock_queue.put_nowait.side_effect = ValueError("queue closed")
        buf = LogBuffer(max_size=100, shared_queue=mock_queue)
        buf.add("INFO", "msg")
        assert buf.total_logs == 1

    def test_log_entry_has_timestamp(self):
        buf = LogBuffer(max_size=100)
        buf.add("INFO", "msg")
        logs = buf.flush()
        assert "timestamp" in logs[0]
        # ISO format check
        assert "T" in logs[0]["timestamp"]

    def test_thread_safety(self):
        """Multiple threads adding concurrently should not lose logs."""
        buf = LogBuffer(max_size=10000)
        num_threads = 10
        logs_per_thread = 100

        def add_logs():
            for i in range(logs_per_thread):
                buf.add("INFO", f"msg-{i}")

        threads = [threading.Thread(target=add_logs) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert buf.total_logs == num_threads * logs_per_thread


class TestWorkerDatabaseLogHandler:
    """Tests for the logging.Handler subclass."""

    def test_handler_is_logging_handler(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        assert isinstance(handler, logging.Handler)

    def test_set_current_task(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        handler.set_current_task("task-abc")
        assert handler.current_task_id == "task-abc"

    def test_set_current_task_none(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        handler.set_current_task("task-abc")
        handler.set_current_task(None)
        assert handler.current_task_id is None

    def test_emit_adds_to_buffer(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="test.py", lineno=10,
            msg="hello", args=(), exc_info=None,
        )
        handler.emit(record)
        logs = buf.flush()
        assert len(logs) == 1
        assert logs[0]["message"] == "hello"
        assert logs[0]["level"] == "INFO"

    def test_emit_includes_metadata(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        record = logging.LogRecord(
            name="test", level=logging.WARNING,
            pathname="test.py", lineno=42,
            msg="warning msg", args=(), exc_info=None,
        )
        record.module = "mymodule"
        record.funcName = "myfunc"
        handler.emit(record)
        logs = buf.flush()
        assert logs[0]["metadata"]["module"] == "mymodule"
        assert logs[0]["metadata"]["funcName"] == "myfunc"
        assert logs[0]["metadata"]["lineno"] == 42

    def test_emit_with_task_id(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        handler.set_current_task("task-xyz")
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="err", args=(), exc_info=None,
        )
        handler.emit(record)
        logs = buf.flush()
        assert logs[0]["task_id"] == "task-xyz"

    def test_emit_with_exception_info(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        handler.setFormatter(logging.Formatter())
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="err with exc", args=(), exc_info=exc_info,
        )
        handler.emit(record)
        logs = buf.flush()
        assert "exception" in logs[0]["metadata"]

    def test_emit_handles_error_gracefully(self):
        """If buffer.add raises, emit should handle it via handleError."""
        buf = MagicMock()
        buf.add.side_effect = TypeError("bad type")
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        handler.handleError = MagicMock()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="test.py", lineno=1,
            msg="test", args=(), exc_info=None,
        )
        handler.emit(record)
        handler.handleError.assert_called_once()

    def test_min_level_default(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf)
        assert handler.level == logging.INFO

    def test_min_level_custom(self):
        buf = LogBuffer()
        handler = WorkerDatabaseLogHandler("worker-1", buf, min_level=logging.WARNING)
        assert handler.level == logging.WARNING


class TestCustomLogInterceptor:
    """Tests for the CustomLogInterceptor class."""

    def test_capture_log_adds_to_buffer(self):
        buf = LogBuffer()
        interceptor = CustomLogInterceptor(buf)
        interceptor.capture_log("INFO", "hello")
        logs = buf.flush()
        assert len(logs) == 1
        assert logs[0]["message"] == "hello"
        assert logs[0]["level"] == "INFO"

    def test_set_current_task(self):
        buf = LogBuffer()
        interceptor = CustomLogInterceptor(buf)
        interceptor.set_current_task("task-abc")
        assert interceptor.current_task_id == "task-abc"

    def test_capture_log_uses_current_task_when_none_provided(self):
        buf = LogBuffer()
        interceptor = CustomLogInterceptor(buf)
        interceptor.set_current_task("task-default")
        interceptor.capture_log("WARNING", "test msg")
        logs = buf.flush()
        assert logs[0]["task_id"] == "task-default"

    def test_capture_log_explicit_task_overrides_current(self):
        buf = LogBuffer()
        interceptor = CustomLogInterceptor(buf)
        interceptor.set_current_task("task-default")
        interceptor.capture_log("ERROR", "test msg", task_id="task-override")
        logs = buf.flush()
        assert logs[0]["task_id"] == "task-override"

    def test_capture_log_no_task(self):
        buf = LogBuffer()
        interceptor = CustomLogInterceptor(buf)
        interceptor.capture_log("DEBUG", "msg")
        logs = buf.flush()
        assert logs[0]["task_id"] is None

    def test_initial_state(self):
        buf = LogBuffer()
        interceptor = CustomLogInterceptor(buf)
        assert interceptor.current_task_id is None
        assert interceptor.original_print is None
        assert interceptor.log_buffer is buf
