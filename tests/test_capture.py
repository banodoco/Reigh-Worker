"""Tests for source/models/wgp/generators/capture.py.

Covers TailBuffer, TeeWriter, CaptureHandler, and run_with_capture.
"""

import io
import logging
import sys
from collections import deque
from unittest.mock import MagicMock

import pytest

from source.models.wgp.generators.capture import (
    CaptureHandler,
    TailBuffer,
    TeeWriter,
    run_with_capture,
)


# ── TailBuffer ──────────────────────────────────────────────────────────────


class TestTailBuffer:
    def test_basic_write_and_getvalue(self):
        buf = TailBuffer(100)
        buf.write("hello")
        assert buf.getvalue() == "hello"

    def test_multiple_writes_concatenate(self):
        buf = TailBuffer(100)
        buf.write("hello ")
        buf.write("world")
        assert buf.getvalue() == "hello world"

    def test_overflow_truncation(self):
        buf = TailBuffer(10)
        buf.write("abcdefghij")  # exactly 10
        assert buf.getvalue() == "abcdefghij"
        buf.write("XYZ")  # total 13, should keep last 10
        assert buf.getvalue() == "defghijXYZ"
        assert len(buf.getvalue()) == 10

    def test_large_single_write_truncates(self):
        buf = TailBuffer(5)
        buf.write("abcdefghij")
        assert buf.getvalue() == "fghij"

    def test_empty_write_is_noop(self):
        buf = TailBuffer(100)
        buf.write("")
        assert buf.getvalue() == ""

    def test_none_write_is_noop(self):
        """None is falsy, so write should skip it."""
        buf = TailBuffer(100)
        buf.write(None)
        assert buf.getvalue() == ""

    def test_getvalue_on_fresh_buffer(self):
        buf = TailBuffer(100)
        assert buf.getvalue() == ""

    def test_non_string_coerced_via_str(self):
        """Non-string input is coerced via str() inside write."""
        buf = TailBuffer(100)
        buf.write(42)
        assert buf.getvalue() == "42"


# ── TeeWriter ───────────────────────────────────────────────────────────────


class TestTeeWriter:
    def _make_pair(self):
        original = io.StringIO()
        capture = TailBuffer(10_000)
        return original, capture, TeeWriter(original, capture)

    def test_write_goes_to_both(self):
        original, capture, tee = self._make_pair()
        tee.write("hello")
        assert original.getvalue() == "hello"
        assert capture.getvalue() == "hello"

    def test_writelines(self):
        original, capture, tee = self._make_pair()
        tee.writelines(["line1\n", "line2\n"])
        assert original.getvalue() == "line1\nline2\n"
        assert capture.getvalue() == "line1\nline2\n"

    def test_flush_does_not_raise(self):
        original, capture, tee = self._make_pair()
        tee.flush()  # Should not raise

    def test_isatty_proxied(self):
        mock_original = MagicMock()
        mock_original.isatty.return_value = True
        capture = TailBuffer(1000)
        tee = TeeWriter(mock_original, capture)
        assert tee.isatty() is True
        mock_original.isatty.return_value = False
        assert tee.isatty() is False

    def test_encoding_property(self):
        mock_original = MagicMock()
        mock_original.encoding = "utf-8"
        capture = TailBuffer(1000)
        tee = TeeWriter(mock_original, capture)
        assert tee.encoding == "utf-8"

    def test_encoding_missing_returns_none(self):
        """If original has no encoding attr, property returns None."""
        original = object()  # no encoding attribute
        capture = TailBuffer(1000)
        tee = TeeWriter(original, capture)
        assert tee.encoding is None

    def test_getattr_proxies_to_original(self):
        mock_original = MagicMock()
        mock_original.custom_attr = "custom_value"
        capture = TailBuffer(1000)
        tee = TeeWriter(mock_original, capture)
        assert tee.custom_attr == "custom_value"

    def test_write_tolerates_broken_original(self):
        """If original.write raises OSError, capture still gets the text."""
        broken = MagicMock()
        broken.write.side_effect = OSError("broken pipe")
        capture = TailBuffer(1000)
        tee = TeeWriter(broken, capture)
        tee.write("still captured")
        assert capture.getvalue() == "still captured"


# ── CaptureHandler ──────────────────────────────────────────────────────────


class TestCaptureHandler:
    def _make_handler(self):
        log_deque = deque(maxlen=100)
        handler = CaptureHandler(log_deque)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        return log_deque, handler

    def test_records_stored(self):
        log_deque, handler = self._make_handler()
        logger = logging.getLogger("test_capture_handler_stored")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("hello from test")
        finally:
            logger.removeHandler(handler)

        assert len(log_deque) == 1
        assert log_deque[0]["level"] == "INFO"
        assert log_deque[0]["name"] == "test_capture_handler_stored"
        assert "hello from test" in log_deque[0]["message"]

    def test_duplicates_deduplicated(self):
        log_deque, handler = self._make_handler()
        logger = logging.getLogger("test_capture_handler_dedup")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            for _ in range(5):
                logger.warning("same message")
        finally:
            logger.removeHandler(handler)

        # Only the first occurrence should be stored
        assert len(log_deque) == 1

    def test_different_messages_not_deduplicated(self):
        log_deque, handler = self._make_handler()
        logger = logging.getLogger("test_capture_handler_diff")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("message A")
            logger.info("message B")
        finally:
            logger.removeHandler(handler)

        assert len(log_deque) == 2


# ── run_with_capture ────────────────────────────────────────────────────────


class TestRunWithCapture:
    def test_captures_stdout(self):
        def fn():
            print("hello stdout")

        result, stdout_buf, stderr_buf, logs = run_with_capture(fn)
        assert "hello stdout" in stdout_buf.getvalue()

    def test_captures_stderr(self):
        def fn():
            print("hello stderr", file=sys.stderr)

        result, stdout_buf, stderr_buf, logs = run_with_capture(fn)
        assert "hello stderr" in stderr_buf.getvalue()

    def test_captures_logging(self):
        def fn():
            logging.getLogger("test_rwc_logging").warning("log warning msg")

        result, stdout_buf, stderr_buf, logs = run_with_capture(fn)
        log_messages = [entry["message"] for entry in logs]
        assert any("log warning msg" in m for m in log_messages)

    def test_returns_function_result(self):
        def fn():
            return 42

        result, stdout_buf, stderr_buf, logs = run_with_capture(fn)
        assert result == 42

    def test_passes_kwargs_to_function(self):
        def fn(x=0, y=0):
            return x + y

        result, _, _, _ = run_with_capture(fn, x=10, y=20)
        assert result == 30

    def test_exception_reraised(self):
        def fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_with_capture(fn)

    def test_exception_has_attached_captures(self):
        def fn():
            print("pre-crash output")
            print("pre-crash stderr", file=sys.stderr)
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError) as exc_info:
            run_with_capture(fn)

        exc = exc_info.value
        assert hasattr(exc, "__captured_stdout__")
        assert hasattr(exc, "__captured_stderr__")
        assert hasattr(exc, "__captured_logs__")
        assert "pre-crash output" in exc.__captured_stdout__.getvalue()
        assert "pre-crash stderr" in exc.__captured_stderr__.getvalue()

    def test_stdout_stderr_restored_after_success(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        run_with_capture(lambda: None)

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_stdout_stderr_restored_after_exception(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with pytest.raises(RuntimeError):
            run_with_capture(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
