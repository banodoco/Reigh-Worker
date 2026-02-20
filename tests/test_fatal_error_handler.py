"""Tests for source/task_handlers/worker/fatal_error_handler.py."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from source.task_handlers.worker.fatal_error_handler import (
    FatalWorkerError,
    is_fatal_error,
    is_retryable_error,
    reset_fatal_error_counter,
    handle_fatal_error_in_worker,
    is_running_as_worker,
    check_and_handle_fatal_error,
    _mark_worker_for_termination,
    FATAL_ERROR_PATTERNS,
    RETRYABLE_ERROR_PATTERNS,
)

# We need to reset global state between tests
import source.task_handlers.worker.fatal_error_handler as feh_module


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module-level global counters before each test."""
    feh_module._consecutive_fatal_errors = 0
    feh_module._last_fatal_category = None
    yield
    feh_module._consecutive_fatal_errors = 0
    feh_module._last_fatal_category = None


# ---------------------------------------------------------------------------
# FatalWorkerError
# ---------------------------------------------------------------------------

class TestFatalWorkerError:
    """Tests for FatalWorkerError exception class."""

    def test_basic_creation(self):
        err = FatalWorkerError("GPU failed")
        assert str(err) == "GPU failed"
        assert err.original_error is None
        assert err.error_category == "unknown"

    def test_with_original_error(self):
        original = RuntimeError("CUDA error")
        err = FatalWorkerError("GPU failed", original_error=original, error_category="cuda_driver")
        assert err.original_error is original
        assert err.error_category == "cuda_driver"

    def test_is_exception(self):
        err = FatalWorkerError("test")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# is_fatal_error
# ---------------------------------------------------------------------------

class TestIsFatalError:
    """Tests for is_fatal_error."""

    def test_empty_message_not_fatal(self):
        is_fatal, category, threshold = is_fatal_error("")
        assert is_fatal is False
        assert category is None

    def test_none_message_not_fatal(self):
        is_fatal, category, threshold = is_fatal_error(None)
        assert is_fatal is False

    def test_normal_error_not_fatal(self):
        is_fatal, category, threshold = is_fatal_error("ValueError: invalid input")
        assert is_fatal is False

    def test_cuda_driver_init_failure(self):
        msg = "CUDA driver initialization failed, you might not have a CUDA gpu"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is True
        assert category == "cuda_driver"
        assert threshold == 2

    def test_no_cuda_device_detected(self):
        msg = "no CUDA-capable device is detected"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is True
        assert category == "cuda_driver"

    def test_gpu_fallen_off_bus(self):
        msg = "GPU has fallen off the bus"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is True
        assert category == "cuda_hardware"
        assert threshold == 1

    def test_nvml_not_found(self):
        msg = "NVML Shared Library Not Found"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is True
        assert category == "nvml"

    def test_segfault(self):
        msg = "Segmentation fault (core dumped)"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is True
        assert category == "system_critical"
        assert threshold == 1

    def test_bus_error(self):
        msg = "Bus error (signal)"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is True
        assert category == "system_critical"

    def test_oom_not_fatal(self):
        """OOM errors should NOT be fatal (removed from fatal patterns)."""
        msg = "CUDA out of memory. Tried to allocate 2.00 GiB"
        is_fatal, category, threshold = is_fatal_error(msg)
        assert is_fatal is False

    def test_exception_type_cuda_driver(self):
        """Exception with torch.cuda type and 'driver' in message should be fatal."""
        exc = MagicMock()
        type(exc).__module__ = "torch.cuda"
        type(exc).__qualname__ = "CudaError"
        # Make str(type(exception)) contain torch.cuda
        exc.__class__ = type("CudaError", (RuntimeError,), {"__module__": "torch.cuda"})
        msg = "CUDA driver initialization failed"
        is_fatal, category, threshold = is_fatal_error(msg, exception=exc)
        assert is_fatal is True

    def test_memory_error_cannot_allocate(self):
        """MemoryError with 'cannot allocate memory' should be fatal."""
        exc = MemoryError("cannot allocate memory")
        msg = "cannot allocate memory"
        is_fatal, category, threshold = is_fatal_error(msg, exception=exc)
        assert is_fatal is True
        assert category == "system_critical"


# ---------------------------------------------------------------------------
# is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:
    """Tests for is_retryable_error."""

    def test_empty_message_not_retryable(self):
        is_retry, category, max_attempts = is_retryable_error("")
        assert is_retry is False
        assert category is None
        assert max_attempts == 0

    def test_generation_no_output(self):
        msg = "No output generated from model"
        is_retry, category, max_attempts = is_retryable_error(msg)
        assert is_retry is True
        assert category == "generation_no_output"
        assert max_attempts == 2

    def test_edge_function_500(self):
        msg = "[EDGE_FAIL:some_func:HTTP_500] Internal server error"
        is_retry, category, max_attempts = is_retryable_error(msg)
        assert is_retry is True
        assert category == "edge_function_transient"
        assert max_attempts == 3

    def test_connection_error(self):
        msg = "ConnectionError: Failed to connect to server"
        is_retry, category, max_attempts = is_retryable_error(msg)
        assert is_retry is True
        assert category == "network_transient"

    def test_timeout_retryable(self):
        msg = "Request timed out after 30s"
        is_retry, category, max_attempts = is_retryable_error(msg)
        assert is_retry is True
        assert category == "network_transient"

    def test_normal_error_not_retryable(self):
        msg = "ValueError: invalid parameter"
        is_retry, category, max_attempts = is_retryable_error(msg)
        assert is_retry is False


# ---------------------------------------------------------------------------
# reset_fatal_error_counter
# ---------------------------------------------------------------------------

class TestResetFatalErrorCounter:
    """Tests for reset_fatal_error_counter."""

    def test_resets_counter(self):
        feh_module._consecutive_fatal_errors = 5
        feh_module._last_fatal_category = "cuda_driver"
        reset_fatal_error_counter()
        assert feh_module._consecutive_fatal_errors == 0
        assert feh_module._last_fatal_category is None


# ---------------------------------------------------------------------------
# handle_fatal_error_in_worker
# ---------------------------------------------------------------------------

class TestHandleFatalErrorInWorker:
    """Tests for handle_fatal_error_in_worker."""

    def test_non_fatal_error_returns_immediately(self):
        """Non-fatal error should not raise or increment counter."""
        handle_fatal_error_in_worker("normal ValueError")
        assert feh_module._consecutive_fatal_errors == 0

    def test_single_cuda_driver_below_threshold(self):
        """One cuda_driver error (threshold=2) should warn but not raise."""
        logger = MagicMock()
        handle_fatal_error_in_worker(
            "CUDA driver initialization failed, you might not have a CUDA gpu",
            logger=logger,
        )
        assert feh_module._consecutive_fatal_errors == 1
        logger.warning.assert_called()

    def test_cuda_driver_at_threshold_raises(self):
        """Two consecutive cuda_driver errors should raise FatalWorkerError."""
        msg = "CUDA driver initialization failed, you might not have a CUDA gpu"
        logger = MagicMock()

        # First error - below threshold
        handle_fatal_error_in_worker(msg, logger=logger)

        # Second error - meets threshold
        with pytest.raises(FatalWorkerError, match="cuda_driver"):
            handle_fatal_error_in_worker(msg, logger=logger)

    def test_hardware_error_immediate_kill(self):
        """cuda_hardware errors (threshold=1) should raise immediately."""
        msg = "GPU has fallen off the bus"
        with pytest.raises(FatalWorkerError, match="cuda_hardware"):
            handle_fatal_error_in_worker(msg)

    def test_category_change_resets_counter(self):
        """Changing error category should reset the counter."""
        logger = MagicMock()
        # First: cuda_driver (threshold=2, count now 1)
        handle_fatal_error_in_worker(
            "CUDA driver initialization failed, you might not have a CUDA gpu",
            logger=logger,
        )
        assert feh_module._consecutive_fatal_errors == 1

        # Switch to nvml (threshold=2) - counter resets then increments to 1
        handle_fatal_error_in_worker(
            "NVML Shared Library Not Found",
            logger=logger,
        )
        assert feh_module._consecutive_fatal_errors == 1
        assert feh_module._last_fatal_category == "nvml"

    def test_mark_worker_for_termination_called(self):
        """When threshold reached, should attempt to mark worker for termination."""
        msg = "GPU has fallen off the bus"
        logger = MagicMock()

        with patch("source.task_handlers.worker.fatal_error_handler._mark_worker_for_termination",
                   return_value=True) as mock_mark:
            with pytest.raises(FatalWorkerError):
                handle_fatal_error_in_worker(
                    msg, logger=logger, worker_id="w-123", task_id="t-456"
                )

        mock_mark.assert_called_once()
        assert mock_mark.call_args[1]["worker_id"] == "w-123"
        assert mock_mark.call_args[1]["task_id"] == "t-456"


# ---------------------------------------------------------------------------
# is_running_as_worker
# ---------------------------------------------------------------------------

class TestIsRunningAsWorker:
    """Tests for is_running_as_worker."""

    def test_env_var_set(self):
        with patch.dict(os.environ, {"WAN2GP_WORKER_MODE": "true"}):
            assert is_running_as_worker() is True

    def test_env_var_not_set(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["test_runner.py"]):
            assert is_running_as_worker() is False

    def test_worker_flag_in_argv(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["main.py", "--worker"]):
            assert is_running_as_worker() is True

    def test_worker_short_flag(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["main.py", "-w"]):
            assert is_running_as_worker() is True

    def test_worker_py_in_argv0(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["worker.py"]):
            assert is_running_as_worker() is True


# ---------------------------------------------------------------------------
# check_and_handle_fatal_error
# ---------------------------------------------------------------------------

class TestCheckAndHandleFatalError:
    """Tests for check_and_handle_fatal_error."""

    def test_non_worker_does_nothing(self):
        """Not running as worker should return without any action."""
        with patch("source.task_handlers.worker.fatal_error_handler.is_running_as_worker", return_value=False):
            # Even a fatal error message should not raise
            check_and_handle_fatal_error("GPU has fallen off the bus")
        # No exception raised

    def test_worker_mode_delegates(self):
        """Worker mode should delegate to handle_fatal_error_in_worker."""
        with patch("source.task_handlers.worker.fatal_error_handler.is_running_as_worker", return_value=True), \
             patch("source.task_handlers.worker.fatal_error_handler.handle_fatal_error_in_worker") as mock_handler:
            check_and_handle_fatal_error("some error", worker_id="w1", task_id="t1")
        mock_handler.assert_called_once()


# ---------------------------------------------------------------------------
# _mark_worker_for_termination
# ---------------------------------------------------------------------------

class TestMarkWorkerForTermination:
    """Tests for _mark_worker_for_termination."""

    def test_no_supabase_credentials_returns_false(self):
        """Missing Supabase credentials should return False."""
        with patch.dict(os.environ, {}, clear=True):
            result = _mark_worker_for_termination("w-1", "t-1", "fatal error")
        assert result is False

    def test_successful_marking(self):
        """Successful DB update should return True."""
        mock_supabase = MagicMock()
        mock_worker_resp = MagicMock()
        mock_worker_resp.data = {"id": "w-1", "metadata": {"runpod_id": "rp-123"}}

        mock_update_resp = MagicMock()
        mock_update_resp.data = [{"id": "w-1"}]

        mock_task_resp = MagicMock()
        mock_task_resp.data = [{"id": "t-1"}]

        # Chain the supabase calls
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_worker_resp
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_resp
        mock_supabase.table.return_value.update.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_task_resp

        mock_supabase_module = MagicMock()
        mock_supabase_module.create_client.return_value = mock_supabase

        with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "key"}), \
             patch.dict("sys.modules", {"supabase": mock_supabase_module}):
            result = _mark_worker_for_termination("w-1", "t-1", "GPU failed")

        assert result is True

    def test_exception_returns_false(self):
        """Any exception during marking should return False."""
        logger = MagicMock()
        mock_supabase_module = MagicMock()
        mock_supabase_module.create_client.side_effect = OSError("network error")

        with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "key"}), \
             patch.dict("sys.modules", {"supabase": mock_supabase_module}):
            result = _mark_worker_for_termination("w-1", "t-1", "error", logger=logger)
        assert result is False
