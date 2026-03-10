"""
Tests for source/task_handlers/worker/heartbeat_utils.py

Covers:
  - start_heartbeat_guardian_process: Process creation + return values
  - get_gpu_memory_usage: CUDA available, not available, exception paths

Run with: python -m pytest tests/test_heartbeat_utils.py -v
"""

import sys
from unittest.mock import patch, MagicMock

# heartbeat_utils imports from heartbeat_guardian at module level,
# so we need to pre-seed sys.modules before the import.
_fake_heartbeat_guardian = MagicMock()
_fake_headless_logger = MagicMock()


def _import_module():
    """Import heartbeat_utils with mocked external dependencies."""
    with patch.dict(sys.modules, {
        "heartbeat_guardian": _fake_heartbeat_guardian,
    }):
        # Reload to pick up the mocked modules
        import importlib
        import source.task_handlers.worker.heartbeat_utils as mod
        importlib.reload(mod)
        return mod


class TestStartHeartbeatGuardianProcess:
    """Tests for start_heartbeat_guardian_process."""

    def test_returns_guardian_and_queue(self):
        mod = _import_module()

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch("source.task_handlers.worker.heartbeat_utils.Process", return_value=mock_process) as MockProcess, \
             patch("source.task_handlers.worker.heartbeat_utils.Queue") as MockQueue, \
             patch("source.task_handlers.worker.heartbeat_utils.headless_logger") as mock_logger:

            mock_queue_instance = MagicMock()
            MockQueue.return_value = mock_queue_instance

            guardian, log_queue = mod.start_heartbeat_guardian_process(
                "worker-1", "https://db.example.com", "key123"
            )

        assert guardian is mock_process
        assert log_queue is mock_queue_instance
        mock_process.start.assert_called_once()
        mock_logger.essential.assert_called_once()

    def test_process_created_as_daemon(self):
        mod = _import_module()

        mock_process = MagicMock()
        mock_process.pid = 99

        with patch("source.task_handlers.worker.heartbeat_utils.Process", return_value=mock_process) as MockProcess, \
             patch("source.task_handlers.worker.heartbeat_utils.Queue") as MockQueue, \
             patch("source.task_handlers.worker.heartbeat_utils.headless_logger"):

            MockQueue.return_value = MagicMock()
            mod.start_heartbeat_guardian_process("w1", "url", "key")

        # The Process constructor should receive daemon=True
        call_kwargs = MockProcess.call_args[1]
        assert call_kwargs["daemon"] is True

    def test_queue_max_size(self):
        mod = _import_module()

        mock_process = MagicMock()
        mock_process.pid = 1

        with patch("source.task_handlers.worker.heartbeat_utils.Process", return_value=mock_process), \
             patch("source.task_handlers.worker.heartbeat_utils.Queue") as MockQueue, \
             patch("source.task_handlers.worker.heartbeat_utils.headless_logger"):

            MockQueue.return_value = MagicMock()
            mod.start_heartbeat_guardian_process("w1", "url", "key")

        MockQueue.assert_called_once_with(maxsize=mod.HEARTBEAT_LOG_QUEUE_MAX_SIZE)


class TestGetGpuMemoryUsage:
    """Tests for get_gpu_memory_usage."""

    def test_cuda_available(self):
        mod = _import_module()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # total_memory = 8 GB in bytes
        props = MagicMock()
        props.total_memory = 8 * (1024 ** 2) * (1024 ** 2 // (1024 ** 2))  # simplified
        props.total_memory = 8192 * (1024 ** 2)  # 8192 MB in bytes
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_allocated.return_value = 2048 * (1024 ** 2)  # 2048 MB

        with patch.dict(sys.modules, {"torch": mock_torch}):
            total, allocated = mod.get_gpu_memory_usage()

        assert total == 8192
        assert allocated == 2048

    def test_cuda_not_available(self):
        mod = _import_module()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            total, allocated = mod.get_gpu_memory_usage()

        assert total is None
        assert allocated is None

    def test_runtime_error_returns_none(self):
        mod = _import_module()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("driver not found")

        with patch.dict(sys.modules, {"torch": mock_torch}), \
             patch("source.task_handlers.worker.heartbeat_utils.headless_logger") as mock_logger:
            total, allocated = mod.get_gpu_memory_usage()

        assert total is None
        assert allocated is None
        mock_logger.debug.assert_called_once()

    def test_os_error_returns_none(self):
        mod = _import_module()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.side_effect = OSError("device error")

        with patch.dict(sys.modules, {"torch": mock_torch}), \
             patch("source.task_handlers.worker.heartbeat_utils.headless_logger") as mock_logger:
            total, allocated = mod.get_gpu_memory_usage()

        assert total is None
        assert allocated is None
