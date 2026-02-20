"""
Tests for source/task_handlers/queue/memory_cleanup.py — cleanup_memory_after_task.

Covers:
  - CUDA available: cache cleared, gc run, memory logged
  - CUDA not available: gc still runs
  - Significant VRAM freed vs. no significant VRAM freed
  - Exception path: warning logged
  - uni3c cache clearing (import success and ImportError)

Run with: python -m pytest tests/test_memory_cleanup.py -v
"""

import sys
import types
from unittest.mock import patch, MagicMock, call

# We need to patch BYTES_PER_GIB since it's imported at module level
BYTES_PER_GIB = 1024 ** 3


def _make_queue():
    """Create a mock HeadlessTaskQueue."""
    queue = MagicMock()
    queue.logger = MagicMock()
    return queue


class TestCleanupCudaAvailable:
    """Tests when torch.cuda is available."""

    def test_full_cleanup_with_vram_freed(self):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # Before: allocated=4GB, reserved=6GB
        # After:  allocated=3GB, reserved=4GB => freed 2GB
        mock_torch.cuda.memory_allocated.side_effect = [
            4 * BYTES_PER_GIB,  # before
            3 * BYTES_PER_GIB,  # after
        ]
        mock_torch.cuda.memory_reserved.side_effect = [
            6 * BYTES_PER_GIB,  # before
            4 * BYTES_PER_GIB,  # after
        ]

        mock_gc = MagicMock()
        mock_gc.collect.return_value = 42

        queue = _make_queue()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
            "Wan2GP.models.wan.uni3c": MagicMock(),
        }):
            cleanup_memory_after_task(queue, "task-1")

        # Should have called empty_cache
        mock_torch.cuda.empty_cache.assert_called_once()

        # Should have called gc.collect
        mock_gc.collect.assert_called_once()

        # Should have logged "Freed" message (2GB > 0.01)
        info_calls = [str(c) for c in queue.logger.info.call_args_list]
        freed_calls = [c for c in info_calls if "Freed" in c]
        assert len(freed_calls) >= 1

    def test_no_significant_vram_freed(self):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # Before and after nearly the same reserved
        mock_torch.cuda.memory_allocated.side_effect = [
            4 * BYTES_PER_GIB,
            4 * BYTES_PER_GIB,
        ]
        mock_torch.cuda.memory_reserved.side_effect = [
            6 * BYTES_PER_GIB,
            6 * BYTES_PER_GIB,  # no change
        ]

        mock_gc = MagicMock()
        mock_gc.collect.return_value = 0

        queue = _make_queue()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
            "Wan2GP.models.wan.uni3c": MagicMock(),
        }):
            cleanup_memory_after_task(queue, "task-2")

        # Should log "No significant VRAM freed"
        info_calls = [str(c) for c in queue.logger.info.call_args_list]
        no_freed = [c for c in info_calls if "No significant" in c]
        assert len(no_freed) >= 1


class TestCleanupCudaNotAvailable:
    """Tests when CUDA is not available."""

    def test_gc_still_runs(self):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_gc = MagicMock()
        mock_gc.collect.return_value = 10

        queue = _make_queue()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
            "Wan2GP.models.wan.uni3c": MagicMock(),
        }):
            cleanup_memory_after_task(queue, "task-3")

        # gc.collect should still be called
        mock_gc.collect.assert_called_once()

        # empty_cache should NOT be called (no CUDA)
        mock_torch.cuda.empty_cache.assert_not_called()


class TestCleanupExceptionPath:
    """Tests for error handling in cleanup."""

    def test_runtime_error_logs_warning(self):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.side_effect = RuntimeError("CUDA error")

        mock_gc = MagicMock()

        queue = _make_queue()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
            "Wan2GP.models.wan.uni3c": MagicMock(),
        }):
            # Should not raise
            cleanup_memory_after_task(queue, "task-err")

        queue.logger.warning.assert_called_once()
        assert "Failed to cleanup" in queue.logger.warning.call_args[0][0]

    def test_os_error_logs_warning(self):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache.side_effect = OSError("device lost")
        mock_torch.cuda.memory_allocated.return_value = 1 * BYTES_PER_GIB
        mock_torch.cuda.memory_reserved.return_value = 2 * BYTES_PER_GIB

        mock_gc = MagicMock()

        queue = _make_queue()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
            "Wan2GP.models.wan.uni3c": MagicMock(),
        }):
            cleanup_memory_after_task(queue, "task-os-err")

        queue.logger.warning.assert_called_once()


class TestUni3cCacheClearing:
    """Tests for uni3c cache clearing behavior."""

    def test_uni3c_import_error_is_ignored(self):
        """When Wan2GP.models.wan.uni3c is not importable, cleanup continues."""
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_gc = MagicMock()
        mock_gc.collect.return_value = 5

        queue = _make_queue()

        # Remove uni3c from sys.modules so the import inside the function fails
        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
        }):
            # Remove it if present so the `from Wan2GP...` import raises ImportError
            sys.modules.pop("Wan2GP.models.wan.uni3c", None)
            sys.modules.pop("Wan2GP.models.wan", None)
            sys.modules.pop("Wan2GP.models", None)
            sys.modules.pop("Wan2GP", None)

            # Should not raise
            cleanup_memory_after_task(queue, "task-no-uni3c")

        # gc should still have been called
        mock_gc.collect.assert_called_once()

    def test_uni3c_cache_called_when_available(self):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_gc = MagicMock()
        mock_gc.collect.return_value = 0

        mock_uni3c = MagicMock()

        queue = _make_queue()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "gc": mock_gc,
            "Wan2GP": MagicMock(),
            "Wan2GP.models": MagicMock(),
            "Wan2GP.models.wan": MagicMock(),
            "Wan2GP.models.wan.uni3c": mock_uni3c,
        }):
            cleanup_memory_after_task(queue, "task-uni3c")

        mock_uni3c.clear_uni3c_cache_if_unused.assert_called_once()


class TestCleanupMemoryDirect:
    """Direct checks with lightweight stubs instead of mock patch decorators."""

    def test_non_cuda_path_logs_gc_and_skips_empty_cache(self, monkeypatch):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        class _Cuda:
            def __init__(self):
                self.empty_cache_calls = 0

            @staticmethod
            def is_available():
                return False

            def empty_cache(self):
                self.empty_cache_calls += 1

        class _Torch:
            def __init__(self):
                self.cuda = _Cuda()

        class _GC:
            @staticmethod
            def collect():
                return 17

        class _Logger:
            def __init__(self):
                self.infos = []
                self.warnings = []

            def info(self, msg):
                self.infos.append(msg)

            def warning(self, msg):
                self.warnings.append(msg)

        class _Queue:
            def __init__(self):
                self.logger = _Logger()

        fake_torch = _Torch()
        queue = _Queue()

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "gc", _GC)
        monkeypatch.setitem(
            sys.modules,
            "Wan2GP.models.wan.uni3c",
            types.SimpleNamespace(clear_uni3c_cache_if_unused=lambda: None),
        )

        cleanup_memory_after_task(queue, "task-direct")

        assert fake_torch.cuda.empty_cache_calls == 0
        assert len(queue.logger.warnings) == 0
        assert len(queue.logger.infos) >= 1
        assert any("Garbage collected 17 objects" in msg for msg in queue.logger.infos)
        assert all("BEFORE - VRAM" not in msg for msg in queue.logger.infos)
        assert all("AFTER - VRAM" not in msg for msg in queue.logger.infos)
        assert all("Freed" not in msg for msg in queue.logger.infos)
        assert all("Cleared CUDA cache" not in msg for msg in queue.logger.infos)
        assert queue.logger.infos[-1].endswith("17 objects")

    def test_cuda_path_logs_before_after_and_freed(self, monkeypatch):
        from source.task_handlers.queue.memory_cleanup import cleanup_memory_after_task

        class _Cuda:
            CAPTURED = []

            def __init__(self):
                self.empty_cache_calls = 0
                self._alloc_calls = 0
                self._reserved_calls = 0

            @staticmethod
            def is_available():
                return True

            def empty_cache(self):
                self.empty_cache_calls += 1

            def memory_allocated(self):
                self._alloc_calls += 1
                return (5 if self._alloc_calls == 1 else 4) * BYTES_PER_GIB

            def memory_reserved(self):
                self._reserved_calls += 1
                return (7 if self._reserved_calls == 1 else 5) * BYTES_PER_GIB

        class _Torch:
            def __init__(self):
                self.cuda = _Cuda()

        class _GC:
            @staticmethod
            def collect():
                return 9

        class _Logger:
            def __init__(self):
                self.infos = []
                self.warnings = []

            def info(self, msg):
                self.infos.append(msg)

            def warning(self, msg):
                self.warnings.append(msg)

        class _Queue:
            def __init__(self):
                self.logger = _Logger()

        fake_torch = _Torch()
        queue = _Queue()

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "gc", _GC)
        monkeypatch.setitem(
            sys.modules,
            "Wan2GP.models.wan.uni3c",
            types.SimpleNamespace(clear_uni3c_cache_if_unused=lambda: None),
        )

        cleanup_memory_after_task(queue, "task-cuda-direct")

        assert fake_torch.cuda.empty_cache_calls == 1
        assert len(queue.logger.warnings) == 0
        assert len(queue.logger.infos) >= 5
        assert any("task-cuda-direct" in msg for msg in queue.logger.infos)
        assert any("BEFORE - VRAM allocated: 5.00GB, reserved: 7.00GB" in msg for msg in queue.logger.infos)
        assert any("Cleared CUDA cache" in msg for msg in queue.logger.infos)
        assert any("Garbage collected 9 objects" in msg for msg in queue.logger.infos)
        assert any("AFTER - VRAM allocated: 4.00GB, reserved: 5.00GB" in msg for msg in queue.logger.infos)
        assert any("Freed 2.00GB of reserved VRAM" in msg for msg in queue.logger.infos)
        assert not any("No significant VRAM freed" in msg for msg in queue.logger.infos)
        assert "task-cuda-direct" in queue.logger.infos[0]
        assert "BEFORE - VRAM" in queue.logger.infos[0]
        assert "AFTER - VRAM" in "\n".join(queue.logger.infos)
