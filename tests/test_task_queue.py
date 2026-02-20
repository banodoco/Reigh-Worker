"""Tests for source/task_handlers/queue/task_queue.py (HeadlessTaskQueue)."""

import queue
import sys
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Pre-import mocking: inject stubs into sys.modules BEFORE any source imports
# to avoid the cv2.dnn.DictValue crash (broken cv2 install) and heavy deps.
# ---------------------------------------------------------------------------

_mock_queue_logger = MagicMock()
_mock_worker_loop = MagicMock()
_mock_process_task_impl = MagicMock()
_mock_execute_generation_impl = MagicMock()
_mock_cleanup = MagicMock()
_mock_start_queue = MagicMock()
_mock_stop_queue = MagicMock()
_mock_submit_task_impl = MagicMock()
_mock_switch_model_impl = MagicMock()
_mock_convert_to_wgp_task_impl = MagicMock()

# Mock cv2 to prevent the cv2.dnn.DictValue AttributeError during import.
# This must happen before importing any source module that touches cv2.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

from source.task_handlers.queue.task_queue import (  # noqa: E402
    HeadlessTaskQueue,
    GenerationTask,
    QueueStatus,
    create_sample_task,
)

# Patch the module-level bindings that task_queue.py imported from its dependencies.
# This works even if the real modules were loaded first (full suite), because
# we directly replace the names in the task_queue module's namespace.
import source.task_handlers.queue.task_queue as _tq_module
_tq_module.queue_logger = _mock_queue_logger
_tq_module.start_queue = _mock_start_queue
_tq_module.stop_queue = _mock_stop_queue
_tq_module.submit_task_impl = _mock_submit_task_impl
_tq_module.worker_loop = _mock_worker_loop
_tq_module.process_task_impl = _mock_process_task_impl
_tq_module.execute_generation_impl = _mock_execute_generation_impl
_tq_module.cleanup_memory_after_task = _mock_cleanup
_tq_module.switch_model_impl = _mock_switch_model_impl
_tq_module.convert_to_wgp_task_impl = _mock_convert_to_wgp_task_impl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset all mock call counts between tests so delegation assertions are clean."""
    for mock_obj in (
        _mock_queue_logger,
        _mock_worker_loop,
        _mock_process_task_impl,
        _mock_execute_generation_impl,
        _mock_cleanup,
        _mock_start_queue,
        _mock_stop_queue,
        _mock_submit_task_impl,
        _mock_switch_model_impl,
        _mock_convert_to_wgp_task_impl,
    ):
        mock_obj.reset_mock()
    yield


def _make_queue():
    """Create a HeadlessTaskQueue with mocked WGP path."""
    with (
        patch("os.path.abspath", return_value="/fake/wan_dir"),
        patch("logging.FileHandler", return_value=MagicMock()),
    ):
        q = HeadlessTaskQueue(wan_dir="/fake/wan_dir")
    return q


# ── GenerationTask ───────────────────────────────────────────────────────────

class TestGenerationTask:
    def test_default_status_is_pending(self):
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        assert task.status == "pending"

    def test_created_at_auto_populated(self):
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        assert task.created_at is not None

    def test_custom_priority(self):
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={}, priority=5)
        assert task.priority == 5


# ── create_sample_task ───────────────────────────────────────────────────────

class TestCreateSampleTask:
    def test_creates_with_correct_fields(self):
        task = create_sample_task("task-1", "vace_14B", "a forest scene", seed=42)
        assert task.id == "task-1"
        assert task.model == "vace_14B"
        assert task.prompt == "a forest scene"
        assert task.parameters.get("seed") == 42


# ── HeadlessTaskQueue initialization ─────────────────────────────────────────

class TestHeadlessTaskQueueInit:
    def test_init_sets_defaults(self):
        q = _make_queue()
        assert q.running is False
        assert q.current_task is None
        assert q.current_model is None
        assert q.orchestrator is None
        assert q.max_workers == 1

    def test_init_custom_workers(self):
        with (
            patch("os.path.abspath", return_value="/fake"),
            patch("logging.FileHandler", return_value=MagicMock()),
        ):
            q = HeadlessTaskQueue(wan_dir="/fake", max_workers=4)
        assert q.max_workers == 4

    def test_stats_initialized(self):
        q = _make_queue()
        assert q.stats["tasks_submitted"] == 0
        assert q.stats["tasks_completed"] == 0
        assert q.stats["tasks_failed"] == 0
        assert q.stats["model_switches"] == 0
        assert q.stats["total_generation_time"] == 0.0


# ── Queue operations ─────────────────────────────────────────────────────────

class TestQueueOperations:
    def test_get_task_status_found(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        q.task_history["t1"] = task
        assert q.get_task_status("t1") is task

    def test_get_task_status_not_found(self):
        q = _make_queue()
        assert q.get_task_status("nonexistent") is None

    def test_get_queue_status(self):
        q = _make_queue()
        status = q.get_queue_status()
        assert status.pending_tasks == 0
        assert status.processing_task is None
        assert status.completed_tasks == 0
        assert status.failed_tasks == 0

    def test_get_queue_status_with_current_task(self):
        q = _make_queue()
        task = GenerationTask(id="active-task", model="m1", prompt="p1", parameters={})
        q.current_task = task
        q.stats["tasks_completed"] = 5
        q.stats["tasks_failed"] = 2

        status = q.get_queue_status()
        assert status.processing_task == "active-task"
        assert status.completed_tasks == 5
        assert status.failed_tasks == 2


# ── Delegation methods ───────────────────────────────────────────────────────

class TestDelegation:
    """Verify that HeadlessTaskQueue methods delegate to the correct implementations."""

    def test_start_delegates(self):
        q = _make_queue()
        q.start()
        _mock_start_queue.assert_called()

    def test_stop_delegates(self):
        q = _make_queue()
        q.stop()
        _mock_stop_queue.assert_called()

    def test_submit_task_delegates(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        q.submit_task(task)
        _mock_submit_task_impl.assert_called()

    def test_cleanup_memory_delegates(self):
        q = _make_queue()
        q._cleanup_memory_after_task("task-1")
        _mock_cleanup.assert_called()


# ── Model support checks ────────────────────────────────────────────────────

class TestModelSupport:
    def test_is_single_image_task_true(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={"video_length": 1})
        assert q._is_single_image_task(task) is True

    def test_is_single_image_task_false(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={"video_length": 49})
        assert q._is_single_image_task(task) is False

    def test_is_single_image_task_no_video_length(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        assert q._is_single_image_task(task) is False

    def test_model_supports_vace_name_fallback(self):
        """When orchestrator has no is_model_vace, falls back to name-based check."""
        q = _make_queue()
        q.orchestrator = MagicMock(spec=[])  # No attributes at all
        assert q._model_supports_vace("wan_vace_14B") is True
        assert q._model_supports_vace("wan_t2v_14B") is False


# ── Memory usage ─────────────────────────────────────────────────────────────

class TestMemoryUsage:
    def test_get_memory_usage_returns_dict(self):
        q = _make_queue()
        mem = q._get_memory_usage()
        assert isinstance(mem, dict)
        assert "gpu_memory_used" in mem
        assert "system_memory_used" in mem


# ── wait_for_completion ──────────────────────────────────────────────────────

class TestWaitForCompletion:
    def test_task_not_found(self):
        q = _make_queue()
        result = q.wait_for_completion("nonexistent", timeout=0.1)
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_completed_task(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        task.status = "completed"
        task.result_path = "/output/result.mp4"
        q.task_history["t1"] = task

        result = q.wait_for_completion("t1", timeout=1.0)
        assert result["success"] is True
        assert result["output_path"] == "/output/result.mp4"

    def test_failed_task(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        task.status = "failed"
        task.error_message = "CUDA OOM"
        q.task_history["t1"] = task

        result = q.wait_for_completion("t1", timeout=1.0)
        assert result["success"] is False
        assert "CUDA OOM" in result["error"]

    def test_timeout(self):
        q = _make_queue()
        task = GenerationTask(id="t1", model="m1", prompt="p1", parameters={})
        task.status = "processing"
        q.task_history["t1"] = task

        with patch("source.task_handlers.queue.task_queue.time.sleep"):
            result = q.wait_for_completion("t1", timeout=0.01)
        assert result["success"] is False
        assert "did not complete" in result["error"]
