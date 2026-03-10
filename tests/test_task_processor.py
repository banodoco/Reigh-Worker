"""Tests for source/task_handlers/queue/task_processor.py."""

import os
import queue as queue_mod
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from source.task_handlers.queue.task_processor import (
    process_task_impl,
    execute_generation_impl,
    worker_loop,
    _monitor_loop,
)


def _make_queue_mock():
    """Create a mock HeadlessTaskQueue with required attributes."""
    q = MagicMock()
    q.queue_lock = threading.RLock()
    q.current_task = None
    q.running = True
    q.shutdown_event = threading.Event()
    q.logger = MagicMock()
    q.stats = {
        "tasks_completed": 0,
        "tasks_failed": 0,
        "total_generation_time": 0.0,
    }
    return q


def _make_task_mock(task_id="test-task-1", model="vace_14B", prompt="test prompt"):
    """Create a mock GenerationTask."""
    task = MagicMock()
    task.id = task_id
    task.model = model
    task.prompt = prompt
    task.status = "pending"
    task.parameters = {}
    task.result_path = None
    task.error_message = None
    task.processing_time = None
    return task


# ── process_task_impl ────────────────────────────────────────────────────────

class TestProcessTaskImpl:
    """Core task processing flow."""

    @patch("source.task_handlers.queue.task_processor.set_current_task_context", create=True)
    def test_successful_task(self, mock_set_ctx, tmp_path):
        """Task that produces output is marked completed."""
        q = _make_queue_mock()
        task = _make_task_mock()

        # Create a real output file so Path.exists() returns True
        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"video data")
        q._execute_generation.return_value = str(output_file)

        process_task_impl(q, task, "Worker-0")

        assert task.status == "completed"
        assert task.result_path == str(output_file)
        assert q.stats["tasks_completed"] == 1
        assert q.stats["tasks_failed"] == 0
        q._switch_model.assert_called_once_with(task.model, "Worker-0")
        q._cleanup_memory_after_task.assert_called_once_with(task.id)

    @patch("source.task_handlers.queue.task_processor.set_current_task_context", create=True)
    def test_task_no_output_marked_failed(self, mock_set_ctx):
        """Task returning falsy result is marked failed."""
        q = _make_queue_mock()
        task = _make_task_mock()
        q._execute_generation.return_value = None

        process_task_impl(q, task, "Worker-0")

        assert task.status == "failed"
        assert task.error_message == "No output generated"
        assert q.stats["tasks_failed"] == 1

    @patch("source.task_handlers.queue.task_processor.set_current_task_context", create=True)
    def test_task_nonexistent_output_marked_failed(self, mock_set_ctx):
        """Task returning path that doesn't exist is marked failed after retry."""
        q = _make_queue_mock()
        task = _make_task_mock()
        q._execute_generation.return_value = "/nonexistent/path/output.mp4"

        # Patch time.sleep in the retry loop to speed up the test
        with patch("source.task_handlers.queue.task_processor.time.sleep"):
            process_task_impl(q, task, "Worker-0")

        assert task.status == "failed"
        assert q.stats["tasks_failed"] == 1

    @patch("source.task_handlers.queue.task_processor.set_current_task_context", create=True)
    def test_generation_error_marks_failed(self, mock_set_ctx):
        """RuntimeError during generation marks task as failed."""
        q = _make_queue_mock()
        task = _make_task_mock()
        q._execute_generation.side_effect = RuntimeError("CUDA OOM")

        # Mock the fatal error handler to not raise
        with patch("source.task_handlers.queue.task_processor.check_and_handle_fatal_error", create=True):
            process_task_impl(q, task, "Worker-0")

        assert task.status == "failed"
        assert "CUDA OOM" in task.error_message
        assert q.stats["tasks_failed"] == 1

    @patch("source.task_handlers.queue.task_processor.set_current_task_context", create=True)
    def test_current_task_cleared_in_finally(self, mock_set_ctx, tmp_path):
        """current_task is always cleared after processing, even on success."""
        q = _make_queue_mock()
        task = _make_task_mock()

        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"video data")
        q._execute_generation.return_value = str(output_file)

        process_task_impl(q, task, "Worker-0")

        assert q.current_task is None

    @patch("source.task_handlers.queue.task_processor.set_current_task_context", create=True)
    def test_billing_reset_failure_does_not_fail_task(self, mock_set_ctx, tmp_path):
        """If billing reset fails, task still proceeds normally."""
        q = _make_queue_mock()
        task = _make_task_mock()

        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"video data")
        q._execute_generation.return_value = str(output_file)

        with patch(
            "source.task_handlers.queue.task_processor.reset_generation_started_at",
            side_effect=OSError("DB unreachable"),
            create=True,
        ):
            process_task_impl(q, task, "Worker-0")

        # Task should still succeed despite billing failure
        assert task.status == "completed"


# ── execute_generation_impl ──────────────────────────────────────────────────

class TestExecuteGenerationImpl:
    """Generation dispatch to orchestrator."""

    def test_vace_generation_path(self):
        """VACE model delegates to generate_vace."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {}

        q._model_supports_vace.return_value = True
        q._convert_to_wgp_task.return_value = {
            "model": "vace_14B",
            "prompt": "test",
            "video_guide": "/path/to/guide.mp4",
        }
        q.orchestrator.generate_vace.return_value = "/output/vace.mp4"
        q._is_single_image_task.return_value = False

        result = execute_generation_impl(q, task, "Worker-0")

        assert result == "/output/vace.mp4"
        q.orchestrator.generate_vace.assert_called_once()

    def test_vace_without_video_guide_raises(self):
        """VACE model without video_guide raises ValueError."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {}

        q._model_supports_vace.return_value = True
        q._convert_to_wgp_task.return_value = {
            "model": "vace_14B",
            "prompt": "test",
        }

        with pytest.raises(ValueError, match="requires a video_guide"):
            execute_generation_impl(q, task, "Worker-0")

    def test_t2v_generation_path(self):
        """Non-VACE, non-Flux model uses generate_t2v."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {}

        q._model_supports_vace.return_value = False
        q.orchestrator._is_flux.return_value = False
        q._convert_to_wgp_task.return_value = {
            "model": "t2v",
            "prompt": "test",
        }
        q.orchestrator.generate_t2v.return_value = "/output/t2v.mp4"
        q._is_single_image_task.return_value = False

        result = execute_generation_impl(q, task, "Worker-0")

        assert result == "/output/t2v.mp4"
        q.orchestrator.generate_t2v.assert_called_once()

    def test_flux_generation_path(self):
        """Flux model uses generate_flux and maps video_length to num_images."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {}

        q._model_supports_vace.return_value = False
        q.orchestrator._is_flux.return_value = True
        q._convert_to_wgp_task.return_value = {
            "model": "flux",
            "prompt": "test",
            "video_length": 4,
        }
        q.orchestrator.generate_flux.return_value = "/output/flux.png"
        q._is_single_image_task.return_value = False

        result = execute_generation_impl(q, task, "Worker-0")

        assert result == "/output/flux.png"
        q.orchestrator.generate_flux.assert_called_once()
        # video_length should have been remapped to num_images
        call_kwargs = q.orchestrator.generate_flux.call_args
        assert "num_images" in call_kwargs.kwargs or "num_images" in (call_kwargs[1] if len(call_kwargs) > 1 else {})

    def test_single_image_task_converts_to_png(self):
        """Single image tasks get converted from video to PNG."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {"video_length": 1}

        q._model_supports_vace.return_value = False
        q.orchestrator._is_flux.return_value = False
        q._convert_to_wgp_task.return_value = {
            "model": "t2v",
            "prompt": "test",
        }
        q.orchestrator.generate_t2v.return_value = "/output/result.mp4"
        q._is_single_image_task.return_value = True
        q._convert_single_frame_video_to_png.return_value = "/output/result.png"

        result = execute_generation_impl(q, task, "Worker-0")

        assert result == "/output/result.png"
        q._convert_single_frame_video_to_png.assert_called_once()

    def test_generation_error_propagates(self):
        """RuntimeError from orchestrator propagates to caller."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {}

        q._model_supports_vace.return_value = False
        q.orchestrator._is_flux.return_value = False
        q._convert_to_wgp_task.return_value = {
            "model": "t2v",
            "prompt": "test",
        }
        q.orchestrator.generate_t2v.side_effect = RuntimeError("generation failed")

        with pytest.raises(RuntimeError, match="generation failed"):
            execute_generation_impl(q, task, "Worker-0")

    def test_phase_config_patch_and_restore(self):
        """Phase config is applied before generation and restored after."""
        q = _make_queue_mock()
        task = _make_task_mock()
        task.parameters = {}

        q._model_supports_vace.return_value = False
        q.orchestrator._is_flux.return_value = False
        q._convert_to_wgp_task.return_value = {
            "model": "t2v",
            "prompt": "test",
            "_parsed_phase_config": {"phases": []},
            "_phase_config_model_name": "test_model",
        }
        q.orchestrator.generate_t2v.return_value = "/output/result.mp4"
        q._is_single_image_task.return_value = False

        with patch("source.core.params.phase_config.apply_phase_config_patch") as mock_apply, \
             patch("source.core.params.phase_config.restore_model_patches") as mock_restore:
            result = execute_generation_impl(q, task, "Worker-0")

        mock_apply.assert_called_once()
        mock_restore.assert_called_once()


# ── worker_loop ──────────────────────────────────────────────────────────────

class TestWorkerLoop:
    """Worker loop task dequeue behavior."""

    def test_exits_when_shutdown(self):
        """Worker loop exits when shutdown_event is set."""
        q = _make_queue_mock()
        q.task_queue = queue_mod.PriorityQueue()
        q.shutdown_event.set()  # Immediately signal shutdown

        # Should exit quickly without processing any tasks
        worker_loop(q)

        # No tasks processed
        assert q.stats["tasks_completed"] == 0

    def test_processes_task_from_queue(self):
        """Worker loop dequeues and processes a task."""
        q = _make_queue_mock()
        q.task_queue = queue_mod.PriorityQueue()

        task = _make_task_mock()
        q.task_queue.put((0, time.time(), task))

        # Shut down after processing one task
        call_count = [0]
        original_process = process_task_impl

        def mock_process(queue, t, worker_name):
            call_count[0] += 1
            queue.shutdown_event.set()  # Stop after first task

        with patch("source.task_handlers.queue.task_processor.process_task_impl", side_effect=mock_process):
            worker_loop(q)

        assert call_count[0] == 1


# ── _monitor_loop ────────────────────────────────────────────────────────────

class TestMonitorLoop:
    """Monitor loop behavior."""

    def test_exits_when_shutdown(self):
        """Monitor loop exits when shutdown_event is set."""
        q = _make_queue_mock()
        q.shutdown_event.set()

        with patch("source.task_handlers.queue.task_processor.time.sleep"):
            _monitor_loop(q)

        q.logger.info.assert_any_call("Queue monitor stopped")
