"""Tests for source/task_handlers/queue/queue_lifecycle.py."""

import time
import threading
from unittest import mock
from unittest.mock import MagicMock, patch, call

import pytest


def _make_queue(**overrides):
    """Build a mock HeadlessTaskQueue with required attributes."""
    queue = MagicMock()
    queue.running = False
    queue.max_workers = 2
    queue.shutdown_event = MagicMock()
    queue.worker_threads = []
    queue.logger = MagicMock()
    queue.orchestrator = MagicMock()
    queue.current_model = None
    queue.queue_lock = threading.Lock()
    queue.task_queue = MagicMock()
    queue.task_history = {}
    queue.stats = {"tasks_submitted": 0}
    for k, v in overrides.items():
        setattr(queue, k, v)
    return queue


def _make_task(**overrides):
    """Build a mock GenerationTask."""
    task = MagicMock()
    task.id = overrides.get("id", "task_001")
    task.prompt = overrides.get("prompt", "a cat")
    task.model = overrides.get("model", "wan_2_1_base")
    task.priority = overrides.get("priority", 5)
    task.parameters = overrides.get("parameters", {})
    return task


class TestStartQueue:
    """Tests for start_queue."""

    @patch("source.task_handlers.queue.queue_lifecycle.threading.Thread")
    def test_starts_workers_and_monitor(self, mock_thread_cls):
        """Starts the correct number of worker threads plus monitor."""
        from source.task_handlers.queue.queue_lifecycle import start_queue

        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        queue = _make_queue(max_workers=3)
        start_queue(queue)

        assert queue.running is True
        queue.shutdown_event.clear.assert_called_once()
        # 3 workers + 1 monitor = 4 threads created
        assert mock_thread_cls.call_count == 4
        assert mock_thread.start.call_count == 4
        assert len(queue.worker_threads) == 4

    @patch("source.task_handlers.queue.queue_lifecycle.threading.Thread")
    def test_already_running_does_nothing(self, mock_thread_cls):
        """If queue is already running, start_queue returns early."""
        from source.task_handlers.queue.queue_lifecycle import start_queue

        queue = _make_queue(running=True)
        start_queue(queue)

        queue.logger.warning.assert_called_once()
        mock_thread_cls.assert_not_called()

    @patch("source.task_handlers.queue.queue_lifecycle.threading.Thread")
    def test_preload_model(self, mock_thread_cls):
        """Pre-loading a model calls orchestrator.load_model."""
        from source.task_handlers.queue.queue_lifecycle import start_queue

        mock_thread_cls.return_value = MagicMock()
        queue = _make_queue()

        start_queue(queue, preload_model="wan_2_2_vace")

        queue._ensure_orchestrator.assert_called_once()
        queue.orchestrator.load_model.assert_called_once_with("wan_2_2_vace")
        assert queue.current_model == "wan_2_2_vace"

    @patch("source.task_handlers.queue.queue_lifecycle.threading.Thread")
    def test_preload_model_failure_recoverable(self, mock_thread_cls):
        """Model preload failure with valid orchestrator logs warning but continues."""
        from source.task_handlers.queue.queue_lifecycle import start_queue

        mock_thread_cls.return_value = MagicMock()
        queue = _make_queue()
        queue.orchestrator.load_model.side_effect = RuntimeError("Model not found")

        # Should not raise since orchestrator is not None
        start_queue(queue, preload_model="bad_model")

        queue.logger.error.assert_called()
        queue.logger.warning.assert_called()

    @patch("source.task_handlers.queue.queue_lifecycle.threading.Thread")
    def test_preload_model_failure_fatal_when_no_orchestrator(self, mock_thread_cls):
        """Fatal error when orchestrator itself fails to initialize."""
        from source.task_handlers.queue.queue_lifecycle import start_queue

        mock_thread_cls.return_value = MagicMock()
        queue = _make_queue()
        queue.orchestrator = None
        queue._ensure_orchestrator.side_effect = None  # doesn't set orchestrator

        # load_model will fail since orchestrator is None
        def fail_load(model):
            raise RuntimeError("init failed")
        # Need orchestrator to still be None after _ensure_orchestrator
        queue._ensure_orchestrator = MagicMock()
        # Simulate: _ensure_orchestrator runs but orchestrator stays None
        queue.orchestrator = None

        with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
            start_queue(queue, preload_model="bad_model")


class TestStopQueue:
    """Tests for stop_queue."""

    def test_stop_sets_flags(self):
        """Stop sets running=False and signals shutdown_event."""
        from source.task_handlers.queue.queue_lifecycle import stop_queue

        worker1 = MagicMock()
        worker2 = MagicMock()
        queue = _make_queue(running=True, worker_threads=[worker1, worker2])

        stop_queue(queue)

        assert queue.running is False
        queue.shutdown_event.set.assert_called_once()
        worker1.join.assert_called_once_with(timeout=30.0)
        worker2.join.assert_called_once_with(timeout=30.0)
        queue._save_queue_state.assert_called_once()

    def test_stop_when_not_running(self):
        """Stop when not running does nothing."""
        from source.task_handlers.queue.queue_lifecycle import stop_queue

        queue = _make_queue(running=False)
        stop_queue(queue)

        queue.shutdown_event.set.assert_not_called()
        queue._save_queue_state.assert_not_called()

    def test_stop_custom_timeout(self):
        """Custom timeout is passed to worker.join."""
        from source.task_handlers.queue.queue_lifecycle import stop_queue

        worker = MagicMock()
        queue = _make_queue(running=True, worker_threads=[worker])

        stop_queue(queue, timeout=5.0)

        worker.join.assert_called_once_with(timeout=5.0)


class TestSubmitTaskImpl:
    """Tests for submit_task_impl."""

    def test_submit_returns_task_id(self):
        """submit_task_impl returns the task ID."""
        from source.task_handlers.queue.queue_lifecycle import submit_task_impl

        queue = _make_queue()
        task = _make_task(id="task_42")

        result = submit_task_impl(queue, task)

        assert result == "task_42"
        assert queue.stats["tasks_submitted"] == 1
        assert "task_42" in queue.task_history

    def test_submit_adds_to_priority_queue(self):
        """Task is added to queue with negative priority for min-heap ordering."""
        from source.task_handlers.queue.queue_lifecycle import submit_task_impl

        queue = _make_queue()
        task = _make_task(id="task_1", priority=10)

        submit_task_impl(queue, task)

        queue.task_queue.put.assert_called_once()
        args = queue.task_queue.put.call_args[0][0]
        # First element should be negative priority
        assert args[0] == -10
        # Third element should be the task
        assert args[2] is task

    def test_submit_validates_conversion(self):
        """_convert_to_wgp_task is called as pre-validation."""
        from source.task_handlers.queue.queue_lifecycle import submit_task_impl

        queue = _make_queue()
        task = _make_task()

        submit_task_impl(queue, task)

        queue._convert_to_wgp_task.assert_called_once_with(task)

    def test_submit_multiple_tasks(self):
        """Multiple tasks can be submitted sequentially."""
        from source.task_handlers.queue.queue_lifecycle import submit_task_impl

        queue = _make_queue()

        for i in range(5):
            task = _make_task(id=f"task_{i}", priority=i)
            submit_task_impl(queue, task)

        assert queue.stats["tasks_submitted"] == 5
        assert len(queue.task_history) == 5
        assert queue.task_queue.put.call_count == 5
