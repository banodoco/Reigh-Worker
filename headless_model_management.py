"""Backward-compatible import location. Actual implementation in source/task_handlers/queue/task_queue.py."""
from source.task_handlers.queue.task_queue import HeadlessTaskQueue, GenerationTask, QueueStatus  # noqa: F401
