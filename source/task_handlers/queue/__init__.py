"""
Task queue sub-package -- extracted from headless_model_management.py.

Re-exports the public implementation functions so that callers (e.g.
``HeadlessTaskQueue``) can import from one place.
"""

from source.task_handlers.queue.download_ops import switch_model_impl, convert_to_wgp_task_impl
from source.task_handlers.queue.worker_thread import (
    worker_loop,
    process_task_impl,
    execute_generation_impl,
    cleanup_memory_after_task,
)
from source.task_handlers.queue.queue_lifecycle import start_queue, stop_queue, submit_task_impl
from source.task_handlers.queue.task_queue import HeadlessTaskQueue, GenerationTask, QueueStatus  # noqa: F401
