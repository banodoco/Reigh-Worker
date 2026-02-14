"""Utility functions for common log patterns."""

from typing import Optional

from source.core.log.core import essential, debug, success, error

__all__ = [
    "log_task_start",
    "log_task_complete",
    "log_task_error",
    "log_model_switch",
    "log_file_operation",
    "log_ffmpeg_command",
    "log_generation_params",
]


# Utility functions for common log patterns
def log_task_start(component: str, task_id: str, task_type: str, **params):
    """Log the start of a task with key parameters."""
    essential(component, f"Starting {task_type} task", task_id)
    if params:
        debug(component, f"Task parameters: {params}", task_id)

def log_task_complete(component: str, task_id: str, task_type: str, output_path: Optional[str] = None, duration: Optional[float] = None):
    """Log the completion of a task."""
    duration_str = f" ({duration:.1f}s)" if duration else ""
    if output_path:
        success(component, f"{task_type} completed{duration_str}: {output_path}", task_id)
    else:
        success(component, f"{task_type} completed{duration_str}", task_id)

def log_task_error(component: str, task_id: str, task_type: str, error_msg: str):
    """Log a task error."""
    error(component, f"{task_type} failed: {error_msg}", task_id)

def log_model_switch(component: str, old_model: Optional[str], new_model: str, duration: Optional[float] = None):
    """Log a model switch operation."""
    duration_str = f" ({duration:.1f}s)" if duration else ""
    if old_model:
        essential(component, f"Model switch: {old_model} \u2192 {new_model}{duration_str}")
    else:
        essential(component, f"Model loaded: {new_model}{duration_str}")

def log_file_operation(component: str, operation: str, source: str, target: Optional[str] = None, task_id: Optional[str] = None):
    """Log file operations like copy, move, download."""
    if target:
        debug(component, f"{operation}: {source} \u2192 {target}", task_id)
    else:
        debug(component, f"{operation}: {source}", task_id)

def log_ffmpeg_command(component: str, command: str, task_id: Optional[str] = None):
    """Log FFmpeg commands (debug only)."""
    debug(component, f"FFmpeg: {command}", task_id)

def log_generation_params(component: str, task_id: str, **params):
    """Log generation parameters (debug only)."""
    debug(component, f"Generation parameters: {params}", task_id)
