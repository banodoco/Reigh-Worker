"""Context manager for timing operations."""

import datetime
from typing import Optional

from source.core.log.core import essential, debug, success, error


# Context manager for timing operations
class LogTimer:
    """Context manager to time and log operations."""

    def __init__(self, component: str, operation: str, task_id: Optional[str] = None, level: str = "essential"):
        self.component = component
        self.operation = operation
        self.task_id = task_id
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        if self.level == "debug":
            debug(self.component, f"Starting {self.operation}...", self.task_id)
        else:
            essential(self.component, f"Starting {self.operation}...", self.task_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            # Success
            if self.level == "debug":
                debug(self.component, f"{self.operation} completed ({duration:.1f}s)", self.task_id)
            else:
                success(self.component, f"{self.operation} completed ({duration:.1f}s)", self.task_id)
        else:
            # Error
            error(self.component, f"{self.operation} failed after {duration:.1f}s: {exc_val}", self.task_id)
