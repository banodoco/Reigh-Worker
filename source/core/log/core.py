"""
Centralized logging utility for Headless-Wan2GP

Provides structured logging with debug vs essential log levels.
Essential logs are always shown, debug logs only appear when debug mode is enabled.
"""

import sys
import datetime
import traceback
from typing import Optional
from pathlib import Path

__all__ = [
    "set_log_file",
    "enable_debug_mode",
    "disable_debug_mode",
    "is_debug_enabled",
    "essential",
    "success",
    "warning",
    "error",
    "critical",
    "debug",
    "progress",
    "status",
    "ComponentLogger",
    "headless_logger",
    "queue_logger",
    "orchestrator_logger",
    "travel_logger",
    "generation_logger",
    "model_logger",
    "task_logger",
    "set_log_interceptor",
    "set_current_task_context",
]

# Global debug mode flag - set by the main application
_debug_mode = False
# Global log file handle
_log_file = None
_log_file_lock = None

def set_log_file(path: str):
    """Set a file path to mirror all logs to."""
    global _log_file, _log_file_lock
    import threading
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(path, 'a', encoding='utf-8')
        _log_file_lock = threading.Lock()
        essential("LOGGING", f"Logging to file enabled: {path}")
    except OSError as e:
        error("LOGGING", f"Failed to set log file {path}: {e}")

def _write_to_log_file(formatted_message: str):
    """Write message to log file if enabled."""
    global _log_file, _log_file_lock
    if _log_file and _log_file_lock:
        try:
            with _log_file_lock:
                _log_file.write(formatted_message + "\n")
                _log_file.flush()
        except OSError as e:
            # Log file write failed (disk full, file closed, etc.)
            # Print to stderr as a last resort since our own logging infrastructure is broken
            print(f"[core.log] Failed to write to log file: {e}", file=sys.stderr)

def enable_debug_mode():
    """Enable debug logging globally."""
    global _debug_mode
    _debug_mode = True

def disable_debug_mode():
    """Disable debug logging globally."""
    global _debug_mode
    _debug_mode = False

def is_debug_enabled() -> bool:
    """Check if debug mode is currently enabled."""
    return _debug_mode

def _get_timestamp() -> str:
    """Get formatted timestamp for logs."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def _format_message(level: str, component: str, message: str, task_id: Optional[str] = None) -> str:
    """Format a log message with consistent structure."""
    timestamp = _get_timestamp()

    if task_id:
        return f"[{timestamp}] {level} {component} [Task {task_id}] {message}"
    else:
        return f"[{timestamp}] {level} {component} {message}"

def _append_exc_info(formatted: str, exc_info: bool) -> str:
    """Append traceback to formatted message if exc_info=True and an exception is active."""
    if exc_info:
        exc_text = traceback.format_exc()
        if exc_text and exc_text.strip() != "NoneType: None":
            formatted = f"{formatted}\n{exc_text.rstrip()}"
    return formatted

def essential(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log an essential message that should always be shown."""
    formatted = _append_exc_info(_format_message("INFO", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

def success(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a success message that should always be shown."""
    formatted = _append_exc_info(_format_message("\u2705", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

def warning(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a warning message that should always be shown."""
    formatted = _append_exc_info(_format_message("\u26a0\ufe0f", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

def error(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log an error message that should always be shown."""
    formatted = _append_exc_info(_format_message("\u274c", component, message, task_id), exc_info)
    print(formatted, file=sys.stderr)
    _write_to_log_file(formatted)

def critical(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a critical/fatal error message that should always be shown."""
    formatted = _append_exc_info(_format_message("\U0001f534 CRITICAL", component, message, task_id), exc_info)
    print(formatted, file=sys.stderr)
    _write_to_log_file(formatted)

def debug(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a debug message that only appears when debug mode is enabled."""
    if _debug_mode:
        formatted = _append_exc_info(_format_message("DEBUG", component, message, task_id), exc_info)
        print(formatted)
        _write_to_log_file(formatted)

def progress(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a progress message that should always be shown."""
    formatted = _append_exc_info(_format_message("\u23f3", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

def status(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a status message that should always be shown."""
    formatted = _append_exc_info(_format_message("\U0001f4ca", component, message, task_id), exc_info)
    print(formatted)
    _write_to_log_file(formatted)

# Component-specific loggers for better organization
class ComponentLogger:
    """Logger for a specific component with consistent naming."""

    def __init__(self, component_name: str):
        self.component = component_name

    def essential(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        essential(self.component, message, task_id, exc_info=exc_info)

    def success(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        success(self.component, message, task_id, exc_info=exc_info)

    def warning(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        warning(self.component, message, task_id, exc_info=exc_info)

    def error(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        error(self.component, message, task_id, exc_info=exc_info)

    def critical(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        critical(self.component, message, task_id, exc_info=exc_info)

    def debug(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        debug(self.component, message, task_id, exc_info=exc_info)

    def progress(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        progress(self.component, message, task_id, exc_info=exc_info)

    def status(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        status(self.component, message, task_id, exc_info=exc_info)

    def info(self, message: str, task_id: Optional[str] = None, exc_info: bool = False):
        """Alias for essential() to maintain compatibility with standard logging."""
        essential(self.component, message, task_id, exc_info=exc_info)

# Pre-configured loggers for main components
headless_logger = ComponentLogger("HEADLESS")
queue_logger = ComponentLogger("QUEUE")
orchestrator_logger = ComponentLogger("ORCHESTRATOR")
travel_logger = ComponentLogger("TRAVEL")
generation_logger = ComponentLogger("GENERATION")
model_logger = ComponentLogger("MODEL")
task_logger = ComponentLogger("TASK")

# -----------------------------------------------------------------------------
# Interceptor globals and redefinitions
# -----------------------------------------------------------------------------

from source.core.log.database import CustomLogInterceptor

# Global log interceptor instance (set in worker.py)
_log_interceptor: Optional[CustomLogInterceptor] = None


def set_log_interceptor(interceptor: Optional[CustomLogInterceptor]):
    """Set the global log interceptor for database logging."""
    global _log_interceptor
    _log_interceptor = interceptor


def set_current_task_context(task_id: Optional[str]):
    """
    Set/clear the task context used for associating intercepted logs with a task_id.

    This is intentionally a thin wrapper around the active interceptor instance so that
    worker threads (e.g. the headless generation queue) can correctly tag logs even if
    the main polling thread is busy or if the queue is multi-threaded.
    """
    if _log_interceptor:
        try:
            _log_interceptor.set_current_task(task_id)
        except (ValueError, TypeError, OSError) as e:
            # Interceptor failed to set task context - log to stderr to avoid recursion
            print(f"[core.log] Failed to set task context on log interceptor: {e}", file=sys.stderr)


# Update logging functions to use interceptor
def _intercept_log(level: str, message: str, task_id: Optional[str] = None):
    """Send log to interceptor if enabled."""
    if _log_interceptor:
        _log_interceptor.capture_log(level, message, task_id)


# Modify existing logging functions to intercept
_original_essential = essential
def essential(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log an essential message that should always be shown."""
    _original_essential(component, message, task_id, exc_info=exc_info)
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_success = success
def success(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a success message that should always be shown."""
    _original_success(component, message, task_id, exc_info=exc_info)
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_warning = warning
def warning(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a warning message that should always be shown."""
    _original_warning(component, message, task_id, exc_info=exc_info)
    _intercept_log("WARNING", f"{component}: {message}", task_id)


_original_error = error
def error(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log an error message that should always be shown."""
    _original_error(component, message, task_id, exc_info=exc_info)
    _intercept_log("ERROR", f"{component}: {message}", task_id)


_original_debug = debug
def debug(component: str, message: str, task_id: Optional[str] = None, exc_info: bool = False):
    """Log a debug message that only appears when debug mode is enabled."""
    _original_debug(component, message, task_id, exc_info=exc_info)
    if _debug_mode:  # Only intercept if debug mode is enabled
        _intercept_log("DEBUG", f"{component}: {message}", task_id)
