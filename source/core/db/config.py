"""
Database configuration: globals, constants, and debug helpers.

All module-level state that other db submodules depend on lives here.
"""
from __future__ import annotations

import datetime
from typing import Optional

# Import centralized logger for system_logs visibility
try:
    from ...core.log import headless_logger
except ImportError:
    # Fallback if core.log not available
    headless_logger = None

try:
    from supabase import Client as SupabaseClient
except ImportError:
    SupabaseClient = None

from source.core.constants import BYTES_PER_MB

# -----------------------------------------------------------------------------
# Global DB Configuration (will be set by worker.py)
# -----------------------------------------------------------------------------
PG_TABLE_NAME = "tasks"
SUPABASE_URL = None
SUPABASE_SERVICE_KEY = None
SUPABASE_VIDEO_BUCKET = "image_uploads"
SUPABASE_CLIENT: SupabaseClient | None = None
SUPABASE_EDGE_COMPLETE_TASK_URL: str | None = None  # Optional override for edge function
SUPABASE_ACCESS_TOKEN: str | None = None # Will be set by worker.py
SUPABASE_EDGE_CREATE_TASK_URL: str | None = None # Will be set by worker.py
SUPABASE_EDGE_CLAIM_TASK_URL: str | None = None # Will be set by worker.py

# -----------------------------------------------------------------------------
# Status Constants
# -----------------------------------------------------------------------------
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"
# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------
debug_mode = False

def dprint(msg: str):
    """Print a debug message if debug_mode is enabled."""
    if debug_mode:
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] {msg}")

def _log_thumbnail(msg: str, level: str = "debug", task_id: str = None):
    """Log thumbnail-related messages to both stdout and centralized logger."""
    full_msg = f"[THUMBNAIL] {msg}"
    print(full_msg)  # Always print to stdout
    if headless_logger:
        if level == "info":
            headless_logger.info(full_msg, task_id=task_id)
        elif level == "warning":
            headless_logger.warning(full_msg, task_id=task_id)
        else:
            headless_logger.debug(full_msg, task_id=task_id)

# -----------------------------------------------------------------------------
# Edge function error prefix (used by debug.py to detect edge failures)
# -----------------------------------------------------------------------------
EDGE_FAIL_PREFIX = "[EDGE_FAIL"  # Used by debug.py to detect edge failures

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}  # 500 included for transient edge function crashes (CDN issues, cold starts)
