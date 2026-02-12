"""Prompt validation, debug helpers, status constants, and task ID generation."""

import uuid
from datetime import datetime

# --- Global Debug Mode ---
# This will be set by the main script (steerable_motion.py)
DEBUG_MODE = False

# --- Constants for DB interaction and defaults ---
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"
DEFAULT_DB_TABLE_NAME = "tasks"


# --- Debug / Verbose Logging Helper ---
def dprint(msg: str):
    """Print a debug message if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(f"[DEBUG SM-COMMON {datetime.utcnow().isoformat()}Z] {msg}")


def _ensure_valid_text(text: str | None) -> str:
    """Return *text* stripped, or a single space when *text* is None/blank."""
    if not text or not text.strip():
        return " "
    return text.strip()


def ensure_valid_prompt(prompt: str | None) -> str:
    """Ensures prompt is valid (not None or empty), returns space as default."""
    return _ensure_valid_text(prompt)


def ensure_valid_negative_prompt(negative_prompt: str | None) -> str:
    """Ensures negative prompt is valid (not None or empty), returns space as default."""
    return _ensure_valid_text(negative_prompt)


def generate_unique_task_id(prefix: str = "") -> str:
    """Generates a UUID4 string.

    The optional *prefix* parameter is now ignored so that the returned value
    is a bare RFC-4122 UUID which can be stored in a Postgres `uuid` column
    without casting errors.  The argument is kept in the signature to avoid
    breaking existing call-sites that still pass a prefix.
    """
    return str(uuid.uuid4())
