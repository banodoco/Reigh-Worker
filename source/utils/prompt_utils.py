"""Prompt validation and task ID generation."""

import uuid

__all__ = [
    "ensure_valid_prompt",
    "ensure_valid_negative_prompt",
    "generate_unique_task_id",
]


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
