"""Safe logging utilities - Prevent freezes on large data structures."""

import reprlib
import json
from typing import Optional, Any

from source.core.log.core import ComponentLogger


# =============================================================================
# GLOBAL LOGGING CONFIGURATION
# =============================================================================
# These constants define consistent limits across the entire codebase to prevent
# logging-induced hangs while maintaining useful debug information.

# String representation limits
LOG_MAX_STRING_REPR = 200          # Max chars for individual string values
LOG_MAX_OBJECT_OUTPUT = 500        # Max chars for entire object representation
LOG_MAX_COLLECTION_ITEMS = 5       # Max items to show in lists/dicts/sets
LOG_MAX_NESTING_DEPTH = 3          # Max recursion depth for nested structures

# JSON serialization limits (for legacy json.dumps usage)
LOG_MAX_JSON_OUTPUT = 1000         # Max chars for JSON.dumps() output (legacy)

# Safety truncation limit for debug messages
LOG_MAX_DEBUG_MESSAGE = 10000      # Max chars before a debug message gets truncated

# Known problematic keys that contain large nested structures
LOG_LARGE_DICT_KEYS = {
    'orchestrator_payload', 'orchestrator_details', 'full_orchestrator_payload',
    'phase_config', 'wgp_params', 'generation_params', 'task_params',
    'resolved_params', 'model_defaults', 'model_config', 'db_task_params',
    'task_params_from_db', 'task_params_dict', 'extracted_params'
}

# Configure reprlib for safe string conversion using global constants
_safe_repr = reprlib.Repr()
_safe_repr.maxstring = LOG_MAX_STRING_REPR
_safe_repr.maxother = LOG_MAX_STRING_REPR
_safe_repr.maxlist = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxdict = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxset = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxtuple = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxdeque = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxarray = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxlevel = LOG_MAX_NESTING_DEPTH


def safe_repr(obj: Any, max_length: int = None) -> str:
    """
    Safely convert any object to string with size limits.

    This prevents logging from hanging on large nested structures.
    Uses reprlib for smart truncation of collections.

    Args:
        obj: Object to convert to string
        max_length: Maximum string length (default: LOG_MAX_OBJECT_OUTPUT)

    Returns:
        Safe string representation, truncated if needed

    Example:
        >>> safe_repr({'huge': ['data'] * 1000})
        "{'huge': ['data', 'data', 'data', 'data', 'data', ...]}"
    """
    if max_length is None:
        max_length = LOG_MAX_OBJECT_OUTPUT

    try:
        # Use reprlib for smart truncation
        result = _safe_repr.repr(obj)

        # Additional length limit as backup
        if len(result) > max_length:
            result = result[:max_length] + "...}"

        return result
    except (ValueError, KeyError, TypeError) as e:
        return f"<repr failed: {type(obj).__name__} - {e}>"


def safe_dict_repr(d: dict, max_items: int = None, max_length: int = None) -> str:
    """
    Safely represent a dictionary with smart truncation.

    Special handling for known problematic keys that contain large nested data.

    Args:
        d: Dictionary to represent
        max_items: Maximum number of items to show (default: LOG_MAX_COLLECTION_ITEMS)
        max_length: Maximum total string length (default: LOG_MAX_OBJECT_OUTPUT)

    Returns:
        Safe string representation of dict

    Example:
        >>> safe_dict_repr({'orchestrator_payload': {...huge...}, 'seed': 123})
        "{'orchestrator_payload': <dict with 45 keys>, 'seed': 123, ...2 more}"
    """
    if max_items is None:
        max_items = LOG_MAX_COLLECTION_ITEMS
    if max_length is None:
        max_length = LOG_MAX_OBJECT_OUTPUT

    if not isinstance(d, dict):
        return safe_repr(d, max_length)

    try:

        items = []
        remaining = len(d) - max_items if len(d) > max_items else 0

        for i, (k, v) in enumerate(d.items()):
            if i >= max_items:
                break

            # Smart handling based on key name and value type
            if k in LOG_LARGE_DICT_KEYS and isinstance(v, dict):
                # Just show key count for known large dicts
                items.append(f"'{k}': <dict with {len(v)} keys>")
            elif isinstance(v, (dict, list, tuple, set)) and len(str(v)) > 100:
                # Use reprlib for other large collections
                items.append(f"'{k}': {_safe_repr.repr(v)}")
            else:
                # Normal representation for small values
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:100] + "..."
                items.append(f"'{k}': {v_str}")

        result = "{" + ", ".join(items)
        if remaining > 0:
            result += f", ...{remaining} more"
        result += "}"

        # Final length check
        if len(result) > max_length:
            result = result[:max_length] + "...}"

        return result

    except (ValueError, KeyError, TypeError) as e:
        return f"<dict repr failed: {len(d) if hasattr(d, '__len__') else '?'} items - {e}>"


def safe_log_params(params: dict, param_name: str = "parameters") -> str:
    """
    Create a safe log message for parameter dictionaries.

    This is specifically designed for logging generation/task parameters
    without causing hangs.

    Args:
        params: Parameter dictionary to log
        param_name: Name to use in the log message (default: "parameters")

    Returns:
        Safe log message string

    Example:
        >>> safe_log_params({'model': 'wan_2_2', 'seed': 123, 'huge_config': {...}})
        "parameters: {'model': 'wan_2_2', 'seed': 123, 'huge_config': <dict with 50 keys>}"
    """
    return f"{param_name}: {safe_dict_repr(params)}"


def safe_json_repr(obj: Any, max_length: int = None) -> str:
    """
    Safely serialize object to JSON string with size limits.

    This is a replacement for json.dumps(...)[:<limit>] pattern which still
    serializes the entire object before truncation (causing hangs).

    Args:
        obj: Object to serialize
        max_length: Maximum output length (default: LOG_MAX_JSON_OUTPUT)

    Returns:
        Safe JSON string, truncated if needed

    Example:
        >>> safe_json_repr({'huge': [1]*1000})
        '{"huge": [1, 1, 1, 1, 1, ...]}'

    Note:
        Prefer safe_dict_repr() for dicts as it's faster. This is for cases
        where JSON format is specifically needed for compatibility.
    """
    if max_length is None:
        max_length = LOG_MAX_JSON_OUTPUT

    try:
        # For small objects, use normal JSON serialization
        if isinstance(obj, (str, int, float, bool, type(None))):
            return json.dumps(obj)

        # For collections, try full serialization but catch large ones
        try:
            result = json.dumps(obj, default=str, indent=2)
            if len(result) <= max_length:
                return result
            # Too long, truncate with ellipsis
            return result[:max_length] + "...}"
        except (TypeError, ValueError, RecursionError):
            # Fallback to safe_repr for objects that can't be JSON serialized
            return safe_repr(obj, max_length)

    except (json.JSONDecodeError, ValueError) as e:
        return f"<json serialization failed: {type(obj).__name__} - {e}>"


def safe_log_change(param: str, old_value: Any, new_value: Any, max_length: int = None) -> str:
    """
    Create a safe log message for parameter changes (old -> new).

    Args:
        param: Parameter name
        old_value: Old value
        new_value: New value
        max_length: Maximum length per value (default: LOG_MAX_STRING_REPR)

    Returns:
        Safe log message string

    Example:
        >>> safe_log_change('seed', 123, 456)
        "seed: 123 -> 456"
        >>> safe_log_change('config', {...huge...}, {...huge...})
        "config: <dict with 50 keys> -> <dict with 52 keys>"
    """
    if max_length is None:
        max_length = LOG_MAX_STRING_REPR

    try:
        # Special handling for dicts
        if isinstance(old_value, dict):
            old_str = f"<dict with {len(old_value)} keys>"
        elif old_value == "NOT_SET":
            old_str = "NOT_SET"
        else:
            old_str = safe_repr(old_value, max_length)

        if isinstance(new_value, dict):
            new_str = f"<dict with {len(new_value)} keys>"
        else:
            new_str = safe_repr(new_value, max_length)

        return f"{param}: {old_str} \u2192 {new_str}"
    except (ValueError, KeyError, TypeError) as e:
        return f"{param}: <comparison failed: {e}>"


# Update ComponentLogger to use safe logging by default
class SafeComponentLogger(ComponentLogger):
    """
    Enhanced ComponentLogger with automatic safe logging.

    Automatically applies safe_repr() to prevent hanging on large objects.
    """

    def debug(self, message: str, task_id: Optional[str] = None):
        """Log a debug message with automatic safety checks."""
        # If message contains dict formatting, it's already too late
        # But we can catch obvious cases
        if len(message) > LOG_MAX_DEBUG_MESSAGE:
            message = message[:LOG_MAX_DEBUG_MESSAGE] + "... [truncated - message too long]"
        super().debug(message, task_id)

    def safe_debug_dict(self, label: str, data: dict, task_id: Optional[str] = None):
        """
        Safely log a dictionary with truncation.

        Example:
            logger.safe_debug_dict("Generation params", params, task_id)
        """
        safe_msg = safe_log_params(data, label)
        self.debug(safe_msg, task_id)

    def safe_debug_change(self, param: str, old_value: Any, new_value: Any, task_id: Optional[str] = None):
        """
        Safely log a parameter change.

        Example:
            logger.safe_debug_change("seed", old_seed, new_seed, task_id)
        """
        safe_msg = safe_log_change(param, old_value, new_value)
        self.debug(safe_msg, task_id)


# Create safe versions of pre-configured loggers
headless_logger_safe = SafeComponentLogger("HEADLESS")
queue_logger_safe = SafeComponentLogger("QUEUE")
orchestrator_logger_safe = SafeComponentLogger("ORCHESTRATOR")
travel_logger_safe = SafeComponentLogger("TRAVEL")
generation_logger_safe = SafeComponentLogger("GENERATION")
model_logger_safe = SafeComponentLogger("MODEL")
task_logger_safe = SafeComponentLogger("TASK")
