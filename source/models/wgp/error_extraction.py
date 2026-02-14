"""
WGP Error Extraction

Extracts meaningful error messages from WGP's captured stdout/stderr output.
WGP often catches exceptions internally and prints them rather than raising,
so we scan the output for known patterns to enable proper retry classification.
"""

import re
from typing import Optional


# Error patterns to look for in WGP output, ordered by priority.
# When generation fails silently (no output), we scan captured stdout/stderr
# for these patterns to extract the actual error for proper retry classification.
WGP_ERROR_PATTERNS = [
    # OOM errors - highest priority, should get 3 retry attempts
    (r"torch\.OutOfMemoryError[:\s]*(.*?)(?:\n|$)", "torch.OutOfMemoryError"),
    (r"CUDA out of memory\.(.*?)(?:See documentation|$)", "CUDA out of memory"),
    (r"Tried to allocate (\d+\.?\d*\s*[GMK]iB)", "CUDA out of memory: Tried to allocate"),
    (r"CUDA error: out of memory", "CUDA error: out of memory"),

    # CUDA/GPU errors
    (r"CUDA error: (.*?)(?:\n|$)", "CUDA error"),
    (r"RuntimeError: CUDA(.*?)(?:\n|$)", "CUDA RuntimeError"),

    # Model loading errors
    (r"Failed to load model[:\s]*(.*?)(?:\n|$)", "Model loading failed"),
    (r"Error loading (.*?) model", "Model loading error"),

    # Generic Python errors (lower priority)
    (r"RuntimeError: (.*?)(?:\n|$)", "RuntimeError"),
    (r"ValueError: (.*?)(?:\n|$)", "ValueError"),
    (r"Exception: (.*?)(?:\n|$)", "Exception"),
]

# Maximum chars of captured stdout/stderr to include in error logs
LOG_TAIL_MAX_CHARS = 2000


def _extract_wgp_error(stdout_content: str, stderr_content: str) -> Optional[str]:
    """
    Extract the actual error from WGP's captured stdout/stderr.

    WGP often catches exceptions internally and prints them rather than raising.
    This function scans the captured output for known error patterns and returns
    a meaningful error message that can be properly classified by the retry system.

    Args:
        stdout_content: Captured stdout from WGP generation
        stderr_content: Captured stderr from WGP generation

    Returns:
        Error message string if found, None otherwise
    """
    combined = (stdout_content or "") + "\n" + (stderr_content or "")

    if not combined.strip():
        return None

    # Check for error patterns in priority order
    for pattern, error_prefix in WGP_ERROR_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE | re.DOTALL)
        if match:
            # Extract matched detail if available
            detail = match.group(1).strip() if match.lastindex else ""
            # Clean up the detail (remove extra whitespace, limit length)
            detail = " ".join(detail.split())[:200]

            if detail:
                return f"{error_prefix}: {detail}"
            return error_prefix

    # If we see generic error indicators but no specific pattern, return a hint
    error_indicators = ['error', 'exception', 'traceback', 'failed']
    combined_lower = combined.lower()
    if any(indicator in combined_lower for indicator in error_indicators):
        # Try to extract traceback's last line (usually the actual error)
        lines = combined.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('File ') and not line.startswith('  '):
                if any(indicator in line.lower() for indicator in ['error', 'exception']):
                    return line[:300]

    return None
