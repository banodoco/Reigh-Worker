"""
VACE quantization calculations for join clips tasks.

Handles the 4n+1 frame count constraint that VACE requires, adjusting
gap frame counts to ensure valid generation parameters.
"""


def _calculate_vace_quantization(
    context_frame_count: int,
    gap_frame_count: int,
    replace_mode: bool,
    context_before: int = None,
    context_after: int = None
) -> dict:
    """
    Calculate VACE quantization adjustments for frame counts.

    VACE requires frame counts to match the pattern 4n+1 (e.g., 45, 49, 53).
    When we request a frame count that doesn't match this pattern, VACE will
    quantize down to the nearest valid count.

    Args:
        context_frame_count: Number of frames from each clip's boundary (symmetric default)
        gap_frame_count: Number of frames to generate in the gap
        replace_mode: Whether we're replacing a portion (True) or inserting (False)
                     Both modes now work similarly for VACE - context + gap + context
        context_before: Override for context frames before gap (for asymmetric cases)
        context_after: Override for context frames after gap (for asymmetric cases)

    Returns:
        dict with:
            - total_frames: Actual frame count VACE will generate
            - gap_for_guide: Adjusted gap to use in guide/mask creation
            - quantization_shift: Number of frames dropped by VACE (0 if no quantization)
    """
    # Support asymmetric context counts
    ctx_before = context_before if context_before is not None else context_frame_count
    ctx_after = context_after if context_after is not None else context_frame_count

    # Calculate desired total: context_before + gap + context_after
    desired_total = ctx_before + gap_frame_count + ctx_after

    # Apply VACE quantization (4n + 1)
    actual_total = ((desired_total - 1) // 4) * 4 + 1
    quantization_shift = desired_total - actual_total

    # Adjust gap to account for dropped frames
    gap_for_guide = max(0, gap_frame_count - quantization_shift)

    return {
        'total_frames': actual_total,
        'gap_for_guide': gap_for_guide,
        'quantization_shift': quantization_shift,
        'context_before': ctx_before,
        'context_after': ctx_after,
    }
