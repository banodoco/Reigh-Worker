"""GuidanceTracker - tracks which frames in the guide video have guidance."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

__all__ = [
    "GuidanceTracker",
]


@dataclass
class GuidanceTracker:
    """Tracks which frames in the guide video have guidance."""
    total_frames: int
    has_guidance: List[bool]  # True = has guidance, False = needs structure motion

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        # Initially, no frames have guidance
        self.has_guidance = [False] * total_frames

    def mark_guided(self, start_idx: int, end_idx: int):
        """Mark a range of frames as having guidance."""
        for i in range(start_idx, min(end_idx + 1, self.total_frames)):
            self.has_guidance[i] = True

    def mark_single_frame(self, idx: int):
        """Mark a single frame as having guidance."""
        if 0 <= idx < self.total_frames:
            self.has_guidance[idx] = True

    def get_unguidanced_ranges(self) -> List[Tuple[int, int]]:
        """
        Get continuous ranges of frames without guidance.

        Returns:
            List of (start_idx, end_idx) tuples for unguidanced ranges
        """
        unguidanced_ranges = []
        in_unguidanced_range = False
        range_start = None

        for i, has_guide in enumerate(self.has_guidance):
            if not has_guide and not in_unguidanced_range:
                # Start of unguidanced range
                range_start = i
                in_unguidanced_range = True
            elif has_guide and in_unguidanced_range:
                # End of unguidanced range
                unguidanced_ranges.append((range_start, i - 1))
                in_unguidanced_range = False

        # Handle case where video ends with unguidanced frames
        if in_unguidanced_range:
            unguidanced_ranges.append((range_start, self.total_frames - 1))

        return unguidanced_ranges

    def get_anchor_frame_index(self, unguidanced_range_start: int) -> Optional[int]:
        """
        Get the index of the last guided frame before an unguidanced range.
        This frame will be used as the anchor for structure motion warping.

        Args:
            unguidanced_range_start: Start index of the unguidanced range

        Returns:
            Index of anchor frame, or None if no guided frame exists before this range
        """
        # Search backward from range start
        for i in range(unguidanced_range_start - 1, -1, -1):
            if self.has_guidance[i]:
                return i
        return None

    def debug_summary(self) -> str:
        """Generate a visual summary of guidance state for debugging."""
        visual = []
        for i, has_guide in enumerate(self.has_guidance):
            if i % 10 == 0:
                visual.append(f"\n{i:3d}: ")
            visual.append("\u2588" if has_guide else "\u2591")

        unguidanced = self.get_unguidanced_ranges()
        summary = [
            "".join(visual),
            f"\nGuided frames: {sum(self.has_guidance)}/{self.total_frames}",
            f"Unguidanced ranges: {len(unguidanced)}"
        ]

        if unguidanced:
            summary.append("Ranges needing structure motion:")
            for start, end in unguidanced:
                summary.append(f"  - Frames {start}-{end} ({end - start + 1} frames)")

        return "\n".join(summary)
