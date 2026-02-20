"""
Tests for source/media/structure/tracker.py - GuidanceTracker.

Run with: python -m pytest tests/test_structure_tracker.py -v
"""

import pytest

from source.media.structure.tracker import GuidanceTracker


class TestGuidanceTrackerInit:
    """Tests for GuidanceTracker initialization."""

    def test_init_sets_total_frames(self):
        tracker = GuidanceTracker(total_frames=10)
        assert tracker.total_frames == 10

    def test_init_all_frames_unguidanced(self):
        tracker = GuidanceTracker(total_frames=5)
        assert tracker.has_guidance == [False, False, False, False, False]

    def test_init_zero_frames(self):
        tracker = GuidanceTracker(total_frames=0)
        assert tracker.total_frames == 0
        assert tracker.has_guidance == []

    def test_init_single_frame(self):
        tracker = GuidanceTracker(total_frames=1)
        assert tracker.has_guidance == [False]


class TestMarkGuided:
    """Tests for mark_guided (range marking)."""

    def test_mark_single_frame_range(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(2, 2)
        assert tracker.has_guidance == [False, False, True, False, False]

    def test_mark_full_range(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(0, 4)
        assert tracker.has_guidance == [True, True, True, True, True]

    def test_mark_partial_range(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(3, 6)
        expected = [False, False, False, True, True, True, True, False, False, False]
        assert tracker.has_guidance == expected

    def test_mark_guided_end_idx_inclusive(self):
        """end_idx is inclusive in mark_guided."""
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(1, 3)
        assert tracker.has_guidance[3] is True

    def test_mark_guided_clamps_to_total_frames(self):
        """end_idx beyond total_frames should not cause errors."""
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(3, 100)
        assert tracker.has_guidance == [False, False, False, True, True]

    def test_mark_guided_overlapping_ranges(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(2, 5)
        tracker.mark_guided(4, 7)
        expected = [False, False, True, True, True, True, True, True, False, False]
        assert tracker.has_guidance == expected

    def test_mark_guided_idempotent(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(1, 3)
        tracker.mark_guided(1, 3)
        assert tracker.has_guidance == [False, True, True, True, False]

    def test_mark_guided_start_at_zero(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(0, 2)
        assert tracker.has_guidance == [True, True, True, False, False]


class TestMarkSingleFrame:
    """Tests for mark_single_frame."""

    def test_mark_single_valid_frame(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(2)
        assert tracker.has_guidance == [False, False, True, False, False]

    def test_mark_single_first_frame(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(0)
        assert tracker.has_guidance[0] is True

    def test_mark_single_last_frame(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(4)
        assert tracker.has_guidance[4] is True

    def test_mark_single_negative_index_ignored(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(-1)
        assert tracker.has_guidance == [False] * 5

    def test_mark_single_out_of_bounds_ignored(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(5)
        assert tracker.has_guidance == [False] * 5

    def test_mark_single_large_out_of_bounds_ignored(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(100)
        assert tracker.has_guidance == [False] * 5

    def test_mark_single_idempotent(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(2)
        tracker.mark_single_frame(2)
        assert tracker.has_guidance == [False, False, True, False, False]


class TestGetUnguidancedRanges:
    """Tests for get_unguidanced_ranges."""

    def test_all_unguidanced(self):
        tracker = GuidanceTracker(total_frames=5)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(0, 4)]

    def test_all_guided(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(0, 4)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []

    def test_single_gap_in_middle(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 3)
        tracker.mark_guided(7, 9)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(4, 6)]

    def test_unguidanced_at_start(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(5, 9)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(0, 4)]

    def test_unguidanced_at_end(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 4)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(5, 9)]

    def test_multiple_gaps(self):
        tracker = GuidanceTracker(total_frames=15)
        tracker.mark_single_frame(0)
        tracker.mark_single_frame(5)
        tracker.mark_single_frame(10)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(1, 4), (6, 9), (11, 14)]

    def test_alternating_guided_unguidanced(self):
        tracker = GuidanceTracker(total_frames=6)
        tracker.mark_single_frame(0)
        tracker.mark_single_frame(2)
        tracker.mark_single_frame(4)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(1, 1), (3, 3), (5, 5)]

    def test_empty_tracker(self):
        tracker = GuidanceTracker(total_frames=0)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []

    def test_single_frame_unguidanced(self):
        tracker = GuidanceTracker(total_frames=1)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(0, 0)]

    def test_single_frame_guided(self):
        tracker = GuidanceTracker(total_frames=1)
        tracker.mark_single_frame(0)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []

    def test_ranges_are_inclusive(self):
        """Both start and end of ranges are inclusive indices."""
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_single_frame(0)
        tracker.mark_single_frame(9)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(1, 8)]
        # Verify the range covers 8 frames (1 through 8 inclusive)
        start, end = ranges[0]
        assert end - start + 1 == 8


class TestGetAnchorFrameIndex:
    """Tests for get_anchor_frame_index."""

    def test_anchor_immediately_before_gap(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 4)
        # Gap starts at 5
        anchor = tracker.get_anchor_frame_index(5)
        assert anchor == 4

    def test_anchor_with_gap_at_start(self):
        """No guided frame before the first frame."""
        tracker = GuidanceTracker(total_frames=10)
        anchor = tracker.get_anchor_frame_index(0)
        assert anchor is None

    def test_anchor_skips_unguidanced_frames(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_single_frame(2)
        # Frames 3-4 unguidanced, gap starts at 3
        anchor = tracker.get_anchor_frame_index(3)
        assert anchor == 2

    def test_anchor_with_distant_guided_frame(self):
        tracker = GuidanceTracker(total_frames=20)
        tracker.mark_single_frame(0)
        # Gap starts at 15
        anchor = tracker.get_anchor_frame_index(15)
        assert anchor == 0

    def test_anchor_picks_nearest_guided_frame(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_single_frame(1)
        tracker.mark_single_frame(5)
        # Gap starts at 7
        anchor = tracker.get_anchor_frame_index(7)
        assert anchor == 5

    def test_anchor_none_when_no_guidance_at_all(self):
        tracker = GuidanceTracker(total_frames=10)
        anchor = tracker.get_anchor_frame_index(5)
        assert anchor is None

    def test_anchor_at_boundary(self):
        """When gap starts at index 1 and frame 0 is guided."""
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_single_frame(0)
        anchor = tracker.get_anchor_frame_index(1)
        assert anchor == 0

    def test_anchor_with_get_unguidanced_ranges(self):
        """Integration: use get_unguidanced_ranges to find gaps, then get anchors."""
        tracker = GuidanceTracker(total_frames=20)
        tracker.mark_guided(0, 4)
        tracker.mark_guided(10, 14)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(5, 9), (15, 19)]

        anchor1 = tracker.get_anchor_frame_index(ranges[0][0])
        assert anchor1 == 4

        anchor2 = tracker.get_anchor_frame_index(ranges[1][0])
        assert anchor2 == 14


class TestDebugSummary:
    """Tests for debug_summary."""

    def test_debug_summary_returns_string(self):
        tracker = GuidanceTracker(total_frames=5)
        result = tracker.debug_summary()
        assert isinstance(result, str)

    def test_debug_summary_contains_frame_count(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 4)
        result = tracker.debug_summary()
        assert "Guided frames: 5/10" in result

    def test_debug_summary_contains_unguidanced_range_count(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 4)
        result = tracker.debug_summary()
        assert "Unguidanced ranges: 1" in result

    def test_debug_summary_all_guided(self):
        tracker = GuidanceTracker(total_frames=5)
        tracker.mark_guided(0, 4)
        result = tracker.debug_summary()
        assert "Guided frames: 5/5" in result
        assert "Unguidanced ranges: 0" in result
        assert "Ranges needing structure motion:" not in result

    def test_debug_summary_shows_range_details(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 2)
        result = tracker.debug_summary()
        # Should show the unguidanced range 3-9 (7 frames)
        assert "Frames 3-9 (7 frames)" in result

    def test_debug_summary_visual_line_breaks_every_10(self):
        """Visual output should have line breaks at multiples of 10."""
        tracker = GuidanceTracker(total_frames=25)
        result = tracker.debug_summary()
        assert "  0: " in result
        assert " 10: " in result
        assert " 20: " in result

    def test_debug_summary_visual_chars(self):
        """Guided frames use block char, unguidanced use light shade."""
        tracker = GuidanceTracker(total_frames=3)
        tracker.mark_single_frame(1)
        result = tracker.debug_summary()
        # The visual portion should contain both chars
        assert "\u2588" in result  # guided (full block)
        assert "\u2591" in result  # unguidanced (light shade)

    def test_debug_summary_zero_frames(self):
        tracker = GuidanceTracker(total_frames=0)
        result = tracker.debug_summary()
        assert "Guided frames: 0/0" in result
        assert "Unguidanced ranges: 0" in result


class TestEdgeCases:
    """Edge cases and combined operations."""

    def test_large_tracker(self):
        tracker = GuidanceTracker(total_frames=1000)
        tracker.mark_guided(100, 200)
        tracker.mark_guided(500, 600)
        ranges = tracker.get_unguidanced_ranges()
        assert len(ranges) == 3
        assert ranges[0] == (0, 99)
        assert ranges[1] == (201, 499)
        assert ranges[2] == (601, 999)

    def test_mark_guided_then_single_frame(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 3)
        tracker.mark_single_frame(5)
        assert tracker.has_guidance == [True, True, True, True, False, True, False, False, False, False]

    def test_mark_single_frame_then_guided_range(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_single_frame(5)
        tracker.mark_guided(3, 7)
        expected = [False, False, False, True, True, True, True, True, False, False]
        assert tracker.has_guidance == expected

    def test_adjacent_guided_ranges_merge(self):
        """Two adjacent guided ranges should leave no gap."""
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(0, 4)
        tracker.mark_guided(5, 9)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []

    def test_guided_count_via_sum(self):
        tracker = GuidanceTracker(total_frames=10)
        tracker.mark_guided(2, 5)
        tracker.mark_single_frame(8)
        assert sum(tracker.has_guidance) == 5  # frames 2,3,4,5,8
