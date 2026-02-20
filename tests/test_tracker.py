"""Tests for source/media/structure/tracker.py.

Covers GuidanceTracker: initialization, marking, range queries,
anchor frame lookup, boundary conditions, and debug output.
"""

import pytest

from source.media.structure.tracker import GuidanceTracker


class TestGuidanceTrackerInit:
    def test_initial_state_all_unguidanced(self):
        tracker = GuidanceTracker(10)
        assert tracker.total_frames == 10
        assert len(tracker.has_guidance) == 10
        assert all(g is False for g in tracker.has_guidance)

    def test_single_frame_tracker(self):
        tracker = GuidanceTracker(1)
        assert len(tracker.has_guidance) == 1
        assert tracker.has_guidance[0] is False

    def test_zero_frames(self):
        tracker = GuidanceTracker(0)
        assert len(tracker.has_guidance) == 0


class TestMarkGuided:
    def test_mark_guided_range(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(2, 5)
        expected = [False, False, True, True, True, True, False, False, False, False]
        assert tracker.has_guidance == expected

    def test_mark_guided_single_frame_via_range(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(3, 3)
        assert tracker.has_guidance[3] is True
        assert sum(tracker.has_guidance) == 1

    def test_mark_guided_entire_range(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(0, 4)
        assert all(tracker.has_guidance)

    def test_mark_guided_clamps_to_total_frames(self):
        """end_idx beyond total_frames is safely clamped."""
        tracker = GuidanceTracker(5)
        tracker.mark_guided(3, 100)
        expected = [False, False, False, True, True]
        assert tracker.has_guidance == expected

    def test_mark_guided_overlapping_ranges(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(2, 5)
        tracker.mark_guided(4, 7)
        expected = [False, False, True, True, True, True, True, True, False, False]
        assert tracker.has_guidance == expected

    def test_mark_guided_first_frame(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(0, 0)
        assert tracker.has_guidance[0] is True
        assert sum(tracker.has_guidance) == 1

    def test_mark_guided_last_frame(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(4, 4)
        assert tracker.has_guidance[4] is True
        assert sum(tracker.has_guidance) == 1


class TestMarkSingleFrame:
    def test_mark_single_frame(self):
        tracker = GuidanceTracker(10)
        tracker.mark_single_frame(5)
        assert tracker.has_guidance[5] is True
        assert sum(tracker.has_guidance) == 1

    def test_mark_single_frame_first(self):
        tracker = GuidanceTracker(5)
        tracker.mark_single_frame(0)
        assert tracker.has_guidance[0] is True

    def test_mark_single_frame_last(self):
        tracker = GuidanceTracker(5)
        tracker.mark_single_frame(4)
        assert tracker.has_guidance[4] is True

    def test_mark_single_frame_out_of_range_negative(self):
        """Negative index should be ignored (bounds check: 0 <= idx)."""
        tracker = GuidanceTracker(5)
        tracker.mark_single_frame(-1)
        assert sum(tracker.has_guidance) == 0

    def test_mark_single_frame_out_of_range_too_high(self):
        """Index >= total_frames should be ignored."""
        tracker = GuidanceTracker(5)
        tracker.mark_single_frame(5)
        assert sum(tracker.has_guidance) == 0

    def test_mark_single_frame_idempotent(self):
        tracker = GuidanceTracker(5)
        tracker.mark_single_frame(2)
        tracker.mark_single_frame(2)
        assert sum(tracker.has_guidance) == 1


class TestGetUnguidancedRanges:
    def test_all_unguidanced(self):
        tracker = GuidanceTracker(5)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(0, 4)]

    def test_all_guided(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(0, 4)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []

    def test_single_gap_in_middle(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(0, 2)
        tracker.mark_guided(7, 9)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(3, 6)]

    def test_multiple_gaps(self):
        tracker = GuidanceTracker(10)
        tracker.mark_single_frame(0)
        tracker.mark_single_frame(3)
        tracker.mark_single_frame(7)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(1, 2), (4, 6), (8, 9)]

    def test_gap_at_start(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(3, 4)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(0, 2)]

    def test_gap_at_end(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(0, 1)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(2, 4)]

    def test_alternating_guided_unguidanced(self):
        tracker = GuidanceTracker(6)
        tracker.mark_single_frame(0)
        tracker.mark_single_frame(2)
        tracker.mark_single_frame(4)
        # Unguidanced: 1, 3, 5
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(1, 1), (3, 3), (5, 5)]

    def test_empty_tracker(self):
        tracker = GuidanceTracker(0)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []

    def test_single_frame_unguidanced(self):
        tracker = GuidanceTracker(1)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == [(0, 0)]

    def test_single_frame_guided(self):
        tracker = GuidanceTracker(1)
        tracker.mark_single_frame(0)
        ranges = tracker.get_unguidanced_ranges()
        assert ranges == []


class TestGetAnchorFrameIndex:
    def test_anchor_found(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(0, 3)
        # Unguidanced range starts at 4; anchor should be 3
        assert tracker.get_anchor_frame_index(4) == 3

    def test_anchor_not_found(self):
        tracker = GuidanceTracker(10)
        # No guided frames before index 0
        assert tracker.get_anchor_frame_index(0) is None

    def test_anchor_not_found_no_guidance_before(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(5, 9)
        # No guided frames before index 3
        assert tracker.get_anchor_frame_index(3) is None

    def test_anchor_skips_unguidanced_to_find_guided(self):
        tracker = GuidanceTracker(10)
        tracker.mark_single_frame(2)
        # Frames 3-6 unguidanced, looking from 7
        assert tracker.get_anchor_frame_index(7) == 2

    def test_anchor_at_frame_zero(self):
        tracker = GuidanceTracker(10)
        tracker.mark_single_frame(0)
        assert tracker.get_anchor_frame_index(1) == 0

    def test_anchor_immediately_preceding(self):
        tracker = GuidanceTracker(10)
        tracker.mark_single_frame(4)
        assert tracker.get_anchor_frame_index(5) == 4

    def test_anchor_with_multiple_guided_returns_nearest(self):
        """Should return the last guided frame before the range (nearest one)."""
        tracker = GuidanceTracker(10)
        tracker.mark_single_frame(1)
        tracker.mark_single_frame(3)
        tracker.mark_single_frame(5)
        # Looking from 7, nearest guided before 7 is 5
        assert tracker.get_anchor_frame_index(7) == 5


class TestDebugSummary:
    def test_debug_summary_format(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(1, 3)
        summary = tracker.debug_summary()
        assert "Guided frames: 3/5" in summary
        assert "Unguidanced ranges: 2" in summary

    def test_debug_summary_contains_visual(self):
        tracker = GuidanceTracker(5)
        tracker.mark_guided(0, 4)
        summary = tracker.debug_summary()
        # All guided: should contain filled blocks
        assert "\u2588" in summary

    def test_debug_summary_all_unguidanced(self):
        tracker = GuidanceTracker(3)
        summary = tracker.debug_summary()
        assert "Guided frames: 0/3" in summary
        assert "Unguidanced ranges: 1" in summary
        assert "Ranges needing structure motion:" in summary

    def test_debug_summary_no_unguidanced(self):
        tracker = GuidanceTracker(3)
        tracker.mark_guided(0, 2)
        summary = tracker.debug_summary()
        assert "Guided frames: 3/3" in summary
        assert "Unguidanced ranges: 0" in summary
        # Should NOT contain "Ranges needing structure motion" when none exist
        assert "Ranges needing structure motion:" not in summary

    def test_debug_summary_shows_range_details(self):
        tracker = GuidanceTracker(10)
        tracker.mark_guided(0, 2)
        summary = tracker.debug_summary()
        # Should show the unguidanced range 3-9 (7 frames)
        assert "Frames 3-9" in summary
        assert "7 frames" in summary
