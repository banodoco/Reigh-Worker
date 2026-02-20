"""Tests for source/task_handlers/join/clip_validator.py."""

import pytest

from source.task_handlers.join.clip_validator import calculate_min_clip_frames


class TestCalculateMinClipFrames:
    """Tests for calculate_min_clip_frames (pure logic, no I/O)."""

    def test_replace_mode_formula(self):
        """Replace mode: gap + 2*context."""
        result = calculate_min_clip_frames(
            gap_frame_count=53,
            context_frame_count=8,
            replace_mode=True,
        )
        assert result == 53 + 2 * 8  # 69

    def test_insert_mode_formula(self):
        """Insert mode: 2*context only."""
        result = calculate_min_clip_frames(
            gap_frame_count=53,
            context_frame_count=8,
            replace_mode=False,
        )
        assert result == 2 * 8  # 16

    def test_replace_mode_with_small_values(self):
        result = calculate_min_clip_frames(
            gap_frame_count=1,
            context_frame_count=1,
            replace_mode=True,
        )
        assert result == 1 + 2 * 1  # 3

    def test_insert_mode_with_small_values(self):
        result = calculate_min_clip_frames(
            gap_frame_count=1,
            context_frame_count=1,
            replace_mode=False,
        )
        assert result == 2

    def test_replace_mode_large_gap(self):
        result = calculate_min_clip_frames(
            gap_frame_count=200,
            context_frame_count=16,
            replace_mode=True,
        )
        assert result == 200 + 32  # 232

    def test_insert_mode_ignores_gap(self):
        """In insert mode, gap_frame_count has no effect on minimum."""
        result_small = calculate_min_clip_frames(1, 8, False)
        result_large = calculate_min_clip_frames(1000, 8, False)
        assert result_small == result_large == 16

    def test_zero_context_replace_mode(self):
        result = calculate_min_clip_frames(50, 0, True)
        assert result == 50

    def test_zero_context_insert_mode(self):
        result = calculate_min_clip_frames(50, 0, False)
        assert result == 0

    def test_zero_gap_replace_mode(self):
        result = calculate_min_clip_frames(0, 8, True)
        assert result == 16  # 0 + 2*8
