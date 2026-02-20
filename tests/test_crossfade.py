"""Tests for source/media/video/crossfade.py."""

import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from source.media.video.crossfade import (
    crossfade_ease,
    cross_fade_overlap_frames,
    stitch_videos_with_crossfade,
)


class TestCrossfadeEase:
    """Tests for the cosine ease-in-out function."""

    def test_zero_returns_zero(self):
        assert crossfade_ease(0.0) == pytest.approx(0.0)

    def test_one_returns_one(self):
        assert crossfade_ease(1.0) == pytest.approx(1.0)

    def test_midpoint_returns_half(self):
        assert crossfade_ease(0.5) == pytest.approx(0.5)

    def test_monotonically_increasing(self):
        values = [crossfade_ease(t / 100.0) for t in range(101)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10

    def test_quarter_point(self):
        # At t=0.25, cosine ease gives (1 - cos(0.25*pi)) / 2
        expected = (1 - math.cos(0.25 * math.pi)) / 2.0
        assert crossfade_ease(0.25) == pytest.approx(expected)


class TestCrossFadeOverlapFrames:
    """Tests for cross_fade_overlap_frames."""

    def _make_frames(self, n, value, shape=(100, 100, 3)):
        """Create n uniform frames of a given value."""
        return [np.full(shape, value, dtype=np.uint8) for _ in range(n)]

    def test_zero_overlap_returns_empty(self):
        seg1 = self._make_frames(5, 0)
        seg2 = self._make_frames(5, 255)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=0)
        assert result == []

    def test_returns_correct_number_of_frames(self):
        seg1 = self._make_frames(10, 0)
        seg2 = self._make_frames(10, 255)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=5)
        assert len(result) == 5

    def test_overlap_limited_to_segment_length(self):
        seg1 = self._make_frames(3, 0)
        seg2 = self._make_frames(3, 255)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=10)
        assert len(result) == 3

    def test_output_frames_are_uint8(self):
        seg1 = self._make_frames(5, 100)
        seg2 = self._make_frames(5, 200)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3)
        for frame in result:
            assert frame.dtype == np.uint8

    def test_output_frame_shape_matches_segment2(self):
        seg1 = self._make_frames(5, 100, shape=(80, 120, 3))
        seg2 = self._make_frames(5, 200, shape=(100, 100, 3))
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3)
        for frame in result:
            assert frame.shape == (100, 100, 3)

    def test_linear_mode(self):
        seg1 = self._make_frames(5, 0)
        seg2 = self._make_frames(5, 255)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3, mode="linear")
        assert len(result) == 3
        for frame in result:
            assert frame.dtype == np.uint8

    def test_linear_sharp_mode(self):
        seg1 = self._make_frames(5, 0)
        seg2 = self._make_frames(5, 255)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3, mode="linear_sharp")
        assert len(result) == 3

    def test_unknown_mode_falls_back_to_linear(self):
        seg1 = self._make_frames(5, 0)
        seg2 = self._make_frames(5, 255)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3, mode="unknown")
        assert len(result) == 3

    def test_empty_segment2_returns_empty(self):
        seg1 = self._make_frames(5, 0)
        seg2 = []
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3)
        assert result == []

    def test_blending_produces_intermediate_values(self):
        """With black and white frames, blended result should have intermediate values."""
        seg1 = self._make_frames(5, 0)
        seg2 = self._make_frames(5, 254)
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3, mode="linear")
        # The middle frame (index 1 of 3) should have intermediate brightness
        mid_frame = result[1]
        mean_val = mid_frame.mean()
        assert 30 < mean_val < 230  # Not purely black or white

    def test_different_resolution_segments_resized(self):
        """Segment1 at different resolution should be resized to match segment2."""
        seg1 = self._make_frames(5, 100, shape=(50, 50, 3))
        seg2 = self._make_frames(5, 200, shape=(100, 100, 3))
        result = cross_fade_overlap_frames(seg1, seg2, overlap_count=3)
        assert len(result) == 3
        for frame in result:
            assert frame.shape == (100, 100, 3)


class TestStitchVideosWithCrossfade:
    """Tests for stitch_videos_with_crossfade."""

    def test_fewer_than_two_videos_raises(self):
        with pytest.raises(ValueError, match="at least 2 videos"):
            stitch_videos_with_crossfade(
                video_paths=[Path("/fake/a.mp4")],
                blend_frame_counts=[],
                output_video_path=Path("/fake/out.mp4"),
                fps=16.0,
            )

    def test_mismatched_blend_counts_raises(self):
        with pytest.raises(ValueError, match="blend_frame_counts must have length"):
            stitch_videos_with_crossfade(
                video_paths=[Path("/fake/a.mp4"), Path("/fake/b.mp4"), Path("/fake/c.mp4")],
                blend_frame_counts=[3],  # Should be 2
                output_video_path=Path("/fake/out.mp4"),
                fps=16.0,
            )

    @patch("source.media.video.crossfade.create_video_from_frames_list")
    @patch("source.media.video.crossfade.extract_frames_from_video")
    def test_successful_stitch(self, mock_extract, mock_create, tmp_path):
        """Two videos with 3-frame crossfade should stitch successfully."""
        shape = (100, 100, 3)
        frames_a = [np.full(shape, 100, dtype=np.uint8) for _ in range(10)]
        frames_b = [np.full(shape, 200, dtype=np.uint8) for _ in range(10)]
        mock_extract.side_effect = [frames_a, frames_b]

        output_path = tmp_path / "stitched.mp4"
        # Mock create_video_from_frames_list to create the file
        def fake_create(frames, path, fps, resolution):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake video data")
            return path
        mock_create.side_effect = fake_create

        result = stitch_videos_with_crossfade(
            video_paths=[Path("/fake/a.mp4"), Path("/fake/b.mp4")],
            blend_frame_counts=[3],
            output_video_path=output_path,
            fps=16.0,
        )
        assert result == output_path
        mock_create.assert_called_once()

    @patch("source.media.video.crossfade.extract_frames_from_video")
    def test_raises_when_extraction_fails(self, mock_extract):
        """Should raise ValueError when frame extraction fails."""
        mock_extract.return_value = []  # No frames extracted

        with pytest.raises(ValueError, match="Failed to extract frames"):
            stitch_videos_with_crossfade(
                video_paths=[Path("/fake/a.mp4"), Path("/fake/b.mp4")],
                blend_frame_counts=[3],
                output_video_path=Path("/fake/out.mp4"),
                fps=16.0,
            )

    @patch("source.media.video.crossfade.create_video_from_frames_list")
    @patch("source.media.video.crossfade.extract_frames_from_video")
    def test_zero_blend_frames(self, mock_extract, mock_create, tmp_path):
        """With zero blend frames, videos should be concatenated without crossfade."""
        shape = (100, 100, 3)
        frames_a = [np.full(shape, 100, dtype=np.uint8) for _ in range(5)]
        frames_b = [np.full(shape, 200, dtype=np.uint8) for _ in range(5)]
        mock_extract.side_effect = [frames_a, frames_b]

        output_path = tmp_path / "stitched.mp4"
        def fake_create(frames, path, fps, resolution):
            path.write_bytes(b"fake video")
            return path
        mock_create.side_effect = fake_create

        result = stitch_videos_with_crossfade(
            video_paths=[Path("/fake/a.mp4"), Path("/fake/b.mp4")],
            blend_frame_counts=[0],
            output_video_path=output_path,
            fps=16.0,
        )
        assert result == output_path
        # All 10 frames should be in the final video
        called_frames = mock_create.call_args[0][0]
        assert len(called_frames) == 10

    @patch("source.media.video.crossfade.create_video_from_frames_list")
    @patch("source.media.video.crossfade.extract_frames_from_video")
    def test_three_videos_with_crossfade(self, mock_extract, mock_create, tmp_path):
        """Three videos with crossfade blending at both boundaries."""
        shape = (100, 100, 3)
        frames_a = [np.full(shape, 50, dtype=np.uint8) for _ in range(10)]
        frames_b = [np.full(shape, 150, dtype=np.uint8) for _ in range(10)]
        frames_c = [np.full(shape, 250, dtype=np.uint8) for _ in range(10)]
        mock_extract.side_effect = [frames_a, frames_b, frames_c]

        output_path = tmp_path / "stitched.mp4"
        def fake_create(frames, path, fps, resolution):
            path.write_bytes(b"fake video data")
            return path
        mock_create.side_effect = fake_create

        result = stitch_videos_with_crossfade(
            video_paths=[Path("/fake/a.mp4"), Path("/fake/b.mp4"), Path("/fake/c.mp4")],
            blend_frame_counts=[2, 2],
            output_video_path=output_path,
            fps=16.0,
        )
        assert result == output_path
