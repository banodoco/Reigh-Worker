"""Tests for source/utils/mask_utils.py."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from source.utils.mask_utils import (
    create_mask_video_from_inactive_indices,
    create_simple_first_frame_mask_video,
)


class TestCreateMaskVideoFromInactiveIndices:
    """Tests for create_mask_video_from_inactive_indices."""

    def test_zero_frames_returns_none(self, tmp_path):
        """Zero total_frames should return None."""
        result = create_mask_video_from_inactive_indices(
            total_frames=0,
            resolution_wh=(320, 240),
            inactive_frame_indices=set(),
            output_path=tmp_path / "mask.mp4",
        )
        assert result is None

    def test_negative_frames_returns_none(self, tmp_path):
        """Negative total_frames should return None."""
        result = create_mask_video_from_inactive_indices(
            total_frames=-5,
            resolution_wh=(320, 240),
            inactive_frame_indices=set(),
            output_path=tmp_path / "mask.mp4",
        )
        assert result is None

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_all_active_frames(self, mock_create_video, tmp_path):
        """When no frames are inactive, all frames should be white (255)."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        result = create_mask_video_from_inactive_indices(
            total_frames=5,
            resolution_wh=(4, 4),
            inactive_frame_indices=set(),
            output_path=output_path,
        )

        assert result == output_path
        mock_create_video.assert_called_once()
        frames = mock_create_video.call_args[0][0]
        assert len(frames) == 5
        for frame in frames:
            assert np.all(frame == 255)
            assert frame.shape == (4, 4, 3)

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_all_inactive_frames(self, mock_create_video, tmp_path):
        """When all frames are inactive, all frames should be black (0)."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        result = create_mask_video_from_inactive_indices(
            total_frames=3,
            resolution_wh=(4, 4),
            inactive_frame_indices={0, 1, 2},
            output_path=output_path,
        )

        assert result == output_path
        frames = mock_create_video.call_args[0][0]
        for frame in frames:
            assert np.all(frame == 0)

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_mixed_active_inactive(self, mock_create_video, tmp_path):
        """Mixed active/inactive frames produce correct pattern."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        result = create_mask_video_from_inactive_indices(
            total_frames=5,
            resolution_wh=(4, 4),
            inactive_frame_indices={0, 2, 4},
            output_path=output_path,
        )

        frames = mock_create_video.call_args[0][0]
        assert len(frames) == 5
        # 0: inactive (black), 1: active (white), 2: inactive, 3: active, 4: inactive
        assert np.all(frames[0] == 0)
        assert np.all(frames[1] == 255)
        assert np.all(frames[2] == 0)
        assert np.all(frames[3] == 255)
        assert np.all(frames[4] == 0)

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_inactive_as_list(self, mock_create_video, tmp_path):
        """Accepts a list (not just set) for inactive_frame_indices."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        result = create_mask_video_from_inactive_indices(
            total_frames=3,
            resolution_wh=(4, 4),
            inactive_frame_indices=[0, 2],
            output_path=output_path,
        )

        frames = mock_create_video.call_args[0][0]
        assert np.all(frames[0] == 0)
        assert np.all(frames[1] == 255)
        assert np.all(frames[2] == 0)

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_resolution_wh_order(self, mock_create_video, tmp_path):
        """resolution_wh is (width, height), frames should be (height, width, 3)."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        create_mask_video_from_inactive_indices(
            total_frames=1,
            resolution_wh=(320, 240),
            inactive_frame_indices=set(),
            output_path=output_path,
        )

        frames = mock_create_video.call_args[0][0]
        assert frames[0].shape == (240, 320, 3)  # (height, width, channels)

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_frame_dtype_uint8(self, mock_create_video, tmp_path):
        """Frames should be uint8."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        create_mask_video_from_inactive_indices(
            total_frames=1,
            resolution_wh=(4, 4),
            inactive_frame_indices=set(),
            output_path=output_path,
        )

        frames = mock_create_video.call_args[0][0]
        assert frames[0].dtype == np.uint8


class TestCreateSimpleFirstFrameMaskVideo:
    """Tests for create_simple_first_frame_mask_video."""

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_first_frame_is_inactive(self, mock_create_video, tmp_path):
        """Only the first frame should be black."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        create_simple_first_frame_mask_video(
            total_frames=5,
            resolution_wh=(4, 4),
            output_path=output_path,
        )

        frames = mock_create_video.call_args[0][0]
        assert len(frames) == 5
        assert np.all(frames[0] == 0)   # first = inactive (black)
        for i in range(1, 5):
            assert np.all(frames[i] == 255)  # rest = active (white)

    def test_zero_frames_returns_none(self, tmp_path):
        """Delegates to main function; zero frames should return None."""
        result = create_simple_first_frame_mask_video(
            total_frames=0,
            resolution_wh=(320, 240),
            output_path=tmp_path / "mask.mp4",
        )
        assert result is None

    @mock.patch("source.utils.mask_utils.create_video_from_frames_list")
    def test_single_frame_all_black(self, mock_create_video, tmp_path):
        """With 1 total frame, that frame is the first frame = black."""
        output_path = tmp_path / "mask.mp4"
        mock_create_video.return_value = output_path

        create_simple_first_frame_mask_video(
            total_frames=1,
            resolution_wh=(4, 4),
            output_path=output_path,
        )

        frames = mock_create_video.call_args[0][0]
        assert len(frames) == 1
        assert np.all(frames[0] == 0)
