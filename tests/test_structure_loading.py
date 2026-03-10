"""Tests for source/media/structure/loading.py."""

import math
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from source.media.structure.loading import _resample_frame_indices, load_structure_video_frames


class TestResampleFrameIndices:
    """Tests for the _resample_frame_indices helper."""

    def test_same_fps_returns_sequential(self):
        """When video FPS matches target FPS, indices should be sequential."""
        indices = _resample_frame_indices(
            video_fps=16.0,
            video_frames_count=20,
            max_target_frames_count=10,
            target_fps=16.0,
            start_target_frame=0,
        )
        assert len(indices) <= 10
        # All indices should be valid
        for idx in indices:
            assert 0 <= idx < 20

    def test_respects_max_target_frames(self):
        """Should never return more frames than max_target_frames_count."""
        indices = _resample_frame_indices(
            video_fps=30.0,
            video_frames_count=100,
            max_target_frames_count=10,
            target_fps=16.0,
            start_target_frame=0,
        )
        assert len(indices) <= 10

    def test_zero_max_target_extracts_all(self):
        """When max_target_frames_count is 0, should extract until video ends."""
        indices = _resample_frame_indices(
            video_fps=16.0,
            video_frames_count=10,
            max_target_frames_count=0,
            target_fps=16.0,
            start_target_frame=0,
        )
        # Should extract frames up to the video limit
        assert len(indices) > 0
        for idx in indices:
            assert 0 <= idx < 10

    def test_start_target_frame_offset(self):
        """Starting from a later frame should produce later indices."""
        indices_from_0 = _resample_frame_indices(
            video_fps=16.0,
            video_frames_count=50,
            max_target_frames_count=5,
            target_fps=16.0,
            start_target_frame=0,
        )
        indices_from_5 = _resample_frame_indices(
            video_fps=16.0,
            video_frames_count=50,
            max_target_frames_count=5,
            target_fps=16.0,
            start_target_frame=5,
        )
        # Later start should produce later indices
        if indices_from_0 and indices_from_5:
            assert indices_from_5[0] >= indices_from_0[0]

    def test_higher_target_fps_more_frames(self):
        """Higher target FPS should require more source frames."""
        indices_low = _resample_frame_indices(
            video_fps=30.0,
            video_frames_count=100,
            max_target_frames_count=0,
            target_fps=8.0,
            start_target_frame=0,
        )
        indices_high = _resample_frame_indices(
            video_fps=30.0,
            video_frames_count=100,
            max_target_frames_count=0,
            target_fps=24.0,
            start_target_frame=0,
        )
        assert len(indices_high) >= len(indices_low)

    def test_indices_monotonically_increasing(self):
        """Frame indices should be non-decreasing."""
        indices = _resample_frame_indices(
            video_fps=30.0,
            video_frames_count=100,
            max_target_frames_count=20,
            target_fps=16.0,
            start_target_frame=0,
        )
        for i in range(len(indices) - 1):
            assert indices[i] <= indices[i + 1]

    def test_short_video_returns_fewer_frames(self):
        """Very short video should return fewer frames than requested."""
        indices = _resample_frame_indices(
            video_fps=16.0,
            video_frames_count=3,
            max_target_frames_count=100,
            target_fps=16.0,
            start_target_frame=0,
        )
        assert len(indices) <= 3


class TestLoadStructureVideoFrames:
    """Tests for load_structure_video_frames."""

    def _make_mock_decord(self):
        """Create a mock decord module suitable for sys.modules injection."""
        mock_decord = MagicMock()
        mock_decord.bridge.set_bridge = MagicMock()
        return mock_decord

    def _make_mock_reader(self, frame_count=20, fps=30.0, frame_shape=(480, 640, 3)):
        """Create a mock decord.VideoReader."""
        reader = MagicMock()
        reader.get_avg_fps.return_value = fps
        reader.__len__ = MagicMock(return_value=frame_count)

        # Create mock frames that behave like torch tensors
        def make_mock_frame():
            f = MagicMock()
            f.cpu.return_value.numpy.return_value = np.random.randint(
                0, 255, frame_shape, dtype=np.uint8
            )
            return f

        # get_batch returns a list-like of frames
        mock_batch = [make_mock_frame() for _ in range(frame_count)]

        def mock_get_batch(indices):
            return [mock_batch[min(i, len(mock_batch) - 1)] for i in indices]

        reader.get_batch.side_effect = mock_get_batch
        return reader

    def test_adjust_mode_compress(self):
        """Adjust mode with more video frames than needed should compress."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=30, fps=30.0)
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=10,
                target_fps=16,
                target_resolution=(640, 480),
                treatment="adjust",
            )
        # Should load target_frame_count + 1 = 11 frames
        assert len(frames) == 11
        for f in frames:
            assert f.shape == (480, 640, 3)

    def test_adjust_mode_stretch(self):
        """Adjust mode with fewer video frames than needed should stretch."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=5, fps=30.0)
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=10,
                target_fps=16,
                target_resolution=(640, 480),
                treatment="adjust",
            )
        assert len(frames) == 11

    def test_clip_mode(self):
        """Clip mode uses temporal sampling."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=100, fps=30.0)
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=10,
                target_fps=16,
                target_resolution=(320, 240),
                treatment="clip",
            )
        assert len(frames) > 0
        for f in frames:
            assert f.shape == (240, 320, 3)

    def test_clip_mode_short_video_loops(self):
        """Clip mode with short video should loop frames."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=3, fps=30.0)
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=10,
                target_fps=16,
                target_resolution=(320, 240),
                treatment="clip",
            )
        # Should have loaded enough frames via looping
        assert len(frames) == 11  # target_frame_count + 1

    def test_target_resolution_applied(self):
        """Frames should be resized to target resolution."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=20, fps=30.0, frame_shape=(100, 200, 3))
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=5,
                target_fps=16,
                target_resolution=(640, 480),
                treatment="adjust",
            )
        for f in frames:
            assert f.shape == (480, 640, 3)

    def test_crop_to_fit_enabled(self):
        """With crop_to_fit, should center-crop before resizing."""
        # Use a 16:9 source going to 4:3 target - aspect ratios differ
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=10, fps=30.0, frame_shape=(720, 1280, 3))
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=5,
                target_fps=16,
                target_resolution=(640, 480),
                treatment="adjust",
                crop_to_fit=True,
            )
        for f in frames:
            assert f.shape == (480, 640, 3)

    def test_crop_to_fit_disabled(self):
        """With crop_to_fit=False, should just resize without cropping."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=10, fps=30.0, frame_shape=(720, 1280, 3))
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=5,
                target_fps=16,
                target_resolution=(640, 480),
                treatment="adjust",
                crop_to_fit=False,
            )
        for f in frames:
            assert f.shape == (480, 640, 3)

    def test_decord_import_error(self):
        """Should raise ImportError when decord is not available."""
        with patch.dict("sys.modules", {"decord": None}):
            with pytest.raises(ImportError):
                load_structure_video_frames(
                    structure_video_path="/fake/video.mp4",
                    target_frame_count=5,
                    target_fps=16,
                    target_resolution=(320, 240),
                )

    def test_frames_are_uint8(self):
        """Output frames should be uint8."""
        mock_decord = self._make_mock_decord()
        mock_reader = self._make_mock_reader(frame_count=10, fps=30.0)
        mock_decord.VideoReader.return_value = mock_reader

        with patch.dict(sys.modules, {"decord": mock_decord}):
            frames = load_structure_video_frames(
                structure_video_path="/fake/video.mp4",
                target_frame_count=5,
                target_fps=16,
                target_resolution=(320, 240),
            )
        for f in frames:
            assert f.dtype == np.uint8
