"""Tests for source/media/video/frame_extraction.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest


class TestExtractFramesFromVideo:
    """Tests for extract_frames_from_video."""

    def _make_mock_cap(self, is_opened=True, frame_count=10, frames=None):
        """Create a mock cv2.VideoCapture with configurable behavior."""
        cap = MagicMock()
        cap.isOpened.return_value = is_opened
        cap.get.return_value = float(frame_count)

        if frames is None:
            # Default: generate dummy frames
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cap.read.return_value = (True, dummy_frame)
        else:
            cap.read.side_effect = frames

        return cap

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_extracts_all_frames(self, mock_cap_cls, mock_sleep):
        mock_cap = self._make_mock_cap(frame_count=5)
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4")
        assert len(frames) == 5

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_extracts_with_start_frame(self, mock_cap_cls, mock_sleep):
        mock_cap = self._make_mock_cap(frame_count=10)
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4", start_frame=5)
        assert len(frames) == 5  # 10 - 5 = 5

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_extracts_with_num_frames(self, mock_cap_cls, mock_sleep):
        mock_cap = self._make_mock_cap(frame_count=20)
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4", num_frames=5)
        assert len(frames) == 5

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_returns_empty_when_cannot_open(self, mock_cap_cls, mock_sleep):
        mock_cap = self._make_mock_cap(is_opened=False)
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4")
        assert frames == []
        # Should have retried 3 times
        assert mock_sleep.call_count == 2  # sleeps between attempts 1->2 and 2->3

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_returns_empty_when_zero_frame_count(self, mock_cap_cls, mock_sleep):
        mock_cap = self._make_mock_cap(is_opened=True, frame_count=0)
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4")
        assert frames == []

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_handles_read_failure_mid_stream(self, mock_cap_cls, mock_sleep):
        """When cap.read() returns False mid-extraction, should stop and return partial frames."""
        mock_cap = self._make_mock_cap(frame_count=10)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Read 3 frames successfully, then fail
        mock_cap.read.side_effect = [
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
            (False, None),
        ]
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4")
        assert len(frames) == 3

    @patch("source.media.video.frame_extraction.time.sleep")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_retry_on_open_failure(self, mock_cap_cls, mock_sleep):
        """Should retry when video fails to open on first attempt."""
        fail_cap = MagicMock()
        fail_cap.isOpened.return_value = False

        success_cap = self._make_mock_cap(frame_count=5)

        mock_cap_cls.side_effect = [fail_cap, success_cap]

        from source.media.video.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video("/fake/video.mp4")
        assert len(frames) == 5


class TestExtractLastFrameAsImage:
    """Tests for extract_last_frame_as_image."""

    @patch("source.media.video.frame_extraction.cv2.cvtColor")
    @patch("source.media.video.frame_extraction.Image")
    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_extracts_last_frame_successfully(self, mock_cap_cls, mock_image, mock_cvt, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10.0
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        mock_cvt.return_value = dummy_frame
        mock_img_instance = MagicMock()
        mock_image.fromarray.return_value = mock_img_instance

        from source.media.video.frame_extraction import extract_last_frame_as_image
        result = extract_last_frame_as_image("/fake/video.mp4", tmp_path, "test-task")

        assert result is not None
        mock_cap.set.assert_called_once()  # Should seek to last frame
        mock_img_instance.save.assert_called_once()

    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_returns_none_when_cannot_open(self, mock_cap_cls, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_last_frame_as_image
        result = extract_last_frame_as_image("/fake/video.mp4", tmp_path, "test-task")
        assert result is None

    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_returns_none_when_zero_frames(self, mock_cap_cls, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_last_frame_as_image
        result = extract_last_frame_as_image("/fake/video.mp4", tmp_path, "test-task")
        assert result is None
        mock_cap.release.assert_called_once()

    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_returns_none_when_read_fails(self, mock_cap_cls, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10.0
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        from source.media.video.frame_extraction import extract_last_frame_as_image
        result = extract_last_frame_as_image("/fake/video.mp4", tmp_path, "test-task")
        assert result is None

    @patch("source.media.video.frame_extraction.cv2.VideoCapture")
    def test_returns_none_on_oserror(self, mock_cap_cls, tmp_path):
        mock_cap_cls.side_effect = OSError("file not found")

        from source.media.video.frame_extraction import extract_last_frame_as_image
        result = extract_last_frame_as_image("/fake/video.mp4", tmp_path, "test-task")
        assert result is None

    def test_returns_none_when_deps_unavailable(self, tmp_path):
        import source.media.video.frame_extraction as fe
        original = fe._COLOR_MATCH_DEPS_AVAILABLE
        try:
            fe._COLOR_MATCH_DEPS_AVAILABLE = False
            result = fe.extract_last_frame_as_image("/fake/video.mp4", tmp_path, "test-task")
            assert result is None
        finally:
            fe._COLOR_MATCH_DEPS_AVAILABLE = original


class TestFrameExtractionDirect:
    """Direct integration checks using real tiny videos."""

    def _write_video(self, path: Path, frame_count: int = 5, fps: int = 10):
        cv2 = pytest.importorskip("cv2")
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (40, 30),
        )
        assert writer.isOpened()
        for i in range(frame_count):
            frame = np.full((30, 40, 3), (10 + i * 30) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        assert path.exists()
        assert path.stat().st_size > 0

    def test_extract_frames_from_real_video(self, tmp_path):
        from source.media.video.frame_extraction import extract_frames_from_video

        video = tmp_path / "tiny.mp4"
        self._write_video(video, frame_count=6, fps=12)

        frames = extract_frames_from_video(video, start_frame=1, num_frames=3)
        assert isinstance(frames, list)
        assert len(frames) == 3
        assert isinstance(frames[0], np.ndarray)
        assert isinstance(frames[1], np.ndarray)
        assert isinstance(frames[2], np.ndarray)
        assert frames[0].dtype == np.uint8
        assert frames[1].dtype == np.uint8
        assert frames[2].dtype == np.uint8
        assert frames[0].shape == (30, 40, 3)
        assert frames[1].shape == (30, 40, 3)
        assert frames[2].shape == (30, 40, 3)

    def test_extract_last_frame_as_image_real_video(self, tmp_path):
        from source.media.video.frame_extraction import extract_last_frame_as_image

        video = tmp_path / "tiny_last.mp4"
        self._write_video(video, frame_count=4, fps=8)

        out = extract_last_frame_as_image(video, tmp_path / "frames", "task-real")
        assert out is not None
        out_path = Path(out)
        assert out_path.exists()
        assert out_path.suffix == ".png"
        assert "last_frame_ref_tiny_last" in out_path.name
        assert out_path.stat().st_size > 0
