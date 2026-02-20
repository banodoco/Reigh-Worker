"""Tests for source/media/structure/download.py."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


class TestDownloadAndExtractMotionFrames:
    """Tests for download_and_extract_motion_frames."""

    @patch("source.media.structure.download.generation_logger")
    def test_local_path_with_decord(self, mock_logger, tmp_path):
        """Extracts frames from a local path using decord."""
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "motion.mp4"
        video_path.write_bytes(b"fake video")

        # Mock decord at import level
        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.__len__ = lambda self: 100

        # Create fake tensor batch
        mock_frame = MagicMock()
        mock_frame.cpu.return_value.numpy.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_batch = MagicMock()
        mock_batch.__len__ = lambda self: 10
        mock_batch.__getitem__ = lambda self, i: mock_frame
        mock_vr.get_batch.return_value = mock_batch

        mock_decord.VideoReader.return_value = mock_vr

        with patch.dict("sys.modules", {"decord": mock_decord, "decord.bridge": mock_decord.bridge}):
            frames = download_and_extract_motion_frames(
                str(video_path), frame_start=0, frame_count=10, download_dir=tmp_path
            )

        assert len(frames) == 10
        for f in frames:
            assert f.shape == (720, 1280, 3)
            assert f.dtype == np.uint8

    @patch("source.media.structure.download.generation_logger")
    def test_local_path_cv2_fallback(self, mock_logger, tmp_path):
        """Falls back to cv2 when decord is not available."""
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "motion.mp4"
        video_path.write_bytes(b"fake video")

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100.0  # FRAME_COUNT
        mock_cap.read.side_effect = [
            (True, np.zeros((720, 1280, 3), dtype=np.uint8)) for _ in range(5)
        ] + [(False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 1
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.cvtColor.side_effect = lambda frame, code: frame  # passthrough

        # Make decord import fail, cv2 succeed
        def fake_import(name, *args, **kwargs):
            if name == "decord":
                raise ModuleNotFoundError("No module named 'decord'")
            if name == "cv2":
                return mock_cv2
            return MagicMock()

        with patch("builtins.__import__", side_effect=fake_import):
            # Need to bypass the existing decord import attempt
            # Instead, patch sys.modules to remove decord and inject cv2
            import sys
            saved_decord = sys.modules.pop("decord", None)
            sys.modules["cv2"] = mock_cv2
            try:
                # Force re-execution by calling with decord unavailable
                frames = download_and_extract_motion_frames(
                    str(video_path), frame_start=0, frame_count=5, download_dir=tmp_path
                )
            finally:
                if saved_decord:
                    sys.modules["decord"] = saved_decord
                sys.modules.pop("cv2", None)

        assert len(frames) == 5
        mock_cap.release.assert_called_once()

    @patch("source.media.structure.download.generation_logger")
    def test_url_download(self, mock_logger, tmp_path):
        """Downloads video from URL then extracts frames."""
        # We test the download logic separately by mocking requests
        mock_response = MagicMock()
        mock_response.content = b"fake video bytes"
        mock_response.raise_for_status = MagicMock()

        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.__len__ = lambda self: 50

        mock_frame = MagicMock()
        mock_frame.cpu.return_value.numpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_batch = MagicMock()
        mock_batch.__len__ = lambda self: 5
        mock_batch.__getitem__ = lambda self, i: mock_frame
        mock_vr.get_batch.return_value = mock_batch
        mock_decord.VideoReader.return_value = mock_vr

        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch.dict("sys.modules", {"decord": mock_decord, "decord.bridge": mock_decord.bridge}):
            from source.media.structure.download import download_and_extract_motion_frames

            frames = download_and_extract_motion_frames(
                "https://example.com/video.mp4",
                frame_start=10,
                frame_count=5,
                download_dir=tmp_path,
            )

        mock_get.assert_called_once_with("https://example.com/video.mp4", timeout=120)
        assert len(frames) == 5
        # Verify the file was written
        assert (tmp_path / "structure_motion.mp4").exists()

    @patch("source.media.structure.download.generation_logger")
    def test_local_file_not_found_raises(self, mock_logger, tmp_path):
        """Raises ValueError when local file doesn't exist."""
        from source.media.structure.download import download_and_extract_motion_frames

        with pytest.raises(ValueError, match="not found"):
            download_and_extract_motion_frames(
                "/nonexistent/video.mp4",
                frame_start=0,
                frame_count=10,
                download_dir=tmp_path,
            )

    @patch("source.media.structure.download.generation_logger")
    def test_frame_start_exceeds_total_raises(self, mock_logger, tmp_path):
        """Raises ValueError when frame_start >= total frames (decord path)."""
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "short.mp4"
        video_path.write_bytes(b"fake video")

        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.__len__ = lambda self: 5  # Only 5 frames
        mock_decord.VideoReader.return_value = mock_vr

        with patch.dict("sys.modules", {"decord": mock_decord, "decord.bridge": mock_decord.bridge}):
            with pytest.raises(ValueError, match="frame_start 10 >= total frames 5"):
                download_and_extract_motion_frames(
                    str(video_path),
                    frame_start=10,
                    frame_count=5,
                    download_dir=tmp_path,
                )

    @patch("source.media.structure.download.generation_logger")
    def test_fewer_frames_available_than_requested(self, mock_logger, tmp_path):
        """When fewer frames available than requested, returns what's available."""
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "motion.mp4"
        video_path.write_bytes(b"fake video")

        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.__len__ = lambda self: 8  # Only 8 total

        mock_frame = MagicMock()
        mock_frame.cpu.return_value.numpy.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        # Only 3 frames available from index 5 (5, 6, 7)
        mock_batch = MagicMock()
        mock_batch.__len__ = lambda self: 3
        mock_batch.__getitem__ = lambda self, i: mock_frame
        mock_vr.get_batch.return_value = mock_batch

        mock_decord.VideoReader.return_value = mock_vr

        with patch.dict("sys.modules", {"decord": mock_decord, "decord.bridge": mock_decord.bridge}):
            frames = download_and_extract_motion_frames(
                str(video_path),
                frame_start=5,
                frame_count=10,  # Requesting 10 but only 3 available
                download_dir=tmp_path,
            )

        assert len(frames) == 3

    @patch("source.media.structure.download.generation_logger")
    def test_file_scheme_url(self, mock_logger, tmp_path):
        """Handles file:// URLs by extracting the local path."""
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "motion.mp4"
        video_path.write_bytes(b"fake video")

        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.__len__ = lambda self: 20

        mock_frame = MagicMock()
        mock_frame.cpu.return_value.numpy.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_batch = MagicMock()
        mock_batch.__len__ = lambda self: 5
        mock_batch.__getitem__ = lambda self, i: mock_frame
        mock_vr.get_batch.return_value = mock_batch
        mock_decord.VideoReader.return_value = mock_vr

        with patch.dict("sys.modules", {"decord": mock_decord, "decord.bridge": mock_decord.bridge}):
            frames = download_and_extract_motion_frames(
                f"file://{video_path}",
                frame_start=0,
                frame_count=5,
                download_dir=tmp_path,
            )

        assert len(frames) == 5

    @patch("source.media.structure.download.generation_logger")
    def test_non_uint8_frames_clipped(self, mock_logger, tmp_path):
        """Frames with non-uint8 dtype are clipped and converted."""
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "motion.mp4"
        video_path.write_bytes(b"fake video")

        mock_decord = MagicMock()
        mock_vr = MagicMock()
        mock_vr.__len__ = lambda self: 10

        # Return float32 frame that needs conversion
        float_frame = np.array([[[300.0, -10.0, 128.5]]], dtype=np.float32)
        mock_frame = MagicMock()
        mock_frame.cpu.return_value.numpy.return_value = float_frame

        mock_batch = MagicMock()
        mock_batch.__len__ = lambda self: 1
        mock_batch.__getitem__ = lambda self, i: mock_frame
        mock_vr.get_batch.return_value = mock_batch
        mock_decord.VideoReader.return_value = mock_vr

        with patch.dict("sys.modules", {"decord": mock_decord, "decord.bridge": mock_decord.bridge}):
            frames = download_and_extract_motion_frames(
                str(video_path),
                frame_start=0,
                frame_count=1,
                download_dir=tmp_path,
            )

        assert len(frames) == 1
        assert frames[0].dtype == np.uint8
        # 300 should be clipped to 255, -10 to 0
        assert frames[0][0, 0, 0] == 255
        assert frames[0][0, 0, 1] == 0


class TestDownloadAndExtractMotionFramesDirect:
    """Direct behavior checks with real tiny videos."""

    def _write_tiny_video(self, video_path: Path, frame_count: int = 6, fps: int = 8):
        cv2 = pytest.importorskip("cv2")
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (32, 24),
        )
        assert writer.isOpened()
        for i in range(frame_count):
            # BGR frame with changing values to ensure non-identical frames.
            frame = np.full((24, 32, 3), (i * 20) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        assert video_path.exists()
        assert video_path.stat().st_size > 0

    def test_local_video_extracts_requested_slice(self, tmp_path):
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "tiny.mp4"
        self._write_tiny_video(video_path, frame_count=6, fps=8)

        frames = download_and_extract_motion_frames(
            str(video_path),
            frame_start=2,
            frame_count=3,
            download_dir=tmp_path / "dl",
        )

        assert isinstance(frames, list)
        assert len(frames) == 3
        assert isinstance(frames[0], np.ndarray)
        assert isinstance(frames[1], np.ndarray)
        assert isinstance(frames[2], np.ndarray)
        assert frames[0].dtype == np.uint8
        assert frames[1].dtype == np.uint8
        assert frames[2].dtype == np.uint8
        assert frames[0].ndim == 3
        assert frames[1].ndim == 3
        assert frames[2].ndim == 3
        assert frames[0].shape[2] == 3
        assert frames[1].shape[2] == 3
        assert frames[2].shape[2] == 3
        assert int(frames[0].min()) >= 0
        assert int(frames[1].min()) >= 0
        assert int(frames[2].min()) >= 0
        assert int(frames[0].max()) <= 255
        assert int(frames[1].max()) <= 255
        assert int(frames[2].max()) <= 255

    def test_file_url_path_works_and_truncates_to_available_frames(self, tmp_path):
        from source.media.structure.download import download_and_extract_motion_frames

        video_path = tmp_path / "tiny_file_url.mp4"
        self._write_tiny_video(video_path, frame_count=5, fps=10)

        frames = download_and_extract_motion_frames(
            f"file://{video_path}",
            frame_start=3,
            frame_count=10,
            download_dir=tmp_path / "dl2",
        )

        assert isinstance(frames, list)
        assert len(frames) >= 1
        assert len(frames) <= 2
        assert isinstance(frames[0], np.ndarray)
        assert frames[0].dtype == np.uint8
        assert frames[0].shape[0] > 0
        assert frames[0].shape[1] > 0
        assert frames[0].shape[2] == 3
