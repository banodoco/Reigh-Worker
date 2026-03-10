"""Tests for source/task_handlers/travel/ffmpeg_fallback.py."""

from pathlib import Path
import shutil
from unittest import mock
from unittest.mock import MagicMock, patch, call

import pytest


class TestAttemptFfmpegCrossfadeFallback:
    """Tests for attempt_ffmpeg_crossfade_fallback."""

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    def test_less_than_two_videos_returns_false(self, mock_logger):
        """Returns False when fewer than 2 videos are provided."""
        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=["video1.mp4"],
            overlaps=[],
            output_path=Path("/output/result.mp4"),
            task_id="task1",
        )
        assert result is False

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    def test_empty_video_list_returns_false(self, mock_logger):
        """Returns False for empty video list."""
        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=[],
            overlaps=[],
            output_path=Path("/output/result.mp4"),
            task_id="task1",
        )
        assert result is False

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    @patch.dict("sys.modules", {"cv2": MagicMock()})
    def test_video_cannot_be_opened(self, mock_logger):
        """Returns False when cv2 cannot open the first video."""
        import sys
        mock_cv2 = sys.modules["cv2"]

        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=["a.mp4", "b.mp4"],
            overlaps=[5],
            output_path=Path("/output/result.mp4"),
            task_id="task1",
        )
        assert result is False


class TestAttemptFfmpegCrossfadeFallbackDirect:
    """Direct integration coverage using tiny real videos."""

    def _write_video(self, path: Path, value: int, frames: int = 6, fps: int = 8):
        import numpy as np
        cv2 = pytest.importorskip("cv2")
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (32, 24),
        )
        assert writer.isOpened()
        for _ in range(frames):
            frame = (value * np.ones((24, 32, 3), dtype=np.uint8))
            writer.write(frame)
        writer.release()
        assert path.exists()
        assert path.stat().st_size > 0

    def test_real_ffmpeg_crossfade_generates_output(self, tmp_path):
        import numpy as np
        cv2 = pytest.importorskip("cv2")
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not available on PATH")

        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        seg1 = tmp_path / "seg1.mp4"
        seg2 = tmp_path / "seg2.mp4"
        out = tmp_path / "stitched.mp4"
        self._write_video(seg1, 40, frames=8, fps=8)
        self._write_video(seg2, 200, frames=8, fps=8)

        ok = attempt_ffmpeg_crossfade_fallback(
            [str(seg1), str(seg2)],
            [2],
            out,
            "task-direct",
        )

        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 0

        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_frame, frame = cap.read()
        cap.release()

        assert frame_count > 0
        assert width == 32
        assert height == 24
        assert got_frame is True
        assert frame is not None

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    @patch.dict("sys.modules", {"cv2": MagicMock()})
    def test_invalid_fps_returns_false(self, mock_logger):
        """Returns False when FPS is zero or negative."""
        import sys
        mock_cv2 = sys.modules["cv2"]

        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0  # Invalid FPS
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FPS = 5  # cv2 constant

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=["a.mp4", "b.mp4"],
            overlaps=[5],
            output_path=Path("/output/result.mp4"),
            task_id="task1",
        )
        assert result is False
        mock_cap.release.assert_called_once()

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    @patch.dict("sys.modules", {"subprocess": MagicMock(), "cv2": MagicMock()})
    def test_successful_crossfade(self, mock_logger, tmp_path):
        """Successful ffmpeg crossfade returns True."""
        import sys
        mock_cv2 = sys.modules["cv2"]
        mock_subprocess = sys.modules["subprocess"]

        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        # Set up cv2 mocks
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 24.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 81.0,
        }.get(prop, 0.0)
        mock_cv2.VideoCapture.return_value = mock_cap

        # Set up subprocess mock
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.run.return_value = mock_result

        # Create actual output file so exists() check works
        output_path = tmp_path / "result.mp4"
        output_path.write_bytes(b"fake video content")

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=["a.mp4", "b.mp4"],
            overlaps=[5],
            output_path=output_path,
            task_id="task1",
        )
        assert result is True
        mock_subprocess.run.assert_called_once()

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    @patch.dict("sys.modules", {"subprocess": MagicMock(), "cv2": MagicMock()})
    def test_ffmpeg_returns_nonzero(self, mock_logger, tmp_path):
        """Returns False when ffmpeg returns a non-zero exit code."""
        import sys
        mock_cv2 = sys.modules["cv2"]
        mock_subprocess = sys.modules["subprocess"]

        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 24.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 81.0,
        }.get(prop, 0.0)
        mock_cv2.VideoCapture.return_value = mock_cap

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: invalid filter"
        mock_subprocess.run.return_value = mock_result

        output_path = tmp_path / "result.mp4"
        # Don't create the file -- output doesn't exist

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=["a.mp4", "b.mp4"],
            overlaps=[5],
            output_path=output_path,
            task_id="task1",
        )
        assert result is False

    @patch("source.task_handlers.travel.ffmpeg_fallback.travel_logger")
    @patch.dict("sys.modules", {"subprocess": MagicMock(), "cv2": MagicMock()})
    def test_timeout_returns_false(self, mock_logger):
        """Returns False on subprocess timeout."""
        import subprocess as real_subprocess
        import sys
        mock_cv2 = sys.modules["cv2"]
        mock_subprocess = sys.modules["subprocess"]

        from source.task_handlers.travel.ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 24.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 81.0,
        }.get(prop, 0.0)
        mock_cv2.VideoCapture.return_value = mock_cap

        # Make subprocess.run raise TimeoutExpired
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired
        mock_subprocess.SubprocessError = real_subprocess.SubprocessError
        mock_subprocess.run.side_effect = real_subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        result = attempt_ffmpeg_crossfade_fallback(
            segment_video_paths=["a.mp4", "b.mp4"],
            overlaps=[5],
            output_path=Path("/output/result.mp4"),
            task_id="task1",
        )
        assert result is False
