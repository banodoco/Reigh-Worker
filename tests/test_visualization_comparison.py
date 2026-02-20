"""Tests for source/media/visualization/comparison.py."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

from source.media.visualization.comparison import (
    create_travel_visualization,
    create_simple_comparison,
    create_opencv_side_by_side,
)


class TestCreateTravelVisualization:
    """Tests for create_travel_visualization."""

    def test_unknown_layout_raises_valueerror(self):
        """Should raise ValueError for unknown layout type."""
        mock_clip = MagicMock()
        mock_clip.duration = 5.0

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip = MagicMock(return_value=mock_clip)

        with patch.dict("sys.modules", {"moviepy": mock_moviepy, "moviepy.editor": mock_moviepy}), \
             patch("source.media.visualization.comparison._apply_video_treatment", return_value=mock_clip):
            with pytest.raises(ValueError, match="Unknown layout"):
                create_travel_visualization(
                    output_video_path="/fake/output.mp4",
                    structure_video_path="/fake/structure.mp4",
                    guidance_video_path=None,
                    input_image_paths=["/fake/img.png"],
                    segment_frames=[10],
                    layout="nonexistent_layout",
                )

    def test_default_viz_output_path(self):
        """When viz_output_path is None, should create one with _viz suffix."""
        mock_clip = MagicMock()
        mock_clip.duration = 5.0

        mock_result = MagicMock()

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip = MagicMock(return_value=mock_clip)

        with patch.dict("sys.modules", {"moviepy": mock_moviepy, "moviepy.editor": mock_moviepy}), \
             patch("source.media.visualization.comparison._apply_video_treatment", return_value=mock_clip), \
             patch("source.media.visualization.comparison._create_side_by_side_layout", return_value=mock_result):
            result = create_travel_visualization(
                output_video_path="/fake/dir/output.mp4",
                structure_video_path="/fake/structure.mp4",
                guidance_video_path=None,
                input_image_paths=["/fake/img.png"],
                segment_frames=[10],
                viz_output_path=None,
                layout="side_by_side",
            )
        # Should have written to path with _viz suffix
        mock_result.write_videofile.assert_called_once()
        written_path = mock_result.write_videofile.call_args[0][0]
        assert "_viz" in written_path

    def test_moviepy_import_error(self):
        """Should raise ImportError when moviepy is not available."""
        # Temporarily make moviepy unavailable
        import sys
        original = sys.modules.get("moviepy.editor")
        sys.modules["moviepy.editor"] = None
        try:
            with pytest.raises((ImportError, TypeError)):
                create_travel_visualization(
                    output_video_path="/fake/output.mp4",
                    structure_video_path="/fake/structure.mp4",
                    guidance_video_path=None,
                    input_image_paths=[],
                    segment_frames=[],
                )
        finally:
            if original is not None:
                sys.modules["moviepy.editor"] = original
            else:
                sys.modules.pop("moviepy.editor", None)


class TestCreateSimpleComparison:
    """Tests for create_simple_comparison."""

    def test_horizontal_orientation(self):
        """Horizontal layout should call clips_array with [[clip1, clip2]]."""
        mock_clip1 = MagicMock()
        mock_clip1.duration = 3.0
        mock_clip2 = MagicMock()
        mock_clip2.duration = 3.0
        mock_final = MagicMock()

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip = MagicMock(side_effect=[mock_clip1, mock_clip2])
        mock_moviepy.clips_array = MagicMock(return_value=mock_final)

        with patch.dict("sys.modules", {"moviepy": mock_moviepy, "moviepy.editor": mock_moviepy}):
            result = create_simple_comparison(
                "/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4",
                orientation="horizontal"
            )
        mock_moviepy.clips_array.assert_called_once()
        # First arg should be [[clip1, clip2]]
        array_arg = mock_moviepy.clips_array.call_args[0][0]
        assert len(array_arg) == 1  # one row
        assert len(array_arg[0]) == 2  # two clips

    def test_vertical_orientation(self):
        """Vertical layout should call clips_array with [[clip1], [clip2]]."""
        mock_clip1 = MagicMock()
        mock_clip1.duration = 3.0
        mock_clip2 = MagicMock()
        mock_clip2.duration = 3.0
        mock_final = MagicMock()

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip = MagicMock(side_effect=[mock_clip1, mock_clip2])
        mock_moviepy.clips_array = MagicMock(return_value=mock_final)

        with patch.dict("sys.modules", {"moviepy": mock_moviepy, "moviepy.editor": mock_moviepy}):
            result = create_simple_comparison(
                "/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4",
                orientation="vertical"
            )
        mock_moviepy.clips_array.assert_called_once()
        array_arg = mock_moviepy.clips_array.call_args[0][0]
        assert len(array_arg) == 2  # two rows
        assert len(array_arg[0]) == 1  # one clip per row

    def test_returns_output_path(self):
        """Should return the output path."""
        mock_clip = MagicMock()
        mock_clip.duration = 2.0
        mock_final = MagicMock()

        mock_moviepy = MagicMock()
        mock_moviepy.VideoFileClip = MagicMock(return_value=mock_clip)
        mock_moviepy.clips_array = MagicMock(return_value=mock_final)

        with patch.dict("sys.modules", {"moviepy": mock_moviepy, "moviepy.editor": mock_moviepy}):
            result = create_simple_comparison(
                "/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4"
            )
        assert result == "/fake/out.mp4"


class TestCreateOpencvSideBySide:
    """Tests for create_opencv_side_by_side."""

    def _make_mock_cap(self, width, height, fps, frames):
        """Create a mock VideoCapture that yields the given frames."""
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {
            3: float(width),    # CAP_PROP_FRAME_WIDTH
            4: float(height),   # CAP_PROP_FRAME_HEIGHT
            5: float(fps),      # CAP_PROP_FPS
        }.get(prop, 0.0)

        read_returns = [(True, f) for f in frames] + [(False, None)]
        cap.read.side_effect = read_returns
        return cap

    def test_basic_side_by_side(self):
        """Should combine frames and write output."""
        frame1 = np.zeros((100, 200, 3), dtype=np.uint8)
        frame2 = np.ones((100, 200, 3), dtype=np.uint8) * 255

        cap1 = self._make_mock_cap(200, 100, 30, [frame1, frame1])
        cap2 = self._make_mock_cap(200, 100, 30, [frame2, frame2])

        mock_writer = MagicMock()

        with patch("source.media.visualization.comparison.cv2.VideoCapture", side_effect=[cap1, cap2]), \
             patch("source.media.visualization.comparison.cv2.VideoWriter", return_value=mock_writer), \
             patch("source.media.visualization.comparison.cv2.VideoWriter_fourcc", return_value=0), \
             patch("source.media.visualization.comparison.cv2.resize", side_effect=lambda f, sz: f):
            result = create_opencv_side_by_side("/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4")
        assert result == "/fake/out.mp4"
        assert mock_writer.write.call_count == 2
        mock_writer.release.assert_called_once()

    def test_custom_fps(self):
        """Should use provided FPS instead of video FPS."""
        cap1 = self._make_mock_cap(100, 100, 30, [])
        cap2 = self._make_mock_cap(100, 100, 30, [])

        mock_writer = MagicMock()

        with patch("source.media.visualization.comparison.cv2.VideoCapture", side_effect=[cap1, cap2]), \
             patch("source.media.visualization.comparison.cv2.VideoWriter", return_value=mock_writer) as mock_vw_cls, \
             patch("source.media.visualization.comparison.cv2.VideoWriter_fourcc", return_value=0):
            create_opencv_side_by_side("/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4", fps=60)
        # Check that VideoWriter was called with fps=60
        vw_call_args = mock_vw_cls.call_args
        assert vw_call_args[0][2] == 60  # third positional arg is fps

    def test_stops_at_shorter_video(self):
        """Should stop when either video runs out of frames."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        # Video 1: 3 frames, Video 2: 1 frame
        cap1 = self._make_mock_cap(50, 50, 30, [frame, frame, frame])
        cap2 = self._make_mock_cap(50, 50, 30, [frame])

        mock_writer = MagicMock()

        with patch("source.media.visualization.comparison.cv2.VideoCapture", side_effect=[cap1, cap2]), \
             patch("source.media.visualization.comparison.cv2.VideoWriter", return_value=mock_writer), \
             patch("source.media.visualization.comparison.cv2.VideoWriter_fourcc", return_value=0), \
             patch("source.media.visualization.comparison.cv2.resize", side_effect=lambda f, sz: f):
            create_opencv_side_by_side("/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4")
        # Should only write 1 frame (limited by shorter video)
        assert mock_writer.write.call_count == 1

    def test_default_fps_from_video1(self):
        """When fps=None, should use fps from first video."""
        cap1 = self._make_mock_cap(100, 100, 24, [])
        cap2 = self._make_mock_cap(100, 100, 30, [])

        mock_writer = MagicMock()

        with patch("source.media.visualization.comparison.cv2.VideoCapture", side_effect=[cap1, cap2]), \
             patch("source.media.visualization.comparison.cv2.VideoWriter", return_value=mock_writer) as mock_vw_cls, \
             patch("source.media.visualization.comparison.cv2.VideoWriter_fourcc", return_value=0):
            create_opencv_side_by_side("/fake/v1.mp4", "/fake/v2.mp4", "/fake/out.mp4")
        vw_call_args = mock_vw_cls.call_args
        assert vw_call_args[0][2] == 24  # fps from video 1


class TestCreateOpencvSideBySideDirect:
    """Direct OpenCV behavior test with real temporary files."""

    def _write_video(self, path, value, frame_count=4, fps=8):
        cv2 = pytest.importorskip("cv2")
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (32, 24),
        )
        assert writer.isOpened()
        for _ in range(frame_count):
            frame = np.full((24, 32, 3), value, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        assert path.exists()
        assert path.stat().st_size > 0

    def test_real_videos_produce_side_by_side_output(self, tmp_path):
        cv2 = pytest.importorskip("cv2")

        video1 = tmp_path / "v1.mp4"
        video2 = tmp_path / "v2.mp4"
        out = tmp_path / "out.mp4"
        self._write_video(video1, 10, frame_count=4, fps=8)
        self._write_video(video2, 220, frame_count=4, fps=8)

        result = create_opencv_side_by_side(str(video1), str(video2), str(out), fps=8)
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        cap.release()

        assert frame_count == 4
        assert width == 64
        assert height == 24
        assert ok is True
        assert frame is not None
        assert frame.shape == (24, 64, 3)
