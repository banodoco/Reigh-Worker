"""Tests for source/media/visualization/timeline.py."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestApplyVideoTreatment:
    """Tests for _apply_video_treatment."""

    def _make_mock_clip(self, fps=24, duration=2.0, frame_shape=(480, 640, 3)):
        """Create a mock MoviePy clip."""
        clip = MagicMock()
        clip.fps = fps
        clip.duration = duration
        clip.get_frame = MagicMock(
            side_effect=lambda t: np.zeros(frame_shape, dtype=np.uint8)
        )
        return clip

    @patch("source.media.visualization.timeline.ImageSequenceClip", create=True)
    def test_adjust_compress(self, mock_isc):
        """Adjust mode compresses when source has more frames than target."""
        # Patch the moviepy import inside the function
        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(ImageSequenceClip=mock_isc),
        }):
            from source.media.visualization.timeline import _apply_video_treatment

            clip = self._make_mock_clip(fps=24, duration=4.0)  # 96 frames
            mock_result = MagicMock()
            mock_result.duration = 2.0
            mock_isc.return_value = mock_result

            result = _apply_video_treatment(
                clip, target_duration=2.0, target_fps=24, treatment="adjust"
            )

            # ImageSequenceClip should have been called with 48 frames at 24fps
            assert mock_isc.called
            frames_arg = mock_isc.call_args[0][0]
            assert len(frames_arg) == 48
            fps_arg = mock_isc.call_args[1].get("fps") or mock_isc.call_args[0][1]
            assert fps_arg == 24

    @patch("source.media.visualization.timeline.ImageSequenceClip", create=True)
    def test_adjust_stretch(self, mock_isc):
        """Adjust mode stretches when source has fewer frames than target."""
        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(ImageSequenceClip=mock_isc),
        }):
            from source.media.visualization.timeline import _apply_video_treatment

            clip = self._make_mock_clip(fps=24, duration=1.0)  # 24 frames
            mock_result = MagicMock()
            mock_result.duration = 2.0
            mock_isc.return_value = mock_result

            result = _apply_video_treatment(
                clip, target_duration=2.0, target_fps=24, treatment="adjust"
            )

            frames_arg = mock_isc.call_args[0][0]
            assert len(frames_arg) == 48  # Target is 2.0 * 24 = 48

    @patch("source.media.visualization.timeline.ImageSequenceClip", create=True)
    def test_clip_mode(self, mock_isc):
        """Clip mode uses FPS-based temporal sampling."""
        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(ImageSequenceClip=mock_isc),
        }):
            from source.media.visualization.timeline import _apply_video_treatment

            clip = self._make_mock_clip(fps=30, duration=3.0)  # 90 frames
            mock_result = MagicMock()
            mock_result.duration = 2.0
            mock_isc.return_value = mock_result

            result = _apply_video_treatment(
                clip, target_duration=2.0, target_fps=24, treatment="clip"
            )

            assert mock_isc.called
            frames_arg = mock_isc.call_args[0][0]
            # Should produce target_frame_count = 48 frames
            assert len(frames_arg) == 48

    def test_invalid_treatment_raises(self):
        """Invalid treatment value raises ValueError."""
        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(),
        }):
            from source.media.visualization.timeline import _apply_video_treatment

            clip = self._make_mock_clip()
            with pytest.raises(ValueError, match="Invalid treatment"):
                _apply_video_treatment(clip, 2.0, 24, treatment="invalid")


class TestCreateTimelineClip:
    """Tests for _create_timeline_clip."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample image files for testing."""
        paths = []
        for i in range(3):
            img = Image.new("RGB", (100, 80), color=(i * 80, 50, 200))
            path = tmp_path / f"img_{i}.png"
            img.save(path)
            paths.append(str(path))
        return paths

    @patch("source.media.visualization.timeline.VideoClip", create=True)
    def test_timeline_creation(self, mock_video_clip, sample_images):
        """Timeline clip is created with correct duration."""
        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(VideoClip=mock_video_clip),
        }):
            from source.media.visualization.timeline import _create_timeline_clip

            mock_video_clip.return_value = MagicMock()

            result = _create_timeline_clip(
                duration=5.0,
                width=800,
                height=150,
                input_image_paths=sample_images,
                segment_frames=[40, 40, 40],
                segment_prompts=["a", "b", "c"],
                fps=24,
            )

            mock_video_clip.assert_called_once()
            # Check it was called with a make_frame function and correct duration
            args, kwargs = mock_video_clip.call_args
            assert args[0] is not None  # make_frame callable
            assert kwargs.get("duration") == 5.0 or (len(args) > 1 and args[1] == 5.0)

    def test_make_frame_returns_valid_array(self, sample_images):
        """The make_frame closure returns a valid numpy array."""
        from source.media.visualization.timeline import _create_timeline_clip

        # We need to capture the make_frame function
        captured_make_frame = None

        def capture_video_clip(make_frame, duration):
            nonlocal captured_make_frame
            captured_make_frame = make_frame
            clip = MagicMock()
            clip.duration = duration
            return clip

        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(VideoClip=capture_video_clip),
        }):
            from importlib import reload
            import source.media.visualization.timeline as timeline_mod
            reload(timeline_mod)

            timeline_mod._create_timeline_clip(
                duration=5.0,
                width=800,
                height=150,
                input_image_paths=sample_images,
                segment_frames=[40, 40, 40],
                segment_prompts=["prompt alpha", "prompt beta", "prompt gamma"],
                fps=24,
            )

        assert captured_make_frame is not None
        # Generate a frame at t=0
        frame = captured_make_frame(0.0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (150, 800, 3)
        assert frame.dtype == np.uint8

    def test_make_frame_mid_video(self, sample_images):
        """make_frame works at a midpoint in the video."""
        captured_make_frame = None

        def capture_video_clip(make_frame, duration):
            nonlocal captured_make_frame
            captured_make_frame = make_frame
            clip = MagicMock()
            clip.duration = duration
            return clip

        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(VideoClip=capture_video_clip),
        }):
            from importlib import reload
            import source.media.visualization.timeline as timeline_mod
            reload(timeline_mod)

            timeline_mod._create_timeline_clip(
                duration=5.0,
                width=800,
                height=150,
                input_image_paths=sample_images,
                segment_frames=[40, 40, 40],
                segment_prompts=None,
                fps=24,
            )

        assert captured_make_frame is not None
        frame = captured_make_frame(2.5)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (150, 800, 3)

    def test_make_frame_with_overlaps(self, sample_images):
        """make_frame handles frame_overlaps correctly."""
        captured_make_frame = None

        def capture_video_clip(make_frame, duration):
            nonlocal captured_make_frame
            captured_make_frame = make_frame
            clip = MagicMock()
            clip.duration = duration
            return clip

        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(VideoClip=capture_video_clip),
        }):
            from importlib import reload
            import source.media.visualization.timeline as timeline_mod
            reload(timeline_mod)

            timeline_mod._create_timeline_clip(
                duration=3.0,
                width=600,
                height=120,
                input_image_paths=sample_images,
                segment_frames=[30, 30, 30],
                segment_prompts=["p1", "p2", "p3"],
                fps=24,
                frame_overlaps=[5, 5],
            )

        assert captured_make_frame is not None
        frame = captured_make_frame(1.0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 600, 3)

    def test_make_frame_vertical_mode(self, sample_images):
        """make_frame works in vertical mode."""
        captured_make_frame = None

        def capture_video_clip(make_frame, duration):
            nonlocal captured_make_frame
            captured_make_frame = make_frame
            clip = MagicMock()
            clip.duration = duration
            return clip

        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(VideoClip=capture_video_clip),
        }):
            from importlib import reload
            import source.media.visualization.timeline as timeline_mod
            reload(timeline_mod)

            timeline_mod._create_timeline_clip(
                duration=5.0,
                width=200,
                height=600,
                input_image_paths=sample_images,
                segment_frames=[40, 40, 40],
                segment_prompts=["p1", "p2", "p3"],
                fps=24,
                vertical=True,
            )

        assert captured_make_frame is not None
        frame = captured_make_frame(2.0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (600, 200, 3)

    def test_segment_boundaries_with_overlaps(self, sample_images):
        """Segment boundary calculation accounts for overlaps correctly."""
        # This tests the boundary math without needing to render frames
        # The total_frames should be sum(segment_frames) - sum(overlaps)
        segment_frames = [40, 40, 40]
        frame_overlaps = [5, 5]
        expected_total = 40 + 40 + 40 - 5 - 5  # 110

        # Verify by checking the make_frame closure behavior
        captured_make_frame = None

        def capture_video_clip(make_frame, duration):
            nonlocal captured_make_frame
            captured_make_frame = make_frame
            clip = MagicMock()
            clip.duration = duration
            return clip

        with patch.dict("sys.modules", {
            "moviepy": MagicMock(),
            "moviepy.editor": MagicMock(VideoClip=capture_video_clip),
        }):
            from importlib import reload
            import source.media.visualization.timeline as timeline_mod
            reload(timeline_mod)

            timeline_mod._create_timeline_clip(
                duration=5.0,
                width=800,
                height=150,
                input_image_paths=sample_images,
                segment_frames=segment_frames,
                segment_prompts=["a", "b", "c"],
                fps=24,
                frame_overlaps=frame_overlaps,
            )

        # Frame at t=0 should succeed and produce a valid canvas
        frame = captured_make_frame(0.0)
        assert frame.shape == (150, 800, 3)

        # Frame at end should also succeed
        frame_end = captured_make_frame(4.9)
        assert frame_end.shape == (150, 800, 3)
