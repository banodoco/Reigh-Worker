"""Tests for source/media/visualization/layouts.py."""

from unittest.mock import MagicMock, patch, call
import pytest


def _make_mock_clip(w=640, h=480, duration=5.0, fps=24):
    """Create a mock video clip with the required attributes."""
    clip = MagicMock()
    clip.w = w
    clip.h = h
    clip.duration = duration
    clip.fps = fps

    def resize_side_effect(**kwargs):
        new_clip = _make_mock_clip(w=w, h=h, duration=duration, fps=fps)
        if "height" in kwargs:
            scale = kwargs["height"] / h
            new_clip.h = kwargs["height"]
            new_clip.w = int(w * scale)
        if "width" in kwargs:
            new_clip.w = kwargs["width"]
        if "height" in kwargs and "width" in kwargs:
            new_clip.w = kwargs["width"]
            new_clip.h = kwargs["height"]
        return new_clip

    clip.resize = MagicMock(side_effect=resize_side_effect)
    clip.set_position = MagicMock(return_value=clip)
    clip.set_duration = MagicMock(return_value=clip)
    return clip


@pytest.fixture
def mock_moviepy():
    """Mock moviepy.editor to avoid needing the real library."""
    mock_module = MagicMock()

    def mock_clips_array(arr):
        """Simulate clips_array by summing widths/heights."""
        result = MagicMock()
        # For a single row: width = sum of clip widths, height = max of clip heights
        if len(arr) == 1:
            row = arr[0]
            result.w = sum(c.w for c in row)
            result.h = max(c.h for c in row)
        elif len(arr) == 2:
            # For two rows: width = max row width, height = sum of row heights
            row_widths = []
            row_heights = []
            for row in arr:
                row_widths.append(sum(c.w for c in row))
                row_heights.append(max(c.h for c in row))
            result.w = max(row_widths)
            result.h = sum(row_heights)
        result.duration = 5.0
        result.resize = MagicMock(return_value=result)
        return result

    mock_module.clips_array = mock_clips_array

    def mock_composite(clips, size=None):
        result = MagicMock()
        if size:
            result.w, result.h = size
        else:
            result.w = clips[0].w
            result.h = clips[0].h
        result.duration = 5.0
        return result

    mock_module.CompositeVideoClip = mock_composite

    def mock_color_clip(size, color, duration):
        clip = MagicMock()
        clip.w, clip.h = size
        clip.duration = duration
        clip.resize = MagicMock(return_value=clip)
        return clip

    mock_module.ColorClip = mock_color_clip

    with patch.dict("sys.modules", {"moviepy": MagicMock(), "moviepy.editor": mock_module}):
        yield mock_module


@pytest.fixture
def mock_timeline():
    """Mock the _create_timeline_clip dependency."""
    with patch("source.media.visualization.layouts._create_timeline_clip") as mock:
        timeline_clip = MagicMock()
        timeline_clip.w = 800
        timeline_clip.h = 150
        timeline_clip.set_position = MagicMock(return_value=timeline_clip)
        mock.return_value = timeline_clip
        yield mock


class TestCreateSideBySideLayout:
    def test_basic_side_by_side(self, mock_moviepy, mock_timeline):
        """Side-by-side layout creates a two-column arrangement with timeline on top."""
        from source.media.visualization.layouts import _create_side_by_side_layout

        output_clip = _make_mock_clip()
        structure_clip = _make_mock_clip()
        guidance_clip = _make_mock_clip()

        result = _create_side_by_side_layout(
            output_clip=output_clip,
            structure_clip=structure_clip,
            guidance_clip=guidance_clip,
            input_image_paths=["img1.png", "img2.png"],
            segment_frames=[40, 40],
            segment_prompts=["prompt1", "prompt2"],
            fps=24,
        )

        # Timeline should have been created
        mock_timeline.assert_called_once()

        # Result should exist (CompositeVideoClip was called)
        assert result is not None

    def test_side_by_side_without_structure_overlay(self, mock_moviepy, mock_timeline):
        """No overlay text when structure_video_type is None."""
        from source.media.visualization.layouts import _create_side_by_side_layout

        result = _create_side_by_side_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=None,
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=None,
            fps=24,
            structure_video_type=None,
            structure_video_strength=None,
        )
        assert result is not None

    def test_side_by_side_passes_frame_overlaps(self, mock_moviepy, mock_timeline):
        """frame_overlaps are forwarded to timeline creation."""
        from source.media.visualization.layouts import _create_side_by_side_layout

        _create_side_by_side_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=None,
            input_image_paths=["img1.png", "img2.png"],
            segment_frames=[40, 40],
            segment_prompts=None,
            fps=24,
            frame_overlaps=[4],
        )

        # Verify frame_overlaps was passed through
        call_kwargs = mock_timeline.call_args
        assert call_kwargs.kwargs.get("frame_overlaps") == [4] or \
               (len(call_kwargs.args) > 7 and call_kwargs.args[7] == [4])


class TestCreateTripleLayout:
    def test_triple_includes_guidance(self, mock_moviepy, mock_timeline):
        """Triple layout includes guidance clip as a third column."""
        from source.media.visualization.layouts import _create_triple_layout

        output_clip = _make_mock_clip()
        structure_clip = _make_mock_clip()
        guidance_clip = _make_mock_clip()

        result = _create_triple_layout(
            output_clip=output_clip,
            structure_clip=structure_clip,
            guidance_clip=guidance_clip,
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=["prompt"],
            fps=24,
        )
        assert result is not None
        mock_timeline.assert_called_once()

    def test_triple_timeline_position_bottom(self, mock_moviepy, mock_timeline):
        """Triple layout uses timeline at the bottom."""
        from source.media.visualization.layouts import _create_multi_layout

        result = _create_multi_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=_make_mock_clip(),
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=None,
            fps=24,
            include_guidance=True,
            timeline_position="bottom",
            timeline_height=200,
            composite_extra_height=100,
        )
        assert result is not None


class TestCreateGridLayout:
    def test_grid_with_guidance(self, mock_moviepy, mock_timeline):
        """Grid layout creates a 2x2 arrangement."""
        from source.media.visualization.layouts import _create_grid_layout

        result = _create_grid_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=_make_mock_clip(),
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=None,
            fps=24,
        )
        assert result is not None
        mock_timeline.assert_called_once()

    def test_grid_without_guidance_uses_color_clip(self, mock_moviepy, mock_timeline):
        """Grid layout creates a black placeholder when guidance_clip is None."""
        from source.media.visualization.layouts import _create_grid_layout

        result = _create_grid_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=None,
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=None,
            fps=24,
        )
        assert result is not None

    def test_grid_with_structure_overlay_error_handled(self, mock_moviepy, mock_timeline):
        """Grid layout handles overlay text creation failure gracefully."""
        from source.media.visualization.layouts import _create_grid_layout

        # TextClip is not mocked, so the overlay will fail silently
        result = _create_grid_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=_make_mock_clip(),
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=None,
            fps=24,
            structure_video_type="depth",
            structure_video_strength=0.8,
        )
        assert result is not None


class TestCreateVerticalLayout:
    def test_vertical_basic(self, mock_moviepy, mock_timeline):
        """Vertical layout creates structure/output stacked with timeline on left."""
        from source.media.visualization.layouts import _create_vertical_layout

        result = _create_vertical_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=None,
            input_image_paths=["img.png"],
            segment_frames=[80],
            segment_prompts=None,
            fps=24,
        )
        assert result is not None
        mock_timeline.assert_called_once()

        # Verify vertical=True was passed to timeline
        call_kwargs = mock_timeline.call_args
        assert call_kwargs.kwargs.get("vertical") is True

    def test_vertical_passes_all_params(self, mock_moviepy, mock_timeline):
        """Vertical layout forwards all parameters to timeline."""
        from source.media.visualization.layouts import _create_vertical_layout

        _create_vertical_layout(
            output_clip=_make_mock_clip(),
            structure_clip=_make_mock_clip(),
            guidance_clip=None,
            input_image_paths=["a.png", "b.png", "c.png"],
            segment_frames=[30, 30, 30],
            segment_prompts=["p1", "p2", "p3"],
            fps=16,
            frame_overlaps=[4, 4],
        )

        call_kwargs = mock_timeline.call_args
        assert call_kwargs.kwargs["fps"] == 16
        assert call_kwargs.kwargs["input_image_paths"] == ["a.png", "b.png", "c.png"]
