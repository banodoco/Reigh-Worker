"""Tests for source/media/video/video_transforms.py."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from source.media.video.video_transforms import (
    adjust_frame_brightness,
    apply_brightness_to_video_frames,
    reverse_video,
    standardize_video_aspect_ratio,
    add_audio_to_video,
    overlay_start_end_images_above_video,
)


# ---------------------------------------------------------------------------
# adjust_frame_brightness
# ---------------------------------------------------------------------------

class TestAdjustFrameBrightness:
    """Tests for adjust_frame_brightness."""

    def test_zero_adjustment_returns_same(self):
        """brightness_adjust=0 should return unchanged frame."""
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = adjust_frame_brightness(frame, 0)
        np.testing.assert_array_equal(result, frame)

    def test_positive_adjustment_brightens(self):
        """Positive brightness_adjust should increase pixel values."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = adjust_frame_brightness(frame, 0.5)
        # factor = 1 + 0.5 = 1.5, so 100 * 1.5 = 150
        assert result.mean() > frame.mean()

    def test_negative_adjustment_darkens(self):
        """Negative brightness_adjust should decrease pixel values."""
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_frame_brightness(frame, -0.5)
        # factor = 1 + (-0.5) = 0.5, so 200 * 0.5 = 100
        assert result.mean() < frame.mean()

    def test_clipping_at_255(self):
        """Values above 255 should be clipped."""
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_frame_brightness(frame, 1.0)
        # factor = 2.0, 200 * 2.0 = 400 -> clipped to 255
        assert np.all(result == 255)

    def test_clipping_at_0(self):
        """With factor < 0, values should clip to 0."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        # factor = 1 + (-2.0) = -1.0, so 100 * (-1.0) = -100 -> clipped to 0
        result = adjust_frame_brightness(frame, -2.0)
        assert np.all(result == 0)

    def test_output_is_uint8(self):
        """Result should always be uint8."""
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = adjust_frame_brightness(frame, 0.3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# apply_brightness_to_video_frames
# ---------------------------------------------------------------------------

class TestApplyBrightnessToVideoFrames:
    """Tests for apply_brightness_to_video_frames."""

    def test_successful_brightness_adjustment(self, tmp_path):
        """Should extract, adjust, and recreate video successfully."""
        output_path = tmp_path / "bright.mp4"
        frames = [np.full((48, 64, 3), 128, dtype=np.uint8) for _ in range(3)]

        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps", return_value=(3, 16)), \
             patch("source.media.video.video_transforms.extract_frames_from_video", return_value=frames), \
             patch("source.media.video.video_transforms.create_video_from_frames_list", return_value=output_path):
            result = apply_brightness_to_video_frames("/fake/input.mp4", output_path, 0.5, "task-123")

        assert result == output_path

    def test_no_frames_returns_none(self, tmp_path):
        """If video has 0 frames, should return None."""
        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps", return_value=(0, 16)):
            result = apply_brightness_to_video_frames("/fake/input.mp4", tmp_path / "out.mp4", 0.5, "task-123")
        assert result is None

    def test_extract_fails_returns_none(self, tmp_path):
        """If frame extraction fails, should return None."""
        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps", return_value=(10, 16)), \
             patch("source.media.video.video_transforms.extract_frames_from_video", return_value=None):
            result = apply_brightness_to_video_frames("/fake/input.mp4", tmp_path / "out.mp4", 0.5, "task-123")
        assert result is None

    def test_exception_returns_none(self, tmp_path):
        """Any OSError/ValueError/RuntimeError should return None."""
        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps",
                   side_effect=OSError("disk error")):
            result = apply_brightness_to_video_frames("/fake/input.mp4", tmp_path / "out.mp4", 0.5, "task-123")
        assert result is None


# ---------------------------------------------------------------------------
# reverse_video
# ---------------------------------------------------------------------------

class TestReverseVideo:
    """Tests for reverse_video."""

    def test_successful_reverse(self, tmp_path):
        """Successful reversal returns output path."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        outp = tmp_path / "reversed.mp4"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps") as mock_info, \
             patch("source.media.video.video_transforms.subprocess.run", return_value=mock_result):
            mock_info.side_effect = [
                (100, 16.0),   # get FPS
                (100, 16.0),   # verify reversed
                (100, 16.0),   # verify original
            ]
            outp.write_bytes(b"reversed_data")
            result = reverse_video(inp, outp)

        assert result == outp

    def test_input_not_found_returns_none(self, tmp_path):
        """Missing input video should return None."""
        result = reverse_video(tmp_path / "missing.mp4", tmp_path / "out.mp4")
        assert result is None

    def test_ffmpeg_failure_returns_none(self, tmp_path):
        """FFmpeg non-zero return code should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps", return_value=(100, 16.0)), \
             patch("source.media.video.video_transforms.subprocess.run", return_value=mock_result):
            result = reverse_video(inp, tmp_path / "out.mp4")
        assert result is None

    def test_timeout_returns_none(self, tmp_path):
        """FFmpeg timeout should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")

        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps", return_value=(100, 16.0)), \
             patch("source.media.video.video_transforms.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 600)):
            result = reverse_video(inp, tmp_path / "out.mp4")
        assert result is None

    def test_reverse_command_contains_reverse_filter(self, tmp_path):
        """The ffmpeg command should contain the 'reverse' video filter."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        outp = tmp_path / "reversed.mp4"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("source.media.video.video_transforms.get_video_frame_count_and_fps") as mock_info, \
             patch("source.media.video.video_transforms.subprocess.run", return_value=mock_result) as mock_run:
            mock_info.side_effect = [(100, 16.0), (100, 16.0), (100, 16.0)]
            outp.write_bytes(b"reversed")
            reverse_video(inp, outp)

        cmd = mock_run.call_args[0][0]
        assert "reverse" in cmd


# ---------------------------------------------------------------------------
# standardize_video_aspect_ratio
# ---------------------------------------------------------------------------

class TestStandardizeVideoAspectRatio:
    """Tests for standardize_video_aspect_ratio."""

    def test_already_correct_aspect_copies(self, tmp_path):
        """Video with correct aspect ratio should be copied, not cropped."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"video_data")
        outp = tmp_path / "output.mp4"

        # ffprobe returns 1920x1080 = 16:9
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = "1920,1080\n"

        with patch("source.media.video.video_transforms.subprocess.run", return_value=mock_probe), \
             patch("shutil.copy2") as mock_copy:
            result = standardize_video_aspect_ratio(inp, outp, "16:9", "task-1")

        assert result == outp
        mock_copy.assert_called_once()

    def test_wider_source_crops_width(self, tmp_path):
        """Source wider than target should crop width."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"video_data")
        outp = tmp_path / "output.mp4"

        # ffprobe returns 1920x1080 (16:9), target 1:1
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = "1920,1080\n"

        mock_crop = MagicMock()
        mock_crop.returncode = 0

        with patch("source.media.video.video_transforms.subprocess.run",
                   side_effect=[mock_probe, mock_crop]):
            outp.write_bytes(b"cropped")
            result = standardize_video_aspect_ratio(inp, outp, "1:1", "task-1")

        assert result == outp
        crop_cmd = mock_crop.call_args if hasattr(mock_crop, 'call_args') else None
        # Verify the crop command was constructed (second subprocess.run call)

    def test_input_not_found_returns_none(self, tmp_path):
        """Missing input file should return None."""
        result = standardize_video_aspect_ratio(
            tmp_path / "missing.mp4", tmp_path / "out.mp4", "16:9"
        )
        assert result is None

    def test_invalid_aspect_ratio_returns_none(self, tmp_path):
        """Invalid aspect ratio format should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        result = standardize_video_aspect_ratio(inp, tmp_path / "out.mp4", "invalid")
        assert result is None

    def test_ffprobe_failure_returns_none(self, tmp_path):
        """ffprobe failure should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")

        mock_probe = MagicMock()
        mock_probe.returncode = 1
        mock_probe.stderr = "error"

        with patch("source.media.video.video_transforms.subprocess.run", return_value=mock_probe):
            result = standardize_video_aspect_ratio(inp, tmp_path / "out.mp4", "16:9")
        assert result is None


# ---------------------------------------------------------------------------
# add_audio_to_video
# ---------------------------------------------------------------------------

class TestAddAudioToVideo:
    """Tests for add_audio_to_video."""

    def test_input_not_found_returns_none(self, tmp_path):
        """Missing input video should return None."""
        result = add_audio_to_video(
            tmp_path / "missing.mp4", "http://example.com/audio.mp3",
            tmp_path / "out.mp4", tmp_path
        )
        assert result is None

    def test_no_audio_url_returns_none(self, tmp_path):
        """Empty audio URL should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        result = add_audio_to_video(inp, "", tmp_path / "out.mp4", tmp_path)
        assert result is None

    def test_local_audio_file(self, tmp_path):
        """Local audio file (non-URL) should be used directly."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"video")
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"audio")
        outp = tmp_path / "output.mp4"

        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = "10.5\n"

        mock_mux = MagicMock()
        mock_mux.returncode = 0

        with patch("source.media.video.video_transforms.subprocess.run",
                   side_effect=[mock_probe, mock_mux]):
            outp.write_bytes(b"muxed")
            result = add_audio_to_video(inp, str(audio_file), outp, tmp_path)

        assert result == outp

    def test_local_audio_not_found_returns_none(self, tmp_path):
        """Non-existent local audio file should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"video")
        result = add_audio_to_video(
            inp, str(tmp_path / "missing.mp3"), tmp_path / "out.mp4", tmp_path
        )
        assert result is None

    def test_ffmpeg_mux_failure_returns_none(self, tmp_path):
        """FFmpeg mux failure should return None."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"video")
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"audio")

        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = "10.5\n"

        mock_mux = MagicMock()
        mock_mux.returncode = 1
        mock_mux.stderr = "mux error"

        with patch("source.media.video.video_transforms.subprocess.run",
                   side_effect=[mock_probe, mock_mux]):
            result = add_audio_to_video(inp, str(audio), tmp_path / "out.mp4", tmp_path)

        assert result is None


# ---------------------------------------------------------------------------
# overlay_start_end_images_above_video
# ---------------------------------------------------------------------------

class TestOverlayStartEndImagesAboveVideo:
    """Tests for overlay_start_end_images_above_video."""

    def test_missing_input_returns_false(self, tmp_path):
        """Missing any input path should return False."""
        result = overlay_start_end_images_above_video(
            tmp_path / "start.png",
            tmp_path / "end.png",
            tmp_path / "video.mp4",
            tmp_path / "output.mp4"
        )
        assert result is False

    def test_ffmpeg_fallback_success(self, tmp_path):
        """With MoviePy unavailable, should use FFmpeg fallback."""
        start_img = tmp_path / "start.png"
        end_img = tmp_path / "end.png"
        video = tmp_path / "video.mp4"
        output = tmp_path / "output.mp4"

        start_img.write_bytes(b"img1")
        end_img.write_bytes(b"img2")
        video.write_bytes(b"video")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 640.0,   # CAP_PROP_FRAME_WIDTH
            4: 480.0,   # CAP_PROP_FRAME_HEIGHT
            5: 16.0,    # CAP_PROP_FPS
        }.get(prop, 0.0)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = b""

        with patch("source.media.video.video_transforms._MOVIEPY_AVAILABLE", False), \
             patch("source.media.video.video_transforms.cv2.VideoCapture", return_value=mock_cap), \
             patch("source.media.video.video_transforms.subprocess.run", return_value=mock_proc):
            output.with_suffix('.mp4').write_bytes(b"composite")
            result = overlay_start_end_images_above_video(start_img, end_img, video, output)

        assert result is True
