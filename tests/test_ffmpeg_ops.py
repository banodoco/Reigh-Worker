"""Tests for source/media/video/ffmpeg_ops.py."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from source.media.video.ffmpeg_ops import (
    ensure_video_fps,
    extract_frame_range_to_video,
    create_video_from_frames_list,
    apply_saturation_to_video_ffmpeg,
)


# ---------------------------------------------------------------------------
# ensure_video_fps
# ---------------------------------------------------------------------------

class TestEnsureVideoFps:
    """Tests for ensure_video_fps."""

    def test_already_at_target_fps(self, tmp_path):
        """Video already at target FPS should return the original path."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake_video_data")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 16.0)):
            result = ensure_video_fps(video_file, target_fps=16.0)
        assert result == video_file

    def test_within_tolerance(self, tmp_path):
        """FPS within tolerance should not trigger resample."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake_video_data")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 16.3)):
            result = ensure_video_fps(video_file, target_fps=16.0, fps_tolerance=0.5)
        assert result == video_file

    def test_resample_triggered(self, tmp_path):
        """FPS outside tolerance should trigger resample via ffmpeg."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake_video_data")

        resampled_path = tmp_path / "input_resampled_16fps.mp4"

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps") as mock_info, \
             patch("source.media.video.ffmpeg_ops.subprocess.run", return_value=mock_run_result) as mock_run:
            # First call: actual fps. Second call: verification.
            mock_info.side_effect = [(100, 30.0), (100, 16.0)]

            # Create the resampled file so the exists() check passes
            resampled_path.write_bytes(b"resampled_data")

            result = ensure_video_fps(video_file, target_fps=16.0, output_dir=tmp_path)

        assert result == resampled_path
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert f"fps=16.0" in cmd[cmd.index("-vf") + 1]

    def test_source_not_found_raises(self, tmp_path):
        """Missing source video should raise OSError."""
        with pytest.raises(OSError, match="does not exist"):
            ensure_video_fps(tmp_path / "nonexistent.mp4", target_fps=16)

    def test_fps_cannot_be_determined_raises(self, tmp_path):
        """None FPS should raise ValueError."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, None)):
            with pytest.raises(ValueError, match="Could not determine"):
                ensure_video_fps(video_file, target_fps=16)

    def test_ffmpeg_timeout_raises_runtime(self, tmp_path):
        """FFmpeg timeout should raise RuntimeError."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 30.0)), \
             patch("source.media.video.ffmpeg_ops.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 600)):
            with pytest.raises(RuntimeError, match="timed out"):
                ensure_video_fps(video_file, target_fps=16)

    def test_ffmpeg_nonzero_returncode_raises(self, tmp_path):
        """FFmpeg non-zero return code should raise RuntimeError."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 30.0)), \
             patch("source.media.video.ffmpeg_ops.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="FFmpeg resample failed"):
                ensure_video_fps(video_file, target_fps=16)

    def test_none_target_fps_defaults_to_16(self, tmp_path):
        """None target_fps should default to 16."""
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(b"fake")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 16.0)):
            result = ensure_video_fps(video_file, target_fps=None)
        assert result == video_file


# ---------------------------------------------------------------------------
# extract_frame_range_to_video
# ---------------------------------------------------------------------------

class TestExtractFrameRangeToVideo:
    """Tests for extract_frame_range_to_video."""

    def test_successful_extraction_with_end_frame(self, tmp_path):
        """Extract a specific range of frames successfully."""
        src = tmp_path / "source.mp4"
        src.write_bytes(b"video_data")
        out = tmp_path / "output.mp4"

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_ffprobe", return_value=300), \
             patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps") as mock_info, \
             patch("source.media.video.ffmpeg_ops.subprocess.run", return_value=mock_run_result):
            mock_info.side_effect = [
                (300, 16.0),  # source check
                (253, 16.0),  # output verification
            ]
            # Create the output file so exists() passes
            out.write_bytes(b"output_data")

            result = extract_frame_range_to_video(src, out, 0, 252, 16)

        assert result == out

    def test_extraction_without_end_frame(self, tmp_path):
        """Extract from start_frame to end of video (end_frame=None)."""
        src = tmp_path / "source.mp4"
        src.write_bytes(b"video_data")
        out = tmp_path / "output.mp4"

        mock_run_result = MagicMock()
        mock_run_result.returncode = 0

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_ffprobe", return_value=100), \
             patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps") as mock_info, \
             patch("source.media.video.ffmpeg_ops.subprocess.run", return_value=mock_run_result) as mock_run:
            mock_info.side_effect = [
                (100, 16.0),  # source
                (87, 16.0),   # output verification
            ]
            out.write_bytes(b"output_data")

            result = extract_frame_range_to_video(src, out, 13, None, 16)

        cmd = mock_run.call_args[0][0]
        vf_arg = cmd[cmd.index("-vf") + 1]
        assert "gte(n" in vf_arg
        assert result == out

    def test_source_not_found_raises(self, tmp_path):
        """Missing source video should raise OSError."""
        with pytest.raises(OSError, match="does not exist"):
            extract_frame_range_to_video(
                tmp_path / "missing.mp4", tmp_path / "out.mp4", 0, 10, 16
            )

    def test_negative_start_frame_raises(self, tmp_path):
        """Negative start_frame should raise ValueError."""
        src = tmp_path / "source.mp4"
        src.write_bytes(b"data")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_ffprobe", return_value=100), \
             patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 16.0)):
            with pytest.raises(ValueError, match="cannot be negative"):
                extract_frame_range_to_video(src, tmp_path / "out.mp4", -1, 50, 16)

    def test_end_frame_exceeds_source_raises(self, tmp_path):
        """end_frame >= source frames should raise ValueError."""
        src = tmp_path / "source.mp4"
        src.write_bytes(b"data")

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_ffprobe", return_value=100), \
             patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps", return_value=(100, 16.0)):
            with pytest.raises(ValueError, match="end_frame 100 >= source frames 100"):
                extract_frame_range_to_video(src, tmp_path / "out.mp4", 0, 100, 16)

    def test_frame_count_mismatch_too_large_raises(self, tmp_path):
        """Large frame count mismatch should raise RuntimeError."""
        src = tmp_path / "source.mp4"
        src.write_bytes(b"data")
        out = tmp_path / "output.mp4"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("source.media.video.ffmpeg_ops.get_video_frame_count_ffprobe", return_value=100), \
             patch("source.media.video.ffmpeg_ops.get_video_frame_count_and_fps") as mock_info, \
             patch("source.media.video.ffmpeg_ops.subprocess.run", return_value=mock_result):
            # Expected 51 frames (0-50), but got 40
            mock_info.side_effect = [
                (100, 16.0),  # source
                (40, 16.0),   # output has way fewer frames
            ]
            out.write_bytes(b"output_data")

            with pytest.raises(RuntimeError, match="Frame count mismatch too large"):
                extract_frame_range_to_video(src, out, 0, 50, 16)


# ---------------------------------------------------------------------------
# create_video_from_frames_list
# ---------------------------------------------------------------------------

class TestCreateVideoFromFramesList:
    """Tests for create_video_from_frames_list."""

    def _make_frame(self, w=64, h=48):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_successful_creation(self, tmp_path):
        """Successful video creation returns output path."""
        frames = [self._make_frame() for _ in range(5)]
        output = tmp_path / "output.mp4"

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch("source.media.video.ffmpeg_ops.subprocess.Popen", return_value=mock_proc):
            # Create the file so exists() returns True
            output.write_bytes(b"video_content")
            result = create_video_from_frames_list(frames, output, 16, (64, 48))

        assert result == output
        assert mock_proc.stdin.write.call_count == 5

    def test_no_valid_frames_raises(self, tmp_path):
        """All None/invalid frames should raise ValueError."""
        frames = [None, None, "not_a_frame"]
        with pytest.raises(ValueError, match="No valid frames"):
            create_video_from_frames_list(frames, tmp_path / "out.mp4", 16, (64, 48))

    def test_skips_none_frames(self, tmp_path):
        """None frames should be skipped, valid ones written."""
        valid_frame = self._make_frame()
        frames = [None, valid_frame, None, valid_frame]
        output = tmp_path / "output.mp4"

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch("source.media.video.ffmpeg_ops.subprocess.Popen", return_value=mock_proc):
            output.write_bytes(b"video_content")
            create_video_from_frames_list(frames, output, 16, (64, 48))

        # Only 2 valid frames should be written
        assert mock_proc.stdin.write.call_count == 2

    def test_ffmpeg_not_found_raises_oserror(self, tmp_path):
        """FileNotFoundError from Popen should become OSError."""
        frames = [self._make_frame()]
        with patch("source.media.video.ffmpeg_ops.subprocess.Popen",
                   side_effect=FileNotFoundError("ffmpeg")):
            with pytest.raises(OSError, match="FFmpeg not found"):
                create_video_from_frames_list(frames, tmp_path / "out.mp4", 16, (64, 48))

    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path):
        """Non-zero return code from FFmpeg should raise RuntimeError."""
        frames = [self._make_frame()]
        output = tmp_path / "output.mp4"

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"encoding error"
        mock_proc.returncode = 1
        mock_proc.wait.return_value = None

        with patch("source.media.video.ffmpeg_ops.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg failed"):
                create_video_from_frames_list(frames, output, 16, (64, 48))

    def test_colorspace_standardization_flag(self, tmp_path):
        """standardize_colorspace=True should add colorspace filters to command."""
        frames = [self._make_frame()]
        output = tmp_path / "output.mp4"

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch("source.media.video.ffmpeg_ops.subprocess.Popen", return_value=mock_proc) as mock_popen:
            output.write_bytes(b"video_content")
            create_video_from_frames_list(
                frames, output, 16, (64, 48), standardize_colorspace=True
            )

        cmd = mock_popen.call_args[0][0]
        assert "-colorspace" in cmd
        assert "bt709" in cmd

    def test_output_suffix_forced_to_mp4(self, tmp_path):
        """Output path should be forced to .mp4 extension."""
        frames = [self._make_frame()]
        output = tmp_path / "output.avi"
        expected = tmp_path / "output.mp4"

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch("source.media.video.ffmpeg_ops.subprocess.Popen", return_value=mock_proc):
            expected.write_bytes(b"video_content")
            result = create_video_from_frames_list(frames, output, 16, (64, 48))

        assert result.suffix == ".mp4"


# ---------------------------------------------------------------------------
# apply_saturation_to_video_ffmpeg
# ---------------------------------------------------------------------------

class TestApplySaturationToVideoFfmpeg:
    """Tests for apply_saturation_to_video_ffmpeg."""

    def test_success_returns_true(self, tmp_path):
        """Successful saturation adjustment returns True."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"input_data")
        outp = tmp_path / "output.mp4"

        with patch("source.media.video.ffmpeg_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Create the output file
            outp.write_bytes(b"saturated_data")
            result = apply_saturation_to_video_ffmpeg(inp, outp, 1.5)

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert "eq=saturation=1.5" in cmd[cmd.index("-vf") + 1]

    def test_ffmpeg_failure_returns_false(self, tmp_path):
        """CalledProcessError should return False."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        outp = tmp_path / "output.mp4"

        with patch("source.media.video.ffmpeg_ops.subprocess.run",
                   side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            result = apply_saturation_to_video_ffmpeg(inp, outp, 1.5)

        assert result is False

    def test_custom_preset_in_command(self, tmp_path):
        """Custom preset should appear in the ffmpeg command."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        outp = tmp_path / "output.mp4"

        with patch("source.media.video.ffmpeg_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            outp.write_bytes(b"data")
            apply_saturation_to_video_ffmpeg(inp, outp, 1.2, preset="fast")

        cmd = mock_run.call_args[0][0]
        assert "fast" in cmd

    def test_output_missing_returns_false(self, tmp_path):
        """If output file is missing after ffmpeg, return False."""
        inp = tmp_path / "input.mp4"
        inp.write_bytes(b"data")
        outp = tmp_path / "output.mp4"

        with patch("source.media.video.ffmpeg_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Don't create output file
            result = apply_saturation_to_video_ffmpeg(inp, outp, 1.5)

        assert result is False
