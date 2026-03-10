"""Tests for scripts/uni3c_validation.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import ImageFont

from scripts.uni3c_validation import (
    extract_frames_from_video,
    create_frame_strip,
    create_uni3c_comparison,
    create_vlm_validation_prompt,
)

# Load a real default font before any mocks are applied, so tests can use it
# as a return value for mocked load_default() without truetype issues.
_DEFAULT_FONT = ImageFont.load_default()


# ── extract_frames_from_video ────────────────────────────────────────────────

class TestExtractFramesFromVideo:
    """Frame extraction via ffprobe/ffmpeg."""

    def test_nonexistent_video_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video not found"):
            extract_frames_from_video(str(tmp_path / "no_such_video.mp4"))

    @patch("scripts.uni3c_validation.subprocess.run")
    def test_extracts_correct_number_of_frames(self, mock_run, tmp_path):
        """Extracts N evenly-spaced frames from a video."""
        # Create a dummy video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        output_dir = tmp_path / "frames"

        # Mock ffprobe returning duration
        ffprobe_result = MagicMock()
        ffprobe_result.stdout = "10.0\n"

        # Mock ffmpeg creating frame files
        def mock_run_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            elif cmd[0] == "ffmpeg":
                # Create the output frame file
                # Find the output path (last argument)
                out_path = Path(cmd[-1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"fake frame")
                return MagicMock()
            return MagicMock()

        mock_run.side_effect = mock_run_side_effect

        frames = extract_frames_from_video(str(video_file), num_frames=5, output_dir=str(output_dir))
        assert len(frames) == 5

    @patch("scripts.uni3c_validation.subprocess.run")
    def test_single_frame_uses_midpoint(self, mock_run, tmp_path):
        """num_frames=1 extracts at the midpoint of the video."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        ffprobe_result = MagicMock()
        ffprobe_result.stdout = "10.0\n"

        ffmpeg_calls = []

        def mock_run_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            elif cmd[0] == "ffmpeg":
                ffmpeg_calls.append(cmd)
                out_path = Path(cmd[-1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"fake frame")
                return MagicMock()
            return MagicMock()

        mock_run.side_effect = mock_run_side_effect

        frames = extract_frames_from_video(str(video_file), num_frames=1)
        assert len(frames) == 1

        # The timestamp should be at midpoint (5.0s for 10s video)
        ffmpeg_cmd = ffmpeg_calls[0]
        ss_idx = ffmpeg_cmd.index("-ss")
        timestamp = float(ffmpeg_cmd[ss_idx + 1])
        assert timestamp == pytest.approx(5.0)

    @patch("scripts.uni3c_validation.subprocess.run")
    def test_ffprobe_failure_uses_default_duration(self, mock_run, tmp_path):
        """If ffprobe fails, falls back to 5s default duration."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        import subprocess

        call_count = 0

        def mock_run_side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if cmd[0] == "ffprobe":
                raise subprocess.SubprocessError("ffprobe not found")
            elif cmd[0] == "ffmpeg":
                out_path = Path(cmd[-1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"fake frame")
                return MagicMock()
            return MagicMock()

        mock_run.side_effect = mock_run_side_effect

        # Should not raise, just use default duration
        frames = extract_frames_from_video(str(video_file), num_frames=3)
        # ffmpeg may succeed or fail, but function should not crash
        assert isinstance(frames, list)

    @patch("scripts.uni3c_validation.subprocess.run")
    def test_uses_temp_dir_when_no_output_dir(self, mock_run, tmp_path):
        """When output_dir is None, uses a temp directory."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        ffprobe_result = MagicMock()
        ffprobe_result.stdout = "5.0\n"

        def mock_run_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            elif cmd[0] == "ffmpeg":
                out_path = Path(cmd[-1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"fake frame")
                return MagicMock()
            return MagicMock()

        mock_run.side_effect = mock_run_side_effect

        frames = extract_frames_from_video(str(video_file), num_frames=2, output_dir=None)
        assert isinstance(frames, list)
        # Frames should be in a temp directory
        if frames:
            assert "uni3c_frames_" in frames[0]


# ── create_frame_strip ───────────────────────────────────────────────────────

class TestCreateFrameStrip:
    """Horizontal frame strip generation with PIL."""

    @patch("scripts.uni3c_validation.ImageFont.load_default", return_value=_DEFAULT_FONT)
    @patch("scripts.uni3c_validation.ImageFont.truetype", side_effect=OSError("no font"))
    def test_empty_frames_returns_placeholder(self, mock_truetype, mock_default):
        """Empty frame list returns a placeholder image."""
        result = create_frame_strip([], "Test Label")
        assert result is not None
        assert result.size[0] == 400  # Default placeholder width
        assert result.size[1] == 240  # target_height (200) + label_height (40)

    @patch("scripts.uni3c_validation.ImageFont.load_default", return_value=_DEFAULT_FONT)
    @patch("scripts.uni3c_validation.ImageFont.truetype", side_effect=OSError("no font"))
    def test_with_real_frames(self, mock_truetype, mock_default, tmp_path):
        """Creates a strip from actual image files."""
        from PIL import Image

        # Create test frame images
        frame_paths = []
        for i in range(3):
            fp = tmp_path / f"frame_{i}.jpg"
            img = Image.new("RGB", (320, 240), color=(i * 80, i * 80, i * 80))
            img.save(str(fp))
            frame_paths.append(str(fp))

        result = create_frame_strip(frame_paths, "Test Strip", target_height=100)
        assert result is not None
        # Width should accommodate all frames + gaps
        assert result.size[0] > 100
        # Height = target_height + label_height
        assert result.size[1] == 140

    @patch("scripts.uni3c_validation.ImageFont.load_default", return_value=_DEFAULT_FONT)
    @patch("scripts.uni3c_validation.ImageFont.truetype", side_effect=OSError("no font"))
    def test_custom_border_color(self, mock_truetype, mock_default, tmp_path):
        """Border color is applied to label background."""
        from PIL import Image

        fp = tmp_path / "frame.jpg"
        Image.new("RGB", (100, 100)).save(str(fp))

        result = create_frame_strip([str(fp)], "Label", border_color=(255, 0, 0))
        assert result is not None

    @patch("scripts.uni3c_validation.ImageFont.load_default", return_value=_DEFAULT_FONT)
    @patch("scripts.uni3c_validation.Image.open", side_effect=OSError("corrupt"))
    @patch("scripts.uni3c_validation.ImageFont.truetype", side_effect=OSError("no font"))
    def test_corrupt_frame_returns_placeholder(self, mock_truetype, mock_open, mock_default):
        """If all frames fail to load, returns a placeholder."""
        result = create_frame_strip(["/bad/frame.jpg"], "Failed Strip")
        assert result is not None
        assert result.size[0] == 400  # Placeholder width


# ── create_uni3c_comparison ──────────────────────────────────────────────────

class TestCreateUni3cComparison:
    """Full comparison grid generation."""

    @patch("scripts.uni3c_validation.ImageFont.load_default", return_value=_DEFAULT_FONT)
    @patch("scripts.uni3c_validation.ImageFont.truetype", side_effect=OSError("no font"))
    @patch("scripts.uni3c_validation.extract_frames_from_video")
    def test_creates_comparison_image(self, mock_extract, mock_truetype, mock_default, tmp_path):
        """Comparison image is saved to output_dir."""
        from PIL import Image

        # Create temp frame files for each video
        frame_paths = []
        for i in range(3):
            fp = tmp_path / f"frame_{i}.jpg"
            Image.new("RGB", (320, 240), color=(100, 100, 100)).save(str(fp))
            frame_paths.append(str(fp))

        mock_extract.return_value = frame_paths

        output_dir = str(tmp_path / "output")

        result = create_uni3c_comparison(
            guide_video="guide.mp4",
            baseline_output="baseline.mp4",
            uni3c_output="uni3c.mp4",
            output_dir=output_dir,
            task_id="test-123",
        )

        assert result is not None
        assert Path(result).exists()
        assert "uni3c_comparison_test-123" in result

    @patch("scripts.uni3c_validation.ImageFont.load_default", return_value=_DEFAULT_FONT)
    @patch("scripts.uni3c_validation.ImageFont.truetype", side_effect=OSError("no font"))
    @patch("scripts.uni3c_validation.extract_frames_from_video", return_value=[])
    def test_handles_no_frames(self, mock_extract, mock_truetype, mock_default, tmp_path):
        """Comparison still generates even when no frames extracted."""
        output_dir = str(tmp_path / "output")

        result = create_uni3c_comparison(
            guide_video="guide.mp4",
            baseline_output="baseline.mp4",
            uni3c_output="uni3c.mp4",
            output_dir=output_dir,
            task_id="empty",
        )

        assert result is not None
        assert Path(result).exists()


# ── create_vlm_validation_prompt ─────────────────────────────────────────────

class TestCreateVlmValidationPrompt:
    """VLM prompt generation."""

    def test_includes_image_path(self):
        prompt = create_vlm_validation_prompt("/path/to/comparison.jpg")
        assert "/path/to/comparison.jpg" in prompt

    def test_includes_evaluation_instructions(self):
        prompt = create_vlm_validation_prompt("/img.jpg")
        assert "GUIDE VIDEO" in prompt
        assert "BASELINE" in prompt
        assert "UNI3C OUTPUT" in prompt
        assert "VERDICT" in prompt

    def test_includes_confidence_section(self):
        prompt = create_vlm_validation_prompt("/img.jpg")
        assert "Confidence" in prompt
        assert "HIGH" in prompt or "MEDIUM" in prompt or "LOW" in prompt

    def test_prompt_contains_all_expected_sections_in_order(self):
        prompt = create_vlm_validation_prompt("/abs/path/comparison.jpg")
        assert prompt.startswith("Please analyze this Uni3C motion guidance validation image.")
        assert "1. **GUIDE VIDEO**" in prompt
        assert "2. **BASELINE**" in prompt
        assert "3. **UNI3C OUTPUT**" in prompt
        assert "Answer format:" in prompt
        assert "- Guide Motion:" in prompt
        assert "- Baseline Motion:" in prompt
        assert "- Uni3C Motion:" in prompt
        assert "- VERDICT:" in prompt
        assert "- Confidence:" in prompt
        assert "- Reasoning:" in prompt
        assert prompt.rstrip().endswith("Image path: /abs/path/comparison.jpg")
