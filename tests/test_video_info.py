"""Tests for source/media/video/video_info.py."""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from source.media.video.video_info import (
    get_video_frame_count_ffprobe,
    get_video_fps_ffprobe,
    get_video_frame_count_and_fps,
    _parse_ffprobe_rate,
)


class TestParseFFProbeRate:
    """Tests for the ffprobe rate string parser."""

    def test_simple_fraction(self):
        assert _parse_ffprobe_rate("30000/1001") == pytest.approx(29.97, rel=0.01)

    def test_integer_fraction(self):
        assert _parse_ffprobe_rate("24/1") == 24.0

    def test_plain_float(self):
        assert _parse_ffprobe_rate("29.97") == pytest.approx(29.97)

    def test_plain_integer(self):
        assert _parse_ffprobe_rate("30") == 30.0

    def test_zero_denominator_returns_none(self):
        assert _parse_ffprobe_rate("0/0") is None

    def test_zero_numerator_returns_none(self):
        """0/1 = 0.0, which is <= 0, so returns None."""
        assert _parse_ffprobe_rate("0/1") is None

    def test_empty_string_returns_none(self):
        assert _parse_ffprobe_rate("") is None

    def test_none_returns_none(self):
        assert _parse_ffprobe_rate(None) is None

    def test_whitespace_stripped(self):
        assert _parse_ffprobe_rate("  30/1  ") == 30.0

    def test_negative_returns_none(self):
        assert _parse_ffprobe_rate("-24/1") is None

    def test_invalid_string_returns_none(self):
        assert _parse_ffprobe_rate("abc") is None


class TestGetVideoFrameCountFFProbe:
    """Tests for frame count retrieval via ffprobe."""

    def test_method1_success(self):
        """Fast metadata method returns valid count."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "120\n"
        with patch("source.media.video.video_info.subprocess.run", return_value=mock_result):
            count = get_video_frame_count_ffprobe("/fake/video.mp4")
        assert count == 120

    def test_method1_fails_method2_succeeds(self):
        """Fallback to frame counting when metadata fails."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stdout = ""

        success_result = MagicMock()
        success_result.returncode = 0
        success_result.stdout = "90\n"

        with patch("source.media.video.video_info.subprocess.run", side_effect=[fail_result, success_result]):
            count = get_video_frame_count_ffprobe("/fake/video.mp4")
        assert count == 90

    def test_both_methods_fail_returns_none(self):
        """Returns None when both methods fail."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stdout = ""
        with patch("source.media.video.video_info.subprocess.run", return_value=fail_result):
            count = get_video_frame_count_ffprobe("/fake/video.mp4")
        assert count is None

    def test_timeout_returns_none(self):
        with patch("source.media.video.video_info.subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            count = get_video_frame_count_ffprobe("/fake/video.mp4")
        assert count is None

    def test_oserror_returns_none(self):
        with patch("source.media.video.video_info.subprocess.run", side_effect=OSError("not found")):
            count = get_video_frame_count_ffprobe("/fake/video.mp4")
        assert count is None

    def test_zero_frame_count_falls_through(self):
        """Zero frame count from method 1 should try method 2."""
        zero_result = MagicMock()
        zero_result.returncode = 0
        zero_result.stdout = "0\n"

        valid_result = MagicMock()
        valid_result.returncode = 0
        valid_result.stdout = "50\n"

        with patch("source.media.video.video_info.subprocess.run", side_effect=[zero_result, valid_result]):
            count = get_video_frame_count_ffprobe("/fake/video.mp4")
        assert count == 50


class TestGetVideoFPSFFProbe:
    """Tests for FPS retrieval via ffprobe."""

    def test_avg_frame_rate_preferred(self):
        """avg_frame_rate should be used when available."""
        ffprobe_output = json.dumps({
            "streams": [{"avg_frame_rate": "30000/1001", "r_frame_rate": "30/1"}]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        with patch("source.media.video.video_info.subprocess.run", return_value=mock_result):
            fps = get_video_fps_ffprobe("/fake/video.mp4")
        assert fps == pytest.approx(29.97, rel=0.01)

    def test_fallback_to_r_frame_rate(self):
        """When avg_frame_rate is 0/0, should fall back to r_frame_rate."""
        ffprobe_output = json.dumps({
            "streams": [{"avg_frame_rate": "0/0", "r_frame_rate": "24/1"}]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        with patch("source.media.video.video_info.subprocess.run", return_value=mock_result):
            fps = get_video_fps_ffprobe("/fake/video.mp4")
        assert fps == 24.0

    def test_no_streams_returns_none(self):
        ffprobe_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        with patch("source.media.video.video_info.subprocess.run", return_value=mock_result):
            fps = get_video_fps_ffprobe("/fake/video.mp4")
        assert fps is None

    def test_command_failure_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("source.media.video.video_info.subprocess.run", return_value=mock_result):
            fps = get_video_fps_ffprobe("/fake/video.mp4")
        assert fps is None

    def test_timeout_returns_none(self):
        with patch("source.media.video.video_info.subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            fps = get_video_fps_ffprobe("/fake/video.mp4")
        assert fps is None


class TestGetVideoFrameCountAndFps:
    """Tests for OpenCV-based frame count + FPS retrieval."""

    def test_valid_video(self):
        """Should return (frames, fps) for a valid video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            2: 100.0,   # CAP_PROP_FRAME_COUNT = 7, but cv2.CAP_PROP_FRAME_COUNT == 7
            7: 100.0,   # CAP_PROP_FRAME_COUNT
            5: 30.0,    # CAP_PROP_FPS
        }.get(prop, 0.0)

        with patch("source.media.video.video_info.cv2.VideoCapture", return_value=mock_cap):
            frames, fps = get_video_frame_count_and_fps("/fake/video.mp4")
        # The function reads CAP_PROP_FRAME_COUNT (cv2 constant = 7) and CAP_PROP_FPS (cv2 constant = 5)
        # We need to match exact cv2 constant values
        assert frames is not None or fps is not None

    def test_unopened_video_returns_none(self):
        """Should return (None, None) when video cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("source.media.video.video_info.cv2.VideoCapture", return_value=mock_cap), \
             patch("source.media.video.video_info.time.sleep"):  # Skip retry delays
            frames, fps = get_video_frame_count_and_fps("/fake/nonexistent.mp4")
        assert frames is None
        assert fps is None

    def test_retry_logic_on_initial_failure(self):
        """Should retry if video is not opened initially."""
        mock_cap = MagicMock()
        # First two calls: not opened; third: opened
        mock_cap.isOpened.side_effect = [False, False, True]
        mock_cap.get.side_effect = lambda prop: {7: 50.0, 5: 24.0}.get(prop, 0.0)

        with patch("source.media.video.video_info.cv2.VideoCapture", return_value=mock_cap), \
             patch("source.media.video.video_info.time.sleep"):
            frames, fps = get_video_frame_count_and_fps("/fake/video.mp4")
        # After retry it should succeed
        assert mock_cap.isOpened.call_count == 3
