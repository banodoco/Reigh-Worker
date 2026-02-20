"""Tests for source/media/video/color_matching.py."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from source.media.video.color_matching import (
    _cm_enhance_saturation,
    _cm_transfer_mean_std_lab,
    apply_color_matching_to_video,
)


class TestCmEnhanceSaturation:
    """Tests for the saturation adjustment helper."""

    def test_full_saturation_no_change(self):
        """Factor 1.0 should leave image nearly unchanged."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = _cm_enhance_saturation(img, saturation_factor=1.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_zero_saturation_produces_grayscale(self):
        """Factor 0.0 should remove all color, producing grayscale."""
        # Create a brightly colored image
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Full red channel (BGR)
        img[:, :, 1] = 0
        img[:, :, 0] = 0
        result = _cm_enhance_saturation(img, saturation_factor=0.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        # All channels should be equal (grayscale) when saturation is zero
        # Allow small rounding differences from color space conversion
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=2)
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=2)

    def test_half_saturation_reduces_color(self):
        """Factor 0.5 should reduce saturation compared to original."""
        import cv2
        # Create saturated image
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # Red in BGR
        img[:, :, 1] = 50
        img[:, :, 0] = 50
        result = _cm_enhance_saturation(img, saturation_factor=0.5)
        # Convert both to HSV and compare saturation channels
        hsv_orig = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        # Result saturation should be lower
        assert hsv_result[:, :, 1].mean() < hsv_orig[:, :, 1].mean()

    def test_output_shape_matches_input(self):
        img = np.random.randint(0, 256, (100, 50, 3), dtype=np.uint8)
        result = _cm_enhance_saturation(img, saturation_factor=0.7)
        assert result.shape == img.shape

    def test_deps_unavailable_returns_input(self):
        """When _COLOR_MATCH_DEPS_AVAILABLE is False, should return input unchanged."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        with patch("source.media.video.color_matching._COLOR_MATCH_DEPS_AVAILABLE", False):
            result = _cm_enhance_saturation(img, saturation_factor=0.5)
            np.testing.assert_array_equal(result, img)


class TestCmTransferMeanStdLab:
    """Tests for LAB color transfer."""

    def test_output_shape_and_dtype(self):
        source = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = _cm_transfer_mean_std_lab(source, target)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_identical_images_similar_output(self):
        """Transferring colors from an image to itself should produce similar result."""
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        result = _cm_transfer_mean_std_lab(img, img)
        # Should be very close to original
        diff = np.abs(result.astype(float) - img.astype(float))
        assert diff.mean() < 10  # Allow some rounding error from LAB conversion

    def test_transfers_color_characteristics(self):
        """Result should take on target's color characteristics."""
        # Source: blue image
        source = np.zeros((32, 32, 3), dtype=np.uint8)
        source[:, :, 0] = 200  # Blue channel in BGR
        source[:, :, 1] = 50
        source[:, :, 2] = 50

        # Target: red image
        target = np.zeros((32, 32, 3), dtype=np.uint8)
        target[:, :, 0] = 50
        target[:, :, 1] = 50
        target[:, :, 2] = 200  # Red channel in BGR

        result = _cm_transfer_mean_std_lab(source, target)
        # Result should be shifted toward red
        assert result[:, :, 2].mean() > source[:, :, 2].mean()

    def test_uniform_source_low_std(self):
        """When source has near-zero std, should fill with target mean."""
        # Uniform source (std ~ 0)
        source = np.full((32, 32, 3), 128, dtype=np.uint8)
        target = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        result = _cm_transfer_mean_std_lab(source, target)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_deps_unavailable_returns_source(self):
        source = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        target = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        with patch("source.media.video.color_matching._COLOR_MATCH_DEPS_AVAILABLE", False):
            result = _cm_transfer_mean_std_lab(source, target)
            np.testing.assert_array_equal(result, source)


class TestApplyColorMatchingToVideo:
    """Tests for the main video color matching function."""

    def test_missing_file_returns_none(self, tmp_path):
        """Should return None when input files don't exist."""
        result = apply_color_matching_to_video(
            str(tmp_path / "nonexistent.mp4"),
            str(tmp_path / "start.png"),
            str(tmp_path / "end.png"),
            str(tmp_path / "output.mp4"),
        )
        assert result is None

    def test_deps_unavailable_returns_none(self, tmp_path):
        """Should return None when deps are not available."""
        # Create dummy files
        for name in ["input.mp4", "start.png", "end.png"]:
            (tmp_path / name).touch()
        with patch("source.media.video.color_matching._COLOR_MATCH_DEPS_AVAILABLE", False):
            result = apply_color_matching_to_video(
                str(tmp_path / "input.mp4"),
                str(tmp_path / "start.png"),
                str(tmp_path / "end.png"),
                str(tmp_path / "output.mp4"),
            )
        assert result is None

    def test_empty_frames_returns_none(self, tmp_path):
        """Should return None when frame extraction returns empty list."""
        for name in ["input.mp4", "start.png", "end.png"]:
            (tmp_path / name).touch()
        with patch("source.media.video.color_matching.extract_frames_from_video", return_value=[]), \
             patch("source.media.video.color_matching.get_video_frame_count_and_fps", return_value=(10, 30.0)):
            result = apply_color_matching_to_video(
                str(tmp_path / "input.mp4"),
                str(tmp_path / "start.png"),
                str(tmp_path / "end.png"),
                str(tmp_path / "output.mp4"),
            )
        assert result is None

    def test_successful_color_matching(self, tmp_path):
        """Should process frames and call create_video_from_frames_list."""
        import cv2
        for name in ["start.png", "end.png"]:
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / name), img)

        # Create dummy input video file
        (tmp_path / "input.mp4").touch()

        fake_frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

        with patch("source.media.video.color_matching.extract_frames_from_video", return_value=fake_frames), \
             patch("source.media.video.color_matching.get_video_frame_count_and_fps", return_value=(5, 30.0)), \
             patch("source.media.video.color_matching.create_video_from_frames_list", return_value=str(tmp_path / "output.mp4")) as mock_create:
            result = apply_color_matching_to_video(
                str(tmp_path / "input.mp4"),
                str(tmp_path / "start.png"),
                str(tmp_path / "end.png"),
                str(tmp_path / "output.mp4"),
            )
        assert result == str(tmp_path / "output.mp4")
        mock_create.assert_called_once()
        # Verify the frames list was passed
        args = mock_create.call_args
        assert len(args[0][0]) == 5  # 5 accumulated frames
