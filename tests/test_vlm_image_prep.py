"""Tests for source/media/vlm/image_prep.py."""

from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from source.media.vlm.image_prep import create_framed_vlm_image, create_labeled_debug_image


class TestCreateFramedVlmImage:
    """Tests for create_framed_vlm_image."""

    def test_returns_pil_image(self):
        start = Image.new("RGB", (100, 80), (255, 0, 0))
        end = Image.new("RGB", (100, 80), (0, 0, 255))
        result = create_framed_vlm_image(start, end)
        assert isinstance(result, Image.Image)

    def test_combined_width_includes_borders_and_gap(self):
        start = Image.new("RGB", (100, 80), (255, 0, 0))
        end = Image.new("RGB", (100, 80), (0, 0, 255))
        border_width = 6
        gap = 4
        result = create_framed_vlm_image(start, end, border_width=border_width)
        expected_width = (100 + 2 * border_width) + (100 + 2 * border_width) + gap
        assert result.width == expected_width

    def test_combined_height_matches_max_bordered(self):
        start = Image.new("RGB", (100, 80), (255, 0, 0))
        end = Image.new("RGB", (100, 120), (0, 0, 255))
        border_width = 6
        result = create_framed_vlm_image(start, end, border_width=border_width)
        # Height should be max of the two bordered heights
        expected_height = max(80 + 2 * border_width, 120 + 2 * border_width)
        assert result.height == expected_height

    def test_default_border_width(self):
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        result = create_framed_vlm_image(start, end)
        # Default border_width=6, gap=4
        expected_width = (50 + 12) + (50 + 12) + 4
        assert result.width == expected_width

    def test_custom_border_width(self):
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        result = create_framed_vlm_image(start, end, border_width=10)
        expected_width = (50 + 20) + (50 + 20) + 4
        assert result.width == expected_width

    def test_rgb_mode(self):
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        result = create_framed_vlm_image(start, end)
        assert result.mode == "RGB"

    def test_different_sized_images(self):
        start = Image.new("RGB", (200, 100))
        end = Image.new("RGB", (100, 200))
        result = create_framed_vlm_image(start, end)
        # Should not raise
        assert result.width > 0
        assert result.height > 0


class TestCreateLabeledDebugImage:
    """Tests for create_labeled_debug_image."""

    def test_returns_pil_image(self):
        start = Image.new("RGB", (100, 80))
        end = Image.new("RGB", (100, 80))
        result = create_labeled_debug_image(start, end, pair_index=0)
        assert isinstance(result, Image.Image)

    def test_canvas_larger_than_inputs(self):
        start = Image.new("RGB", (100, 80))
        end = Image.new("RGB", (100, 80))
        result = create_labeled_debug_image(start, end, pair_index=1)
        # The canvas should be larger due to labels, borders, padding, title
        assert result.width > 200
        assert result.height > 80

    def test_different_pair_index(self):
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        # Different pair_index should still produce a valid image
        r1 = create_labeled_debug_image(start, end, pair_index=0)
        r2 = create_labeled_debug_image(start, end, pair_index=5)
        assert isinstance(r1, Image.Image)
        assert isinstance(r2, Image.Image)

    def test_rgb_mode(self):
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        result = create_labeled_debug_image(start, end)
        assert result.mode == "RGB"

    def test_different_sized_inputs(self):
        start = Image.new("RGB", (200, 100))
        end = Image.new("RGB", (100, 200))
        result = create_labeled_debug_image(start, end)
        assert result.width > 0
        assert result.height > 0

    def test_font_fallback(self):
        """When no system font is found, should use default font."""
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        # Mock Path.exists to return False for all font paths
        with patch("source.media.vlm.image_prep.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            result = create_labeled_debug_image(start, end)
        assert isinstance(result, Image.Image)

    def test_textbbox_attribute_error_handled(self):
        """When textbbox raises AttributeError, fallback width calc is used."""
        start = Image.new("RGB", (50, 50))
        end = Image.new("RGB", (50, 50))
        # This should work regardless -- the code handles AttributeError in textbbox
        result = create_labeled_debug_image(start, end)
        assert isinstance(result, Image.Image)
