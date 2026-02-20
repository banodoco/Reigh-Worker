"""Tests for source/utils/resolution_utils.py."""

import pytest

from source.utils.resolution_utils import parse_resolution, snap_resolution_to_model_grid


class TestParseResolution:
    """parse_resolution: 'WIDTHxHEIGHT' → (width, height)."""

    @pytest.mark.parametrize("input_str,expected", [
        ("960x544", (960, 544)),
        ("1920x1080", (1920, 1080)),
        ("1x1", (1, 1)),
        ("512x512", (512, 512)),
    ])
    def test_valid_formats(self, input_str, expected):
        assert parse_resolution(input_str) == expected

    @pytest.mark.parametrize("bad_input", [
        "960",
        "960:544",
        "widthxheight",
        "",
        "x",
        "960x",
        "x544",
    ])
    def test_invalid_format_raises(self, bad_input):
        with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
            parse_resolution(bad_input)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("0x544")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("-1x544")


class TestSnapResolution:
    """snap_resolution_to_model_grid: snap to multiples of 16."""

    def test_already_aligned(self):
        assert snap_resolution_to_model_grid((960, 544)) == (960, 544)

    def test_needs_snapping(self):
        assert snap_resolution_to_model_grid((967, 550)) == (960, 544)

    def test_small_values(self):
        assert snap_resolution_to_model_grid((17, 17)) == (16, 16)

    def test_exact_multiples(self):
        assert snap_resolution_to_model_grid((32, 64)) == (32, 64)

    def test_rounds_down(self):
        # 31 // 16 * 16 = 16
        assert snap_resolution_to_model_grid((31, 31)) == (16, 16)
