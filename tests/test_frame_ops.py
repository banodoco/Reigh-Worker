"""Tests for source/media/structure/frame_ops.py."""

import numpy as np
import pytest

from source.media.structure.frame_ops import create_neutral_frame


class TestCreateNeutralFrame:
    """Tests for create_neutral_frame."""

    def test_flow_returns_gray(self):
        """Flow neutral frame should be mid-gray (128)."""
        frame = create_neutral_frame("flow", (320, 240))
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert np.all(frame == 128)

    def test_canny_returns_black(self):
        """Canny neutral frame should be black (0)."""
        frame = create_neutral_frame("canny", (64, 64))
        assert frame.shape == (64, 64, 3)
        assert np.all(frame == 0)

    def test_depth_returns_gray(self):
        """Depth neutral frame should be mid-gray (128)."""
        frame = create_neutral_frame("depth", (100, 50))
        assert frame.shape == (50, 100, 3)
        assert np.all(frame == 128)

    def test_raw_returns_black(self):
        """Raw/Uni3C neutral frame should be black (0)."""
        frame = create_neutral_frame("raw", (200, 100))
        assert frame.shape == (100, 200, 3)
        assert np.all(frame == 0)

    def test_unknown_type_returns_black(self):
        """Unknown structure types should default to black."""
        frame = create_neutral_frame("some_unknown_type", (80, 60))
        assert frame.shape == (60, 80, 3)
        assert np.all(frame == 0)

    def test_resolution_width_height_order(self):
        """Resolution is (width, height) but numpy shape is (height, width, channels)."""
        frame = create_neutral_frame("canny", (320, 240))
        assert frame.shape == (240, 320, 3)

    def test_dtype_is_uint8(self):
        for stype in ("flow", "canny", "depth", "raw", "unknown"):
            frame = create_neutral_frame(stype, (10, 10))
            assert frame.dtype == np.uint8, f"dtype mismatch for {stype}"

    def test_channels_count(self):
        """All neutral frames should have 3 channels (RGB)."""
        for stype in ("flow", "canny", "depth", "raw"):
            frame = create_neutral_frame(stype, (16, 16))
            assert frame.shape[2] == 3

    def test_large_resolution(self):
        """Should handle large resolutions without error."""
        frame = create_neutral_frame("flow", (1920, 1080))
        assert frame.shape == (1080, 1920, 3)

    def test_small_resolution(self):
        """Should handle 1x1 resolution."""
        frame = create_neutral_frame("canny", (1, 1))
        assert frame.shape == (1, 1, 3)
