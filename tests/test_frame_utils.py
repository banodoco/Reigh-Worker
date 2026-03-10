"""Tests for source/utils/frame_utils.py."""

import math

import numpy as np
import pytest

from source.utils.frame_utils import (
    create_color_frame,
    get_easing_function,
    adjust_frame_brightness,
    get_sequential_target_path,
)


class TestCreateColorFrame:
    def test_default_black(self):
        frame = create_color_frame((320, 240))
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert np.all(frame == 0)

    def test_custom_color(self):
        frame = create_color_frame((64, 64), color_bgr=(255, 0, 128))
        assert np.all(frame[:, :, 0] == 255)
        assert np.all(frame[:, :, 1] == 0)
        assert np.all(frame[:, :, 2] == 128)

    def test_shape_is_height_width_3(self):
        frame = create_color_frame((100, 50))
        # size is (width, height) → numpy is (height, width, 3)
        assert frame.shape == (50, 100, 3)

    def test_dtype_uint8(self):
        frame = create_color_frame((10, 10))
        assert frame.dtype == np.uint8


class TestGetEasingFunction:
    @pytest.mark.parametrize("name", [
        "linear",
        "ease_in_quad",
        "ease_out_quad",
        "ease_in_out_quad",
        "ease_in_out",
        "ease_in_cubic",
        "ease_out_cubic",
        "ease_in_out_cubic",
        "ease_in_sine",
        "ease_out_sine",
        "ease_in_out_sine",
        "ease_in_expo",
        "ease_out_expo",
        "ease_in_out_expo",
    ])
    def test_boundaries(self, name):
        fn = get_easing_function(name)
        assert fn(0) == pytest.approx(0, abs=1e-9)
        assert fn(1) == pytest.approx(1, abs=1e-9)

    def test_linear_midpoint(self):
        fn = get_easing_function("linear")
        assert fn(0.5) == pytest.approx(0.5)

    def test_ease_in_quad_values(self):
        fn = get_easing_function("ease_in_quad")
        assert fn(0.5) == pytest.approx(0.25)

    def test_unknown_defaults_to_ease_in_out(self):
        fn = get_easing_function("nonexistent_easing")
        ease_in_out = get_easing_function("ease_in_out_quad")
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            assert fn(t) == pytest.approx(ease_in_out(t))


class TestAdjustFrameBrightness:
    def test_zero_factor_no_change(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = adjust_frame_brightness(frame, 0.0)
        # alpha = 1.0 - 0.0 = 1.0 → same values
        np.testing.assert_array_equal(result, frame)

    def test_positive_factor_darker(self):
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_frame_brightness(frame, 0.5)
        # alpha = 1.0 - 0.5 = 0.5 → darker
        assert result.mean() < frame.mean()

    def test_negative_factor_brighter(self):
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = adjust_frame_brightness(frame, -0.5)
        # alpha = 1.0 - (-0.5) = 1.5 → brighter
        assert result.mean() > frame.mean()

    def test_result_is_uint8(self):
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_frame_brightness(frame, -1.0)
        assert result.dtype == np.uint8


class TestGetSequentialTargetPath:
    def test_no_collision(self, tmp_path):
        result = get_sequential_target_path(tmp_path, "video", ".mp4")
        assert result == tmp_path / "video.mp4"

    def test_single_collision(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        result = get_sequential_target_path(tmp_path, "video", ".mp4")
        assert result == tmp_path / "video_1.mp4"

    def test_multiple_collisions(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        (tmp_path / "video_1.mp4").touch()
        (tmp_path / "video_2.mp4").touch()
        result = get_sequential_target_path(tmp_path, "video", ".mp4")
        assert result == tmp_path / "video_3.mp4"

    def test_suffix_without_dot(self, tmp_path):
        result = get_sequential_target_path(tmp_path, "file", "txt")
        assert result == tmp_path / "file.txt"
