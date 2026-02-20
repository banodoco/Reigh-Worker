"""
Tests for source/models/model_handlers/qwen_compositor.py.

Covers:
- QWEN_MAX_DIMENSION constant value
- cap_qwen_resolution: no-op, capping, invalid input, landscape/portrait, defaults
- create_qwen_masked_composite: mocked network + PIL workflow, error wrapping,
  black mask passthrough, raise_for_status, default task_id, portrait resize
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from source.models.model_handlers.qwen_compositor import (
    QWEN_MAX_DIMENSION,
    cap_qwen_resolution,
    create_qwen_masked_composite,
)


# ── Constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_max_dimension_value(self):
        assert QWEN_MAX_DIMENSION == 1200


# ── cap_qwen_resolution ────────────────────────────────────────────────


@patch("source.models.model_handlers.qwen_compositor.model_logger")
class TestCapQwenResolution:
    """Tests for resolution capping logic."""

    def test_no_capping_needed(self, _mock_logger):
        assert cap_qwen_resolution("800x600") == "800x600"

    def test_exact_max_not_capped(self, _mock_logger):
        assert cap_qwen_resolution("1200x900") == "1200x900"

    def test_caps_width(self, _mock_logger):
        # 2400x1200 → ratio = 1200/2400 = 0.5 → 1200x600
        result = cap_qwen_resolution("2400x1200")
        assert result == "1200x600"

    def test_caps_height(self, _mock_logger):
        # 800x2400 → ratio = 1200/2400 = 0.5 → 400x1200
        result = cap_qwen_resolution("800x2400")
        assert result == "400x1200"

    def test_caps_both_dimensions(self, _mock_logger):
        # 2400x2400 → ratio = 0.5 → 1200x1200
        result = cap_qwen_resolution("2400x2400")
        assert result == "1200x1200"

    def test_landscape_aspect_ratio_preserved(self, _mock_logger):
        # 1920x1080 → ratio = min(1200/1920, 1200/1080) = min(0.625, 1.111) = 0.625
        # → int(1920*0.625)=1200, int(1080*0.625)=675
        result = cap_qwen_resolution("1920x1080")
        assert result == "1200x675"

    def test_portrait_aspect_ratio_preserved(self, _mock_logger):
        # 1080x1920 → ratio = min(1200/1080, 1200/1920) = min(1.111, 0.625) = 0.625
        # → int(1080*0.625)=675, int(1920*0.625)=1200
        result = cap_qwen_resolution("1080x1920")
        assert result == "675x1200"

    def test_invalid_format_no_x(self, _mock_logger):
        assert cap_qwen_resolution("800-600") is None

    def test_empty_string_returns_none(self, _mock_logger):
        assert cap_qwen_resolution("") is None

    def test_none_input_returns_none(self, _mock_logger):
        # resolution_str is falsy → returns None
        assert cap_qwen_resolution(None) is None  # type: ignore[arg-type]

    def test_non_numeric_returns_none(self, mock_logger):
        result = cap_qwen_resolution("abcxdef")
        assert result is None
        mock_logger.warning.assert_called_once()

    def test_custom_max_dimension(self, _mock_logger):
        # 800x600 with max_dimension=500 → ratio = min(500/800, 500/600) = 0.625
        # → int(800*0.625)=500, int(600*0.625)=375
        result = cap_qwen_resolution("800x600", max_dimension=500)
        assert result == "500x375"

    def test_task_id_passed_to_logger(self, mock_logger):
        cap_qwen_resolution("2000x2000", task_id="test-42")
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args
        assert "test-42" in call_kwargs[0][0]

    def test_default_task_id_is_unknown(self, mock_logger):
        """Default task_id 'unknown' appears in the warning for invalid input."""
        cap_qwen_resolution("abcxdef")
        msg = mock_logger.warning.call_args[0][0]
        assert "unknown" in msg

    def test_extra_x_separator_returns_none(self, mock_logger):
        """Resolution string with multiple 'x' characters is invalid."""
        result = cap_qwen_resolution("100x200x300")
        assert result is None
        mock_logger.warning.assert_called_once()

    def test_single_dimension_with_x_returns_none(self, mock_logger):
        """A string like 'x600' has an empty first part -> ValueError."""
        result = cap_qwen_resolution("x600")
        assert result is None
        mock_logger.warning.assert_called_once()

    def test_no_capping_at_boundary_minus_one(self, _mock_logger):
        """Both dimensions exactly one below max are not capped."""
        result = cap_qwen_resolution("1199x1199")
        assert result == "1199x1199"

    def test_capping_at_boundary_plus_one(self, _mock_logger):
        """One dimension just above max triggers capping."""
        # 1201x1200 -> ratio = min(1200/1201, 1200/1200) = 1200/1201 ~ 0.99917
        # -> int(1201*0.99917)=1200, int(1200*0.99917)=1199
        result = cap_qwen_resolution("1201x1200")
        assert result is not None
        w, h = map(int, result.split("x"))
        assert w <= 1200
        assert h <= 1200


# ── create_qwen_masked_composite ───────────────────────────────────────


def _make_fake_image_bytes(width: int, height: int, color: str = "red", mode: str = "RGB") -> bytes:
    """Helper to create fake PNG image bytes for mock responses."""
    img = Image.new(mode, (width, height), color)
    buf = BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf.getvalue()


@patch("source.models.model_handlers.qwen_compositor.model_logger")
@patch("source.models.model_handlers.qwen_compositor.requests.get")
class TestCreateQwenMaskedComposite:
    """Tests for the masked composite creation workflow."""

    def test_creates_composite_file(self, mock_get, _mock_logger, tmp_path):
        """End-to-end test: downloads image+mask, produces JPEG composite."""
        image_bytes = _make_fake_image_bytes(100, 100, "red")
        mask_bytes = _make_fake_image_bytes(100, 100, "white", mode="L")

        # Mock requests.get to return image bytes then mask bytes
        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/image.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="test-1",
        )

        # File was created
        result_path = Path(result)
        assert result_path.exists()
        assert result_path.suffix == ".jpg"
        assert "test-1" in result_path.name

        # The result is a valid image
        composite = Image.open(result_path)
        assert composite.size == (100, 100)

    def test_creates_output_directory(self, mock_get, _mock_logger, tmp_path):
        """Output directory is created if it doesn't exist."""
        nested_dir = tmp_path / "sub" / "dir"

        image_bytes = _make_fake_image_bytes(50, 50, "blue")
        mask_bytes = _make_fake_image_bytes(50, 50, "black", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=nested_dir,
            task_id="nested",
        )

        assert nested_dir.exists()
        assert Path(result).exists()

    def test_resizes_large_image(self, mock_get, _mock_logger, tmp_path):
        """Images larger than QWEN_MAX_DIMENSION get resized."""
        image_bytes = _make_fake_image_bytes(2400, 1200, "green")
        mask_bytes = _make_fake_image_bytes(2400, 1200, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/big.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="big",
        )

        composite = Image.open(result)
        w, h = composite.size
        assert w <= QWEN_MAX_DIMENSION
        assert h <= QWEN_MAX_DIMENSION

    def test_mask_resized_to_match_image(self, mock_get, _mock_logger, tmp_path):
        """Mask of different size gets resized to match the image."""
        image_bytes = _make_fake_image_bytes(200, 200, "red")
        # Mask is a different size
        mask_bytes = _make_fake_image_bytes(100, 100, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/image.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="resize-mask",
        )

        composite = Image.open(result)
        assert composite.size == (200, 200)

    def test_green_overlay_applied_on_white_mask(self, mock_get, _mock_logger, tmp_path):
        """Where the mask is white (255), the composite should be green."""
        image_bytes = _make_fake_image_bytes(10, 10, "red")
        mask_bytes = _make_fake_image_bytes(10, 10, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="green",
        )

        composite = Image.open(result).convert("RGB")
        pixel = composite.getpixel((5, 5))
        # Where mask is white → green overlay: (0, 255, 0)
        # JPEG compression may cause slight variation, so check approximate
        assert pixel[1] > 200  # green channel is high
        assert pixel[0] < 50   # red channel is low
        assert pixel[2] < 50   # blue channel is low

    def test_calls_requests_with_timeout(self, mock_get, _mock_logger, tmp_path):
        """Verify requests.get is called with timeout=30."""
        image_bytes = _make_fake_image_bytes(50, 50, "red")
        mask_bytes = _make_fake_image_bytes(50, 50, "black", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="timeout",
        )

        assert mock_get.call_count == 2
        for call in mock_get.call_args_list:
            assert call.kwargs.get("timeout") == 30 or call[1].get("timeout") == 30

    def test_raises_value_error_on_download_failure(self, mock_get, _mock_logger, tmp_path):
        """Network errors are wrapped in ValueError."""
        mock_get.side_effect = OSError("Connection failed")

        with pytest.raises(ValueError, match="Composite image creation failed"):
            create_qwen_masked_composite(
                image_url="http://example.com/bad.png",
                mask_url="http://example.com/mask.png",
                output_dir=tmp_path,
                task_id="fail",
            )

    def test_raises_value_error_on_runtime_error(self, mock_get, _mock_logger, tmp_path):
        """RuntimeError (e.g. from PIL internals) is wrapped in ValueError."""
        mock_get.side_effect = RuntimeError("PIL internal error")

        with pytest.raises(ValueError, match="Composite image creation failed"):
            create_qwen_masked_composite(
                image_url="http://example.com/bad.png",
                mask_url="http://example.com/mask.png",
                output_dir=tmp_path,
                task_id="runtime-fail",
            )

    def test_raises_value_error_on_value_error(self, mock_get, _mock_logger, tmp_path):
        """ValueError from inner code is re-wrapped in ValueError."""
        mock_get.side_effect = ValueError("Bad data")

        with pytest.raises(ValueError, match="Composite image creation failed"):
            create_qwen_masked_composite(
                image_url="http://example.com/bad.png",
                mask_url="http://example.com/mask.png",
                output_dir=tmp_path,
                task_id="val-fail",
            )

    def test_black_mask_preserves_original_image(self, mock_get, _mock_logger, tmp_path):
        """Where the mask is black (0), the original image pixels should show through."""
        image_bytes = _make_fake_image_bytes(10, 10, "red")
        mask_bytes = _make_fake_image_bytes(10, 10, "black", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="black-mask",
        )

        composite = Image.open(result).convert("RGB")
        pixel = composite.getpixel((5, 5))
        # Where mask is black → original red image shows through
        # JPEG compression may cause slight variation
        assert pixel[0] > 200  # red channel is high
        assert pixel[1] < 50   # green channel is low
        assert pixel[2] < 50   # blue channel is low

    def test_default_task_id_in_filename(self, mock_get, _mock_logger, tmp_path):
        """When task_id is not provided, 'unknown' appears in the filename."""
        image_bytes = _make_fake_image_bytes(50, 50, "red")
        mask_bytes = _make_fake_image_bytes(50, 50, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
        )

        assert "unknown" in Path(result).name

    def test_resizes_portrait_image(self, mock_get, _mock_logger, tmp_path):
        """Portrait images taller than QWEN_MAX_DIMENSION are resized correctly."""
        image_bytes = _make_fake_image_bytes(800, 2400, "blue")
        mask_bytes = _make_fake_image_bytes(800, 2400, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        result = create_qwen_masked_composite(
            image_url="http://example.com/tall.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="portrait",
        )

        composite = Image.open(result)
        w, h = composite.size
        assert w <= QWEN_MAX_DIMENSION
        assert h <= QWEN_MAX_DIMENSION
        # Height was the limiting dimension, should be capped to 1200
        assert h == QWEN_MAX_DIMENSION

    def test_raise_for_status_called_for_both_requests(self, mock_get, _mock_logger, tmp_path):
        """Verify raise_for_status is called on both image and mask responses."""
        image_bytes = _make_fake_image_bytes(50, 50, "red")
        mask_bytes = _make_fake_image_bytes(50, 50, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes

        mask_response = MagicMock()
        mask_response.content = mask_bytes

        mock_get.side_effect = [img_response, mask_response]

        create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="status-check",
        )

        img_response.raise_for_status.assert_called_once()
        mask_response.raise_for_status.assert_called_once()

    def test_error_logged_on_failure(self, mock_get, mock_logger, tmp_path):
        """The error logger is called when compositing fails."""
        mock_get.side_effect = OSError("Network failure")

        with pytest.raises(ValueError):
            create_qwen_masked_composite(
                image_url="http://example.com/bad.png",
                mask_url="http://example.com/mask.png",
                output_dir=tmp_path,
                task_id="log-err",
            )

        mock_logger.error.assert_called_once()
        assert "log-err" in mock_logger.error.call_args[0][0]

    def test_success_logged_on_completion(self, mock_get, mock_logger, tmp_path):
        """The info logger is called on successful composite creation."""
        image_bytes = _make_fake_image_bytes(50, 50, "red")
        mask_bytes = _make_fake_image_bytes(50, 50, "white", mode="L")

        img_response = MagicMock()
        img_response.content = image_bytes
        img_response.raise_for_status = MagicMock()

        mask_response = MagicMock()
        mask_response.content = mask_bytes
        mask_response.raise_for_status = MagicMock()

        mock_get.side_effect = [img_response, mask_response]

        create_qwen_masked_composite(
            image_url="http://example.com/img.png",
            mask_url="http://example.com/mask.png",
            output_dir=tmp_path,
            task_id="log-ok",
        )

        mock_logger.info.assert_called_once()
        assert "log-ok" in mock_logger.info.call_args[0][0]
