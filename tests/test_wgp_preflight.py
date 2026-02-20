"""Tests for source/models/wgp/generators/preflight.py."""

import pytest
from unittest.mock import patch, MagicMock


class TestPrepareSviImageRefs:
    """Tests for prepare_svi_image_refs."""

    def test_no_image_refs_paths_does_nothing(self):
        """Should not modify kwargs when image_refs_paths is missing."""
        from source.models.wgp.generators.preflight import prepare_svi_image_refs

        kwargs = {"prompt": "test"}
        prepare_svi_image_refs(kwargs)
        assert "image_refs" not in kwargs

    def test_empty_image_refs_paths_does_nothing(self):
        """Should not modify kwargs when image_refs_paths is empty list."""
        from source.models.wgp.generators.preflight import prepare_svi_image_refs

        kwargs = {"image_refs_paths": []}
        prepare_svi_image_refs(kwargs)
        assert "image_refs" not in kwargs

    def test_existing_image_refs_prevents_conversion(self):
        """Should not overwrite existing image_refs with loaded images."""
        from source.models.wgp.generators.preflight import prepare_svi_image_refs

        existing_refs = [MagicMock()]
        kwargs = {
            "image_refs_paths": ["/path/img1.png"],
            "image_refs": existing_refs,
        }
        prepare_svi_image_refs(kwargs)
        assert kwargs["image_refs"] is existing_refs

    @patch("source.models.wgp.generators.preflight.is_debug_enabled", return_value=False)
    def test_converts_paths_to_pil_images(self, mock_debug):
        """Should convert string paths to PIL Image objects."""
        from source.models.wgp.generators.preflight import prepare_svi_image_refs

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        with patch("PIL.Image.open", return_value=mock_img) as mock_open:
            with patch("PIL.ImageOps.exif_transpose", return_value=mock_img):
                kwargs = {"image_refs_paths": ["/path/img1.png", "/path/img2.png"]}
                prepare_svi_image_refs(kwargs)

        assert "image_refs" in kwargs
        assert len(kwargs["image_refs"]) == 2

    @patch("source.models.wgp.generators.preflight.is_debug_enabled", return_value=True)
    def test_skips_empty_paths(self, mock_debug):
        """Should skip empty/None paths in the list."""
        from source.models.wgp.generators.preflight import prepare_svi_image_refs

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        with patch("PIL.Image.open", return_value=mock_img):
            with patch("PIL.ImageOps.exif_transpose", return_value=mock_img):
                kwargs = {"image_refs_paths": ["", "/path/img1.png", None]}
                prepare_svi_image_refs(kwargs)

        assert len(kwargs["image_refs"]) == 1

    @patch("source.models.wgp.generators.preflight.is_debug_enabled", return_value=True)
    def test_os_error_on_load_skips_image(self, mock_debug):
        """Should skip images that fail to load."""
        from source.models.wgp.generators.preflight import prepare_svi_image_refs

        with patch("PIL.Image.open", side_effect=OSError("file not found")):
            with patch("PIL.ImageOps.exif_transpose"):
                kwargs = {"image_refs_paths": ["/bad/path.png"]}
                prepare_svi_image_refs(kwargs)

        # No image_refs set since no images loaded successfully
        assert "image_refs" not in kwargs


class TestConfigureModelSpecificParams:
    """Tests for configure_model_specific_params."""

    def test_flux_model_params(self):
        """Flux models should use image_mode=1 and embedded guidance."""
        from source.models.wgp.generators.preflight import configure_model_specific_params

        result = configure_model_specific_params(
            is_flux=True,
            is_qwen=False,
            is_vace=False,
            resolved_params={},
            final_video_length=81,
            final_batch_size=4,
            final_guidance_scale=7.0,
            final_embedded_guidance=3.5,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["image_mode"] == 1
        assert result["actual_video_length"] == 1
        assert result["actual_batch_size"] == 81  # = final_video_length for flux
        assert result["actual_guidance"] == 3.5  # embedded guidance

    def test_qwen_model_params(self):
        """Qwen models should use image_mode=1 and standard guidance."""
        from source.models.wgp.generators.preflight import configure_model_specific_params

        result = configure_model_specific_params(
            is_flux=False,
            is_qwen=True,
            is_vace=False,
            resolved_params={"batch_size": 2},
            final_video_length=81,
            final_batch_size=4,
            final_guidance_scale=7.0,
            final_embedded_guidance=3.5,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["image_mode"] == 1
        assert result["actual_video_length"] == 1
        assert result["actual_batch_size"] == 2  # from resolved_params
        assert result["actual_guidance"] == 7.0

    def test_t2v_model_params(self):
        """T2V/VACE models should use image_mode=0."""
        from source.models.wgp.generators.preflight import configure_model_specific_params

        result = configure_model_specific_params(
            is_flux=False,
            is_qwen=False,
            is_vace=False,
            resolved_params={},
            final_video_length=81,
            final_batch_size=4,
            final_guidance_scale=7.0,
            final_embedded_guidance=3.5,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["image_mode"] == 0
        assert result["actual_video_length"] == 81
        assert result["actual_batch_size"] == 4

    def test_t2v_short_video_length_boosted(self):
        """T2V models with video_length < 5 should be boosted to 5."""
        from source.models.wgp.generators.preflight import configure_model_specific_params

        result = configure_model_specific_params(
            is_flux=False,
            is_qwen=False,
            is_vace=False,
            resolved_params={},
            final_video_length=3,
            final_batch_size=1,
            final_guidance_scale=7.0,
            final_embedded_guidance=3.5,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["actual_video_length"] == 5

    def test_non_vace_disables_controls(self):
        """Non-VACE models without guides should have controls disabled."""
        from source.models.wgp.generators.preflight import configure_model_specific_params

        result = configure_model_specific_params(
            is_flux=False,
            is_qwen=False,
            is_vace=False,
            resolved_params={},
            final_video_length=81,
            final_batch_size=4,
            final_guidance_scale=7.0,
            final_embedded_guidance=3.5,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=1.0,
            control_net_weight2=0.5,
        )

        assert result["video_guide"] is None
        assert result["video_mask"] is None
        assert result["video_prompt_type"] == "disabled"
        assert result["control_net_weight"] == 0.0
        assert result["control_net_weight2"] == 0.0

    def test_vace_preserves_controls(self):
        """VACE models should preserve video_guide and control weights."""
        from source.models.wgp.generators.preflight import configure_model_specific_params

        result = configure_model_specific_params(
            is_flux=False,
            is_qwen=False,
            is_vace=True,
            resolved_params={},
            final_video_length=81,
            final_batch_size=4,
            final_guidance_scale=7.0,
            final_embedded_guidance=3.5,
            video_guide="/path/guide.mp4",
            video_mask="/path/mask.mp4",
            video_prompt_type="i2v",
            control_net_weight=0.8,
            control_net_weight2=0.3,
        )

        assert result["video_guide"] == "/path/guide.mp4"
        assert result["video_mask"] == "/path/mask.mp4"
        assert result["video_prompt_type"] == "i2v"
        assert result["control_net_weight"] == 0.8
        assert result["control_net_weight2"] == 0.3


class TestPrepareImageInputs:
    """Tests for prepare_image_inputs."""

    def test_string_image_start_loaded(self):
        """Should load image_start from string path."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_img = MagicMock()
        mock_img.size = (896, 496)
        mock_load = MagicMock(return_value=mock_img)

        wgp_params = {
            "image_start": "/path/to/start.png",
            "resolution": "896x496",
        }

        prepare_image_inputs(
            wgp_params,
            is_qwen=False,
            image_mode=0,
            load_image_fn=mock_load,
        )

        mock_load.assert_called_with("/path/to/start.png", mask=False)
        assert wgp_params["image_start"] is mock_img

    def test_image_resized_to_target(self):
        """Should resize image when it doesn't match target resolution."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_img = MagicMock()
        mock_img.size = (640, 480)
        resized_img = MagicMock()
        mock_img.resize.return_value = resized_img
        mock_load = MagicMock(return_value=mock_img)

        wgp_params = {
            "image_start": "/path/to/start.png",
            "resolution": "896x496",
        }

        prepare_image_inputs(
            wgp_params,
            is_qwen=False,
            image_mode=0,
            load_image_fn=mock_load,
        )

        mock_img.resize.assert_called_once()
        assert wgp_params["image_start"] is resized_img

    def test_list_of_image_paths_loaded(self):
        """Should load list of image paths for image_start."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_img = MagicMock()
        mock_img.size = (896, 496)
        mock_load = MagicMock(return_value=mock_img)

        wgp_params = {
            "image_start": ["/path/img1.png", "/path/img2.png"],
            "resolution": "896x496",
        }

        prepare_image_inputs(
            wgp_params,
            is_qwen=False,
            image_mode=0,
            load_image_fn=mock_load,
        )

        assert isinstance(wgp_params["image_start"], list)
        assert len(wgp_params["image_start"]) == 2

    def test_qwen_loads_guide_and_mask(self):
        """Qwen models should load image_guide and image_mask."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_guide = MagicMock()
        mock_mask = MagicMock()

        def mock_load(path, mask=False):
            if mask:
                return mock_mask
            return mock_guide

        wgp_params = {
            "image_guide": "/path/guide.png",
            "image_mask": "/path/mask.png",
            "resolution": "896x496",
        }

        prepare_image_inputs(
            wgp_params,
            is_qwen=True,
            image_mode=1,
            load_image_fn=mock_load,
        )

        assert wgp_params["image_guide"] is mock_guide
        assert wgp_params["image_mask"] is mock_mask
        assert wgp_params["model_mode"] == 1

    def test_qwen_without_mask_sets_none(self):
        """Qwen models without mask should set image_mask=None."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_guide = MagicMock()
        mock_load = MagicMock(return_value=mock_guide)

        wgp_params = {
            "image_guide": "/path/guide.png",
            "resolution": "896x496",
        }

        prepare_image_inputs(
            wgp_params,
            is_qwen=True,
            image_mode=1,
            load_image_fn=mock_load,
        )

        assert wgp_params["image_mask"] is None

    def test_empty_image_refs_sanitized_to_none(self):
        """Empty image_refs list should be sanitized to None."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_load = MagicMock()
        wgp_params = {"image_refs": [], "resolution": "896x496"}

        prepare_image_inputs(
            wgp_params,
            is_qwen=False,
            image_mode=0,
            load_image_fn=mock_load,
        )

        assert wgp_params["image_refs"] is None

    def test_no_resolution_still_works(self):
        """Should work even without a resolution parameter."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_img = MagicMock()
        mock_load = MagicMock(return_value=mock_img)

        wgp_params = {"image_start": "/path/img.png"}

        prepare_image_inputs(
            wgp_params,
            is_qwen=False,
            image_mode=0,
            load_image_fn=mock_load,
        )

        # Image loaded but not resized (no target dimensions)
        mock_load.assert_called_with("/path/img.png", mask=False)

    def test_invalid_resolution_handled(self):
        """Should handle unparseable resolution gracefully."""
        from source.models.wgp.generators.preflight import prepare_image_inputs

        mock_img = MagicMock()
        mock_load = MagicMock(return_value=mock_img)

        wgp_params = {
            "image_start": "/path/img.png",
            "resolution": "invalid",
        }

        # Should not raise
        prepare_image_inputs(
            wgp_params,
            is_qwen=False,
            image_mode=0,
            load_image_fn=mock_load,
        )
