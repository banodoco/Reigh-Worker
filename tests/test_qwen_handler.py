"""Tests for source/models/model_handlers/qwen_handler.py."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from source.models.model_handlers.qwen_handler import (
    QwenHandler,
    QWEN_EDIT_MODEL_CONFIG,
)


class TestQwenEditModelConfig:
    def test_has_expected_variants(self):
        assert "qwen-edit" in QWEN_EDIT_MODEL_CONFIG
        assert "qwen-edit-2509" in QWEN_EDIT_MODEL_CONFIG
        assert "qwen-edit-2511" in QWEN_EDIT_MODEL_CONFIG

    def test_all_variants_have_required_keys(self):
        required_keys = {"model_name", "lightning_fname", "lightning_repo", "hf_subfolder"}
        for variant, config in QWEN_EDIT_MODEL_CONFIG.items():
            assert required_keys <= set(config.keys()), f"Missing keys in {variant}"


class TestQwenHandlerInit:
    def test_creates_lora_dir(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        assert handler.qwen_lora_dir.exists()
        assert handler.qwen_lora_dir == tmp_path / "loras_qwen"

    def test_resolves_wan_root(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        assert handler.wan_root == tmp_path.resolve()


class TestGetEditModelConfig:
    def test_default_variant(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        config = handler._get_edit_model_config({})
        assert config["model_name"] == "qwen_image_edit_20B"

    def test_specific_variant(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        config = handler._get_edit_model_config({"qwen_edit_model": "qwen-edit-2511"})
        assert config["model_name"] == "qwen_image_edit_plus2_20B"

    def test_unknown_variant_fallback(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        config = handler._get_edit_model_config({"qwen_edit_model": "nonexistent"})
        assert config["model_name"] == "qwen_image_edit_20B"

    def test_get_edit_model_name(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        assert handler.get_edit_model_name({}) == "qwen_image_edit_20B"
        assert handler.get_edit_model_name({"qwen_edit_model": "qwen-edit-2509"}) == "qwen_image_edit_plus_20B"


class TestEnsureLoraLists:
    def test_creates_missing_keys(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        params = {}
        handler._ensure_lora_lists(params)
        assert params["lora_names"] == []
        assert params["lora_multipliers"] == []

    def test_preserves_existing_keys(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        params = {"lora_names": ["a.safetensors"], "lora_multipliers": [1.0]}
        handler._ensure_lora_lists(params)
        assert params["lora_names"] == ["a.safetensors"]
        assert params["lora_multipliers"] == [1.0]


class TestApplyAdditionalLoras:
    def test_array_format(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {}
        db_params = {
            "loras": [
                {"path": "https://example.com/a.safetensors", "scale": 0.8},
                {"path": "https://example.com/b.safetensors", "scale": 0.5},
            ]
        }
        handler._apply_additional_loras(db_params, gen_params)
        assert gen_params["additional_loras"]["https://example.com/a.safetensors"] == 0.8
        assert gen_params["additional_loras"]["https://example.com/b.safetensors"] == 0.5

    def test_dict_format(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {}
        db_params = {
            "additional_loras": {"https://example.com/a.safetensors": 0.9}
        }
        handler._apply_additional_loras(db_params, gen_params)
        assert gen_params["additional_loras"]["https://example.com/a.safetensors"] == 0.9

    def test_empty_loras(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {}
        handler._apply_additional_loras({}, gen_params)
        assert "additional_loras" not in gen_params

    def test_empty_path_skipped(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {}
        handler._apply_additional_loras({"loras": [{"path": "", "scale": 0.5}]}, gen_params)
        assert "additional_loras" not in gen_params


class TestMaybeAddHiresConfig:
    def test_no_hires_scale_skips(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {"lora_names": [], "lora_multipliers": []}
        handler._maybe_add_hires_config({}, gen_params)
        assert "hires_config" not in gen_params

    def test_adds_hires_config(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {"lora_names": [], "lora_multipliers": []}
        handler._maybe_add_hires_config({"hires_scale": 2.0}, gen_params)
        assert gen_params["hires_config"]["enabled"] is True
        assert gen_params["hires_config"]["scale"] == 2.0

    def test_hires_config_defaults(self, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {"lora_names": [], "lora_multipliers": []}
        handler._maybe_add_hires_config({"hires_scale": 1.5}, gen_params)
        assert gen_params["hires_config"]["hires_steps"] == 6
        assert gen_params["hires_config"]["denoising_strength"] == 0.5
        assert gen_params["hires_config"]["upscale_method"] == "bicubic"


class TestHandleQwenImageEdit:
    @patch("source.models.model_handlers.qwen_handler.download_image_if_url")
    def test_requires_image(self, mock_dl, tmp_path):
        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        with pytest.raises(ValueError, match="image.*required"):
            handler.handle_qwen_image_edit({}, {})

    @patch("source.models.model_handlers.qwen_handler.download_image_if_url")
    def test_sets_defaults(self, mock_dl, tmp_path):
        mock_dl.return_value = "/fake/local/image.png"
        # Create the lightning LoRA file so download isn't attempted
        lora_dir = tmp_path / "loras_qwen"
        lora_dir.mkdir(exist_ok=True)
        (lora_dir / "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors").touch()

        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {}
        handler.handle_qwen_image_edit({"image": "https://example.com/img.png"}, gen_params)

        assert gen_params["image_guide"] == "/fake/local/image.png"
        assert gen_params["video_prompt_type"] == "KI"
        assert gen_params["guidance_scale"] == 1
        assert gen_params["num_inference_steps"] == 12
        assert gen_params["video_length"] == 1
        assert "system_prompt" in gen_params


class TestHandleQwenImage:
    def test_sets_text_to_image_defaults(self, tmp_path):
        # Create the lightning LoRA file
        lora_dir = tmp_path / "loras_qwen"
        lora_dir.mkdir(exist_ok=True)
        (lora_dir / "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors").touch()

        handler = QwenHandler(wan_root=str(tmp_path), task_id="t1")
        gen_params = {}
        handler.handle_qwen_image({"resolution": "1024x1024"}, gen_params)

        assert gen_params["video_prompt_type"] == ""  # No input image
        assert gen_params["guidance_scale"] == 3.5
        assert gen_params["num_inference_steps"] == 4
        assert gen_params["video_length"] == 1
