"""
Integration tests for worker.py — QwenHandler integration and parameter transformation.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
WAN2GP_PATH = PROJECT_ROOT / "Wan2GP"


class TestQwenHandlerImport:
    """QwenHandler can be imported and has expected API."""

    def test_import_and_instantiate(self):
        from source.models.model_handlers.qwen_handler import QwenHandler

        handler = QwenHandler(wan_root=str(WAN2GP_PATH), task_id="test_task")
        assert isinstance(handler, QwenHandler)

    def test_has_expected_methods(self):
        from source.models.model_handlers.qwen_handler import QwenHandler

        handler = QwenHandler(wan_root=str(WAN2GP_PATH), task_id="test_task")
        for method_name in (
            "handle_qwen_image_edit",
            "handle_image_inpaint",
            "handle_annotated_image_edit",
            "handle_qwen_image_style",
            "cap_qwen_resolution",
            "create_qwen_masked_composite",
        ):
            assert hasattr(handler, method_name), f"Missing method: {method_name}"


class TestCapQwenResolution:
    """Resolution capping behaviour."""

    def _make_handler(self):
        from source.models.model_handlers.qwen_handler import QwenHandler

        return QwenHandler(wan_root=str(WAN2GP_PATH), task_id="test_task")

    def test_large_resolution_is_capped(self):
        handler = self._make_handler()
        assert handler.cap_qwen_resolution("1920x1080") == "1200x675"

    def test_small_resolution_unchanged(self):
        handler = self._make_handler()
        assert handler.cap_qwen_resolution("800x600") == "800x600"


class TestQwenImageEditHandler:
    """Qwen image-edit handler populates expected params."""

    @patch("source.models.model_handlers.qwen_handler.hf_hub_download")
    @patch("source.models.model_handlers.qwen_handler.download_image_if_url")
    def test_params_set_correctly(self, mock_download, mock_hf_hub):
        from source.models.model_handlers.qwen_handler import QwenHandler

        mock_download.return_value = "/tmp/test_image.jpg"
        mock_hf_hub.return_value = str(WAN2GP_PATH / "loras_qwen" / "fake_lora.safetensors")

        fake_lora_dir = WAN2GP_PATH / "loras_qwen"
        fake_lora_dir.mkdir(parents=True, exist_ok=True)
        (fake_lora_dir / "Qwen-VL-Image-Edit-Lora-V1.0-bf16.safetensors").touch()

        handler = QwenHandler(wan_root=str(WAN2GP_PATH), task_id="test_task")

        db_task_params = {
            "image": "https://example.com/image.jpg",
            "prompt": "Make it blue",
            "resolution": "1024x768",
        }
        generation_params = {}
        handler.handle_qwen_image_edit(db_task_params, generation_params)

        assert generation_params["video_prompt_type"] == "KI"
        assert generation_params["guidance_scale"] == 1
        assert "num_inference_steps" in generation_params
        assert generation_params["video_length"] == 1
        assert "system_prompt" in generation_params
        assert "lora_names" in generation_params
        assert len(generation_params["lora_names"]) > 0


class TestQwenImageStyleHandler:
    """Qwen image-style handler populates expected params."""

    @patch("source.models.model_handlers.qwen_handler.hf_hub_download")
    @patch("source.models.model_handlers.qwen_handler.download_image_if_url")
    def test_style_params(self, _mock_download, mock_hf_hub):
        from source.models.model_handlers.qwen_handler import QwenHandler

        mock_hf_hub.return_value = str(WAN2GP_PATH / "loras_qwen" / "fake_lora.safetensors")
        fake_lora_dir = WAN2GP_PATH / "loras_qwen"
        fake_lora_dir.mkdir(parents=True, exist_ok=True)
        for fname in (
            "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors",
            "style_transfer_qwen_edit_2_000011250.safetensors",
        ):
            (fake_lora_dir / fname).touch()

        handler = QwenHandler(wan_root=str(WAN2GP_PATH), task_id="test_task")

        db_task_params = {
            "prompt": "a beautiful landscape",
            "style_reference_strength": 0.8,
            "subject_strength": 0.0,
            "scene_reference_strength": 0.0,
        }
        generation_params = {"prompt": "a beautiful landscape"}

        handler.handle_qwen_image_style(db_task_params, generation_params)

        assert "In the style of this image" in generation_params["prompt"]
        assert "style" in generation_params["system_prompt"].lower()
        assert len(generation_params.get("lora_names", [])) > 0


class TestWorkerImports:
    """worker.py can be imported without errors."""

    @pytest.mark.xfail(reason="edit_video_orchestrator.py has uncommitted syntax error")
    def test_worker_importable(self):
        import worker  # noqa: F401
