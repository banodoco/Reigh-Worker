"""Tests for source/task_handlers/tasks/task_conversion.py.

This module has heavy external dependencies (QwenHandler, GenerationTask, db_operations),
so we mock them and test the core conversion logic paths.
"""

import pytest
from unittest import mock
from unittest.mock import MagicMock, patch

# We need to mock several imports before importing the module under test
# because task_conversion.py imports QwenHandler, GenerationTask, etc. at module level.


@pytest.fixture(autouse=True)
def _mock_heavy_deps():
    """Mock external dependencies that task_conversion imports at module level."""
    # Already imported at module level, so we patch the references in the module
    pass


# Import the module constants directly (no external deps needed)
from source.task_handlers.tasks.task_conversion import (
    IMG2IMG_TARGET_MEGAPIXELS,
    DEFAULT_IMAGE_RESOLUTION,
)


class TestModuleConstants:
    def test_img2img_target_megapixels(self):
        assert IMG2IMG_TARGET_MEGAPIXELS == 1024 * 1024

    def test_default_image_resolution(self):
        assert DEFAULT_IMAGE_RESOLUTION == "1024x1024"


class TestDbTaskToGenerationTask:
    """Test db_task_to_generation_task with mocked dependencies."""

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_basic_text_generation(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {
            "prompt": "a beautiful sunset",
            "model": "t2v_22",
            "seed": 42,
        }
        result = db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        # Verify GenerationTask was called with correct args
        MockGenTask.assert_called_once()
        call_kwargs = MockGenTask.call_args[1]
        assert call_kwargs["id"] == "task-1"
        assert call_kwargs["model"] == "t2v_22"
        assert call_kwargs["prompt"] == "a beautiful sunset"
        assert call_kwargs["parameters"]["seed"] == 42
        assert call_kwargs["priority"] == 0

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_empty_prompt_raises_for_non_img2img(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        db_params = {"model": "t2v_22"}

        with pytest.raises(ValueError, match="prompt is required"):
            db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_empty_prompt_ok_for_img2img(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        # img2img task types accept empty prompts
        for task_type in ["z_image_turbo_i2i", "qwen_image_edit", "qwen_image_style", "image_inpaint"]:
            db_params = {"model": "some_model"}
            # z_image_turbo_i2i needs extra params but qwen_image_edit etc. go through QwenHandler
            if task_type == "z_image_turbo_i2i":
                continue  # skip - needs image download
            result = db_task_to_generation_task(db_params, "task-1", task_type, "/wan2gp")
            # Should not raise

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_param_whitelist_filtering(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {
            "prompt": "test",
            "model": "t2v_22",
            "resolution": "1280x720",
            "guidance_scale": 7.5,
            "seed": 42,
            "secret_internal_field": "should_not_pass",
            "random_unknown_param": 123,
        }
        db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        gen_params = call_kwargs["parameters"]
        assert gen_params["resolution"] == "1280x720"
        assert gen_params["guidance_scale"] == 7.5
        assert gen_params["seed"] == 42
        # Non-whitelisted params should be excluded
        assert "secret_internal_field" not in gen_params
        assert "random_unknown_param" not in gen_params

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_essential_defaults(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {"prompt": "test", "model": "t2v"}
        db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        gen_params = call_kwargs["parameters"]
        assert gen_params["seed"] == -1
        assert gen_params["negative_prompt"] == ""

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_steps_alias(self, MockGenTask, mock_extract, MockQwen):
        """'steps' in db params should map to num_inference_steps."""
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {"prompt": "test", "model": "t2v", "steps": 30}
        db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        assert call_kwargs["parameters"]["num_inference_steps"] == 30

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_orchestrator_priority_boost(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {"prompt": "test", "model": "t2v"}
        db_task_to_generation_task(db_params, "task-1", "travel_orchestrator", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        assert call_kwargs["priority"] >= 10

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_z_image_turbo_setup(self, MockGenTask, mock_extract, MockQwen):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {"prompt": "test image", "model": "z_image", "resolution": "512x512"}
        db_task_to_generation_task(db_params, "task-1", "z_image_turbo", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        assert call_kwargs["model"] == "z_image"
        gen_params = call_kwargs["parameters"]
        assert gen_params["video_length"] == 1
        assert gen_params["guidance_scale"] == 0
        assert gen_params["resolution"] == "512x512"

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_default_model_fallback(self, MockGenTask, mock_extract, MockQwen):
        """When model is not provided, get_default_model should be called."""
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        # get_default_model is imported lazily inside the function from task_types,
        # so we patch it at its source module
        with patch("source.task_handlers.tasks.task_types.get_default_model", return_value="fallback_model"):
            db_params = {"prompt": "test"}  # No model key
            db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        assert call_kwargs["model"] == "fallback_model"

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_uni3c_params_whitelisted(self, MockGenTask, mock_extract, MockQwen):
        """Uni3C parameters should pass through the whitelist."""
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {}
        MockGenTask.return_value = MagicMock()

        db_params = {
            "prompt": "test",
            "model": "t2v",
            "use_uni3c": True,
            "uni3c_guide_video": "/guide.mp4",
            "uni3c_strength": 0.8,
            "uni3c_start_percent": 0.1,
            "uni3c_end_percent": 0.9,
        }
        db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        gen_params = call_kwargs["parameters"]
        assert gen_params["use_uni3c"] is True
        assert gen_params["uni3c_guide_video"] == "/guide.mp4"
        assert gen_params["uni3c_strength"] == 0.8

    @patch("source.task_handlers.tasks.task_conversion.QwenHandler")
    @patch("source.task_handlers.tasks.task_conversion.extract_orchestrator_parameters")
    @patch("source.task_handlers.tasks.task_conversion.GenerationTask")
    def test_additional_loras_from_extracted(self, MockGenTask, mock_extract, MockQwen):
        """additional_loras from extract_orchestrator_parameters should be used if not in generation_params."""
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task

        mock_extract.return_value = {
            "additional_loras": {"https://example.com/lora.safetensors": "1.0"}
        }
        MockGenTask.return_value = MagicMock()

        db_params = {"prompt": "test", "model": "t2v"}
        db_task_to_generation_task(db_params, "task-1", "t2v", "/wan2gp")

        call_kwargs = MockGenTask.call_args[1]
        gen_params = call_kwargs["parameters"]
        assert gen_params["additional_loras"] == {"https://example.com/lora.safetensors": "1.0"}


class TestDbTaskToGenerationTaskDirect:
    """Direct conversion checks with monkeypatch stubs instead of patch decorators."""

    def test_direct_conversion_keeps_whitelisted_fields_and_defaults(self, monkeypatch):
        import source.task_handlers.tasks.task_conversion as tc

        created = {}

        class _Task:
            def __init__(self, id, model, prompt, parameters, priority):
                created["id"] = id
                created["model"] = model
                created["prompt"] = prompt
                created["parameters"] = parameters
                created["priority"] = priority

        class _Qwen:
            def __init__(self, wan_root, task_id):
                self.wan_root = wan_root
                self.task_id = task_id

            def handle_qwen_image_edit(self, *_args, **_kwargs):
                return None

            def handle_qwen_image_hires(self, *_args, **_kwargs):
                return None

            def handle_image_inpaint(self, *_args, **_kwargs):
                return None

            def handle_annotated_image_edit(self, *_args, **_kwargs):
                return None

            def handle_qwen_image_style(self, *_args, **_kwargs):
                return None

            def handle_qwen_image(self, *_args, **_kwargs):
                return None

            def handle_qwen_image_2512(self, *_args, **_kwargs):
                return None

        monkeypatch.setattr(tc, "GenerationTask", _Task)
        monkeypatch.setattr(tc, "QwenHandler", _Qwen)
        monkeypatch.setattr(tc, "extract_orchestrator_parameters", lambda _db, _task: {})

        result = tc.db_task_to_generation_task(
            {
                "prompt": "direct prompt",
                "model": "wan22",
                "resolution": "768x432",
                "guidance_scale": 6.5,
                "num_inference_steps": 12,
                "seed": 123,
                "negative_prompt": "blurry",
                "use_uni3c": True,
                "uni3c_guide_video": "/tmp/guide.mp4",
                "uni3c_strength": 0.75,
                "priority": 3,
                "not_whitelisted": "drop-me",
            },
            "task-direct",
            "t2v",
            "/opt/Wan2GP",
        )

        assert isinstance(result, _Task)
        assert created["id"] == "task-direct"
        assert created["model"] == "wan22"
        assert created["prompt"] == "direct prompt"
        assert created["priority"] == 3
        assert isinstance(created["parameters"], dict)
        assert "resolution" in created["parameters"]
        assert "guidance_scale" in created["parameters"]
        assert "num_inference_steps" in created["parameters"]
        assert "seed" in created["parameters"]
        assert "negative_prompt" in created["parameters"]
        assert "use_uni3c" in created["parameters"]
        assert "uni3c_guide_video" in created["parameters"]
        assert "uni3c_strength" in created["parameters"]
        assert created["parameters"]["resolution"] == "768x432"
        assert created["parameters"]["guidance_scale"] == 6.5
        assert created["parameters"]["num_inference_steps"] == 12
        assert created["parameters"]["seed"] == 123
        assert created["parameters"]["negative_prompt"] == "blurry"
        assert created["parameters"]["use_uni3c"] is True
        assert created["parameters"]["uni3c_guide_video"] == "/tmp/guide.mp4"
        assert created["parameters"]["uni3c_strength"] == 0.75
        assert "not_whitelisted" not in created["parameters"]
