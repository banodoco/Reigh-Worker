"""Tests for source/models/wgp/param_resolution.py."""

from unittest import mock
import sys

import pytest

from source.models.wgp.param_resolution import resolve_parameters


class TestResolveParameters:
    """Tests for resolve_parameters."""

    def _make_mock_orchestrator(self):
        """Create a minimal mock orchestrator."""
        return mock.MagicMock()

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_system_defaults(self):
        """With no model config and no task params, returns system defaults."""
        # Make wgp.get_default_settings return None
        sys.modules["wgp"].get_default_settings.return_value = None

        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", {})

        assert result["resolution"] == "1280x720"
        assert result["video_length"] == 49
        assert result["num_inference_steps"] == 25
        assert result["guidance_scale"] == 7.5
        assert result["guidance2_scale"] == 7.5
        assert result["flow_shift"] == 7.0
        assert result["sample_solver"] == "euler"
        assert result["switch_threshold"] == 500
        assert result["seed"] == -1
        assert result["negative_prompt"] == ""
        assert result["activated_loras"] == []
        assert result["loras_multipliers"] == ""

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_model_config_overrides_defaults(self):
        """Model JSON config overrides system defaults."""
        model_defaults = {
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "flow_shift": 3.0,
        }
        sys.modules["wgp"].get_default_settings.return_value = model_defaults

        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", {})

        assert result["num_inference_steps"] == 50
        assert result["guidance_scale"] == 5.0
        assert result["flow_shift"] == 3.0
        # Unmodified defaults remain
        assert result["resolution"] == "1280x720"
        assert result["video_length"] == 49

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_task_params_override_model_config(self):
        """Task explicit params override both model config and defaults."""
        model_defaults = {
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
        }
        sys.modules["wgp"].get_default_settings.return_value = model_defaults

        task_params = {
            "num_inference_steps": 100,
            "resolution": "1920x1080",
        }

        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", task_params)

        assert result["num_inference_steps"] == 100  # task wins over model
        assert result["resolution"] == "1920x1080"   # task wins over default
        assert result["guidance_scale"] == 5.0        # model wins over default

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_none_task_params_do_not_override(self):
        """Task params with None values should NOT override existing values."""
        sys.modules["wgp"].get_default_settings.return_value = None

        task_params = {
            "resolution": None,
            "seed": None,
        }

        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", task_params)

        # Defaults should remain because None values are skipped
        assert result["resolution"] == "1280x720"
        assert result["seed"] == -1

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_prompt_excluded_from_model_config(self):
        """'prompt' from model config should NOT be applied."""
        model_defaults = {
            "prompt": "model default prompt",
            "num_inference_steps": 30,
        }
        sys.modules["wgp"].get_default_settings.return_value = model_defaults

        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", {})

        # prompt should NOT be in result from model config
        assert result.get("prompt") != "model default prompt"
        # But num_inference_steps should
        assert result["num_inference_steps"] == 30

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_task_prompt_is_applied(self):
        """But prompt from task params IS applied."""
        sys.modules["wgp"].get_default_settings.return_value = None

        task_params = {"prompt": "user specified prompt"}
        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", task_params)

        assert result["prompt"] == "user specified prompt"

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_wgp_import_failure_graceful(self):
        """If wgp import fails, still returns defaults + task overrides."""
        # Make the import inside the function raise
        sys.modules["wgp"].get_default_settings.side_effect = ValueError("mock error")

        task_params = {"seed": 42}
        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", task_params)

        # Should still have defaults
        assert result["resolution"] == "1280x720"
        # And task overrides
        assert result["seed"] == 42

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_returns_dict(self):
        """Always returns a dict."""
        sys.modules["wgp"].get_default_settings.return_value = None
        result = resolve_parameters(self._make_mock_orchestrator(), "any-model", {})
        assert isinstance(result, dict)

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_extra_task_params_passthrough(self):
        """Unknown task params are passed through to resolved result."""
        sys.modules["wgp"].get_default_settings.return_value = None

        task_params = {
            "custom_param": "custom_value",
            "another_param": 42,
        }
        result = resolve_parameters(self._make_mock_orchestrator(), "test-model", task_params)

        assert result["custom_param"] == "custom_value"
        assert result["another_param"] == 42

    @mock.patch.dict(sys.modules, {"wgp": mock.MagicMock()})
    def test_model_type_passed_to_get_default_settings(self):
        """The model_type argument is passed to wgp.get_default_settings."""
        sys.modules["wgp"].get_default_settings.return_value = None

        resolve_parameters(self._make_mock_orchestrator(), "my-special-model", {})

        sys.modules["wgp"].get_default_settings.assert_called_with("my-special-model")
