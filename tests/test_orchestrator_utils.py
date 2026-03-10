"""Tests for source/utils/orchestrator_utils.py."""

from unittest import mock

import pytest

from source.utils.orchestrator_utils import (
    extract_orchestrator_parameters,
    report_orchestrator_failure,
)


class TestExtractOrchestratorParameters:
    """Tests for extract_orchestrator_parameters."""

    def test_empty_params_returns_copy(self):
        """Empty input returns a copy of the input."""
        params = {"some_key": "value"}
        result = extract_orchestrator_parameters(params)
        assert result == params
        assert result is not params  # must be a copy

    def test_no_orchestrator_details(self):
        """When there are no orchestrator_details, return params as-is."""
        params = {"prompt": "hello", "seed": 42}
        result = extract_orchestrator_parameters(params)
        assert result["prompt"] == "hello"
        assert result["seed"] == 42

    def test_extracts_from_orchestrator_details(self):
        """Parameters from orchestrator_details are extracted to top level."""
        params = {
            "orchestrator_details": {
                "prompt": "cinematic video",
                "seed": 123,
                "resolution": "1920x1080",
            }
        }
        result = extract_orchestrator_parameters(params)
        assert result["prompt"] == "cinematic video"
        assert result["seed"] == 123
        assert result["resolution"] == "1920x1080"

    def test_top_level_takes_precedence(self):
        """Top-level params are NOT overridden by orchestrator_details."""
        params = {
            "prompt": "top level prompt",
            "orchestrator_details": {
                "prompt": "orchestrator prompt",
                "seed": 42,
            }
        }
        result = extract_orchestrator_parameters(params)
        assert result["prompt"] == "top level prompt"  # top level wins
        assert result["seed"] == 42  # extracted since not at top level

    def test_empty_additional_loras_not_extracted(self):
        """Empty additional_loras dicts are skipped."""
        params = {
            "orchestrator_details": {
                "additional_loras": {},
            }
        }
        result = extract_orchestrator_parameters(params)
        assert "additional_loras" not in result or result.get("additional_loras") is None or result.get("additional_loras") == {}
        # The key should not be added because the value is empty
        # Check that it wasn't extracted (key only exists if it was in the original params)
        if "additional_loras" in result:
            # It must have come from the original params, not extraction
            assert "additional_loras" in params

    def test_nonempty_additional_loras_extracted(self):
        """Non-empty additional_loras are extracted."""
        params = {
            "orchestrator_details": {
                "additional_loras": {"lora_a": 0.5},
            }
        }
        result = extract_orchestrator_parameters(params)
        assert result["additional_loras"] == {"lora_a": 0.5}

    def test_all_extraction_map_keys(self):
        """All known extraction map keys are extracted when present."""
        all_keys = [
            "additional_loras", "prompt", "negative_prompt", "resolution",
            "video_length", "seed", "model", "num_inference_steps",
            "guidance_scale", "guidance2_scale", "guidance3_scale",
            "guidance_phases", "flow_shift", "switch_threshold",
            "switch_threshold2", "model_switch_phase", "sample_solver",
            "lora_names", "lora_multipliers", "activated_loras",
            "image_url", "in_scene", "style_reference_image",
            "style_reference_strength", "video_guide", "video_mask",
            "video_prompt_type", "control_net_weight", "amount_of_motion",
            "phase_config",
        ]
        orch_details = {k: f"value_{k}" for k in all_keys}
        # additional_loras needs to be non-empty to be extracted
        orch_details["additional_loras"] = {"test": 1.0}
        params = {"orchestrator_details": orch_details}

        result = extract_orchestrator_parameters(params)
        for key in all_keys:
            assert key in result, f"Key '{key}' was not extracted"

    def test_unknown_keys_not_extracted(self):
        """Keys not in the extraction map are not pulled to top level."""
        params = {
            "orchestrator_details": {
                "unknown_key": "should_not_appear",
                "prompt": "should appear",
            }
        }
        result = extract_orchestrator_parameters(params)
        assert result["prompt"] == "should appear"
        assert "unknown_key" not in result

    def test_empty_orchestrator_details(self):
        """Empty orchestrator_details dict is handled gracefully."""
        params = {"orchestrator_details": {}}
        result = extract_orchestrator_parameters(params)
        # Should just return a copy of params
        assert "orchestrator_details" in result

    def test_task_id_parameter_accepted(self):
        """task_id parameter is accepted (used for logging only)."""
        result = extract_orchestrator_parameters({}, task_id="task-abc")
        assert isinstance(result, dict)


class TestReportOrchestratorFailure:
    """Tests for report_orchestrator_failure."""

    @mock.patch("source.utils.orchestrator_utils.headless_logger")
    def test_no_orchestrator_ref_logs_warning(self, mock_logger):
        """When no orchestrator reference found, should log warning."""
        report_orchestrator_failure({}, "something failed")
        mock_logger.warning.assert_called()

    @mock.patch("source.utils.orchestrator_utils.headless_logger")
    def test_finds_orchestrator_task_id_ref(self, mock_logger):
        """Finds orchestrator ID via 'orchestrator_task_id_ref' key."""
        with mock.patch("source.utils.orchestrator_utils.report_orchestrator_failure") as patched:
            # Just verify the key lookup logic by calling the real function
            pass

        # Call the actual function - it will try to import db_operations
        # which is fine, we just need to verify it finds the ID
        params = {"orchestrator_task_id_ref": "orch-123"}
        # This will attempt to call db_ops.update_task_status; we mock that
        with mock.patch("source.db_operations.update_task_status") as mock_update:
            with mock.patch("source.db_operations.STATUS_FAILED", "FAILED"):
                report_orchestrator_failure(params, "test error")
                mock_update.assert_called_once_with("orch-123", "FAILED", "test error")

    @mock.patch("source.utils.orchestrator_utils.headless_logger")
    def test_truncates_long_messages(self, mock_logger):
        """Messages over 500 chars should be truncated."""
        long_msg = "x" * 1000
        params = {"orchestrator_task_id": "orch-456"}

        with mock.patch("source.db_operations.update_task_status") as mock_update:
            with mock.patch("source.db_operations.STATUS_FAILED", "FAILED"):
                report_orchestrator_failure(params, long_msg)
                call_args = mock_update.call_args[0]
                assert len(call_args[2]) == 500

    @mock.patch("source.utils.orchestrator_utils.headless_logger")
    def test_tries_multiple_key_names(self, mock_logger):
        """Tries multiple key names to find orchestrator ID."""
        # Test with 'orchestrator_id' (last in the priority list)
        params = {"orchestrator_id": "orch-789"}

        with mock.patch("source.db_operations.update_task_status") as mock_update:
            with mock.patch("source.db_operations.STATUS_FAILED", "FAILED"):
                report_orchestrator_failure(params, "error msg")
                mock_update.assert_called_once_with("orch-789", "FAILED", "error msg")
