"""Tests for source/task_handlers/join/shared.py."""

from unittest import mock

import pytest

from source.core.params.task_result import TaskOutcome
from source.task_handlers.join.shared import (
    _check_existing_join_tasks,
    _extract_join_settings_from_payload,
)


class TestExtractJoinSettingsFromPayload:
    """Tests for _extract_join_settings_from_payload."""

    def test_default_values_on_empty_payload(self):
        """Empty payload should return sensible defaults."""
        result = _extract_join_settings_from_payload({})
        assert result["context_frame_count"] == 8
        assert result["gap_frame_count"] == 53
        assert result["replace_mode"] is False
        assert result["prompt"] == "smooth transition"
        assert result["negative_prompt"] == ""
        assert result["model"] == "wan_2_2_vace_lightning_baseline_2_2_2"
        assert result["seed"] == -1
        assert result["additional_loras"] == {}
        assert result["keep_bridging_images"] is False
        assert result["use_input_video_fps"] is False

    def test_explicit_values_override_defaults(self):
        """Payload values should take precedence over defaults."""
        payload = {
            "context_frame_count": 16,
            "gap_frame_count": 100,
            "replace_mode": True,
            "prompt": "cinematic dissolve",
            "negative_prompt": "blurry",
            "model": "custom_model",
            "seed": 42,
        }
        result = _extract_join_settings_from_payload(payload)
        assert result["context_frame_count"] == 16
        assert result["gap_frame_count"] == 100
        assert result["replace_mode"] is True
        assert result["prompt"] == "cinematic dissolve"
        assert result["negative_prompt"] == "blurry"
        assert result["model"] == "custom_model"
        assert result["seed"] == 42

    def test_use_input_res_nullifies_resolution(self):
        """When use_input_video_resolution is True, resolution should be None."""
        payload = {
            "use_input_video_resolution": True,
            "resolution": "1920x1080",
        }
        result = _extract_join_settings_from_payload(payload)
        assert result["resolution"] is None
        assert result["use_input_video_resolution"] is True

    def test_use_input_res_false_keeps_resolution(self):
        """When use_input_video_resolution is False, resolution is passed through."""
        payload = {
            "use_input_video_resolution": False,
            "resolution": "1280x720",
        }
        result = _extract_join_settings_from_payload(payload)
        assert result["resolution"] == "1280x720"
        assert result["use_input_video_resolution"] is False

    def test_optional_fields_pass_through(self):
        """Optional fields like fps, phase_config, inference steps."""
        payload = {
            "fps": 24,
            "phase_config": {"phases": [1, 2, 3]},
            "num_inference_steps": 50,
            "guidance_scale": 10.0,
            "vid2vid_init_strength": 0.8,
            "audio_url": "https://example.com/audio.mp3",
        }
        result = _extract_join_settings_from_payload(payload)
        assert result["fps"] == 24
        assert result["phase_config"] == {"phases": [1, 2, 3]}
        assert result["num_inference_steps"] == 50
        assert result["guidance_scale"] == 10.0
        assert result["vid2vid_init_strength"] == 0.8
        assert result["audio_url"] == "https://example.com/audio.mp3"

    def test_none_optional_fields(self):
        """Absent optional fields return None via .get()."""
        result = _extract_join_settings_from_payload({})
        assert result["aspect_ratio"] is None
        assert result["fps"] is None
        assert result["phase_config"] is None
        assert result["num_inference_steps"] is None
        assert result["guidance_scale"] is None
        assert result["vid2vid_init_strength"] is None
        assert result["audio_url"] is None

    def test_lora_params(self):
        """LoRA parameters are extracted."""
        payload = {
            "additional_loras": {"lora1": 0.8, "lora2": 1.0},
        }
        result = _extract_join_settings_from_payload(payload)
        assert result["additional_loras"] == {"lora1": 0.8, "lora2": 1.0}

    def test_result_keys_count(self):
        """Verify the result contains exactly the expected number of keys."""
        result = _extract_join_settings_from_payload({})
        expected_keys = {
            "context_frame_count", "gap_frame_count", "replace_mode",
            "prompt", "negative_prompt", "model", "aspect_ratio",
            "resolution", "use_input_video_resolution", "fps",
            "use_input_video_fps", "phase_config", "num_inference_steps",
            "guidance_scale", "seed", "additional_loras",
            "keep_bridging_images", "vid2vid_init_strength", "audio_url",
        }
        assert set(result.keys()) == expected_keys


class TestCheckExistingJoinTasks:
    """Tests for _check_existing_join_tasks returning TaskResult."""

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_no_existing_tasks_returns_none(self, mock_db_ops):
        mock_db_ops.get_orchestrator_child_tasks.return_value = {}
        result = _check_existing_join_tasks("orch-1", num_joins=3)
        assert result is None

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_parallel_all_complete_returns_orchestrator_complete(self, mock_db_ops):
        mock_db_ops.get_orchestrator_child_tasks.return_value = {
            "join_clips_segment": [
                {"status": "Complete", "output_location": "/j1.mp4"},
                {"status": "Complete", "output_location": "/j2.mp4"},
            ],
            "join_final_stitch": [
                {"status": "Complete", "output_location": "/final.mp4"},
            ],
        }
        result = _check_existing_join_tasks("orch-1", num_joins=2)
        assert result is not None
        assert result.outcome == TaskOutcome.ORCHESTRATOR_COMPLETE
        assert result.output_path == "/final.mp4"

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_parallel_in_progress_returns_orchestrating(self, mock_db_ops):
        mock_db_ops.get_orchestrator_child_tasks.return_value = {
            "join_clips_segment": [
                {"status": "Complete", "output_location": "/j1.mp4"},
                {"status": "In Progress"},
            ],
            "join_final_stitch": [
                {"status": "Queued"},
            ],
        }
        result = _check_existing_join_tasks("orch-1", num_joins=2)
        assert result is not None
        assert result.outcome == TaskOutcome.ORCHESTRATING
        ok, msg = result  # backward compat
        assert ok is True

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_parallel_failed_returns_failed(self, mock_db_ops):
        mock_db_ops.get_orchestrator_child_tasks.return_value = {
            "join_clips_segment": [
                {"status": "Complete", "output_location": "/j1.mp4"},
                {"status": "Failed"},
            ],
            "join_final_stitch": [
                {"status": "Queued"},
            ],
        }
        result = _check_existing_join_tasks("orch-1", num_joins=2)
        assert result is not None
        assert result.outcome == TaskOutcome.FAILED
        ok, msg = result
        assert ok is False
        assert "failed" in msg.lower()

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_chain_all_complete_returns_orchestrator_complete(self, mock_db_ops):
        mock_db_ops.get_orchestrator_child_tasks.return_value = {
            "join_clips_segment": [
                {"status": "Complete", "output_location": "/j1.mp4",
                 "task_params": {"join_index": 0}},
                {"status": "Complete", "output_location": "/j2.mp4",
                 "task_params": {"join_index": 1, "thumbnail_url": "http://thumb.jpg"}},
            ],
            "join_final_stitch": [],
        }
        result = _check_existing_join_tasks("orch-1", num_joins=2)
        assert result is not None
        assert result.outcome == TaskOutcome.ORCHESTRATOR_COMPLETE
        assert result.output_path == "/j2.mp4"
        assert result.thumbnail_url == "http://thumb.jpg"

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_chain_in_progress_returns_orchestrating(self, mock_db_ops):
        mock_db_ops.get_orchestrator_child_tasks.return_value = {
            "join_clips_segment": [
                {"status": "Complete", "output_location": "/j1.mp4",
                 "task_params": {"join_index": 0}},
                {"status": "In Progress",
                 "task_params": {"join_index": 1}},
            ],
            "join_final_stitch": [],
        }
        result = _check_existing_join_tasks("orch-1", num_joins=2)
        assert result is not None
        assert result.outcome == TaskOutcome.ORCHESTRATING

    @mock.patch("source.task_handlers.join.shared.db_ops")
    def test_not_enough_joins_returns_none(self, mock_db_ops):
        """If fewer joins exist than expected, return None to proceed with creation."""
        mock_db_ops.get_orchestrator_child_tasks.return_value = {
            "join_clips_segment": [
                {"status": "Complete", "output_location": "/j1.mp4"},
            ],
            "join_final_stitch": [],
        }
        result = _check_existing_join_tasks("orch-1", num_joins=3)
        assert result is None
