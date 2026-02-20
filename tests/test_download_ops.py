"""Tests for source/task_handlers/queue/download_ops.py."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock
import os

import pytest


def _make_queue(**overrides):
    """Build a mock HeadlessTaskQueue with common defaults."""
    queue = MagicMock()
    queue.logger = MagicMock()
    queue.orchestrator = MagicMock()
    queue.current_model = "wan_2_1_base"
    queue.wan_dir = "/fake/wan2gp"
    queue.stats = {"model_switches": 0, "tasks_submitted": 0}
    for k, v in overrides.items():
        setattr(queue, k, v)
    return queue


def _make_task(**overrides):
    """Build a mock GenerationTask."""
    task = MagicMock()
    task.id = overrides.get("id", "task_001")
    task.prompt = overrides.get("prompt", "a cat")
    task.model = overrides.get("model", "wan_2_1_base")
    task.priority = overrides.get("priority", 1)
    task.parameters = overrides.get("parameters", {})
    return task


class TestSwitchModelImpl:
    """Tests for switch_model_impl."""

    def test_switch_occurred(self):
        """When orchestrator reports a switch, stats are updated."""
        from source.task_handlers.queue.download_ops import switch_model_impl

        queue = _make_queue()
        queue.orchestrator.load_model.return_value = True

        result = switch_model_impl(queue, "wan_2_2_vace", "Worker-0")

        assert result is True
        assert queue.stats["model_switches"] == 1
        assert queue.current_model == "wan_2_2_vace"
        queue._ensure_orchestrator.assert_called_once()

    def test_no_switch_needed(self):
        """When model is already loaded, no switch stats update."""
        from source.task_handlers.queue.download_ops import switch_model_impl

        queue = _make_queue()
        queue.orchestrator.load_model.return_value = False

        result = switch_model_impl(queue, "wan_2_1_base", "Worker-0")

        assert result is False
        assert queue.stats["model_switches"] == 0
        assert queue.current_model == "wan_2_1_base"

    def test_switch_failure_propagates(self):
        """RuntimeError from load_model propagates."""
        from source.task_handlers.queue.download_ops import switch_model_impl

        queue = _make_queue()
        queue.orchestrator.load_model.side_effect = RuntimeError("VRAM OOM")

        with pytest.raises(RuntimeError, match="VRAM OOM"):
            switch_model_impl(queue, "wan_2_2_vace", "Worker-0")


class TestConvertToWgpTaskImpl:
    """Tests for convert_to_wgp_task_impl."""

    @patch("source.task_handlers.queue.download_ops.is_debug_enabled", return_value=False)
    @patch("source.task_handlers.queue.download_ops.TaskConfig", create=True)
    def test_basic_conversion(self, mock_tc_class, mock_debug):
        """Basic task conversion produces wgp_params with prompt and model."""
        # We need to patch the import inside the function
        with patch("source.core.params.TaskConfig") as mock_tc_class:
            from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

            mock_config = MagicMock()
            mock_config.lora.has_pending_downloads.return_value = False
            mock_config.validate.return_value = []
            mock_config.to_wgp_format.return_value = {"steps": 30, "cfg": 7.0}
            mock_tc_class.from_db_task.return_value = mock_config

            queue = _make_queue()
            task = _make_task(parameters={"_source_task_type": "t2v"})

            result = convert_to_wgp_task_impl(queue, task)

            assert result["prompt"] == "a cat"
            assert result["model"] == "wan_2_1_base"
            assert result["steps"] == 30
            # Infrastructure params should be stripped
            assert "supabase_url" not in result

    @patch("source.task_handlers.queue.download_ops.is_debug_enabled", return_value=False)
    def test_lora_download_success(self, mock_debug):
        """LoRA downloads are attempted when pending."""
        with patch("source.core.params.TaskConfig") as mock_tc_class:
            from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

            mock_config = MagicMock()
            mock_config.lora.has_pending_downloads.return_value = True
            mock_config.lora.get_pending_downloads.return_value = {
                "https://example.com/lora.safetensors": 1.0
            }
            mock_config.validate.return_value = []
            mock_config.to_wgp_format.return_value = {}
            mock_tc_class.from_db_task.return_value = mock_config

            queue = _make_queue()
            task = _make_task()

            with patch("source.models.lora.lora_utils._download_lora_from_url") as mock_dl:
                mock_dl.return_value = "/loras/downloaded.safetensors"

                with patch("os.getcwd", return_value="/fake/wan2gp"), \
                     patch("os.chdir"):
                    result = convert_to_wgp_task_impl(queue, task)

                mock_dl.assert_called_once()
                mock_config.lora.mark_downloaded.assert_called_once()

    @patch("source.task_handlers.queue.download_ops.is_debug_enabled", return_value=False)
    def test_lora_download_failure_logged(self, mock_debug):
        """Failed LoRA download is logged as warning, does not crash."""
        with patch("source.core.params.TaskConfig") as mock_tc_class:
            from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

            mock_config = MagicMock()
            mock_config.lora.has_pending_downloads.return_value = True
            mock_config.lora.get_pending_downloads.return_value = {
                "https://example.com/bad.safetensors": 1.0
            }
            mock_config.validate.return_value = []
            mock_config.to_wgp_format.return_value = {}
            mock_tc_class.from_db_task.return_value = mock_config

            queue = _make_queue()
            task = _make_task()

            with patch("source.models.lora.lora_utils._download_lora_from_url") as mock_dl:
                mock_dl.return_value = None  # Failed download

                with patch("os.getcwd", return_value="/fake/wan2gp"), \
                     patch("os.chdir"):
                    result = convert_to_wgp_task_impl(queue, task)

                # Should not crash, should log warning
                queue.logger.warning.assert_called()
                mock_config.lora.mark_downloaded.assert_not_called()

    @patch("source.task_handlers.queue.download_ops.is_debug_enabled", return_value=False)
    def test_infrastructure_params_stripped(self, mock_debug):
        """Supabase params are removed from output."""
        with patch("source.core.params.TaskConfig") as mock_tc_class:
            from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

            mock_config = MagicMock()
            mock_config.lora.has_pending_downloads.return_value = False
            mock_config.validate.return_value = []
            mock_config.to_wgp_format.return_value = {
                "steps": 30,
                "supabase_url": "https://xxx.supabase.co",
                "supabase_anon_key": "secret",
                "supabase_access_token": "token",
            }
            mock_tc_class.from_db_task.return_value = mock_config

            queue = _make_queue()
            task = _make_task()

            result = convert_to_wgp_task_impl(queue, task)

            assert "supabase_url" not in result
            assert "supabase_anon_key" not in result
            assert "supabase_access_token" not in result
            assert result["steps"] == 30

    @patch("source.task_handlers.queue.download_ops.is_debug_enabled", return_value=True)
    def test_debug_mode_logs_summary(self, mock_debug):
        """In debug mode, config summary and conversion count are logged."""
        with patch("source.core.params.TaskConfig") as mock_tc_class:
            from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

            mock_config = MagicMock()
            mock_config.lora.has_pending_downloads.return_value = False
            mock_config.validate.return_value = []
            mock_config.to_wgp_format.return_value = {"a": 1, "b": 2}
            mock_tc_class.from_db_task.return_value = mock_config

            queue = _make_queue()
            task = _make_task()

            convert_to_wgp_task_impl(queue, task)

            mock_config.log_summary.assert_called_once()
            # Should log conversion info
            queue.logger.info.assert_called()

    @patch("source.task_handlers.queue.download_ops.is_debug_enabled", return_value=False)
    def test_validation_warnings_logged(self, mock_debug):
        """Validation errors are logged as warnings."""
        with patch("source.core.params.TaskConfig") as mock_tc_class:
            from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

            mock_config = MagicMock()
            mock_config.lora.has_pending_downloads.return_value = False
            mock_config.validate.return_value = ["bad width", "bad height"]
            mock_config.to_wgp_format.return_value = {}
            mock_tc_class.from_db_task.return_value = mock_config

            queue = _make_queue()
            task = _make_task()

            convert_to_wgp_task_impl(queue, task)

            # Should log validation warnings
            queue.logger.warning.assert_called()
