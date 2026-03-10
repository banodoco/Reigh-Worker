"""
Tests for source/core/log/tasks.py

Covers all 7 utility functions: log_task_start, log_task_complete,
log_task_error, log_model_switch, log_file_operation, log_ffmpeg_command,
log_generation_params.

Run with: python -m pytest tests/test_log_tasks.py -v
"""

from unittest.mock import patch


# All functions delegate to core.{essential, debug, success, error}.
# We patch them at the module level where they are imported in tasks.py.

_LOG_MOD = "source.core.log.tasks"


class TestLogTaskStart:
    """Tests for log_task_start."""

    @patch(f"{_LOG_MOD}.debug")
    @patch(f"{_LOG_MOD}.essential")
    def test_start_without_params(self, mock_essential, mock_debug):
        from source.core.log.tasks import log_task_start

        log_task_start("TRAVEL", "abc-123", "travel_segment")

        mock_essential.assert_called_once_with(
            "TRAVEL", "Starting travel_segment task", "abc-123"
        )
        mock_debug.assert_not_called()

    @patch(f"{_LOG_MOD}.debug")
    @patch(f"{_LOG_MOD}.essential")
    def test_start_with_params(self, mock_essential, mock_debug):
        from source.core.log.tasks import log_task_start

        log_task_start("GEN", "t1", "inpaint", width=512, height=512)

        mock_essential.assert_called_once()
        mock_debug.assert_called_once()
        # The debug message should contain the params dict
        args = mock_debug.call_args[0]
        assert "width" in args[1]
        assert "512" in args[1]

    @patch(f"{_LOG_MOD}.debug")
    @patch(f"{_LOG_MOD}.essential")
    def test_start_empty_params_dict(self, mock_essential, mock_debug):
        """Passing **{} should behave the same as no params."""
        from source.core.log.tasks import log_task_start

        log_task_start("COMP", "t2", "render", **{})
        mock_debug.assert_not_called()


class TestLogTaskComplete:
    """Tests for log_task_complete."""

    @patch(f"{_LOG_MOD}.success")
    def test_complete_minimal(self, mock_success):
        from source.core.log.tasks import log_task_complete

        log_task_complete("C", "t1", "encode")

        mock_success.assert_called_once_with("C", "encode completed", "t1")

    @patch(f"{_LOG_MOD}.success")
    def test_complete_with_output_path(self, mock_success):
        from source.core.log.tasks import log_task_complete

        log_task_complete("C", "t1", "encode", output_path="/tmp/out.mp4")

        msg = mock_success.call_args[0][1]
        assert "/tmp/out.mp4" in msg

    @patch(f"{_LOG_MOD}.success")
    def test_complete_with_duration(self, mock_success):
        from source.core.log.tasks import log_task_complete

        log_task_complete("C", "t1", "encode", duration=12.345)

        msg = mock_success.call_args[0][1]
        assert "12.3s" in msg

    @patch(f"{_LOG_MOD}.success")
    def test_complete_with_both(self, mock_success):
        from source.core.log.tasks import log_task_complete

        log_task_complete("C", "t1", "encode", output_path="/out.mp4", duration=5.0)

        msg = mock_success.call_args[0][1]
        assert "/out.mp4" in msg
        assert "5.0s" in msg


class TestLogTaskError:
    """Tests for log_task_error."""

    @patch(f"{_LOG_MOD}.error")
    def test_error_message(self, mock_error):
        from source.core.log.tasks import log_task_error

        log_task_error("Q", "t9", "upload", "disk full")

        mock_error.assert_called_once_with("Q", "upload failed: disk full", "t9")


class TestLogModelSwitch:
    """Tests for log_model_switch."""

    @patch(f"{_LOG_MOD}.essential")
    def test_switch_with_old_model(self, mock_essential):
        from source.core.log.tasks import log_model_switch

        log_model_switch("MODEL", "t2v", "i2v")

        msg = mock_essential.call_args[0][1]
        assert "t2v" in msg
        assert "i2v" in msg
        assert "\u2192" in msg  # arrow

    @patch(f"{_LOG_MOD}.essential")
    def test_initial_load_no_old_model(self, mock_essential):
        from source.core.log.tasks import log_model_switch

        log_model_switch("MODEL", None, "t2v")

        msg = mock_essential.call_args[0][1]
        assert "Model loaded: t2v" in msg

    @patch(f"{_LOG_MOD}.essential")
    def test_switch_with_duration(self, mock_essential):
        from source.core.log.tasks import log_model_switch

        log_model_switch("M", "a", "b", duration=3.14)

        msg = mock_essential.call_args[0][1]
        assert "3.1s" in msg


class TestLogFileOperation:
    """Tests for log_file_operation."""

    @patch(f"{_LOG_MOD}.debug")
    def test_operation_with_target(self, mock_debug):
        from source.core.log.tasks import log_file_operation

        log_file_operation("IO", "copy", "/a", target="/b", task_id="t1")

        msg = mock_debug.call_args[0][1]
        assert "/a" in msg and "/b" in msg

    @patch(f"{_LOG_MOD}.debug")
    def test_operation_without_target(self, mock_debug):
        from source.core.log.tasks import log_file_operation

        log_file_operation("IO", "delete", "/a")

        msg = mock_debug.call_args[0][1]
        assert "delete: /a" in msg


class TestLogFfmpegCommand:
    """Tests for log_ffmpeg_command."""

    @patch(f"{_LOG_MOD}.debug")
    def test_ffmpeg_logged(self, mock_debug):
        from source.core.log.tasks import log_ffmpeg_command

        log_ffmpeg_command("VID", "ffmpeg -i in.mp4 out.mp4", task_id="t3")

        mock_debug.assert_called_once()
        msg = mock_debug.call_args[0][1]
        assert "FFmpeg:" in msg


class TestLogGenerationParams:
    """Tests for log_generation_params."""

    @patch(f"{_LOG_MOD}.debug")
    def test_generation_params(self, mock_debug):
        from source.core.log.tasks import log_generation_params

        log_generation_params("GEN", "t5", steps=20, cfg=7.5)

        mock_debug.assert_called_once()
        msg = mock_debug.call_args[0][1]
        assert "steps" in msg
        assert "20" in msg
