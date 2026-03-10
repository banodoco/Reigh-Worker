"""Tests for source/task_handlers/extract_frame.py."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest


class TestHandleExtractFrameTask:
    """Tests for handle_extract_frame_task."""

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.upload_and_get_final_output_location")
    @patch("source.task_handlers.extract_frame.save_frame_from_video")
    @patch("source.task_handlers.extract_frame.prepare_output_path_with_upload")
    @patch("source.task_handlers.extract_frame.cv2")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_success_path(
        self, mock_db_ops, mock_cv2, mock_prepare, mock_save, mock_upload, mock_logger
    ):
        """Successful frame extraction returns (True, db_location)."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.return_value = "/videos/test.mp4"
        mock_db_ops.get_abs_path_from_db_path.return_value = Path("/abs/videos/test.mp4")

        save_path = Path("/output/task1_frame_0.png")
        mock_prepare.return_value = (save_path, "db://initial")

        # Mock cv2.VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        }.get(prop, 0.0)
        mock_cv2.VideoCapture.return_value = mock_cap

        mock_save.return_value = True
        mock_upload.return_value = "db://final/task1_frame_0.png"

        params = {"input_video_task_id": "vid_123", "frame_index": 5}
        success, result = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is True
        assert result == "db://final/task1_frame_0.png"
        mock_save.assert_called_once()
        mock_cap.release.assert_called_once()

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.report_orchestrator_failure")
    def test_missing_input_video_task_id(self, mock_report, mock_logger):
        """Missing input_video_task_id returns (False, error message)."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        params = {}  # no input_video_task_id
        success, msg = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is False
        assert "Missing 'input_video_task_id'" in msg
        mock_report.assert_called_once()

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.report_orchestrator_failure")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_video_not_found_in_db(self, mock_db_ops, mock_report, mock_logger):
        """Returns failure when video task output not found in DB."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.return_value = None

        params = {"input_video_task_id": "vid_123"}
        success, msg = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is False
        assert "Could not find output location" in msg

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.report_orchestrator_failure")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_abs_path_resolution_fails(self, mock_db_ops, mock_report, mock_logger):
        """Returns failure when DB path cannot be resolved to absolute path."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.return_value = "/some/path.mp4"
        mock_db_ops.get_abs_path_from_db_path.return_value = None

        params = {"input_video_task_id": "vid_123"}
        success, msg = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is False
        assert "Could not resolve" in msg

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.report_orchestrator_failure")
    @patch("source.task_handlers.extract_frame.prepare_output_path_with_upload")
    @patch("source.task_handlers.extract_frame.cv2")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_video_cannot_be_opened(
        self, mock_db_ops, mock_cv2, mock_prepare, mock_report, mock_logger
    ):
        """Returns failure when cv2 cannot open the video file."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.return_value = "/videos/test.mp4"
        mock_db_ops.get_abs_path_from_db_path.return_value = Path("/abs/videos/test.mp4")
        mock_prepare.return_value = (Path("/output/frame.png"), "db://initial")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        params = {"input_video_task_id": "vid_123"}
        success, msg = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is False
        assert "Could not open video" in msg

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.report_orchestrator_failure")
    @patch("source.task_handlers.extract_frame.save_frame_from_video")
    @patch("source.task_handlers.extract_frame.prepare_output_path_with_upload")
    @patch("source.task_handlers.extract_frame.cv2")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_save_frame_fails(
        self, mock_db_ops, mock_cv2, mock_prepare, mock_save, mock_report, mock_logger
    ):
        """Returns failure when save_frame_from_video returns False."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.return_value = "/videos/test.mp4"
        mock_db_ops.get_abs_path_from_db_path.return_value = Path("/abs/videos/test.mp4")
        mock_prepare.return_value = (Path("/output/frame.png"), "db://initial")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640.0
        mock_cv2.VideoCapture.return_value = mock_cap

        mock_save.return_value = False

        params = {"input_video_task_id": "vid_123"}
        success, msg = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is False
        assert "save_frame_from_video utility failed" in msg

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.report_orchestrator_failure")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_exception_during_extraction(self, mock_db_ops, mock_report, mock_logger):
        """OSError during extraction is caught and returns (False, error str)."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.side_effect = OSError("disk error")

        params = {"input_video_task_id": "vid_123"}
        success, msg = handle_extract_frame_task(params, Path("/output"), "task1")

        assert success is False
        assert "disk error" in msg
        mock_report.assert_called_once()

    @patch("source.task_handlers.extract_frame.task_logger")
    @patch("source.task_handlers.extract_frame.upload_and_get_final_output_location")
    @patch("source.task_handlers.extract_frame.save_frame_from_video")
    @patch("source.task_handlers.extract_frame.prepare_output_path_with_upload")
    @patch("source.task_handlers.extract_frame.cv2")
    @patch("source.task_handlers.extract_frame.db_ops")
    def test_default_frame_index_zero(
        self, mock_db_ops, mock_cv2, mock_prepare, mock_save, mock_upload, mock_logger
    ):
        """When frame_index is not specified, defaults to 0."""
        from source.task_handlers.extract_frame import handle_extract_frame_task

        mock_db_ops.get_task_output_location_from_db.return_value = "/v.mp4"
        mock_db_ops.get_abs_path_from_db_path.return_value = Path("/abs/v.mp4")
        mock_prepare.return_value = (Path("/o/frame.png"), "db://x")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 320.0
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_save.return_value = True
        mock_upload.return_value = "db://final"

        params = {"input_video_task_id": "vid_123"}  # no frame_index
        handle_extract_frame_task(params, Path("/output"), "task1")

        # save_frame_from_video should have been called with frame_index=0
        call_kwargs = mock_save.call_args
        assert call_kwargs[1]["frame_index"] == 0 or call_kwargs.kwargs.get("frame_index") == 0


class TestHandleExtractFrameTaskDirect:
    """Low-mock behavior coverage using monkeypatch and simple stubs."""

    def test_missing_input_id_returns_descriptive_error(self):
        from source.task_handlers.extract_frame import handle_extract_frame_task

        success, msg = handle_extract_frame_task({}, Path("/tmp"), "tid-001")
        assert success is False
        assert isinstance(msg, str)
        assert "tid-001" in msg
        assert "input_video_task_id" in msg
        assert "Missing" in msg
        assert msg.startswith("Task tid-001:")
        assert msg.endswith("in payload.")

    def test_direct_success_with_monkeypatch(self, monkeypatch, tmp_path):
        import source.task_handlers.extract_frame as mod

        captured = {
            "reported": [],
            "save_kwargs": None,
            "uploaded": None,
            "prepare_args": None,
        }

        class _DBOps:
            @staticmethod
            def get_task_output_location_from_db(_task_id):
                return "db://video.mp4"

            @staticmethod
            def get_abs_path_from_db_path(_db_path):
                return tmp_path / "input.mp4"

        class _Cap:
            def __init__(self):
                self._opened = True
                self.released = False

            def isOpened(self):
                return self._opened

            def get(self, prop):
                if prop == mod.cv2.CAP_PROP_FRAME_WIDTH:
                    return 640.0
                if prop == mod.cv2.CAP_PROP_FRAME_HEIGHT:
                    return 360.0
                return 0.0

            def release(self):
                self.released = True

        class _CV2:
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4

            @staticmethod
            def VideoCapture(_path):
                return _Cap()

        def _prepare(task_id, output_filename, main_output_dir_base, task_type, custom_output_dir):
            captured["prepare_args"] = {
                "task_id": task_id,
                "output_filename": output_filename,
                "main_output_dir_base": main_output_dir_base,
                "task_type": task_type,
                "custom_output_dir": custom_output_dir,
            }
            return (
                tmp_path / output_filename,
                f"db://{task_id}/{task_type}",
            )

        def _save_frame(**kwargs):
            captured["save_kwargs"] = kwargs
            return True

        def _upload(path, initial_db):
            captured["uploaded"] = (path, initial_db)
            return f"{initial_db}/final"

        def _report(_params, msg):
            captured["reported"].append(msg)

        (tmp_path / "input.mp4").write_bytes(b"video")
        monkeypatch.setattr(mod, "db_ops", _DBOps)
        monkeypatch.setattr(mod, "cv2", _CV2)
        monkeypatch.setattr(mod, "prepare_output_path_with_upload", _prepare)
        monkeypatch.setattr(mod, "save_frame_from_video", _save_frame)
        monkeypatch.setattr(mod, "upload_and_get_final_output_location", _upload)
        monkeypatch.setattr(mod, "report_orchestrator_failure", _report)

        success, result = mod.handle_extract_frame_task(
            {"input_video_task_id": "dep-1", "frame_index": 7, "output_dir": "custom"},
            tmp_path,
            "task-xyz",
        )

        assert success is True
        assert result == "db://task-xyz/extract_frame/final"
        assert result.endswith("/final")
        assert captured["reported"] == []
        assert captured["save_kwargs"] is not None
        assert captured["prepare_args"] is not None
        assert captured["prepare_args"]["task_id"] == "task-xyz"
        assert captured["prepare_args"]["task_type"] == "extract_frame"
        assert captured["prepare_args"]["custom_output_dir"] == "custom"
        assert captured["prepare_args"]["main_output_dir_base"] == tmp_path
        assert captured["prepare_args"]["output_filename"] == "task-xyz_frame_7.png"
        assert captured["save_kwargs"]["frame_index"] == 7
        assert captured["save_kwargs"]["resolution"] == (640, 360)
        assert captured["save_kwargs"]["input_video_path"] == tmp_path / "input.mp4"
        assert captured["save_kwargs"]["input_video_path"].name == "input.mp4"
        assert captured["save_kwargs"]["resolution"][0] == 640
        assert captured["save_kwargs"]["resolution"][1] == 360
        assert Path(captured["save_kwargs"]["output_image_path"]).name == "task-xyz_frame_7.png"
        assert Path(captured["save_kwargs"]["output_image_path"]).suffix == ".png"
        assert captured["uploaded"] is not None
        assert Path(captured["uploaded"][0]).name == "task-xyz_frame_7.png"
        assert Path(captured["uploaded"][0]).parent == tmp_path
        assert captured["uploaded"][1] == "db://task-xyz/extract_frame"
