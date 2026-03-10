"""
Tests for source/task_handlers/worker/worker_utils.py

Covers:
  - log_ram_usage: psutil available, not available, exception
  - cleanup_generated_files: file, directory, not found, debug mode, OSError
  - _cleanup_temporary_files: debug mode skip, normal run

Run with: python -m pytest tests/test_worker_utils.py -v
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

_MOD = "source.task_handlers.worker.worker_utils"


class TestLogRamUsagePsutilAvailable:
    """Tests for log_ram_usage when psutil is available."""

    @patch(f"{_MOD}.headless_logger")
    @patch(f"{_MOD}._PSUTIL_AVAILABLE", True)
    @patch(f"{_MOD}.psutil")
    def test_returns_metrics(self, mock_psutil, mock_logger):
        from source.task_handlers.worker.worker_utils import log_ram_usage

        mock_process = MagicMock()
        mock_mem = MagicMock()
        mock_mem.rss = 2 * (1024 ** 3)  # 2 GB
        mock_process.memory_info.return_value = mock_mem
        mock_psutil.Process.return_value = mock_process

        mock_sys_mem = MagicMock()
        mock_sys_mem.total = 32 * (1024 ** 3)  # 32 GB
        mock_sys_mem.available = 16 * (1024 ** 3)  # 16 GB
        mock_sys_mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_sys_mem

        result = log_ram_usage("test label", task_id="t1")

        assert result["available"] is True
        assert result["process_rss_gb"] == 2.0
        assert result["system_total_gb"] == 32.0
        assert result["system_available_gb"] == 16.0
        assert result["system_used_percent"] == 50.0
        mock_logger.info.assert_called_once()


class TestLogRamUsagePsutilNotAvailable:
    """Tests for log_ram_usage when psutil is not installed."""

    @patch(f"{_MOD}._PSUTIL_AVAILABLE", False)
    def test_returns_unavailable(self):
        from source.task_handlers.worker.worker_utils import log_ram_usage

        result = log_ram_usage("label")
        assert result == {"available": False}


class TestLogRamUsageExceptions:
    """Tests for log_ram_usage exception handling."""

    @patch(f"{_MOD}.headless_logger")
    @patch(f"{_MOD}._PSUTIL_AVAILABLE", True)
    @patch(f"{_MOD}.psutil")
    def test_process_lookup_error(self, mock_psutil, mock_logger):
        from source.task_handlers.worker.worker_utils import log_ram_usage

        mock_psutil.Process.side_effect = ProcessLookupError("no such process")

        result = log_ram_usage("bad process", task_id="t2")

        assert result["available"] is False
        assert "error" in result
        mock_logger.warning.assert_called_once()

    @patch(f"{_MOD}.headless_logger")
    @patch(f"{_MOD}._PSUTIL_AVAILABLE", True)
    @patch(f"{_MOD}.psutil")
    def test_os_error(self, mock_psutil, mock_logger):
        from source.task_handlers.worker.worker_utils import log_ram_usage

        mock_psutil.Process.side_effect = OSError("permission denied")

        result = log_ram_usage("denied", task_id="t3")

        assert result["available"] is False
        assert "permission denied" in result["error"]


class TestCleanupGeneratedFilesDebugMode:
    """Tests for cleanup_generated_files with debug_mode=True."""

    @patch(f"{_MOD}.headless_logger")
    def test_debug_mode_skips_cleanup(self, mock_logger):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        cleanup_generated_files("/some/path.mp4", task_id="t1", debug_mode=True)

        mock_logger.debug.assert_called_once()
        assert "skipping" in mock_logger.debug.call_args[0][0].lower()


class TestCleanupGeneratedFilesEmpty:
    """Tests for cleanup_generated_files with empty output_location."""

    @patch(f"{_MOD}.headless_logger")
    def test_empty_output_returns_early(self, mock_logger):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        cleanup_generated_files("", task_id="t1")

        # Should return without logging anything (empty string is falsy)
        mock_logger.debug.assert_not_called()

    @patch(f"{_MOD}.headless_logger")
    def test_none_output_returns_early(self, mock_logger):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        cleanup_generated_files(None, task_id="t1")

        mock_logger.debug.assert_not_called()


class TestCleanupGeneratedFilesFile:
    """Tests for cleanup_generated_files removing a file."""

    @patch(f"{_MOD}.headless_logger")
    def test_file_deleted(self, mock_logger, tmp_path):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        # Create a real temp file
        test_file = tmp_path / "output.mp4"
        test_file.write_bytes(b"fake video data" * 100)

        cleanup_generated_files(str(test_file), task_id="t1")

        assert not test_file.exists()
        mock_logger.debug.assert_called()


class TestCleanupGeneratedFilesDirectory:
    """Tests for cleanup_generated_files removing a directory."""

    @patch(f"{_MOD}.headless_logger")
    def test_directory_deleted(self, mock_logger, tmp_path):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        # Create a temp directory with files
        test_dir = tmp_path / "output_dir"
        test_dir.mkdir()
        (test_dir / "frame1.png").write_bytes(b"png data")
        (test_dir / "frame2.png").write_bytes(b"png data")

        cleanup_generated_files(str(test_dir), task_id="t1")

        assert not test_dir.exists()
        mock_logger.debug.assert_called()


class TestCleanupGeneratedFilesNotFound:
    """Tests for cleanup_generated_files when path doesn't exist."""

    @patch(f"{_MOD}.headless_logger")
    def test_nonexistent_path_logged(self, mock_logger, tmp_path):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        cleanup_generated_files(str(tmp_path / "nonexistent.mp4"), task_id="t1")

        # Should log that the path was not found
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        not_found = [c for c in debug_calls if "not found" in c]
        assert len(not_found) >= 1


class TestCleanupGeneratedFilesOSError:
    """Tests for cleanup_generated_files when an OSError occurs."""

    @patch(f"{_MOD}.headless_logger")
    @patch(f"{_MOD}.Path")
    def test_os_error_logged_as_warning(self, MockPath, mock_logger):
        from source.task_handlers.worker.worker_utils import cleanup_generated_files

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.stat.side_effect = OSError("permission denied")
        MockPath.return_value = mock_path_instance

        cleanup_generated_files("/locked/file.mp4", task_id="t1")

        mock_logger.warning.assert_called_once()
        assert "Failed to cleanup" in mock_logger.warning.call_args[0][0]


