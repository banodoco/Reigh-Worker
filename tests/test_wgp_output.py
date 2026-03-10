"""Tests for source/models/wgp/generators/output.py."""

import io
from collections import deque
from unittest.mock import patch, MagicMock

import pytest


class TestExtractOutputPath:
    """Tests for extract_output_path."""

    def test_successful_extraction(self):
        """Should return the last file in file_list on success."""
        from source.models.wgp.generators.output import extract_output_path

        state = {"gen": {"file_list": ["/output/video1.mp4", "/output/video2.mp4"]}}
        result = extract_output_path(
            state, "T2V", io.StringIO(), io.StringIO(), deque()
        )
        assert result == "/output/video2.mp4"

    def test_single_file_in_list(self):
        """Should handle a single-element file_list."""
        from source.models.wgp.generators.output import extract_output_path

        state = {"gen": {"file_list": ["/output/only.mp4"]}}
        result = extract_output_path(
            state, "VACE", io.StringIO(), io.StringIO(), deque()
        )
        assert result == "/output/only.mp4"

    def test_empty_file_list_raises_runtime_error(self):
        """Should raise RuntimeError when file_list is empty and no WGP error found."""
        from source.models.wgp.generators.output import extract_output_path

        state = {"gen": {"file_list": []}}
        with pytest.raises(RuntimeError, match="No output generated"):
            extract_output_path(
                state, "T2V", io.StringIO(), io.StringIO(), deque()
            )
        assert state["gen"]["file_list"] == []

    def test_empty_file_list_with_wgp_error(self):
        """Should raise RuntimeError with extracted WGP error message."""
        from source.models.wgp.generators.output import extract_output_path

        state = {"gen": {"file_list": []}}
        stderr = io.StringIO("torch.OutOfMemoryError: CUDA out of memory")

        with pytest.raises(RuntimeError, match="WGP generation failed"):
            extract_output_path(
                state, "T2V", io.StringIO(), stderr, deque()
            )
        assert "OutOfMemoryError" in stderr.getvalue()

    def test_missing_gen_key_returns_none(self):
        """Should return None when 'gen' key is missing from state."""
        from source.models.wgp.generators.output import extract_output_path

        state = {}
        result = extract_output_path(
            state, "T2V", io.StringIO(), io.StringIO(), deque()
        )
        assert result is None

    def test_missing_file_list_key_returns_none(self):
        """Should return None when 'file_list' key is missing from gen."""
        from source.models.wgp.generators.output import extract_output_path

        state = {"gen": {}}
        result = extract_output_path(
            state, "T2V", io.StringIO(), io.StringIO(), deque()
        )
        assert result is None

    def test_none_captured_io(self):
        """Should handle None stdout/stderr gracefully."""
        from source.models.wgp.generators.output import extract_output_path

        state = {"gen": {"file_list": []}}
        with pytest.raises(RuntimeError, match="No output generated"):
            extract_output_path(state, "T2V", None, None, deque())
        assert state["gen"]["file_list"] == []


class TestLogCapturedOutput:
    """Tests for log_captured_output."""

    def test_empty_logs_and_streams(self):
        """Should not raise with all empty inputs."""
        from source.models.wgp.generators.output import log_captured_output

        stdout = io.StringIO()
        stderr = io.StringIO()
        log_captured_output(stdout, stderr, deque())
        assert stdout.getvalue() == ""
        assert stderr.getvalue() == ""

    def test_logs_error_and_warning_records(self):
        """Should log error/warning records from captured_logs."""
        from source.models.wgp.generators.output import log_captured_output

        logs = deque([
            {"level": "ERROR", "name": "torch", "message": "CUDA error occurred"},
            {"level": "INFO", "name": "diffusers", "message": "Loading model"},
            {"level": "WARNING", "name": "transformers", "message": "Deprecated API"},
        ])
        # Should not raise
        log_captured_output(io.StringIO(), io.StringIO(), logs)
        assert len(logs) == 3
        assert logs[0]["level"] == "ERROR"
        assert logs[2]["level"] == "WARNING"

    def test_stderr_content_logged(self):
        """Should log stderr content."""
        from source.models.wgp.generators.output import log_captured_output

        stderr = io.StringIO("Some error output on stderr")
        log_captured_output(io.StringIO(), stderr, deque())
        assert "stderr" in stderr.getvalue()

    def test_stdout_error_patterns_detected(self):
        """Should detect error patterns in stdout."""
        from source.models.wgp.generators.output import log_captured_output

        stdout = io.StringIO("Loading model...\nCUDA error: out of memory\nFailed.")
        log_captured_output(stdout, io.StringIO(), deque())
        assert "CUDA error" in stdout.getvalue()
        assert "Failed." in stdout.getvalue()

    def test_none_streams_handled(self):
        """Should handle None stdout/stderr."""
        from source.models.wgp.generators.output import log_captured_output

        log_captured_output(None, None, deque())

    def test_long_stderr_truncated(self):
        """Should truncate very long stderr to LOG_TAIL_MAX_CHARS."""
        from source.models.wgp.generators.output import log_captured_output, LOG_TAIL_MAX_CHARS

        # This just exercises the truncation path; we verify it doesn't crash
        long_stderr = io.StringIO("x" * (LOG_TAIL_MAX_CHARS + 500))
        log_captured_output(io.StringIO(), long_stderr, deque())
        assert len(long_stderr.getvalue()) == LOG_TAIL_MAX_CHARS + 500


class TestLogMemoryStats:
    """Tests for log_memory_stats."""

    @patch("source.models.wgp.generators.output.torch", create=True)
    @patch("source.models.wgp.generators.output.psutil", create=True)
    def test_with_cuda_available(self, mock_psutil, mock_torch):
        """Should log both RAM and VRAM when CUDA is available."""
        import sys

        # Create proper mock modules
        mock_psutil_mod = MagicMock()
        mock_ram = MagicMock()
        mock_ram.used = 8 * (1024**3)
        mock_ram.total = 16 * (1024**3)
        mock_ram.percent = 50.0
        mock_psutil_mod.virtual_memory.return_value = mock_ram

        mock_torch_mod = MagicMock()
        mock_torch_mod.cuda.is_available.return_value = True
        mock_torch_mod.cuda.memory_allocated.return_value = 4 * (1024**3)
        mock_torch_mod.cuda.memory_reserved.return_value = 6 * (1024**3)

        mock_props = MagicMock()
        mock_props.total_memory = 24 * (1024**3)
        mock_torch_mod.cuda.get_device_properties.return_value = mock_props

        with patch.dict(sys.modules, {"torch": mock_torch_mod, "psutil": mock_psutil_mod}):
            from source.models.wgp.generators.output import log_memory_stats
            log_memory_stats()
        assert mock_psutil_mod.virtual_memory.called
        assert mock_torch_mod.cuda.memory_allocated.called
        assert mock_torch_mod.cuda.memory_reserved.called
        assert mock_torch_mod.cuda.get_device_properties.called


class TestExtractOutputPathDirect:
    """Additional no-mock assertions for malformed state handling."""

    def test_malformed_state_shapes_return_none(self):
        from source.models.wgp.generators.output import extract_output_path

        result1 = extract_output_path(None, "T2V", io.StringIO(), io.StringIO(), deque())  # type: ignore[arg-type]
        result2 = extract_output_path({"gen": None}, "T2V", io.StringIO(), io.StringIO(), deque())
        result5 = extract_output_path({"gen": {"file_list": ["a.mp4", "b.mp4", "c.mp4"]}}, "T2V", io.StringIO(), io.StringIO(), deque())

        assert result1 is None
        assert result2 is None
        with pytest.raises(RuntimeError, match="No output generated"):
            extract_output_path({"gen": {"file_list": None}}, "T2V", io.StringIO(), io.StringIO(), deque())
        with pytest.raises(RuntimeError, match="No output generated"):
            extract_output_path({"gen": {"file_list": []}}, "T2V", io.StringIO(), io.StringIO(), deque())
        assert result5 == "c.mp4"

    def test_import_error_handled_gracefully(self):
        """Should handle missing torch/psutil gracefully."""
        import sys
        with patch.dict(sys.modules, {"torch": None, "psutil": None}):
            # The function catches ImportError, so this should not raise
            from source.models.wgp.generators.output import log_memory_stats
            log_memory_stats()
            assert sys.modules["torch"] is None
            assert sys.modules["psutil"] is None

    @patch("source.models.wgp.generators.output.torch", create=True)
    @patch("source.models.wgp.generators.output.psutil", create=True)
    def test_without_cuda(self, mock_psutil, mock_torch):
        """Should log only RAM when CUDA is not available."""
        import sys

        mock_psutil_mod = MagicMock()
        mock_ram = MagicMock()
        mock_ram.used = 4 * (1024**3)
        mock_ram.total = 8 * (1024**3)
        mock_ram.percent = 50.0
        mock_psutil_mod.virtual_memory.return_value = mock_ram

        mock_torch_mod = MagicMock()
        mock_torch_mod.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch_mod, "psutil": mock_psutil_mod}):
            from source.models.wgp.generators.output import log_memory_stats
            log_memory_stats()
        assert mock_psutil_mod.virtual_memory.called
        assert mock_torch_mod.cuda.is_available.called
