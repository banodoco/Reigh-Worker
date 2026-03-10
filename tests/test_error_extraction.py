"""Tests for source/models/wgp/error_extraction.py."""

from source.models.wgp.error_extraction import _extract_wgp_error, WGP_ERROR_PATTERNS, LOG_TAIL_MAX_CHARS


class TestExtractWgpError:
    # --- OOM errors (highest priority) ---

    def test_torch_oom_error(self):
        stderr = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "torch.OutOfMemoryError" in result

    def test_cuda_oom_error(self):
        stdout = "CUDA out of memory. Tried to allocate 512 MiB. See documentation for tips."
        result = _extract_wgp_error(stdout, "")
        assert result is not None
        assert "CUDA out of memory" in result

    def test_tried_to_allocate(self):
        stderr = "some other stuff\nTried to allocate 4.50 GiB\nmore stuff"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "Tried to allocate" in result
        assert "4.50 GiB" in result

    def test_cuda_error_out_of_memory(self):
        result = _extract_wgp_error("CUDA error: out of memory", "")
        assert result is not None
        assert "CUDA error: out of memory" in result

    # --- CUDA/GPU errors ---

    def test_cuda_generic_error(self):
        stderr = "CUDA error: device-side assert triggered\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "CUDA error" in result

    def test_cuda_runtime_error(self):
        stderr = "RuntimeError: CUDA driver version is insufficient\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "CUDA" in result

    # --- Model loading errors ---

    def test_failed_to_load_model(self):
        stdout = "Failed to load model: corrupted weights file\n"
        result = _extract_wgp_error(stdout, "")
        assert result is not None
        assert "Model loading failed" in result
        assert "corrupted weights file" in result

    def test_error_loading_model(self):
        stderr = "Error loading VACE model from /path/to/model\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "Model loading error" in result

    # --- Generic Python errors ---

    def test_runtime_error(self):
        stderr = "RuntimeError: something went wrong\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "RuntimeError" in result

    def test_value_error(self):
        stderr = "ValueError: invalid value for parameter\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "ValueError" in result

    def test_generic_exception(self):
        stderr = "Exception: unexpected failure\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "Exception" in result

    # --- Priority: OOM should match before generic RuntimeError ---

    def test_oom_priority_over_runtime_error(self):
        stderr = "RuntimeError: some stuff\ntorch.OutOfMemoryError: CUDA out of memory"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        # OOM patterns come first, so should match
        assert "torch.OutOfMemoryError" in result

    # --- No error / empty / None handling ---

    def test_empty_input_returns_none(self):
        result = _extract_wgp_error("", "")
        assert result is None

    def test_none_input_returns_none(self):
        result = _extract_wgp_error(None, None)
        assert result is None

    def test_whitespace_only_returns_none(self):
        result = _extract_wgp_error("   ", "  \n  ")
        assert result is None

    def test_normal_output_no_error(self):
        stdout = "Loading model...\nGeneration complete in 30.5s\nSaved to /output/video.mp4"
        result = _extract_wgp_error(stdout, "")
        assert result is None

    # --- Fallback: generic error indicators ---

    def test_generic_error_indicator_fallback(self):
        stderr = "File something.py line 42\n  something\nMyCustomError: whoopsie daisy"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "MyCustomError" in result

    def test_traceback_last_line_extraction(self):
        """SomeWeirdException contains 'Exception' so the regex pattern matches it."""
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"model.py\", line 10, in forward\n"
            "    result = self.compute()\n"
            "SomeWeirdException: dimension mismatch"
        )
        result = _extract_wgp_error("", stderr)
        assert result is not None
        # The generic Exception pattern matches (case insensitive)
        assert "Exception" in result
        assert "dimension mismatch" in result

    def test_traceback_fallback_no_regex_match(self):
        """When no regex pattern matches, fallback scans for 'error' in the last line."""
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"model.py\", line 10, in forward\n"
            "    result = self.compute()\n"
            "MyCustomError: totally broken"
        )
        # 'MyCustomError' doesn't match any regex pattern (no 'Exception' or known prefix),
        # but the fallback indicator search finds 'error' in it
        result = _extract_wgp_error("", stderr)
        assert result is not None
        assert "MyCustomError" in result

    def test_no_error_indicator_in_output(self):
        # Output with none of the indicator words
        stdout = "Loading model...\nProcessing frames...\nDone."
        result = _extract_wgp_error(stdout, "")
        assert result is None

    # --- Detail truncation ---

    def test_long_detail_is_truncated(self):
        long_detail = "x" * 500
        stderr = f"RuntimeError: {long_detail}\n"
        result = _extract_wgp_error("", stderr)
        assert result is not None
        # Detail is truncated to 200 chars
        assert len(result) <= len("RuntimeError: ") + 200 + 10  # some slack

    # --- Combined stdout + stderr ---

    def test_error_in_stdout_detected(self):
        result = _extract_wgp_error("ValueError: bad input\n", "")
        assert result is not None

    def test_error_in_stderr_detected(self):
        result = _extract_wgp_error("", "ValueError: bad input\n")
        assert result is not None

    def test_combined_stdout_stderr(self):
        stdout = "Loading model...\n"
        stderr = "RuntimeError: GPU not available\n"
        result = _extract_wgp_error(stdout, stderr)
        assert result is not None
        assert "RuntimeError" in result


class TestConstants:
    def test_log_tail_max_chars_is_positive(self):
        assert LOG_TAIL_MAX_CHARS > 0

    def test_error_patterns_is_nonempty(self):
        assert len(WGP_ERROR_PATTERNS) > 0

    def test_each_pattern_is_tuple(self):
        for pattern in WGP_ERROR_PATTERNS:
            assert isinstance(pattern, tuple)
            assert len(pattern) == 2
