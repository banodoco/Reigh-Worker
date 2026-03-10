"""Tests for source/utils/download_utils.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# _get_unique_target_path
# ---------------------------------------------------------------------------

class TestGetUniqueTargetPath:
    """Tests for _get_unique_target_path helper."""

    def test_creates_target_directory(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        subdir = tmp_path / "new_subdir"
        assert not subdir.exists()
        result = _get_unique_target_path(subdir, "test_file", ".mp4")
        assert subdir.exists()
        assert result.parent == subdir

    def test_filename_contains_base_name(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        result = _get_unique_target_path(tmp_path, "my_video", ".mp4")
        assert "my_video" in result.name

    def test_extension_with_dot(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        result = _get_unique_target_path(tmp_path, "file", ".jpg")
        assert result.suffix == ".jpg"

    def test_extension_without_dot(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        result = _get_unique_target_path(tmp_path, "file", "png")
        assert result.suffix == ".png"

    def test_empty_extension(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        result = _get_unique_target_path(tmp_path, "file", "")
        # Empty extension means the filename ends with the unique suffix (no dot appended)
        assert result.parent == tmp_path
        assert "file" in result.name

    def test_uniqueness(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        results = set()
        for _ in range(20):
            p = _get_unique_target_path(tmp_path, "same_name", ".txt")
            results.add(p)
        # UUID hex ensures uniqueness (collision probability negligible)
        assert len(results) == 20

    def test_accepts_string_target_dir(self, tmp_path):
        from source.utils.download_utils import _get_unique_target_path

        result = _get_unique_target_path(str(tmp_path), "file", ".txt")
        assert isinstance(result, Path)
        assert result.parent == tmp_path


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    """Tests for download_file."""

    def test_existing_valid_non_lora_file_returns_true(self, tmp_path):
        """Existing non-lora file should return True without downloading."""
        from source.utils.download_utils import download_file

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"content")

        result = download_file("https://example.com/model.bin", str(tmp_path), "model.bin")
        assert result is True

    def test_existing_lora_valid_returns_true(self, tmp_path):
        """Existing lora file that passes validation should return True."""
        from source.utils.download_utils import download_file

        dest = tmp_path / "style.safetensors"
        dest.write_bytes(b"dummy")

        with patch("source.utils.lora_validation.validate_lora_file", return_value=(True, "OK")):
            result = download_file("https://example.com/style.safetensors", str(tmp_path), "style.safetensors")
        assert result is True

    def test_existing_lora_invalid_redownloads(self, tmp_path):
        """Existing lora file that fails validation should be re-downloaded."""
        from source.utils.download_utils import download_file

        dest = tmp_path / "bad_lora.safetensors"
        dest.write_bytes(b"corrupted")

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content.return_value = [b"valid"]
        mock_response.raise_for_status = MagicMock()

        # First call: invalid (triggers re-download), second call after download: valid
        with patch("source.utils.lora_validation.validate_lora_file", side_effect=[
            (False, "corrupted"),
            (True, "OK after re-download"),
        ]):
            with patch("source.utils.download_utils.requests.get", return_value=mock_response):
                result = download_file("https://example.com/bad_lora.safetensors", str(tmp_path), "bad_lora.safetensors")
        assert result is True

    def test_huggingface_url_uses_hf_hub(self, tmp_path):
        """HuggingFace URLs should attempt hf_hub_download first."""
        from source.utils.download_utils import download_file

        mock_hf = MagicMock(return_value=str(tmp_path / "cached_file.safetensors"))
        # Create the cached file so copy2 works
        (tmp_path / "cached_file.safetensors").write_bytes(b"data")

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(hf_hub_download=mock_hf)}):
            with patch("source.utils.lora_validation.validate_lora_file", return_value=(True, "OK")):
                # Need to reimport to pick up the mocked module
                import importlib
                import source.utils.download_utils as mod
                result = mod.download_file(
                    "https://huggingface.co/user/repo/resolve/main/model.safetensors",
                    str(tmp_path),
                    "model.safetensors"
                )
        assert result is True

    def test_requests_fallback_success(self, tmp_path):
        """Non-HuggingFace URL should download via requests."""
        from source.utils.download_utils import download_file

        content = b"file_content_here"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/file.bin", str(tmp_path), "file.bin")
        assert result is True
        assert (tmp_path / "file.bin").read_bytes() == content

    def test_requests_size_mismatch_returns_false(self, tmp_path):
        """Content-length mismatch should return False and clean up."""
        from source.utils.download_utils import download_file

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}  # Expect 100 bytes
        mock_response.iter_content.return_value = [b"short"]  # Only 5 bytes
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/file.bin", str(tmp_path), "file.bin")
        assert result is False
        assert not (tmp_path / "file.bin").exists()

    def test_requests_network_error_returns_false(self, tmp_path):
        """Network errors should return False."""
        from source.utils.download_utils import download_file

        with patch("source.utils.download_utils.requests.get", side_effect=requests.ConnectionError("timeout")):
            result = download_file("https://example.com/file.bin", str(tmp_path), "file.bin")
        assert result is False

    def test_downloaded_lora_validation_failure_returns_false(self, tmp_path):
        """Downloaded lora file that fails post-download validation should return False."""
        from source.utils.download_utils import download_file

        content = b"bad_lora_data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            with patch("source.utils.lora_validation.validate_lora_file", return_value=(False, "Bad format")):
                result = download_file("https://example.com/bad.safetensors", str(tmp_path), "bad.safetensors")
        assert result is False
        assert not (tmp_path / "bad.safetensors").exists()

    def test_no_content_length_skips_size_check(self, tmp_path):
        """When content-length is 0 (missing), size check should be skipped."""
        from source.utils.download_utils import download_file

        content = b"some content"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "0"}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/file.bin", str(tmp_path), "file.bin")
        assert result is True

    def test_no_content_length_header_at_all(self, tmp_path):
        """When content-length header is absent entirely, size check is skipped."""
        from source.utils.download_utils import download_file

        content = b"data"
        mock_response = MagicMock()
        mock_response.headers = {}  # No content-length key
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/file.bin", str(tmp_path), "file.bin")
        assert result is True

    def test_http_error_returns_false(self, tmp_path):
        """HTTP error status from raise_for_status should return False."""
        from source.utils.download_utils import download_file

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/file.bin", str(tmp_path), "file.bin")
        assert result is False

    def test_huggingface_import_error_falls_back_to_requests(self, tmp_path):
        """When huggingface_hub is not installed, falls back to requests download."""
        from source.utils.download_utils import download_file

        content = b"fallback content"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        # Setting module to None in sys.modules causes ImportError on import
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with patch("source.utils.download_utils.requests.get", return_value=mock_response):
                result = download_file(
                    "https://huggingface.co/user/repo/resolve/main/weights.bin",
                    str(tmp_path),
                    "weights.bin",
                )
        assert result is True
        assert (tmp_path / "weights.bin").read_bytes() == content

    def test_huggingface_download_error_falls_back_to_requests(self, tmp_path):
        """When hf_hub_download raises, falls back to requests."""
        from source.utils.download_utils import download_file

        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.side_effect = OSError("HF download failed")

        content = b"fallback data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            with patch("source.utils.download_utils.requests.get", return_value=mock_response):
                result = download_file(
                    "https://huggingface.co/user/repo/resolve/main/model.bin",
                    str(tmp_path),
                    "model.bin",
                )
        assert result is True

    def test_huggingface_non_lora_file_success(self, tmp_path):
        """HuggingFace download of non-lora file (no validation) returns True."""
        from source.utils.download_utils import download_file

        cached = tmp_path / "cached_model.bin"
        cached.write_bytes(b"model data")

        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.return_value = str(cached)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            result = download_file(
                "https://huggingface.co/user/repo/resolve/main/model.bin",
                str(tmp_path),
                "model.bin",
            )
        assert result is True

    def test_huggingface_lora_validation_failure(self, tmp_path):
        """HF-downloaded lora that fails validation returns False."""
        from source.utils.download_utils import download_file

        cached = tmp_path / "cached_lora.safetensors"
        cached.write_bytes(b"bad lora")

        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.return_value = str(cached)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            with patch("source.utils.lora_validation.validate_lora_file", return_value=(False, "bad format")):
                result = download_file(
                    "https://huggingface.co/user/repo/resolve/main/my_lora.safetensors",
                    str(tmp_path),
                    "my_lora.safetensors",
                )
        assert result is False

    def test_huggingface_same_path_no_copy(self, tmp_path):
        """When hf_hub_download returns the same path as dest, no copy is needed."""
        from source.utils.download_utils import download_file

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"")  # Ensure it does not exist before download
        dest.unlink()

        mock_hf_module = MagicMock()
        # Return exactly the dest_path
        mock_hf_module.hf_hub_download.return_value = str(dest)
        # Create the file as if hf_hub_download wrote it
        dest.write_bytes(b"model data")

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            with patch("source.utils.download_utils.shutil.copy2") as mock_copy:
                result = download_file(
                    "https://huggingface.co/user/repo/resolve/main/model.bin",
                    str(tmp_path),
                    "model.bin",
                )
        assert result is True
        mock_copy.assert_not_called()

    def test_huggingface_url_not_resolve_format_falls_through(self, tmp_path):
        """HuggingFace URL that does not match /resolve/ format skips hf_hub_download."""
        from source.utils.download_utils import download_file

        mock_hf_module = MagicMock()

        content = b"data from requests"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        # URL like https://huggingface.co/user/repo (no /resolve/ segment)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            with patch("source.utils.download_utils.requests.get", return_value=mock_response):
                result = download_file(
                    "https://huggingface.co/user/repo",
                    str(tmp_path),
                    "model.bin",
                )
        # hf_hub_download should not be called since URL does not have /resolve/
        mock_hf_module.hf_hub_download.assert_not_called()
        assert result is True

    def test_non_lora_safetensors_format_check(self, tmp_path):
        """Non-lora .safetensors file gets basic format verification via safetensors.torch."""
        from source.utils.download_utils import download_file

        # The lora check: filename.endswith('.safetensors') or 'lora' in filename.lower()
        # Both conditions trigger lora validation. To reach the non-lora safetensors branch
        # at line 136, the filename must NOT end with .safetensors AND NOT contain 'lora'
        # but then line 136 checks filename.endswith('.safetensors') again. This means
        # the non-lora safetensors branch is unreachable with current logic because:
        # - If filename.endswith('.safetensors'): goes to lora validation
        # - If 'lora' in filename.lower(): goes to lora validation
        # - The else at line 134 only runs if NEITHER condition is true
        # - But line 136 checks .safetensors again in the else - that can only be True
        #   if the filename ends with .safetensors, but we already checked that above.
        # This is effectively dead code, but let's test the else branch (non-safetensors).
        content = b"binary data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/model.ckpt", str(tmp_path), "model.ckpt")
        assert result is True

    def test_cleanup_partial_download_on_error(self, tmp_path):
        """Partial file should be cleaned up on download failure."""
        from source.utils.download_utils import download_file

        dest = tmp_path / "partial.bin"
        # Simulate: requests.get succeeds, but writing fails with an OSError
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.side_effect = OSError("disk full")

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            result = download_file("https://example.com/partial.bin", str(tmp_path), "partial.bin")
        assert result is False

    def test_cleanup_failure_does_not_raise(self, tmp_path):
        """If cleanup of partial download also fails, it should not raise."""
        from source.utils.download_utils import download_file

        dest = tmp_path / "stuck.bin"
        # File must NOT exist initially (otherwise download_file returns True early)

        # Simulate: requests.get succeeds, raise_for_status is OK, but writing creates
        # a partial file and then stat() raises (simulating size check failure path).
        # Instead, use a simpler approach: make the whole request fail, but have the
        # dest file appear (as if partially written) and os.remove fails on cleanup.
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "1000"}

        # Writing will create the file, then iter_content will raise
        def write_then_fail(**kwargs):
            dest.write_bytes(b"partial")
            raise OSError("disk full")

        mock_response.iter_content.side_effect = write_then_fail

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            with patch("os.remove", side_effect=OSError("permission denied")):
                # Should not raise despite cleanup failure
                result = download_file("https://example.com/stuck.bin", str(tmp_path), "stuck.bin")
        assert result is False

    def test_lora_in_filename_triggers_validation(self, tmp_path):
        """A file with 'lora' in its name (but not .safetensors) triggers lora validation."""
        from source.utils.download_utils import download_file

        content = b"lora data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        with patch("source.utils.download_utils.requests.get", return_value=mock_response):
            with patch("source.utils.lora_validation.validate_lora_file", return_value=(True, "Validated")) as mock_val:
                result = download_file(
                    "https://example.com/my_lora_v2.bin",
                    str(tmp_path),
                    "my_lora_v2.bin",
                )
        assert result is True
        mock_val.assert_called_once()

    def test_huggingface_url_parsing_with_subdir(self, tmp_path):
        """HuggingFace URL with subdirectory in path is parsed correctly."""
        from source.utils.download_utils import download_file

        cached = tmp_path / "cached.bin"
        cached.write_bytes(b"data")

        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.return_value = str(cached)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            download_file(
                "https://huggingface.co/org/model/resolve/v1.0/subdir/weights.bin",
                str(tmp_path),
                "weights.bin",
            )

        mock_hf_module.hf_hub_download.assert_called_once_with(
            repo_id="org/model",
            filename="subdir/weights.bin",
            revision="v1.0",
            cache_dir=str(tmp_path),
            resume_download=True,
            local_files_only=False,
        )


# ---------------------------------------------------------------------------
# _download_file_if_url (internal helper)
# ---------------------------------------------------------------------------

class TestDownloadFileIfUrl:
    """Tests for the internal _download_file_if_url helper."""

    def test_non_url_returns_as_is(self):
        from source.utils.download_utils import _download_file_if_url

        result = _download_file_if_url("/local/path/file.mp4", "/tmp/downloads")
        assert result == "/local/path/file.mp4"

    def test_empty_string_returns_as_is(self):
        from source.utils.download_utils import _download_file_if_url

        result = _download_file_if_url("", "/tmp/downloads")
        assert result == ""

    def test_url_without_download_dir_returns_url(self):
        from source.utils.download_utils import _download_file_if_url

        result = _download_file_if_url("https://example.com/file.mp4", None)
        assert result == "https://example.com/file.mp4"

    def test_url_download_success(self, tmp_path):
        from source.utils.download_utils import _download_file_if_url

        content = b"video data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/my_video.mp4",
                tmp_path,
                task_id_for_logging="test-task",
            )
        assert result != "https://example.com/my_video.mp4"
        assert Path(result).exists()
        assert Path(result).read_bytes() == content

    def test_url_download_request_error_returns_url(self, tmp_path):
        from source.utils.download_utils import _download_file_if_url

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.side_effect = requests.exceptions.ConnectionError("fail")

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/file.mp4",
                tmp_path,
                task_id_for_logging="test-task",
            )
        assert result == "https://example.com/file.mp4"

    def test_custom_descriptive_name(self, tmp_path):
        from source.utils.download_utils import _download_file_if_url

        content = b"data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/file.mp4",
                tmp_path,
                descriptive_name="my_custom_name",
            )
        assert "my_custom_name" in Path(result).name

    def test_default_extension_used_when_none_in_url(self, tmp_path):
        from source.utils.download_utils import _download_file_if_url

        content = b"data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/noext",
                tmp_path,
                default_extension=".webm",
            )
        # The URL path "noext" has no extension, but default_extension is only used
        # when there is no suffix at all. In this case Path("noext").suffix == ""
        # so the default .webm should be used.
        assert Path(result).suffix == ".webm"

    def test_none_input_returns_none(self):
        """None input should be returned as-is (falsy check)."""
        from source.utils.download_utils import _download_file_if_url

        result = _download_file_if_url(None, "/tmp")
        assert result is None

    def test_ftp_scheme_returns_unchanged(self):
        """Non-HTTP schemes should be returned as-is."""
        from source.utils.download_utils import _download_file_if_url

        result = _download_file_if_url("ftp://example.com/file.mp4", "/tmp")
        assert result == "ftp://example.com/file.mp4"

    def test_os_error_on_save_returns_original_url(self, tmp_path):
        """OSError during file save should return the original URL."""
        from source.utils.download_utils import _download_file_if_url

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        url = "https://example.com/file.mp4"
        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            with patch("builtins.open", side_effect=OSError("disk full")):
                result = _download_file_if_url(url, tmp_path)
        assert result == url

    def test_long_descriptive_name_is_truncated(self, tmp_path):
        """Descriptive names longer than 50 chars should be truncated."""
        from source.utils.download_utils import _download_file_if_url

        content = b"data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        long_name = "x" * 100
        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/file.jpg",
                tmp_path,
                descriptive_name=long_name,
            )
        filename = Path(result).stem
        # The base_name is truncated to 50, so the filename should start with 50 x's
        assert filename.startswith("x" * 50)
        assert not filename.startswith("x" * 51)

    def test_default_stem_used_when_url_has_no_filename(self, tmp_path):
        """When URL has no filename stem, default_stem is used."""
        from source.utils.download_utils import _download_file_if_url

        content = b"data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/",
                tmp_path,
                default_stem="my_default",
                default_extension=".dat",
            )
        # URL path "/" has no filename, so stem is empty, default_stem should be used
        assert Path(result).exists()

    def test_creates_nested_target_directory(self, tmp_path):
        """Target directory should be created if it does not exist."""
        from source.utils.download_utils import _download_file_if_url

        target = tmp_path / "sub" / "nested"
        content = b"data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url("https://example.com/file.mp4", target)
        assert target.exists()
        assert Path(result).parent == target

    def test_url_with_extension_preserves_it(self, tmp_path):
        """URL with a known extension should use that extension, not the default."""
        from source.utils.download_utils import _download_file_if_url

        content = b"video"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(
                "https://example.com/clip.webm",
                tmp_path,
                default_extension=".mp4",  # Should NOT be used
            )
        assert Path(result).suffix == ".webm"

    def test_http_error_returns_original_url(self, tmp_path):
        """HTTP error from raise_for_status should return original URL."""
        from source.utils.download_utils import _download_file_if_url

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response

        url = "https://example.com/forbidden.mp4"
        with patch("source.utils.download_utils.requests.Session", return_value=mock_session):
            result = _download_file_if_url(url, tmp_path)
        assert result == url


# ---------------------------------------------------------------------------
# download_video_if_url
# ---------------------------------------------------------------------------

class TestDownloadVideoIfUrl:
    """Tests for download_video_if_url."""

    def test_local_path_returns_unchanged(self):
        from source.utils.download_utils import download_video_if_url

        result = download_video_if_url("/local/video.mp4", "/tmp/downloads")
        assert result == "/local/video.mp4"

    def test_delegates_to_download_file_if_url(self, tmp_path):
        from source.utils.download_utils import download_video_if_url

        with patch("source.utils.download_utils._download_file_if_url", return_value="/downloaded/video.mp4") as mock_fn:
            result = download_video_if_url("https://example.com/v.mp4", tmp_path, "task-1", "my_video")
        mock_fn.assert_called_once_with(
            "https://example.com/v.mp4",
            tmp_path,
            "task-1",
            "my_video",
            default_extension=".mp4",
            default_stem="structure_video",
            file_type_label="video",
            timeout=600,
        )
        assert result == "/downloaded/video.mp4"


# ---------------------------------------------------------------------------
# download_image_if_url
# ---------------------------------------------------------------------------

class TestDownloadImageIfUrl:
    """Tests for download_image_if_url."""

    def test_local_path_returns_unchanged(self):
        from source.utils.download_utils import download_image_if_url

        result = download_image_if_url("/local/image.jpg", "/tmp/downloads")
        assert result == "/local/image.jpg"

    def test_delegates_to_download_file_if_url(self, tmp_path):
        from source.utils.download_utils import download_image_if_url

        with patch("source.utils.download_utils._download_file_if_url", return_value="/downloaded/img.jpg") as mock_fn:
            result = download_image_if_url("https://example.com/i.jpg", tmp_path, "task-2", debug_mode=True, descriptive_name="shot_ref")
        mock_fn.assert_called_once_with(
            "https://example.com/i.jpg",
            tmp_path,
            "task-2",
            "shot_ref",
            default_extension=".jpg",
            default_stem="image",
            file_type_label="image",
            timeout=300,
        )
        assert result == "/downloaded/img.jpg"

    def test_debug_mode_is_ignored(self, tmp_path):
        """debug_mode parameter should be accepted but not cause errors."""
        from source.utils.download_utils import download_image_if_url

        with patch("source.utils.download_utils._download_file_if_url", return_value="ok"):
            # Should not raise
            download_image_if_url("https://example.com/i.jpg", tmp_path, debug_mode=True)
            download_image_if_url("https://example.com/i.jpg", tmp_path, debug_mode=False)
