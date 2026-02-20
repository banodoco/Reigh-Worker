"""Tests for source/models/wgp/generation_helpers.py."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestNotifyWorkerModelSwitch:
    """Tests for notify_worker_model_switch."""

    def test_no_worker_id_returns_early(self):
        """Should return immediately when WORKER_ID is not set."""
        from source.models.wgp.generation_helpers import notify_worker_model_switch

        with patch.dict(os.environ, {}, clear=True):
            # Should not raise, should not try to import httpx
            notify_worker_model_switch("old_model", "new_model")

    def test_no_supabase_key_returns_early(self):
        """Should return when WORKER_ID is set but no supabase key."""
        from source.models.wgp.generation_helpers import notify_worker_model_switch

        env = {"WORKER_ID": "worker-123"}
        with patch.dict(os.environ, env, clear=True):
            notify_worker_model_switch("old_model", "new_model")

    @patch("source.models.wgp.generation_helpers.httpx", create=True)
    def test_successful_notification(self, mock_httpx_module):
        """Should POST to edge function with correct payload on success."""
        import sys
        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.post.return_value = mock_response
        mock_httpx.HTTPError = Exception

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            from source.models.wgp.generation_helpers import notify_worker_model_switch

            env = {
                "WORKER_ID": "worker-123",
                "SUPABASE_SERVICE_ROLE_KEY": "test-key",
                "SUPABASE_URL": "https://test.supabase.co",
            }
            with patch.dict(os.environ, env, clear=True):
                notify_worker_model_switch("t2v", "vace_14B")

            mock_httpx.post.assert_called_once()
            call_kwargs = mock_httpx.post.call_args
            assert "update-worker-model" in call_kwargs[0][0]

    @patch("source.models.wgp.generation_helpers.httpx", create=True)
    def test_http_error_does_not_raise(self, mock_httpx_module):
        """HTTP errors should be caught and not propagate."""
        import sys
        mock_httpx = MagicMock()

        class FakeHTTPError(Exception):
            pass

        mock_httpx.HTTPError = FakeHTTPError
        mock_httpx.post.side_effect = FakeHTTPError("connection refused")

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            from source.models.wgp.generation_helpers import notify_worker_model_switch

            env = {
                "WORKER_ID": "worker-123",
                "SUPABASE_SERVICE_ROLE_KEY": "test-key",
            }
            with patch.dict(os.environ, env, clear=True):
                # Should not raise
                notify_worker_model_switch("t2v", "vace_14B")

    def test_httpx_import_error_does_not_raise(self):
        """Missing httpx should be caught gracefully."""
        import sys
        from source.models.wgp.generation_helpers import notify_worker_model_switch

        env = {
            "WORKER_ID": "worker-123",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        }

        # Remove httpx from sys.modules to force ImportError
        with patch.dict(os.environ, env, clear=True), \
             patch.dict(sys.modules, {"httpx": None}):
            # Should not raise
            notify_worker_model_switch("t2v", "vace_14B")


class TestVerifyWgpDirectory:
    """Tests for verify_wgp_directory."""

    def test_correct_directory_logs_debug(self):
        """Should log debug when in a Wan2GP directory."""
        from source.models.wgp.generation_helpers import verify_wgp_directory

        mock_logger = MagicMock()

        with patch("os.getcwd", return_value="/workspace/Wan2GP"):
            with patch("os.path.exists", return_value=True):
                result = verify_wgp_directory(mock_logger, "test context")

        assert result == "/workspace/Wan2GP"
        mock_logger.debug.assert_called()

    def test_wrong_directory_logs_warning(self):
        """Should log warning when not in a Wan2GP directory."""
        from source.models.wgp.generation_helpers import verify_wgp_directory

        mock_logger = MagicMock()

        with patch("os.getcwd", return_value="/home/user/random"):
            with patch("os.path.exists", return_value=True):
                result = verify_wgp_directory(mock_logger, "after generation")

        assert result == "/home/user/random"
        mock_logger.warning.assert_called()

    def test_missing_defaults_dir_logs_error(self):
        """Should log error when defaults/ dir is not accessible."""
        from source.models.wgp.generation_helpers import verify_wgp_directory

        mock_logger = MagicMock()

        with patch("os.getcwd", return_value="/workspace/Wan2GP"):
            with patch("os.path.exists", return_value=False):
                verify_wgp_directory(mock_logger, "test")

        mock_logger.error.assert_called()
        error_msg = mock_logger.error.call_args[0][0]
        assert "defaults/" in error_msg


class TestCreateVaceFixedGenerateVideo:
    """Tests for create_vace_fixed_generate_video."""

    def test_passthrough_call(self):
        """Wrapper should call original function with args and kwargs."""
        from source.models.wgp.generation_helpers import create_vace_fixed_generate_video

        mock_original = MagicMock(return_value="output.mp4")
        wrapper = create_vace_fixed_generate_video(mock_original)

        result = wrapper("arg1", key="value")
        mock_original.assert_called_once_with("arg1", key="value")
        assert result == "output.mp4"

    def test_denoise_strength_remapped(self):
        """Should rename denoise_strength to denoising_strength."""
        from source.models.wgp.generation_helpers import create_vace_fixed_generate_video

        mock_original = MagicMock(return_value="output.mp4")
        wrapper = create_vace_fixed_generate_video(mock_original)

        wrapper(prompt="test", denoise_strength=0.7)

        call_kwargs = mock_original.call_args[1]
        assert "denoising_strength" in call_kwargs
        assert call_kwargs["denoising_strength"] == 0.7
        assert "denoise_strength" not in call_kwargs

    def test_denoising_strength_not_overwritten(self):
        """Without denoise_strength kwarg, denoising_strength should not appear."""
        from source.models.wgp.generation_helpers import create_vace_fixed_generate_video

        mock_original = MagicMock()
        wrapper = create_vace_fixed_generate_video(mock_original)

        wrapper(prompt="test", steps=20)

        call_kwargs = mock_original.call_args[1]
        assert "denoising_strength" not in call_kwargs


class TestLoadImage:
    """Tests for load_image."""

    def test_none_path_returns_none(self):
        """Should return None for empty/None path."""
        from source.models.wgp.generation_helpers import load_image

        mock_orch = MagicMock()
        assert load_image(mock_orch, None) is None
        assert load_image(mock_orch, "") is None

    @patch("source.models.wgp.generation_helpers.resolve_media_path")
    def test_load_rgb_image(self, mock_resolve):
        """Should load and convert image to RGB."""
        from source.models.wgp.generation_helpers import load_image

        mock_resolve.return_value = "/resolved/path.png"
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        mock_orch = MagicMock()

        mock_pil = MagicMock()
        mock_pil.open.return_value = mock_img
        with patch.dict("sys.modules", {"PIL": MagicMock(Image=mock_pil), "PIL.Image": mock_pil}):
            # Need to call the function fresh so the local import picks up our mock
            result = load_image(mock_orch, "/some/path.png", mask=False)

        mock_img.convert.assert_called_with("RGB")

    @patch("source.models.wgp.generation_helpers.resolve_media_path")
    def test_load_mask_image(self, mock_resolve):
        """Should load and convert image to grayscale for masks."""
        from source.models.wgp.generation_helpers import load_image

        mock_resolve.return_value = "/resolved/mask.png"
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        mock_orch = MagicMock()

        mock_pil = MagicMock()
        mock_pil.open.return_value = mock_img
        with patch.dict("sys.modules", {"PIL": MagicMock(Image=mock_pil), "PIL.Image": mock_pil}):
            result = load_image(mock_orch, "/some/mask.png", mask=True)

        mock_img.convert.assert_called_with("L")

    @patch("source.models.wgp.generation_helpers.resolve_media_path")
    def test_load_image_os_error_returns_none(self, mock_resolve):
        """Should return None and not raise on file errors."""
        from source.models.wgp.generation_helpers import load_image

        mock_resolve.side_effect = OSError("file not found")
        mock_orch = MagicMock()
        result = load_image(mock_orch, "/bad/path.png")
        assert result is None


class TestResolveMediaPath:
    """Tests for resolve_media_path."""

    def test_none_path_returns_none(self):
        """Should return None for None path."""
        from source.models.wgp.generation_helpers import resolve_media_path

        mock_orch = MagicMock()
        assert resolve_media_path(mock_orch, None) is None

    def test_empty_path_returns_empty(self):
        """Should return empty string for empty path."""
        from source.models.wgp.generation_helpers import resolve_media_path

        mock_orch = MagicMock()
        assert resolve_media_path(mock_orch, "") == ""

    def test_absolute_existing_path(self, tmp_path):
        """Should return resolved path if it exists."""
        from source.models.wgp.generation_helpers import resolve_media_path

        mock_orch = MagicMock()

        # Create a real file so Path(path).exists() returns True
        img_file = tmp_path / "img.png"
        img_file.write_bytes(b"fake image")

        result = resolve_media_path(mock_orch, str(img_file))

        assert "img.png" in result


class TestModelTypeCheckers:
    """Tests for is_vace, is_model_vace, is_flux, is_t2v, is_qwen."""

    def test_is_vace_delegates_to_orchestrator(self):
        from source.models.wgp.generation_helpers import is_vace

        mock_orch = MagicMock()
        mock_orch.current_model = "vace_14B"
        mock_orch._test_vace_module.return_value = True

        assert is_vace(mock_orch) is True
        mock_orch._test_vace_module.assert_called_with("vace_14B")

    def test_is_model_vace_delegates(self):
        from source.models.wgp.generation_helpers import is_model_vace

        mock_orch = MagicMock()
        mock_orch._test_vace_module.return_value = False

        assert is_model_vace(mock_orch, "t2v") is False
        mock_orch._test_vace_module.assert_called_with("t2v")

    def test_is_flux(self):
        from source.models.wgp.generation_helpers import is_flux

        mock_orch = MagicMock()
        mock_orch.current_model = "flux_dev"
        mock_orch._get_base_model_type.return_value = "flux"

        assert is_flux(mock_orch) is True

    def test_is_t2v_standard(self):
        from source.models.wgp.generation_helpers import is_t2v

        mock_orch = MagicMock()
        mock_orch.current_model = "t2v_model"
        mock_orch._get_base_model_type.return_value = "t2v"

        assert is_t2v(mock_orch) is True

    def test_is_t2v_hunyuan(self):
        from source.models.wgp.generation_helpers import is_t2v

        mock_orch = MagicMock()
        mock_orch.current_model = "hunyuan_model"
        mock_orch._get_base_model_type.return_value = "hunyuan"

        assert is_t2v(mock_orch) is True

    def test_is_qwen_by_family(self):
        from source.models.wgp.generation_helpers import is_qwen

        mock_orch = MagicMock()
        mock_orch.current_model = "qwen_model"
        mock_orch._get_model_family.return_value = "qwen"

        assert is_qwen(mock_orch) is True

    def test_is_qwen_by_base_type_fallback(self):
        from source.models.wgp.generation_helpers import is_qwen

        mock_orch = MagicMock()
        mock_orch.current_model = "qwen2vl"
        mock_orch._get_model_family.side_effect = ValueError("unknown")
        mock_orch._get_base_model_type.return_value = "qwen2vl"

        assert is_qwen(mock_orch) is True

    def test_is_qwen_false(self):
        from source.models.wgp.generation_helpers import is_qwen

        mock_orch = MagicMock()
        mock_orch.current_model = "t2v_model"
        mock_orch._get_model_family.return_value = "wan"
        mock_orch._get_base_model_type.return_value = "t2v"

        assert is_qwen(mock_orch) is False
