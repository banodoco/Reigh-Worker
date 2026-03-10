"""Tests for source/models/comfy/comfy_utils.py."""

import signal
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

import httpx


class TestComfyUIManager:
    """Tests for ComfyUIManager."""

    def test_init_defaults(self):
        """Should initialize with default comfy_path and port."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager()
        assert manager.process is None
        assert manager.port == 8188

    def test_init_custom_path_and_port(self):
        """Should accept custom path and port."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager(comfy_path="/custom/path", port=9999)
        assert manager.comfy_path == "/custom/path"
        assert manager.port == 9999

    def test_start_already_running(self):
        """Should warn and return True if already running."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager()
        manager.process = MagicMock()
        result = manager.start()
        assert result is True

    def test_start_missing_main_py_raises(self):
        """Should raise FileNotFoundError if ComfyUI main.py doesn't exist."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager(comfy_path="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="ComfyUI not found"):
            manager.start()

    @patch("source.models.comfy.comfy_utils.subprocess.Popen")
    @patch("source.models.comfy.comfy_utils.Path")
    def test_start_success(self, mock_path_cls, mock_popen):
        """Should start subprocess and return True."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        # Make main.py appear to exist
        mock_main_py = MagicMock()
        mock_main_py.exists.return_value = True
        mock_path_cls.return_value.__truediv__ = MagicMock(return_value=mock_main_py)

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        manager = ComfyUIManager(comfy_path="/workspace/ComfyUI")
        result = manager.start()

        assert result is True
        assert manager.process is mock_process
        mock_popen.assert_called_once()

    @patch("source.models.comfy.comfy_utils.os.killpg")
    @patch("source.models.comfy.comfy_utils.os.getpgid")
    def test_stop_success(self, mock_getpgid, mock_killpg):
        """Should send SIGTERM and wait for process."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        mock_getpgid.return_value = 100
        mock_process = MagicMock()
        mock_process.pid = 12345

        manager = ComfyUIManager()
        manager.process = mock_process
        manager.stop()

        mock_killpg.assert_called_with(100, signal.SIGTERM)
        mock_process.wait.assert_called_with(timeout=10)
        assert manager.process is None

    @patch("source.models.comfy.comfy_utils.os.killpg")
    @patch("source.models.comfy.comfy_utils.os.getpgid")
    def test_stop_timeout_sends_sigkill(self, mock_getpgid, mock_killpg):
        """Should send SIGKILL if SIGTERM times out."""
        import subprocess
        from source.models.comfy.comfy_utils import ComfyUIManager

        mock_getpgid.return_value = 100
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="comfy", timeout=10)

        manager = ComfyUIManager()
        manager.process = mock_process
        manager.stop()

        # Should have been called twice: once with SIGTERM, once with SIGKILL
        calls = mock_killpg.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == signal.SIGTERM
        assert calls[1][0][1] == signal.SIGKILL
        assert manager.process is None

    def test_stop_no_process(self):
        """Should do nothing when no process is running."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager()
        manager.stop()  # Should not raise
        assert manager.process is None

    @pytest.mark.asyncio
    async def test_wait_for_ready_success(self):
        """Should return True when ComfyUI responds with 200."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager()
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        with patch("source.models.comfy.comfy_utils.time.sleep"):
            result = await manager.wait_for_ready(mock_client, timeout=5)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_ready_timeout(self):
        """Should return False when ComfyUI never becomes ready."""
        from source.models.comfy.comfy_utils import ComfyUIManager

        manager = ComfyUIManager()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("refused")

        with patch("source.models.comfy.comfy_utils.time.sleep"), \
             patch("source.models.comfy.comfy_utils.time.time") as mock_time:
            # Simulate timeout: first call returns 0, second call returns 200 (> timeout=1)
            mock_time.side_effect = [0, 0, 200]
            result = await manager.wait_for_ready(mock_client, timeout=1)

        assert result is False


class TestComfyUIClient:
    """Tests for ComfyUIClient."""

    def test_init_defaults(self):
        """Should initialize with default host and port."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        assert client.host == "localhost"
        assert client.port == 8188
        assert client.base_url == "http://localhost:8188"

    def test_init_custom(self):
        """Should accept custom host and port."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient(host="10.0.0.1", port=9999)
        assert client.base_url == "http://10.0.0.1:9999"

    @pytest.mark.asyncio
    async def test_upload_video(self):
        """Should POST video bytes and return filename."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "uploaded_video.mp4"}
        mock_response.raise_for_status = MagicMock()
        mock_http.post.return_value = mock_response

        result = await client.upload_video(mock_http, b"fakevideo", "input.mp4")

        assert result == "uploaded_video.mp4"
        mock_http.post.assert_called_once()
        call_kwargs = mock_http.post.call_args
        assert "/upload/image" in call_kwargs[0][0]

    @pytest.mark.asyncio
    async def test_upload_video_fallback_filename(self):
        """Should return original filename if server doesn't return name."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_http.post.return_value = mock_response

        result = await client.upload_video(mock_http, b"data", "original.mp4")
        assert result == "original.mp4"

    @pytest.mark.asyncio
    async def test_queue_workflow(self):
        """Should POST workflow and return prompt_id."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.json.return_value = {"prompt_id": "abc-123"}
        mock_response.raise_for_status = MagicMock()
        mock_http.post.return_value = mock_response

        workflow = {"3": {"class_type": "KSampler"}}
        result = await client.queue_workflow(mock_http, workflow)

        assert result == "abc-123"
        call_kwargs = mock_http.post.call_args
        assert "/prompt" in call_kwargs[0][0]

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self):
        """Should return history when workflow completes."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        history_data = {"status": {"completed": True}, "outputs": {}}
        mock_response = MagicMock()
        mock_response.json.return_value = {"prompt-123": history_data}
        mock_response.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_response

        with patch("source.models.comfy.comfy_utils.time.sleep"):
            result = await client.wait_for_completion(mock_http, "prompt-123", timeout=10)

        assert result == history_data

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        """Should raise TimeoutError when workflow doesn't complete."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        # Never completed
        mock_response = MagicMock()
        mock_response.json.return_value = {"prompt-123": {"status": {}}}
        mock_response.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_response

        with patch("source.models.comfy.comfy_utils.time.sleep"), \
             patch("source.models.comfy.comfy_utils.time.time") as mock_time:
            mock_time.side_effect = [0, 0, 700]  # Exceeds timeout

            with pytest.raises(TimeoutError, match="did not complete"):
                await client.wait_for_completion(mock_http, "prompt-123", timeout=600)

    @pytest.mark.asyncio
    async def test_download_output_videos(self):
        """Should download video outputs from completed workflow."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.content = b"video_bytes_here"
        mock_response.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_response

        history = {
            "outputs": {
                "node_7": {
                    "videos": [
                        {"filename": "output.mp4", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }

        result = await client.download_output(mock_http, history)

        assert len(result) == 1
        assert result[0]["filename"] == "output.mp4"
        assert result[0]["content"] == b"video_bytes_here"

    @pytest.mark.asyncio
    async def test_download_output_gifs(self):
        """Should also download from 'gifs' key (VHS_VideoCombine)."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        mock_response = MagicMock()
        mock_response.content = b"gif_content"
        mock_response.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_response

        history = {
            "outputs": {
                "node_5": {
                    "gifs": [
                        {"filename": "output.gif"}
                    ]
                }
            }
        }

        result = await client.download_output(mock_http, history)

        assert len(result) == 1
        assert result[0]["filename"] == "output.gif"

    @pytest.mark.asyncio
    async def test_download_output_empty(self):
        """Should return empty list when no outputs."""
        from source.models.comfy.comfy_utils import ComfyUIClient

        client = ComfyUIClient()
        mock_http = AsyncMock(spec=httpx.AsyncClient)

        history = {"outputs": {}}
        result = await client.download_output(mock_http, history)

        assert result == []
