"""GPU tests for LTX-2 multi-frame workflow (requires CUDA + ComfyUI)."""

import io
import pytest
import asyncio

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

skip_no_gpu = pytest.mark.skipif(
    not HAS_CUDA, reason="CUDA not available"
)
skip_no_pil = pytest.mark.skipif(
    not HAS_PIL, reason="Pillow not installed"
)


def _make_test_image(width=768, height=512, color=(255, 0, 0)):
    """Create a simple solid-color test image as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@skip_no_gpu
@skip_no_pil
class TestImageUpload:
    """Test image upload to ComfyUI."""

    def test_image_upload_to_comfyui(self):
        from source.models.comfy.comfy_handler import _ensure_comfy_running
        from source.models.comfy.comfy_utils import ComfyUIClient
        import httpx

        async def _upload():
            if not await _ensure_comfy_running():
                pytest.skip("ComfyUI not available")

            client_api = ComfyUIClient()
            image_bytes = _make_test_image()

            async with httpx.AsyncClient(timeout=60.0) as client:
                name = await client_api.upload_image(
                    client, image_bytes, "test_upload.png"
                )
                assert isinstance(name, str)
                assert len(name) > 0

        asyncio.run(_upload())


@skip_no_gpu
@skip_no_pil
class TestFullGeneration:
    """End-to-end multiframe generation test."""

    @pytest.mark.timeout(900)
    def test_full_multiframe_generation(self, tmp_path):
        from source.models.comfy.ltx2_multiframe_handler import handle_ltx2_multiframe_task

        # Create 4 test images (different colors for visual distinction)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        image_paths = []
        for i, color in enumerate(colors):
            path = tmp_path / f"test_img_{i}.png"
            img = Image.new("RGB", (768, 512), color)
            img.save(path)
            image_paths.append(str(path))

        params = {
            "prompt": "a colorful animation test",
            "negative_prompt": "blurry",
            "image_urls": image_paths,
            "seed": 42,
            "num_frames": 25,  # Short for faster test
            "steps": 8,
        }

        success, result = handle_ltx2_multiframe_task(
            task_params_from_db=params,
            main_output_dir_base=tmp_path,
            task_id="gpu_test_001",
        )

        assert success, f"Generation failed: {result}"
        assert result is not None
        from pathlib import Path
        assert Path(result).exists(), f"Output file not found: {result}"
        assert Path(result).stat().st_size > 0, "Output file is empty"
