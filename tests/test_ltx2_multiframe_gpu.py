"""
GPU integration tests for LTX-2 multi-frame guide images.

Requires CUDA-capable GPU and LTX-2 model weights to be available.
Skipped automatically when no GPU is present.

Run with:
    python -m pytest tests/test_ltx2_multiframe_gpu.py -v -s
"""

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Skip entire module if no CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _chdir_to_wan2gp():
    original_cwd = os.getcwd()
    wan_root = str(PROJECT_ROOT / "Wan2GP")
    os.chdir(wan_root)
    yield wan_root
    os.chdir(original_cwd)


@pytest.fixture()
def output_dir(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture()
def guide_images(tmp_path):
    """Create 4 small PIL guide images."""
    from PIL import Image
    images = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        img = Image.new("RGB", (768, 512), color=color)
        path = tmp_path / f"guide_{i}.png"
        img.save(str(path))
        images.append(str(path))
    return images


# ===================================================================
# GPU Tests
# ===================================================================

class TestMultiframeGenerationGPU:
    """Full GPU integration tests for multi-frame guide images."""

    def test_full_multiframe_generation(self, _chdir_to_wan2gp, output_dir, guide_images):
        """4 guide images at different frame positions → output video."""
        from headless_wgp import WanOrchestrator

        orch = WanOrchestrator(_chdir_to_wan2gp, main_output_dir=str(output_dir))
        orch.load_model("ltx2_19B")

        result = orch.generate(
            prompt="3d animated cartoon character is reading a book",
            negative_prompt="blurry, low quality",
            resolution="768x512",
            video_length=105,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=10,
            guide_images=[
                {"image": guide_images[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_images[1], "frame_idx": 40, "strength": 1.0},
                {"image": guide_images[2], "frame_idx": 80, "strength": 1.0},
                {"image": guide_images[3], "frame_idx": -1, "strength": 1.0},
            ],
        )

        assert result is not None
        assert os.path.exists(result)
        # Output should be a video file
        assert result.endswith((".mp4", ".webm", ".gif"))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(exit_code)
