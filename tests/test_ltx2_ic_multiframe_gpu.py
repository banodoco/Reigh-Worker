"""
REAL GPU test: IC LoRA + multi-frame guide images with LTX-2 19B.

Combines structural control (pose/depth/canny via IC LoRA) with multiple
guide images at specific frame positions. Runs on actual GPU with real
model weights.

Run with:
    python -m pytest tests/test_ltx2_ic_multiframe_gpu.py -v -s --tb=short
    python tests/test_ltx2_ic_multiframe_gpu.py          # standalone
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Skip entire module if no GPU
# ---------------------------------------------------------------------------
import torch
if not torch.cuda.is_available():
    pytest.skip("No CUDA GPU available", allow_module_level=True)


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
def control_video():
    """Donald1_00003.mp4 control video for IC LoRA."""
    path = str(TESTS_DIR / "Donald1_00003.mp4")
    assert os.path.isfile(path), f"Control video missing: {path}"
    return path


@pytest.fixture()
def start_image():
    path = str(TESTS_DIR / "CICEK .png")
    assert os.path.isfile(path), f"Start image missing: {path}"
    return path


@pytest.fixture()
def guide_image_paths(tmp_path):
    """Create 3 guide images from the start image at different colors."""
    from PIL import Image
    paths = []
    colors = [(200, 100, 50), (50, 200, 100), (100, 50, 200)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (768, 512), color=color)
        path = tmp_path / f"guide_{i}.png"
        img.save(str(path))
        paths.append(str(path))
    return paths


def _make_orchestrator(wan_root: str, output_dir: Path):
    """Real GPU orchestrator -- NO smoke mode."""
    os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    from headless_wgp import WanOrchestrator
    return WanOrchestrator(wan_root, main_output_dir=str(output_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestICLoraMultiframeGPU:
    """Real GPU tests: IC LoRA + multi-frame guide images with LTX-2 19B."""

    def test_pose_with_multiframe_guides(
        self, _chdir_to_wan2gp, output_dir, control_video, start_image, guide_image_paths
    ):
        """PVG (pose) IC LoRA + 3 guide images at frames 0, 16, -1."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        print("[GPU] Model loaded.")

        print(f"[GPU] IC LoRA pose + 3 guide images")
        result = orch.generate(
            prompt="A person performing expressive gestures, pose-guided animation",
            resolution="512x320",
            video_length=33,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
            start_image=start_image,
            video_guide=control_video,
            video_prompt_type="PVG",
            control_net_weight=0.8,
            guide_images=[
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 16, "strength": 0.8},
                {"image": guide_image_paths[2], "frame_idx": -1, "strength": 0.8},
            ],
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Pose + multiframe output: {result} ({size / 1024:.1f} KB)")

    def test_depth_with_multiframe_guides(
        self, _chdir_to_wan2gp, output_dir, control_video, start_image, guide_image_paths
    ):
        """DVG (depth) IC LoRA + 3 guide images at frames 0, 16, -1."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        print("[GPU] Model loaded.")

        print(f"[GPU] IC LoRA depth + 3 guide images")
        result = orch.generate(
            prompt="A person walking in a cinematic scene, depth-guided motion",
            resolution="512x320",
            video_length=33,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
            start_image=start_image,
            video_guide=control_video,
            video_prompt_type="DVG",
            control_net_weight=0.8,
            guide_images=[
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 16, "strength": 0.8},
                {"image": guide_image_paths[2], "frame_idx": -1, "strength": 0.8},
            ],
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Depth + multiframe output: {result} ({size / 1024:.1f} KB)")

    def test_union_with_multiframe_guides(
        self, _chdir_to_wan2gp, output_dir, control_video, start_image, guide_image_paths
    ):
        """PDVG (union pose+depth) IC LoRA + 3 guide images at frames 0, 16, -1."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        print("[GPU] Model loaded.")

        print(f"[GPU] IC LoRA union (PDVG) + 3 guide images")
        result = orch.generate(
            prompt="A person dancing with detailed depth and pose, multi-frame guided",
            resolution="512x320",
            video_length=33,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
            start_image=start_image,
            video_guide=control_video,
            video_prompt_type="PDVG",
            control_net_weight=0.8,
            guide_images=[
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 16, "strength": 0.8},
                {"image": guide_image_paths[2], "frame_idx": -1, "strength": 0.8},
            ],
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Union + multiframe output: {result} ({size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(exit_code)
