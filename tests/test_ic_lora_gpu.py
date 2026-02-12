"""
REAL GPU test: IC LoRA (depth/pose/canny control) with LTX-2 19B.

Uses Donald1_00003.mp4 as control video with depth IC LoRA to generate
a depth-guided video. Runs on actual 4090 GPU with real model weights.

Run with:
    python -m pytest tests/test_ic_lora_gpu.py -v -s --tb=short
    python tests/test_ic_lora_gpu.py          # standalone
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


def _make_orchestrator(wan_root: str, output_dir: Path):
    """Real GPU orchestrator — NO smoke mode."""
    os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    from headless_wgp import WanOrchestrator
    return WanOrchestrator(wan_root, main_output_dir=str(output_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestICLoraGPU:
    """Real GPU IC LoRA tests with LTX-2 19B and depth/pose/canny control."""

    def test_ic_lora_depth_control(
        self, _chdir_to_wan2gp, output_dir, control_video, start_image
    ):
        """Generate video using depth IC LoRA with Donald1_00003.mp4 as control."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        print("[GPU] Model loaded.")

        print(f"[GPU] IC LoRA depth control with: {Path(control_video).name}")
        result = orch.generate(
            prompt="A person walking in a cinematic scene, depth-guided motion, smooth camera",
            resolution="512x320",
            video_length=33,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
            start_image=start_image,
            video_guide=control_video,
            video_prompt_type="DVG",
            control_net_weight=0.8,
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Depth IC LoRA output: {result} ({size / 1024:.1f} KB)")

    def test_ic_lora_pose_control(
        self, _chdir_to_wan2gp, output_dir, control_video, start_image
    ):
        """Generate video using pose IC LoRA with Donald1_00003.mp4 as control."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        print("[GPU] Model loaded.")

        print(f"[GPU] IC LoRA pose control with: {Path(control_video).name}")
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
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Pose IC LoRA output: {result} ({size / 1024:.1f} KB)")

    def test_ic_lora_canny_control(
        self, _chdir_to_wan2gp, output_dir, control_video, start_image
    ):
        """Generate video using canny edge IC LoRA with Donald1_00003.mp4 as control."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        print("[GPU] Model loaded.")

        print(f"[GPU] IC LoRA canny control with: {Path(control_video).name}")
        result = orch.generate(
            prompt="A figure with sharp edges, canny edge guided video generation",
            resolution="512x320",
            video_length=33,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
            start_image=start_image,
            video_guide=control_video,
            video_prompt_type="EVG",
            control_net_weight=0.8,
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Canny IC LoRA output: {result} ({size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(exit_code)
