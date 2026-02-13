"""
REAL GPU test: LTX-2 travel between images using actual 4090 GPU.

Generates a short video transitioning between the neon botanical test images
using the LTX-2 19B model (fp8). No smoke mode — actual inference on GPU.

Run with:
    python -m pytest tests/test_travel_real_gpu.py -v -s --tb=short
    python tests/test_travel_real_gpu.py          # standalone
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
def test_images():
    """The 3 neon botanical test images."""
    paths = [
        str(TESTS_DIR / "CICEK .png"),
        str(TESTS_DIR / "CICEK 2.png"),
        str(TESTS_DIR / "2AF816ADD8BC6EE937FEA5D5B3061DC047BBD8BA5A77E91A5158BF6C86C9FAD1.jpeg"),
    ]
    for p in paths:
        assert os.path.isfile(p), f"Test image missing: {p}"
    return paths


def _make_orchestrator(wan_root: str, output_dir: Path):
    """Real GPU orchestrator — NO smoke mode."""
    # Ensure smoke mode is OFF
    os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    from headless_wgp import WanOrchestrator
    return WanOrchestrator(wan_root, main_output_dir=str(output_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRealLTX2Travel:
    """Real GPU generation with LTX-2 model and test images."""

    def test_ltx2_single_segment_start_end(
        self, _chdir_to_wan2gp, output_dir, test_images
    ):
        """Generate a short video from CICEK.png → CICEK2.png using LTX-2."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        assert orch._is_ltx2() is True
        print("[GPU] Model loaded.")

        print(f"[GPU] Generating: {Path(test_images[0]).name} → {Path(test_images[1]).name}")
        result = orch.generate(
            prompt="A neon green sprout slowly transforms into a cyan sprout, bioluminescent glow, smooth cinematic transition",
            resolution="512x320",
            video_length=33,          # short: 33 frames (~2s)
            num_inference_steps=20,   # fewer steps for speed
            guidance_scale=4.0,
            seed=42,
            start_image=test_images[0],
            end_image=test_images[1],
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Output: {result} ({size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(exit_code)
