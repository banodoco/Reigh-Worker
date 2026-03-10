"""
REAL GPU test: LTX-2 travel between images with Deforum Evolution LoRA.

Uses 5 JPG test images (1.jpg–5.jpg) and the Deforum Evolution LoRA to
generate transition videos on actual GPU hardware.

Run with:
    python -m pytest tests/test_travel_ltx2_lora_gpu.py -v -s --tb=short
    python tests/test_travel_ltx2_lora_gpu.py          # standalone
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


LORA_FILENAME = "LTXV_2_Deforum-Evolution_v1.safetensors"
LORA_REPO_ID = "s4f3tymarc/LTX-2_Deforum_Evolution_v1"


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
    """The 5 JPG test images for travel-between-images."""
    paths = [str(TESTS_DIR / f"{i}.jpg") for i in range(1, 6)]
    for p in paths:
        assert os.path.isfile(p), f"Test image missing: {p}"
    return paths


@pytest.fixture()
def deforum_lora(_chdir_to_wan2gp):
    """Download Deforum Evolution LoRA to Wan2GP/loras/ltx2/ if not present."""
    lora_dir = Path(_chdir_to_wan2gp) / "loras" / "ltx2"
    lora_dir.mkdir(parents=True, exist_ok=True)
    lora_path = lora_dir / LORA_FILENAME

    if not lora_path.is_file():
        print(f"\n[LoRA] Downloading {LORA_FILENAME} from {LORA_REPO_ID}...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=LORA_REPO_ID,
            filename=LORA_FILENAME,
            local_dir=str(lora_dir),
        )
        assert lora_path.is_file(), f"LoRA download failed: {lora_path}"
        print(f"[LoRA] Downloaded to {lora_path} ({lora_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"\n[LoRA] Already present: {lora_path}")

    return LORA_FILENAME


def _make_orchestrator(wan_root: str, output_dir: Path):
    """Real GPU orchestrator — NO smoke mode."""
    os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    from headless_wgp import WanOrchestrator
    return WanOrchestrator(wan_root, main_output_dir=str(output_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTravelLTX2LoraGPU:
    """Real GPU travel-between-images with LTX-2 and Deforum Evolution LoRA."""

    def test_single_segment_with_lora(
        self, _chdir_to_wan2gp, output_dir, test_images, deforum_lora
    ):
        """Generate 1→2 transition (33 frames, 512x320) with Deforum LoRA."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        assert orch._is_ltx2() is True
        print("[GPU] Model loaded.")

        print(f"[GPU] Generating: 1.jpg → 2.jpg with LoRA {deforum_lora}")
        result = orch.generate(
            prompt="A smooth cinematic transition, dreamy motion, evolving forms, deforum style",
            resolution="512x320",
            video_length=33,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
            start_image=test_images[0],
            end_image=test_images[1],
            lora_names=[deforum_lora],
            lora_multipliers=[1.0],
        )

        assert result is not None, "generate() returned None"
        assert os.path.isfile(result), f"Output not found: {result}"
        size = os.path.getsize(result)
        assert size > 1000, f"Output too small ({size} bytes), likely not a real video"
        print(f"[GPU] Output: {result} ({size / 1024:.1f} KB)")

    def test_multi_segment_travel_with_lora(
        self, _chdir_to_wan2gp, output_dir, test_images, deforum_lora
    ):
        """Generate all 4 segments (1→2→3→4→5) with LoRA, verify all outputs."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        print("\n[GPU] Loading LTX-2 19B model...")
        switched = orch.load_model("ltx2_19B")
        assert switched is True
        assert orch._is_ltx2() is True
        print("[GPU] Model loaded.")

        results = []
        for i in range(len(test_images) - 1):
            seg = i + 1
            src = test_images[i]
            dst = test_images[i + 1]
            print(f"\n[GPU] Segment {seg}/4: {Path(src).name} → {Path(dst).name} (LoRA: {deforum_lora})")

            result = orch.generate(
                prompt="A smooth cinematic transition, dreamy motion, evolving forms, deforum style",
                resolution="512x320",
                video_length=33,
                num_inference_steps=20,
                guidance_scale=4.0,
                seed=42 + i,
                start_image=src,
                end_image=dst,
                lora_names=[deforum_lora],
                lora_multipliers=[1.0],
            )

            assert result is not None, f"Segment {seg}: generate() returned None"
            assert os.path.isfile(result), f"Segment {seg}: output not found: {result}"
            size = os.path.getsize(result)
            assert size > 1000, f"Segment {seg}: output too small ({size} bytes)"
            print(f"[GPU] Segment {seg} output: {result} ({size / 1024:.1f} KB)")
            results.append(result)

        assert len(results) == 4, f"Expected 4 segments, got {len(results)}"
        print(f"\n[GPU] All 4 segments generated successfully.")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(exit_code)
