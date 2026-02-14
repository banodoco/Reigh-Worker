"""
WAN core services smoke tests — the most important new coverage.

WAN services (T2V, I2V, VACE, Flux, Hunyuan) previously had zero headless
test coverage.  These tests verify that all families still work correctly
after the LTX2 addition.

Uses HEADLESS_WAN2GP_SMOKE=1 so no GPU or model weights are required.

Run with:
    python -m pytest tests/test_wan_headless.py -v
    python tests/test_wan_headless.py          # standalone
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


# ---------------------------------------------------------------------------
# Fixtures (reused from test_ltx2_headless.py patterns)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _chdir_to_wan2gp(tmp_path):
    """WanOrchestrator asserts cwd == wan_root."""
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
def start_image(tmp_path):
    """Create a small RGB PNG to act as the start anchor image."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    path = tmp_path / "start.png"
    img.save(str(path))
    return str(path)


@pytest.fixture()
def end_image(tmp_path):
    """Create a small RGB PNG to act as the end anchor image."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(50, 100, 200))
    path = tmp_path / "end.png"
    img.save(str(path))
    return str(path)


@pytest.fixture()
def sample_video(tmp_path):
    """Create a placeholder file that smoke mode can copy as output."""
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    sample = samples_dir / "test.mp4"
    if not sample.exists():
        sample.write_bytes(b"\x00" * 128)
    return str(sample)


def _make_orchestrator(wan_root: str, output_dir: Path):
    """Instantiate a WanOrchestrator in smoke mode (no GPU needed)."""
    os.environ["HEADLESS_WAN2GP_SMOKE"] = "1"
    try:
        from headless_wgp import WanOrchestrator
        orch = WanOrchestrator(wan_root, main_output_dir=str(output_dir))
    finally:
        os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    return orch


# ===================================================================
# TestT2VSmoke — Text-to-Video
# ===================================================================

class TestT2VSmoke:
    """Verify T2V still works after LTX2 addition."""

    def test_t2v_model_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("t2v")
        assert switched is True
        assert orch.current_model == "t2v"
        assert orch._is_t2v() is True

    def test_t2v_smoke_generation(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("t2v")
        result = orch.generate(prompt="A sunset over the ocean")
        assert result is not None
        assert os.path.exists(result)

    def test_t2v_with_parameters(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("t2v")
        result = orch.generate(
            prompt="A cat playing in a garden",
            resolution="1280x720",
            video_length=49,
            seed=42,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_t2v_22_model_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("t2v_2_2")
        assert switched is True
        assert orch.current_model == "t2v_2_2"
        assert orch._is_t2v() is True


# ===================================================================
# TestI2VSmoke — Image-to-Video
# ===================================================================

class TestI2VSmoke:
    """Verify I2V image inputs still work."""

    def test_i2v_model_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("i2v_14B")
        assert switched is True
        assert orch.current_model == "i2v_14B"

    def test_i2v_with_start_image(self, _chdir_to_wan2gp, output_dir, start_image, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("i2v_14B")
        result = orch.generate(
            prompt="Zoom into the scene",
            start_image=start_image,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_i2v_with_both_images(self, _chdir_to_wan2gp, output_dir, start_image, end_image, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("i2v_14B")
        result = orch.generate(
            prompt="Smooth transition between two scenes",
            start_image=start_image,
            end_image=end_image,
        )
        assert result is not None
        assert os.path.exists(result)


# ===================================================================
# TestVACESmoke — VACE models
# ===================================================================

class TestVACESmoke:
    """Verify VACE still works."""

    def test_vace_model_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("vace_14B")
        assert switched is True
        assert orch.current_model == "vace_14B"
        assert orch._is_vace() is True

    def test_vace_smoke_generation(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("vace_14B")
        result = orch.generate(prompt="A forest path in autumn")
        assert result is not None
        assert os.path.exists(result)

    def test_vace_cocktail_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("vace_14B_cocktail_2_2")
        assert switched is True
        assert orch.current_model == "vace_14B_cocktail_2_2"
        assert orch._is_vace() is True


# ===================================================================
# TestFluxSmoke — Flux model
# ===================================================================

class TestFluxSmoke:
    """Verify Flux still works."""

    def test_flux_model_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("flux")
        assert switched is True
        assert orch.current_model == "flux"
        assert orch._is_flux() is True

    def test_flux_smoke_generation(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("flux")
        result = orch.generate(prompt="A photorealistic portrait")
        assert result is not None
        assert os.path.exists(result)

    def test_flux_is_not_ltx2(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("flux")
        assert orch._is_ltx2() is False
        assert orch._is_flux() is True


# ===================================================================
# TestHunyuanSmoke — Hunyuan model
# ===================================================================

class TestHunyuanSmoke:
    """Verify Hunyuan still works."""

    def test_hunyuan_model_loads(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        switched = orch.load_model("hunyuan")
        assert switched is True
        assert orch.current_model == "hunyuan"

    def test_hunyuan_smoke_generation(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("hunyuan")
        result = orch.generate(prompt="A traditional ink painting")
        assert result is not None
        assert os.path.exists(result)


# ===================================================================
# TestModelSwitching — Model switching across all families
# ===================================================================

class TestModelSwitching:
    """Verify model switching is intact across all families."""

    def test_full_rotation(self, _chdir_to_wan2gp, output_dir, sample_video):
        """Switch through t2v → i2v → vace → ltx2 → flux → t2v."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        models = ["t2v", "i2v_14B", "vace_14B", "ltx2_19B", "flux", "t2v"]
        for model in models:
            orch.load_model(model)
            assert orch.current_model == model, (
                f"Expected current_model={model}, got {orch.current_model}"
            )

    def test_reload_same_is_noop(self, _chdir_to_wan2gp, output_dir, sample_video):
        """Loading the same model twice should not 'switch'."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        assert orch.load_model("t2v") is True   # first load
        assert orch.load_model("t2v") is False   # noop


# ===================================================================
# TestWANParameterPassthrough — Parameters still flow through generate()
# ===================================================================

class TestWANParameterPassthrough:
    """Verify parameters are accepted by generate() without error."""

    def test_t2v_standard_params(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("t2v")
        result = orch.generate(
            prompt="A test prompt",
            resolution="1280x720",
            video_length=49,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=123,
            negative_prompt="blurry",
        )
        assert result is not None

    def test_i2v_image_params(self, _chdir_to_wan2gp, output_dir, start_image, end_image, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("i2v_14B")
        result = orch.generate(
            prompt="Transition scene",
            start_image=start_image,
            end_image=end_image,
            resolution="768x512",
            video_length=25,
        )
        assert result is not None

    def test_vace_control_params(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("vace_14B")
        result = orch.generate(
            prompt="A controlled edit",
            resolution="1280x720",
            guidance_scale=5.0,
            num_inference_steps=30,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
