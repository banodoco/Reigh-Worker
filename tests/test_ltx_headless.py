"""
LTX family tests — LTXv detection, task types, smoke generation, and
cross-model switching between LTX variants.

Expands LTX coverage beyond the existing test_ltx2_headless.py to include
LTXv and cross-model switching scenarios.

Uses HEADLESS_WAN2GP_SMOKE=1 so no GPU or model weights are required.

Run with:
    python -m pytest tests/test_ltx_headless.py -v
    python tests/test_ltx_headless.py          # standalone
"""

import json
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
# Fixtures
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
# TestLTXvModelDetection — LTXv detection distinct from LTX2
# ===================================================================

class TestLTXvModelDetection:
    """Verify LTXv is detected correctly and distinctly from LTX2."""

    @pytest.fixture()
    def orch(self, _chdir_to_wan2gp, output_dir, sample_video):
        return _make_orchestrator(_chdir_to_wan2gp, output_dir)

    def test_ltxv_detected_as_t2v(self, orch):
        """LTXv model maps to base type 't2v' (non-flux, non-ltx2)."""
        orch.load_model("ltxv_13B")
        assert orch._is_t2v() is True

    def test_ltxv_not_ltx2(self, orch):
        """LTXv should NOT be detected as LTX2."""
        orch.load_model("ltxv_13B")
        assert orch._is_ltx2() is False

    def test_ltxv_not_flux(self, orch):
        orch.load_model("ltxv_13B")
        assert orch._is_flux() is False

    def test_ltxv_not_vace(self, orch):
        orch.load_model("ltxv_13B")
        assert orch._is_vace() is False


# ===================================================================
# TestLTXvTaskTypes — LTXv registration intact
# ===================================================================

class TestLTXvTaskTypes:
    """Verify LTXv task type registrations."""

    def test_ltxv_in_wgp(self):
        from source.task_types import WGP_TASK_TYPES
        assert "ltxv" in WGP_TASK_TYPES

    def test_ltxv_in_direct_queue(self):
        from source.task_types import DIRECT_QUEUE_TASK_TYPES
        assert "ltxv" in DIRECT_QUEUE_TASK_TYPES

    def test_ltxv_model_mapping(self):
        from source.task_types import get_default_model
        assert get_default_model("ltxv") == "ltxv_13B"

    def test_ltxv_config_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "defaults" / "ltxv_13B.json").is_file()


# ===================================================================
# TestLTXvSmoke — LTXv generates in smoke mode
# ===================================================================

class TestLTXvSmoke:
    """Verify LTXv generates in smoke mode."""

    def test_ltxv_smoke_generation(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltxv_13B")
        result = orch.generate(prompt="A nature documentary clip")
        assert result is not None
        assert os.path.exists(result)

    def test_ltxv_with_parameters(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltxv_13B")
        result = orch.generate(
            prompt="A detailed scene",
            resolution="768x512",
            video_length=33,
            seed=99,
        )
        assert result is not None
        assert os.path.exists(result)


# ===================================================================
# TestCrossModelSwitching — Switching between LTX variants
# ===================================================================

class TestCrossModelSwitching:
    """Verify switching between LTX variants works."""

    def test_ltx2_to_ltxv(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True
        orch.load_model("ltxv_13B")
        assert orch._is_ltx2() is False
        assert orch.current_model == "ltxv_13B"

    def test_ltxv_to_t2v(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltxv_13B")
        orch.load_model("t2v")
        assert orch.current_model == "t2v"
        assert orch._is_t2v() is True

    def test_ltxv_to_ltx2(self, _chdir_to_wan2gp, output_dir, sample_video):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltxv_13B")
        assert orch._is_ltx2() is False
        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True
        assert orch.current_model == "ltx2_19B"

    def test_full_ltx_family_rotation(self, _chdir_to_wan2gp, output_dir, sample_video):
        """Rotate through all LTX-family variants and back."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        # ltx2 → ltxv → t2v → ltx2
        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True

        orch.load_model("ltxv_13B")
        assert orch._is_ltx2() is False
        assert orch._is_t2v() is True

        orch.load_model("t2v")
        assert orch._is_ltx2() is False

        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True


# ===================================================================
# TestLTXModelConfigs — LTX filesystem artifacts intact
# ===================================================================

class TestLTXModelConfigs:
    """Verify LTX config files and directory structure."""

    def test_ltx2_config_valid(self):
        cfg_path = PROJECT_ROOT / "Wan2GP" / "defaults" / "ltx2_19B.json"
        data = json.loads(cfg_path.read_text())
        assert isinstance(data, dict)
        assert "model" in data

    def test_ltxv_config_valid(self):
        cfg_path = PROJECT_ROOT / "Wan2GP" / "defaults" / "ltxv_13B.json"
        data = json.loads(cfg_path.read_text())
        assert isinstance(data, dict)
        assert "model" in data

    def test_ltx2_handler_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2_handler.py").is_file()

    def test_ltx2_model_dir_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2").is_dir()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
