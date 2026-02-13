"""
Registration & model detection tests for all services.

Verifies exact membership of frozen sets, model mappings, model family
detection methods, and helper functions.

Uses HEADLESS_WAN2GP_SMOKE=1 for model detection tests (no GPU required).

Run with:
    python -m pytest tests/test_all_services_headless.py -v
    python tests/test_all_services_headless.py          # standalone
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
# TestTaskTypeRegistration — Exact membership of frozen sets
# ===================================================================

class TestTaskTypeRegistration:
    """Verify exact set contents — catches any accidental removal."""

    EXPECTED_WGP = frozenset({
        "vace", "vace_21", "vace_22",
        "flux",
        "t2v", "t2v_22", "wan_2_2_t2i",
        "i2v", "i2v_22",
        "hunyuan", "ltxv", "ltx2",
        "qwen_image_edit", "qwen_image_style", "image_inpaint", "annotated_image_edit",
        "inpaint_frames",
        "generate_video",
        "z_image_turbo", "z_image_turbo_i2i",
    })

    EXPECTED_DIRECT = EXPECTED_WGP - {"inpaint_frames"} | {
        "qwen_image_hires", "qwen_image", "qwen_image_2512",
    }

    def test_all_wgp_types_present(self):
        from source.task_types import WGP_TASK_TYPES
        assert WGP_TASK_TYPES == self.EXPECTED_WGP, (
            f"WGP_TASK_TYPES mismatch.\n"
            f"  Missing: {self.EXPECTED_WGP - WGP_TASK_TYPES}\n"
            f"  Extra:   {WGP_TASK_TYPES - self.EXPECTED_WGP}"
        )

    def test_all_direct_queue_types_present(self):
        from source.task_types import DIRECT_QUEUE_TASK_TYPES
        assert DIRECT_QUEUE_TASK_TYPES == self.EXPECTED_DIRECT, (
            f"DIRECT_QUEUE_TASK_TYPES mismatch.\n"
            f"  Missing: {self.EXPECTED_DIRECT - DIRECT_QUEUE_TASK_TYPES}\n"
            f"  Extra:   {DIRECT_QUEUE_TASK_TYPES - self.EXPECTED_DIRECT}"
        )

    def test_ltx2_in_both_sets(self):
        from source.task_types import WGP_TASK_TYPES, DIRECT_QUEUE_TASK_TYPES
        assert "ltx2" in WGP_TASK_TYPES
        assert "ltx2" in DIRECT_QUEUE_TASK_TYPES

    def test_ltxv_in_both_sets(self):
        from source.task_types import WGP_TASK_TYPES, DIRECT_QUEUE_TASK_TYPES
        assert "ltxv" in WGP_TASK_TYPES
        assert "ltxv" in DIRECT_QUEUE_TASK_TYPES


# ===================================================================
# TestModelMappings — Every task type → correct model name
# ===================================================================

class TestModelMappings:
    """Parametrized tests over all entries in TASK_TYPE_TO_MODEL."""

    EXPECTED_MAPPINGS = {
        "generate_video": "t2v",
        "vace": "vace_14B_cocktail_2_2",
        "vace_21": "vace_14B",
        "vace_22": "vace_14B_cocktail_2_2",
        "wan_2_2_t2i": "t2v_2_2",
        "t2v": "t2v",
        "t2v_22": "t2v_2_2",
        "flux": "flux",
        "i2v": "i2v_14B",
        "i2v_22": "i2v_2_2",
        "hunyuan": "hunyuan",
        "ltxv": "ltxv_13B",
        "ltx2": "ltx2_19B",
        "join_clips_segment": "wan_2_2_vace_lightning_baseline_2_2_2",
        "inpaint_frames": "wan_2_2_vace_lightning_baseline_2_2_2",
        "qwen_image_edit": "qwen_image_edit_20B",
        "qwen_image_hires": "qwen_image_edit_20B",
        "qwen_image_style": "qwen_image_edit_20B",
        "image_inpaint": "qwen_image_edit_20B",
        "annotated_image_edit": "qwen_image_edit_20B",
        "qwen_image": "qwen_image_edit_20B",
        "qwen_image_2512": "qwen_image_2512_20B",
        "z_image_turbo": "z_image",
        "z_image_turbo_i2i": "z_image_img2img",
    }

    @pytest.mark.parametrize("task_type,expected_model", list(EXPECTED_MAPPINGS.items()))
    def test_model_mapping(self, task_type, expected_model):
        from source.task_types import get_default_model
        assert get_default_model(task_type) == expected_model

    def test_fallback_for_unknown_type(self):
        from source.task_types import get_default_model
        assert get_default_model("nonexistent_task") == "t2v"


# ===================================================================
# TestModelDetectionMethods — _is_*() correctness in smoke mode
# ===================================================================

class TestModelDetectionMethods:
    """Load each model family and verify _is_*() methods return correct True/False."""

    @pytest.fixture()
    def orch(self, _chdir_to_wan2gp, output_dir, sample_video):
        return _make_orchestrator(_chdir_to_wan2gp, output_dir)

    def test_t2v_detection(self, orch):
        orch.load_model("t2v")
        assert orch._is_t2v() is True
        assert orch._is_flux() is False
        assert orch._is_vace() is False
        assert orch._is_ltx2() is False
        assert orch._is_qwen() is False

    def test_i2v_detection(self, orch):
        """i2v maps to base type 't2v' in smoke mode (non-flux, non-vace)."""
        orch.load_model("i2v_14B")
        assert orch._is_t2v() is True
        assert orch._is_flux() is False
        assert orch._is_vace() is False
        assert orch._is_ltx2() is False

    def test_vace_detection(self, orch):
        orch.load_model("vace_14B")
        assert orch._is_vace() is True
        assert orch._is_flux() is False
        assert orch._is_ltx2() is False

    def test_flux_detection(self, orch):
        orch.load_model("flux")
        assert orch._is_flux() is True
        assert orch._is_t2v() is False
        assert orch._is_vace() is False
        assert orch._is_ltx2() is False

    def test_ltx2_detection(self, orch):
        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True
        assert orch._is_t2v() is True  # ltx2 base type is "ltx2_19B", in T2V family
        assert orch._is_flux() is False
        assert orch._is_vace() is False

    def test_qwen_detection(self, orch):
        orch.load_model("qwen_image_edit_20B")
        assert orch._is_qwen() is True
        assert orch._is_flux() is False
        assert orch._is_vace() is False
        assert orch._is_ltx2() is False


# ===================================================================
# TestHelperFunctions — is_wgp_task / is_direct_queue_task
# ===================================================================

class TestHelperFunctions:
    """Verify helper functions are consistent with the frozen sets."""

    WGP_TYPES = [
        "t2v", "i2v", "vace", "flux", "hunyuan", "ltxv", "ltx2",
        "generate_video", "z_image_turbo", "inpaint_frames",
    ]
    DIRECT_TYPES = [
        "t2v", "i2v", "vace", "flux", "hunyuan", "ltxv", "ltx2",
        "qwen_image_edit", "qwen_image_hires", "qwen_image",
        "z_image_turbo", "z_image_turbo_i2i",
    ]
    NEGATIVE_TYPES = [
        "travel_orchestrator", "travel_segment", "magic_edit",
        "nonexistent", "join_clips_orchestrator",
    ]

    @pytest.mark.parametrize("task_type", WGP_TYPES)
    def test_is_wgp_task_positive(self, task_type):
        from source.task_types import is_wgp_task
        assert is_wgp_task(task_type) is True

    @pytest.mark.parametrize("task_type", NEGATIVE_TYPES)
    def test_is_wgp_task_negative(self, task_type):
        from source.task_types import is_wgp_task
        assert is_wgp_task(task_type) is False

    @pytest.mark.parametrize("task_type", DIRECT_TYPES)
    def test_is_direct_queue_task_positive(self, task_type):
        from source.task_types import is_direct_queue_task
        assert is_direct_queue_task(task_type) is True

    @pytest.mark.parametrize("task_type", NEGATIVE_TYPES)
    def test_is_direct_queue_task_negative(self, task_type):
        from source.task_types import is_direct_queue_task
        assert is_direct_queue_task(task_type) is False


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
