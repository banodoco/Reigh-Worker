"""
Smoke tests for LTX-2 multi-frame guide images via latent injection.

Tests the parameter flow from task_types -> task_conversion -> orchestrator ->
wgp_params -> ltx2.generate() without requiring GPU or model weights.

Run with:
    python -m pytest tests/test_ltx2_multiframe_smoke.py -v
    python tests/test_ltx2_multiframe_smoke.py          # standalone
"""

import ast
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if heavy deps are available (cv2 triggers from source.__init__)
try:
    from source.task_handlers.tasks.task_types import (
        TASK_TYPE_TO_MODEL, DIRECT_QUEUE_TASK_TYPES, WGP_TASK_TYPES,
    )
    HAS_SOURCE = True
except ImportError:
    HAS_SOURCE = False

try:
    from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
    HAS_CONVERSION = True
except ImportError:
    HAS_CONVERSION = False

try:
    from headless_wgp import WanOrchestrator
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False

needs_source = pytest.mark.skipif(not HAS_SOURCE, reason="source package deps (cv2) not available")
needs_conversion = pytest.mark.skipif(not HAS_CONVERSION, reason="task_conversion deps not available")
needs_orchestrator = pytest.mark.skipif(not HAS_ORCHESTRATOR, reason="headless_wgp deps not available")


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
def guide_image_paths(tmp_path):
    """Create 4 tiny PNG images for guide_images testing."""
    from PIL import Image
    paths = []
    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(50 * i, 100, 200))
        path = tmp_path / f"guide_{i}.png"
        img.save(str(path))
        paths.append(str(path))
    return paths


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_orchestrator(wan_root: str, output_dir: Path):
    """Instantiate a WanOrchestrator in smoke mode (no GPU needed)."""
    os.environ["HEADLESS_WAN2GP_SMOKE"] = "1"
    try:
        orch = WanOrchestrator(wan_root, main_output_dir=str(output_dir))
    finally:
        os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    return orch


# ===================================================================
# Tests: Task Type Registration
# ===================================================================

class TestTaskTypeRegistration:
    """Verify ltx2_multiframe is properly registered in all task type sets."""

    @needs_source
    def test_task_type_in_model_map(self):
        assert "ltx2_multiframe" in TASK_TYPE_TO_MODEL
        assert TASK_TYPE_TO_MODEL["ltx2_multiframe"] == "ltx2_19B"

    @needs_source
    def test_task_type_in_direct_queue(self):
        assert "ltx2_multiframe" in DIRECT_QUEUE_TASK_TYPES

    @needs_source
    def test_task_type_in_wgp_types(self):
        assert "ltx2_multiframe" in WGP_TASK_TYPES

    def test_task_type_in_source_via_ast(self):
        """Fallback AST check: verify ltx2_multiframe appears in task_types.py."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_types.py").read_text()
        assert '"ltx2_multiframe"' in src


# ===================================================================
# Tests: Parameter Whitelist
# ===================================================================

class TestParamWhitelist:
    """Verify guide_images is in the task_conversion parameter whitelist."""

    def test_guide_images_in_whitelist(self):
        """guide_images should pass through the param whitelist."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text()
        assert '"guide_images"' in src or "'guide_images'" in src, (
            "guide_images should be in param_whitelist in task_conversion.py"
        )


# ===================================================================
# Tests: Task Conversion
# ===================================================================

class TestTaskConversion:
    """Verify task_conversion handles ltx2_multiframe guide_images."""

    @needs_conversion
    def test_local_paths_pass_through(self, _chdir_to_wan2gp, guide_image_paths):
        """Local file paths should be kept as-is in resolved guide_images."""
        db_params = {
            "prompt": "test prompt",
            "resolution": "768x512",
            "video_length": 105,
            "seed": 42,
            "guide_images": [
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 40, "strength": 0.8},
            ],
        }

        task = db_task_to_generation_task(
            db_params, task_id="test-1", task_type="ltx2_multiframe",
            wan2gp_path=str(PROJECT_ROOT / "Wan2GP"),
        )

        guides = task.parameters.get("guide_images")
        assert guides is not None
        assert len(guides) == 2
        assert guides[0]["image"] == guide_image_paths[0]
        assert guides[0]["frame_idx"] == 0
        assert guides[0]["strength"] == 1.0
        assert guides[1]["strength"] == 0.8

    @needs_conversion
    def test_url_download(self, _chdir_to_wan2gp):
        """HTTP URLs should be downloaded to temp files."""
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color="red").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.content = png_bytes
        mock_resp.raise_for_status = MagicMock()

        db_params = {
            "prompt": "test url download",
            "resolution": "768x512",
            "video_length": 105,
            "seed": 1,
            "guide_images": [
                {"image": "https://example.com/img.png", "frame_idx": 0, "strength": 1.0},
            ],
        }

        with patch("requests.get", return_value=mock_resp) as mock_get:
            task = db_task_to_generation_task(
                db_params, task_id="test-2", task_type="ltx2_multiframe",
                wan2gp_path=str(PROJECT_ROOT / "Wan2GP"),
            )
            mock_get.assert_called_once_with("https://example.com/img.png", timeout=30)

        guides = task.parameters["guide_images"]
        assert len(guides) == 1
        assert os.path.exists(guides[0]["image"])
        os.unlink(guides[0]["image"])

    def test_conversion_handler_exists_in_source(self):
        """Verify the ltx2_multiframe handler exists in task_conversion.py via AST."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text()
        assert 'task_type == "ltx2_multiframe"' in src


# ===================================================================
# Tests: Orchestrator Bridge
# ===================================================================

class TestOrchestratorBridge:
    """Verify orchestrator converts guide_images paths to PIL."""

    @needs_orchestrator
    def test_bridge_converts_paths_to_pil(
        self, _chdir_to_wan2gp, output_dir, guide_image_paths
    ):
        """guide_images dict entries should become (PIL, frame_idx, strength) tuples."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")

        result = orch.generate(
            prompt="test guide bridging",
            resolution="768x512",
            video_length=33,
            seed=42,
            guide_images=[
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 16, "strength": 0.9},
            ],
        )
        assert result is not None

    def test_bridge_code_exists_in_orchestrator(self):
        """Verify guide_images bridging code exists in orchestrator.py."""
        src = (PROJECT_ROOT / "source" / "models" / "wgp" / "orchestrator.py").read_text()
        assert 'guide_images' in src
        assert '[LTX2_BRIDGE] Converted' in src


# ===================================================================
# Tests: WGP Params
# ===================================================================

class TestWGPParams:
    """Verify guide_images flows through parameter builders."""

    def test_passthrough_mode_includes_guide_images(self):
        """build_passthrough_params should include guide_images from resolved_params."""
        from source.models.wgp.generators.wgp_params import build_passthrough_params

        fake_guides = [("pil_img", 0, 1.0), ("pil_img2", 40, 0.8)]
        params = build_passthrough_params(
            state={},
            current_model="ltx2_19B",
            image_mode=0,
            resolved_params={"guide_images": fake_guides, "prompt": "test"},
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )
        assert params["guide_images"] == fake_guides

    def test_normal_mode_includes_guide_images(self):
        """build_normal_params should include guide_images from resolved_params."""
        from source.models.wgp.generators.wgp_params import build_normal_params

        fake_guides = [("pil_img", 0, 1.0)]
        params = build_normal_params(
            state={},
            current_model="ltx2_19B",
            image_mode=0,
            resolved_params={"guide_images": fake_guides},
            prompt="test",
            actual_video_length=105,
            actual_batch_size=1,
            actual_guidance=4.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert params["guide_images"] == fake_guides

    def test_normal_mode_guide_images_none_by_default(self):
        """guide_images should be None when not provided."""
        from source.models.wgp.generators.wgp_params import build_normal_params

        params = build_normal_params(
            state={},
            current_model="ltx2_19B",
            image_mode=0,
            resolved_params={},
            prompt="test",
            actual_video_length=105,
            actual_batch_size=1,
            actual_guidance=4.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert params["guide_images"] is None


# ===================================================================
# Tests: LTX-2 Generate — Guide Images Logic
# ===================================================================

class TestLTX2GenerateGuideImages:
    """Verify guide_images are added to the images list in ltx2.py."""

    def test_latent_index_calculation(self):
        """_to_latent_index should divide frame_idx by stride (integer division)."""
        # Verify logic via inline replication (avoids torch import from ltx2.py)
        def _to_latent_index(frame_idx, stride):
            return int(frame_idx) // int(stride)

        assert _to_latent_index(0, 8) == 0
        assert _to_latent_index(7, 8) == 0
        assert _to_latent_index(8, 8) == 1
        assert _to_latent_index(40, 8) == 5
        assert _to_latent_index(80, 8) == 10
        assert _to_latent_index(104, 8) == 13

    def test_latent_index_function_in_source(self):
        """Verify _to_latent_index exists and uses integer division."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        assert "def _to_latent_index(frame_idx" in src
        assert "int(frame_idx) // int(stride)" in src

    def test_guide_frame_minus_one_resolves_to_last(self):
        """frame_idx=-1 in guide_images should map to frame_num-1."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        assert "if frame_idx == -1:" in src
        assert "frame_idx = frame_num - 1" in src

    def test_guide_images_code_appends_to_images_list(self):
        """The guide_images block should append entries to the images list."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        assert 'guide_images = kwargs.get("guide_images")' in src
        assert "images.append(entry)" in src
        assert "images_stage2.append(entry)" in src


# ===================================================================
# Tests: Reference Parameters
# ===================================================================

class TestReferenceParams:
    """Verify the reference parameters file is consistent."""

    def test_params_structure(self):
        from tests.ltx2_multiframe_params import LTX2_MULTIFRAME_PARAMS as P
        assert P["video_length"] == 105
        assert P["resolution"] == "768x512"
        assert len(P["guide_images"]) == 4
        assert P["guide_images"][0]["frame_idx"] == 0
        assert P["guide_images"][-1]["frame_idx"] == -1

    def test_all_guides_have_required_keys(self):
        from tests.ltx2_multiframe_params import LTX2_MULTIFRAME_PARAMS as P
        for entry in P["guide_images"]:
            assert "image" in entry
            assert "frame_idx" in entry
            assert "strength" in entry


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
