"""
Smoke tests for combined IC LoRA + multi-frame guide images (ltx2_ic_multiframe).

Tests the parameter flow from task_types -> task_conversion -> orchestrator ->
wgp_params -> ltx2.generate() for both IC LoRA control and guide images
without requiring GPU or model weights.

Run with:
    python -m pytest tests/test_ltx2_ic_multiframe_smoke.py -v
    python tests/test_ltx2_ic_multiframe_smoke.py          # standalone
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
    """Verify ltx2_ic_multiframe is properly registered in all task type sets."""

    @needs_source
    def test_task_type_in_model_map(self):
        assert "ltx2_ic_multiframe" in TASK_TYPE_TO_MODEL
        assert TASK_TYPE_TO_MODEL["ltx2_ic_multiframe"] == "ltx2_19B"

    @needs_source
    def test_task_type_in_direct_queue(self):
        assert "ltx2_ic_multiframe" in DIRECT_QUEUE_TASK_TYPES

    @needs_source
    def test_task_type_in_wgp_types(self):
        assert "ltx2_ic_multiframe" in WGP_TASK_TYPES

    def test_task_type_in_source_via_ast(self):
        """Fallback AST check: verify ltx2_ic_multiframe appears in task_types.py."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_types.py").read_text()
        assert '"ltx2_ic_multiframe"' in src


# ===================================================================
# Tests: Parameter Whitelist
# ===================================================================

class TestParamWhitelist:
    """Verify both guide_images and video_prompt_type are in the whitelist."""

    def test_guide_images_in_whitelist(self):
        """guide_images should pass through the param whitelist."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text()
        assert '"guide_images"' in src or "'guide_images'" in src, (
            "guide_images should be in param_whitelist in task_conversion.py"
        )

    def test_video_prompt_type_in_whitelist(self):
        """video_prompt_type should pass through the param whitelist."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text()
        assert '"video_prompt_type"' in src or "'video_prompt_type'" in src, (
            "video_prompt_type should be in param_whitelist in task_conversion.py"
        )

    def test_control_net_weight_in_whitelist(self):
        """control_net_weight should pass through the param whitelist."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text()
        assert '"control_net_weight"' in src or "'control_net_weight'" in src, (
            "control_net_weight should be in param_whitelist in task_conversion.py"
        )


# ===================================================================
# Tests: Task Conversion
# ===================================================================

class TestTaskConversion:
    """Verify task_conversion handles ltx2_ic_multiframe with both features."""

    @needs_conversion
    def test_local_paths_with_ic_lora_params(self, _chdir_to_wan2gp, guide_image_paths):
        """Local file paths + IC LoRA params should coexist in resolved task."""
        db_params = {
            "prompt": "test combined IC LoRA + multiframe",
            "resolution": "768x512",
            "video_length": 97,
            "seed": 42,
            "video_prompt_type": "PVG",
            "control_net_weight": 0.8,
            "guide_images": [
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 32, "strength": 0.9},
                {"image": guide_image_paths[2], "frame_idx": 64, "strength": 0.8},
            ],
        }

        task = db_task_to_generation_task(
            db_params, task_id="test-ic-mf-1", task_type="ltx2_ic_multiframe",
            wan2gp_path=str(PROJECT_ROOT / "Wan2GP"),
        )

        # Guide images resolved
        guides = task.parameters.get("guide_images")
        assert guides is not None
        assert len(guides) == 3
        assert guides[0]["image"] == guide_image_paths[0]
        assert guides[0]["frame_idx"] == 0
        assert guides[2]["strength"] == 0.8

        # IC LoRA params pass through
        assert task.parameters.get("video_prompt_type") == "PVG"
        assert task.parameters.get("control_net_weight") == 0.8

    @needs_conversion
    def test_url_download_with_ic_lora(self, _chdir_to_wan2gp):
        """HTTP URLs should be downloaded; IC LoRA params preserved."""
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color="red").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.content = png_bytes
        mock_resp.raise_for_status = MagicMock()

        db_params = {
            "prompt": "test url download with IC LoRA",
            "resolution": "768x512",
            "video_length": 97,
            "seed": 1,
            "video_prompt_type": "DVG",
            "control_net_weight": 0.7,
            "guide_images": [
                {"image": "https://example.com/guide.png", "frame_idx": 0, "strength": 1.0},
            ],
        }

        with patch("requests.get", return_value=mock_resp) as mock_get:
            task = db_task_to_generation_task(
                db_params, task_id="test-ic-mf-2", task_type="ltx2_ic_multiframe",
                wan2gp_path=str(PROJECT_ROOT / "Wan2GP"),
            )
            mock_get.assert_called_once_with("https://example.com/guide.png", timeout=30)

        guides = task.parameters["guide_images"]
        assert len(guides) == 1
        assert os.path.exists(guides[0]["image"])
        assert task.parameters["video_prompt_type"] == "DVG"
        assert task.parameters["control_net_weight"] == 0.7
        os.unlink(guides[0]["image"])

    def test_conversion_handler_covers_ic_multiframe(self):
        """Verify the ltx2_ic_multiframe handler exists in task_conversion.py."""
        src = (PROJECT_ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text()
        assert '"ltx2_ic_multiframe"' in src


# ===================================================================
# Tests: Orchestrator Bridge
# ===================================================================

class TestOrchestratorBridge:
    """Verify orchestrator handles both guide_images and IC LoRA params."""

    @needs_orchestrator
    def test_bridge_with_both_features(
        self, _chdir_to_wan2gp, output_dir, guide_image_paths
    ):
        """guide_images + video_prompt_type should both flow through orchestrator."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")

        result = orch.generate(
            prompt="test combined IC LoRA + guide bridging",
            resolution="768x512",
            video_length=33,
            seed=42,
            video_prompt_type="PVG",
            control_net_weight=0.8,
            guide_images=[
                {"image": guide_image_paths[0], "frame_idx": 0, "strength": 1.0},
                {"image": guide_image_paths[1], "frame_idx": 16, "strength": 0.9},
            ],
        )
        assert result is not None


# ===================================================================
# Tests: WGP Params
# ===================================================================

class TestWGPParams:
    """Verify both guide_images and video_prompt_type flow through param builders."""

    def test_passthrough_mode_includes_both(self):
        """build_passthrough_params should include guide_images and video_prompt_type."""
        from source.models.wgp.generators.wgp_params import build_passthrough_params

        fake_guides = [("pil_img", 0, 1.0), ("pil_img2", 32, 0.8)]
        params = build_passthrough_params(
            state={},
            current_model="ltx2_19B",
            image_mode=0,
            resolved_params={
                "guide_images": fake_guides,
                "video_prompt_type": "PVG",
                "control_net_weight": 0.8,
                "prompt": "test",
            },
            video_guide=None,
            video_mask=None,
            video_prompt_type="PVG",
            control_net_weight=0.8,
            control_net_weight2=None,
        )
        assert params["guide_images"] == fake_guides
        assert params["video_prompt_type"] == "PVG"
        assert params["control_net_weight"] == 0.8

    def test_normal_mode_includes_both(self):
        """build_normal_params should include guide_images and video_prompt_type."""
        from source.models.wgp.generators.wgp_params import build_normal_params

        fake_guides = [("pil_img", 0, 1.0)]
        params = build_normal_params(
            state={},
            current_model="ltx2_19B",
            image_mode=0,
            resolved_params={"guide_images": fake_guides},
            prompt="test",
            actual_video_length=97,
            actual_batch_size=1,
            actual_guidance=4.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type="PVG",
            control_net_weight=0.8,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert params["guide_images"] == fake_guides
        assert params["video_prompt_type"] == "PVG"
        assert params["control_net_weight"] == 0.8


# ===================================================================
# Tests: Combined Code Paths (AST)
# ===================================================================

class TestCombinedCodePaths:
    """AST checks that ltx2.py generate() handles both guide_images and IC LoRA."""

    def test_generate_handles_guide_images(self):
        """ltx2.py generate() should process guide_images from kwargs."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        assert 'guide_images = kwargs.get("guide_images")' in src
        assert "images.append(entry)" in src

    def test_generate_handles_video_conditioning(self):
        """ltx2.py generate() should process video_conditioning for IC LoRA."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        assert "video_conditioning" in src

    def test_generate_has_both_kwargs(self):
        """generate() should accept both guide_images and video_prompt_type paths."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        # Both features are accessed from kwargs in generate()
        assert "guide_images" in src
        assert "video_prompt_type" in src


# ===================================================================
# Tests: Reference Parameters
# ===================================================================

class TestReferenceParams:
    """Validate the combined params structure."""

    def test_params_structure(self):
        from tests.ltx2_ic_multiframe_params import LTX2_IC_MULTIFRAME_PARAMS as P
        assert P["video_length"] == 97
        assert P["resolution"] == "768x512"
        assert P["num_inference_steps"] == 20
        assert P["guidance_scale"] == 4.0

    def test_ic_lora_params_present(self):
        from tests.ltx2_ic_multiframe_params import LTX2_IC_MULTIFRAME_PARAMS as P
        assert P["video_prompt_type"] == "PVG"
        assert P["control_net_weight"] == 0.8

    def test_guide_images_present(self):
        from tests.ltx2_ic_multiframe_params import LTX2_IC_MULTIFRAME_PARAMS as P
        assert len(P["guide_images"]) == 4
        assert P["guide_images"][0]["frame_idx"] == 0
        assert P["guide_images"][-1]["frame_idx"] == -1

    def test_all_guides_have_required_keys(self):
        from tests.ltx2_ic_multiframe_params import LTX2_IC_MULTIFRAME_PARAMS as P
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
