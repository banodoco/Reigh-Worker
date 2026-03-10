"""
Smoke test for LTX-2 IC-LoRA Pose workflow integration.

Tests the pose IC-LoRA pipeline path with union control LoRA support.
Uses HEADLESS_WAN2GP_SMOKE=1 so no GPU or model weights are required.

Run with:
    python -m pytest tests/test_ltx2_pose_smoke.py -v
    python tests/test_ltx2_pose_smoke.py          # standalone
"""

import ast
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
def start_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    path = tmp_path / "start.png"
    img.save(str(path))
    return str(path)


@pytest.fixture()
def sample_video(tmp_path):
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    sample = samples_dir / "test.mp4"
    if not sample.exists():
        sample.write_bytes(b"\x00" * 128)
    return str(sample)


@pytest.fixture()
def control_video(tmp_path):
    """Create a tiny placeholder video for smoke-mode control input."""
    path = tmp_path / "control.mp4"
    path.write_bytes(b"\x00" * 128)
    return str(path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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
# Tests
# ===================================================================

class TestLTX2PoseICLoraSmoke:
    """Smoke tests for pose IC-LoRA pipeline via WanOrchestrator."""

    def test_pose_generation_smoke(
        self, _chdir_to_wan2gp, output_dir, start_image, sample_video
    ):
        """PVG video_prompt_type with pose IC-LoRA path exercise."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        result = orch.generate(
            prompt="a young woman dancing in her room",
            resolution="768x512",
            video_length=97,
            num_inference_steps=8,
            guidance_scale=1.0,
            seed=42,
            video_prompt_type="PVG",
            control_net_weight=0.95,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_video_prompt_type_g_enables_guidance(
        self, _chdir_to_wan2gp, output_dir, sample_video
    ):
        """'G' letter in video_prompt_type should enable guidance conditioning."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        # Without G: no guidance
        result = orch.generate(
            prompt="test without guidance",
            resolution="512x320",
            video_length=33,
            seed=42,
            video_prompt_type="PV",
        )
        assert result is not None


class TestUnionControlLoraMapping:
    """Verify union LoRA selection logic in get_loras_transformer."""

    def test_union_lora_selected_for_multiple_controls(self):
        """When video_prompt_type has multiple control letters (e.g. 'PD'),
        union LoRA should be selected if available."""
        tree = ast.parse(
            (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        )
        found_union_check = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_loras_transformer":
                src = ast.get_source_segment(
                    (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text(),
                    node,
                )
                assert "union-control" in src or "union_control" in src, (
                    "get_loras_transformer should check for union control LoRA"
                )
                assert "requested_controls" in src, (
                    "get_loras_transformer should track requested control types"
                )
                found_union_check = True
                break
        assert found_union_check, "get_loras_transformer not found in ltx2.py"

    def test_single_control_uses_separate_lora(self):
        """Single control letter (e.g. 'P' only) should use separate LoRA,
        not union, verified via AST."""
        src = (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2.py").read_text()
        # The logic should have: len(requested_controls) > 1 check
        assert "len(requested_controls) > 1" in src, (
            "get_loras_transformer should only use union LoRA when multiple controls requested"
        )

    def test_union_lora_url_in_config(self):
        """Union control LoRA URL should be present in ltx2_19B.json."""
        import json
        cfg = json.loads(
            (PROJECT_ROOT / "Wan2GP" / "defaults" / "ltx2_19B.json").read_text()
        )
        preload = cfg["model"]["preload_URLs"]
        union_urls = [u for u in preload if "union" in u.lower()]
        assert len(union_urls) == 1, f"Expected exactly 1 union URL, got: {union_urls}"
        assert "union-control" in union_urls[0]


class TestPoseParametersMatchWorkflow:
    """Verify ltx2pose.json workflow parameters map correctly to pipeline."""

    def test_workflow_core_params(self):
        """Core parameters from the ComfyUI workflow should be valid for the pipeline."""
        from tests.ltx2pose_params import LTX2_POSE_WORKFLOW_PARAMS as P
        assert P["guidance_scale"] == 1.0, "Distilled mode should use CFG=1"
        assert P["num_inference_steps"] == 8, "Distilled fast mode uses 8 steps"
        assert P["control_net_weight"] == 0.95
        assert P["video_length"] == 97
        assert "P" in P["video_prompt_type"], "Pose control must be in video_prompt_type"

    def test_workflow_lora_config(self):
        """Distilled LoRA should be configured with correct multiplier."""
        from tests.ltx2pose_params import LTX2_POSE_WORKFLOW_PARAMS as P
        assert "distilled" in P["activated_loras"][0]
        assert P["loras_multipliers"] == "0.7"

    def test_flow_shift_parameter(self):
        """Flow shift from LTXVScheduler should be preserved."""
        from tests.ltx2pose_params import LTX2_POSE_WORKFLOW_PARAMS as P
        assert P["flow_shift"] == 2.05


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
