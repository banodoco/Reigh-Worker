"""
DB → GenerationTask pipeline tests.

Tests the critical db_task_to_generation_task() function that converts
Supabase task rows into GenerationTask objects for the queue, and
the parse_phase_config() function that handles multi-phase generation.

Uses mocking to avoid network calls (image downloads, etc.).

Run with:
    python -m pytest tests/test_task_conversion_headless.py -v
    python tests/test_task_conversion_headless.py          # standalone
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WAN2GP_PATH = str(PROJECT_ROOT / "Wan2GP")

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies that are not needed for headless tests.
# These modules are imported at module level by source/common_utils.py and
# other modules but are not exercised by the conversion logic under test.
# ---------------------------------------------------------------------------
for _mod_name in ("cv2", "mediapipe", "mediapipe.solutions"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)


# ===================================================================
# TestBasicConversion — One test per core task type
# ===================================================================

class TestBasicConversion:
    """Each core task type produces correct model + params."""

    def test_t2v_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "A beautiful sunset"},
            task_id="test-t2v-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "t2v"
        assert task.prompt == "A beautiful sunset"
        assert task.id == "test-t2v-001"

    def test_ltx2_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "A cinematic scene"},
            task_id="test-ltx2-001",
            task_type="ltx2",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "ltx2_19B"
        assert task.prompt == "A cinematic scene"

    def test_ltxv_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Nature documentary"},
            task_id="test-ltxv-001",
            task_type="ltxv",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "ltxv_13B"

    def test_i2v_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Zoom into the scene", "resolution": "1280x720"},
            task_id="test-i2v-001",
            task_type="i2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "i2v_14B"
        assert task.parameters.get("resolution") == "1280x720"

    def test_vace_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "A forest path"},
            task_id="test-vace-001",
            task_type="vace",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "vace_14B_cocktail_2_2"

    def test_flux_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "A portrait photo"},
            task_id="test-flux-001",
            task_type="flux",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "flux"

    def test_hunyuan_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "An ink painting"},
            task_id="test-hunyuan-001",
            task_type="hunyuan",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "hunyuan"

    def test_z_image_turbo_conversion(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "A quick sketch"},
            task_id="test-zimage-001",
            task_type="z_image_turbo",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "z_image"
        assert task.parameters.get("video_length") == 1
        assert task.parameters.get("guidance_scale") == 0


# ===================================================================
# TestModelOverride — Model selection priority
# ===================================================================

class TestModelOverride:
    """Verify model override behavior."""

    def test_explicit_model_overrides_default(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "model": "t2v_2_2"},
            task_id="test-override-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "t2v_2_2"

    def test_missing_model_uses_default(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test"},
            task_id="test-default-001",
            task_type="ltx2",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.model == "ltx2_19B"


# ===================================================================
# TestParameterPassthrough — Parameter mapping correct
# ===================================================================

class TestParameterPassthrough:
    """Verify whitelisted parameters pass through correctly."""

    def test_whitelisted_params_pass_through(self):
        from source.task_conversion import db_task_to_generation_task
        params = {
            "prompt": "Test prompt",
            "resolution": "1280x720",
            "video_length": 49,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": 42,
            "negative_prompt": "blurry",
            "flow_shift": 5.0,
        }
        task = db_task_to_generation_task(
            db_task_params=params,
            task_id="test-params-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["resolution"] == "1280x720"
        assert task.parameters["video_length"] == 49
        assert task.parameters["num_inference_steps"] == 30
        assert task.parameters["guidance_scale"] == 7.5
        assert task.parameters["seed"] == 42
        assert task.parameters["negative_prompt"] == "blurry"
        assert task.parameters["flow_shift"] == 5.0

    def test_steps_alias_maps_to_num_inference_steps(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "steps": 25},
            task_id="test-steps-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["num_inference_steps"] == 25

    def test_non_whitelisted_params_filtered(self):
        """Infrastructure params like supabase_url should not leak through."""
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={
                "prompt": "Test",
                "supabase_url": "https://example.com",
                "api_key": "secret",
                "random_junk": 42,
            },
            task_id="test-filter-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert "supabase_url" not in task.parameters
        assert "api_key" not in task.parameters
        assert "random_junk" not in task.parameters


# ===================================================================
# TestDefaults — Default values applied
# ===================================================================

class TestDefaults:
    """Verify essential default values are applied."""

    def test_seed_defaults(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test"},
            task_id="test-seed-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["seed"] == -1

    def test_negative_prompt_defaults(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test"},
            task_id="test-neg-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["negative_prompt"] == ""

    def test_empty_prompt_raises_for_non_img2img(self):
        from source.task_conversion import db_task_to_generation_task
        with pytest.raises(ValueError, match="prompt is required"):
            db_task_to_generation_task(
                db_task_params={"prompt": ""},
                task_id="test-empty-001",
                task_type="t2v",
                wan2gp_path=WAN2GP_PATH,
            )

    def test_empty_prompt_allowed_for_img2img(self):
        """img2img tasks (z_image_turbo_i2i, qwen_image_edit, etc.) allow empty prompts."""
        from source.task_conversion import db_task_to_generation_task

        # z_image_turbo_i2i needs an image URL; mock requests (imported inside
        # the handler) and PIL.Image.open to avoid network and file I/O.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_resp.raise_for_status = MagicMock()

        mock_img = MagicMock()
        mock_img.size = (512, 512)
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)

        with patch("requests.get", return_value=mock_resp), \
             patch("PIL.Image.open", return_value=mock_img):
            task = db_task_to_generation_task(
                db_task_params={"prompt": "", "image_url": "https://example.com/img.png"},
                task_id="test-img2img-001",
                task_type="z_image_turbo_i2i",
                wan2gp_path=WAN2GP_PATH,
            )
            assert task.prompt == " "  # minimal prompt


# ===================================================================
# TestPhaseConfigIntegration — Phase config parsing into gen params
# ===================================================================

class TestPhaseConfigIntegration:
    """Verify phase_config override processing in db_task_to_generation_task."""

    def _make_phase_config(self, num_phases=2, steps=None, guidance=None):
        """Helper to build a valid phase_config dict."""
        if steps is None:
            steps = [15, 15] if num_phases == 2 else [10, 10, 10]
        if guidance is None:
            guidance = [5.0, 1.0] if num_phases == 2 else [5.0, 3.0, 1.0]

        phases = []
        for i in range(num_phases):
            phases.append({
                "guidance_scale": guidance[i],
                "loras": [],
            })

        return {
            "num_phases": num_phases,
            "steps_per_phase": steps,
            "phases": phases,
            "flow_shift": 5.0,
            "sample_solver": "euler",
        }

    def test_phase_config_sets_steps(self):
        from source.task_conversion import db_task_to_generation_task
        pc = self._make_phase_config(num_phases=2, steps=[20, 10])
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "phase_config": pc},
            task_id="test-phase-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["num_inference_steps"] == 30  # 20 + 10

    def test_phase_config_sets_guidance_phases(self):
        from source.task_conversion import db_task_to_generation_task
        pc = self._make_phase_config(num_phases=3, steps=[10, 10, 10])
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "phase_config": pc},
            task_id="test-phase-002",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["guidance_phases"] == 3
        assert task.parameters["guidance_scale"] == 5.0
        assert task.parameters["guidance2_scale"] == 3.0
        assert task.parameters["guidance3_scale"] == 1.0

    def test_phase_config_sets_flow_shift(self):
        from source.task_conversion import db_task_to_generation_task
        pc = self._make_phase_config(num_phases=2, steps=[15, 15])
        pc["flow_shift"] = 12.0
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "phase_config": pc},
            task_id="test-phase-003",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.parameters["flow_shift"] == 12.0

    def test_phase_config_with_loras(self):
        from source.task_conversion import db_task_to_generation_task
        pc = self._make_phase_config(num_phases=2, steps=[15, 15])
        pc["phases"][0]["loras"] = [{"url": "https://example.com/lora1.safetensors", "multiplier": "1.2"}]
        pc["phases"][1]["loras"] = [{"url": "https://example.com/lora1.safetensors", "multiplier": "0.5"}]
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "phase_config": pc},
            task_id="test-phase-004",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        # LoRA URLs should be in activated_loras
        assert "https://example.com/lora1.safetensors" in task.parameters.get("activated_loras", [])
        # Multipliers in phase format
        assert "1.2;0.5" in task.parameters.get("loras_multipliers", "")

    def test_invalid_phase_count_raises(self):
        from source.task_conversion import db_task_to_generation_task
        pc = {
            "num_phases": 4,
            "steps_per_phase": [5, 5, 5, 5],
            "phases": [
                {"guidance_scale": 5.0, "loras": []},
                {"guidance_scale": 3.0, "loras": []},
                {"guidance_scale": 2.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
            "flow_shift": 5.0,
            "sample_solver": "euler",
        }
        with pytest.raises(ValueError, match="num_phases must be 2 or 3"):
            db_task_to_generation_task(
                db_task_params={"prompt": "Test", "phase_config": pc},
                task_id="test-phase-bad",
                task_type="t2v",
                wan2gp_path=WAN2GP_PATH,
            )


# ===================================================================
# TestParsePhaseConfig — Direct tests on parse_phase_config
# ===================================================================

class TestParsePhaseConfig:
    """Direct tests on the parse_phase_config function."""

    def test_basic_2_phase(self):
        from source.task_conversion import parse_phase_config
        result = parse_phase_config(
            phase_config={
                "num_phases": 2,
                "steps_per_phase": [20, 10],
                "phases": [
                    {"guidance_scale": 5.0, "loras": []},
                    {"guidance_scale": 1.0, "loras": []},
                ],
                "flow_shift": 5.0,
                "sample_solver": "euler",
            },
            num_inference_steps=30,
            task_id="test-parse-001",
        )
        assert result["guidance_phases"] == 2
        assert result["guidance_scale"] == 5.0
        assert result["guidance2_scale"] == 1.0
        assert result["flow_shift"] == 5.0
        assert result["switch_threshold"] is not None

    def test_basic_3_phase(self):
        from source.task_conversion import parse_phase_config
        result = parse_phase_config(
            phase_config={
                "num_phases": 3,
                "steps_per_phase": [10, 10, 10],
                "phases": [
                    {"guidance_scale": 5.0, "loras": []},
                    {"guidance_scale": 3.0, "loras": []},
                    {"guidance_scale": 1.0, "loras": []},
                ],
                "flow_shift": 5.0,
                "sample_solver": "euler",
            },
            num_inference_steps=30,
            task_id="test-parse-002",
        )
        assert result["guidance_phases"] == 3
        assert result["guidance3_scale"] == 1.0
        assert result["switch_threshold2"] is not None

    def test_unipc_solver(self):
        from source.task_conversion import parse_phase_config
        result = parse_phase_config(
            phase_config={
                "num_phases": 2,
                "steps_per_phase": [15, 15],
                "phases": [
                    {"guidance_scale": 4.0, "loras": []},
                    {"guidance_scale": 1.0, "loras": []},
                ],
                "flow_shift": 5.0,
                "sample_solver": "unipc",
            },
            num_inference_steps=30,
            task_id="test-parse-unipc",
        )
        assert result["sample_solver"] == "unipc"
        assert result["switch_threshold"] is not None

    def test_dpm_solver(self):
        from source.task_conversion import parse_phase_config
        result = parse_phase_config(
            phase_config={
                "num_phases": 2,
                "steps_per_phase": [15, 15],
                "phases": [
                    {"guidance_scale": 4.0, "loras": []},
                    {"guidance_scale": 1.0, "loras": []},
                ],
                "flow_shift": 5.0,
                "sample_solver": "dpm++",
            },
            num_inference_steps=30,
            task_id="test-parse-dpm",
        )
        assert result["sample_solver"] == "dpm++"
        assert result["switch_threshold"] is not None

    def test_steps_mismatch_raises(self):
        from source.task_conversion import parse_phase_config
        with pytest.raises(ValueError, match="steps_per_phase.*sum to"):
            parse_phase_config(
                phase_config={
                    "num_phases": 2,
                    "steps_per_phase": [10, 10],
                    "phases": [
                        {"guidance_scale": 5.0, "loras": []},
                        {"guidance_scale": 1.0, "loras": []},
                    ],
                    "flow_shift": 5.0,
                    "sample_solver": "euler",
                },
                num_inference_steps=50,  # Mismatch: 10+10 != 50
                task_id="test-parse-mismatch",
            )

    def test_lora_deduplication(self):
        from source.task_conversion import parse_phase_config
        result = parse_phase_config(
            phase_config={
                "num_phases": 2,
                "steps_per_phase": [15, 15],
                "phases": [
                    {"guidance_scale": 5.0, "loras": [
                        {"url": "https://example.com/lora_a.safetensors", "multiplier": "1.0"},
                        {"url": "https://example.com/lora_b.safetensors", "multiplier": "0.8"},
                    ]},
                    {"guidance_scale": 1.0, "loras": [
                        {"url": "https://example.com/lora_a.safetensors", "multiplier": "0.5"},
                    ]},
                ],
                "flow_shift": 5.0,
                "sample_solver": "euler",
            },
            num_inference_steps=30,
            task_id="test-parse-dedup",
        )
        # lora_a appears in both phases, should be deduplicated
        assert len(result["lora_names"]) == 2
        assert "https://example.com/lora_a.safetensors" in result["lora_names"]
        assert "https://example.com/lora_b.safetensors" in result["lora_names"]
        # Multipliers for lora_a: "1.0;0.5", for lora_b: "0.8;0" (not in phase 2)
        assert len(result["lora_multipliers"]) == 2


# ===================================================================
# TestOrchestratorPriority — Orchestrator tasks get boosted priority
# ===================================================================

class TestOrchestratorPriority:
    """Verify orchestrator tasks get priority boost."""

    def test_normal_task_default_priority(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test"},
            task_id="test-priority-001",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.priority == 0

    def test_explicit_priority_passed_through(self):
        from source.task_conversion import db_task_to_generation_task
        task = db_task_to_generation_task(
            db_task_params={"prompt": "Test", "priority": 5},
            task_id="test-priority-002",
            task_type="t2v",
            wan2gp_path=WAN2GP_PATH,
        )
        assert task.priority == 5


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
