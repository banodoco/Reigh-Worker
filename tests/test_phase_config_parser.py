"""Tests for source/core/params/phase_config_parser.py."""

import pytest

from source.core.params.phase_config_parser import parse_phase_config, DIFFUSION_TIMESTEP_SCALE


class TestBasicParsing:
    def test_two_phase_euler(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 3.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["guidance_phases"] == 2
        assert result["guidance_scale"] == 3.0
        assert result["guidance2_scale"] == 1.0
        assert result["switch_threshold"] is not None
        assert result["switch_threshold2"] is None
        assert result["flow_shift"] == 5.0
        assert result["sample_solver"] == "euler"

    def test_three_phase_euler(self):
        config = {
            "num_phases": 3,
            "steps_per_phase": [2, 2, 2],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 3.0, "loras": []},
                {"guidance_scale": 2.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["guidance_phases"] == 3
        assert result["guidance_scale"] == 3.0
        assert result["guidance2_scale"] == 2.0
        assert result["guidance3_scale"] == 1.0
        assert result["switch_threshold"] is not None
        assert result["switch_threshold2"] is not None


class TestSolvers:
    def _make_config(self, solver):
        return {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": solver,
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }

    def test_unipc_solver(self):
        result = parse_phase_config(self._make_config("unipc"), num_inference_steps=6)
        assert result["sample_solver"] == "unipc"
        assert result["switch_threshold"] is not None

    def test_dpmpp_solver(self):
        result = parse_phase_config(self._make_config("dpm++"), num_inference_steps=6)
        assert result["sample_solver"] == "dpm++"

    def test_dpmpp_sde_solver(self):
        result = parse_phase_config(self._make_config("dpm++_sde"), num_inference_steps=6)
        assert result["sample_solver"] == "dpm++_sde"

    def test_unknown_solver_fallback(self):
        result = parse_phase_config(self._make_config("unknown_solver"), num_inference_steps=6)
        assert result["sample_solver"] == "unknown_solver"


class TestValidation:
    def test_steps_mismatch_raises(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        with pytest.raises(ValueError, match="steps_per_phase.*sum"):
            parse_phase_config(config, num_inference_steps=10)

    def test_wrong_phases_count_raises(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        with pytest.raises(ValueError, match="num_phases must be 2 or 3"):
            parse_phase_config(config, num_inference_steps=6)

    def test_auto_corrects_num_phases(self):
        """num_phases=5 but steps_per_phase and phases both have 2 entries."""
        config = {
            "num_phases": 5,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["guidance_phases"] == 2


class TestLoraProcessing:
    def test_single_lora_two_phases(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": 0.9}]},
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": 0.5}]},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["lora_names"] == ["https://example.com/a.safetensors"]
        assert result["lora_multipliers"] == ["0.9;0.5"]
        assert "https://example.com/a.safetensors" in result["additional_loras"]

    def test_lora_absent_in_second_phase_gets_zero(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": 0.9}]},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["lora_multipliers"] == ["0.9;0"]

    def test_no_loras(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["lora_names"] == []
        assert result["lora_multipliers"] == []
        assert result["additional_loras"] == {}

    def test_empty_url_lora_skipped(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": [{"url": "", "multiplier": 0.9}]},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["lora_names"] == []

    def test_duplicate_lora_urls_deduplicated(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": [
                    {"url": "https://example.com/a.safetensors", "multiplier": 0.9},
                    {"url": "https://example.com/a.safetensors", "multiplier": 0.5},
                ]},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert len(result["lora_names"]) == 1

    def test_per_step_multiplier_string(self):
        """Comma-separated multiplier string for per-step control."""
        config = {
            "num_phases": 2,
            "steps_per_phase": [2, 2],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": "0.9,0.5"}]},
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": "0.3,0.1"}]},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=4)
        assert result["lora_multipliers"] == ["0.9,0.5;0.3,0.1"]

    def test_per_step_multiplier_count_mismatch_raises(self):
        config = {
            "num_phases": 2,
            "steps_per_phase": [3, 3],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": "0.9,0.5"}]},
                {"guidance_scale": 1.0, "loras": [{"url": "https://example.com/a.safetensors", "multiplier": 0.3}]},
            ],
        }
        with pytest.raises(ValueError, match="values.*but phase has"):
            parse_phase_config(config, num_inference_steps=6)


class TestSwitchThresholds:
    def test_thresholds_are_floats(self):
        config = {
            "num_phases": 3,
            "steps_per_phase": [2, 2, 2],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert isinstance(result["switch_threshold"], float)
        assert isinstance(result["switch_threshold2"], float)

    def test_threshold_ordering(self):
        """switch_threshold should be > switch_threshold2 (timesteps decrease)."""
        config = {
            "num_phases": 3,
            "steps_per_phase": [2, 2, 2],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
                {"guidance_scale": 1.0, "loras": []},
            ],
        }
        result = parse_phase_config(config, num_inference_steps=6)
        assert result["switch_threshold"] > result["switch_threshold2"]


class TestDiffusionTimestepScale:
    def test_is_positive_integer(self):
        """Timestep scale is used as a denominator and range bound — must be > 0."""
        assert isinstance(DIFFUSION_TIMESTEP_SCALE, int)
        assert DIFFUSION_TIMESTEP_SCALE > 0
