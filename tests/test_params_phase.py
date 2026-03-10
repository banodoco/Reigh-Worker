"""Tests for source/core/params/phase.py."""

from source.core.params.phase import PhaseConfig


class TestPhaseConfig:
    def test_empty_check(self):
        pc = PhaseConfig()
        assert pc.is_empty() is True

    def test_not_empty_with_raw_config(self):
        pc = PhaseConfig(raw_config={"phases": []})
        assert pc.is_empty() is False

    def test_to_wgp_format_empty(self):
        pc = PhaseConfig()
        assert pc.to_wgp_format() == {}

    def test_to_wgp_format_excludes_internals(self):
        pc = PhaseConfig(
            parsed_output={
                "guidance_phases": [1.0, 1.0],
                "switch_threshold": 0.5,
                "_patch_config": {"should": "be excluded"},
                "_parsed_phase_config": {"also": "excluded"},
                "_phase_config_model_name": "excluded_too",
            }
        )
        wgp = pc.to_wgp_format()
        assert "guidance_phases" in wgp
        assert "switch_threshold" in wgp
        assert "_patch_config" not in wgp
        assert "_parsed_phase_config" not in wgp
        assert "_phase_config_model_name" not in wgp

    def test_get_lora_info(self):
        pc = PhaseConfig(
            parsed_output={
                "lora_names": ["a.safetensors"],
                "lora_multipliers": ["0.9;0.5"],
                "additional_loras": {"https://hf.co/a.safetensors": "0.9;0.5"},
            }
        )
        info = pc.get_lora_info()
        assert info["lora_names"] == ["a.safetensors"]
        assert info["additional_loras"] == {"https://hf.co/a.safetensors": "0.9;0.5"}

    def test_get_lora_info_empty(self):
        pc = PhaseConfig()
        info = pc.get_lora_info()
        assert info["lora_names"] == []
        assert info["lora_multipliers"] == []
        assert info["additional_loras"] == {}

    def test_validate_raw_without_parsed(self):
        pc = PhaseConfig(raw_config={"phases": [1, 2]}, parsed_output={})
        errors = pc.validate()
        assert any("failed to parse" in e.lower() for e in errors)

    def test_validate_success(self):
        pc = PhaseConfig(raw_config={"phases": [1]}, parsed_output={"guidance_phases": [1.0]})
        assert pc.validate() == []
