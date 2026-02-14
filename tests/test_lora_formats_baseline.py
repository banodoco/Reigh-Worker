"""
Baseline tests for LoRA format resolution through LoRAConfig.

Tests all major LoRA input formats and their code paths through the actual
LoRAConfig.from_params() system.
"""

import pytest

from source.core.params.lora import LoRAConfig, LoRAStatus


class TestFormat1WgpLegacy:
    """Format 1: WGP Legacy — activated_loras list + loras_multipliers string."""

    def test_list_with_space_separated_multipliers(self):
        params = {
            "activated_loras": ["lora1.safetensors", "lora2.safetensors"],
            "loras_multipliers": "1.0 0.8",
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 2
        assert config.entries[0].filename == "lora1.safetensors"
        assert config.entries[1].filename == "lora2.safetensors"
        assert config.entries[0].multiplier == 1.0
        assert config.entries[1].multiplier == 0.8

    def test_csv_string_activated_loras(self):
        params = {
            "activated_loras": "lora1.safetensors,lora2.safetensors",
            "loras_multipliers": "1.0,0.8",
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 2
        assert config.entries[0].filename == "lora1.safetensors"
        assert config.entries[1].filename == "lora2.safetensors"

    def test_phase_config_multipliers_preserved(self):
        params = {
            "activated_loras": ["lora1.safetensors", "lora2.safetensors"],
            "loras_multipliers": "1.0;0;0 0;0.8;1.0",
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 2
        # Phase-config multipliers contain semicolons and stay as strings
        assert config.entries[0].is_phase_config_multiplier()
        assert config.entries[1].is_phase_config_multiplier()


class TestFormat3InternalList:
    """Format 3: Internal List — lora_names + lora_multipliers as lists."""

    def test_float_multipliers(self):
        params = {
            "lora_names": ["lora1.safetensors", "lora2.safetensors"],
            "lora_multipliers": [1.0, 0.8],
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 2
        assert config.entries[0].multiplier == 1.0
        assert config.entries[1].multiplier == 0.8

    def test_phase_config_string_multipliers(self):
        params = {
            "lora_names": ["lora1.safetensors"],
            "lora_multipliers": ["1.0;0;0"],
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 1
        assert config.entries[0].is_phase_config_multiplier()


class TestFormat5AdditionalDict:
    """Format 5: Additional LoRAs Dict — URL to multiplier map."""

    def test_urls_with_multipliers(self):
        params = {
            "additional_loras": {
                "https://example.com/lora1.safetensors": 1.0,
                "https://example.com/lora2.safetensors": 0.8,
            }
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 2
        filenames = {e.filename for e in config.entries}
        assert "lora1.safetensors" in filenames
        assert "lora2.safetensors" in filenames
        assert all(e.status == LoRAStatus.PENDING for e in config.entries)

    def test_merge_with_existing(self):
        params = {
            "activated_loras": ["existing.safetensors"],
            "loras_multipliers": "1.0",
            "additional_loras": {"https://example.com/new.safetensors": 0.8},
        }
        config = LoRAConfig.from_params(params)
        filenames = [e.filename for e in config.entries]
        assert "existing.safetensors" in filenames
        assert "new.safetensors" in filenames


class TestFormat8FeatureFlags:
    """Format 8: Boolean feature flags are tested via the actual lora_validation helpers."""

    def test_empty_params_no_crash(self):
        config = LoRAConfig.from_params({})
        assert len(config.entries) == 0
        assert config.to_wgp_format() == {"activated_loras": [], "loras_multipliers": ""}


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_activated_loras(self):
        params = {"activated_loras": [], "loras_multipliers": ""}
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 0

    @pytest.mark.parametrize(
        "multipliers_str",
        ["1.0 0.8", "1.0,0.8"],
        ids=["space-separated", "comma-separated"],
    )
    def test_multiplier_parsing_variants(self, multipliers_str):
        params = {
            "activated_loras": ["a.safetensors", "b.safetensors"],
            "loras_multipliers": multipliers_str,
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 2

    def test_whitespace_in_csv_loras(self):
        params = {
            "activated_loras": " lora1.safetensors , lora2.safetensors ",
            "loras_multipliers": "1.0, 0.8",
        }
        config = LoRAConfig.from_params(params)
        assert config.entries[0].filename == "lora1.safetensors"
        assert config.entries[1].filename == "lora2.safetensors"

    def test_url_deduplication(self):
        """Same URL in activated_loras and additional_loras is deduplicated."""
        url = "https://example.com/lora.safetensors"
        params = {
            "activated_loras": [url],
            "loras_multipliers": "0.5;0.8",
            "additional_loras": {url: 1.0},
        }
        config = LoRAConfig.from_params(params)
        assert len(config.entries) == 1
        # Phase-config multiplier from activated_loras is preserved
        assert config.entries[0].multiplier == "0.5;0.8"

    def test_missing_multipliers_default_to_1(self):
        params = {"activated_loras": ["lora1.safetensors", "lora2.safetensors"]}
        config = LoRAConfig.from_params(params)
        assert config.entries[0].multiplier == 1.0
        assert config.entries[1].multiplier == 1.0
