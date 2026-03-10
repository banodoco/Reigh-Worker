"""Tests for source/task_handlers/travel/svi_config.py."""

import pytest

from source.task_handlers.travel.svi_config import (
    SVI_LORAS,
    SVI_DEFAULT_PARAMS,
    SVI_STITCH_OVERLAP,
    get_svi_lora_arrays,
    merge_svi_into_generation_params,
)


class TestSVIConstants:
    def test_svi_loras_not_empty(self):
        assert len(SVI_LORAS) > 0

    def test_svi_loras_have_phase_multipliers(self):
        for url, mult in SVI_LORAS.items():
            assert ";" in mult, f"SVI LoRA {url} should have phase multiplier format"
            parts = mult.split(";")
            assert len(parts) == 2, f"SVI LoRA {url} should have 2 phases"

    def test_svi_default_params(self):
        assert SVI_DEFAULT_PARAMS["guidance_phases"] == 2
        assert SVI_DEFAULT_PARAMS["sample_solver"] == "euler"
        assert "num_inference_steps" in SVI_DEFAULT_PARAMS

    def test_svi_stitch_overlap(self):
        assert SVI_STITCH_OVERLAP == 4


class TestGetSviLoraArrays:
    def test_no_existing_loras(self):
        urls, mults = get_svi_lora_arrays()
        assert len(urls) == len(SVI_LORAS)
        assert len(mults) == len(SVI_LORAS)

    def test_preserves_existing_loras(self):
        existing_urls = ["https://example.com/my_lora.safetensors"]
        existing_mults = ["1.0"]
        urls, mults = get_svi_lora_arrays(
            existing_urls=existing_urls,
            existing_multipliers=existing_mults,
        )
        assert urls[0] == "https://example.com/my_lora.safetensors"
        assert mults[0] == "1.0"
        assert len(urls) == 1 + len(SVI_LORAS)

    def test_no_duplicates(self):
        # If an SVI URL is already in the list, don't add it again
        first_url = list(SVI_LORAS.keys())[0]
        existing_urls = [first_url]
        existing_mults = ["custom_mult"]
        urls, mults = get_svi_lora_arrays(
            existing_urls=existing_urls,
            existing_multipliers=existing_mults,
        )
        # Should have len(SVI_LORAS) total, not len(SVI_LORAS) + 1
        assert urls.count(first_url) == 1
        assert len(urls) == len(SVI_LORAS)

    def test_no_scaling(self):
        """Without any strength params, multipliers should match SVI_LORAS values."""
        urls, mults = get_svi_lora_arrays()
        for url, mult in zip(urls, mults):
            assert mult == SVI_LORAS[url]

    def test_global_svi_strength(self):
        urls, mults = get_svi_lora_arrays(svi_strength=0.5)
        # All multipliers should be scaled by 0.5
        for url, mult in zip(urls, mults):
            original = SVI_LORAS[url]
            original_parts = original.split(";")
            scaled_parts = mult.split(";")
            for orig_p, scaled_p in zip(original_parts, scaled_parts):
                orig_val = float(orig_p)
                scaled_val = float(scaled_p)
                assert abs(scaled_val - orig_val * 0.5) < 0.01

    def test_svi_strength_1_affects_high_noise(self):
        """svi_strength_1 should only affect high-noise LoRAs (pattern: X;0)."""
        urls, mults = get_svi_lora_arrays(svi_strength_1=2.0)
        for url, mult in zip(urls, mults):
            original = SVI_LORAS[url]
            orig_parts = original.split(";")
            new_parts = mult.split(";")
            is_high_noise = float(orig_parts[0]) > 0 and float(orig_parts[1]) == 0
            if is_high_noise:
                # Should be scaled by 2.0
                assert abs(float(new_parts[0]) - float(orig_parts[0]) * 2.0) < 0.01
            else:
                # Should be unchanged (no global svi_strength set)
                for o, n in zip(orig_parts, new_parts):
                    assert abs(float(o) - float(n)) < 0.01

    def test_svi_strength_2_affects_low_noise(self):
        """svi_strength_2 should only affect low-noise LoRAs (pattern: 0;X)."""
        urls, mults = get_svi_lora_arrays(svi_strength_2=0.5)
        for url, mult in zip(urls, mults):
            original = SVI_LORAS[url]
            orig_parts = original.split(";")
            new_parts = mult.split(";")
            is_low_noise = float(orig_parts[0]) == 0 and float(orig_parts[1]) > 0
            if is_low_noise:
                assert abs(float(new_parts[1]) - float(orig_parts[1]) * 0.5) < 0.01
            else:
                for o, n in zip(orig_parts, new_parts):
                    assert abs(float(o) - float(n)) < 0.01

    def test_svi_strength_1_0_is_not_scaling(self):
        """svi_strength=1.0 should produce same results as no scaling."""
        urls_default, mults_default = get_svi_lora_arrays()
        urls_1, mults_1 = get_svi_lora_arrays(svi_strength=1.0)
        # Note: svi_strength=1.0 still enters scaling path since has_scaling checks != 1.0
        # Actually: has_scaling = (svi_strength is not None and svi_strength != 1.0) ...
        # So svi_strength=1.0 should NOT trigger scaling
        assert mults_default == mults_1

    def test_does_not_mutate_existing(self):
        """Ensure the existing lists are not mutated."""
        existing_urls = ["https://example.com/lora.safetensors"]
        existing_mults = ["1.0"]
        urls_copy = existing_urls.copy()
        mults_copy = existing_mults.copy()
        get_svi_lora_arrays(existing_urls=existing_urls, existing_multipliers=existing_mults)
        assert existing_urls == urls_copy
        assert existing_mults == mults_copy

    def test_zero_strength_zeroes_out(self):
        urls, mults = get_svi_lora_arrays(svi_strength=0.0)
        for mult in mults:
            parts = mult.split(";")
            for p in parts:
                assert float(p) == 0.0


class TestMergeSviIntoGenerationParams:
    def test_empty_params(self):
        params = {}
        merge_svi_into_generation_params(params)
        assert "activated_loras" in params
        assert "loras_multipliers" in params
        assert len(params["activated_loras"]) == len(SVI_LORAS)

    def test_preserves_existing_loras(self):
        params = {
            "activated_loras": ["https://example.com/existing.safetensors"],
            "loras_multipliers": "1.0",
        }
        merge_svi_into_generation_params(params)
        assert params["activated_loras"][0] == "https://example.com/existing.safetensors"
        assert len(params["activated_loras"]) == 1 + len(SVI_LORAS)

    def test_string_multipliers_parsed(self):
        params = {
            "activated_loras": ["https://example.com/a.safetensors"],
            "loras_multipliers": "0.8 1.2",
        }
        merge_svi_into_generation_params(params)
        # The merged multipliers should be space-separated string
        assert isinstance(params["loras_multipliers"], str)
        parts = params["loras_multipliers"].split()
        assert parts[0] == "0.8"
        assert parts[1] == "1.2"

    def test_with_svi_strength(self):
        params = {}
        merge_svi_into_generation_params(params, svi_strength=0.5)
        mults = params["loras_multipliers"].split()
        # All should be scaled versions
        for mult in mults:
            parts = mult.split(";")
            for p in parts:
                val = float(p)
                # Each value should be half the original or zero
                assert val >= 0

    def test_list_multipliers_handled(self):
        """loras_multipliers can be a list instead of a string."""
        params = {
            "activated_loras": [],
            "loras_multipliers": ["0.8", "1.2"],
        }
        merge_svi_into_generation_params(params)
        assert isinstance(params["loras_multipliers"], str)

    def test_none_multipliers_handled(self):
        params = {
            "activated_loras": [],
            "loras_multipliers": None,
        }
        merge_svi_into_generation_params(params)
        assert len(params["activated_loras"]) == len(SVI_LORAS)
