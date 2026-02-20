"""Tests for source/core/params/phase_multiplier_utils.py."""

import pytest

from source.core.params.phase_multiplier_utils import (
    LIGHTNING_PATTERNS,
    is_lightning_lora,
    parse_phase_multiplier,
    convert_to_phase_format,
    format_phase_multipliers,
    extract_phase_values,
    get_phase_loras,
)


class TestIsLightningLora:
    def test_empty_string(self):
        assert is_lightning_lora("") is False

    def test_none_value(self):
        assert is_lightning_lora(None) is False

    def test_lightning_keyword(self):
        assert is_lightning_lora("Qwen-Image-Edit-Lightning-8steps.safetensors") is True

    def test_distill_keyword(self):
        assert is_lightning_lora("model-distill-v2.safetensors") is True

    def test_turbo_keyword(self):
        assert is_lightning_lora("turbo_lora.safetensors") is True

    def test_fast_keyword(self):
        assert is_lightning_lora("fast_generation.safetensors") is True

    def test_speed_keyword(self):
        assert is_lightning_lora("speed_optimized.safetensors") is True

    def test_accelerator_keyword(self):
        assert is_lightning_lora("my_accelerator.safetensors") is True

    def test_case_insensitive(self):
        assert is_lightning_lora("LIGHTNING_LORA.safetensors") is True
        assert is_lightning_lora("Turbo_Model.safetensors") is True

    def test_non_lightning(self):
        assert is_lightning_lora("style_transfer.safetensors") is False
        assert is_lightning_lora("detail_enhance.safetensors") is False

    def test_all_patterns_present(self):
        for pattern in LIGHTNING_PATTERNS:
            assert is_lightning_lora(f"test_{pattern}_lora.safetensors") is True


class TestParsePhaseMultiplier:
    def test_empty_string(self):
        result, valid = parse_phase_multiplier("", 2)
        assert result == [1.0, 1.0]
        assert valid is True

    def test_none_value(self):
        result, valid = parse_phase_multiplier(None, 2)
        assert result == [1.0, 1.0]
        assert valid is True

    def test_simple_value_two_phases(self):
        result, valid = parse_phase_multiplier("1.0", 2)
        assert result == [1.0, 1.0]
        assert valid is True

    def test_simple_value_three_phases(self):
        result, valid = parse_phase_multiplier("0.8", 3)
        assert result == [0.8, 0.8, 0.8]
        assert valid is True

    def test_phase_format_two_phases(self):
        result, valid = parse_phase_multiplier("1.0;0.5", 2)
        assert result == [1.0, 0.5]
        assert valid is True

    def test_phase_format_three_phases(self):
        result, valid = parse_phase_multiplier("1.0;0.5;0.3", 3)
        assert result == [1.0, 0.5, 0.3]
        assert valid is True

    def test_missing_trailing_value(self):
        result, valid = parse_phase_multiplier("1.0;", 2)
        assert result == [1.0, 0.0]
        assert valid is True

    def test_missing_leading_value(self):
        result, valid = parse_phase_multiplier(";0.5", 2)
        assert result == [0.0, 0.5]
        assert valid is True

    def test_fills_missing_phases(self):
        result, valid = parse_phase_multiplier("1.0", 1)
        assert result == [1.0]
        assert valid is True

    def test_truncates_extra_phases(self):
        result, valid = parse_phase_multiplier("1.0;0.5;0.3;0.1", 2)
        assert result == [1.0, 0.5]
        assert valid is True

    def test_fills_missing_when_fewer_parts(self):
        result, valid = parse_phase_multiplier("1.0;", 3)
        assert result == [1.0, 0.0, 0.0]
        assert valid is True

    def test_invalid_simple_value_raises(self):
        with pytest.raises(ValueError, match="Invalid multiplier"):
            parse_phase_multiplier("not_a_number", 2)

    def test_invalid_phase_value_raises(self):
        with pytest.raises(ValueError, match="Invalid phase 2 multiplier"):
            parse_phase_multiplier("1.0;bad", 2)

    def test_invalid_with_lora_name_in_error(self):
        with pytest.raises(ValueError, match="for LoRA 'my_lora'"):
            parse_phase_multiplier("bad", 2, lora_name="my_lora")

    def test_whitespace_handling(self):
        result, valid = parse_phase_multiplier("  1.0 ; 0.5  ", 2)
        assert result == [1.0, 0.5]
        assert valid is True

    def test_zero_values(self):
        result, valid = parse_phase_multiplier("0;0", 2)
        assert result == [0.0, 0.0]
        assert valid is True


class TestConvertToPhaseFormat:
    def test_already_phase_format(self):
        assert convert_to_phase_format("1.0;0.5", "any.safetensors", 2) == "1.0;0.5"

    def test_standard_lora(self):
        assert convert_to_phase_format("1.0", "style.safetensors", 2) == "1.0;1.0"

    def test_lightning_lora_auto_detect(self):
        assert convert_to_phase_format("1.0", "Lightning-8steps.safetensors", 2) == "1.0;0"

    def test_lightning_lora_auto_detect_disabled(self):
        assert convert_to_phase_format("1.0", "Lightning.safetensors", 2, auto_detect_lightning=False) == "1.0;1.0"

    def test_three_phases_standard(self):
        assert convert_to_phase_format("0.8", "style.safetensors", 3) == "0.8;0.8;0.8"

    def test_three_phases_lightning(self):
        assert convert_to_phase_format("1.0", "turbo.safetensors", 3) == "1.0;0;0"


class TestFormatPhaseMultipliers:
    def test_mixed_loras(self):
        result = format_phase_multipliers(
            ["Lightning.safetensors", "style.safetensors"],
            ["1.0", "1.1"],
            num_phases=2,
        )
        assert result == ["1.0;0", "1.1;1.1"]

    def test_all_standard(self):
        result = format_phase_multipliers(
            ["a.safetensors", "b.safetensors"],
            ["0.9", "1.2"],
            num_phases=2,
        )
        assert result == ["0.9;0.9", "1.2;1.2"]

    def test_already_phase_format(self):
        result = format_phase_multipliers(
            ["a.safetensors"],
            ["1.0;0.5"],
            num_phases=2,
        )
        assert result == ["1.0;0.5"]

    def test_more_multipliers_than_names(self):
        result = format_phase_multipliers(
            ["a.safetensors"],
            ["1.0", "0.8"],
            num_phases=2,
        )
        # Second multiplier has empty lora_name so not lightning
        assert result == ["1.0;1.0", "0.8;0.8"]

    def test_empty_inputs(self):
        result = format_phase_multipliers([], [], num_phases=2)
        assert result == []


class TestExtractPhaseValues:
    def test_phase_0(self):
        result = extract_phase_values(["1.0;0", "1.1;1.2"], phase_index=0)
        assert result == ["1.0", "1.1"]

    def test_phase_1(self):
        result = extract_phase_values(["1.0;0", "1.1;1.2"], phase_index=1)
        assert result == ["0.0", "1.2"]

    def test_simple_multiplier(self):
        result = extract_phase_values(["1.0", "0.8"], phase_index=0)
        assert result == ["1.0", "0.8"]

    def test_simple_multiplier_phase_1(self):
        result = extract_phase_values(["1.0", "0.8"], phase_index=1)
        assert result == ["1.0", "0.8"]

    def test_empty_list(self):
        assert extract_phase_values([], phase_index=0) == []

    def test_malformed_input_fallback_phase_0(self):
        result = extract_phase_values(["bad_value"], phase_index=0)
        assert result == ["1.0"]

    def test_malformed_input_fallback_phase_1(self):
        result = extract_phase_values(["bad_value"], phase_index=1)
        assert result == ["0"]


class TestGetPhaseLoras:
    def test_filter_phase_1(self):
        loras, mults = get_phase_loras(
            ["lightning.safetensors", "style.safetensors", "detail.safetensors"],
            ["1.0;0", "1.1;1.2", "0;0.8"],
            phase_index=1,
            num_phases=2,
        )
        assert loras == ["style.safetensors", "detail.safetensors"]
        assert mults == ["1.2", "0.8"]

    def test_filter_phase_0(self):
        loras, mults = get_phase_loras(
            ["lightning.safetensors", "style.safetensors"],
            ["1.0;0", "1.1;1.2"],
            phase_index=0,
            num_phases=2,
        )
        assert loras == ["lightning.safetensors", "style.safetensors"]
        assert mults == ["1.0", "1.1"]

    def test_empty_lora_list(self):
        loras, mults = get_phase_loras([], [], phase_index=0)
        assert loras == []
        assert mults == []

    def test_pads_missing_multipliers(self):
        loras, mults = get_phase_loras(
            ["a.safetensors", "b.safetensors"],
            ["1.0;0.5"],  # Only one multiplier for two loras
            phase_index=0,
            num_phases=2,
        )
        assert loras == ["a.safetensors", "b.safetensors"]
        assert mults == ["1.0", "1.0"]

    def test_string_multipliers(self):
        """When multipliers is a string, it should be parsed."""
        loras, mults = get_phase_loras(
            ["a.safetensors", "b.safetensors"],
            "1.0;0 1.1;1.2",
            phase_index=1,
            num_phases=2,
        )
        assert loras == ["b.safetensors"]
        assert mults == ["1.2"]

    def test_all_zero_phase(self):
        """When all loras have zero multipliers for a phase, return empty."""
        loras, mults = get_phase_loras(
            ["a.safetensors", "b.safetensors"],
            ["1.0;0", "0.5;0"],
            phase_index=1,
            num_phases=2,
        )
        assert loras == []
        assert mults == []
