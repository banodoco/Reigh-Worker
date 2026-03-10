"""Tests for source/utils/lora_validation.py."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from source.utils.lora_validation import (
    validate_lora_file,
    check_loras_in_directory,
    _normalize_activated_loras_list,
)


def _fake_safe_open(keys=None):
    """Create a mock context manager for safetensors.torch.safe_open."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.keys.return_value = keys or ["lora_up.weight", "lora_down.weight"]
    return ctx


class TestValidateLoraFile:
    def test_nonexistent(self, tmp_path):
        ok, msg = validate_lora_file(tmp_path / "missing.safetensors", "missing.safetensors")
        assert ok is False
        assert "does not exist" in msg

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.safetensors"
        p.write_bytes(b"")
        ok, msg = validate_lora_file(p, "empty.safetensors")
        assert ok is False
        assert "too small" in msg.lower()

    def test_html_error_page(self, tmp_path):
        """HTML sniffing happens after safetensors parsing — use .bin to hit that path cleanly."""
        p = tmp_path / "lora.bin"
        p.write_bytes(b"<!DOCTYPE html><html><body>Error</body></html>" + b"\x00" * 2_000_000)
        ok, msg = validate_lora_file(p, "lora.bin")
        assert ok is False
        assert "HTML" in msg

    def test_valid_size(self, tmp_path):
        """A file within a valid size range passes basic checks."""
        p = tmp_path / "lora.safetensors"
        p.write_bytes(b"\x00" * 2_000_000)
        with patch("safetensors.torch.safe_open", return_value=_fake_safe_open()):
            ok, msg = validate_lora_file(p, "lora.safetensors")
        assert ok is True
        assert "validated" in msg.lower()

    def test_too_small(self, tmp_path):
        p = tmp_path / "lora.safetensors"
        p.write_bytes(b"\x00" * 100)  # 100 bytes - way too small
        ok, msg = validate_lora_file(p, "lora.safetensors")
        assert ok is False
        assert "too small" in msg.lower()

    def test_valid_non_safetensors(self, tmp_path):
        """A .bin file within valid size range passes without safetensors parsing."""
        p = tmp_path / "lora.bin"
        p.write_bytes(b"\x00" * 2_000_000)
        ok, msg = validate_lora_file(p, "lora.bin")
        assert ok is True
        assert "validated" in msg.lower()


class TestCheckLorasInDirectory:
    def test_nonexistent_dir(self, tmp_path):
        result = check_loras_in_directory(tmp_path / "nonexistent")
        assert "error" in result

    def test_empty_dir(self, tmp_path):
        result = check_loras_in_directory(tmp_path)
        assert result["total_files"] == 0

    def test_dir_with_valid_lora(self, tmp_path):
        p = tmp_path / "my_lora.safetensors"
        p.write_bytes(b"\x00" * 2_000_000)
        with patch("safetensors.torch.safe_open", return_value=_fake_safe_open()):
            result = check_loras_in_directory(tmp_path)
        assert result["total_files"] >= 1
        assert result["valid_files"] >= 1
        assert result["invalid_files"] == 0


class TestNormalizeActivatedLorasList:
    def test_list_passthrough(self):
        assert _normalize_activated_loras_list(["a", "b"]) == ["a", "b"]

    def test_string_splitting(self):
        result = _normalize_activated_loras_list("a,b,c")
        assert result == ["a", "b", "c"]

    def test_empty_list(self):
        assert _normalize_activated_loras_list([]) == []

    def test_empty_string(self):
        assert _normalize_activated_loras_list("") == []

    def test_single_item_string(self):
        result = _normalize_activated_loras_list("lora_a.safetensors")
        assert result == ["lora_a.safetensors"]

    def test_whitespace_stripped(self):
        result = _normalize_activated_loras_list(" a , b , c ")
        assert result == ["a", "b", "c"]


class TestApplySpecialLoraSettings:
    """Tests for _apply_special_lora_settings."""

    def test_steps_from_task_params(self):
        """When 'steps' is in task_params_dict, use it."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {"activated_loras": [], "loras_multipliers": ""}
        task_params = {"steps": 8}

        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict=task_params,
        )
        assert ui_defaults["num_inference_steps"] == 8

    def test_steps_from_num_inference_steps(self):
        """When 'num_inference_steps' is in task_params_dict but not 'steps', use it."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {"activated_loras": [], "loras_multipliers": ""}
        task_params = {"num_inference_steps": 12}

        _apply_special_lora_settings(
            task_id="t1",
            lora_type="LightI2X",
            lora_basename="light.safetensors",
            default_steps=6,
            guidance_scale=2.0,
            flow_shift=3.0,
            ui_defaults=ui_defaults,
            task_params_dict=task_params,
        )
        assert ui_defaults["num_inference_steps"] == 12

    def test_default_steps_used_when_no_override(self):
        """When neither 'steps' nor 'num_inference_steps' is present, use default."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {"activated_loras": [], "loras_multipliers": ""}
        task_params = {}

        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=7.0,
            ui_defaults=ui_defaults,
            task_params_dict=task_params,
        )
        assert ui_defaults["num_inference_steps"] == 4

    def test_guidance_and_flow_shift_set(self):
        """guidance_scale and flow_shift are set on ui_defaults."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {"activated_loras": [], "loras_multipliers": ""}
        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.5,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict={},
        )
        assert ui_defaults["guidance_scale"] == 1.5
        assert ui_defaults["flow_shift"] == 5.0

    def test_tea_cache_set_when_provided(self):
        """tea_cache_setting is set when not None."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {"activated_loras": [], "loras_multipliers": ""}
        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict={},
            tea_cache_setting=0.15,
        )
        assert ui_defaults["tea_cache_setting"] == 0.15

    def test_tea_cache_not_set_when_none(self):
        """tea_cache_setting is not set when None."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {"activated_loras": [], "loras_multipliers": ""}
        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict={},
            tea_cache_setting=None,
        )
        assert "tea_cache_setting" not in ui_defaults

    def test_lora_appended_to_activated_list(self):
        """LoRA basename is appended to activated_loras."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {
            "activated_loras": ["existing.safetensors"],
            "loras_multipliers": "1.0",
        }
        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict={},
        )
        assert "causvid.safetensors" in ui_defaults["activated_loras"]
        assert "existing.safetensors" in ui_defaults["activated_loras"]

    def test_lora_not_duplicated(self):
        """If lora_basename already in list, it is not added again."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {
            "activated_loras": ["causvid.safetensors"],
            "loras_multipliers": "1.0",
        }
        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict={},
        )
        assert ui_defaults["activated_loras"].count("causvid.safetensors") == 1

    def test_multipliers_padded_to_match_loras(self):
        """loras_multipliers is padded with '1.0' entries if shorter than activated_loras."""
        from source.utils.lora_validation import _apply_special_lora_settings

        ui_defaults = {
            "activated_loras": [],
            "loras_multipliers": "",
        }
        _apply_special_lora_settings(
            task_id="t1",
            lora_type="CausVid",
            lora_basename="causvid.safetensors",
            default_steps=4,
            guidance_scale=1.0,
            flow_shift=5.0,
            ui_defaults=ui_defaults,
            task_params_dict={},
        )
        # Should have 1 lora and 1 multiplier
        assert len(ui_defaults["activated_loras"]) == 1
        mults = ui_defaults["loras_multipliers"].split()
        assert len(mults) == 1
        assert mults[0] == "1.0"
