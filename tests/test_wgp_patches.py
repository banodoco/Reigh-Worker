"""Tests for source/models/wgp/wgp_patches.py.

Each patch function takes a mock WGP module and returns True/False.
Tests verify: (1) patches modify the module correctly, (2) patched
functions route correctly, (3) missing attributes return False gracefully.
"""

import types
from unittest.mock import MagicMock

from source.models.wgp.wgp_patches import (
    apply_qwen_model_routing_patch,
    apply_qwen_lora_directory_patch,
    apply_lora_multiplier_parser_patch,
    apply_lora_key_tolerance_patch,
    apply_all_wgp_patches,
)


def _make_mock_wgp():
    """Create a mock WGP module with the attributes patch functions expect."""
    wgp = types.ModuleType("mock_wgp")
    wgp.load_wan_model = MagicMock(return_value=("proc", "pipe"))
    wgp.get_lora_dir = MagicMock(return_value="/default/lora/dir")
    wgp.get_base_model_type = MagicMock(side_effect=lambda x: x)
    wgp.text_encoder_quantization = False
    wgp.get_loras_preprocessor = MagicMock(return_value=None)
    wgp.parse_loras_multipliers = MagicMock()
    wgp.preparse_loras_multipliers = MagicMock()
    return wgp


class TestQwenModelRoutingPatch:
    def test_non_qwen_model_calls_original(self):
        wgp = _make_mock_wgp()
        orig_load = wgp.load_wan_model
        apply_qwen_model_routing_patch(wgp, "/fake/wan")

        wgp.load_wan_model("model.safetensors", "wan", "wan_14B", {})
        orig_load.assert_called_once()

    def test_replaces_load_wan_model(self):
        wgp = _make_mock_wgp()
        orig = wgp.load_wan_model
        apply_qwen_model_routing_patch(wgp, "/fake")
        assert wgp.load_wan_model is not orig

    def test_returns_false_on_missing_attribute(self):
        wgp = types.ModuleType("bad_wgp")
        assert apply_qwen_model_routing_patch(wgp, "/fake") is False


class TestQwenLoraDirectoryPatch:
    def test_non_qwen_model_falls_through(self):
        wgp = _make_mock_wgp()
        orig = wgp.get_lora_dir
        apply_qwen_lora_directory_patch(wgp, "/fake/wan")

        wgp.get_lora_dir("wan_14B")
        orig.assert_called_once_with("wan_14B")

    def test_qwen_model_redirects_to_loras_qwen(self, tmp_path):
        wgp = _make_mock_wgp()
        qwen_dir = tmp_path / "loras_qwen"
        qwen_dir.mkdir()

        apply_qwen_lora_directory_patch(wgp, str(tmp_path))
        assert wgp.get_lora_dir("qwen_image_edit") == str(qwen_dir)

    def test_qwen_without_dir_falls_through(self, tmp_path):
        """If loras_qwen/ doesn't exist, falls back to original."""
        wgp = _make_mock_wgp()
        orig = wgp.get_lora_dir
        apply_qwen_lora_directory_patch(wgp, str(tmp_path))

        wgp.get_lora_dir("qwen_image_edit")
        orig.assert_called_once_with("qwen_image_edit")

    def test_none_model_type_does_not_crash(self):
        wgp = _make_mock_wgp()
        apply_qwen_lora_directory_patch(wgp, "/fake")
        wgp.get_lora_dir(None)  # Should not raise

    def test_returns_false_on_missing_attribute(self):
        wgp = types.ModuleType("bad_wgp")
        assert apply_qwen_lora_directory_patch(wgp, "/fake") is False


class TestLoraMultiplierParserPatch:
    def test_replaces_parse_functions(self):
        """If shared.utils is available, both functions get replaced."""
        wgp = _make_mock_wgp()
        orig_parse = wgp.parse_loras_multipliers
        orig_preparse = wgp.preparse_loras_multipliers

        result = apply_lora_multiplier_parser_patch(wgp)
        if not result:
            return  # shared.utils not available in this env, skip

        assert wgp.parse_loras_multipliers is not orig_parse
        assert wgp.preparse_loras_multipliers is not orig_preparse


class TestLoraKeyTolerancePatch:
    def test_wraps_preprocessor(self):
        """Patched get_loras_preprocessor should return a new function."""
        wgp = _make_mock_wgp()
        result = apply_lora_key_tolerance_patch(wgp)
        assert result is True

        # The patched function should return a preprocessor (not None)
        mock_transformer = MagicMock()
        mock_transformer.named_modules.return_value = [("layer1", MagicMock())]
        preproc = wgp.get_loras_preprocessor(mock_transformer, "wan")
        assert preproc is not None
        assert callable(preproc)

    def test_tolerant_preprocessor_strips_invalid_keys(self):
        """Keys with no valid LoRA suffix should be stripped."""
        wgp = _make_mock_wgp()
        apply_lora_key_tolerance_patch(wgp)

        mock_transformer = MagicMock()
        mock_transformer.named_modules.return_value = [
            ("blocks.0.attn", MagicMock()),
        ]

        preproc = wgp.get_loras_preprocessor(mock_transformer, "wan")
        sd = {
            "blocks.0.attn.lora_down.weight": "valid_tensor",
            "blocks.0.attn.lora_up.weight": "valid_tensor",
            "blocks.0.attn.diff_m": "invalid_suffix",
        }
        result = preproc(sd)
        assert "blocks.0.attn.lora_down.weight" in result
        assert "blocks.0.attn.lora_up.weight" in result
        assert "blocks.0.attn.diff_m" not in result

    def test_returns_false_on_missing_attribute(self):
        wgp = types.ModuleType("bad_wgp")
        assert apply_lora_key_tolerance_patch(wgp) is False


class TestApplyAllPatches:
    def test_returns_dict_with_all_patch_names(self):
        wgp = _make_mock_wgp()
        results = apply_all_wgp_patches(wgp, "/fake/wan")
        expected_keys = {
            "qwen_model_routing", "qwen_lora_directory",
            "lora_multiplier_parser", "qwen_inpainting_lora",
            "lora_key_tolerance", "lora_caching",
        }
        assert set(results.keys()) == expected_keys

    def test_all_values_are_bool(self):
        """Even failed patches should return False, never raise."""
        wgp = _make_mock_wgp()
        results = apply_all_wgp_patches(wgp, "/fake/wan")
        assert all(isinstance(v, bool) for v in results.values())

    def test_basic_patches_succeed(self):
        wgp = _make_mock_wgp()
        results = apply_all_wgp_patches(wgp, "/fake/wan")
        assert results["qwen_model_routing"] is True
        assert results["qwen_lora_directory"] is True
        assert results["lora_key_tolerance"] is True
