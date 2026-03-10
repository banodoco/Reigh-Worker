"""Tests for source/models/wgp/generators/wgp_params.py."""

import pytest

from source.models.wgp.generators.wgp_params import (
    make_send_cmd,
    build_passthrough_params,
    build_normal_params,
    apply_kwargs_overrides,
)


class TestMakeSendCmd:
    def test_returns_callable(self):
        send_cmd = make_send_cmd()
        assert callable(send_cmd)

    def test_status_command(self):
        """send_cmd handles 'status' without error."""
        send_cmd = make_send_cmd()
        send_cmd("status", "loading model")  # Should not raise

    def test_progress_list(self):
        """send_cmd handles 'progress' with a list."""
        send_cmd = make_send_cmd()
        send_cmd("progress", [50, "halfway done"])

    def test_progress_scalar(self):
        """send_cmd handles 'progress' with a scalar value."""
        send_cmd = make_send_cmd()
        send_cmd("progress", 75)

    def test_output_command(self):
        send_cmd = make_send_cmd()
        send_cmd("output")

    def test_exit_command(self):
        send_cmd = make_send_cmd()
        send_cmd("exit")

    def test_error_command(self):
        send_cmd = make_send_cmd()
        send_cmd("error", "something went wrong")

    def test_info_command(self):
        send_cmd = make_send_cmd()
        send_cmd("info", "some info")

    def test_preview_command(self):
        send_cmd = make_send_cmd()
        send_cmd("preview")

    def test_unknown_command_no_crash(self):
        """Unknown commands don't crash (no-op)."""
        send_cmd = make_send_cmd()
        send_cmd("unknown_command", "data")  # Should not raise


class TestBuildPassthroughParams:
    def test_basic_output_structure(self):
        """Build params with minimal input and verify core keys."""
        result = build_passthrough_params(
            state={"loaded": True},
            current_model="wan2_1_t2v",
            image_mode=0,
            resolved_params={"prompt": "a cat", "seed": 123},
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["state"] == {"loaded": True}
        assert result["model_type"] == "wan2_1_t2v"
        assert result["image_mode"] == 0
        assert result["prompt"] == "a cat"
        assert result["seed"] == 123

    def test_default_values(self):
        """Missing resolved_params get defaults."""
        result = build_passthrough_params(
            state={},
            current_model="test",
            image_mode=0,
            resolved_params={},
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["prompt"] == ""
        assert result["negative_prompt"] == ""
        assert result["resolution"] == "1280x720"
        assert result["video_length"] == 81
        assert result["seed"] == 42
        assert result["batch_size"] == 1

    def test_resolved_params_override_defaults(self):
        """JSON resolved_params override the defaults."""
        result = build_passthrough_params(
            state={},
            current_model="test",
            image_mode=0,
            resolved_params={
                "prompt": "override prompt",
                "resolution": "896x496",
                "video_length": 61,
                "guidance_scale": 3.5,
                "guidance2_scale": 2.0,
            },
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        assert result["prompt"] == "override prompt"
        assert result["resolution"] == "896x496"
        assert result["video_length"] == 61
        assert result["guidance_scale"] == 3.5
        assert result["guidance2_scale"] == 2.0

    def test_core_params_not_overridden_by_json(self):
        """Core params (task, send_cmd, state, model_type) are never overridden."""
        result = build_passthrough_params(
            state={"real": True},
            current_model="real_model",
            image_mode=1,
            resolved_params={
                "state": {"fake": True},
                "model_type": "fake_model",
                "task": {"id": 999},
            },
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )

        # Core params should be preserved (not from resolved_params)
        assert result["state"] == {"real": True}
        assert result["model_type"] == "real_model"
        assert result["task"]["id"] == 1  # Fixed ID, not from resolved

    def test_video_guide_passthrough(self):
        """Video guide and mask are passed through."""
        guide_data = [1, 2, 3]
        mask_data = [4, 5, 6]
        result = build_passthrough_params(
            state={},
            current_model="test",
            image_mode=0,
            resolved_params={},
            video_guide=guide_data,
            video_mask=mask_data,
            video_prompt_type="VM",
            control_net_weight=0.8,
            control_net_weight2=0.5,
        )

        assert result["video_guide"] == guide_data
        assert result["video_mask"] == mask_data
        assert result["video_prompt_type"] == "VM"
        assert result["control_net_weight"] == 0.8
        assert result["control_net_weight2"] == 0.5

    def test_none_video_prompt_type_defaults(self):
        """None video_prompt_type defaults to 'VM'."""
        result = build_passthrough_params(
            state={},
            current_model="test",
            image_mode=0,
            resolved_params={},
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
        )
        assert result["video_prompt_type"] == "VM"
        assert result["control_net_weight"] == 1.0
        assert result["control_net_weight2"] == 1.0


class TestBuildNormalParams:
    def test_basic_output_structure(self):
        """Normal params have core generation fields."""
        result = build_normal_params(
            state={"model_loaded": True},
            current_model="wan2_1_i2v",
            image_mode=1,
            resolved_params={"prompt": "original"},
            prompt="a dog running",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=6.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )

        assert result["model_type"] == "wan2_1_i2v"
        assert result["image_mode"] == 1
        assert result["video_length"] == 81
        assert result["batch_size"] == 1
        assert result["guidance_scale"] == 5.0
        # Prompt from resolved_params takes precedence
        assert result["prompt"] == "original"

    def test_flux_embedded_guidance(self):
        """is_flux=True enables embedded_guidance_scale."""
        result = build_normal_params(
            state={},
            current_model="flux",
            image_mode=0,
            resolved_params={},
            prompt="test",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=6.0,
            is_flux=True,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert result["embedded_guidance_scale"] == 6.0

    def test_non_flux_zero_embedded_guidance(self):
        """is_flux=False sets embedded_guidance_scale to 0.0."""
        result = build_normal_params(
            state={},
            current_model="wan",
            image_mode=0,
            resolved_params={},
            prompt="test",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=6.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert result["embedded_guidance_scale"] == 0.0

    def test_lora_params(self):
        """LoRA params are passed through."""
        result = build_normal_params(
            state={},
            current_model="wan",
            image_mode=0,
            resolved_params={},
            prompt="test",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=["lora_a.safetensors", "lora_b.safetensors"],
            loras_multipliers_str="1.0;0.5 0.8;0.3",
        )
        assert result["activated_loras"] == ["lora_a.safetensors", "lora_b.safetensors"]
        assert result["loras_multipliers"] == "1.0;0.5 0.8;0.3"

    def test_guidance_phases_from_resolved(self):
        """Phase config values come from resolved_params."""
        result = build_normal_params(
            state={},
            current_model="wan",
            image_mode=0,
            resolved_params={
                "guidance_phases": 3,
                "switch_threshold": 300,
                "switch_threshold2": 150,
                "model_switch_phase": 2,
            },
            prompt="test",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert result["guidance_phases"] == 3
        assert result["switch_threshold"] == 300
        assert result["switch_threshold2"] == 150
        assert result["model_switch_phase"] == 2

    def test_v91_params_present(self):
        """v9.1 required parameters are present with defaults."""
        result = build_normal_params(
            state={},
            current_model="wan",
            image_mode=0,
            resolved_params={},
            prompt="test",
            actual_video_length=81,
            actual_batch_size=1,
            actual_guidance=5.0,
            final_embedded_guidance=0.0,
            is_flux=False,
            video_guide=None,
            video_mask=None,
            video_prompt_type=None,
            control_net_weight=None,
            control_net_weight2=None,
            activated_loras=[],
            loras_multipliers_str="",
        )
        assert "alt_guidance_scale" in result
        assert "masking_strength" in result
        assert "motion_amplitude" in result
        assert "pace" in result
        assert "temperature" in result
        assert result["alt_guidance_scale"] == 0.0
        assert result["masking_strength"] == 1.0
        assert result["motion_amplitude"] == 1.0


class TestApplyKwargsOverrides:
    def test_override_existing(self):
        """Override an existing key."""
        params = {"seed": 42, "prompt": "original"}
        apply_kwargs_overrides(params, {"seed": 999})
        assert params["seed"] == 999
        assert params["prompt"] == "original"

    def test_add_new_key(self):
        """Add a key that doesn't exist yet."""
        params = {"prompt": "test"}
        apply_kwargs_overrides(params, {"custom_param": "value"})
        assert params["custom_param"] == "value"

    def test_multiple_overrides(self):
        """Override multiple keys at once."""
        params = {"a": 1, "b": 2, "c": 3}
        apply_kwargs_overrides(params, {"a": 10, "c": 30, "d": 40})
        assert params == {"a": 10, "b": 2, "c": 30, "d": 40}

    def test_empty_kwargs(self):
        """Empty kwargs dict is a no-op."""
        params = {"a": 1}
        apply_kwargs_overrides(params, {})
        assert params == {"a": 1}

    def test_in_place_modification(self):
        """apply_kwargs_overrides modifies dict in place, returns None."""
        params = {"x": 1}
        result = apply_kwargs_overrides(params, {"x": 2})
        assert result is None
        assert params["x"] == 2
