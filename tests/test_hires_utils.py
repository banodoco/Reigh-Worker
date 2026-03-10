"""Tests for source/media/video/hires_utils.py."""

import pytest
import torch

from source.media.video.hires_utils import HiresFixHelper


class TestParseConfig:
    """Tests for HiresFixHelper.parse_config."""

    def test_defaults(self):
        """Empty config should return default values."""
        scale, steps, denoise, method = HiresFixHelper.parse_config({})
        assert scale == 1.5
        assert steps == 12
        assert denoise == 0.5
        assert method == "bicubic"

    def test_custom_values(self):
        config = {
            "scale": 2.0,
            "steps": 20,
            "denoise": 0.7,
            "upscale_method": "bilinear",
        }
        scale, steps, denoise, method = HiresFixHelper.parse_config(config)
        assert scale == 2.0
        assert steps == 20
        assert denoise == 0.7
        assert method == "bilinear"

    def test_string_values_coerced(self):
        """String values from JSON should be properly coerced to numbers."""
        config = {"scale": "1.8", "steps": "15", "denoise": "0.3"}
        scale, steps, denoise, method = HiresFixHelper.parse_config(config)
        assert scale == 1.8
        assert steps == 15
        assert denoise == 0.3

    def test_partial_config(self):
        """Partial config should fill missing keys with defaults."""
        config = {"scale": 3.0}
        scale, steps, denoise, method = HiresFixHelper.parse_config(config)
        assert scale == 3.0
        assert steps == 12  # default
        assert denoise == 0.5  # default
        assert method == "bicubic"  # default


class TestUpscaleLatents:
    """Tests for HiresFixHelper.upscale_latents."""

    def test_basic_upscale(self):
        """1.5x upscale should produce correctly sized output."""
        latents = torch.randn(1, 4, 16, 16)
        result = HiresFixHelper.upscale_latents(latents, scale_factor=1.5)
        assert result.shape == (1, 4, 24, 24)

    def test_2x_upscale(self):
        latents = torch.randn(1, 4, 10, 10)
        result = HiresFixHelper.upscale_latents(latents, scale_factor=2.0)
        assert result.shape == (1, 4, 20, 20)

    def test_nearest_method(self):
        latents = torch.randn(1, 4, 8, 8)
        result = HiresFixHelper.upscale_latents(latents, scale_factor=2.0, method="nearest")
        assert result.shape == (1, 4, 16, 16)

    def test_bilinear_method(self):
        latents = torch.randn(1, 4, 8, 8)
        result = HiresFixHelper.upscale_latents(latents, scale_factor=2.0, method="bilinear")
        assert result.shape == (1, 4, 16, 16)

    def test_preserves_batch_and_channels(self):
        """Batch size and channel count should be unchanged."""
        latents = torch.randn(2, 8, 12, 12)
        result = HiresFixHelper.upscale_latents(latents, scale_factor=1.5)
        assert result.shape[0] == 2
        assert result.shape[1] == 8

    def test_scale_factor_one_preserves_size(self):
        latents = torch.randn(1, 4, 16, 16)
        result = HiresFixHelper.upscale_latents(latents, scale_factor=1.0)
        assert result.shape == latents.shape


class TestAddDenoiseNoise:
    """Tests for HiresFixHelper.add_denoise_noise."""

    def test_zero_denoise_unchanged(self):
        """Denoise strength 0 should return the original latents exactly."""
        gen = torch.Generator().manual_seed(42)
        latents = torch.randn(1, 4, 8, 8)
        result = HiresFixHelper.add_denoise_noise(latents, 0.0, gen)
        torch.testing.assert_close(result, latents)

    def test_full_denoise_is_pure_noise(self):
        """Denoise strength 1.0 should produce pure random noise (no original signal)."""
        gen = torch.Generator().manual_seed(42)
        latents = torch.ones(1, 4, 8, 8)
        result = HiresFixHelper.add_denoise_noise(latents, 1.0, gen)
        # Result should NOT be all ones (it should be noise)
        assert not torch.allclose(result, latents)
        # Verify formula: result = latents * 0 + noise * 1 = noise
        gen2 = torch.Generator().manual_seed(42)
        expected_noise = torch.randn(latents.shape, generator=gen2, device=latents.device, dtype=latents.dtype)
        torch.testing.assert_close(result, expected_noise)

    def test_partial_denoise_blends(self):
        """Partial denoise should blend original and noise."""
        gen = torch.Generator().manual_seed(42)
        latents = torch.ones(1, 4, 8, 8) * 5.0
        result = HiresFixHelper.add_denoise_noise(latents, 0.5, gen)
        # Result should be between pure signal and pure noise
        assert result.shape == latents.shape
        # Mean should be shifted from 5.0 toward 0.0 (noise has ~0 mean)
        assert result.mean().item() < 5.0

    def test_output_shape_matches_input(self):
        gen = torch.Generator().manual_seed(42)
        latents = torch.randn(2, 8, 16, 16)
        result = HiresFixHelper.add_denoise_noise(latents, 0.5, gen)
        assert result.shape == latents.shape

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce same result."""
        latents = torch.randn(1, 4, 8, 8)
        gen1 = torch.Generator().manual_seed(123)
        result1 = HiresFixHelper.add_denoise_noise(latents, 0.5, gen1)
        gen2 = torch.Generator().manual_seed(123)
        result2 = HiresFixHelper.add_denoise_noise(latents, 0.5, gen2)
        torch.testing.assert_close(result1, result2)


class TestPrintPass2LoraSummary:
    """Tests for HiresFixHelper.print_pass2_lora_summary (smoke tests)."""

    def test_basic_summary_runs(self):
        """Should not raise with valid inputs."""
        HiresFixHelper.print_pass2_lora_summary(
            lora_names=["lora_a", "lora_b"],
            phase_values=["0.8", "0.0"],
            active_count=1,
        )

    def test_all_active(self):
        """All non-zero multipliers."""
        HiresFixHelper.print_pass2_lora_summary(
            lora_names=["a", "b", "c"],
            phase_values=["1.0", "0.5", "0.3"],
            active_count=3,
        )

    def test_empty_lists(self):
        """Empty lora lists should not crash."""
        HiresFixHelper.print_pass2_lora_summary(
            lora_names=[],
            phase_values=[],
            active_count=0,
        )
