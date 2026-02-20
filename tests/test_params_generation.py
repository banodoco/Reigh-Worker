"""Tests for source/core/params/generation.py."""

import pytest

from source.core.params.generation import GenerationConfig


class TestGenerationConfigFromParams:
    def test_basic(self):
        cfg = GenerationConfig.from_params({
            "prompt": "a dog",
            "resolution": "960x544",
            "video_length": 81,
            "seed": 42,
        })
        assert cfg.prompt == "a dog"
        assert cfg.resolution == "960x544"
        assert cfg.video_length == 81
        assert cfg.seed == 42

    def test_image_aliases(self):
        """'image' and 'image_url' are aliases for image_start."""
        cfg1 = GenerationConfig.from_params({"image": "img.png"})
        assert cfg1.image_start == "img.png"

        cfg2 = GenerationConfig.from_params({"image_url": "http://img.png"})
        assert cfg2.image_start == "http://img.png"

    def test_image_start_takes_precedence(self):
        cfg = GenerationConfig.from_params({"image_start": "a.png", "image": "b.png"})
        assert cfg.image_start == "a.png"

    def test_steps_alias(self):
        cfg = GenerationConfig.from_params({"steps": 30})
        assert cfg.num_inference_steps == 30

    def test_denoising_aliases(self):
        cfg1 = GenerationConfig.from_params({"denoise_strength": 0.7})
        assert cfg1.denoising_strength == 0.7

        cfg2 = GenerationConfig.from_params({"strength": 0.5})
        assert cfg2.denoising_strength == 0.5

    def test_empty(self):
        cfg = GenerationConfig.from_params({})
        assert cfg.prompt == ""
        assert cfg.resolution is None


class TestGenerationConfigToWgpFormat:
    def test_includes_set_values(self):
        cfg = GenerationConfig(prompt="test", seed=42, video_length=81)
        wgp = cfg.to_wgp_format()
        assert wgp["prompt"] == "test"
        assert wgp["seed"] == 42
        assert wgp["video_length"] == 81

    def test_excludes_none(self):
        cfg = GenerationConfig(prompt="test")
        wgp = cfg.to_wgp_format()
        assert "resolution" not in wgp
        assert "seed" not in wgp


class TestGenerationConfigValidate:
    def test_valid_video_length(self):
        cfg = GenerationConfig(video_length=81)  # (81-1) % 4 == 0
        assert cfg.validate() == []

    def test_invalid_video_length(self):
        cfg = GenerationConfig(video_length=80)  # (80-1) % 4 == 3
        errors = cfg.validate()
        assert any("4N+1" in e for e in errors)

    def test_none_video_length_ok(self):
        cfg = GenerationConfig()
        assert cfg.validate() == []
