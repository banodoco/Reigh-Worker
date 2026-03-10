"""Tests for source/core/params/vace.py."""

from source.core.params.vace import VACEConfig


class TestVACEConfigFromParams:
    def test_all_fields(self):
        cfg = VACEConfig.from_params({
            "video_guide": "/path/guide.mp4",
            "video_mask": "/path/mask.mp4",
            "video_prompt_type": "image2video",
            "control_net_weight": 0.8,
            "control_net_weight2": 0.6,
        })
        assert cfg.guide_path == "/path/guide.mp4"
        assert cfg.mask_path == "/path/mask.mp4"
        assert cfg.prompt_type == "image2video"
        assert cfg.control_weight == 0.8
        assert cfg.control_weight2 == 0.6

    def test_empty_params(self):
        cfg = VACEConfig.from_params({})
        assert cfg.guide_path is None
        assert cfg.mask_path is None
        assert cfg.prompt_type is None
        assert cfg.control_weight is None
        assert cfg.control_weight2 is None

    def test_partial_params(self):
        """Only guide_path set — mask and weights stay None."""
        cfg = VACEConfig.from_params({"video_guide": "/guide.mp4"})
        assert cfg.guide_path == "/guide.mp4"
        assert cfg.mask_path is None
        assert cfg.control_weight is None


class TestVACEConfigToWgpFormat:
    def test_only_set_values_included(self):
        """to_wgp_format should omit None fields entirely."""
        cfg = VACEConfig(guide_path="/guide.mp4", control_weight=0.5)
        wgp = cfg.to_wgp_format()
        assert wgp == {"video_guide": "/guide.mp4", "control_net_weight": 0.5}
        assert "video_mask" not in wgp
        assert "video_prompt_type" not in wgp

    def test_empty_config_returns_empty(self):
        assert VACEConfig().to_wgp_format() == {}

    def test_roundtrip_from_params_to_wgp(self):
        """Params → VACEConfig → WGP format should preserve field names/values."""
        params = {
            "video_guide": "/g.mp4",
            "video_mask": "/m.mp4",
            "video_prompt_type": "KI",
            "control_net_weight": 0.8,
            "control_net_weight2": 0.6,
        }
        wgp = VACEConfig.from_params(params).to_wgp_format()
        for key in params:
            assert wgp[key] == params[key]


class TestVACEConfigValidate:
    def test_always_valid(self):
        """VACEConfig.validate() currently has no failure conditions."""
        assert VACEConfig().validate() == []
        assert VACEConfig(guide_path="/guide.mp4", control_weight=0.8).validate() == []
