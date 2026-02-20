"""Tests for source/core/params/structure_guidance.py."""

import pytest

from source.core.params.structure_guidance import (
    StructureVideoEntry,
    StructureGuidanceConfig,
)


class TestStructureVideoEntry:
    def test_defaults(self):
        entry = StructureVideoEntry(path="/video.mp4")
        assert entry.path == "/video.mp4"
        assert entry.start_frame == 0
        assert entry.end_frame is None
        assert entry.treatment == "adjust"
        assert entry.source_start_frame == 0
        assert entry.source_end_frame is None

    def test_from_dict_full(self):
        d = {
            "path": "/video.mp4",
            "start_frame": 10,
            "end_frame": 50,
            "treatment": "clip",
            "source_start_frame": 5,
            "source_end_frame": 45,
        }
        entry = StructureVideoEntry.from_dict(d)
        assert entry.path == "/video.mp4"
        assert entry.start_frame == 10
        assert entry.end_frame == 50
        assert entry.treatment == "clip"
        assert entry.source_start_frame == 5
        assert entry.source_end_frame == 45

    def test_from_dict_defaults(self):
        entry = StructureVideoEntry.from_dict({})
        assert entry.path == ""
        assert entry.start_frame == 0
        assert entry.treatment == "adjust"

    def test_to_dict_roundtrip(self):
        original = {
            "path": "/test.mp4",
            "start_frame": 5,
            "end_frame": 20,
            "treatment": "clip",
            "source_start_frame": 0,
            "source_end_frame": None,
        }
        entry = StructureVideoEntry.from_dict(original)
        result = entry.to_dict()
        assert result == original


class TestStructureGuidanceConfigDefaults:
    def test_default_values(self):
        config = StructureGuidanceConfig()
        assert config.target == "vace"
        assert config.preprocessing == "flow"
        assert config.strength == 1.0
        assert config.canny_intensity == 1.0
        assert config.depth_contrast == 1.0
        assert config.step_window == (0.0, 1.0)
        assert config.frame_policy == "fit"
        assert config.zero_empty_frames is True
        assert config.keep_on_gpu is False
        assert config.videos == []

    def test_has_guidance_false(self):
        config = StructureGuidanceConfig()
        assert config.has_guidance is False

    def test_has_guidance_with_videos(self):
        config = StructureGuidanceConfig()
        config.videos = [StructureVideoEntry(path="/video.mp4")]
        assert config.has_guidance is True

    def test_has_guidance_with_url(self):
        config = StructureGuidanceConfig()
        config._guidance_video_url = "https://example.com/video.mp4"
        assert config.has_guidance is True


class TestStructureGuidanceConfigProperties:
    def test_is_uni3c(self):
        config = StructureGuidanceConfig()
        config.target = "uni3c"
        assert config.is_uni3c is True
        assert config.is_vace is False

    def test_is_vace(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        assert config.is_vace is True
        assert config.is_uni3c is False

    def test_legacy_structure_type_uni3c(self):
        config = StructureGuidanceConfig()
        config.target = "uni3c"
        assert config.legacy_structure_type == "uni3c"

    def test_legacy_structure_type_raw(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "none"
        assert config.legacy_structure_type == "raw"

    def test_legacy_structure_type_flow(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "flow"
        assert config.legacy_structure_type == "flow"

    def test_legacy_structure_type_canny(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "canny"
        assert config.legacy_structure_type == "canny"

    def test_repr(self):
        config = StructureGuidanceConfig()
        r = repr(config)
        assert "StructureGuidanceConfig" in r
        assert "target='vace'" in r


class TestStructureGuidanceConfigNewFormat:
    def test_from_new_format(self):
        params = {
            "structure_guidance": {
                "target": "uni3c",
                "preprocessing": "none",
                "strength": 0.75,
                "step_window": [0.1, 0.9],
                "frame_policy": "stretch",
                "zero_empty_frames": False,
                "keep_on_gpu": True,
                "videos": [{"path": "/v.mp4", "start_frame": 5}],
            }
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "uni3c"
        assert config.preprocessing == "none"
        assert config.strength == 0.75
        assert config.step_window == (0.1, 0.9)
        assert config.frame_policy == "stretch"
        assert config.zero_empty_frames is False
        assert config.keep_on_gpu is True
        assert len(config.videos) == 1
        assert config.videos[0].path == "/v.mp4"
        assert config.videos[0].start_frame == 5

    def test_from_new_format_with_internal_fields(self):
        params = {
            "structure_guidance": {
                "target": "vace",
                "_guidance_video_url": "https://example.com/guide.mp4",
                "_frame_offset": 10,
                "_original_video_url": "https://example.com/orig.mp4",
                "_trimmed_video_url": "https://example.com/trim.mp4",
            }
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config._guidance_video_url == "https://example.com/guide.mp4"
        assert config._frame_offset == 10
        assert config._original_video_url == "https://example.com/orig.mp4"
        assert config._trimmed_video_url == "https://example.com/trim.mp4"

    def test_from_new_format_defaults(self):
        params = {"structure_guidance": {}}
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "flow"
        assert config.strength == 1.0


class TestStructureGuidanceConfigLegacyFormat:
    def test_uni3c_via_use_uni3c(self):
        params = {
            "use_uni3c": True,
            "uni3c_strength": 0.8,
            "uni3c_start_percent": 0.1,
            "uni3c_end_percent": 0.9,
            "uni3c_frame_policy": "stretch",
            "uni3c_guide_video": "https://example.com/guide.mp4",
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "uni3c"
        assert config.strength == 0.8
        assert config.step_window == (0.1, 0.9)
        assert config.frame_policy == "stretch"
        assert config._guidance_video_url == "https://example.com/guide.mp4"

    def test_uni3c_via_structure_type(self):
        params = {"structure_type": "uni3c"}
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "uni3c"
        assert config.preprocessing == "none"

    def test_vace_flow(self):
        params = {
            "structure_type": "flow",
            "structure_video_motion_strength": 0.7,
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "flow"
        assert config.strength == 0.7

    def test_vace_canny(self):
        params = {
            "structure_type": "canny",
            "structure_canny_intensity": 0.5,
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "canny"
        assert config.canny_intensity == 0.5

    def test_vace_depth(self):
        params = {
            "structure_type": "depth",
            "structure_depth_contrast": 0.9,
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "depth"
        assert config.depth_contrast == 0.9

    def test_vace_raw(self):
        params = {"structure_type": "raw"}
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "none"

    def test_single_video_path(self):
        params = {
            "structure_video_path": "/video.mp4",
            "structure_video_treatment": "clip",
        }
        config = StructureGuidanceConfig.from_params(params)
        assert len(config.videos) == 1
        assert config.videos[0].path == "/video.mp4"
        assert config.videos[0].treatment == "clip"

    def test_structure_videos_array(self):
        params = {
            "structure_videos": [
                {"path": "/a.mp4", "start_frame": 0},
                {"path": "/b.mp4", "start_frame": 10},
            ]
        }
        config = StructureGuidanceConfig.from_params(params)
        assert len(config.videos) == 2

    def test_structure_videos_with_type_override(self):
        params = {
            "structure_videos": [
                {"path": "/a.mp4", "structure_type": "canny"},
            ]
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "canny"

    def test_no_structure_type(self):
        params = {}
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "flow"

    def test_legacy_guidance_video_urls(self):
        params = {
            "structure_guidance_video_url": "https://example.com/guide.mp4",
            "structure_guidance_frame_offset": 5,
        }
        config = StructureGuidanceConfig.from_params(params)
        assert config._guidance_video_url == "https://example.com/guide.mp4"
        assert config._frame_offset == 5

    def test_legacy_motion_video_url_fallback(self):
        params = {"structure_motion_video_url": "https://example.com/motion.mp4"}
        config = StructureGuidanceConfig.from_params(params)
        assert config._guidance_video_url == "https://example.com/motion.mp4"

    def test_structure_video_type_alias(self):
        params = {"structure_video_type": "depth"}
        config = StructureGuidanceConfig.from_params(params)
        assert config.target == "vace"
        assert config.preprocessing == "depth"


class TestStructureGuidanceConfigOutput:
    def test_to_vace_params(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "canny"
        config.strength = 0.8
        config.canny_intensity = 0.5
        config.videos = [StructureVideoEntry(path="/v.mp4")]
        config._guidance_video_url = "https://example.com/guide.mp4"
        config._frame_offset = 10

        result = config.to_vace_params()
        assert result["structure_type"] == "canny"
        assert result["structure_video_motion_strength"] == 0.8
        assert result["structure_canny_intensity"] == 0.5
        assert result["structure_video_path"] == "/v.mp4"
        assert result["structure_guidance_video_url"] == "https://example.com/guide.mp4"
        assert result["structure_guidance_frame_offset"] == 10

    def test_to_vace_params_raw_type(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "none"
        result = config.to_vace_params()
        assert result["structure_type"] == "raw"

    def test_to_uni3c_params(self):
        config = StructureGuidanceConfig()
        config.target = "uni3c"
        config.strength = 0.75
        config.step_window = (0.1, 0.9)
        config.frame_policy = "stretch"
        config.zero_empty_frames = False
        config.keep_on_gpu = True
        config._guidance_video_url = "https://example.com/guide.mp4"

        result = config.to_uni3c_params()
        assert result["use_uni3c"] is True
        assert result["uni3c_strength"] == 0.75
        assert result["uni3c_start_percent"] == 0.1
        assert result["uni3c_end_percent"] == 0.9
        assert result["uni3c_frame_policy"] == "stretch"
        assert result["uni3c_zero_empty_frames"] is False
        assert result["uni3c_keep_on_gpu"] is True
        assert result["uni3c_guide_video"] == "https://example.com/guide.mp4"

    def test_to_wgp_format_dispatches_vace(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "flow"
        result = config.to_wgp_format()
        assert "structure_type" in result
        assert "use_uni3c" not in result

    def test_to_wgp_format_dispatches_uni3c(self):
        config = StructureGuidanceConfig()
        config.target = "uni3c"
        result = config.to_wgp_format()
        assert result["use_uni3c"] is True
        assert "structure_type" not in result

    def test_to_segment_payload(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "flow"
        config.strength = 0.8
        config._guidance_video_url = "https://example.com/guide.mp4"
        config.videos = [StructureVideoEntry(path="/v.mp4")]

        payload = config.to_segment_payload(segment_index=2, frame_offset=100)
        sg = payload["structure_guidance"]
        assert sg["target"] == "vace"
        assert sg["preprocessing"] == "flow"
        assert sg["strength"] == 0.8
        assert sg["_guidance_video_url"] == "https://example.com/guide.mp4"
        assert sg["_frame_offset"] == 100
        assert len(sg["videos"]) == 1

    def test_to_segment_payload_no_guidance_url(self):
        config = StructureGuidanceConfig()
        payload = config.to_segment_payload()
        sg = payload["structure_guidance"]
        assert "_guidance_video_url" not in sg
        assert sg["_frame_offset"] == 0


class TestStructureGuidanceConfigValidation:
    def test_valid_config(self):
        config = StructureGuidanceConfig()
        assert config.validate() == []

    def test_invalid_target(self):
        config = StructureGuidanceConfig()
        config.target = "invalid"
        errors = config.validate()
        assert any("Invalid target" in e for e in errors)

    def test_invalid_preprocessing(self):
        config = StructureGuidanceConfig()
        config.target = "vace"
        config.preprocessing = "invalid"
        errors = config.validate()
        assert any("Invalid preprocessing" in e for e in errors)

    def test_negative_strength(self):
        config = StructureGuidanceConfig()
        config.strength = -0.1
        errors = config.validate()
        assert any("non-negative" in e for e in errors)

    def test_step_window_out_of_range(self):
        config = StructureGuidanceConfig()
        config.step_window = (-0.1, 1.0)
        errors = config.validate()
        assert any("Step window" in e for e in errors)

    def test_step_window_end_greater_than_one(self):
        config = StructureGuidanceConfig()
        config.step_window = (0.0, 1.5)
        errors = config.validate()
        assert any("Step window" in e for e in errors)

    def test_step_window_start_greater_than_end(self):
        config = StructureGuidanceConfig()
        config.step_window = (0.8, 0.3)
        errors = config.validate()
        assert any("start must be <= end" in e for e in errors)

    def test_empty_video_path(self):
        config = StructureGuidanceConfig()
        config.videos = [StructureVideoEntry(path="")]
        errors = config.validate()
        assert any("empty path" in e for e in errors)

    def test_uni3c_skips_preprocessing_validation(self):
        """Uni3C target should not validate preprocessing."""
        config = StructureGuidanceConfig()
        config.target = "uni3c"
        config.preprocessing = "invalid"
        errors = config.validate()
        assert not any("Invalid preprocessing" in e for e in errors)
