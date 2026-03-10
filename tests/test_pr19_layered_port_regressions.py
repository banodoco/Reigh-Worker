"""PR #19 layered-port regression tests.

These tests focus on the compatibility and safety contracts introduced during
PR #19 integration:
- Legacy compatibility model IDs remain loadable.
- Critical task defaults keep resolving to legacy compatibility IDs.
- VACE detection remains strict (no broad keyword matching).
- Uni3C params are forwarded when supported and fail-fast when dropped.
"""

from __future__ import annotations

import io
import json
import os
import sys
import types
from collections import deque
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LEGACY_COMPAT_MODELS = [
    "wan_2_2_i2v_lightning_baseline_2_2_2",
    "wan_2_2_i2v_lightning_baseline_3_3",
    "wan_2_2_i2v_lightning_svi_3_3",
    "wan_2_2_i2v_lightning_svi_endframe",
    "wan_2_2_vace_lightning_baseline_2_2_2",
    "wan_2_2_vace_lightning_baseline_3_3",
    "z_image_img2img",
]


def _make_fake_wgp_with_signature(include_uni3c: bool):
    """Build a fake wgp module with a controllable generate_video signature."""

    if include_uni3c:
        def _generate_video(  # noqa: PLR0913
            task=None,
            send_cmd=None,
            state=None,
            model_type=None,
            prompt=None,
            resolution=None,
            video_length=None,
            batch_size=None,
            seed=None,
            image_mode=None,
            use_uni3c=False,
            uni3c_guide_video=None,
            uni3c_strength=1.0,
            uni3c_start_percent=0.0,
            uni3c_end_percent=1.0,
            uni3c_keep_on_gpu=False,
            uni3c_frame_policy="fit",
            uni3c_zero_empty_frames=True,
            uni3c_blackout_last_frame=False,
            uni3c_controlnet=None,
        ):
            return None
    else:
        def _generate_video(  # noqa: PLR0913
            task=None,
            send_cmd=None,
            state=None,
            model_type=None,
            prompt=None,
            resolution=None,
            video_length=None,
            batch_size=None,
            seed=None,
            image_mode=None,
        ):
            return None

    return types.SimpleNamespace(generate_video=_generate_video)


def _build_stub_orchestrator_instance():
    """Create a WanOrchestrator instance without running __init__."""
    from source.models.wgp.orchestrator import WanOrchestrator

    orch = object.__new__(WanOrchestrator)
    orch.current_model = "t2v"
    orch.smoke_mode = False
    orch.passthrough_mode = False
    orch.state = {
        "model_type": "t2v",
        "loras": [],
        "gen": {"file_list": ["/tmp/pr19-uni3c-out.mp4"], "process_status": ""},
    }

    # Lightweight behavior stubs for generate() dependencies.
    orch._resolve_parameters = lambda _model_type, params: dict(params)
    orch._get_base_model_type = lambda _model_name: "t2v"
    orch._test_vace_module = lambda _model_name: False
    orch._is_vace = lambda: False
    orch._is_flux = lambda: False
    orch._is_qwen = lambda: False
    orch._is_t2v = lambda: True
    orch._is_ltx2 = lambda: False
    orch._resolve_media_path = lambda value: value
    orch._load_image = lambda path, mask=False: path  # noqa: ARG005
    orch._get_or_load_uni3c_controlnet = lambda: "cached-uni3c-controlnet"
    orch._generate_video = lambda **_kwargs: None

    return orch


class TestLegacyCompatModelDefinitions:
    """Compatibility defaults should still load as model definitions."""

    @pytest.mark.parametrize("model_id", LEGACY_COMPAT_MODELS)
    def test_legacy_model_json_loads_into_registry(self, model_id, monkeypatch):
        import source.models.wgp.model_ops as model_ops

        json_path = PROJECT_ROOT / "Wan2GP" / "defaults" / f"{model_id}.json"
        assert json_path.is_file(), f"Missing compatibility default: {json_path}"

        # Validate parseability first to keep failures clear.
        cfg = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)
        assert "model" in cfg

        fake_wgp = types.SimpleNamespace(
            models_def={},
            init_model_def=lambda _key, model_def: dict(model_def, initialized=True),
        )
        monkeypatch.setitem(sys.modules, "wgp", fake_wgp)

        model_ops.load_missing_model_definition(
            orchestrator=types.SimpleNamespace(wan_root=str(PROJECT_ROOT / "Wan2GP")),
            model_key=model_id,
            json_path=str(json_path),
        )

        assert model_id in fake_wgp.models_def
        loaded = fake_wgp.models_def[model_id]
        assert loaded.get("path") == str(json_path)
        assert isinstance(loaded.get("settings"), dict)


class TestCriticalDefaultMappings:
    """Task defaults that must remain stable across PR #19 merge."""

    def test_join_clips_and_inpaint_defaults_preserved(self):
        from source.task_handlers.tasks.task_types import get_default_model

        assert get_default_model("join_clips_segment") == "wan_2_2_vace_lightning_baseline_2_2_2"
        assert get_default_model("inpaint_frames") == "wan_2_2_vace_lightning_baseline_2_2_2"

    def test_z_image_i2i_maps_to_z_image_img2img(self):
        from source.task_handlers.tasks.task_types import get_default_model

        assert get_default_model("z_image_turbo_i2i") == "z_image_img2img"


class TestStrictVaceDetection:
    """Travel segment VACE detection should only trigger on explicit vace names."""

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("wan_2_2_vace_lightning_baseline_2_2_2", True),
            ("wan_2_2_i2v_lightning_baseline_2_2_2", False),
            ("flux2_klein_4b_lightning", False),
            ("VACE_14B", True),
        ],
    )
    def test_detect_vace_model_is_strict(self, model_name, expected, tmp_path):
        from source.task_handlers.travel.segment_processor import (
            TravelSegmentContext,
            TravelSegmentProcessor,
        )

        ctx = TravelSegmentContext(
            task_id="test-task",
            segment_idx=0,
            model_name=model_name,
            total_frames_for_segment=17,
            parsed_res_wh=(896, 512),
            segment_processing_dir=tmp_path,
            main_output_dir_base=tmp_path,
            orchestrator_details={},
            segment_params={},
            mask_active_frames=False,
            debug_enabled=False,
        )

        proc = TravelSegmentProcessor(ctx)
        assert proc.is_vace_model is expected


class TestUni3cRuntimeFiltering:
    """Uni3C options must be kept when supported and fail-fast when unsupported."""

    def test_filter_keeps_uni3c_keys_when_wgp_supports_them(self, monkeypatch):
        from source.models.wgp.orchestrator import WanOrchestrator

        fake_wgp = _make_fake_wgp_with_signature(include_uni3c=True)
        monkeypatch.setitem(sys.modules, "wgp", fake_wgp)

        orch = object.__new__(WanOrchestrator)
        wgp_params = {
            "task": {"id": 1},
            "send_cmd": lambda *_a, **_k: None,
            "state": {},
            "model_type": "t2v",
            "prompt": "p",
            "use_uni3c": True,
            "uni3c_guide_video": "/tmp/guide.mp4",
            "uni3c_strength": 0.75,
            "uni3c_start_percent": 0.1,
            "uni3c_end_percent": 0.9,
            "uni3c_keep_on_gpu": False,
            "uni3c_frame_policy": "fit",
            "uni3c_zero_empty_frames": True,
            "uni3c_blackout_last_frame": False,
            "uni3c_controlnet": object(),
        }

        filtered = WanOrchestrator._filter_wgp_params(orch, wgp_params)

        assert filtered["use_uni3c"] is True
        assert filtered["uni3c_guide_video"] == "/tmp/guide.mp4"
        assert filtered["uni3c_strength"] == 0.75
        assert "uni3c_controlnet" in filtered

    def test_filter_fails_fast_when_uni3c_requested_but_not_supported(self, monkeypatch):
        from source.models.wgp.orchestrator import WanOrchestrator

        fake_wgp = _make_fake_wgp_with_signature(include_uni3c=False)
        monkeypatch.setitem(sys.modules, "wgp", fake_wgp)

        orch = object.__new__(WanOrchestrator)
        wgp_params = {
            "task": {"id": 1},
            "send_cmd": lambda *_a, **_k: None,
            "state": {},
            "model_type": "t2v",
            "prompt": "p",
            "use_uni3c": True,
            "uni3c_guide_video": "/tmp/guide.mp4",
            "uni3c_strength": 0.75,
        }

        with pytest.raises(RuntimeError, match="silent Uni3C degradation"):
            WanOrchestrator._filter_wgp_params(orch, wgp_params)


class TestUni3cGenerateCallPath:
    """End-to-end unit-level generate() wiring for Uni3C params."""

    def test_generate_forwards_uni3c_fields_to_generate_video(self, monkeypatch):
        import source.models.wgp.orchestrator as orch_mod

        fake_wgp = _make_fake_wgp_with_signature(include_uni3c=True)
        monkeypatch.setitem(sys.modules, "wgp", fake_wgp)

        orch = _build_stub_orchestrator_instance()

        monkeypatch.setattr(orch_mod, "prepare_svi_image_refs", lambda _kwargs: None)
        monkeypatch.setattr(
            orch_mod,
            "configure_model_specific_params",
            lambda **kwargs: {
                "image_mode": 0,
                "actual_video_length": kwargs.get("final_video_length", 49),
                "actual_batch_size": kwargs.get("final_batch_size", 1),
                "actual_guidance": kwargs.get("final_guidance_scale", 7.5),
                "video_guide": kwargs.get("video_guide"),
                "video_mask": kwargs.get("video_mask"),
                "video_prompt_type": kwargs.get("video_prompt_type") or "disabled",
                "control_net_weight": kwargs.get("control_net_weight") or 0.0,
                "control_net_weight2": kwargs.get("control_net_weight2") or 0.0,
            },
        )
        monkeypatch.setattr(
            orch_mod,
            "build_normal_params",
            lambda **kwargs: {
                "task": {"id": 1, "params": {}, "repeats": 1},
                "send_cmd": lambda *_a, **_k: None,
                "state": kwargs["state"],
                "model_type": kwargs["current_model"],
                "prompt": kwargs["prompt"],
                "resolution": kwargs["resolved_params"].get("resolution", "896x512"),
                "video_length": kwargs["actual_video_length"],
                "batch_size": kwargs["actual_batch_size"],
                "seed": kwargs["resolved_params"].get("seed", 42),
                "image_mode": kwargs["image_mode"],
            },
        )
        monkeypatch.setattr(orch_mod, "prepare_image_inputs", lambda *args, **kwargs: None)

        captured = {}

        def _fake_run_with_capture(_fn, **kwargs):
            captured.update(kwargs)
            return None, io.StringIO(), io.StringIO(), deque()

        monkeypatch.setattr(orch_mod, "run_with_capture", _fake_run_with_capture)
        monkeypatch.setattr(orch_mod, "extract_output_path", lambda *args, **kwargs: "/tmp/pr19-uni3c-out.mp4")
        monkeypatch.setattr(orch_mod, "log_memory_stats", lambda: None)

        out_path = orch.generate(
            prompt="uni3c test",
            resolution="896x512",
            video_length=17,
            seed=123,
            use_uni3c=True,
            uni3c_guide_video="/tmp/guide.mp4",
            uni3c_strength=0.8,
            uni3c_start_percent=0.1,
            uni3c_end_percent=0.9,
            uni3c_frame_policy="fit",
            uni3c_zero_empty_frames=True,
            uni3c_blackout_last_frame=False,
        )

        assert out_path == "/tmp/pr19-uni3c-out.mp4"
        assert captured.get("use_uni3c") is True
        assert captured.get("uni3c_guide_video") == "/tmp/guide.mp4"
        assert captured.get("uni3c_strength") == 0.8
        assert captured.get("uni3c_start_percent") == 0.1
        assert captured.get("uni3c_end_percent") == 0.9
        assert captured.get("uni3c_controlnet") == "cached-uni3c-controlnet"


class TestPr19NewModelSmokePath:
    """Smoke-mode coverage for a PR #19-introduced model config flow."""

    @pytest.fixture(autouse=True)
    def _chdir_to_wan2gp(self):
        original_cwd = os.getcwd()
        wan_root = str(PROJECT_ROOT / "Wan2GP")
        os.chdir(wan_root)
        yield wan_root
        os.chdir(original_cwd)

    def test_flux2_klein_smoke_generation(self, tmp_path):
        os.environ["HEADLESS_WAN2GP_SMOKE"] = "1"
        try:
            from headless_wgp import WanOrchestrator

            out_dir = tmp_path / "outputs"
            out_dir.mkdir()

            orch = WanOrchestrator(str(PROJECT_ROOT / "Wan2GP"), main_output_dir=str(out_dir))
            assert orch.load_model("flux2_klein_4b") is True

            result = orch.generate(prompt="PR19 Flux2 Klein smoke")
            assert result is not None
            assert Path(result).exists()
        finally:
            os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)

        cfg_path = PROJECT_ROOT / "Wan2GP" / "defaults" / "flux2_klein_4b.json"
        assert cfg_path.is_file(), "Expected PR #19 config flux2_klein_4b.json to exist"
