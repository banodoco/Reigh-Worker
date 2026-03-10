from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_vlm_model_download_skips_when_files_present(tmp_path):
    import source.media.vlm.model as vm

    model_files = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "config.json",
        "tokenizer_config.json",
    ]
    for name in model_files:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert vm.download_qwen_vlm_if_needed(tmp_path) == tmp_path


def test_vlm_single_and_transition_batch_validation():
    import source.media.vlm.single_image_prompts as sp
    import source.media.vlm.transition_prompts as tp

    with pytest.raises(ValueError):
        sp.generate_single_image_prompts_batch(["a.png"], [], device="cpu")
    with pytest.raises(ValueError):
        tp.generate_transition_prompts_batch([("a.png", "b.png")], [], device="cpu")
    assert sp.generate_single_image_prompts_batch([], [], device="cpu") == []
    assert tp.generate_transition_prompts_batch([], [], device="cpu") == []


def test_generation_strategies_direct_wrappers():
    import source.models.wgp.generators.generation_strategies as gs

    class _Dummy:
        current_model = "m"

        def _is_t2v(self):
            return True

        def _is_vace(self):
            return True

        def _is_flux(self):
            return True

        def generate(self, **kwargs):
            return kwargs

    d = _Dummy()
    assert gs.generate_t2v(d, prompt="p")["prompt"] == "p"
    assert gs.generate_vace(d, prompt="p", video_guide="g.mp4")["video_guide"] == "g.mp4"
    assert gs.generate_flux(d, prompt="p", images=3)["video_length"] == 3


def test_wgp_init_mixin_short_circuit_when_orchestrator_exists():
    import source.task_handlers.queue.wgp_init as wi

    class _Queue(wi.WgpInitMixin):
        def __init__(self):
            self.orchestrator = object()
            self._orchestrator_init_attempted = False
            self.logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None, error=lambda *_a, **_k: None)

    q = _Queue()
    assert q._ensure_orchestrator() is None


def test_rife_interpolate_direct_missing_params(tmp_path):
    import source.task_handlers.rife_interpolate as rf

    ok, msg = rf.handle_rife_interpolate_task({}, main_output_dir_base=tmp_path, task_id="r1")
    assert ok is False
    assert "Missing required parameters" in msg


def test_segment_processor_direct_non_vace_prompt_type():
    import source.task_handlers.travel.segment_processor as sp

    ctx = sp.TravelSegmentContext(
        task_id="t1",
        segment_idx=0,
        model_name="qwen_image_edit_20B",
        total_frames_for_segment=81,
        parsed_res_wh=(512, 512),
        segment_processing_dir=Path("/tmp"),
        main_output_dir_base=Path("/tmp"),
        orchestrator_details={},
        segment_params={},
        mask_active_frames=False,
        debug_enabled=False,
    )
    proc = sp.TravelSegmentProcessor(ctx)
    assert proc.is_vace_model is False
    assert proc.create_video_prompt_type(mask_video_path=None) == "U"


def test_segment_processor_lightning_name_not_misclassified_as_vace():
    import source.task_handlers.travel.segment_processor as sp

    ctx = sp.TravelSegmentContext(
        task_id="t2",
        segment_idx=0,
        model_name="wan_2_2_i2v_lightning_baseline_2_2_2",
        total_frames_for_segment=81,
        parsed_res_wh=(512, 512),
        segment_processing_dir=Path("/tmp"),
        main_output_dir_base=Path("/tmp"),
        orchestrator_details={},
        segment_params={},
        mask_active_frames=False,
        debug_enabled=False,
    )
    proc = sp.TravelSegmentProcessor(ctx)
    assert proc.is_vace_model is False


def test_visualization_layouts_direct_side_by_side_calls_shared(monkeypatch):
    import source.media.visualization.layouts as lo

    monkeypatch.setattr(lo, "_create_multi_layout", lambda *args, **kwargs: "layout-ok")
    out = lo._create_side_by_side_layout(
        output_clip=object(),
        structure_clip=object(),
        guidance_clip=None,
        input_image_paths=[],
        segment_frames=[],
        segment_prompts=None,
        fps=16,
    )
    assert out == "layout-ok"
