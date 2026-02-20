from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_mask_builder_direct_marks_overlap_and_last_frame(monkeypatch, tmp_path):
    import source.task_handlers.travel.mask_builder as mb

    out = tmp_path / "mask.mp4"
    out.write_bytes(b"x")
    captured: dict[str, object] = {}

    def _create_mask(**kwargs):
        captured.update(kwargs)
        return out

    monkeypatch.setattr(mb, "prepare_output_path", lambda **_kwargs: (out, str(out)))
    monkeypatch.setattr(mb, "create_mask_video_from_inactive_indices", _create_mask)
    monkeypatch.setattr(mb, "get_video_frame_count_and_fps", lambda _p: (81, 16))

    ctx = SimpleNamespace(
        task_id="t1",
        segment_idx=1,
        mask_active_frames=True,
        segment_params={"frame_overlap_from_previous": 3, "is_first_segment": False},
        orchestrator_details={"fps_helpers": 16, "chain_segments": True},
        total_frames_for_segment=81,
        parsed_res_wh=(512, 512),
        main_output_dir_base=tmp_path,
        debug_enabled=True,
    )
    proc = SimpleNamespace(ctx=ctx, is_vace_model=True, _detect_single_image_journey=lambda: False)

    created = mb.create_mask_video(proc)
    assert created == out
    inactive = set(captured["inactive_frame_indices"])
    assert {0, 1, 2}.issubset(inactive)
    assert 80 in inactive


def test_mask_builder_direct_non_vace_non_debug_skips(monkeypatch, tmp_path):
    import source.task_handlers.travel.mask_builder as mb

    monkeypatch.setattr(mb, "prepare_output_path", lambda **_kwargs: (tmp_path / "x.mp4", "db://x"))

    ctx = SimpleNamespace(
        task_id="t2",
        segment_idx=0,
        mask_active_frames=True,
        segment_params={"frame_overlap_from_previous": 0, "is_first_segment": True},
        orchestrator_details={"fps_helpers": 16, "chain_segments": True},
        total_frames_for_segment=20,
        parsed_res_wh=(512, 512),
        main_output_dir_base=tmp_path,
        debug_enabled=False,
    )
    proc = SimpleNamespace(ctx=ctx, is_vace_model=False, _detect_single_image_journey=lambda: True)
    assert mb.create_mask_video(proc) is None
