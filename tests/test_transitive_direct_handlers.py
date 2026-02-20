from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_wgp_orchestrator_direct_sets_output_paths():
    import source.models.wgp.orchestrator as orch

    module_obj = SimpleNamespace()
    orch._set_wgp_output_paths(module_obj, "/tmp/out")
    assert module_obj.save_path == "/tmp/out"
    assert module_obj.image_save_path == "/tmp/out"
    assert module_obj.server_config["save_path"] == "/tmp/out"


def test_create_visualization_direct_missing_required_param(tmp_path):
    import source.task_handlers.create_visualization as cv

    ok, msg = cv.handle_create_visualization_task(
        task_params_from_db={"params": {"output_video_path": "/a.mp4"}},
        main_output_dir_base=tmp_path,
        viz_task_id_str="v1",
    )
    assert ok is False
    assert "Missing required parameter" in msg


def test_edit_video_orchestrator_direct_calculate_keeper_segments():
    import source.task_handlers.edit_video_orchestrator as evo

    keepers = evo._calculate_keeper_segments(
        portions=[{"start_frame": 10, "end_frame": 19}, {"start_frame": 30, "end_frame": 39}],
        total_frames=50,
        replace_mode=False,
    )
    assert keepers[0]["start_frame"] == 0
    assert keepers[0]["end_frame"] == 9
    assert keepers[-1]["start_frame"] == 40
    assert keepers[-1]["end_frame"] == 49


def test_inpaint_frames_direct_missing_video_path(tmp_path):
    import source.task_handlers.inpaint_frames as ip

    ok, msg = ip._handle_inpaint_frames_task(
        task_params_from_db={"inpaint_start_frame": 1, "inpaint_end_frame": 2},
        main_output_dir_base=tmp_path,
        task_id="ip1",
        task_queue=object(),
    )
    assert ok is False
    assert "video_path is required" in msg


def test_join_orchestrator_direct_missing_payload(tmp_path):
    import source.task_handlers.join.orchestrator as jo

    ok, msg = jo._handle_join_clips_orchestrator_task(
        task_params_from_db={},
        main_output_dir_base=tmp_path,
        orchestrator_task_id_str="j1",
        orchestrator_project_id=None,
    )
    assert ok is False
    assert "orchestrator_details missing" in msg


def test_join_generation_direct_missing_ending_path(tmp_path):
    import source.task_handlers.join.generation as jg

    ok, msg = jg.handle_join_clips_task(
        task_params_from_db={"starting_video_path": "/tmp/a.mp4"},
        main_output_dir_base=tmp_path,
        task_id="g1",
        task_queue=object(),
    )
    assert ok is False
    assert "ending_video_path is required" in msg


def test_join_final_stitch_direct_chain_mode_missing_transition_ids(tmp_path):
    import source.task_handlers.join.final_stitch as fs

    ok, msg = fs.handle_join_final_stitch(
        task_params_from_db={"chain_mode": True, "transition_task_ids": []},
        main_output_dir_base=tmp_path,
        task_id="s1",
    )
    assert ok is False
    assert "no transition_task_ids" in msg


def test_join_vlm_enhancement_direct_invalid_quad_short_circuit():
    import source.task_handlers.join.vlm_enhancement as ve

    prompts = ve._generate_vlm_prompts_for_joins(
        image_quads=[(None, None, None, None)],
        base_prompt="prompt",
        vlm_device="cpu",
    )
    assert prompts == [None]
