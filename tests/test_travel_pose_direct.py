from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_travel_chaining_direct_missing_chain_details():
    import source.task_handlers.travel.chaining as chaining

    ok, msg, output = chaining.handle_travel_chaining_after_wgp(
        wgp_task_params={"task_id": "t1"},
        actual_wgp_output_video_path="/tmp/does-not-matter.mp4",
    )
    assert ok is False
    assert "travel_chain_details" in msg
    assert output is None


def test_travel_orchestrator_helpers_direct():
    import source.task_handlers.travel.orchestrator as to

    assert to._get_model_fps(None) == 24
    assert to._get_frame_step(None) == 4
    assert to._quantize_frames(18, 4) == 17


def test_travel_stitch_direct_missing_refs(tmp_path):
    import source.task_handlers.travel.stitch as stitch

    ok, msg = stitch.handle_travel_stitch_task(
        task_params_from_db={},
        main_output_dir_base=tmp_path,
        stitch_task_id_str="s1",
    )
    assert ok is False
    assert "missing critical orchestrator refs" in msg


def test_pose_utils_transform_all_keypoints_direct():
    import source.utils.pose_utils as pu

    key1 = {"pose_keypoints_2d": [0, 0, 1.0, 10, 10, 1.0]}
    key2 = {"pose_keypoints_2d": [20, 20, 1.0, 30, 30, 1.0]}
    out = pu.transform_all_keypoints(key1, key2, frames=3, interpolation="linear")
    assert len(out) == 3
    assert out[0]["pose_keypoints_2d"][0] == 0
    assert out[-1]["pose_keypoints_2d"][0] == 20
