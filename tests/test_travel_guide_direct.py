from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_travel_guide_prepare_ref_direct(tmp_path):
    from source.media.video.travel_guide import prepare_vace_ref_for_segment

    out = prepare_vace_ref_for_segment(
        ref_instruction={"type": "start"},
        segment_processing_dir=Path(tmp_path),
        target_resolution_wh=(512, 512),
        task_id_for_logging="t1",
    )
    assert out is None


def test_vace_frame_utils_validate_range_direct():
    from source.media.video.vace_frame_utils import validate_frame_range

    ok, err = validate_frame_range(
        total_frame_count=100,
        start_frame=20,
        end_frame=40,
        context_frame_count=8,
    )
    assert ok is True
    assert err in ("", None)
