from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_debug_task_tasks_workers_commands_direct(monkeypatch, capsys):
    import debug.commands.task as task
    import debug.commands.tasks as tasks
    import debug.commands.workers as workers

    monkeypatch.setattr(task.Formatter, "format_task", lambda *_args, **_kwargs: "task-ok")
    monkeypatch.setattr(tasks.Formatter, "format_tasks_summary", lambda *_args, **_kwargs: "tasks-ok")
    monkeypatch.setattr(workers.Formatter, "format_workers_summary", lambda *_args, **_kwargs: "workers-ok")

    client = SimpleNamespace(
        get_task_info=lambda _task_id: {"id": "t1"},
        get_recent_tasks=lambda **_kwargs: {"count": 1},
        get_workers_summary=lambda **_kwargs: {"count": 1},
    )
    task.run(client, "t1", {"format": "text"})
    tasks.run(client, {"format": "text"})
    workers.run(client, {"format": "text"})
    out = capsys.readouterr().out
    assert "task-ok" in out
    assert "tasks-ok" in out
    assert "workers-ok" in out


def test_heartbeat_guardian_direct_core_helpers(monkeypatch):
    import heartbeat_guardian as hg

    monkeypatch.setattr(hg.os, "kill", lambda *_args, **_kwargs: None)
    assert hg.check_process_alive(123) is True

    logs = hg.collect_logs_from_queue(SimpleNamespace(get_nowait=lambda: (_ for _ in ()).throw(Exception("empty"))))
    assert logs == []


def test_structure_generation_direct_empty_processed_raises(monkeypatch, tmp_path):
    import source.media.structure.generation as sg

    monkeypatch.setattr(sg, "load_structure_video_frames", lambda *_args, **_kwargs: [np.zeros((8, 8, 3), dtype=np.uint8)])
    monkeypatch.setattr(sg, "process_structure_frames", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError):
        sg.create_structure_guidance_video(
            structure_video_path="in.mp4",
            max_frames_needed=1,
            target_resolution=(8, 8),
            target_fps=16,
            output_path=tmp_path / "out.mp4",
        )


def test_structure_segments_direct_overlap_math():
    import source.media.structure.segments as ss

    assert ss.segment_has_structure_overlap(
        segment_index=1,
        segment_frames_expanded=[10, 10, 10],
        frame_overlap_expanded=[2, 2],
        structure_videos=[{"start_frame": 12, "end_frame": 18}],
    ) is True
    start, frames = ss.calculate_segment_guidance_position(2, [10, 10, 10])
    assert (start, frames) == (20, 10)


def test_frame_extraction_direct_retries_then_success(monkeypatch):
    import source.media.video.frame_extraction as fe

    class _CapClosed:
        def isOpened(self):
            return False

        def release(self):
            pass

    class _CapOpen:
        def __init__(self):
            self._reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fe.cv2.CAP_PROP_FRAME_COUNT:
                return 2
            return 0

        def set(self, *_args, **_kwargs):
            return None

        def read(self):
            self._reads += 1
            if self._reads <= 2:
                return True, np.zeros((4, 4, 3), dtype=np.uint8)
            return False, None

        def release(self):
            pass

    caps = [_CapClosed(), _CapOpen()]
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda *_args, **_kwargs: caps.pop(0))
    monkeypatch.setattr(fe.time, "sleep", lambda *_args, **_kwargs: None)

    frames = fe.extract_frames_from_video("x.mp4")
    assert len(frames) == 2


def test_visualization_comparison_direct_unknown_layout(monkeypatch):
    import source.media.visualization.comparison as cmp

    class _Clip:
        def __init__(self, _path):
            self.duration = 1.0

        def close(self):
            pass

    moviepy_editor = types.ModuleType("moviepy.editor")
    moviepy_editor.VideoFileClip = _Clip
    monkeypatch.setitem(sys.modules, "moviepy.editor", moviepy_editor)
    monkeypatch.setattr(cmp, "_apply_video_treatment", lambda clip, **_kwargs: clip)

    with pytest.raises(ValueError):
        cmp.create_travel_visualization(
            output_video_path="out.mp4",
            structure_video_path="struct.mp4",
            guidance_video_path=None,
            input_image_paths=[],
            segment_frames=[],
            layout="bad_layout",
            show_guidance=False,
        )
