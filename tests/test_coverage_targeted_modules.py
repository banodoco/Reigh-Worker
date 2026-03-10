"""Targeted direct tests for modules flagged by test_coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _TableChain:
    def __init__(self, data: list[dict] | None = None):
        self._data = data or []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class _FakeSupabase:
    def table(self, _name: str):
        return _TableChain([{"id": "w1", "status": "active", "metadata": {"runpod_id": "rp1"}}])


@dataclass
class _FakeTaskInfo:
    task_id: str = "t1"
    state: dict[str, Any] | None = None
    logs: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []


def test_debug_commands_config_run(capsys):
    from debug.commands import config

    config.run(client=SimpleNamespace(), options={"explain": False})
    out = capsys.readouterr().out
    assert "SYSTEM CONFIGURATION" in out


def test_debug_commands_runpod_run(monkeypatch, capsys):
    from debug.commands import runpod

    fake_runpod = SimpleNamespace(
        api_key="",
        get_pods=lambda: [{"id": "rp1", "name": "gpu_x", "desiredStatus": "RUNNING", "costPerHr": 0.2}],
        terminate_pod=lambda _pid: True,
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "abc123")
    monkeypatch.setitem(__import__("sys").modules, "runpod", fake_runpod)
    client = SimpleNamespace(supabase=_FakeSupabase())

    runpod.run(client=client, options={"terminate": False})
    out = capsys.readouterr().out
    assert "RUNPOD SYNC STATUS" in out


def test_debug_commands_storage_run_handles_missing_dep(capsys):
    from debug.commands import storage

    storage.run(client=SimpleNamespace(supabase=_FakeSupabase()), options={})
    out = capsys.readouterr().out
    assert "Error checking storage" in out or "No active workers" in out


def test_debug_commands_worker_run_calls_formatter(monkeypatch, capsys):
    from debug.commands import worker

    class _Client:
        def get_worker_info(self, *_args, **_kwargs):
            return SimpleNamespace(
                worker_id="w1",
                state={"status": "active"},
                logs=[],
                tasks=[],
            )

    monkeypatch.setattr(worker.Formatter, "format_worker", lambda *_args, **_kwargs: "ok")
    worker.run(client=_Client(), worker_id="w1", options={"format": "text"})
    out = capsys.readouterr().out
    assert "ok" in out


def test_debug_formatter_task_text():
    from debug.formatters import Formatter
    from debug.models import TaskInfo

    info = TaskInfo(task_id="t1", state={"status": "Queued", "task_type": "x"}, logs=[])
    text = Formatter.format_task(info, format_type="text")
    assert "TASK: t1" in text


def test_examples_short_circuit_without_files(tmp_path):
    from examples.inpaint_frames_example import run_inpaint_frames
    from examples.join_clips_example import run_join_clips

    assert run_inpaint_frames(
        video_path=str(tmp_path / "missing.mp4"),
        inpaint_start_frame=1,
        inpaint_end_frame=2,
        output_path=str(tmp_path / "o.mp4"),
        prompt="x",
    ) is False
    assert run_join_clips(
        starting_video_path=str(tmp_path / "missing1.mp4"),
        ending_video_path=str(tmp_path / "missing2.mp4"),
        output_path=str(tmp_path / "o.mp4"),
        prompt="x",
    ) is False


def test_magic_edit_reports_missing_replicate(tmp_path):
    from source.task_handlers import magic_edit
    original = magic_edit.replicate
    magic_edit.replicate = None

    try:
        ok, msg = magic_edit.handle_magic_edit_task(
            task_params_from_db={"image_url": "https://example.com/x.png"},
            main_output_dir_base=Path(tmp_path),
            task_id="m1",
        )
        assert ok is False
        assert "Replicate library" in msg or "REPLICATE_API_TOKEN" in msg
    finally:
        magic_edit.replicate = original


def test_convert_lora_helpers_work():
    import torch
    from scripts.convert_lora_rank import extract_module_pairs, svd_truncate

    sd = {
        "x.lora_A.weight": torch.randn(64, 8),
        "x.lora_B.weight": torch.randn(16, 64),
        "x.alpha": torch.tensor(64.0),
    }
    modules, other = extract_module_pairs(sd)
    assert "x" in modules
    assert other == {}
    new_a, new_b, new_alpha, err = svd_truncate(sd["x.lora_A.weight"], sd["x.lora_B.weight"], target_rank=32, alpha=64.0)
    assert new_a.shape[0] <= 32
    assert new_b.shape[1] <= 32
    assert new_alpha <= 32.0
    assert err >= 0.0


def test_uni3c_frame_strip_placeholder():
    from scripts.uni3c_validation import create_frame_strip

    img = create_frame_strip([], "Test")
    assert img.width > 0
    assert img.height > 0
