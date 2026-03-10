from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_comfy_handler_direct_missing_workflow_returns_error(tmp_path):
    import source.models.comfy.comfy_handler as ch

    ok, msg = ch.handle_comfy_task(
        task_params_from_db={},
        main_output_dir_base=tmp_path,
        task_id="c1",
    )
    assert ok is False
    assert "workflow" in msg


def test_comfy_handler_direct_ensure_running_missing_comfy_path(monkeypatch, tmp_path):
    import asyncio
    import source.models.comfy.comfy_handler as ch

    ch._comfy_manager = None
    ch._comfy_startup_failed = False
    monkeypatch.setattr(ch, "COMFY_PATH", str(tmp_path / "does-not-exist"))

    ok = asyncio.run(ch._ensure_comfy_running())
    assert ok is False
    assert ch._comfy_startup_failed is True
