from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_worker_run_direct_formats_default_view(monkeypatch, capsys):
    import debug.commands.worker as worker

    class _Client:
        def get_worker_info(self, *_args, **_kwargs):
            return SimpleNamespace(worker_id="w1", state={"status": "active"}, logs=[], tasks=[])

    monkeypatch.setattr(worker.Formatter, "format_worker", lambda *_args, **_kwargs: "worker-ok")
    worker.run(client=_Client(), worker_id="w1", options={"format": "text"})
    assert "worker-ok" in capsys.readouterr().out


def test_worker_run_direct_disk_check_branch(capsys):
    import debug.commands.worker as worker

    class _Client:
        def check_worker_disk_space(self, _worker_id):
            return {"available": True, "runpod_id": "rp1", "disk_info": "Filesystem 70%"}

    worker.run(client=_Client(), worker_id="w1", options={"check_disk": True})
    out = capsys.readouterr().out
    assert "DISK SPACE CHECK" in out
    assert "SSH successful" in out
