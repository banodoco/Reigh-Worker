from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _TableChain:
    def __init__(self, data):
        self._data = data

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class _FakeSupabase:
    def table(self, _name):
        return _TableChain([{"id": "w1", "status": "active", "metadata": {"runpod_id": "rp1"}}])


def test_runpod_run_direct_reports_orphan_and_terminates(monkeypatch, capsys):
    import debug.commands.runpod as runpod

    terminated: list[str] = []
    fake_runpod = SimpleNamespace(
        api_key="",
        get_pods=lambda: [
            {"id": "rp1", "name": "gpu_x", "desiredStatus": "RUNNING", "costPerHr": 0.1},
            {"id": "rp2", "name": "gpu_orphan", "desiredStatus": "RUNNING", "costPerHr": 0.2},
        ],
        terminate_pod=lambda pod_id: terminated.append(pod_id),
    )
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)
    monkeypatch.setenv("RUNPOD_API_KEY", "abc123")

    runpod.run(client=SimpleNamespace(supabase=_FakeSupabase()), options={"terminate": True})
    out = capsys.readouterr().out
    assert "RUNPOD SYNC STATUS" in out
    assert "Summary" in out
    assert "RunPod active pods: 2" in out
    assert "Database active workers: 2" in out
    assert "Orphaned pods (RunPod only): 1" in out
    assert "Stale workers (DB only): 0" in out
    assert "ORPHANED PODS" in out
    assert "gpu_orphan" in out
    assert "rp2" in out
    assert "TERMINATING 1 ORPHANED PODS" in out
    assert "Terminated rp2" in out
    assert terminated == ["rp2"]


def test_runpod_run_direct_missing_key_exits(monkeypatch):
    import debug.commands.runpod as runpod

    monkeypatch.setitem(sys.modules, "runpod", SimpleNamespace(api_key="", get_pods=lambda: []))
    monkeypatch.setattr(runpod, "load_dotenv", lambda: None)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    try:
        runpod.run(client=SimpleNamespace(supabase=_FakeSupabase()), options={})
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when RUNPOD_API_KEY is missing")


def test_runpod_run_direct_import_error_exits(monkeypatch):
    import debug.commands.runpod as runpod

    monkeypatch.setattr(runpod, "load_dotenv", lambda: None)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "runpod", raising=False)
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "runpod":
            raise ImportError("no runpod")
        return real_import(name, *args, **kwargs)

    import builtins
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    try:
        runpod.run(client=SimpleNamespace(supabase=_FakeSupabase()), options={})
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when runpod module is unavailable")


def test_runpod_run_direct_no_orphans_message(monkeypatch, capsys):
    import debug.commands.runpod as runpod

    fake_runpod = SimpleNamespace(
        api_key="",
        get_pods=lambda: [{"id": "rp1", "name": "gpu_x", "desiredStatus": "RUNNING", "costPerHr": 0.1}],
        terminate_pod=lambda _pod_id: None,
    )
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)
    monkeypatch.setenv("RUNPOD_API_KEY", "abc123")
    runpod.run(client=SimpleNamespace(supabase=_FakeSupabase()), options={"terminate": False})
    out = capsys.readouterr().out
    assert "No orphaned pods found" in out
