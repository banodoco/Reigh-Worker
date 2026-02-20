from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_fake_runpod_client(monkeypatch):
    pkg = types.ModuleType("gpu_orchestrator")
    mod = types.ModuleType("gpu_orchestrator.runpod_client")

    class _RunpodClient:
        api_key = "k"

        def _get_storage_volume_id(self, _name):
            return "vol_1"

        def check_storage_health(self, **_kwargs):
            return {"healthy": True, "total_gb": 100, "used_gb": 30, "percent_used": 30, "free_gb": 70, "message": "ok"}

        def _expand_network_volume(self, _volume_id, _new_size):
            return True

    mod.create_runpod_client = lambda: _RunpodClient()
    mod.get_network_volumes = lambda _api_key: [{"name": "shared", "id": "vol_1", "size": 100, "dataCenter": {"name": "US"}}]
    monkeypatch.setitem(sys.modules, "gpu_orchestrator", pkg)
    monkeypatch.setitem(sys.modules, "gpu_orchestrator.runpod_client", mod)


def test_storage_run_direct_reports_volumes_and_health(monkeypatch, capsys):
    import debug.commands.storage as storage

    _install_fake_runpod_client(monkeypatch)

    class _WorkersTable:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def execute(self):
            return SimpleNamespace(data=[{"id": "w1", "metadata": {"runpod_id": "rp1", "storage_volume": "shared"}}])

    client = SimpleNamespace(supabase=SimpleNamespace(table=lambda _name: _WorkersTable()))
    storage.run(client=client, options={})
    out = capsys.readouterr().out
    assert "STORAGE HEALTH CHECK" in out
    assert "RunPod Network Volumes" in out
    assert "HEALTHY" in out
