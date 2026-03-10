from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_health_command_direct_formats_output(monkeypatch, capsys):
    import debug.commands.health as health

    monkeypatch.setattr(health.Formatter, "format_health", lambda *_args, **_kwargs: "health-ok")
    client = SimpleNamespace(get_system_health=lambda: {"ok": True})
    health.run(client=client, options={"format": "text"})
    assert "health-ok" in capsys.readouterr().out


def test_orchestrator_command_direct_formats_output(monkeypatch, capsys):
    import debug.commands.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator.Formatter, "format_orchestrator", lambda *_args, **_kwargs: "orch-ok")
    client = SimpleNamespace(get_orchestrator_status=lambda **_kwargs: {"status": "ok"})
    orchestrator.run(client=client, options={"format": "text", "hours": 2})
    assert "orch-ok" in capsys.readouterr().out
