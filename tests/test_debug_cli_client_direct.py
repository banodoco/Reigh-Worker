from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_debug_cli_create_parser_has_expected_commands():
    import debug.cli as cli

    parser = cli.create_parser()
    help_text = parser.format_help()
    assert "task" in help_text
    assert "worker" in help_text
    assert "runpod" in help_text
    assert "storage" in help_text


def test_debug_cli_main_routes_to_health(monkeypatch):
    import debug.cli as cli

    called = {"health": False}
    fake_commands = types.ModuleType("debug.commands")
    fake_commands.task = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.worker = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.tasks = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.workers = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.health = SimpleNamespace(run=lambda *_a, **_k: called.update({"health": True}))
    fake_commands.orchestrator = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.config = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.runpod = SimpleNamespace(run=lambda *_a, **_k: None)
    fake_commands.storage = SimpleNamespace(run=lambda *_a, **_k: None)

    monkeypatch.setitem(sys.modules, "debug.commands", fake_commands)
    monkeypatch.setattr(cli, "DebugClient", lambda: object())
    monkeypatch.setattr(sys, "argv", ["debug.py", "health"])
    cli.main()
    assert called["health"] is True


def test_debug_client_log_query_get_logs():
    from debug.client import LogQueryClient

    class _Query:
        def select(self, *_args, **_kwargs):
            return self

        def gte(self, *_args, **_kwargs):
            return self

        def lte(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def ilike(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            return SimpleNamespace(data=[{"message": "ok"}])

    fake_supabase = SimpleNamespace(table=lambda _name: _Query())
    client = LogQueryClient(fake_supabase)
    logs = client.get_logs(source_type="worker", search_term="ok", limit=5)
    assert len(logs) == 1
    assert logs[0]["message"] == "ok"


def test_debug_client_init_requires_env(monkeypatch):
    import debug.client as dc

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    try:
        dc.DebugClient()
    except ValueError as exc:
        assert "SUPABASE_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError when env vars are missing")


def test_debug_client_check_worker_logging_uses_log_client(monkeypatch):
    import debug.client as dc

    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret")
    monkeypatch.setattr(dc, "create_client", lambda *_args, **_kwargs: object())
    client = dc.DebugClient()
    client.log_client = SimpleNamespace(get_logs=lambda **_kwargs: [{"message": "m1"}, {"message": "m2"}])
    result = client.check_worker_logging("w1")
    assert result["is_logging"] is True
    assert result["log_count"] == 2
