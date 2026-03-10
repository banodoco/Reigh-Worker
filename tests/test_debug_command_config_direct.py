from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_config_run_direct_masks_keys_and_explains(monkeypatch, capsys):
    import debug.commands.config as config

    monkeypatch.setenv("RUNPOD_API_KEY", "abcdefghijklmnop")
    monkeypatch.setenv("WORKER_GRACE_PERIOD_SEC", "120")
    config.run(client=SimpleNamespace(), options={"explain": True})
    out = capsys.readouterr().out
    assert "SYSTEM CONFIGURATION" in out
    assert "RUNPOD_API_KEY: abcdefgh...mnop" in out
    assert "2.0 minutes after promotion to active" in out
