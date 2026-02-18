"""Static checks for Uni3C filter protection in orchestrator."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_PATH = REPO_ROOT / "source" / "models" / "wgp" / "orchestrator.py"


def _src() -> str:
    return ORCH_PATH.read_text(encoding="utf-8")


def test_orchestrator_filter_has_uni3c_key_set():
    src = _src()
    assert "uni3c_keys = {" in src
    assert '"use_uni3c"' in src
    assert '"uni3c_guide_video"' in src
    assert '"uni3c_strength"' in src


def test_orchestrator_filter_has_fail_fast_message():
    src = _src()
    assert "Unsupported Uni3C params dropped" in src
    assert "silent Uni3C degradation" in src


def test_orchestrator_filter_re_raises_runtimeerror():
    src = _src()
    assert "except RuntimeError:" in src
    assert "raise" in src
