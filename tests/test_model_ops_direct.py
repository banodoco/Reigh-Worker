from __future__ import annotations

import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_model_ops_direct_load_missing_model_definition(tmp_path, monkeypatch):
    import source.models.wgp.model_ops as mo

    json_path = tmp_path / "model.json"
    json_path.write_text(json.dumps({"model": {"architecture": "wan", "name": "m"}, "foo": "bar"}), encoding="utf-8")

    fake_wgp = types.SimpleNamespace(
        models_def={},
        init_model_def=lambda _key, model_def: dict(model_def, initialized=True),
    )
    monkeypatch.setitem(sys.modules, "wgp", fake_wgp)

    mo.load_missing_model_definition(orchestrator=types.SimpleNamespace(), model_key="m1", json_path=str(json_path))
    assert "m1" in fake_wgp.models_def
    assert fake_wgp.models_def["m1"]["path"] == str(json_path)
    assert fake_wgp.models_def["m1"]["settings"]["foo"] == "bar"


def test_model_ops_direct_smoke_mode_load_and_unload():
    import source.models.wgp.model_ops as mo

    orchestrator = types.SimpleNamespace(
        smoke_mode=True,
        current_model=None,
        state={},
        offloadobj=object(),
        _cached_uni3c_controlnet=object(),
    )
    switched = mo.load_model_impl(orchestrator, "vace_14B")
    assert switched is True
    assert orchestrator.current_model == "vace_14B"
    assert orchestrator.state["model_type"] == "vace_14B"

    mo.unload_model_impl(orchestrator)
    assert orchestrator.current_model is None
    assert orchestrator.state["model_type"] is None
    assert orchestrator._cached_uni3c_controlnet is None
