from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_magic_edit_direct_missing_replicate(tmp_path):
    import source.task_handlers.magic_edit as magic_edit

    original = magic_edit.replicate
    magic_edit.replicate = None
    try:
        ok, msg = magic_edit.handle_magic_edit_task(
            task_params_from_db={"image_url": "https://example.com/x.png"},
            main_output_dir_base=Path(tmp_path),
            task_id="m2",
        )
        assert ok is False
        assert "Replicate library" in msg
    finally:
        magic_edit.replicate = original


def test_magic_edit_direct_success_url_output(monkeypatch, tmp_path):
    import source.task_handlers.magic_edit as magic_edit

    class _Output:
        def url(self):
            return "https://example.com/out.webp"

    monkeypatch.setenv("REPLICATE_API_TOKEN", "token")
    monkeypatch.setattr(magic_edit, "replicate", SimpleNamespace(run=lambda *_args, **_kwargs: _Output()))
    monkeypatch.setattr(magic_edit, "prepare_output_path_with_upload", lambda **_kwargs: (tmp_path / "final.webp", "db://initial"))
    monkeypatch.setattr(magic_edit, "upload_and_get_final_output_location", lambda *_args, **_kwargs: "db://final")
    monkeypatch.setattr(
        magic_edit.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(content=b"img", raise_for_status=lambda: None),
    )

    ok, out = magic_edit.handle_magic_edit_task(
        task_params_from_db={"image_url": "https://example.com/in.png", "prompt": "p"},
        main_output_dir_base=Path(tmp_path),
        task_id="m3",
    )

    assert ok is True
    assert out == "db://final"
    assert (tmp_path / "final.webp").exists()
