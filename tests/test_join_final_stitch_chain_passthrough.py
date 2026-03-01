from __future__ import annotations

from pathlib import Path

import source.task_handlers.join.final_stitch as final_stitch


def test_chain_mode_passthrough_materializes_remote_output_for_current_task(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task_id = "final-task-1234"
    remote_output = (
        "https://example.supabase.co/storage/v1/object/public/image_uploads/"
        "user/tasks/prev-task/prev-task_joined.mp4"
    )

    monkeypatch.setattr(
        final_stitch.db_ops,
        "get_task_output_location_from_db",
        lambda _task_id: remote_output,
    )

    def _fake_download_video_if_url(
        _url,
        *,
        download_target_dir,
        task_id_for_logging,
        descriptive_name,
    ):
        assert task_id_for_logging == task_id
        assert descriptive_name == "chain_passthrough"
        download_path = Path(download_target_dir) / "downloaded_prev_output.mp4"
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"video-bytes")
        return str(download_path)

    monkeypatch.setattr(final_stitch, "download_video_if_url", _fake_download_video_if_url)

    ok, output_location = final_stitch.handle_join_final_stitch(
        {"chain_mode": True, "transition_task_ids": ["prev-task"]},
        main_output_dir_base=tmp_path,
        task_id=task_id,
    )

    assert ok is True
    assert output_location != remote_output
    materialized_path = Path(output_location)
    assert materialized_path.exists()
    assert materialized_path.name.startswith(task_id)
    assert materialized_path.read_bytes() == b"video-bytes"


def test_chain_mode_passthrough_copies_existing_local_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task_id = "final-task-5678"
    previous_local_output = tmp_path / "prev_output.mp4"
    previous_local_output.write_bytes(b"existing-local-video")

    monkeypatch.setattr(
        final_stitch.db_ops,
        "get_task_output_location_from_db",
        lambda _task_id: str(previous_local_output),
    )

    def _fail_download(*_args, **_kwargs):
        raise AssertionError("download_video_if_url should not be called for existing local output")

    monkeypatch.setattr(final_stitch, "download_video_if_url", _fail_download)

    ok, output_location = final_stitch.handle_join_final_stitch(
        {"chain_mode": True, "transition_task_ids": ["prev-task"]},
        main_output_dir_base=tmp_path,
        task_id=task_id,
    )

    assert ok is True
    materialized_path = Path(output_location)
    assert materialized_path.exists()
    assert materialized_path != previous_local_output
    assert materialized_path.name.startswith(task_id)
    assert materialized_path.read_bytes() == b"existing-local-video"
