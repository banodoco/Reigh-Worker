"""Tests for source/core/db/task_completion.py — add_task_to_db."""

import json
from unittest.mock import MagicMock, patch

import pytest

from source.core.db.task_completion import add_task_to_db


def _edge_ok(data=None):
    resp = MagicMock(status_code=200, text=json.dumps(data or {}))
    resp.json.return_value = data or {}
    return resp, None


def _edge_err(msg="failed"):
    return None, msg


# ── add_task_to_db ──────────────────────────────────────────────────────────

class TestAddTaskToDb:
    def test_success_returns_uuid(self, mock_db_config, mock_sleep):
        """Successful creation returns a UUID string."""
        # Mock the verification query too
        mock_verify = MagicMock()
        mock_verify.data = {"status": "Queued", "created_at": "2024-01-01", "project_id": "proj", "task_type": "t2v"}
        mock_db_config["client"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_verify

        with patch(
            "source.core.db.task_completion._call_edge_function_with_retry",
            return_value=_edge_ok(),
        ) as mock_edge:
            task_id = add_task_to_db(
                {"prompt": "test", "project_id": "proj"},
                "t2v",
            )
        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID format
        # Verify edge function was called with correct payload
        call_kwargs = mock_edge.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[0][1]
        assert payload["task_type"] == "t2v"
        assert payload["project_id"] == "proj"

    def test_raises_on_edge_failure(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_completion._call_edge_function_with_retry",
            return_value=_edge_err("server error"),
        ):
            with pytest.raises(RuntimeError, match="create-task failed"):
                add_task_to_db({"prompt": "test"}, "t2v")

    def test_raises_on_non_200(self, mock_db_config, mock_sleep):
        resp = MagicMock(status_code=400, text="bad request")
        with patch(
            "source.core.db.task_completion._call_edge_function_with_retry",
            return_value=(resp, None),
        ):
            with pytest.raises(RuntimeError, match="400"):
                add_task_to_db({"prompt": "test"}, "t2v")

    def test_list_normalization(self, mock_db_config, mock_sleep):
        """When dependant_on is a string, it should be normalized to a list."""
        mock_verify = MagicMock()
        mock_verify.data = {"status": "Queued", "created_at": "2024-01-01", "project_id": "proj", "task_type": "stitch"}
        mock_db_config["client"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_verify

        with patch(
            "source.core.db.task_completion._call_edge_function_with_retry",
            return_value=_edge_ok(),
        ) as mock_edge:
            add_task_to_db(
                {"prompt": "test", "project_id": "proj"},
                "stitch",
                dependant_on="dep-id-123",
            )

        call_kwargs = mock_edge.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[0][1]
        assert payload["dependant_on"] == ["dep-id-123"]

    def test_no_edge_url_raises(self, mock_sleep):
        """Raises ValueError when edge URL is not configured."""
        import source.core.db.config as _cfg
        orig_url = _cfg.SUPABASE_URL
        orig_create = _cfg.SUPABASE_EDGE_CREATE_TASK_URL
        _cfg.SUPABASE_URL = None
        _cfg.SUPABASE_EDGE_CREATE_TASK_URL = None
        try:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="not configured"):
                    add_task_to_db({"prompt": "test"}, "t2v")
        finally:
            _cfg.SUPABASE_URL = orig_url
            _cfg.SUPABASE_EDGE_CREATE_TASK_URL = orig_create

    def test_raises_on_none_response(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_completion._call_edge_function_with_retry",
            return_value=(None, None),
        ):
            with pytest.raises(RuntimeError, match="no response"):
                add_task_to_db({"prompt": "test"}, "t2v")
