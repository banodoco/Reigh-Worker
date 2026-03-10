"""Tests for source/core/db/task_polling.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from source.core.db.task_polling import (
    get_task_output_location_from_db,
    get_abs_path_from_db_path,
)


def _edge_ok(data):
    """Helper: mock _call_edge_function_with_retry returning success."""
    resp = MagicMock(status_code=200, text=json.dumps(data))
    resp.json.return_value = data
    return resp, None


def _edge_err(msg="failed"):
    """Helper: mock _call_edge_function_with_retry returning error."""
    return None, msg


# ── get_task_output_location_from_db ────────────────────────────────────────

class TestGetTaskOutputLocationFromDb:
    def test_success_returns_output(self, mock_db_config, mock_sleep):
        data = {"status": "Complete", "output_location": "https://storage/file.mp4"}
        with patch(
            "source.core.db.task_polling._call_edge_function_with_retry",
            return_value=_edge_ok(data),
        ):
            result = get_task_output_location_from_db("task-1")
        assert result == "https://storage/file.mp4"

    def test_none_when_not_complete(self, mock_db_config, mock_sleep):
        data = {"status": "In Progress", "output_location": None}
        with patch(
            "source.core.db.task_polling._call_edge_function_with_retry",
            return_value=_edge_ok(data),
        ):
            result = get_task_output_location_from_db("task-1")
        assert result is None

    def test_none_on_edge_error(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_polling._call_edge_function_with_retry",
            return_value=_edge_err(),
        ):
            result = get_task_output_location_from_db("task-1")
        assert result is None

    def test_none_on_404(self, mock_db_config, mock_sleep):
        resp = MagicMock(status_code=404, text="not found")
        resp.json.return_value = {}
        with patch(
            "source.core.db.task_polling._call_edge_function_with_retry",
            return_value=(resp, None),
        ):
            result = get_task_output_location_from_db("task-1")
        assert result is None

    def test_none_when_no_url(self, mock_sleep):
        """No SUPABASE_URL and no env var → returns None."""
        import source.core.db.config as _cfg
        orig_url = _cfg.SUPABASE_URL
        orig_token = _cfg.SUPABASE_ACCESS_TOKEN
        _cfg.SUPABASE_URL = None
        _cfg.SUPABASE_ACCESS_TOKEN = None
        try:
            with patch.dict("os.environ", {}, clear=True):
                result = get_task_output_location_from_db("task-1")
            assert result is None
        finally:
            _cfg.SUPABASE_URL = orig_url
            _cfg.SUPABASE_ACCESS_TOKEN = orig_token


# ── get_abs_path_from_db_path ───────────────────────────────────────────────

class TestGetAbsPathFromDbPath:
    def test_none_input(self):
        assert get_abs_path_from_db_path(None) is None

    def test_empty_string(self):
        assert get_abs_path_from_db_path("") is None

    def test_existing_path(self, tmp_path):
        f = tmp_path / "output.mp4"
        f.write_bytes(b"video")
        result = get_abs_path_from_db_path(str(f))
        assert result is not None
        assert result.exists()

    def test_nonexistent_path(self):
        result = get_abs_path_from_db_path("/nonexistent/file.mp4")
        assert result is None
