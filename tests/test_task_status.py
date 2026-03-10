"""Tests for source/core/db/task_status.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from source.core.db.task_status import (
    requeue_task_for_retry,
    _requeue_task_direct_db,
    reset_generation_started_at,
)


def _edge_ok(data=None):
    resp = MagicMock(status_code=200, text=json.dumps(data or {}))
    resp.json.return_value = data or {}
    return resp, None


def _edge_err(msg="failed"):
    return None, msg


# ── requeue_task_for_retry ───────────────────────────────────────────────────

class TestRequeueTaskForRetry:
    def test_success_via_edge(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_status._call_edge_function_with_retry",
            return_value=_edge_ok(),
        ) as mock_edge:
            result = requeue_task_for_retry("task-1", "OOM error", current_attempts=0)
        assert result is True
        mock_edge.assert_called_once()
        payload = mock_edge.call_args.kwargs.get("payload") or mock_edge.call_args[0][1]
        assert payload["task_id"] == "task-1"
        assert payload["attempts"] == 1

    def test_fallback_to_direct_db(self, mock_db_config, mock_sleep):
        """When edge function fails, falls back to direct DB update."""
        mock_result = MagicMock()
        mock_result.data = [{"id": "task-1"}]
        mock_db_config["client"].table.return_value.update.return_value.eq.return_value.execute.return_value = mock_result

        with patch(
            "source.core.db.task_status._call_edge_function_with_retry",
            return_value=_edge_err(),
        ):
            result = requeue_task_for_retry("task-1", "OOM error", current_attempts=1)
        assert result is True

    def test_increments_attempts(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_status._call_edge_function_with_retry",
            return_value=_edge_ok(),
        ) as mock_edge:
            requeue_task_for_retry("task-1", "error", current_attempts=2)

        # Check the payload sent to the edge function
        call_kwargs = mock_edge.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[0][1]
        assert payload["attempts"] == 3

    def test_includes_error_category(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_status._call_edge_function_with_retry",
            return_value=_edge_ok(),
        ) as mock_edge:
            requeue_task_for_retry("task-1", "error", current_attempts=0, error_category="OOM")

        call_kwargs = mock_edge.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[0][1]
        assert "OOM" in payload["error_details"]

    def test_false_when_all_fail(self, mock_sleep):
        """Returns False when no edge URL and no DB client."""
        import source.core.db.config as _cfg
        orig_url = _cfg.SUPABASE_URL
        orig_token = _cfg.SUPABASE_ACCESS_TOKEN
        orig_client = _cfg.SUPABASE_CLIENT
        _cfg.SUPABASE_URL = None
        _cfg.SUPABASE_ACCESS_TOKEN = None
        _cfg.SUPABASE_CLIENT = None
        try:
            with patch.dict("os.environ", {}, clear=True):
                result = requeue_task_for_retry("task-1", "error", current_attempts=0)
            assert result is False
        finally:
            _cfg.SUPABASE_URL = orig_url
            _cfg.SUPABASE_ACCESS_TOKEN = orig_token
            _cfg.SUPABASE_CLIENT = orig_client


# ── reset_generation_started_at ──────────────────────────────────────────────

class TestResetGenerationStartedAt:
    def test_success(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_status._call_edge_function_with_retry",
            return_value=_edge_ok(),
        ):
            assert reset_generation_started_at("task-1") is True

    def test_failure(self, mock_db_config, mock_sleep):
        with patch(
            "source.core.db.task_status._call_edge_function_with_retry",
            return_value=_edge_err(),
        ):
            assert reset_generation_started_at("task-1") is False

    def test_no_edge_url(self, mock_sleep):
        """Returns False when no edge URL available."""
        import source.core.db.config as _cfg
        orig_url = _cfg.SUPABASE_URL
        orig_token = _cfg.SUPABASE_ACCESS_TOKEN
        _cfg.SUPABASE_URL = None
        _cfg.SUPABASE_ACCESS_TOKEN = None
        try:
            with patch.dict("os.environ", {}, clear=True):
                assert reset_generation_started_at("task-1") is False
        finally:
            _cfg.SUPABASE_URL = orig_url
            _cfg.SUPABASE_ACCESS_TOKEN = orig_token


# ── _requeue_task_direct_db ──────────────────────────────────────────────────

class TestRequeueTaskDirectDb:
    def test_success(self, mock_db_config):
        mock_result = MagicMock()
        mock_result.data = [{"id": "task-1"}]
        mock_db_config["client"].table.return_value.update.return_value.eq.return_value.execute.return_value = mock_result

        assert _requeue_task_direct_db("task-1", 2, "error details") is True

    def test_no_client(self):
        import source.core.db.config as _cfg
        orig = _cfg.SUPABASE_CLIENT
        _cfg.SUPABASE_CLIENT = None
        try:
            assert _requeue_task_direct_db("task-1", 1, "error") is False
        finally:
            _cfg.SUPABASE_CLIENT = orig

    def test_db_error(self, mock_db_config):
        from postgrest.exceptions import APIError
        mock_db_config["client"].table.return_value.update.return_value.eq.return_value.execute.side_effect = APIError({"message": "error", "code": "42", "details": "", "hint": ""})

        assert _requeue_task_direct_db("task-1", 1, "error") is False
