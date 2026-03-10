"""Tests for source/core/db/task_claim.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from source.core.db.task_claim import (
    check_task_counts_supabase,
    check_my_assigned_tasks,
    _orchestrator_has_incomplete_children,
)


# ── check_task_counts_supabase ───────────────────────────────────────────────

class TestCheckTaskCountsSupabase:
    def test_returns_counts_on_success(self, mock_db_config, mock_httpx):
        counts = {"totals": {"queued_only": 5, "active_only": 2}}
        mock_httpx["post"].return_value = MagicMock(
            status_code=200,
            text=json.dumps(counts),
            json=MagicMock(return_value=counts),
        )
        result = check_task_counts_supabase("gpu")
        assert result is not None
        assert result["totals"]["queued_only"] == 5

    def test_none_on_failure(self, mock_db_config, mock_httpx):
        mock_httpx["post"].return_value = MagicMock(
            status_code=500,
            text="error",
        )
        assert check_task_counts_supabase("gpu") is None

    def test_none_when_no_client(self):
        """Returns None when SUPABASE_CLIENT is not set."""
        import source.core.db.config as _cfg
        orig_client = _cfg.SUPABASE_CLIENT
        orig_token = _cfg.SUPABASE_ACCESS_TOKEN
        _cfg.SUPABASE_CLIENT = None
        _cfg.SUPABASE_ACCESS_TOKEN = None
        try:
            assert check_task_counts_supabase("gpu") is None
        finally:
            _cfg.SUPABASE_CLIENT = orig_client
            _cfg.SUPABASE_ACCESS_TOKEN = orig_token


# ── check_my_assigned_tasks ─────────────────────────────────────────────────

class TestCheckMyAssignedTasks:
    def test_returns_task_when_found(self, mock_db_config):
        task_data = {
            "id": "task-123",
            "task_type": "travel_segment",
            "params": {"prompt": "test"},
            "project_id": "proj-1",
        }
        mock_result = MagicMock()
        mock_result.data = [task_data]
        mock_db_config["client"].table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        result = check_my_assigned_tasks("worker-1")
        assert result is not None
        assert result["task_id"] == "task-123"
        assert result["task_type"] == "travel_segment"

    def test_none_when_empty(self, mock_db_config):
        mock_result = MagicMock()
        mock_result.data = []
        mock_db_config["client"].table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result

        assert check_my_assigned_tasks("worker-1") is None

    def test_none_on_error(self, mock_db_config):
        from postgrest.exceptions import APIError
        mock_db_config["client"].table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.side_effect = APIError({"message": "db error", "code": "42", "details": "", "hint": ""})

        assert check_my_assigned_tasks("worker-1") is None

    def test_none_when_no_client(self):
        import source.core.db.config as _cfg
        orig = _cfg.SUPABASE_CLIENT
        _cfg.SUPABASE_CLIENT = None
        try:
            assert check_my_assigned_tasks("worker-1") is None
        finally:
            _cfg.SUPABASE_CLIENT = orig

    def test_none_when_no_worker_id(self, mock_db_config):
        assert check_my_assigned_tasks("") is None


# ── _orchestrator_has_incomplete_children ───────────────────────────────────

class TestOrchestratorHasIncompleteChildren:
    def test_no_children_returns_false(self, mock_db_config, mock_httpx, mock_sleep):
        mock_httpx["post"].return_value = MagicMock(
            status_code=200,
            text=json.dumps({"tasks": []}),
            json=MagicMock(return_value={"tasks": []}),
        )
        assert _orchestrator_has_incomplete_children("orch-1") is False

    def test_all_complete_returns_false(self, mock_db_config, mock_httpx, mock_sleep):
        tasks = [
            {"id": "s1", "status": "Complete"},
            {"id": "s2", "status": "Failed"},
        ]
        mock_httpx["post"].return_value = MagicMock(
            status_code=200,
            text=json.dumps({"tasks": tasks}),
            json=MagicMock(return_value={"tasks": tasks}),
        )
        assert _orchestrator_has_incomplete_children("orch-1") is False

    def test_incomplete_returns_true(self, mock_db_config, mock_httpx, mock_sleep):
        tasks = [
            {"id": "s1", "status": "Complete"},
            {"id": "s2", "status": "In Progress"},
        ]
        mock_httpx["post"].return_value = MagicMock(
            status_code=200,
            text=json.dumps({"tasks": tasks}),
            json=MagicMock(return_value={"tasks": tasks}),
        )
        assert _orchestrator_has_incomplete_children("orch-1") is True

    def test_fallback_to_direct_query(self, mock_db_config, mock_httpx, mock_sleep):
        """When edge function fails, uses direct DB query."""
        import httpx as real_httpx
        mock_httpx["post"].side_effect = real_httpx.ConnectError("refused")

        mock_resp = MagicMock()
        mock_resp.data = [{"id": "s1", "status": "Queued"}]
        mock_db_config["client"].table.return_value.select.return_value.contains.return_value.execute.return_value = mock_resp

        assert _orchestrator_has_incomplete_children("orch-1") is True
