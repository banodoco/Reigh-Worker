"""Tests for source/core/db/task_dependencies.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from source.core.db.task_dependencies import (
    cancel_orchestrator_children,
    cleanup_duplicate_child_tasks,
    get_orchestrator_child_tasks,
    get_task_current_status,
)


def _edge_ok(data):
    resp = MagicMock(status_code=200, text=json.dumps(data))
    resp.json.return_value = data
    return resp, None


def _edge_err(msg="failed"):
    return None, msg


ORCH_ID = "orch-123"


# ── cancel_orchestrator_children ────────────────────────────────────────────

class TestCancelOrchestratorChildren:
    def test_no_children_returns_zero(self, mock_db_config, mock_httpx, mock_sleep):
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value={"segments": [], "stitch": [], "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": []},
        ):
            assert cancel_orchestrator_children(ORCH_ID) == 0

    def test_skips_terminal_statuses(self, mock_db_config, mock_httpx, mock_sleep):
        children = {
            "segments": [
                {"id": "s1", "status": "Complete"},
                {"id": "s2", "status": "Failed"},
                {"id": "s3", "status": "Cancelled"},
            ],
            "stitch": [], "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": [],
        }
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value=children,
        ), patch("source.core.db.task_dependencies.update_task_status") as mock_update:
            count = cancel_orchestrator_children(ORCH_ID)
        assert count == 0
        mock_update.assert_not_called()

    def test_cancels_non_terminal(self, mock_db_config, mock_httpx, mock_sleep):
        children = {
            "segments": [
                {"id": "s1", "status": "Queued"},
                {"id": "s2", "status": "In Progress"},
                {"id": "s3", "status": "Complete"},
            ],
            "stitch": [], "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": [],
        }
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value=children,
        ), patch("source.core.db.task_dependencies.update_task_status") as mock_update:
            count = cancel_orchestrator_children(ORCH_ID)
        assert count == 2
        # Verify correct task IDs were cancelled
        cancelled_ids = {call.args[0] for call in mock_update.call_args_list}
        assert cancelled_ids == {"s1", "s2"}

    def test_returns_count(self, mock_db_config, mock_httpx, mock_sleep):
        children = {
            "segments": [{"id": "s1", "status": "Queued"}],
            "stitch": [{"id": "st1", "status": "In Progress"}],
            "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": [],
        }
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value=children,
        ), patch("source.core.db.task_dependencies.update_task_status"):
            count = cancel_orchestrator_children(ORCH_ID)
        assert count == 2


# ── cleanup_duplicate_child_tasks ───────────────────────────────────────────

class TestCleanupDuplicateChildTasks:
    def test_no_duplicates(self, mock_db_config, mock_httpx, mock_sleep):
        children = {
            "segments": [
                {"id": "s1", "params": {"segment_index": 0}},
                {"id": "s2", "params": {"segment_index": 1}},
            ],
            "stitch": [{"id": "st1"}],
            "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": [],
        }
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value=children,
        ):
            summary = cleanup_duplicate_child_tasks(ORCH_ID, expected_segments=2)
        assert summary["duplicate_segments_removed"] == 0
        assert summary["duplicate_stitch_removed"] == 0

    def test_duplicate_segments_removed(self, mock_db_config, mock_httpx, mock_sleep):
        children = {
            "segments": [
                {"id": "s1", "params": {"segment_index": 0}},
                {"id": "s2", "params": {"segment_index": 0}},  # duplicate
            ],
            "stitch": [],
            "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": [],
        }
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value=children,
        ), patch("source.core.db.task_dependencies._delete_task_by_id", return_value=True):
            summary = cleanup_duplicate_child_tasks(ORCH_ID, expected_segments=1)
        assert summary["duplicate_segments_removed"] == 1

    def test_duplicate_stitch_removed(self, mock_db_config, mock_httpx, mock_sleep):
        children = {
            "segments": [],
            "stitch": [
                {"id": "st1", "created_at": "2024-01-01"},
                {"id": "st2", "created_at": "2024-01-02"},
            ],
            "join_clips_segment": [], "join_clips_orchestrator": [], "join_final_stitch": [],
        }
        with patch(
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            return_value=children,
        ), patch("source.core.db.task_dependencies._delete_task_by_id", return_value=True):
            summary = cleanup_duplicate_child_tasks(ORCH_ID, expected_segments=0)
        assert summary["duplicate_stitch_removed"] == 1


# ── get_orchestrator_child_tasks ────────────────────────────────────────────

class TestGetOrchestratorChildTasks:
    def test_categorizes_by_type(self, mock_db_config, mock_httpx, mock_sleep):
        tasks = [
            {"id": "1", "task_type": "travel_segment", "status": "Complete", "params": {}},
            {"id": "2", "task_type": "travel_stitch", "status": "Queued", "params": {}},
            {"id": "3", "task_type": "join_clips_segment", "status": "Queued", "params": {}},
        ]
        with patch(
            "source.core.db.task_dependencies._call_edge_function_with_retry",
            return_value=_edge_ok({"tasks": tasks}),
        ):
            result = get_orchestrator_child_tasks(ORCH_ID)
        assert len(result["segments"]) == 1
        assert len(result["stitch"]) == 1
        assert len(result["join_clips_segment"]) == 1

    def test_empty_result(self, mock_db_config, mock_httpx, mock_sleep):
        with patch(
            "source.core.db.task_dependencies._call_edge_function_with_retry",
            return_value=_edge_ok({"tasks": []}),
        ):
            result = get_orchestrator_child_tasks(ORCH_ID)
        assert all(len(v) == 0 for v in result.values())

    def test_edge_function_fallback_to_direct(self, mock_db_config, mock_httpx, mock_sleep):
        """When edge function fails, falls back to direct DB query."""
        mock_db_response = MagicMock()
        mock_db_response.data = [
            {"id": "1", "task_type": "travel_segment", "status": "Complete", "params": {}, "output_location": ""},
        ]
        mock_db_config["client"].table.return_value.select.return_value.contains.return_value.order.return_value.execute.return_value = mock_db_response

        with patch(
            "source.core.db.task_dependencies._call_edge_function_with_retry",
            return_value=_edge_err(),
        ):
            result = get_orchestrator_child_tasks(ORCH_ID)
        assert len(result["segments"]) == 1


# ── get_task_current_status ────────────────────────────────────────────────

class TestGetTaskCurrentStatus:
    def test_via_edge(self, mock_db_config, mock_httpx, mock_sleep):
        with patch(
            "source.core.db.task_dependencies._call_edge_function_with_retry",
            return_value=_edge_ok({"status": "In Progress"}),
        ):
            assert get_task_current_status("task-1") == "In Progress"

    def test_fallback_to_direct(self, mock_db_config, mock_httpx, mock_sleep):
        """When edge fails, falls back to direct DB query."""
        mock_resp = MagicMock()
        mock_resp.data = {"status": "Queued"}
        mock_db_config["client"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_resp

        with patch(
            "source.core.db.task_dependencies._call_edge_function_with_retry",
            return_value=_edge_err(),
        ):
            assert get_task_current_status("task-1") == "Queued"

    def test_none_on_all_failure(self, mock_sleep):
        """Returns None when no edge function and no DB client."""
        import source.core.db.config as _cfg
        orig_url = _cfg.SUPABASE_URL
        orig_token = _cfg.SUPABASE_ACCESS_TOKEN
        orig_client = _cfg.SUPABASE_CLIENT
        _cfg.SUPABASE_URL = None
        _cfg.SUPABASE_ACCESS_TOKEN = None
        _cfg.SUPABASE_CLIENT = None
        try:
            with patch.dict("os.environ", {}, clear=True):
                assert get_task_current_status("task-1") is None
        finally:
            _cfg.SUPABASE_URL = orig_url
            _cfg.SUPABASE_ACCESS_TOKEN = orig_token
            _cfg.SUPABASE_CLIENT = orig_client
