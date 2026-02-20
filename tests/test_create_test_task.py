"""Tests for scripts/create_test_task.py."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from scripts.create_test_task import (
    TEST_TASKS,
    create_task,
    list_tasks,
    create_all_tasks,
    main,
)


# ── TEST_TASKS structure ─────────────────────────────────────────────────────

class TestTaskTemplates:
    """Validate the built-in test task templates."""

    def test_all_templates_have_required_fields(self):
        for name, template in TEST_TASKS.items():
            assert "task_type" in template, f"{name} missing task_type"
            assert "params" in template, f"{name} missing params"
            assert "project_id" in template, f"{name} missing project_id"
            assert "description" in template, f"{name} missing description"

    def test_known_task_types_present(self):
        assert "uni3c_basic" in TEST_TASKS
        assert "travel_orchestrator" in TEST_TASKS
        assert "qwen_image_style" in TEST_TASKS

    def test_task_types_are_valid_strings(self):
        valid_types = {"individual_travel_segment", "travel_orchestrator", "qwen_image_style"}
        for name, template in TEST_TASKS.items():
            assert template["task_type"] in valid_types, (
                f"{name} has unexpected task_type: {template['task_type']}"
            )


# ── create_task ──────────────────────────────────────────────────────────────

class TestCreateTask:
    """Test task creation logic."""

    def test_unknown_task_type_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            create_task("nonexistent_type")
        assert exc_info.value.code == 1

    def test_dry_run_returns_id_without_network(self):
        """Dry run should not make any HTTP calls."""
        task_id = create_task("uni3c_basic", dry_run=True)
        assert task_id is not None
        assert len(task_id) == 36  # UUID format

    def test_dry_run_travel_orchestrator(self):
        task_id = create_task("travel_orchestrator", dry_run=True)
        assert task_id is not None

    def test_dry_run_qwen_image_style(self):
        task_id = create_task("qwen_image_style", dry_run=True)
        assert task_id is not None

    def test_create_with_supabase_success(self):
        """Successful Supabase insert returns task ID."""
        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_httpx.post.return_value = mock_response

        with patch.dict("os.environ", {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        }), patch.dict("sys.modules", {"httpx": mock_httpx}):
            task_id = create_task("uni3c_basic", dry_run=False)

        assert task_id is not None
        mock_httpx.post.assert_called_once()

        # Verify the POST payload
        call_kwargs = mock_httpx.post.call_args
        posted_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert posted_json["status"] == "Queued"
        assert posted_json["task_type"] == "individual_travel_segment"

    def test_create_missing_env_vars_exits(self):
        """Missing SUPABASE_URL or key should exit with code 1."""
        mock_httpx = MagicMock()
        with patch.dict("os.environ", {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=True), \
             patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(SystemExit) as exc_info:
                create_task("uni3c_basic", dry_run=False)
            assert exc_info.value.code == 1

    def test_create_http_error_exits(self):
        """HTTP error during insert exits with code 1."""
        mock_httpx = MagicMock()

        class FakeHTTPError(Exception):
            pass

        mock_httpx.post.side_effect = FakeHTTPError("connection failed")
        mock_httpx.HTTPError = FakeHTTPError

        with patch.dict("os.environ", {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        }), patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(SystemExit) as exc_info:
                create_task("uni3c_basic", dry_run=False)
            assert exc_info.value.code == 1


# ── list_tasks ───────────────────────────────────────────────────────────────

class TestListTasks:
    def test_list_tasks_no_error(self, capsys):
        list_tasks()
        captured = capsys.readouterr()
        # Should list all task names
        for name in TEST_TASKS:
            assert name in captured.out


# ── create_all_tasks ─────────────────────────────────────────────────────────

class TestCreateAllTasks:
    def test_dry_run_all(self):
        """Dry run of all tasks should succeed without network."""
        create_all_tasks(dry_run=True)


# ── main (argparse) ──────────────────────────────────────────────────────────

class TestMain:
    def test_list_flag(self, capsys):
        with patch("sys.argv", ["prog", "--list"]):
            main()
        captured = capsys.readouterr()
        assert "uni3c_basic" in captured.out

    def test_dry_run_flag(self):
        with patch("sys.argv", ["prog", "--dry-run", "uni3c_basic"]):
            main()  # Should not raise

    def test_all_dry_run(self):
        with patch("sys.argv", ["prog", "--all", "--dry-run"]):
            main()  # Should not raise

    def test_no_args_prints_help(self, capsys):
        with patch("sys.argv", ["prog"]):
            main()
        captured = capsys.readouterr()
        # Should print help and list
        assert "uni3c_basic" in captured.out
