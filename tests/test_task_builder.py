"""Tests for source/task_handlers/join/task_builder.py.

This module depends on db_operations and _check_orchestrator_cancelled,
so we mock those and test the task creation logic.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call


class TestCreateJoinChainTasks:
    """Test _create_join_chain_tasks with mocked DB operations."""

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_basic_two_clip_join(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        mock_db.add_task_to_db.side_effect = ["join-1", "stitch-1"]

        clips = [{"url": "/clip1.mp4"}, {"url": "/clip2.mp4"}]
        success, msg = _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "smooth transition", "context_frame_count": 8},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output/run-1"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id="proj-1",
            orchestrator_payload={"fps": 24},
            parent_generation_id="gen-1",
        )

        assert success is True
        assert "1 chain joins" in msg
        assert "1 final stitch" in msg
        # Should create 1 join + 1 stitch = 2 db calls
        assert mock_db.add_task_to_db.call_count == 2

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_three_clips_creates_two_joins(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        mock_db.add_task_to_db.side_effect = ["join-1", "join-2", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}, {"url": "/c.mp4"}]
        success, msg = _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "context_frame_count": 8},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        assert success is True
        assert "2 chain joins" in msg
        assert mock_db.add_task_to_db.call_count == 3

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_single_clip_fails(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        clips = [{"url": "/clip1.mp4"}]
        success, msg = _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        assert success is False
        assert "at least 2 clips" in msg

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled")
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_cancellation_during_join_creation(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        # First check passes, second returns cancellation
        mock_cancel.side_effect = [None, "Orchestrator cancelled: test"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}, {"url": "/c.mp4"}]
        mock_db.add_task_to_db.return_value = "join-1"

        success, msg = _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test"},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        assert success is False
        assert "cancelled" in msg.lower()

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_per_join_settings_applied(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        mock_db.add_task_to_db.side_effect = ["join-1", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}]
        per_join = [{"prompt": "custom transition", "guidance_scale": 3.0}]

        _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "default"},
            per_join_settings=per_join,
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        # Check the first call (join task) payload
        first_call_payload = mock_db.add_task_to_db.call_args_list[0][1]["task_payload"]
        assert first_call_payload["prompt"] == "custom transition"
        assert first_call_payload["guidance_scale"] == 3.0

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_vlm_prompt_override(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        mock_db.add_task_to_db.side_effect = ["join-1", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}]

        _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "default"},
            per_join_settings=[],
            vlm_enhanced_prompts=["vlm-generated transition"],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        first_call_payload = mock_db.add_task_to_db.call_args_list[0][1]["task_payload"]
        assert first_call_payload["prompt"] == "vlm-generated transition"

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_chain_dependency_structure(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        mock_db.add_task_to_db.side_effect = ["join-1", "join-2", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}, {"url": "/c.mp4"}]

        _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "context_frame_count": 8},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        calls = mock_db.add_task_to_db.call_args_list
        # First join: no dependency
        assert calls[0][1]["dependant_on"] is None
        # Second join: depends on first
        assert calls[1][1]["dependant_on"] == "join-1"
        # Stitch: depends on last join
        assert calls[2][1]["dependant_on"] == "join-2"

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_final_stitch_payload(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_join_chain_tasks

        mock_db.add_task_to_db.side_effect = ["join-1", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}]

        _create_join_chain_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "fps": 30, "context_frame_count": 10},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id="proj-1",
            orchestrator_payload={"audio_url": "https://example.com/audio.mp3"},
            parent_generation_id="gen-1",
        )

        stitch_call = mock_db.add_task_to_db.call_args_list[1]
        stitch_payload = stitch_call[1]["task_payload"]
        assert stitch_payload["chain_mode"] is True
        assert stitch_payload["fps"] == 30
        assert stitch_payload["blend_frames"] == 10  # min(10, 15) = 10
        assert stitch_payload["audio_url"] == "https://example.com/audio.mp3"
        assert stitch_payload["parent_generation_id"] == "gen-1"
        assert stitch_call[1]["task_type_str"] == "join_final_stitch"


class TestCreateParallelJoinTasks:
    """Test _create_parallel_join_tasks with mocked DB operations."""

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_basic_parallel_two_clips(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        mock_db.add_task_to_db.side_effect = ["trans-1", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}]
        success, msg = _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "context_frame_count": 8},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        assert success is True
        assert "1 parallel transitions" in msg
        assert "1 final stitch" in msg

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_parallel_no_dependencies_between_transitions(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        mock_db.add_task_to_db.side_effect = ["trans-1", "trans-2", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}, {"url": "/c.mp4"}]

        _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "context_frame_count": 8},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        calls = mock_db.add_task_to_db.call_args_list
        # Transitions should have NO dependencies
        assert calls[0][1]["dependant_on"] is None
        assert calls[1][1]["dependant_on"] is None
        # Stitch depends on ALL transitions
        assert calls[2][1]["dependant_on"] == ["trans-1", "trans-2"]

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_single_clip_fails(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        clips = [{"url": "/a.mp4"}]
        success, msg = _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        assert success is False
        assert "at least 2 clips" in msg

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled")
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_cancellation_during_transition_creation(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        mock_cancel.side_effect = [None, "Orchestrator cancelled: test"]
        mock_db.add_task_to_db.return_value = "trans-1"

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}, {"url": "/c.mp4"}]
        success, msg = _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test"},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        assert success is False
        assert "cancelled" in msg.lower()

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_transition_payload_fields(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        mock_db.add_task_to_db.side_effect = ["trans-1", "stitch-1"]

        clips = [{"url": "/a.mp4", "name": "clip_a"}, {"url": "/b.mp4", "name": "clip_b"}]

        _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "smooth"},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id="proj-1",
            orchestrator_payload={},
            parent_generation_id=None,
        )

        trans_payload = mock_db.add_task_to_db.call_args_list[0][1]["task_payload"]
        assert trans_payload["transition_only"] is True
        assert trans_payload["starting_video_path"] == "/a.mp4"
        assert trans_payload["ending_video_path"] == "/b.mp4"
        assert trans_payload["transition_index"] == 0
        assert trans_payload["orchestrator_run_id"] == "run-1"
        assert trans_payload["project_id"] == "proj-1"

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_parallel_final_stitch_has_all_transitions(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        mock_db.add_task_to_db.side_effect = ["trans-1", "trans-2", "trans-3", "stitch-1"]

        clips = [{"url": f"/{i}.mp4"} for i in range(4)]

        _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "context_frame_count": 8},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={"fps": 30},
            parent_generation_id=None,
        )

        stitch_call = mock_db.add_task_to_db.call_args_list[3]
        stitch_payload = stitch_call[1]["task_payload"]
        assert stitch_payload["transition_task_ids"] == ["trans-1", "trans-2", "trans-3"]
        assert stitch_payload["fps"] == 30
        assert stitch_call[1]["dependant_on"] == ["trans-1", "trans-2", "trans-3"]

    @patch("source.task_handlers.join.task_builder._check_orchestrator_cancelled", return_value=None)
    @patch("source.task_handlers.join.task_builder.db_ops")
    def test_blend_frames_capped_at_15(self, mock_db, mock_cancel):
        from source.task_handlers.join.task_builder import _create_parallel_join_tasks

        mock_db.add_task_to_db.side_effect = ["trans-1", "stitch-1"]

        clips = [{"url": "/a.mp4"}, {"url": "/b.mp4"}]

        _create_parallel_join_tasks(
            clip_list=clips,
            run_id="run-1",
            join_settings={"prompt": "test", "context_frame_count": 30},
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=Path("/output"),
            orchestrator_task_id_str="orch-1",
            orchestrator_project_id=None,
            orchestrator_payload={},
            parent_generation_id=None,
        )

        stitch_payload = mock_db.add_task_to_db.call_args_list[1][1]["task_payload"]
        assert stitch_payload["blend_frames"] == 15  # min(30, 15) = 15
