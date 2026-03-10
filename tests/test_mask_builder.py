"""Tests for source/task_handlers/travel/mask_builder.py."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def _make_proc(
    mask_active_frames=True,
    segment_idx=0,
    task_id="task_123",
    total_frames=81,
    parsed_res_wh=(1280, 720),
    frame_overlap_from_previous=0,
    chain_segments=True,
    is_first_segment=True,
    continue_from_video=None,
    debug_enabled=True,
    is_vace_model=True,
    single_image_journey=False,
    consolidated_keyframe_positions=None,
    fps_helpers=16,
    main_output_dir_base=Path("/output"),
):
    """Build a mock TravelSegmentProcessor with a nested ctx."""
    proc = MagicMock()
    ctx = MagicMock()

    ctx.mask_active_frames = mask_active_frames
    ctx.segment_idx = segment_idx
    ctx.task_id = task_id
    ctx.total_frames_for_segment = total_frames
    ctx.parsed_res_wh = parsed_res_wh
    ctx.debug_enabled = debug_enabled
    ctx.main_output_dir_base = main_output_dir_base

    segment_params = {
        "frame_overlap_from_previous": frame_overlap_from_previous,
        "is_first_segment": is_first_segment,
    }
    if consolidated_keyframe_positions is not None:
        segment_params["consolidated_keyframe_positions"] = consolidated_keyframe_positions
    ctx.segment_params = segment_params

    orchestrator_details = {
        "chain_segments": chain_segments,
        "fps_helpers": fps_helpers,
    }
    if continue_from_video is not None:
        orchestrator_details["continue_from_video_resolved_path"] = continue_from_video
    ctx.orchestrator_details = orchestrator_details

    proc.ctx = ctx
    proc.is_vace_model = is_vace_model
    proc._detect_single_image_journey.return_value = single_image_journey

    return proc


class TestCreateMaskVideo:
    """Tests for create_mask_video."""

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    def test_mask_disabled_returns_none(self, mock_logger):
        """When mask_active_frames is False, returns None immediately."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        proc = _make_proc(mask_active_frames=False)
        result = create_mask_video(proc)
        assert result is None

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.get_video_frame_count_and_fps")
    @patch("source.task_handlers.travel.mask_builder.create_mask_video_from_inactive_indices")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_vace_model_creates_mask(
        self, mock_prepare, mock_create_mask, mock_get_info, mock_logger, tmp_path
    ):
        """VACE model with debug disabled still creates mask video."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mask_path = tmp_path / "mask.mp4"
        mask_path.write_bytes(b"fake mask")

        mock_prepare.return_value = (mask_path, "db://mask")
        mock_create_mask.return_value = mask_path
        mock_get_info.return_value = (81, 16.0)

        proc = _make_proc(
            debug_enabled=False,
            is_vace_model=True,
            is_first_segment=True,
        )

        result = create_mask_video(proc)
        assert result == mask_path
        mock_create_mask.assert_called_once()

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_non_vace_no_debug_skips_mask(self, mock_prepare, mock_logger, tmp_path):
        """Non-VACE model with debug disabled skips mask creation."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mock_prepare.return_value = (tmp_path / "mask.mp4", "db://mask")

        proc = _make_proc(debug_enabled=False, is_vace_model=False)
        result = create_mask_video(proc)
        assert result is None

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.get_video_frame_count_and_fps")
    @patch("source.task_handlers.travel.mask_builder.create_mask_video_from_inactive_indices")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_overlap_frames_marked_inactive(
        self, mock_prepare, mock_create_mask, mock_get_info, mock_logger, tmp_path
    ):
        """Overlap frames should be in the inactive set."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mask_path = tmp_path / "mask.mp4"
        mask_path.write_bytes(b"fake")
        mock_prepare.return_value = (mask_path, "db://mask")
        mock_create_mask.return_value = mask_path
        mock_get_info.return_value = (81, 16.0)

        proc = _make_proc(
            frame_overlap_from_previous=5,
            is_first_segment=False,
            chain_segments=True,
            single_image_journey=False,
            total_frames=81,
        )

        create_mask_video(proc)

        # Check the inactive_frame_indices argument
        call_kwargs = mock_create_mask.call_args[1]
        inactive = call_kwargs["inactive_frame_indices"]
        # Overlap frames 0-4 should be inactive
        for i in range(5):
            assert i in inactive
        # Last frame should also be inactive (multi-image journey, not single)
        assert 80 in inactive

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.get_video_frame_count_and_fps")
    @patch("source.task_handlers.travel.mask_builder.create_mask_video_from_inactive_indices")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_chain_segments_false_forces_overlap_zero(
        self, mock_prepare, mock_create_mask, mock_get_info, mock_logger, tmp_path
    ):
        """Independent mode (chain_segments=False) sets overlap to 0 and marks frame 0."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mask_path = tmp_path / "mask.mp4"
        mask_path.write_bytes(b"fake")
        mock_prepare.return_value = (mask_path, "db://mask")
        mock_create_mask.return_value = mask_path
        mock_get_info.return_value = (81, 16.0)

        proc = _make_proc(
            frame_overlap_from_previous=10,
            chain_segments=False,
            is_first_segment=False,
            single_image_journey=False,
            total_frames=81,
        )

        create_mask_video(proc)

        call_kwargs = mock_create_mask.call_args[1]
        inactive = call_kwargs["inactive_frame_indices"]
        # Frame 0 should be marked (independent mode anchor)
        assert 0 in inactive
        # Overlap frames 1-9 should NOT be in inactive since chain_segments=False
        for i in range(1, 10):
            assert i not in inactive

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.get_video_frame_count_and_fps")
    @patch("source.task_handlers.travel.mask_builder.create_mask_video_from_inactive_indices")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_single_image_journey_no_last_frame(
        self, mock_prepare, mock_create_mask, mock_get_info, mock_logger, tmp_path
    ):
        """Single image journey does NOT mark the last frame as inactive."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mask_path = tmp_path / "mask.mp4"
        mask_path.write_bytes(b"fake")
        mock_prepare.return_value = (mask_path, "db://mask")
        mock_create_mask.return_value = mask_path
        mock_get_info.return_value = (81, 16.0)

        proc = _make_proc(
            single_image_journey=True,
            is_first_segment=True,
            total_frames=81,
        )

        create_mask_video(proc)

        call_kwargs = mock_create_mask.call_args[1]
        inactive = call_kwargs["inactive_frame_indices"]
        # Last frame (80) should NOT be in inactive set
        assert 80 not in inactive
        # Frame 0 should be there (first segment from scratch)
        assert 0 in inactive

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.get_video_frame_count_and_fps")
    @patch("source.task_handlers.travel.mask_builder.create_mask_video_from_inactive_indices")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_consolidated_keyframe_positions(
        self, mock_prepare, mock_create_mask, mock_get_info, mock_logger, tmp_path
    ):
        """Consolidated keyframe positions should all be marked inactive."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mask_path = tmp_path / "mask.mp4"
        mask_path.write_bytes(b"fake")
        mock_prepare.return_value = (mask_path, "db://mask")
        mock_create_mask.return_value = mask_path
        mock_get_info.return_value = (81, 16.0)

        proc = _make_proc(
            is_first_segment=True,
            consolidated_keyframe_positions=[0, 20, 40, 60, 80],
            total_frames=81,
            single_image_journey=False,
        )

        create_mask_video(proc)

        call_kwargs = mock_create_mask.call_args[1]
        inactive = call_kwargs["inactive_frame_indices"]
        for pos in [0, 20, 40, 60, 80]:
            assert pos in inactive

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    @patch("source.task_handlers.travel.mask_builder.create_mask_video_from_inactive_indices")
    @patch("source.task_handlers.travel.mask_builder.prepare_output_path")
    def test_mask_creation_fails_returns_none(
        self, mock_prepare, mock_create_mask, mock_logger, tmp_path
    ):
        """Returns None when create_mask_video_from_inactive_indices returns None."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        mock_prepare.return_value = (tmp_path / "mask.mp4", "db://mask")
        mock_create_mask.return_value = None

        proc = _make_proc(is_vace_model=True, debug_enabled=True)
        result = create_mask_video(proc)
        assert result is None

    @patch("source.task_handlers.travel.mask_builder.travel_logger")
    def test_exception_returns_none(self, mock_logger):
        """Returns None when an exception is raised internally."""
        from source.task_handlers.travel.mask_builder import create_mask_video

        proc = _make_proc()
        # Make segment_params.get raise to trigger exception path
        proc.ctx.segment_params = MagicMock()
        proc.ctx.segment_params.get.side_effect = RuntimeError("boom")

        result = create_mask_video(proc)
        assert result is None


class TestCreateMaskVideoDirect:
    """Direct behavior checks with monkeypatch stubs."""

    def test_inactive_index_composition_for_overlap_first_and_last(self, monkeypatch, tmp_path):
        import source.task_handlers.travel.mask_builder as mb

        captured = {}

        class _Ctx:
            mask_active_frames = True
            segment_idx = 2
            task_id = "task-direct-mask"
            total_frames_for_segment = 10
            parsed_res_wh = (640, 360)
            debug_enabled = True
            main_output_dir_base = tmp_path
            segment_params = {
                "frame_overlap_from_previous": 3,
                "is_first_segment": True,
                "consolidated_keyframe_positions": [4, 9],
            }
            orchestrator_details = {
                "chain_segments": True,
                "fps_helpers": 12,
            }

        class _Proc:
            def __init__(self):
                self.ctx = _Ctx()
                self.is_vace_model = True

            @staticmethod
            def _detect_single_image_journey():
                return False

        def _prepare_output_path(task_id, filename, main_output_dir_base, task_type):
            return (tmp_path / filename, f"db://{task_id}/{task_type}")

        def _create_mask(**kwargs):
            captured.update(kwargs)
            out = kwargs["output_path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"mask")
            return out

        monkeypatch.setattr(mb, "prepare_output_path", _prepare_output_path)
        monkeypatch.setattr(mb, "create_mask_video_from_inactive_indices", _create_mask)
        monkeypatch.setattr(mb, "get_video_frame_count_and_fps", lambda _path: (10, 12.0))

        result = mb.create_mask_video(_Proc())

        assert result is not None
        assert result.exists()
        assert result.suffix == ".mp4"
        assert result.stat().st_size > 0
        assert "task-direct-mask" in result.name
        assert "_seg02_" in result.name
        assert captured["total_frames"] == 10
        assert captured["resolution_wh"] == (640, 360)
        assert captured["fps"] == 12
        assert captured["task_id_for_logging"] == "task-direct-mask"
        assert captured["output_path"] == result
        inactive = captured["inactive_frame_indices"]
        assert isinstance(inactive, set)
        assert 0 in inactive
        assert 1 in inactive
        assert 2 in inactive
        assert 3 not in inactive
        assert 4 in inactive
        assert 5 not in inactive
        assert 6 not in inactive
        assert 7 not in inactive
        assert 8 not in inactive
        assert 9 in inactive
        assert len(inactive) >= 5
