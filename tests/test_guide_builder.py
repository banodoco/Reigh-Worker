"""Tests for source/task_handlers/travel/guide_builder.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from source.task_handlers.travel.guide_builder import (
    get_previous_segment_video,
    prepare_input_images_for_guide,
    create_guide_video,
)


def _make_ctx(**overrides):
    """Create a mock TravelSegmentContext with sensible defaults."""
    ctx = MagicMock()
    ctx.task_id = "task-abc-123"
    ctx.segment_idx = 1
    ctx.orchestrator_details = {
        "chain_segments": True,
        "input_image_paths_resolved": ["/img/a.png", "/img/b.png", "/img/c.png"],
        "fps_helpers": 16,
    }
    ctx.segment_params = {
        "is_first_segment": False,
    }
    ctx.segment_processing_dir = Path("/tmp/test_processing")
    ctx.main_output_dir_base = "/tmp/output"
    ctx.total_frames_for_segment = 81
    ctx.parsed_res_wh = (832, 480)
    ctx.debug_enabled = True
    ctx.model_name = "wan2.1-vace-14b"

    # Apply overrides
    for k, v in overrides.items():
        if isinstance(v, dict) and hasattr(ctx, k) and isinstance(getattr(ctx, k), dict):
            getattr(ctx, k).update(v)
        else:
            setattr(ctx, k, v)
    return ctx


def _make_proc(ctx=None, is_vace=True):
    """Create a mock TravelSegmentProcessor."""
    proc = MagicMock()
    proc.ctx = ctx or _make_ctx()
    proc.is_vace_model = is_vace
    proc._detected_structure_type = None
    proc._structure_config = None
    proc._detect_single_image_journey.return_value = False
    return proc


# ---------------------------------------------------------------------------
# get_previous_segment_video
# ---------------------------------------------------------------------------

class TestGetPreviousSegmentVideo:
    """Tests for get_previous_segment_video."""

    def test_chain_segments_false_returns_none(self):
        """When chain_segments=False, should return None (independent segments)."""
        proc = _make_proc()
        proc.ctx.orchestrator_details["chain_segments"] = False
        result = get_previous_segment_video(proc)
        assert result is None

    def test_first_segment_with_continue_video(self):
        """First segment with continue_from_video_resolved_path should return that path."""
        proc = _make_proc()
        proc.ctx.segment_params["is_first_segment"] = True
        proc.ctx.segment_idx = 0
        proc.ctx.orchestrator_details["continue_from_video_resolved_path"] = "/videos/continue.mp4"
        result = get_previous_segment_video(proc)
        assert result == "/videos/continue.mp4"

    def test_first_segment_from_scratch_returns_none(self):
        """First segment without continue_from_video returns None."""
        proc = _make_proc()
        proc.ctx.segment_params["is_first_segment"] = True
        proc.ctx.segment_idx = 0
        result = get_previous_segment_video(proc)
        assert result is None

    def test_subsequent_segment_local_path(self):
        """Subsequent segment with local predecessor path returns it directly."""
        proc = _make_proc()
        proc.ctx.segment_params["is_first_segment"] = False
        proc.ctx.segment_idx = 2

        with patch("source.task_handlers.travel.guide_builder.db_ops") as mock_db:
            mock_db.get_predecessor_output_via_edge_function.return_value = ("dep-id", "/local/previous.mp4")
            result = get_previous_segment_video(proc)

        assert result == "/local/previous.mp4"

    def test_subsequent_segment_remote_url_downloaded(self, tmp_path):
        """Remote URL predecessor should be downloaded locally."""
        proc = _make_proc()
        proc.ctx.segment_params["is_first_segment"] = False
        proc.ctx.segment_idx = 1
        proc.ctx.segment_processing_dir = tmp_path

        local_path = tmp_path / "prev_01_segment_00.mp4"

        with patch("source.task_handlers.travel.guide_builder.db_ops") as mock_db, \
             patch("source.utils.download_file") as mock_download:
            mock_db.get_predecessor_output_via_edge_function.return_value = (
                "dep-id", "https://storage.example.com/segment_00.mp4"
            )

            # Simulate download creating the file
            def fake_download(url, dest_dir, filename):
                (dest_dir / filename).write_bytes(b"video_data")

            mock_download.side_effect = fake_download

            result = get_previous_segment_video(proc)

        assert result is not None
        assert "prev_01_" in result

    def test_no_predecessor_returns_none(self):
        """When DB returns no predecessor, should return None."""
        proc = _make_proc()
        proc.ctx.segment_params["is_first_segment"] = False

        with patch("source.task_handlers.travel.guide_builder.db_ops") as mock_db:
            mock_db.get_predecessor_output_via_edge_function.return_value = (None, None)
            result = get_previous_segment_video(proc)

        assert result is None


# ---------------------------------------------------------------------------
# prepare_input_images_for_guide
# ---------------------------------------------------------------------------

class TestPrepareInputImagesForGuide:
    """Tests for prepare_input_images_for_guide."""

    def test_individual_segment_images_preferred(self):
        """Individual segment params images should be used when available."""
        proc = _make_proc()
        proc.ctx.segment_params["individual_segment_params"] = {
            "input_image_paths_resolved": ["/img/start.png", "/img/end.png"]
        }
        result = prepare_input_images_for_guide(proc)
        assert result == ["/img/start.png", "/img/end.png"]

    def test_falls_back_to_orchestrator_images(self):
        """Without individual params, should use orchestrator images."""
        proc = _make_proc()
        proc.ctx.segment_params = {}  # No individual params
        result = prepare_input_images_for_guide(proc)
        assert result == ["/img/a.png", "/img/b.png", "/img/c.png"]

    def test_returns_copy_not_reference(self):
        """Should return a copy of the image list, not a reference."""
        proc = _make_proc()
        proc.ctx.segment_params = {}
        result = prepare_input_images_for_guide(proc)
        original = proc.ctx.orchestrator_details["input_image_paths_resolved"]
        assert result == original
        assert result is not original

    def test_empty_individual_images_uses_orchestrator(self):
        """Empty individual images list should fall back to orchestrator."""
        proc = _make_proc()
        proc.ctx.segment_params["individual_segment_params"] = {
            "input_image_paths_resolved": []
        }
        result = prepare_input_images_for_guide(proc)
        assert result == ["/img/a.png", "/img/b.png", "/img/c.png"]


# ---------------------------------------------------------------------------
# create_guide_video
# ---------------------------------------------------------------------------

class TestCreateGuideVideo:
    """Tests for create_guide_video."""

    def test_non_vace_non_debug_skips(self):
        """Non-VACE model with debug disabled should skip guide creation."""
        proc = _make_proc(is_vace=False)
        proc.ctx.debug_enabled = False
        result = create_guide_video(proc)
        assert result is None

    def test_vace_model_creates_guide(self, tmp_path):
        """VACE model should create guide video."""
        ctx = _make_ctx(
            segment_processing_dir=tmp_path,
            main_output_dir_base=str(tmp_path),
        )
        proc = _make_proc(ctx=ctx, is_vace=True)
        proc.ctx.debug_enabled = False

        guide_path = tmp_path / "guide.mp4"

        mock_config = MagicMock()
        mock_config.videos = []
        mock_config.legacy_structure_type = None
        mock_config.strength = 1.0
        mock_config.canny_intensity = 1.0
        mock_config.depth_contrast = 1.0
        mock_config.guidance_video_url = None
        mock_config._frame_offset = 0

        with patch("source.task_handlers.travel.guide_builder.prepare_output_path",
                   return_value=(str(guide_path), str(tmp_path))), \
             patch("source.task_handlers.travel.guide_builder.get_previous_segment_video", return_value=None), \
             patch("source.task_handlers.travel.guide_builder.prepare_input_images_for_guide",
                   return_value=["/img/a.png", "/img/b.png"]), \
             patch("source.task_handlers.travel.guide_builder.StructureGuidanceConfig.from_params",
                   return_value=mock_config), \
             patch("source.task_handlers.travel.guide_builder.create_guide_video_for_travel_segment",
                   return_value=str(guide_path)) as mock_create:
            guide_path.write_bytes(b"guide_video_data")
            result = create_guide_video(proc)

        assert result == guide_path
        mock_create.assert_called_once()

    def test_vace_guide_failure_raises(self, tmp_path):
        """VACE model with failed guide creation should raise ValueError."""
        ctx = _make_ctx(
            segment_processing_dir=tmp_path,
            main_output_dir_base=str(tmp_path),
        )
        proc = _make_proc(ctx=ctx, is_vace=True)
        proc.ctx.debug_enabled = False

        mock_config = MagicMock()
        mock_config.videos = []
        mock_config.legacy_structure_type = None
        mock_config.strength = 1.0
        mock_config.canny_intensity = 1.0
        mock_config.depth_contrast = 1.0
        mock_config.guidance_video_url = None
        mock_config._frame_offset = 0

        with patch("source.task_handlers.travel.guide_builder.prepare_output_path",
                   return_value=("/fake/guide.mp4", "/fake")), \
             patch("source.task_handlers.travel.guide_builder.get_previous_segment_video", return_value=None), \
             patch("source.task_handlers.travel.guide_builder.prepare_input_images_for_guide",
                   return_value=["/img/a.png", "/img/b.png"]), \
             patch("source.task_handlers.travel.guide_builder.StructureGuidanceConfig.from_params",
                   return_value=mock_config), \
             patch("source.task_handlers.travel.guide_builder.create_guide_video_for_travel_segment",
                   return_value=None):
            with pytest.raises(ValueError, match="requires guide video"):
                create_guide_video(proc)

    def test_debug_mode_creates_guide_for_non_vace(self, tmp_path):
        """Debug mode should create guide even for non-VACE models."""
        ctx = _make_ctx(
            segment_processing_dir=tmp_path,
            main_output_dir_base=str(tmp_path),
            debug_enabled=True,
        )
        proc = _make_proc(ctx=ctx, is_vace=False)

        guide_path = tmp_path / "guide.mp4"

        mock_config = MagicMock()
        mock_config.videos = []
        mock_config.legacy_structure_type = None
        mock_config.strength = 1.0
        mock_config.canny_intensity = 1.0
        mock_config.depth_contrast = 1.0
        mock_config.guidance_video_url = None
        mock_config._frame_offset = 0

        with patch("source.task_handlers.travel.guide_builder.prepare_output_path",
                   return_value=(str(guide_path), str(tmp_path))), \
             patch("source.task_handlers.travel.guide_builder.get_previous_segment_video", return_value=None), \
             patch("source.task_handlers.travel.guide_builder.prepare_input_images_for_guide",
                   return_value=["/img/a.png", "/img/b.png"]), \
             patch("source.task_handlers.travel.guide_builder.StructureGuidanceConfig.from_params",
                   return_value=mock_config), \
             patch("source.task_handlers.travel.guide_builder.create_guide_video_for_travel_segment",
                   return_value=str(guide_path)):
            guide_path.write_bytes(b"guide_video_data")
            result = create_guide_video(proc)

        assert result == guide_path

    def test_chain_segments_false_forces_first_segment(self, tmp_path):
        """chain_segments=False should force is_first_segment_from_scratch=True."""
        ctx = _make_ctx(
            segment_processing_dir=tmp_path,
            main_output_dir_base=str(tmp_path),
        )
        ctx.orchestrator_details["chain_segments"] = False
        proc = _make_proc(ctx=ctx, is_vace=True)
        proc.ctx.debug_enabled = False

        guide_path = tmp_path / "guide.mp4"

        mock_config = MagicMock()
        mock_config.videos = []
        mock_config.legacy_structure_type = None
        mock_config.strength = 1.0
        mock_config.canny_intensity = 1.0
        mock_config.depth_contrast = 1.0
        mock_config.guidance_video_url = None
        mock_config._frame_offset = 0

        with patch("source.task_handlers.travel.guide_builder.prepare_output_path",
                   return_value=(str(guide_path), str(tmp_path))), \
             patch("source.task_handlers.travel.guide_builder.get_previous_segment_video", return_value=None), \
             patch("source.task_handlers.travel.guide_builder.prepare_input_images_for_guide",
                   return_value=["/img/a.png", "/img/b.png"]), \
             patch("source.task_handlers.travel.guide_builder.StructureGuidanceConfig.from_params",
                   return_value=mock_config), \
             patch("source.task_handlers.travel.guide_builder.create_guide_video_for_travel_segment",
                   return_value=str(guide_path)) as mock_create:
            guide_path.write_bytes(b"guide_video_data")
            create_guide_video(proc)

        # Verify is_first_segment_from_scratch was True in the call
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["is_first_segment_from_scratch"] is True
