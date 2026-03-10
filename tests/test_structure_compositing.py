"""Tests for source/media/structure/compositing.py."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestValidateStructureVideoConfigs:
    """Tests for validate_structure_video_configs."""

    def test_empty_configs_returns_empty(self):
        from source.media.structure.compositing import validate_structure_video_configs

        result = validate_structure_video_configs([], 100)
        assert result == []

    def test_valid_single_config(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/video.mp4", "start_frame": 0, "end_frame": 50}]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 1
        assert result[0]["start_frame"] == 0
        assert result[0]["end_frame"] == 50

    def test_valid_multiple_non_overlapping(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 30},
            {"path": "/b.mp4", "start_frame": 30, "end_frame": 60},
            {"path": "/c.mp4", "start_frame": 60, "end_frame": 100},
        ]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 3

    def test_configs_sorted_by_start_frame(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [
            {"path": "/b.mp4", "start_frame": 50, "end_frame": 80},
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 30},
        ]
        result = validate_structure_video_configs(configs, 100)
        assert result[0]["start_frame"] == 0
        assert result[1]["start_frame"] == 50

    def test_missing_path_raises(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"start_frame": 0, "end_frame": 50}]
        with pytest.raises(ValueError, match="missing 'path'"):
            validate_structure_video_configs(configs, 100)

    def test_missing_start_frame_raises(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "end_frame": 50}]
        with pytest.raises(ValueError, match="missing 'start_frame'"):
            validate_structure_video_configs(configs, 100)

    def test_missing_end_frame_raises(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "start_frame": 0}]
        with pytest.raises(ValueError, match="missing 'end_frame'"):
            validate_structure_video_configs(configs, 100)

    def test_negative_start_frame_raises(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "start_frame": -1, "end_frame": 50}]
        with pytest.raises(ValueError, match="start_frame -1 < 0"):
            validate_structure_video_configs(configs, 100)

    def test_start_frame_beyond_total_skipped(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "start_frame": 200, "end_frame": 300}]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 0

    def test_end_frame_clipped_to_total(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "start_frame": 0, "end_frame": 200}]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 1
        assert result[0]["end_frame"] == 100

    def test_start_equals_end_raises(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "start_frame": 50, "end_frame": 50}]
        with pytest.raises(ValueError, match="start_frame 50 >= end_frame 50"):
            validate_structure_video_configs(configs, 100)

    def test_overlapping_configs_raises(self):
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 50},
            {"path": "/b.mp4", "start_frame": 40, "end_frame": 80},
        ]
        with pytest.raises(ValueError, match="overlaps with previous"):
            validate_structure_video_configs(configs, 100)


class TestCreateCompositeGuidanceVideo:
    """Tests for create_composite_guidance_video (heavily mocked)."""

    @patch("source.media.structure.compositing.process_structure_frames")
    @patch("source.media.structure.compositing.load_structure_video_frames_with_range")
    @patch("source.media.structure.compositing.create_neutral_frame")
    def test_basic_composite_creation(
        self, mock_neutral, mock_load, mock_process
    ):
        """Should create a composite video from configs."""
        from source.media.structure.compositing import create_composite_guidance_video

        neutral = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_neutral.return_value = neutral

        source_frames = [np.ones((64, 64, 3), dtype=np.uint8) * 128 for _ in range(5)]
        mock_load.return_value = source_frames

        processed_frames = [np.ones((64, 64, 3), dtype=np.uint8) * 200 for _ in range(5)]
        mock_process.return_value = processed_frames

        configs = [{"path": "/src.mp4", "start_frame": 0, "end_frame": 5}]
        output_path = MagicMock(spec=Path)
        output_path.parent = MagicMock()
        output_path.exists.return_value = True
        output_path.stat.return_value = MagicMock(st_size=1024)
        output_path.name = "composite.mp4"

        import sys
        # Mock the save_video import path
        mock_save_video = MagicMock()
        fake_modules = {
            "shared": MagicMock(),
            "shared.utils": MagicMock(),
            "shared.utils.audio_video": MagicMock(save_video=mock_save_video),
        }

        with patch.dict(sys.modules, fake_modules), \
             patch("source.media.structure.compositing.torch") as mock_torch, \
             patch("source.media.structure.compositing.Path") as mock_path_cls, \
             patch("source.media.structure.compositing.sys") as mock_sys:
            mock_torch.cuda.is_available.return_value = False
            # Make the Wan2GP path setup work
            mock_parent = MagicMock()
            mock_parent.parent = mock_parent
            mock_path_cls.return_value.parent = mock_parent
            mock_parent.__truediv__ = MagicMock(return_value=MagicMock(__str__=lambda s: "/fake/Wan2GP"))
            mock_sys.path = []

            result = create_composite_guidance_video(
                structure_configs=configs,
                total_frames=10,
                structure_type="canny",
                target_resolution=(64, 64),
                target_fps=24,
                output_path=output_path,
            )

        assert result == output_path

    @patch("source.media.structure.compositing.process_structure_frames")
    @patch("source.media.structure.compositing.load_structure_video_frames_with_range")
    @patch("source.media.structure.compositing.create_neutral_frame")
    def test_no_valid_configs_raises(self, mock_neutral, mock_load, mock_process):
        """Should raise ValueError when no valid configs are provided."""
        from source.media.structure.compositing import create_composite_guidance_video

        # Config starts beyond total_frames, so it's filtered out
        configs = [{"path": "/src.mp4", "start_frame": 200, "end_frame": 300}]
        output_path = MagicMock(spec=Path)

        with pytest.raises(ValueError, match="No valid structure video configs"):
            create_composite_guidance_video(
                structure_configs=configs,
                total_frames=100,
                structure_type="canny",
                target_resolution=(64, 64),
                target_fps=24,
                output_path=output_path,
            )

    @patch("source.media.structure.compositing.process_structure_frames")
    @patch("source.media.structure.compositing.load_structure_video_frames_with_range")
    @patch("source.media.structure.compositing.create_neutral_frame")
    def test_padding_when_processed_frames_short(
        self, mock_neutral, mock_load, mock_process
    ):
        """When processed frames < needed frames, should pad with last frame."""
        from source.media.structure.compositing import create_composite_guidance_video

        neutral = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_neutral.return_value = neutral

        source_frames = [np.ones((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        mock_load.return_value = source_frames

        # Return only 3 frames when 5 are needed
        processed_frames = [np.ones((64, 64, 3), dtype=np.uint8) * i for i in range(3)]
        mock_process.return_value = processed_frames

        configs = [{"path": "/src.mp4", "start_frame": 0, "end_frame": 5}]
        output_path = MagicMock(spec=Path)
        output_path.parent = MagicMock()
        output_path.exists.return_value = True
        output_path.stat.return_value = MagicMock(st_size=2048)
        output_path.name = "composite.mp4"

        import sys
        mock_save_video = MagicMock()
        fake_modules = {
            "shared": MagicMock(),
            "shared.utils": MagicMock(),
            "shared.utils.audio_video": MagicMock(save_video=mock_save_video),
        }

        with patch.dict(sys.modules, fake_modules), \
             patch("source.media.structure.compositing.torch") as mock_torch, \
             patch("source.media.structure.compositing.Path") as mock_path_cls, \
             patch("source.media.structure.compositing.sys") as mock_sys:
            mock_torch.cuda.is_available.return_value = False
            mock_parent = MagicMock()
            mock_parent.parent = mock_parent
            mock_path_cls.return_value.parent = mock_parent
            mock_parent.__truediv__ = MagicMock(return_value=MagicMock(__str__=lambda s: "/fake/Wan2GP"))
            mock_sys.path = []

            result = create_composite_guidance_video(
                structure_configs=configs,
                total_frames=10,
                structure_type="canny",
                target_resolution=(64, 64),
                target_fps=24,
                output_path=output_path,
            )

        # save_video should have been called with 10 frames
        assert mock_save_video.called
        call_args = mock_save_video.call_args
        video_tensor = call_args[0][0]
        assert video_tensor.shape[0] == 10  # total_frames


class TestValidateEdgeCases:
    """Additional edge case tests for validation."""

    def test_adjacent_configs_no_overlap(self):
        """Configs touching at boundary should not overlap."""
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 50},
            {"path": "/b.mp4", "start_frame": 50, "end_frame": 100},
        ]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 2

    def test_single_frame_config(self):
        """Config spanning a single frame should be valid."""
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [{"path": "/a.mp4", "start_frame": 0, "end_frame": 1}]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 1

    def test_mixed_valid_and_out_of_range(self):
        """Should keep valid configs and skip out-of-range ones."""
        from source.media.structure.compositing import validate_structure_video_configs

        configs = [
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 50},
            {"path": "/b.mp4", "start_frame": 150, "end_frame": 200},  # beyond total
        ]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 1
        assert result[0]["path"] == "/a.mp4"
