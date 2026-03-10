"""Tests for source/utils/output_path_utils.py."""

from pathlib import Path

import pytest

from source.utils.output_path_utils import (
    prepare_output_path,
    sanitize_filename_for_storage,
)


class TestPrepareOutputPath:
    def test_creates_directory(self, tmp_path):
        output_dir = tmp_path / "outputs"
        path, db_loc = prepare_output_path("task-1", "video.mp4", output_dir)
        assert output_dir.exists()
        assert path.parent == output_dir

    def test_returns_path_and_string(self, tmp_path):
        path, db_loc = prepare_output_path("task-1", "video.mp4", tmp_path)
        assert isinstance(path, Path)
        assert isinstance(db_loc, str)

    def test_prefixes_filename_with_task_id(self, tmp_path):
        path, _ = prepare_output_path("task-1", "video.mp4", tmp_path)
        assert path.name == "task-1_video.mp4"

    def test_no_double_prefix(self, tmp_path):
        """Filename already starting with task_id is not re-prefixed."""
        path, _ = prepare_output_path("task-1", "task-1_video.mp4", tmp_path)
        assert path.name == "task-1_video.mp4"

    def test_custom_output_dir(self, tmp_path):
        custom = tmp_path / "custom"
        path, _ = prepare_output_path("task-1", "video.mp4", tmp_path, custom_output_dir=custom)
        assert path.parent == custom
        # custom_output_dir skips task_id prefix
        assert path.name == "video.mp4"

    def test_task_type_subdirectory(self, tmp_path):
        path, _ = prepare_output_path("task-1", "video.mp4", tmp_path, task_type="vace")
        assert "vace" in str(path.parent)

    def test_collision_resolution(self, tmp_path):
        # Create existing file
        (tmp_path / "task-1_video.mp4").write_bytes(b"existing")
        path, _ = prepare_output_path("task-1", "video.mp4", tmp_path)
        assert path.name == "task-1_video_1.mp4"


class TestSanitizeFilenameForStorage:
    def test_normal_filename_unchanged(self):
        assert sanitize_filename_for_storage("video.mp4") == "video.mp4"

    def test_removes_unsafe_characters(self):
        result = sanitize_filename_for_storage("video§®.mp4")
        assert "§" not in result
        assert "®" not in result

    def test_removes_control_characters(self):
        result = sanitize_filename_for_storage("video\x00\x01.mp4")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_collapses_multiple_spaces(self):
        result = sanitize_filename_for_storage("video   test.mp4")
        assert "   " not in result

    def test_strips_leading_trailing(self):
        result = sanitize_filename_for_storage("  video.mp4  ")
        assert result == "video.mp4"

    def test_empty_becomes_sanitized_file(self):
        assert sanitize_filename_for_storage("§®©") == "sanitized_file"
