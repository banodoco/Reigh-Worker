"""Tests for source/utils/prompt_utils.py."""

import uuid

from source.utils.prompt_utils import (
    ensure_valid_prompt,
    ensure_valid_negative_prompt,
    generate_unique_task_id,
)


class TestEnsureValidPrompt:
    def test_normal_text(self):
        assert ensure_valid_prompt("a sunset") == "a sunset"

    def test_none_returns_space(self):
        assert ensure_valid_prompt(None) == " "

    def test_empty_returns_space(self):
        assert ensure_valid_prompt("") == " "

    def test_whitespace_only_returns_space(self):
        assert ensure_valid_prompt("   ") == " "

    def test_strips_surrounding_whitespace(self):
        assert ensure_valid_prompt("  hello  ") == "hello"


class TestEnsureValidNegativePrompt:
    def test_normal_text(self):
        assert ensure_valid_negative_prompt("blurry") == "blurry"

    def test_none_returns_space(self):
        assert ensure_valid_negative_prompt(None) == " "

    def test_empty_returns_space(self):
        assert ensure_valid_negative_prompt("") == " "


class TestGenerateUniqueTaskId:
    def test_valid_uuid_format(self):
        task_id = generate_unique_task_id()
        # Should be parseable as UUID4
        parsed = uuid.UUID(task_id, version=4)
        assert str(parsed) == task_id

    def test_uniqueness(self):
        ids = {generate_unique_task_id() for _ in range(100)}
        assert len(ids) == 100

    def test_prefix_ignored(self):
        task_id = generate_unique_task_id(prefix="my_prefix_")
        # Prefix is ignored — result is still a bare UUID
        assert "my_prefix_" not in task_id
        uuid.UUID(task_id, version=4)
