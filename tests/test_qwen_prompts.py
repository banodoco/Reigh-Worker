"""
Tests for source/models/model_handlers/qwen_prompts.py.

Covers:
- All string constants are non-empty
- select_style_system_prompt for every bool combination
- apply_system_prompt with/without custom prompt
- build_style_prompt with various strength combinations
"""

import pytest

from source.models.model_handlers.qwen_prompts import (
    SYSTEM_PROMPT_ANNOTATED_EDIT,
    SYSTEM_PROMPT_IMAGE_2512,
    SYSTEM_PROMPT_IMAGE_EDIT,
    SYSTEM_PROMPT_IMAGE_GEN,
    SYSTEM_PROMPT_IMAGE_HIRES,
    SYSTEM_PROMPT_IMG2IMG,
    SYSTEM_PROMPT_INPAINT,
    SYSTEM_PROMPT_TURBO,
    apply_system_prompt,
    build_style_prompt,
    select_style_system_prompt,
)


# ── String constants ────────────────────────────────────────────────────


class TestConstants:
    """All exported prompt constants must be non-empty strings."""

    @pytest.mark.parametrize(
        "constant",
        [
            SYSTEM_PROMPT_IMAGE_EDIT,
            SYSTEM_PROMPT_INPAINT,
            SYSTEM_PROMPT_ANNOTATED_EDIT,
            SYSTEM_PROMPT_IMAGE_GEN,
            SYSTEM_PROMPT_IMAGE_HIRES,
            SYSTEM_PROMPT_IMAGE_2512,
            SYSTEM_PROMPT_TURBO,
            SYSTEM_PROMPT_IMG2IMG,
        ],
    )
    def test_constants_are_nonempty_strings(self, constant):
        assert isinstance(constant, str)
        assert len(constant) > 0


# ── select_style_system_prompt ──────────────────────────────────────────


class TestSelectStyleSystemPrompt:
    """Exhaustive boolean-combination tests for select_style_system_prompt."""

    def test_all_three_true(self):
        result = select_style_system_prompt(has_subject=True, has_style=True, has_scene=True)
        assert "subjects" in result
        assert "styles" in result
        assert "scenes" in result

    def test_subject_and_style_only(self):
        result = select_style_system_prompt(has_subject=True, has_style=True, has_scene=False)
        assert "subjects" in result
        assert "styles" in result
        assert "scenes" not in result

    def test_style_only(self):
        result = select_style_system_prompt(has_subject=False, has_style=True, has_scene=False)
        assert "artistic styles" in result

    def test_style_and_scene_no_subject(self):
        # has_style=True but has_subject=False → falls into the has_style branch
        result = select_style_system_prompt(has_subject=False, has_style=True, has_scene=True)
        assert "artistic styles" in result

    def test_subject_only_no_style(self):
        # has_subject=True but has_style=False → falls through to IMG2IMG default
        result = select_style_system_prompt(has_subject=True, has_style=False, has_scene=False)
        assert result == SYSTEM_PROMPT_IMG2IMG

    def test_scene_only(self):
        result = select_style_system_prompt(has_subject=False, has_style=False, has_scene=True)
        assert result == SYSTEM_PROMPT_IMG2IMG

    def test_none_true(self):
        result = select_style_system_prompt(has_subject=False, has_style=False, has_scene=False)
        assert result == SYSTEM_PROMPT_IMG2IMG

    def test_subject_and_scene_no_style(self):
        result = select_style_system_prompt(has_subject=True, has_style=False, has_scene=True)
        assert result == SYSTEM_PROMPT_IMG2IMG


# ── apply_system_prompt ─────────────────────────────────────────────────


class TestApplySystemPrompt:
    """Tests for apply_system_prompt helper."""

    def test_uses_default_when_no_custom(self):
        db_params: dict = {}
        gen_params: dict = {}
        apply_system_prompt(db_params, gen_params, "my default")
        assert gen_params["system_prompt"] == "my default"

    def test_uses_default_when_custom_is_none(self):
        db_params = {"system_prompt": None}
        gen_params: dict = {}
        apply_system_prompt(db_params, gen_params, "fallback")
        assert gen_params["system_prompt"] == "fallback"

    def test_uses_default_when_custom_is_empty_string(self):
        db_params = {"system_prompt": ""}
        gen_params: dict = {}
        apply_system_prompt(db_params, gen_params, "fallback")
        assert gen_params["system_prompt"] == "fallback"

    def test_uses_custom_when_provided(self):
        db_params = {"system_prompt": "custom instruction"}
        gen_params: dict = {}
        apply_system_prompt(db_params, gen_params, "default")
        assert gen_params["system_prompt"] == "custom instruction"

    def test_overwrites_existing_value(self):
        db_params = {"system_prompt": "override"}
        gen_params = {"system_prompt": "old value"}
        apply_system_prompt(db_params, gen_params, "default")
        assert gen_params["system_prompt"] == "override"

    def test_modifies_dict_in_place(self):
        db_params: dict = {}
        gen_params: dict = {}
        apply_system_prompt(db_params, gen_params, "prompt")
        # Ensure the original dict was modified, not a copy
        assert "system_prompt" in gen_params


# ── build_style_prompt ──────────────────────────────────────────────────


class TestBuildStylePrompt:
    """Tests for build_style_prompt with various strength combinations."""

    def test_no_strengths_returns_original(self):
        result = build_style_prompt("paint a cat", 0.0, 0.0, "cat", False)
        assert result == "paint a cat"

    def test_style_only(self):
        result = build_style_prompt("a sunset", 0.8, 0.0, "", False)
        assert result.startswith("In the style of this image,")
        assert result.endswith("a sunset")

    def test_subject_only_no_scene(self):
        result = build_style_prompt("running", 0.0, 0.7, "dog", False)
        assert result.startswith("Make an image of this dog:")
        assert "running" in result

    def test_subject_only_with_scene(self):
        result = build_style_prompt("park", 0.0, 0.7, "dog", True)
        assert "Make an image of this dog in this scene:" in result
        assert "park" in result

    def test_style_and_subject_no_scene(self):
        result = build_style_prompt("dancing", 0.5, 0.5, "person", False)
        assert result.startswith("In the style of this image,")
        # lowercase 'make' when preceded by style prefix
        assert "make an image of this person:" in result
        assert result.endswith("dancing")

    def test_style_and_subject_with_scene(self):
        result = build_style_prompt("beach", 0.5, 0.5, "person", True)
        assert "In the style of this image," in result
        assert "make an image of this person in this scene:" in result
        assert result.endswith("beach")

    def test_subject_without_description_ignored(self):
        # subject_strength > 0 but empty description => no subject part
        result = build_style_prompt("hello", 0.0, 0.9, "", False)
        assert result == "hello"

    def test_style_and_subject_without_description(self):
        # style active, subject_strength > 0 but no description
        result = build_style_prompt("hello", 0.5, 0.9, "", False)
        assert result == "In the style of this image, hello"

    def test_prompt_preserved_exactly(self):
        original = "  leading spaces and CAPS!  "
        result = build_style_prompt(original, 0.0, 0.0, "x", False)
        assert result == original
