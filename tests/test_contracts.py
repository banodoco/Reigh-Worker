"""Tests for source.core.params.contracts — validate_orchestrator_details."""

import pytest

from source.core.params.contracts import (
    _REQUIRED_ORCHESTRATOR_KEYS,
    validate_orchestrator_details,
)


def _make_valid_details(**overrides) -> dict:
    """Return a minimal dict satisfying all required orchestrator keys."""
    base = {
        "model_name": "wan2.1-vace",
        "parsed_resolution_wh": "1280x720",
        "segment_frames_expanded": [81, 81],
        "num_new_segments_to_generate": 2,
        "base_prompts_expanded": ["prompt A", "prompt B"],
        "negative_prompts_expanded": ["", ""],
        "frame_overlap_expanded": [8, 8],
        "input_image_paths_resolved": ["/img/a.png", "/img/b.png"],
    }
    base.update(overrides)
    return base


# ── Sanity check on the required-keys constant ──────────────────────────

class TestRequiredKeys:
    def test_is_frozenset(self):
        assert isinstance(_REQUIRED_ORCHESTRATOR_KEYS, frozenset)

    def test_expected_keys(self):
        expected = {
            "model_name",
            "parsed_resolution_wh",
            "segment_frames_expanded",
            "num_new_segments_to_generate",
            "base_prompts_expanded",
            "negative_prompts_expanded",
            "frame_overlap_expanded",
            "input_image_paths_resolved",
        }
        assert _REQUIRED_ORCHESTRATOR_KEYS == expected


# ── Valid dicts pass ─────────────────────────────────────────────────────

class TestValidDetails:
    def test_minimal_valid(self):
        # Should not raise
        validate_orchestrator_details(_make_valid_details())

    def test_extra_keys_allowed(self):
        details = _make_valid_details(
            run_id="abc-123",
            debug_mode_enabled=True,
            some_future_key="whatever",
        )
        validate_orchestrator_details(details)


# ── Missing keys raise ValueError ────────────────────────────────────────

class TestMissingKeys:
    def test_empty_dict(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_orchestrator_details({})

    def test_single_missing_key(self):
        details = _make_valid_details()
        del details["model_name"]
        with pytest.raises(ValueError, match="model_name"):
            validate_orchestrator_details(details)

    def test_multiple_missing_keys(self):
        details = _make_valid_details()
        del details["model_name"]
        del details["parsed_resolution_wh"]
        with pytest.raises(ValueError) as exc_info:
            validate_orchestrator_details(details)
        msg = str(exc_info.value)
        assert "model_name" in msg
        assert "parsed_resolution_wh" in msg

    def test_all_keys_missing(self):
        with pytest.raises(ValueError) as exc_info:
            validate_orchestrator_details({})
        msg = str(exc_info.value)
        # Every required key should appear in the error
        for key in _REQUIRED_ORCHESTRATOR_KEYS:
            assert key in msg


# ── Context and task_id appear in error messages ─────────────────────────

class TestErrorContext:
    def test_default_context(self):
        with pytest.raises(ValueError, match="orchestrator"):
            validate_orchestrator_details({})

    def test_default_task_id(self):
        with pytest.raises(ValueError, match="unknown"):
            validate_orchestrator_details({})

    def test_custom_context(self):
        with pytest.raises(ValueError, match="travel_orch"):
            validate_orchestrator_details({}, context="travel_orch")

    def test_custom_task_id(self):
        with pytest.raises(ValueError, match="task-42"):
            validate_orchestrator_details({}, task_id="task-42")

    def test_both_custom(self):
        with pytest.raises(ValueError, match=r"my_ctx.*task-99"):
            validate_orchestrator_details(
                {}, context="my_ctx", task_id="task-99"
            )


# ── Edge: only optional keys present (all required missing) ─────────────

class TestOnlyOptionalKeys:
    def test_only_optional_keys(self):
        details = {"run_id": "r1", "debug_mode_enabled": False}
        with pytest.raises(ValueError, match="missing required keys"):
            validate_orchestrator_details(details)
