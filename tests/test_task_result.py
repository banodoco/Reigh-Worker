"""Tests for source.core.params.task_result — TaskResult and TaskOutcome."""

import pytest
from dataclasses import FrozenInstanceError

from source.core.params.task_result import TaskOutcome, TaskResult


# ── TaskOutcome enum values ──────────────────────────────────────────────

class TestTaskOutcome:
    def test_enum_members(self):
        assert TaskOutcome.SUCCESS.value == "success"
        assert TaskOutcome.FAILED.value == "failed"
        assert TaskOutcome.ORCHESTRATING.value == "orchestrating"
        assert TaskOutcome.ORCHESTRATOR_COMPLETE.value == "orchestrator_complete"

    def test_enum_count(self):
        assert len(TaskOutcome) == 4


# ── Factory: success ─────────────────────────────────────────────────────

class TestSuccess:
    def test_basic(self):
        r = TaskResult.success(output_path="/out/video.mp4")
        assert r.outcome == TaskOutcome.SUCCESS
        assert r.output_path == "/out/video.mp4"
        assert r.error_message is None
        assert r.thumbnail_url is None
        assert r.metadata == {}

    def test_with_metadata(self):
        r = TaskResult.success(output_path="/out/v.mp4", fps=30, codec="h264")
        assert r.metadata == {"fps": 30, "codec": "h264"}

    def test_is_success(self):
        assert TaskResult.success("/p").is_success is True

    def test_is_terminal(self):
        assert TaskResult.success("/p").is_terminal is True

    def test_tuple_unpacking(self):
        ok, path = TaskResult.success("/out/v.mp4")
        assert ok is True
        assert path == "/out/v.mp4"


# ── Factory: failed ──────────────────────────────────────────────────────

class TestFailed:
    def test_basic(self):
        r = TaskResult.failed("Model not found")
        assert r.outcome == TaskOutcome.FAILED
        assert r.error_message == "Model not found"
        assert r.output_path is None
        assert r.thumbnail_url is None
        assert r.metadata == {}

    def test_is_success(self):
        assert TaskResult.failed("err").is_success is False

    def test_is_terminal(self):
        assert TaskResult.failed("err").is_terminal is True

    def test_tuple_unpacking(self):
        ok, msg = TaskResult.failed("boom")
        assert ok is False
        assert msg == "boom"


# ── Factory: orchestrator_complete ───────────────────────────────────────

class TestOrchestratorComplete:
    def test_basic(self):
        r = TaskResult.orchestrator_complete(output_path="/final.mp4")
        assert r.outcome == TaskOutcome.ORCHESTRATOR_COMPLETE
        assert r.output_path == "/final.mp4"
        assert r.thumbnail_url is None

    def test_with_thumbnail(self):
        r = TaskResult.orchestrator_complete("/f.mp4", thumbnail_url="https://thumb.png")
        assert r.thumbnail_url == "https://thumb.png"

    def test_is_success(self):
        assert TaskResult.orchestrator_complete("/p").is_success is True

    def test_is_terminal(self):
        assert TaskResult.orchestrator_complete("/p").is_terminal is True

    def test_tuple_unpacking(self):
        ok, path = TaskResult.orchestrator_complete("/final.mp4")
        assert ok is True
        assert path == "/final.mp4"


# ── Factory: orchestrating ───────────────────────────────────────────────

class TestOrchestrating:
    def test_basic(self):
        r = TaskResult.orchestrating("3/5 segments complete")
        assert r.outcome == TaskOutcome.ORCHESTRATING
        # message is stored in output_path
        assert r.output_path == "3/5 segments complete"
        assert r.error_message is None

    def test_is_success_false(self):
        # ORCHESTRATING is not in the is_success set
        assert TaskResult.orchestrating("msg").is_success is False

    def test_is_terminal_false(self):
        # Still in progress — not terminal
        assert TaskResult.orchestrating("msg").is_terminal is False

    def test_tuple_unpacking(self):
        ok, path = TaskResult.orchestrating("2/5 done")
        assert ok is True
        assert path == "2/5 done"


# ── Frozen immutability ─────────────────────────────────────────────────

class TestFrozen:
    def test_cannot_set_outcome(self):
        r = TaskResult.success("/p")
        with pytest.raises(FrozenInstanceError):
            r.outcome = TaskOutcome.FAILED

    def test_cannot_set_output_path(self):
        r = TaskResult.success("/p")
        with pytest.raises(FrozenInstanceError):
            r.output_path = "/other"

    def test_cannot_set_error_message(self):
        r = TaskResult.failed("err")
        with pytest.raises(FrozenInstanceError):
            r.error_message = "new"

    def test_cannot_set_thumbnail_url(self):
        r = TaskResult.orchestrator_complete("/p", thumbnail_url="url")
        with pytest.raises(FrozenInstanceError):
            r.thumbnail_url = "other"


# ── Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_multiple_unpacks_are_idempotent(self):
        r = TaskResult.success("/p")
        a1, b1 = r
        a2, b2 = r
        assert (a1, b1) == (a2, b2)

    def test_iter_yields_exactly_two(self):
        r = TaskResult.failed("err")
        items = list(r)
        assert len(items) == 2

    def test_metadata_default_is_not_shared(self):
        r1 = TaskResult.success("/a")
        r2 = TaskResult.success("/b")
        assert r1.metadata is not r2.metadata
