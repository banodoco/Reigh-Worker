"""Tests for source/task_handlers/tasks/task_registry.py.

Focus areas:
- _get_param helper (pure function, no external deps)
- TaskRegistry.dispatch routing logic
- _handle_direct_queue_task behavior
- SegmentContext / GenerationInputs / ImageRefs / StructureOutputs dataclasses
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields

import pytest


# ---------------------------------------------------------------------------
# Heavy-import mocking: patch sys.modules BEFORE importing the module under test.
# This avoids pulling in torch, PIL, cv2, supabase, etc.
# ---------------------------------------------------------------------------

def _make_mock_module(name="mock_module"):
    """Create a MagicMock that behaves like a module (has __path__, __name__)."""
    m = MagicMock()
    m.__name__ = name
    m.__path__ = []
    m.__file__ = f"<mocked {name}>"
    return m


# Modules that need mocking for import to succeed
_MOCK_MODULES = {
    "torch": _make_mock_module("torch"),
    "torch.nn": _make_mock_module("torch.nn"),
    "torch.nn.functional": _make_mock_module("torch.nn.functional"),
    "cv2": _make_mock_module("cv2"),
    "PIL": _make_mock_module("PIL"),
    "PIL.Image": _make_mock_module("PIL.Image"),
    "safetensors": _make_mock_module("safetensors"),
    "safetensors.torch": _make_mock_module("safetensors.torch"),
    "numpy": _make_mock_module("numpy"),
}


# ---------------------------------------------------------------------------
# We import the module once with mocked heavy dependencies. Since the module
# also imports many internal project modules that may themselves import heavy
# deps, we mock at the source.* level where needed.
# ---------------------------------------------------------------------------

# Mock the headless_model_management module to provide HeadlessTaskQueue and GenerationTask
_mock_hmm = _make_mock_module("headless_model_management")
_mock_hmm.HeadlessTaskQueue = MagicMock
_mock_hmm.GenerationTask = MagicMock


# We need to be careful — the task_registry module imports many things.
# Rather than mocking individual internal modules, we just import and let
# it fail-fast, fixing any import issues.

@pytest.fixture(autouse=True)
def _patch_time_sleep():
    """Prevent any accidental real sleeps during tests."""
    with patch("time.sleep"):
        yield


# ---------------------------------------------------------------------------
# _get_param tests
# ---------------------------------------------------------------------------

class TestGetParam:
    """Tests for the _get_param helper function."""

    def test_single_source_found(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("key", {"key": "value"}) == "value"

    def test_single_source_not_found_returns_none(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("missing", {"other": "value"}) is None

    def test_single_source_not_found_with_default(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("missing", {"other": "value"}, default="fallback") == "fallback"

    def test_first_source_wins(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": "first"}
        src2 = {"key": "second"}
        assert _get_param("key", src1, src2) == "first"

    def test_skips_none_values(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": None}
        src2 = {"key": "second"}
        assert _get_param("key", src1, src2) == "second"

    def test_skips_none_source(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("key", None, {"key": "value"}) == "value"

    def test_skips_empty_dict_source(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("key", {}, {"key": "value"}) == "value"

    def test_prefer_truthy_skips_empty_string(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": ""}
        src2 = {"key": "real_value"}
        assert _get_param("key", src1, src2, prefer_truthy=True) == "real_value"

    def test_prefer_truthy_skips_empty_dict(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": {}}
        src2 = {"key": {"a": 1}}
        assert _get_param("key", src1, src2, prefer_truthy=True) == {"a": 1}

    def test_prefer_truthy_skips_empty_list(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": []}
        src2 = {"key": [1, 2]}
        assert _get_param("key", src1, src2, prefer_truthy=True) == [1, 2]

    def test_prefer_truthy_skips_zero(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": 0}
        src2 = {"key": 42}
        assert _get_param("key", src1, src2, prefer_truthy=True) == 42

    def test_prefer_truthy_preserves_bool_false(self):
        """Explicit False should NOT be skipped even with prefer_truthy."""
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": False}
        src2 = {"key": True}
        assert _get_param("key", src1, src2, prefer_truthy=True) is False

    def test_prefer_truthy_preserves_bool_true(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": True}
        assert _get_param("key", src1, prefer_truthy=True) is True

    def test_no_prefer_truthy_keeps_empty_string(self):
        """Without prefer_truthy, empty string is returned (only None is skipped)."""
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": ""}
        src2 = {"key": "other"}
        assert _get_param("key", src1, src2) == ""

    def test_no_prefer_truthy_keeps_zero(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": 0}
        src2 = {"key": 99}
        assert _get_param("key", src1, src2) == 0

    def test_all_sources_none_returns_none(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": None}
        src2 = {"key": None}
        assert _get_param("key", src1, src2) is None

    def test_all_sources_none_with_default(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": None}
        assert _get_param("key", src1, default="fallback") == "fallback"

    def test_key_missing_from_all_sources(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("key", {"a": 1}, {"b": 2}) is None

    def test_no_sources(self):
        from source.task_handlers.tasks.task_registry import _get_param
        assert _get_param("key") is None

    def test_three_sources_cascading(self):
        from source.task_handlers.tasks.task_registry import _get_param
        src1 = {"key": None}
        src2 = {}
        src3 = {"key": "found_in_third"}
        assert _get_param("key", src1, src2, src3) == "found_in_third"


# ---------------------------------------------------------------------------
# Dataclass structure tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Verify the dataclass structures are correct."""

    def test_segment_context_fields(self):
        from source.task_handlers.tasks.task_registry import SegmentContext
        ctx = SegmentContext(
            mode="orchestrator",
            orchestrator_details={"model_name": "vace"},
            individual_params={},
            segment_idx=0,
            segment_params={"some": "param"},
        )
        assert ctx.mode == "orchestrator"
        assert ctx.segment_idx == 0
        assert ctx.orchestrator_task_id_ref is None
        assert ctx.orchestrator_run_id is None

    def test_generation_inputs_fields(self):
        from source.task_handlers.tasks.task_registry import GenerationInputs
        gen = GenerationInputs(
            model_name="vace",
            prompt_for_wgp="test prompt",
            negative_prompt_for_wgp="",
            parsed_res_wh=(896, 496),
            total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp/out"),
            segment_processing_dir=Path("/tmp/proc"),
            debug_enabled=False,
            travel_mode="vace",
        )
        assert gen.model_name == "vace"
        assert gen.parsed_res_wh == (896, 496)
        assert gen.total_frames_for_segment == 81

    def test_image_refs_defaults(self):
        from source.task_handlers.tasks.task_registry import ImageRefs
        refs = ImageRefs()
        assert refs.start_ref_path is None
        assert refs.end_ref_path is None
        assert refs.svi_predecessor_video_for_source is None
        assert refs.use_svi is False
        assert refs.is_continuing is False

    def test_structure_outputs_defaults(self):
        from source.task_handlers.tasks.task_registry import StructureOutputs
        out = StructureOutputs()
        assert out.guide_video_path is None
        assert out.mask_video_path_for_wgp is None
        assert out.video_prompt_type_str is None
        assert out.structure_config is None


# ---------------------------------------------------------------------------
# TaskRegistry.dispatch routing tests
# ---------------------------------------------------------------------------

class TestTaskRegistryDispatch:
    """Test that dispatch routes task types to the correct handlers."""

    def _make_context(self, task_type="vace", task_params=None, task_queue=None, **overrides):
        """Build a minimal TaskDispatchContext dict."""
        ctx = {
            "task_id": "test-task-001",
            "task_params_dict": task_params or {"prompt": "test", "resolution": "896x496", "video_length": 81},
            "main_output_dir_base": Path("/tmp/output"),
            "project_id": "proj-1",
            "task_queue": task_queue,
            "colour_match_videos": False,
            "mask_active_frames": False,
            "debug_mode": False,
            "wan2gp_path": "/opt/wan2gp",
        }
        ctx.update(overrides)
        return ctx

    def test_direct_queue_task_routes_to_queue_handler(self):
        """Direct queue tasks (vace, t2v, etc.) should use _handle_direct_queue_task."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("vace", task_queue=mock_queue)

        with patch.object(TaskRegistry, "_handle_direct_queue_task", return_value=(True, "/output/video.mp4")) as mock_handler:
            result = TaskRegistry.dispatch("vace", ctx)
        mock_handler.assert_called_once_with("vace", ctx)
        assert result == (True, "/output/video.mp4")

    def test_travel_orchestrator_routes_correctly(self):
        """travel_orchestrator should route to travel_orchestrator.handle_travel_orchestrator_task."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        params = {"prompt": "test", "orchestrator_details": {}}
        ctx = self._make_context("travel_orchestrator", task_params=params, task_queue=None)

        with patch("source.task_handlers.tasks.task_registry.travel_orchestrator.handle_travel_orchestrator_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("travel_orchestrator", ctx)
        mock_handler.assert_called_once()
        assert result == (True, "/out")

    def test_travel_orchestrator_sets_task_id(self):
        """travel_orchestrator should set task_id in params and orchestrator_details."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        orch_details = {"model_name": "vace"}
        params = {"prompt": "test", "orchestrator_details": orch_details}
        ctx = self._make_context("travel_orchestrator", task_params=params, task_queue=None)

        with patch("source.task_handlers.tasks.task_registry.travel_orchestrator.handle_travel_orchestrator_task", return_value=(True, "/out")):
            TaskRegistry.dispatch("travel_orchestrator", ctx)
        # dispatch sets task_id on params and orchestrator_details
        assert params["task_id"] == "test-task-001"
        assert orch_details["orchestrator_task_id"] == "test-task-001"

    def test_travel_segment_routes_to_handle_travel_segment_via_queue(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("travel_segment", task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.handle_travel_segment_via_queue", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("travel_segment", ctx)
        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        assert call_kwargs.kwargs.get("is_standalone") is False or call_kwargs[1].get("is_standalone") is False

    def test_individual_travel_segment_is_standalone(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("individual_travel_segment", task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.handle_travel_segment_via_queue", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("individual_travel_segment", ctx)
        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        assert call_kwargs.kwargs.get("is_standalone") is True or call_kwargs[1].get("is_standalone") is True

    def test_magic_edit_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        ctx = self._make_context("magic_edit", task_queue=None)

        with patch("source.task_handlers.tasks.task_registry.me.handle_magic_edit_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("magic_edit", ctx)
        mock_handler.assert_called_once()

    def test_extract_frame_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        ctx = self._make_context("extract_frame", task_queue=None)

        with patch("source.task_handlers.tasks.task_registry.handle_extract_frame_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("extract_frame", ctx)
        mock_handler.assert_called_once()

    def test_join_clips_orchestrator_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        params = {"prompt": "test", "orchestrator_details": {}}
        ctx = self._make_context("join_clips_orchestrator", task_params=params, task_queue=None)

        with patch("source.task_handlers.tasks.task_registry._handle_join_clips_orchestrator_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("join_clips_orchestrator", ctx)
        mock_handler.assert_called_once()

    def test_edit_video_orchestrator_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        params = {"prompt": "test", "orchestrator_details": {}}
        ctx = self._make_context("edit_video_orchestrator", task_params=params, task_queue=None)

        with patch("source.task_handlers.tasks.task_registry._handle_edit_video_orchestrator_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("edit_video_orchestrator", ctx)
        mock_handler.assert_called_once()

    def test_inpaint_frames_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("inpaint_frames", task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry._handle_inpaint_frames_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("inpaint_frames", ctx)
        mock_handler.assert_called_once()

    def test_rife_interpolate_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("rife_interpolate_images", task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.handle_rife_interpolate_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("rife_interpolate_images", ctx)
        mock_handler.assert_called_once()

    def test_comfy_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        ctx = self._make_context("comfy", task_queue=None)

        with patch("source.task_handlers.tasks.task_registry.handle_comfy_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("comfy", ctx)
        mock_handler.assert_called_once()

    def test_create_visualization_routes_correctly(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        ctx = self._make_context("create_visualization", task_queue=None)

        with patch("source.task_handlers.tasks.task_registry._handle_create_visualization_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("create_visualization", ctx)
        mock_handler.assert_called_once()

    def test_unknown_type_with_queue_falls_through_to_direct_queue(self):
        """Unknown task types with a queue should fall through to _handle_direct_queue_task."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("totally_unknown_type", task_queue=mock_queue)

        with patch.object(TaskRegistry, "_handle_direct_queue_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("totally_unknown_type", ctx)
        mock_handler.assert_called_once_with("totally_unknown_type", ctx)

    def test_unknown_type_without_queue_raises(self):
        """Unknown task types without a queue should raise ValueError."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        ctx = self._make_context("totally_unknown_type", task_queue=None)

        with pytest.raises(ValueError, match="Unknown task type"):
            TaskRegistry.dispatch("totally_unknown_type", ctx)

    def test_wan_2_2_t2i_routes_to_direct_queue(self):
        """wan_2_2_t2i is a DIRECT_QUEUE_TASK_TYPE."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context("wan_2_2_t2i", task_queue=mock_queue)

        with patch.object(TaskRegistry, "_handle_direct_queue_task", return_value=(True, "/out")) as mock_handler:
            result = TaskRegistry.dispatch("wan_2_2_t2i", ctx)
        mock_handler.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_direct_queue_task tests
# ---------------------------------------------------------------------------

class TestHandleDirectQueueTask:
    """Tests for TaskRegistry._handle_direct_queue_task."""

    def _make_context(self, task_type="vace", task_params=None, **overrides):
        mock_queue = MagicMock()
        ctx = {
            "task_id": "test-task-002",
            "task_params_dict": task_params or {"prompt": "test", "resolution": "896x496", "video_length": 81},
            "main_output_dir_base": Path("/tmp/output"),
            "project_id": "proj-1",
            "task_queue": mock_queue,
            "colour_match_videos": False,
            "mask_active_frames": False,
            "debug_mode": False,
            "wan2gp_path": "/opt/wan2gp",
        }
        ctx.update(overrides)
        return ctx

    def test_submits_task_to_queue(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        # Make task complete immediately
        mock_status = MagicMock()
        mock_status.status = "completed"
        mock_status.result_path = "/output/result.mp4"
        mock_queue.get_task_status.return_value = mock_status

        ctx = self._make_context(task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task") as mock_convert:
            mock_task = MagicMock()
            mock_task.parameters = {}
            mock_convert.return_value = mock_task

            result = TaskRegistry._handle_direct_queue_task("vace", ctx)

        mock_queue.submit_task.assert_called_once_with(mock_task)
        assert result == (True, "/output/result.mp4")

    def test_failed_task_returns_error(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        mock_status = MagicMock()
        mock_status.status = "failed"
        mock_status.error_message = "GPU OOM"
        mock_queue.get_task_status.return_value = mock_status

        ctx = self._make_context(task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task") as mock_convert:
            mock_task = MagicMock()
            mock_task.parameters = {}
            mock_convert.return_value = mock_task

            result = TaskRegistry._handle_direct_queue_task("vace", ctx)

        assert result == (False, "GPU OOM")

    def test_none_status_returns_error(self):
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        mock_queue.get_task_status.return_value = None

        ctx = self._make_context(task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task") as mock_convert:
            mock_task = MagicMock()
            mock_task.parameters = {}
            mock_convert.return_value = mock_task

            result = TaskRegistry._handle_direct_queue_task("vace", ctx)

        assert result == (False, "Task status became None")

    def test_wan_2_2_t2i_sets_video_length_to_1(self):
        """wan_2_2_t2i tasks should have video_length forced to 1."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        mock_status = MagicMock()
        mock_status.status = "completed"
        mock_status.result_path = "/output/img.png"
        mock_queue.get_task_status.return_value = mock_status

        ctx = self._make_context("wan_2_2_t2i", task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task") as mock_convert:
            mock_task = MagicMock()
            mock_task.parameters = {"video_length": 81}
            mock_convert.return_value = mock_task

            TaskRegistry._handle_direct_queue_task("wan_2_2_t2i", ctx)

        assert mock_task.parameters["video_length"] == 1

    def test_colour_match_and_mask_flags_propagated(self):
        """colour_match_videos and mask_active_frames should be set on parameters."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        mock_status = MagicMock()
        mock_status.status = "completed"
        mock_status.result_path = "/output/result.mp4"
        mock_queue.get_task_status.return_value = mock_status

        ctx = self._make_context(
            task_queue=mock_queue,
            colour_match_videos=True,
            mask_active_frames=True,
        )

        with patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task") as mock_convert:
            mock_task = MagicMock()
            mock_task.parameters = {}
            mock_convert.return_value = mock_task

            TaskRegistry._handle_direct_queue_task("vace", ctx)

        assert mock_task.parameters["colour_match_videos"] is True
        assert mock_task.parameters["mask_active_frames"] is True

    def test_conversion_error_returns_failure(self):
        """Errors during task conversion should return (False, error_msg)."""
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        ctx = self._make_context(task_queue=mock_queue)

        with patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task", side_effect=ValueError("bad params")):
            result = TaskRegistry._handle_direct_queue_task("vace", ctx)

        assert result[0] is False
        assert "bad params" in result[1]


# ---------------------------------------------------------------------------
# Handler coverage: ensure ALL named handler keys are tested
# ---------------------------------------------------------------------------

class TestAllHandlersCovered:
    """Verify that the dispatch handlers dict covers all expected task types."""

    def test_all_named_handlers_present(self):
        """Ensure the handlers dict in dispatch() has entries for all known specialized types."""
        expected_specialized = {
            "travel_orchestrator",
            "travel_segment",
            "individual_travel_segment",
            "travel_stitch",
            "magic_edit",
            "join_clips_orchestrator",
            "edit_video_orchestrator",
            "join_clips_segment",
            "join_final_stitch",
            "inpaint_frames",
            "create_visualization",
            "extract_frame",
            "rife_interpolate_images",
            "comfy",
        }
        # We verify by checking that none of these raise ValueError when dispatched
        # (they either route to a handler or fall through to direct queue).
        from source.task_handlers.tasks.task_registry import TaskRegistry

        mock_queue = MagicMock()
        for task_type in expected_specialized:
            ctx = {
                "task_id": "test",
                "task_params_dict": {"prompt": "t", "orchestrator_details": {}},
                "main_output_dir_base": Path("/tmp"),
                "project_id": "p",
                "task_queue": mock_queue,
                "colour_match_videos": False,
                "mask_active_frames": False,
                "debug_mode": False,
                "wan2gp_path": "/opt",
            }
            # Patch all the handler functions to avoid actual execution
            with patch("source.task_handlers.tasks.task_registry.travel_orchestrator.handle_travel_orchestrator_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.handle_travel_segment_via_queue", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry._handle_travel_stitch_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.me.handle_magic_edit_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry._handle_join_clips_orchestrator_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry._handle_edit_video_orchestrator_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.handle_join_clips_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.handle_join_final_stitch", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry._handle_inpaint_frames_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry._handle_create_visualization_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.handle_extract_frame_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.handle_rife_interpolate_task", return_value=(True, "")), \
                 patch("source.task_handlers.tasks.task_registry.handle_comfy_task", return_value=(True, "")):
                # Should not raise ValueError
                result = TaskRegistry.dispatch(task_type, ctx)
                assert isinstance(result, tuple), f"dispatch({task_type!r}) did not return a tuple"


# ---------------------------------------------------------------------------
# _resolve_segment_context tests
# ---------------------------------------------------------------------------

class TestResolveSegmentContext:
    """Tests for _resolve_segment_context."""

    def test_standalone_mode_with_orchestrator_details(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "orchestrator_details": {"model_name": "vace"},
            "segment_index": 2,
        }
        ctx = _resolve_segment_context(params, is_standalone=True, task_id="task-1")
        assert ctx.mode == "standalone"
        assert ctx.segment_idx == 2
        assert ctx.orchestrator_details == {"model_name": "vace"}

    def test_standalone_defaults_segment_idx_to_0(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "orchestrator_details": {"model_name": "vace"},
        }
        ctx = _resolve_segment_context(params, is_standalone=True, task_id="task-1")
        assert ctx.segment_idx == 0

    def test_standalone_missing_orchestrator_details_raises(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {}
        with pytest.raises(ValueError, match="missing orchestrator_details"):
            _resolve_segment_context(params, is_standalone=True, task_id="task-1")

    def test_orchestrator_mode_missing_segment_index_raises(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {"orchestrator_details": {"model_name": "vace"}}
        with pytest.raises(ValueError, match="missing segment_index"):
            _resolve_segment_context(params, is_standalone=False, task_id="task-1")

    def test_orchestrator_mode_with_inline_details(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "orchestrator_details": {"model_name": "vace"},
            "segment_index": 1,
            "orchestrator_task_id_ref": "orch-1",
            "orchestrator_run_id": "run-1",
        }
        ctx = _resolve_segment_context(params, is_standalone=False, task_id="task-1")
        assert ctx.mode == "orchestrator"
        assert ctx.segment_idx == 1
        assert ctx.orchestrator_task_id_ref == "orch-1"
        assert ctx.orchestrator_run_id == "run-1"

    def test_orchestrator_mode_fetches_from_db_when_no_inline_details(self):
        """When orchestrator_details is not inline, it should fetch from parent task via DB."""
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "segment_index": 0,
            "orchestrator_task_id_ref": "orch-1",
        }

        fetched_params = {"orchestrator_details": {"model_name": "i2v"}}
        with patch("source.task_handlers.tasks.task_registry.db_ops.get_task_params", return_value=fetched_params):
            ctx = _resolve_segment_context(params, is_standalone=False, task_id="task-1")
        assert ctx.orchestrator_details == {"model_name": "i2v"}

    def test_orchestrator_mode_no_details_no_ref_raises(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {"segment_index": 0}
        with pytest.raises(ValueError, match="missing orchestrator_details and orchestrator_task_id_ref"):
            _resolve_segment_context(params, is_standalone=False, task_id="task-1")

    def test_legacy_full_orchestrator_payload_alias(self):
        """full_orchestrator_payload should be accepted as an alias for orchestrator_details."""
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "full_orchestrator_payload": {"model_name": "t2v"},
            "segment_index": 0,
        }
        ctx = _resolve_segment_context(params, is_standalone=False, task_id="task-1")
        assert ctx.orchestrator_details == {"model_name": "t2v"}

    def test_individual_params_extracted(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "orchestrator_details": {"model_name": "vace"},
            "segment_index": 0,
            "individual_segment_params": {"num_frames": 41},
        }
        ctx = _resolve_segment_context(params, is_standalone=False, task_id="task-1")
        assert ctx.individual_params == {"num_frames": 41}


# ---------------------------------------------------------------------------
# _resolve_generation_inputs tests
# ---------------------------------------------------------------------------

class TestResolveGenerationInputs:
    """Tests for _resolve_generation_inputs."""

    def _make_ctx(self, **overrides):
        from source.task_handlers.tasks.task_registry import SegmentContext
        defaults = dict(
            mode="orchestrator",
            orchestrator_details={
                "model_name": "vace_14B_cocktail_2_2",
                "parsed_resolution_wh": "896x496",
                "segment_frames_expanded": [81, 81],
                "model_type": "vace",
            },
            individual_params={},
            segment_idx=0,
            segment_params={
                "parsed_resolution_wh": "896x496",
            },
        )
        defaults.update(overrides)
        return SegmentContext(**defaults)

    def test_resolves_basic_inputs(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx()
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.model_name == "vace_14B_cocktail_2_2"
        assert gen.parsed_res_wh[0] > 0
        assert gen.parsed_res_wh[1] > 0
        assert gen.total_frames_for_segment == 81
        assert gen.travel_mode == "vace"

    def test_missing_model_name_raises(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            orchestrator_details={"parsed_resolution_wh": "896x496", "segment_frames_expanded": [81]},
            segment_params={"parsed_resolution_wh": "896x496"},
        )
        with pytest.raises(ValueError, match="model_name missing"):
            _resolve_generation_inputs(ctx, "task-1", tmp_path)

    def test_missing_resolution_raises(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            orchestrator_details={"model_name": "vace", "segment_frames_expanded": [81], "model_type": "vace"},
            segment_params={},
        )
        with pytest.raises(ValueError, match="parsed_resolution_wh missing"):
            _resolve_generation_inputs(ctx, "task-1", tmp_path)

    def test_individual_params_num_frames_takes_priority(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            individual_params={"num_frames": 41},
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.total_frames_for_segment == 41

    def test_enhanced_prompt_preferred_over_base(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            segment_params={
                "parsed_resolution_wh": "896x496",
                "base_prompt": "original",
                "enhanced_prompt": "AI enhanced version",
            },
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert "AI enhanced" in gen.prompt_for_wgp or gen.prompt_for_wgp == "AI enhanced version"

    def test_debug_enabled_from_segment_params(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            segment_params={
                "parsed_resolution_wh": "896x496",
                "debug_mode_enabled": True,
            },
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.debug_enabled is True

    def test_model_name_from_segment_params(self, tmp_path):
        """model_name from segment_params takes priority over orchestrator_details."""
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            segment_params={
                "model_name": "i2v_22",
                "parsed_resolution_wh": "896x496",
            },
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.model_name == "i2v_22"

    def test_frame_count_from_segment_frames_target(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            segment_params={
                "parsed_resolution_wh": "896x496",
                "segment_frames_target": 33,
            },
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.total_frames_for_segment == 33

    def test_frame_count_out_of_bounds_raises(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            segment_idx=5,
            orchestrator_details={
                "model_name": "vace",
                "parsed_resolution_wh": "896x496",
                "segment_frames_expanded": [81, 81],
                "model_type": "vace",
            },
            segment_params={"parsed_resolution_wh": "896x496"},
        )
        with pytest.raises(ValueError, match="no frame count found"):
            _resolve_generation_inputs(ctx, "task-1", tmp_path)

    def test_invalid_resolution_format_raises(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            segment_params={"parsed_resolution_wh": "not_a_resolution"},
        )
        with pytest.raises(ValueError):
            _resolve_generation_inputs(ctx, "task-1", tmp_path)

    def test_output_dir_created(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        output_dir = tmp_path / "new_subdir" / "deep"
        ctx = self._make_ctx(
            segment_params={
                "parsed_resolution_wh": "896x496",
                "current_run_base_output_dir": str(output_dir),
            },
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.segment_processing_dir.exists()

    def test_debug_enabled_from_orchestrator(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _resolve_generation_inputs

        ctx = self._make_ctx(
            orchestrator_details={
                "model_name": "vace",
                "parsed_resolution_wh": "896x496",
                "segment_frames_expanded": [81],
                "model_type": "vace",
                "debug_mode_enabled": True,
            },
        )
        gen = _resolve_generation_inputs(ctx, "task-1", tmp_path)
        assert gen.debug_enabled is True


# ---------------------------------------------------------------------------
# _build_generation_params tests
# ---------------------------------------------------------------------------

class TestBuildGenerationParams:
    """Tests for _build_generation_params."""

    def _make_inputs(self, **overrides):
        from source.task_handlers.tasks.task_registry import (
            SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        ctx_defaults = dict(
            mode="orchestrator",
            orchestrator_details={"model_type": "vace"},
            individual_params={},
            segment_idx=0,
            segment_params={},
            orchestrator_task_id_ref=None,
            orchestrator_run_id=None,
        )
        gen_defaults = dict(
            model_name="vace_14B",
            prompt_for_wgp="test prompt",
            negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720),
            total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp/test"),
            segment_processing_dir=Path("/tmp/test"),
            debug_enabled=False,
            travel_mode="vace",
        )
        img_defaults = dict(
            start_ref_path=None,
            end_ref_path=None,
            svi_predecessor_video_for_source=None,
            use_svi=False,
            is_continuing=False,
        )
        struct_defaults = dict(
            guide_video_path=None,
            mask_video_path_for_wgp=None,
            video_prompt_type_str="I",
            structure_config=MagicMock(is_uni3c=False),
        )

        # Separate overrides by destination
        ctx_kw = {k: overrides.pop(k) for k in list(overrides) if k in ctx_defaults}
        gen_kw = {k: overrides.pop(k) for k in list(overrides) if k in gen_defaults}
        img_kw = {k: overrides.pop(k) for k in list(overrides) if k in img_defaults}
        struct_kw = {k: overrides.pop(k) for k in list(overrides) if k in struct_defaults}

        ctx_defaults.update(ctx_kw)
        gen_defaults.update(gen_kw)
        img_defaults.update(img_kw)
        struct_defaults.update(struct_kw)

        return (
            SegmentContext(**ctx_defaults),
            GenerationInputs(**gen_defaults),
            ImageRefs(**img_defaults),
            StructureOutputs(**struct_defaults),
        )

    def test_base_params(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs()
        result = _build_generation_params(ctx, gen, img, struct, "task-1")

        assert result["model_name"] == "vace_14B"
        assert result["negative_prompt"] == " "
        assert result["resolution"] == "1280x720"
        assert result["video_length"] == 81
        assert result["video_prompt_type"] == "I"
        assert result["seed"] == 12345  # default seed

    def test_seed_from_individual_params(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs(individual_params={"seed_to_use": 999})
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["seed"] == 999

    def test_seed_from_segment_params(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs(segment_params={"seed_to_use": 777})
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["seed"] == 777

    def test_image_start_and_end(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        start_img = tmp_path / "start.png"
        end_img = tmp_path / "end.png"
        start_img.touch()
        end_img.touch()
        ctx, gen, img, struct = self._make_inputs(
            start_ref_path=str(start_img), end_ref_path=str(end_img),
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert "image_start" in result
        assert "image_end" in result

    def test_no_images_when_none(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs()
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert "image_start" not in result
        assert "image_end" not in result

    def test_guide_and_mask_attached(self, tmp_path):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        mask_path = tmp_path / "mask.mp4"
        mask_path.touch()
        ctx, gen, img, struct = self._make_inputs(
            guide_video_path="/tmp/guide.mp4",
            mask_video_path_for_wgp=mask_path,
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["video_guide"] == "/tmp/guide.mp4"
        assert "video_mask" in result

    def test_explicit_steps_guidance_flow_shift(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs(
            segment_params={
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "flow_shift": 3.0,
            },
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["num_inference_steps"] == 20
        assert result["guidance_scale"] == 7.5
        assert result["flow_shift"] == 3.0

    def test_segment_loras(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs(
            individual_params={
                "segment_loras": [
                    {"path": "lora_a.safetensors", "strength": 0.8},
                    {"path": "lora_b.safetensors", "strength": 1.0},
                ],
            },
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["activated_loras"] == ["lora_a.safetensors", "lora_b.safetensors"]
        assert result["loras_multipliers"] == "0.8 1.0"

    def test_segment_loras_skip_empty_path(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs(
            individual_params={
                "segment_loras": [
                    {"path": "", "strength": 0.8},
                    {"path": "lora_b.safetensors", "strength": 1.0},
                ],
            },
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        # Only non-empty paths included
        assert result["activated_loras"] == ["lora_b.safetensors"]
        assert result["loras_multipliers"] == "1.0"

    @patch("source.task_handlers.tasks.task_registry.parse_phase_config")
    def test_phase_config_applied(self, mock_parse_phase):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        mock_parse_phase.return_value = {
            "guidance_phases": [1, 2, 3],
            "switch_threshold": 0.5,
            "lora_names": ["phase_lora.safetensors"],
            "lora_multipliers": [0.9],
        }
        ctx, gen, img, struct = self._make_inputs(
            orchestrator_details={
                "model_type": "vace",
                "phase_config": {
                    "steps_per_phase": [3, 3, 3],
                    "preset_name": "test_preset",
                },
            },
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["num_inference_steps"] == 9
        assert result["guidance_phases"] == [1, 2, 3]
        assert result["switch_threshold"] == 0.5
        assert result["activated_loras"] == ["phase_lora.safetensors"]
        assert result["loras_multipliers"] == "0.9"

    @patch("source.task_handlers.tasks.task_registry.parse_phase_config")
    def test_phase_config_loras_skipped_when_segment_loras_present(self, mock_parse_phase):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        mock_parse_phase.return_value = {
            "lora_names": ["should_be_skipped.safetensors"],
            "lora_multipliers": [1.0],
        }
        ctx, gen, img, struct = self._make_inputs(
            individual_params={
                "segment_loras": [{"path": "winner.safetensors", "strength": 0.5}],
            },
            orchestrator_details={
                "model_type": "vace",
                "phase_config": {"steps_per_phase": [2, 2, 2]},
            },
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert result["activated_loras"] == ["winner.safetensors"]

    @patch("source.task_handlers.tasks.task_registry.parse_phase_config")
    def test_invalid_phase_config_raises(self, mock_parse_phase):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        mock_parse_phase.side_effect = ValueError("bad config")
        ctx, gen, img, struct = self._make_inputs(
            segment_params={"phase_config": {"steps_per_phase": [2, 2, 2]}},
        )
        with pytest.raises(ValueError, match="Invalid phase_config"):
            _build_generation_params(ctx, gen, img, struct, "task-1")

    @patch("source.task_handlers.tasks.task_registry.parse_phase_config")
    def test_phase_config_patch_data_passed_through(self, mock_parse_phase):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        mock_parse_phase.return_value = {
            "_patch_config": {"some_key": "some_value"},
        }
        ctx, gen, img, struct = self._make_inputs(
            segment_params={"phase_config": {"steps_per_phase": [2, 2, 2]}},
        )
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert "_parsed_phase_config" in result
        assert "_phase_config_model_name" in result
        assert result["_phase_config_model_name"] == "vace_14B"

    def test_no_phase_config_no_loras(self):
        from source.task_handlers.tasks.task_registry import _build_generation_params
        ctx, gen, img, struct = self._make_inputs()
        result = _build_generation_params(ctx, gen, img, struct, "task-1")
        assert "activated_loras" not in result
        assert "loras_multipliers" not in result


# ---------------------------------------------------------------------------
# _apply_svi_config tests
# ---------------------------------------------------------------------------

class TestApplySviConfig:
    """Tests for _apply_svi_config."""

    def _make_inputs(self, **ctx_overrides):
        from source.task_handlers.tasks.task_registry import SegmentContext, GenerationInputs
        ctx_defaults = dict(
            mode="orchestrator",
            orchestrator_details={},
            individual_params={},
            segment_idx=1,
            segment_params={},
        )
        ctx_defaults.update(ctx_overrides)
        ctx = SegmentContext(**ctx_defaults)
        gen = GenerationInputs(
            model_name="vace_14B",
            prompt_for_wgp="test",
            negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720),
            total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp/test"),
            segment_processing_dir=Path("/tmp/test"),
            debug_enabled=False,
            travel_mode="vace",
        )
        return ctx, gen

    def test_noop_when_svi_disabled(self):
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs()
        image_refs = ImageRefs(use_svi=False)
        params = {"video_length": 81}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert "svi2pro" not in params

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    def test_svi_enables_basic_flags(self, mock_merge):
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs()
        image_refs = ImageRefs(use_svi=True, start_ref_path="/tmp/start.png")
        params = {"video_length": 81}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert params["svi2pro"] is True
        assert params["video_prompt_type"] == "I"
        assert params["sliding_window_overlap"] == 4
        mock_merge.assert_called_once()

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    def test_svi_sets_image_refs_paths(self, mock_merge):
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs()
        image_refs = ImageRefs(use_svi=True, start_ref_path="/tmp/start.png")
        params = {"video_length": 81}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert "image_refs_paths" in params
        assert len(params["image_refs_paths"]) == 1

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    @patch("source.media.video.get_video_frame_count_and_fps", return_value=(81, 16))
    def test_svi_with_predecessor_video(self, mock_vfc, mock_merge):
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs()
        image_refs = ImageRefs(
            use_svi=True,
            start_ref_path="/tmp/start.png",
            svi_predecessor_video_for_source="/tmp/pred.mp4",
        )
        params = {"video_length": 81, "image_start": "/tmp/start.png"}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert params["video_source"] is not None
        assert "image_start" not in params  # removed for SVI
        assert params["image_prompt_type"] == "SV"
        assert params["video_length"] == 85  # 81 + 4 overlap

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    @patch("source.media.video.get_video_frame_count_and_fps", side_effect=OSError("no video"))
    def test_svi_predecessor_analysis_error_doesnt_crash(self, mock_vfc, mock_merge):
        """If video analysis of predecessor fails, SVI config should still be applied."""
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs()
        image_refs = ImageRefs(
            use_svi=True,
            svi_predecessor_video_for_source="/tmp/pred.mp4",
        )
        params = {"video_length": 81}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert params["svi2pro"] is True
        assert params["video_source"] is not None

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    def test_svi_does_not_double_bump_video_length(self, mock_merge):
        """If video_length was already adjusted, should not bump again."""
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs()
        image_refs = ImageRefs(
            use_svi=True,
            svi_predecessor_video_for_source="/tmp/pred.mp4",
        )
        # video_length already bumped (85 != 81 desired)
        params = {"video_length": 85}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        # Should NOT bump further
        assert params["video_length"] == 85

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    def test_svi_copies_segment_params(self, mock_merge):
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs(
            segment_params={
                "guidance_scale": 5.0,
                "num_inference_steps": 10,
                "flow_shift": 2.0,
            },
        )
        image_refs = ImageRefs(use_svi=True)
        params = {"video_length": 81}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert params["guidance_scale"] == 5.0
        assert params["num_inference_steps"] == 10
        assert params["flow_shift"] == 2.0

    @patch("source.task_handlers.travel.svi_config.merge_svi_into_generation_params")
    def test_svi_does_not_copy_none_segment_params(self, mock_merge):
        from source.task_handlers.tasks.task_registry import _apply_svi_config, ImageRefs
        ctx, gen = self._make_inputs(
            segment_params={"guidance_scale": None},
        )
        image_refs = ImageRefs(use_svi=True)
        params = {"video_length": 81}
        _apply_svi_config(params, ctx, gen, image_refs, "task-1")
        assert "guidance_scale" not in params


# ---------------------------------------------------------------------------
# _apply_uni3c_config tests
# ---------------------------------------------------------------------------

class TestApplyUni3cConfig:
    """Tests for _apply_uni3c_config."""

    def _make_inputs(self, **config_overrides):
        from source.task_handlers.tasks.task_registry import (
            SegmentContext, GenerationInputs, StructureOutputs,
        )
        ctx = SegmentContext(
            mode="orchestrator",
            orchestrator_details={},
            individual_params={},
            segment_idx=0,
            segment_params={},
        )
        gen = GenerationInputs(
            model_name="vace_14B",
            prompt_for_wgp="test",
            negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720),
            total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp/test"),
            segment_processing_dir=Path("/tmp/test"),
            debug_enabled=False,
            travel_mode="vace",
        )
        config_defaults = dict(
            is_uni3c=True,
            guidance_video_url="/tmp/guide.mp4",
            strength=0.8,
            step_window=(0.0, 0.5),
            keep_on_gpu=False,
            frame_policy="loop",
            zero_empty_frames=True,
        )
        config_defaults.update(config_overrides)
        mock_config = MagicMock(**config_defaults)
        struct = StructureOutputs(structure_config=mock_config)
        return ctx, gen, struct

    def test_noop_when_not_uni3c(self):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs(is_uni3c=False)
        params = {}
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        assert "use_uni3c" not in params

    def test_noop_when_no_guide_video(self):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs(guidance_video_url=None)
        params = {}
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        assert "use_uni3c" not in params

    def test_injects_params_for_local_guide(self):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs()
        params = {}
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        assert params["use_uni3c"] is True
        assert params["uni3c_guide_video"] == "/tmp/guide.mp4"
        assert params["uni3c_strength"] == 0.8
        assert params["uni3c_start_percent"] == 0.0
        assert params["uni3c_end_percent"] == 0.5
        assert params["uni3c_frame_policy"] == "loop"
        assert params["uni3c_zero_empty_frames"] is True
        assert params["uni3c_keep_on_gpu"] is False

    @patch("source.utils.download_file")
    def test_downloads_url_guide(self, mock_download):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs(
            guidance_video_url="https://example.com/guide.mp4",
        )
        params = {}
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        mock_download.assert_called_once()
        assert params["use_uni3c"] is True

    def test_blackout_last_frame_for_last_segment_with_end_anchor(self):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs()
        ctx.segment_params = {"is_last_segment": True}
        params = {"image_end": "/tmp/end.png"}
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        assert params["uni3c_blackout_last_frame"] is True

    def test_no_blackout_when_not_last_segment(self):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs()
        ctx.segment_params = {"is_last_segment": False}
        params = {"image_end": "/tmp/end.png"}
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        assert "uni3c_blackout_last_frame" not in params

    def test_no_blackout_when_no_end_anchor(self):
        from source.task_handlers.tasks.task_registry import _apply_uni3c_config
        ctx, gen, struct = self._make_inputs()
        ctx.segment_params = {"is_last_segment": True}
        params = {}  # no image_end
        _apply_uni3c_config(params, ctx, gen, struct, "task-1")
        assert "uni3c_blackout_last_frame" not in params


# ---------------------------------------------------------------------------
# handle_travel_segment_via_queue integration tests
# ---------------------------------------------------------------------------

class TestHandleTravelSegmentViaQueue:
    """Integration tests for handle_travel_segment_via_queue with all sub-functions mocked."""

    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry._apply_uni3c_config")
    @patch("source.task_handlers.tasks.task_registry._apply_svi_config")
    @patch("source.task_handlers.tasks.task_registry._build_generation_params", return_value={"video_length": 81})
    @patch("source.task_handlers.tasks.task_registry._process_structure_guidance")
    @patch("source.task_handlers.tasks.task_registry._resolve_image_references")
    @patch("source.task_handlers.tasks.task_registry._resolve_generation_inputs")
    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_standalone_completed(
        self, mock_ctx, mock_gen, mock_img, mock_struct, mock_build, mock_svi, mock_uni3c, mock_sleep
    ):
        from source.task_handlers.tasks.task_registry import (
            handle_travel_segment_via_queue, SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        mock_ctx.return_value = SegmentContext(
            mode="standalone", orchestrator_details={}, individual_params={},
            segment_idx=0, segment_params={},
        )
        mock_gen.return_value = GenerationInputs(
            model_name="vace", prompt_for_wgp="test", negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720), total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp"), segment_processing_dir=Path("/tmp"),
            debug_enabled=False, travel_mode="vace",
        )
        mock_img.return_value = ImageRefs()
        mock_struct.return_value = StructureOutputs()

        mock_queue = MagicMock()
        mock_status = MagicMock(status="completed", result_path="/output/seg.mp4")
        mock_queue.get_task_status.return_value = mock_status

        success, path = handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-1",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
            is_standalone=True,
        )
        assert success is True
        assert path == "/output/seg.mp4"
        mock_queue.submit_task.assert_called_once()

    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry._apply_uni3c_config")
    @patch("source.task_handlers.tasks.task_registry._apply_svi_config")
    @patch("source.task_handlers.tasks.task_registry._build_generation_params", return_value={"video_length": 81})
    @patch("source.task_handlers.tasks.task_registry._process_structure_guidance")
    @patch("source.task_handlers.tasks.task_registry._resolve_image_references")
    @patch("source.task_handlers.tasks.task_registry._resolve_generation_inputs")
    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_failed_returns_error(
        self, mock_ctx, mock_gen, mock_img, mock_struct, mock_build, mock_svi, mock_uni3c, mock_sleep
    ):
        from source.task_handlers.tasks.task_registry import (
            handle_travel_segment_via_queue, SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        mock_ctx.return_value = SegmentContext(
            mode="standalone", orchestrator_details={}, individual_params={},
            segment_idx=0, segment_params={},
        )
        mock_gen.return_value = GenerationInputs(
            model_name="vace", prompt_for_wgp="test", negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720), total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp"), segment_processing_dir=Path("/tmp"),
            debug_enabled=False, travel_mode="vace",
        )
        mock_img.return_value = ImageRefs()
        mock_struct.return_value = StructureOutputs()

        mock_queue = MagicMock()
        mock_status = MagicMock(status="failed", error_message="GPU OOM")
        mock_queue.get_task_status.return_value = mock_status

        success, msg = handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-2",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
            is_standalone=True,
        )
        assert success is False
        assert "GPU OOM" in msg

    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_exception_returns_error(self, mock_ctx):
        from source.task_handlers.tasks.task_registry import handle_travel_segment_via_queue
        mock_ctx.side_effect = ValueError("bad params")
        mock_queue = MagicMock()

        success, msg = handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-3",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
        )
        assert success is False
        assert "bad params" in msg

    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry._apply_uni3c_config")
    @patch("source.task_handlers.tasks.task_registry._apply_svi_config")
    @patch("source.task_handlers.tasks.task_registry._build_generation_params", return_value={"video_length": 81})
    @patch("source.task_handlers.tasks.task_registry._process_structure_guidance")
    @patch("source.task_handlers.tasks.task_registry._resolve_image_references")
    @patch("source.task_handlers.tasks.task_registry._resolve_generation_inputs")
    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_none_status_returns_error(
        self, mock_ctx, mock_gen, mock_img, mock_struct, mock_build, mock_svi, mock_uni3c, mock_sleep
    ):
        from source.task_handlers.tasks.task_registry import (
            handle_travel_segment_via_queue, SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        mock_ctx.return_value = SegmentContext(
            mode="standalone", orchestrator_details={}, individual_params={},
            segment_idx=0, segment_params={},
        )
        mock_gen.return_value = GenerationInputs(
            model_name="vace", prompt_for_wgp="test", negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720), total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp"), segment_processing_dir=Path("/tmp"),
            debug_enabled=False, travel_mode="vace",
        )
        mock_img.return_value = ImageRefs()
        mock_struct.return_value = StructureOutputs()

        mock_queue = MagicMock()
        mock_queue.get_task_status.return_value = None

        success, msg = handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-4",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
            is_standalone=True,
        )
        assert success is False
        assert "status became None" in msg

    @patch("source.task_handlers.travel.chaining._handle_travel_chaining_after_wgp")
    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry._apply_uni3c_config")
    @patch("source.task_handlers.tasks.task_registry._apply_svi_config")
    @patch("source.task_handlers.tasks.task_registry._build_generation_params", return_value={"video_length": 81})
    @patch("source.task_handlers.tasks.task_registry._process_structure_guidance")
    @patch("source.task_handlers.tasks.task_registry._resolve_image_references")
    @patch("source.task_handlers.tasks.task_registry._resolve_generation_inputs")
    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_orchestrator_mode_runs_chaining(
        self, mock_ctx, mock_gen, mock_img, mock_struct, mock_build, mock_svi, mock_uni3c, mock_sleep, mock_chaining
    ):
        """In orchestrator mode, chaining should be invoked after WGP completion."""
        from source.task_handlers.tasks.task_registry import (
            handle_travel_segment_via_queue, SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        mock_ctx.return_value = SegmentContext(
            mode="orchestrator", orchestrator_details={}, individual_params={},
            segment_idx=0, segment_params={},
            orchestrator_task_id_ref="orch-1", orchestrator_run_id="run-1",
        )
        mock_gen.return_value = GenerationInputs(
            model_name="vace", prompt_for_wgp="test", negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720), total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp"), segment_processing_dir=Path("/tmp"),
            debug_enabled=False, travel_mode="vace",
        )
        mock_img.return_value = ImageRefs()
        mock_struct.return_value = StructureOutputs()
        mock_chaining.return_value = (True, "chaining OK", "/chained/output.mp4")

        mock_queue = MagicMock()
        mock_status = MagicMock(status="completed", result_path="/raw/output.mp4")
        mock_queue.get_task_status.return_value = mock_status

        success, path = handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-5",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
            is_standalone=False,
        )
        assert success is True
        assert path == "/chained/output.mp4"
        mock_chaining.assert_called_once()

    @patch("source.task_handlers.travel.chaining._handle_travel_chaining_after_wgp")
    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry._apply_uni3c_config")
    @patch("source.task_handlers.tasks.task_registry._apply_svi_config")
    @patch("source.task_handlers.tasks.task_registry._build_generation_params", return_value={"video_length": 81})
    @patch("source.task_handlers.tasks.task_registry._process_structure_guidance")
    @patch("source.task_handlers.tasks.task_registry._resolve_image_references")
    @patch("source.task_handlers.tasks.task_registry._resolve_generation_inputs")
    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_orchestrator_mode_chaining_failure_returns_raw_path(
        self, mock_ctx, mock_gen, mock_img, mock_struct, mock_build, mock_svi, mock_uni3c, mock_sleep, mock_chaining
    ):
        """If chaining fails, should fall back to raw WGP output path."""
        from source.task_handlers.tasks.task_registry import (
            handle_travel_segment_via_queue, SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        mock_ctx.return_value = SegmentContext(
            mode="orchestrator", orchestrator_details={}, individual_params={},
            segment_idx=0, segment_params={},
            orchestrator_task_id_ref="orch-1", orchestrator_run_id="run-1",
        )
        mock_gen.return_value = GenerationInputs(
            model_name="vace", prompt_for_wgp="test", negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720), total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp"), segment_processing_dir=Path("/tmp"),
            debug_enabled=False, travel_mode="vace",
        )
        mock_img.return_value = ImageRefs()
        mock_struct.return_value = StructureOutputs()
        mock_chaining.return_value = (False, "chaining failed", None)

        mock_queue = MagicMock()
        mock_status = MagicMock(status="completed", result_path="/raw/output.mp4")
        mock_queue.get_task_status.return_value = mock_status

        success, path = handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-6",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
            is_standalone=False,
        )
        assert success is True
        assert path == "/raw/output.mp4"

    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry._apply_uni3c_config")
    @patch("source.task_handlers.tasks.task_registry._apply_svi_config")
    @patch("source.task_handlers.tasks.task_registry._build_generation_params", return_value={"video_length": 81})
    @patch("source.task_handlers.tasks.task_registry._process_structure_guidance")
    @patch("source.task_handlers.tasks.task_registry._resolve_image_references")
    @patch("source.task_handlers.tasks.task_registry._resolve_generation_inputs")
    @patch("source.task_handlers.tasks.task_registry._resolve_segment_context")
    def test_source_task_type_injected_into_params(
        self, mock_ctx, mock_gen, mock_img, mock_struct, mock_build, mock_svi, mock_uni3c, mock_sleep
    ):
        """generation_params should have _source_task_type='travel_segment'."""
        from source.task_handlers.tasks.task_registry import (
            handle_travel_segment_via_queue, SegmentContext, GenerationInputs, ImageRefs, StructureOutputs,
        )
        mock_ctx.return_value = SegmentContext(
            mode="standalone", orchestrator_details={}, individual_params={},
            segment_idx=0, segment_params={},
        )
        mock_gen.return_value = GenerationInputs(
            model_name="vace", prompt_for_wgp="test", negative_prompt_for_wgp=" ",
            parsed_res_wh=(1280, 720), total_frames_for_segment=81,
            current_run_base_output_dir=Path("/tmp"), segment_processing_dir=Path("/tmp"),
            debug_enabled=False, travel_mode="vace",
        )
        mock_img.return_value = ImageRefs()
        mock_struct.return_value = StructureOutputs()

        # Return params dict that we can inspect
        build_result = {"video_length": 81}
        mock_build.return_value = build_result

        mock_queue = MagicMock()
        mock_status = MagicMock(status="completed", result_path="/out.mp4")
        mock_queue.get_task_status.return_value = mock_status

        handle_travel_segment_via_queue(
            task_params_dict={},
            main_output_dir_base=Path("/tmp"),
            task_id="seg-task-7",
            colour_match_videos=False,
            mask_active_frames=False,
            task_queue=mock_queue,
            is_standalone=True,
        )
        assert build_result["_source_task_type"] == "travel_segment"


# ---------------------------------------------------------------------------
# Additional _handle_direct_queue_task edge cases
# ---------------------------------------------------------------------------

class TestHandleDirectQueueTaskEdgeCases:
    """Additional edge case tests for _handle_direct_queue_task."""

    def _make_context(self, **overrides):
        mock_queue = MagicMock()
        ctx = {
            "task_id": "edge-task-1",
            "task_params_dict": {"prompt": "test"},
            "main_output_dir_base": Path("/tmp/output"),
            "project_id": "proj-1",
            "task_queue": mock_queue,
            "colour_match_videos": False,
            "mask_active_frames": False,
            "debug_mode": False,
            "wan2gp_path": "/opt/wan2gp",
        }
        ctx.update(overrides)
        return ctx

    @patch("source.task_handlers.tasks.task_registry.time.sleep")
    @patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task")
    def test_failed_without_message(self, mock_convert, mock_sleep):
        from source.task_handlers.tasks.task_registry import TaskRegistry
        mock_gen_task = MagicMock()
        mock_gen_task.parameters = {}
        mock_convert.return_value = mock_gen_task

        ctx = self._make_context()
        mock_status = MagicMock(status="failed", error_message=None)
        ctx["task_queue"].get_task_status.return_value = mock_status

        result = TaskRegistry._handle_direct_queue_task("t2v", ctx)
        assert result == (False, "Failed without message")

    @patch("source.task_handlers.tasks.task_registry.db_task_to_generation_task")
    def test_runtime_error_returns_failure(self, mock_convert):
        from source.task_handlers.tasks.task_registry import TaskRegistry
        mock_convert.side_effect = RuntimeError("unexpected error")
        ctx = self._make_context()
        result = TaskRegistry._handle_direct_queue_task("t2v", ctx)
        assert result[0] is False
        assert "unexpected error" in result[1]


# ---------------------------------------------------------------------------
# _resolve_segment_context additional edge cases
# ---------------------------------------------------------------------------

class TestResolveSegmentContextEdgeCases:
    """Additional edge cases for _resolve_segment_context."""

    def test_db_fetch_returns_json_string(self):
        """DB may return a JSON string that needs parsing."""
        import json
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "segment_index": 0,
            "orchestrator_task_id_ref": "orch-1",
        }
        fetched_json = json.dumps({"orchestrator_details": {"model_name": "from_json"}})
        with patch("source.task_handlers.tasks.task_registry.db_ops.get_task_params", return_value=fetched_json):
            ctx = _resolve_segment_context(params, is_standalone=False, task_id="task-1")
        assert ctx.orchestrator_details == {"model_name": "from_json"}

    def test_db_fetch_returns_none_raises(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "segment_index": 0,
            "orchestrator_task_id_ref": "orch-1",
        }
        with patch("source.task_handlers.tasks.task_registry.db_ops.get_task_params", return_value=None):
            with pytest.raises(ValueError, match="Could not retrieve orchestrator_details"):
                _resolve_segment_context(params, is_standalone=False, task_id="task-1")

    def test_individual_params_default_empty_dict(self):
        from source.task_handlers.tasks.task_registry import _resolve_segment_context

        params = {
            "orchestrator_details": {"model_name": "vace"},
        }
        ctx = _resolve_segment_context(params, is_standalone=True, task_id="task-1")
        assert ctx.individual_params == {}
