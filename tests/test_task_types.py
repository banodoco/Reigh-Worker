"""Tests for source/task_handlers/tasks/task_types.py."""

import sys
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TASK_TYPES_PATH = REPO_ROOT / "source" / "task_handlers" / "tasks" / "task_types.py"
_SPEC = importlib.util.spec_from_file_location("task_types_module", _TASK_TYPES_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

WGP_TASK_TYPES = _MODULE.WGP_TASK_TYPES
DIRECT_QUEUE_TASK_TYPES = _MODULE.DIRECT_QUEUE_TASK_TYPES
TASK_TYPE_TO_MODEL = _MODULE.TASK_TYPE_TO_MODEL
get_default_model = _MODULE.get_default_model
is_wgp_task = _MODULE.is_wgp_task
is_direct_queue_task = _MODULE.is_direct_queue_task


class TestGetDefaultModel:
    def test_known_task_type(self):
        assert get_default_model("vace") == "vace_14B_cocktail_2_2"
        assert get_default_model("t2v") == "t2v"
        assert get_default_model("flux") == "flux"

    def test_qwen_types(self):
        assert get_default_model("qwen_image_edit") == "qwen_image_edit_20B"
        assert get_default_model("qwen_image_2512") == "qwen_image_2512_20B"
        assert get_default_model("z_image_turbo") == "z_image"
        assert get_default_model("z_image_turbo_i2i") == "z_image_img2img"

    def test_unknown_type_returns_t2v(self):
        assert get_default_model("nonexistent") == "t2v"

    def test_empty_string(self):
        assert get_default_model("") == "t2v"


class TestIsWgpTask:
    def test_wgp_task(self):
        assert is_wgp_task("vace") is True
        assert is_wgp_task("t2v") is True
        assert is_wgp_task("qwen_image_edit") is True

    def test_non_wgp_task(self):
        assert is_wgp_task("join_clips") is False
        assert is_wgp_task("travel_orchestrator") is False
        assert is_wgp_task("") is False


class TestIsDirectQueueTask:
    def test_direct_queue_task(self):
        assert is_direct_queue_task("vace") is True
        assert is_direct_queue_task("qwen_image") is True
        assert is_direct_queue_task("z_image_turbo") is True

    def test_non_direct_queue_task(self):
        assert is_direct_queue_task("join_clips") is False
        assert is_direct_queue_task("") is False


class TestFrozenSetConsistency:
    def test_direct_queue_is_superset_of_wgp_minus_orchestrated(self):
        """Most WGP types should also be direct-queueable."""
        # These are WGP tasks that need orchestration (NOT direct queue)
        orchestrated = {"inpaint_frames"}
        for task_type in WGP_TASK_TYPES:
            if task_type not in orchestrated:
                assert task_type in DIRECT_QUEUE_TASK_TYPES, (
                    f"{task_type} is a WGP task but not direct-queueable"
                )

    def test_all_model_mapped_tasks_have_entries(self):
        """Every task in TASK_TYPE_TO_MODEL should map to a non-empty model."""
        for task_type, model in TASK_TYPE_TO_MODEL.items():
            assert model, f"Task type '{task_type}' maps to empty model"

    def test_segment_task_defaults_preserved(self):
        """Critical task defaults should remain on legacy compatibility IDs."""
        assert TASK_TYPE_TO_MODEL["join_clips_segment"] == "wan_2_2_vace_lightning_baseline_2_2_2"
        assert TASK_TYPE_TO_MODEL["inpaint_frames"] == "wan_2_2_vace_lightning_baseline_2_2_2"

    def test_sets_are_frozen(self):
        assert isinstance(WGP_TASK_TYPES, frozenset)
        assert isinstance(DIRECT_QUEUE_TASK_TYPES, frozenset)
