"""
Structural regression gate — pure import-based checks.

No orchestrator, no smoke mode.  Catches accidental removals/additions
in task-type sets, model mappings, handler coverage, and config files.

Run with:
    python -m pytest tests/test_service_health.py -v
    python tests/test_service_health.py          # standalone
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# TestTaskTypeIntegrity — Set sizes haven't changed
# ===================================================================

class TestTaskTypeIntegrity:
    """Verify exact sizes of the task-type sets and model mapping."""

    def test_direct_queue_count(self):
        from source.task_types import DIRECT_QUEUE_TASK_TYPES
        assert len(DIRECT_QUEUE_TASK_TYPES) == 22, (
            f"DIRECT_QUEUE_TASK_TYPES has {len(DIRECT_QUEUE_TASK_TYPES)} entries, expected 22. "
            f"Contents: {sorted(DIRECT_QUEUE_TASK_TYPES)}"
        )

    def test_wgp_task_count(self):
        from source.task_types import WGP_TASK_TYPES
        assert len(WGP_TASK_TYPES) == 20, (
            f"WGP_TASK_TYPES has {len(WGP_TASK_TYPES)} entries, expected 20. "
            f"Contents: {sorted(WGP_TASK_TYPES)}"
        )

    def test_model_mapping_count(self):
        from source.task_types import TASK_TYPE_TO_MODEL
        assert len(TASK_TYPE_TO_MODEL) >= 24, (
            f"TASK_TYPE_TO_MODEL has {len(TASK_TYPE_TO_MODEL)} entries, expected >= 24. "
            f"Keys: {sorted(TASK_TYPE_TO_MODEL.keys())}"
        )


# ===================================================================
# TestSetConsistency — WGP / DIRECT sets are consistent
# ===================================================================

class TestSetConsistency:
    """WGP_TASK_TYPES and DIRECT_QUEUE_TASK_TYPES are related but not identical."""

    def test_wgp_minus_direct_is_inpaint_frames(self):
        """WGP has 'inpaint_frames' which is NOT in DIRECT_QUEUE (it's orchestrated)."""
        from source.task_types import WGP_TASK_TYPES, DIRECT_QUEUE_TASK_TYPES
        wgp_only = WGP_TASK_TYPES - DIRECT_QUEUE_TASK_TYPES
        assert "inpaint_frames" in wgp_only, (
            f"Expected 'inpaint_frames' in WGP-only set, got: {wgp_only}"
        )

    def test_direct_minus_wgp_is_qwen_extras(self):
        """DIRECT has extra Qwen variants not in WGP."""
        from source.task_types import WGP_TASK_TYPES, DIRECT_QUEUE_TASK_TYPES
        direct_only = DIRECT_QUEUE_TASK_TYPES - WGP_TASK_TYPES
        expected_extras = {"qwen_image_hires", "qwen_image", "qwen_image_2512"}
        assert direct_only == expected_extras, (
            f"Expected DIRECT-only to be {expected_extras}, got: {direct_only}"
        )

    def test_all_direct_types_have_model_mapping(self):
        """Every DIRECT_QUEUE task type must have an explicit model mapping."""
        from source.task_types import DIRECT_QUEUE_TASK_TYPES, TASK_TYPE_TO_MODEL
        missing = [t for t in DIRECT_QUEUE_TASK_TYPES if t not in TASK_TYPE_TO_MODEL]
        assert not missing, (
            f"These DIRECT_QUEUE types lack an explicit model mapping: {missing}"
        )

    def test_no_direct_type_uses_fallback(self):
        """No DIRECT_QUEUE task should fall through to the 't2v' fallback."""
        from source.task_types import DIRECT_QUEUE_TASK_TYPES, get_default_model, TASK_TYPE_TO_MODEL
        fallbacks = [
            t for t in DIRECT_QUEUE_TASK_TYPES
            if t not in TASK_TYPE_TO_MODEL
        ]
        assert not fallbacks, (
            f"These DIRECT_QUEUE types would use the 't2v' fallback: {fallbacks}"
        )


# ===================================================================
# TestHandlerCoverage — Dispatch covers all non-direct task types
# ===================================================================

class TestHandlerCoverage:
    """Verify the task_registry handler dict covers the expected types."""

    EXPECTED_HANDLER_TYPES = frozenset({
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
    })

    def test_all_non_direct_types_have_handlers(self):
        """All 14 expected handler types must appear in task_registry.py."""
        registry_src = (PROJECT_ROOT / "source" / "task_registry.py").read_text()
        missing = [h for h in self.EXPECTED_HANDLER_TYPES if f'"{h}"' not in registry_src]
        assert not missing, (
            f"Handler types missing from task_registry.py: {missing}"
        )
        assert len(self.EXPECTED_HANDLER_TYPES) == 14

    def test_no_handler_overlaps_direct_queue(self):
        """Handler types must not also be in DIRECT_QUEUE_TASK_TYPES
        (except inpaint_frames which appears in both WGP and handlers)."""
        from source.task_types import DIRECT_QUEUE_TASK_TYPES
        overlap = self.EXPECTED_HANDLER_TYPES & DIRECT_QUEUE_TASK_TYPES
        assert not overlap, (
            f"These handler types also appear in DIRECT_QUEUE_TASK_TYPES: {overlap}. "
            f"Handler types should be dispatched via the handler dict, not the direct queue."
        )


# ===================================================================
# TestModelConfigHealth — Config files parseable
# ===================================================================

class TestModelConfigHealth:
    """Verify Wan2GP/defaults/*.json are valid and key configs exist."""

    DEFAULTS_DIR = PROJECT_ROOT / "Wan2GP" / "defaults"

    def test_all_config_files_valid_json(self):
        """Every .json in defaults/ must parse without errors."""
        bad_files = []
        json_files = list(self.DEFAULTS_DIR.glob("*.json"))
        assert len(json_files) > 50, (
            f"Expected > 50 config files in defaults/, found {len(json_files)}"
        )
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                assert isinstance(data, dict), f"{jf.name} did not parse to a dict"
            except Exception as e:
                bad_files.append(f"{jf.name}: {e}")
        assert not bad_files, (
            f"Invalid JSON configs:\n" + "\n".join(bad_files)
        )

    def test_ltx2_config_not_corrupted(self):
        """ltx2_19B.json must parse and contain a 'model' key."""
        cfg = json.loads((self.DEFAULTS_DIR / "ltx2_19B.json").read_text())
        assert "model" in cfg, "ltx2_19B.json missing 'model' key"

    def test_ltxv_config_exists(self):
        """ltxv_13B.json must exist."""
        assert (self.DEFAULTS_DIR / "ltxv_13B.json").is_file()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
