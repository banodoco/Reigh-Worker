"""
Tests for centralized modules created during refactoring.

Tests cover:
- source/task_handlers/tasks/task_types.py - Task type definitions and helpers
- source/models/lora/lora_paths.py - LoRA directory configuration
- (deleted — dead code) - Parameter name normalization
- source/core/platform_utils.py - Platform-specific utilities
- source/models/wgp/wgp_patches.py - WGP monkeypatch functions
"""

import pytest
from pathlib import Path


class TestTaskTypes:
    """Tests for source/task_handlers/tasks/task_types.py"""

    def test_imports(self):
        """Verify all exports are importable."""
        from source.task_handlers.tasks.task_types import (
            WGP_TASK_TYPES,
            DIRECT_QUEUE_TASK_TYPES,
            TASK_TYPE_TO_MODEL,
            get_default_model,
        )
        assert WGP_TASK_TYPES is not None
        assert DIRECT_QUEUE_TASK_TYPES is not None
        assert TASK_TYPE_TO_MODEL is not None
        assert callable(get_default_model)

    def test_wgp_task_types_is_frozenset(self):
        """WGP_TASK_TYPES should be immutable."""
        from source.task_handlers.tasks.task_types import WGP_TASK_TYPES
        assert isinstance(WGP_TASK_TYPES, frozenset)
        with pytest.raises(AttributeError):
            WGP_TASK_TYPES.add("test")

    def test_direct_queue_task_types_is_frozenset(self):
        """DIRECT_QUEUE_TASK_TYPES should be immutable."""
        from source.task_handlers.tasks.task_types import DIRECT_QUEUE_TASK_TYPES
        assert isinstance(DIRECT_QUEUE_TASK_TYPES, frozenset)

    def test_common_task_types_present(self):
        """Verify common task types are defined."""
        from source.task_handlers.tasks.task_types import WGP_TASK_TYPES, DIRECT_QUEUE_TASK_TYPES

        # These should be in WGP_TASK_TYPES
        assert "vace" in WGP_TASK_TYPES
        assert "t2v" in WGP_TASK_TYPES
        assert "i2v" in WGP_TASK_TYPES
        assert "flux" in WGP_TASK_TYPES

        # These should be in DIRECT_QUEUE_TASK_TYPES
        assert "vace" in DIRECT_QUEUE_TASK_TYPES
        assert "t2v" in DIRECT_QUEUE_TASK_TYPES

    def test_get_default_model_known_types(self):
        """get_default_model should return correct defaults."""
        from source.task_handlers.tasks.task_types import get_default_model, TASK_TYPE_TO_MODEL

        # Test exact values match the mapping dict
        for task_type, expected_model in TASK_TYPE_TO_MODEL.items():
            assert get_default_model(task_type) == expected_model
        assert get_default_model("flux") == "flux"

    def test_get_default_model_unknown_type(self):
        """get_default_model should return fallback for unknown types."""
        from source.task_handlers.tasks.task_types import get_default_model

        result = get_default_model("unknown_task_type_xyz")
        assert result == "t2v"  # Default fallback

    def test_task_type_to_model_dict(self):
        """TASK_TYPE_TO_MODEL should be a dict with string keys/values."""
        from source.task_handlers.tasks.task_types import TASK_TYPE_TO_MODEL

        assert isinstance(TASK_TYPE_TO_MODEL, dict)
        assert len(TASK_TYPE_TO_MODEL) > 0

        for key, value in TASK_TYPE_TO_MODEL.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestLoraPaths:
    """Tests for source/models/lora/lora_paths.py"""

    def test_imports(self):
        """Verify all exports are importable."""
        from source.models.lora.lora_paths import (
            get_lora_search_dirs,
            get_lora_dir_for_model,
        )
        assert callable(get_lora_search_dirs)
        assert callable(get_lora_dir_for_model)

    def test_get_lora_search_dirs_returns_list(self):
        """get_lora_search_dirs should return a list of Paths."""
        from source.models.lora.lora_paths import get_lora_search_dirs

        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        result = get_lora_search_dirs(wan_dir)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(p, Path) for p in result)

    def test_get_lora_search_dirs_includes_common_dirs(self):
        """get_lora_search_dirs should include standard LoRA directories."""
        from source.models.lora.lora_paths import get_lora_search_dirs

        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        result = get_lora_search_dirs(wan_dir)
        dir_names = [p.name for p in result]

        # Should include common directories
        assert "loras" in dir_names or any("loras" in str(p) for p in result)

    def test_get_lora_dir_for_model_wan(self):
        """get_lora_dir_for_model should return wan directory for wan models."""
        from source.models.lora.lora_paths import get_lora_dir_for_model

        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        result = get_lora_dir_for_model("wan_2_1_t2v", wan_dir)

        assert isinstance(result, Path)
        assert "loras" in str(result)

    def test_get_lora_dir_for_model_flux(self):
        """get_lora_dir_for_model should return flux directory for flux models."""
        from source.models.lora.lora_paths import get_lora_dir_for_model

        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        result = get_lora_dir_for_model("flux", wan_dir)

        assert isinstance(result, Path)
        assert "flux" in str(result).lower()

    def test_get_lora_dir_for_model_qwen(self):
        """get_lora_dir_for_model should return qwen directory for qwen models."""
        from source.models.lora.lora_paths import get_lora_dir_for_model

        wan_dir = Path(__file__).parent.parent / "Wan2GP"
        result = get_lora_dir_for_model("qwen_image_edit", wan_dir)

        assert isinstance(result, Path)
        assert "qwen" in str(result).lower()


class TestParamAliases:
    """Tests for source/param_aliases.py - REMOVED (dead code)"""

    def test_module_removed(self):
        """param_aliases was identified as dead code and deleted."""
        import importlib
        assert importlib.util.find_spec("source.param_aliases") is None



class TestPlatformUtils:
    """Tests for source/core/platform_utils.py"""

    def test_imports(self):
        """Verify all exports are importable."""
        from source.core.platform_utils import (
            suppress_alsa_errors,
            setup_headless_environment,
        )
        assert callable(suppress_alsa_errors)
        assert callable(setup_headless_environment)

    def test_suppress_alsa_errors_does_not_crash(self):
        """suppress_alsa_errors should not raise on any platform."""
        from source.core.platform_utils import suppress_alsa_errors

        # Should not raise even on non-Linux
        suppress_alsa_errors()

    def test_setup_headless_environment_does_not_crash(self, monkeypatch):
        """setup_headless_environment should not raise."""
        from source.core.platform_utils import setup_headless_environment

        # monkeypatch.delenv restores automatically after the test
        for key in ("SDL_AUDIODRIVER", "SDL_VIDEODRIVER"):
            monkeypatch.delenv(key, raising=False)

        setup_headless_environment()
        import os
        assert "SDL_AUDIODRIVER" in os.environ


class TestWgpPatches:
    """Tests for source/models/wgp/wgp_patches.py"""

    def test_imports(self):
        """Verify all exports are importable."""
        from source.models.wgp.wgp_patches import (
            apply_qwen_model_routing_patch,
            apply_qwen_lora_directory_patch,
            apply_lora_multiplier_parser_patch,
            apply_qwen_inpainting_lora_patch,
            apply_all_wgp_patches,
        )
        assert callable(apply_qwen_model_routing_patch)
        assert callable(apply_qwen_lora_directory_patch)
        assert callable(apply_lora_multiplier_parser_patch)
        assert callable(apply_qwen_inpainting_lora_patch)
        assert callable(apply_all_wgp_patches)

    def test_apply_all_wgp_patches_returns_dict(self):
        """apply_all_wgp_patches should return a results dict."""
        from source.models.wgp.wgp_patches import apply_all_wgp_patches
        from types import ModuleType

        # Create a mock wgp module
        mock_wgp = ModuleType("mock_wgp")
        mock_wgp.load_wan_model = lambda *args, **kwargs: None
        mock_wgp.get_lora_dir = lambda model_type: "/tmp"
        mock_wgp.get_base_model_type = lambda x: x

        wan_root = str(Path(__file__).parent.parent / "Wan2GP")

        # Should return dict even with mock module
        result = apply_all_wgp_patches(mock_wgp, wan_root)
        assert isinstance(result, dict)
        assert "qwen_model_routing" in result
        assert "qwen_lora_directory" in result


class TestTaskHandlersImports:
    """Tests for source/task_handlers/ imports."""

    def test_task_handlers_package_imports(self):
        """Verify task_handlers package is importable."""
        from source import task_handlers
        assert task_handlers is not None

    def test_travel_orchestrator_imports(self):
        """Verify travel orchestrator module imports."""
        from source.task_handlers.travel import orchestrator
        assert hasattr(orchestrator, 'handle_travel_orchestrator_task')

    def test_join_clips_imports(self):
        """Verify join_clips module imports."""
        from source.task_handlers.join import generation
        assert hasattr(generation, 'handle_join_clips_task')

    def test_magic_edit_imports(self):
        """Verify magic_edit module imports."""
        from source.task_handlers import magic_edit
        assert hasattr(magic_edit, 'handle_magic_edit_task')

    def test_inpaint_frames_imports(self):
        """Verify inpaint_frames module imports."""
        from source.task_handlers import inpaint_frames
        assert hasattr(inpaint_frames, '_handle_inpaint_frames_task')


class TestPhase1BugFixes:
    """Tests verifying bug fixes are in place."""

    def test_db_operations_uses_httpx(self):
        """db_operations should use httpx, not requests."""
        # db_operations is now a re-export facade; check the actual submodules
        from source.core.db import edge_helpers
        import inspect

        source_code = inspect.getsource(edge_helpers)

        # Should have httpx import
        assert "import httpx" in source_code

    def test_no_hardcoded_worker_id(self):
        """db_operations should not have hardcoded worker_id."""
        # db_operations is now a re-export facade; check the actual submodules
        from source.core.db import task_claim, task_completion
        import inspect

        for mod in (task_claim, task_completion):
            source_code = inspect.getsource(mod)
            # Should not contain the old hardcoded worker_id
            assert "gpu-20250723_221138-afa8403b" not in source_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
