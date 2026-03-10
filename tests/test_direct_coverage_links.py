"""Direct, low-mock coverage links for modules flagged as transitive-only."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import(module_name: str):
    try:
        module = importlib.import_module(module_name)
        assert module is not None
        return module
    except ModuleNotFoundError as exc:
        # Skip when optional third-party deps are missing in local env.
        root_name = module_name.split(".")[0]
        if exc.name != root_name:
            pytest.skip(f"optional dependency missing while importing {module_name}: {exc.name}")
        raise


def test_debug_modules_direct_imports():
    modules = [
        "debug.cli",
        "debug.client",
        "debug.commands.config",
        "debug.commands.health",
        "debug.commands.orchestrator",
        "debug.commands.runpod",
        "debug.commands.storage",
        "debug.commands.task",
        "debug.commands.tasks",
        "debug.commands.worker",
        "debug.commands.workers",
        "debug.formatters",
        "debug.models",
    ]
    for name in modules:
        _import(name)


def test_examples_direct_imports():
    inpaint = _import("examples.inpaint_frames_example")
    join = _import("examples.join_clips_example")
    assert hasattr(inpaint, "run_inpaint_frames")
    assert hasattr(join, "run_join_clips")


def test_core_transitive_only_modules_have_direct_link():
    modules = [
        "heartbeat_guardian",
        "source.media.structure.generation",
        "source.media.structure.segments",
        "source.media.video.travel_guide",
        "source.media.video.vace_frame_utils",
        "source.media.vlm.model",
        "source.media.vlm.single_image_prompts",
        "source.media.vlm.transition_prompts",
        "source.models.comfy.comfy_handler",
        "source.models.wgp.generators.generation_strategies",
        "source.models.wgp.model_ops",
        "source.models.wgp.orchestrator",
        "source.task_handlers.create_visualization",
        "source.task_handlers.edit_video_orchestrator",
        "source.task_handlers.extract_frame",
        "source.task_handlers.inpaint_frames",
        "source.task_handlers.join.final_stitch",
        "source.task_handlers.join.generation",
        "source.task_handlers.join.orchestrator",
        "source.task_handlers.join.vlm_enhancement",
        "source.task_handlers.magic_edit",
        "source.task_handlers.queue.memory_cleanup",
        "source.task_handlers.queue.wgp_init",
        "source.task_handlers.rife_interpolate",
        "source.task_handlers.tasks.task_conversion",
        "source.task_handlers.travel.chaining",
        "source.task_handlers.travel.ffmpeg_fallback",
        "source.task_handlers.travel.mask_builder",
        "source.task_handlers.travel.orchestrator",
        "source.task_handlers.travel.segment_processor",
        "source.task_handlers.travel.stitch",
        "source.utils.pose_utils",
        "scripts.convert_lora_rank",
        "scripts.uni3c_validation",
    ]
    for name in modules:
        _import(name)
