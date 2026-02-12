"""Join clips task handlers package."""
from source.task_handlers.join.generation import _handle_join_clips_task
from source.task_handlers.join.final_stitch import _handle_join_final_stitch
from source.task_handlers.join.orchestrator import _handle_join_clips_orchestrator_task
from source.task_handlers.join.shared import (
    _check_orchestrator_cancelled,
    _extract_join_settings_from_payload,
    _check_existing_join_tasks,
    _create_join_chain_tasks,
    _create_parallel_join_tasks,
    calculate_min_clip_frames,
    validate_clip_frames_for_join,
)
from source.task_handlers.join.vlm_enhancement import (
    _extract_boundary_frames_for_vlm,
    _generate_vlm_prompts_for_joins,
)
