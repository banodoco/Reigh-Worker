"""
Shared core logic for join orchestrators.

Used by both join_clips_orchestrator and edit_video_orchestrator.
"""

import json
import traceback
from pathlib import Path
from typing import Tuple, List, Optional

from source import db_operations as db_ops
from source.utils import download_video_if_url, get_video_frame_count_and_fps

__all__ = [
    "_check_orchestrator_cancelled",
    "calculate_min_clip_frames",
    "validate_clip_frames_for_join",
    "_extract_join_settings_from_payload",
    "_check_existing_join_tasks",
    "_create_join_chain_tasks",
    "_create_parallel_join_tasks",
]


def _check_orchestrator_cancelled(orchestrator_task_id: str, context_msg: str, dprint=print) -> str | None:
    """Check if orchestrator was cancelled and cancel children if so.

    Returns an error message string if cancelled, None if still active.
    """
    status = db_ops.get_task_current_status(orchestrator_task_id)
    if status and status.lower() in ('cancelled', 'canceled'):
        dprint(f"[CANCELLATION] Join orchestrator {orchestrator_task_id} was cancelled - {context_msg}")
        db_ops.cancel_orchestrator_children(orchestrator_task_id, reason="Orchestrator cancelled by user")
        return f"Orchestrator cancelled: {context_msg}"
    return None


def calculate_min_clip_frames(gap_frame_count: int, context_frame_count: int, replace_mode: bool) -> int:
    """
    Calculate the minimum number of frames a clip must have to safely join.

    In REPLACE mode, we need:
        gap_frame_count + 2 * context_frame_count <= min_clip_frames

    This ensures that context frames don't overlap with previously blended regions
    in chained joins, avoiding the "double-blending" artifact.

    Args:
        gap_frame_count: Number of frames in the generated gap/transition
        context_frame_count: Number of context frames from each clip boundary
        replace_mode: Whether REPLACE mode is enabled

    Returns:
        Minimum required frames for each clip
    """
    if replace_mode:
        return gap_frame_count + 2 * context_frame_count
    else:
        return 2 * context_frame_count


def validate_clip_frames_for_join(
    clip_list: List[dict],
    gap_frame_count: int,
    context_frame_count: int,
    replace_mode: bool,
    temp_dir: Path,
    orchestrator_task_id: str,
    dprint
) -> Tuple[bool, str, List[int]]:
    """
    Validate that all clips have enough frames for safe joining.

    Args:
        clip_list: List of clip dicts with 'url' keys
        gap_frame_count: Gap frames for transition
        context_frame_count: Context frames from each boundary
        replace_mode: Whether REPLACE mode is enabled
        temp_dir: Directory to download clips for frame counting
        orchestrator_task_id: Task ID for logging
        dprint: Debug print function

    Returns:
        Tuple of (is_valid, error_message, frame_counts_per_clip)
    """
    min_frames = calculate_min_clip_frames(gap_frame_count, context_frame_count, replace_mode)
    dprint(f"[VALIDATE_CLIPS] Minimum required frames per clip: {min_frames}")
    dprint(f"[VALIDATE_CLIPS]   (gap={gap_frame_count} + 2\u00d7context={context_frame_count}, replace_mode={replace_mode})")

    frame_counts = []
    violations = []

    for idx, clip in enumerate(clip_list):
        clip_url = clip.get("url")
        if not clip_url:
            return False, f"Clip {idx} missing 'url' field", []

        # Download clip to count frames
        local_path = download_video_if_url(
            clip_url,
            download_target_dir=temp_dir,
            task_id_for_logging=orchestrator_task_id,
            descriptive_name=f"validate_clip_{idx}"
        )

        if not local_path or not Path(local_path).exists():
            return False, f"Failed to download clip {idx} for validation: {clip_url}", []

        # Get frame count
        frames, fps = get_video_frame_count_and_fps(str(local_path))
        if not frames:
            return False, f"Could not determine frame count for clip {idx}", []

        frame_counts.append(frames)
        dprint(f"[VALIDATE_CLIPS] Clip {idx}: {frames} frames (min required: {min_frames})")

        # First and last clips only need half the minimum (only one boundary)
        is_first = (idx == 0)
        is_last = (idx == len(clip_list) - 1)

        if is_first or is_last:
            if replace_mode:
                gap_from_side = gap_frame_count // 2 if is_first else (gap_frame_count - gap_frame_count // 2)
                required = context_frame_count + gap_from_side
            else:
                required = context_frame_count
        else:
            required = min_frames

        if frames < required:
            violations.append({
                "idx": idx,
                "frames": frames,
                "required": required,
                "shortfall": required - frames
            })

    if violations:
        min_available = min(frame_counts)
        total_needed = gap_frame_count + 2 * context_frame_count
        ratio = min_available / total_needed
        reduced_gap = max(1, int(gap_frame_count * ratio))
        reduced_context = max(1, int(context_frame_count * ratio))

        warning_parts = []
        for v in violations:
            warning_parts.append(f"Clip {v['idx']}: {v['frames']} frames < {v['required']} required")

        warning_msg = (
            f"[PROPORTIONAL_REDUCTION] Some clips are shorter than ideal:\n  "
            + "\n  ".join(warning_parts)
            + f"\n  Original settings: gap={gap_frame_count}, context={context_frame_count}"
            + f"\n  Will reduce to approximately: gap\u2248{reduced_gap}, context\u2248{reduced_context} ({ratio:.0%} of original)"
            + f"\n  Transitions will be shorter but still generated successfully."
        )
        dprint(f"[VALIDATE_CLIPS] {warning_msg}")

        return True, warning_msg, frame_counts

    dprint(f"[VALIDATE_CLIPS] All {len(clip_list)} clips have sufficient frames")
    return True, "", frame_counts


def _extract_join_settings_from_payload(orchestrator_payload: dict) -> dict:
    """
    Extract standardized join settings from an orchestrator payload.

    Used by both join_clips_orchestrator and edit_video_orchestrator.

    Args:
        orchestrator_payload: The orchestrator_details dict

    Returns:
        Dict of join settings for join_clips_segment tasks
    """
    use_input_res = orchestrator_payload.get("use_input_video_resolution", False)

    return {
        "context_frame_count": orchestrator_payload.get("context_frame_count", 8),
        "gap_frame_count": orchestrator_payload.get("gap_frame_count", 53),
        "replace_mode": orchestrator_payload.get("replace_mode", False),
        "prompt": orchestrator_payload.get("prompt", "smooth transition"),
        "negative_prompt": orchestrator_payload.get("negative_prompt", ""),
        "model": orchestrator_payload.get("model", "wan_2_2_vace_lightning_baseline_2_2_2"),
        "aspect_ratio": orchestrator_payload.get("aspect_ratio"),
        "resolution": None if use_input_res else orchestrator_payload.get("resolution"),
        "use_input_video_resolution": use_input_res,
        "fps": orchestrator_payload.get("fps"),
        "use_input_video_fps": orchestrator_payload.get("use_input_video_fps", False),
        "phase_config": orchestrator_payload.get("phase_config"),
        "num_inference_steps": orchestrator_payload.get("num_inference_steps"),
        "guidance_scale": orchestrator_payload.get("guidance_scale"),
        "seed": orchestrator_payload.get("seed", -1),
        # LoRA parameters
        "additional_loras": orchestrator_payload.get("additional_loras", {}),
        # Keep bridging image param
        "keep_bridging_images": orchestrator_payload.get("keep_bridging_images", False),
        # Vid2vid initialization for replace mode
        "vid2vid_init_strength": orchestrator_payload.get("vid2vid_init_strength"),
        # Audio to add to final output (only used by last join)
        "audio_url": orchestrator_payload.get("audio_url"),
    }


def _check_existing_join_tasks(
    orchestrator_task_id_str: str,
    num_joins: int,
    dprint
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Check for existing child tasks (idempotency check).

    Handles both patterns:
    - Chain pattern: num_joins join_clips_segment tasks
    - Parallel pattern: num_joins join_clips_segment (transitions) + 1 join_final_stitch

    Returns:
        (None, None) if no existing tasks or should proceed with creation
        (success: bool, message: str) if should return early (complete/failed/in-progress)
    """
    dprint(f"[JOIN_CORE] Checking for existing child tasks")
    existing_child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
    existing_joins = existing_child_tasks.get('join_clips_segment', [])
    existing_final_stitch = existing_child_tasks.get('join_final_stitch', [])

    # Determine which pattern was used
    is_parallel_pattern = len(existing_final_stitch) > 0

    if not existing_joins and not existing_final_stitch:
        return None, None

    dprint(f"[JOIN_CORE] Found {len(existing_joins)} join tasks, {len(existing_final_stitch)} final stitch tasks")

    # Check completion status helper
    def is_complete(task):
        return (task.get('status', '') or '').lower() == 'complete'

    def is_terminal_failure(task):
        status = task.get('status', '').lower()
        return status in ('failed', 'cancelled', 'canceled', 'error')

    if is_parallel_pattern:
        # === PARALLEL PATTERN ===
        if len(existing_joins) < num_joins:
            return None, None

        all_tasks = existing_joins + existing_final_stitch
        any_failed = any(is_terminal_failure(t) for t in all_tasks)

        if any_failed:
            failed_tasks = [t for t in all_tasks if is_terminal_failure(t)]
            error_msg = f"{len(failed_tasks)} task(s) failed/cancelled"
            dprint(f"[JOIN_CORE] FAILED: {error_msg}")
            return False, f"[ORCHESTRATOR_FAILED] {error_msg}"

        if existing_final_stitch and is_complete(existing_final_stitch[0]):
            final_stitch = existing_final_stitch[0]
            final_output = final_stitch.get('output_location', 'Completed via idempotency')
            dprint(f"[JOIN_CORE] COMPLETE (parallel): Final stitch done, output: {final_output}")
            completion_data = json.dumps({"output_location": final_output, "thumbnail_url": ""})
            return True, f"[ORCHESTRATOR_COMPLETE]{completion_data}"

        trans_complete = sum(1 for j in existing_joins if is_complete(j))
        stitch_status = "complete" if existing_final_stitch and is_complete(existing_final_stitch[0]) else "pending"
        dprint(f"[JOIN_CORE] IDEMPOTENT (parallel): {trans_complete}/{num_joins} transitions, stitch: {stitch_status}")
        return True, f"[IDEMPOTENT] Parallel: {trans_complete}/{num_joins} transitions complete, stitch: {stitch_status}"

    else:
        # === CHAIN PATTERN (legacy) ===
        if len(existing_joins) < num_joins:
            return None, None

        dprint(f"[JOIN_CORE] All {num_joins} join tasks already exist (chain pattern)")

        all_joins_complete = all(is_complete(join) for join in existing_joins)
        any_join_failed = any(is_terminal_failure(join) for join in existing_joins)

        if any_join_failed:
            failed_joins = [j for j in existing_joins if is_terminal_failure(j)]
            error_msg = f"{len(failed_joins)} join task(s) failed/cancelled"
            dprint(f"[JOIN_CORE] FAILED: {error_msg}")
            return False, f"[ORCHESTRATOR_FAILED] {error_msg}"

        if all_joins_complete:
            def get_join_index(task):
                params = task.get('task_params', {})
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except (json.JSONDecodeError, ValueError):
                        return 0
                return params.get('join_index', 0)

            sorted_joins = sorted(existing_joins, key=get_join_index)
            final_join = sorted_joins[-1]
            final_output = final_join.get('output_location', 'Completed via idempotency')

            final_params = final_join.get('task_params', {})
            if isinstance(final_params, str):
                try:
                    final_params = json.loads(final_params)
                except (json.JSONDecodeError, ValueError):
                    final_params = {}

            final_thumbnail = final_params.get('thumbnail_url', '')

            dprint(f"[JOIN_CORE] COMPLETE: All joins finished, final output: {final_output}")
            completion_data = json.dumps({"output_location": final_output, "thumbnail_url": final_thumbnail})
            return True, f"[ORCHESTRATOR_COMPLETE]{completion_data}"

        complete_count = sum(1 for j in existing_joins if is_complete(j))
        dprint(f"[JOIN_CORE] IDEMPOTENT: {complete_count}/{num_joins} joins complete")
        return True, f"[IDEMPOTENT] Join tasks in progress: {complete_count}/{num_joins} complete"


def _create_join_chain_tasks(
    clip_list: List[dict],
    run_id: str,
    join_settings: dict,
    per_join_settings: List[dict],
    vlm_enhanced_prompts: List[Optional[str]],
    current_run_output_dir: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    orchestrator_payload: dict,
    parent_generation_id: str | None,
    dprint
) -> Tuple[bool, str]:
    """
    Core logic: Create chained join_clips_segment tasks (LEGACY - sequential pattern).

    DEPRECATED: Use _create_parallel_join_tasks for better quality (avoids re-encoding).

    Args:
        clip_list: List of clip dicts with 'url' and optional 'name' keys
        run_id: Unique run identifier
        join_settings: Base settings for all join tasks
        per_join_settings: Per-join overrides (list, one per join)
        vlm_enhanced_prompts: VLM-generated prompts (or None for each join)
        current_run_output_dir: Output directory for this run
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        orchestrator_payload: Full orchestrator payload for reference
        parent_generation_id: Parent generation ID for variant linking
        dprint: Debug print function

    Returns:
        (success: bool, message: str)
    """
    num_joins = len(clip_list) - 1

    if num_joins < 1:
        return False, "clip_list must contain at least 2 clips"

    dprint(f"[JOIN_CORE] Creating {num_joins} join tasks in dependency chain")

    previous_join_task_id = None
    joins_created = 0

    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]

        dprint(f"[JOIN_CORE] Creating join {idx}: {clip_start.get('name', 'clip')} + {clip_end.get('name', 'clip')}")

        # Merge global settings with per-join overrides
        task_join_settings = join_settings.copy()
        if idx < len(per_join_settings):
            task_join_settings.update(per_join_settings[idx])
            dprint(f"[JOIN_CORE] Applied per-join overrides for join {idx}")

        # Apply VLM-enhanced prompt if available (overrides base prompt)
        if idx < len(vlm_enhanced_prompts) and vlm_enhanced_prompts[idx] is not None:
            task_join_settings["prompt"] = vlm_enhanced_prompts[idx]
            dprint(f"[JOIN_CORE] Join {idx}: Using VLM-enhanced prompt")

        # Build join payload
        join_payload = {
            "orchestrator_task_id_ref": orchestrator_task_id_str,
            "orchestrator_run_id": run_id,
            "project_id": orchestrator_project_id,
            "join_index": idx,
            "is_first_join": (idx == 0),
            "is_last_join": (idx == num_joins - 1),
            "child_order": idx,
            "skip_generation": True,
            "starting_video_path": clip_start.get("url") if idx == 0 else None,
            "ending_video_path": clip_end.get("url"),
            **task_join_settings,
            "current_run_base_output_dir": str(current_run_output_dir.resolve()),
            "join_output_dir": str((current_run_output_dir / f"join_{idx}").resolve()),
            "full_orchestrator_payload": orchestrator_payload,
        }

        # === CANCELLATION CHECK ===
        cancel_msg = _check_orchestrator_cancelled(
            orchestrator_task_id_str,
            f"aborting join creation at index {idx} ({joins_created} joins already created)",
            dprint=dprint,
        )
        if cancel_msg:
            return False, cancel_msg

        dprint(f"[JOIN_CORE] Submitting join {idx} to database, depends_on={previous_join_task_id}")

        actual_db_row_id = db_ops.add_task_to_db(
            task_payload=join_payload,
            task_type_str="join_clips_segment",
            dependant_on=previous_join_task_id
        )

        dprint(f"[JOIN_CORE] Join {idx} created with DB ID: {actual_db_row_id}")

        previous_join_task_id = actual_db_row_id
        joins_created += 1

    # === Create join_final_stitch that depends on the last join ===
    context_frame_count = join_settings.get("context_frame_count", 8)

    final_stitch_payload = {
        "orchestrator_task_id_ref": orchestrator_task_id_str,
        "orchestrator_run_id": run_id,
        "project_id": orchestrator_project_id,
        "parent_generation_id": parent_generation_id,
        "clip_list": clip_list,
        "transition_task_ids": [previous_join_task_id],
        "chain_mode": True,
        "blend_frames": min(context_frame_count, 15),
        "fps": join_settings.get("fps") or orchestrator_payload.get("fps", 16),
        "audio_url": orchestrator_payload.get("audio_url"),
        "current_run_base_output_dir": str(current_run_output_dir.resolve()),
    }

    cancel_msg = _check_orchestrator_cancelled(
        orchestrator_task_id_str,
        f"aborting before final stitch ({joins_created} joins cancelled)",
        dprint=dprint,
    )
    if cancel_msg:
        return False, cancel_msg

    final_stitch_task_id = db_ops.add_task_to_db(
        task_payload=final_stitch_payload,
        task_type_str="join_final_stitch",
        dependant_on=previous_join_task_id
    )

    dprint(f"[JOIN_CORE] Final stitch task created with DB ID: {final_stitch_task_id}")
    dprint(f"[JOIN_CORE] Complete: {joins_created} chain joins + 1 final stitch = {joins_created + 1} total tasks")

    return True, f"Successfully enqueued {joins_created} chain joins + 1 final stitch for run {run_id}"


def _create_parallel_join_tasks(
    clip_list: List[dict],
    run_id: str,
    join_settings: dict,
    per_join_settings: List[dict],
    vlm_enhanced_prompts: List[Optional[str]],
    current_run_output_dir: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    orchestrator_payload: dict,
    parent_generation_id: str | None,
    dprint
) -> Tuple[bool, str]:
    """
    Create parallel join tasks with a final stitch (NEW - parallel pattern).

    This pattern avoids quality loss from re-encoding:
    1. Create N-1 transition tasks in parallel (no dependencies between them)
    2. Create a single join_final_stitch task that depends on ALL transition tasks

    Args:
        clip_list: List of clip dicts with 'url' and optional 'name' keys
        run_id: Unique run identifier
        join_settings: Base settings for all join tasks
        per_join_settings: Per-join overrides (list, one per join)
        vlm_enhanced_prompts: VLM-generated prompts (or None for each join)
        current_run_output_dir: Output directory for this run
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        orchestrator_payload: Full orchestrator payload for reference
        parent_generation_id: Parent generation ID for variant linking
        dprint: Debug print function

    Returns:
        (success: bool, message: str)
    """
    num_joins = len(clip_list) - 1

    if num_joins < 1:
        return False, "clip_list must contain at least 2 clips"

    dprint(f"[JOIN_PARALLEL] Creating {num_joins} parallel transition tasks + 1 final stitch")

    transition_task_ids = []

    # === Phase 1: Create transition tasks in parallel (no dependencies) ===
    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]

        dprint(f"[JOIN_PARALLEL] Creating transition {idx}: {clip_start.get('name', 'clip')} \u2192 {clip_end.get('name', 'clip')}")

        task_join_settings = join_settings.copy()
        if idx < len(per_join_settings):
            task_join_settings.update(per_join_settings[idx])
            dprint(f"[JOIN_PARALLEL] Applied per-join overrides for transition {idx}")

        if idx < len(vlm_enhanced_prompts) and vlm_enhanced_prompts[idx] is not None:
            task_join_settings["prompt"] = vlm_enhanced_prompts[idx]
            dprint(f"[JOIN_PARALLEL] Transition {idx}: Using VLM-enhanced prompt")

        transition_payload = {
            "orchestrator_task_id_ref": orchestrator_task_id_str,
            "orchestrator_run_id": run_id,
            "project_id": orchestrator_project_id,
            "join_index": idx,
            "transition_index": idx,
            "is_first_join": False,
            "is_last_join": False,
            "child_order": idx,
            "skip_generation": True,
            "starting_video_path": clip_start.get("url"),
            "ending_video_path": clip_end.get("url"),
            "transition_only": True,
            **task_join_settings,
            "current_run_base_output_dir": str(current_run_output_dir.resolve()),
            "join_output_dir": str((current_run_output_dir / f"transition_{idx}").resolve()),
            "full_orchestrator_payload": orchestrator_payload,
        }

        cancel_msg = _check_orchestrator_cancelled(
            orchestrator_task_id_str,
            f"aborting transition creation at index {idx} ({len(transition_task_ids)} transitions already created)",
            dprint=dprint,
        )
        if cancel_msg:
            return False, cancel_msg

        dprint(f"[JOIN_PARALLEL] Submitting transition {idx} to database (no dependency)")

        trans_task_id = db_ops.add_task_to_db(
            task_payload=transition_payload,
            task_type_str="join_clips_segment",
            dependant_on=None
        )

        dprint(f"[JOIN_PARALLEL] Transition {idx} created with DB ID: {trans_task_id}")
        transition_task_ids.append(trans_task_id)

    # === Phase 2: Create final stitch task that depends on ALL transitions ===
    dprint(f"[JOIN_PARALLEL] Creating final stitch task, depends on {len(transition_task_ids)} transitions")

    context_frame_count = join_settings.get("context_frame_count", 8)

    final_stitch_payload = {
        "orchestrator_task_id_ref": orchestrator_task_id_str,
        "orchestrator_run_id": run_id,
        "project_id": orchestrator_project_id,
        "parent_generation_id": parent_generation_id,
        "clip_list": clip_list,
        "transition_task_ids": transition_task_ids,
        "blend_frames": min(context_frame_count, 15),
        "fps": join_settings.get("fps") or orchestrator_payload.get("fps", 16),
        "audio_url": orchestrator_payload.get("audio_url"),
        "current_run_base_output_dir": str(current_run_output_dir.resolve()),
    }

    cancel_msg = _check_orchestrator_cancelled(
        orchestrator_task_id_str,
        f"aborting before final stitch ({len(transition_task_ids)} transitions cancelled)",
        dprint=dprint,
    )
    if cancel_msg:
        return False, cancel_msg

    final_stitch_task_id = db_ops.add_task_to_db(
        task_payload=final_stitch_payload,
        task_type_str="join_final_stitch",
        dependant_on=transition_task_ids
    )

    dprint(f"[JOIN_PARALLEL] Final stitch task created with DB ID: {final_stitch_task_id}")
    dprint(f"[JOIN_PARALLEL] Complete: {num_joins} transitions + 1 final stitch = {num_joins + 1} total tasks")

    return True, f"Successfully enqueued {num_joins} parallel transitions + 1 final stitch for run {run_id}"
