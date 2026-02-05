"""
Join Clips Orchestrator - Sequentially join multiple video clips

This orchestrator takes a list of video clips and creates a chain of join_clips_child
tasks to progressively build them into a single seamless video.

Pattern:
    Input: [clip_A, clip_B, clip_C, clip_D]

    Creates:
        join_0: clip_A + clip_B → AB.mp4 (no dependency)
        join_1: AB.mp4 + clip_C → ABC.mp4 (depends on join_0)
        join_2: ABC.mp4 + clip_D → ABCD.mp4 (depends on join_1)

    Each join task fetches the output of its predecessor via get_predecessor_output_via_edge_function()

Shared Core Logic:
    The _create_join_chain_tasks() function is the shared core that creates the
    dependency chain of join tasks. It is used by:
    - join_clips_orchestrator: Takes a clip_list directly
    - edit_video_orchestrator: Preprocesses portions_to_regenerate into keeper clips first
"""

import traceback
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
import cv2

from .. import db_operations as db_ops
from ..common_utils import download_video_if_url, get_video_frame_count_and_fps, upload_intermediate_file_to_storage
from ..video_utils import extract_frames_from_video, reverse_video


def _get_video_resolution(video_path: str | Path, dprint=print) -> Tuple[int, int] | None:
    """
    Get video resolution (width, height) using ffprobe.

    Returns:
        (width, height) tuple, or None if detection fails
    """
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            dprint(f"[GET_RESOLUTION] ffprobe failed: {result.stderr}")
            return None
        width_str, height_str = result.stdout.strip().split(',')
        return (int(width_str), int(height_str))
    except Exception as e:
        dprint(f"[GET_RESOLUTION] Error detecting resolution: {e}")
        return None


def _extract_boundary_frames_for_vlm(
    clip_list: List[dict],
    temp_dir: Path,
    orchestrator_task_id: str,
    dprint
) -> List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Extract boundary frames from clips for VLM prompt generation.

    For each join (clip[i] → clip[i+1]), extracts 4 frames:
    - First frame from clip[i] (scene context)
    - Last frame from clip[i] (boundary)
    - First frame from clip[i+1] (boundary)
    - Last frame from clip[i+1] (scene context)

    Note: We use simple first/last frames rather than trying to calculate exact
    transition boundaries. The VLM just needs approximate visual context to
    generate a good prompt - frame-level precision doesn't matter.

    Args:
        clip_list: List of clip dicts with 'url' keys
        temp_dir: Directory to save temporary frame images
        orchestrator_task_id: Task ID for logging
        dprint: Debug print function

    Returns:
        List of (start_first, start_boundary, end_boundary, end_last) tuples for each join
    """
    image_quads = []
    num_joins = len(clip_list) - 1
    
    # Cache downloaded videos and their extracted frames to avoid re-downloading
    # url -> (frames_list, first_path, last_path)
    clip_data_cache = {}
    
    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]
        
        start_url = clip_start.get("url")
        end_url = clip_end.get("url")
        
        if not start_url or not end_url:
            dprint(f"[VLM_EXTRACT] Join {idx}: Missing URL, skipping")
            image_quads.append((None, None, None, None))
            continue
            
        try:
            # === Extract frames from clip_start ===
            if start_url in clip_data_cache:
                start_frames, start_first_path, start_last_path = clip_data_cache[start_url]
            else:
                dprint(f"[VLM_EXTRACT] Join {idx}: Downloading clip_start: {start_url[:80]}...")
                local_start_path = download_video_if_url(
                    start_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx}_start"
                )
                
                # Extract all frames
                start_frames = extract_frames_from_video(local_start_path, dprint_func=dprint)
                if not start_frames:
                    dprint(f"[VLM_EXTRACT] Join {idx}: Failed to extract frames from start clip")
                    image_quads.append((None, None, None, None))
                    continue
                    
                # Save first and absolute last frames (for context)
                start_first_path = temp_dir / f"vlm_clip{idx}_first.jpg"
                start_last_path = temp_dir / f"vlm_clip{idx}_last.jpg"
                cv2.imwrite(str(start_first_path), cv2.cvtColor(start_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(start_last_path), cv2.cvtColor(start_frames[-1], cv2.COLOR_RGB2BGR))
                
                clip_data_cache[start_url] = (start_frames, str(start_first_path), str(start_last_path))
                dprint(f"[VLM_EXTRACT] Join {idx}: Extracted {len(start_frames)} frames from start clip")
            
            # Boundary is simply the last frame of clip_start
            start_boundary_idx = len(start_frames) - 1
            
            # Save boundary frame
            start_boundary_path = temp_dir / f"vlm_clip{idx}_boundary.jpg"
            cv2.imwrite(str(start_boundary_path), cv2.cvtColor(start_frames[start_boundary_idx], cv2.COLOR_RGB2BGR))
            
            # === Extract frames from clip_end ===
            if end_url in clip_data_cache:
                end_frames, end_first_path, end_last_path = clip_data_cache[end_url]
            else:
                dprint(f"[VLM_EXTRACT] Join {idx}: Downloading clip_end: {end_url[:80]}...")
                local_end_path = download_video_if_url(
                    end_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx+1}_end"
                )
                
                # Extract all frames
                end_frames = extract_frames_from_video(local_end_path, dprint_func=dprint)
                if not end_frames:
                    dprint(f"[VLM_EXTRACT] Join {idx}: Failed to extract frames from end clip")
                    image_quads.append((None, None, None, None))
                    continue
                    
                # Save first and absolute last frames (for context)
                end_first_path = temp_dir / f"vlm_clip{idx+1}_first.jpg"
                end_last_path = temp_dir / f"vlm_clip{idx+1}_last.jpg"
                cv2.imwrite(str(end_first_path), cv2.cvtColor(end_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(end_last_path), cv2.cvtColor(end_frames[-1], cv2.COLOR_RGB2BGR))
                
                clip_data_cache[end_url] = (end_frames, str(end_first_path), str(end_last_path))
                dprint(f"[VLM_EXTRACT] Join {idx}: Extracted {len(end_frames)} frames from end clip")
            
            # Boundary is simply the first frame of clip_end
            end_boundary_idx = 0
            
            # Save boundary frame
            end_boundary_path = temp_dir / f"vlm_clip{idx+1}_boundary.jpg"
            cv2.imwrite(str(end_boundary_path), cv2.cvtColor(end_frames[end_boundary_idx], cv2.COLOR_RGB2BGR))
            
            # Return quad: (start_first, start_boundary, end_boundary, end_last)
            image_quads.append((
                str(clip_data_cache[start_url][1]),  # start first
                str(start_boundary_path),             # start boundary
                str(end_boundary_path),               # end boundary
                str(clip_data_cache[end_url][2])      # end last
            ))
            dprint(f"[VLM_EXTRACT] Join {idx}: 4 frames ready (first, boundary, boundary, last)")
            
        except Exception as e:
            dprint(f"[VLM_EXTRACT] Join {idx}: ERROR extracting frames: {e}")
            image_quads.append((None, None, None, None))
            
    return image_quads


def _generate_join_transition_prompt(
    start_first_path: str,
    start_boundary_path: str,
    end_boundary_path: str,
    end_last_path: str,
    base_prompt: str,
    extender,
    dprint
) -> str:
    """
    Generate a single transition prompt for join_clips.

    Uses 4 images side-by-side for full context:
    - First frame of clip A (scene context)
    - Last frame of clip A (boundary)
    - First frame of clip B (boundary)
    - Last frame of clip B (scene context)

    Args:
        start_first_path: Path to first frame of starting clip
        start_boundary_path: Path to last frame of starting clip
        end_boundary_path: Path to first frame of ending clip
        end_last_path: Path to last frame of ending clip
        base_prompt: Base prompt for context
        extender: QwenPromptExpander instance
        dprint: Debug print function

    Returns:
        Generated prompt
    """
    from PIL import Image
    
    # Load all 4 images
    start_first = Image.open(start_first_path).convert("RGB")
    start_boundary = Image.open(start_boundary_path).convert("RGB")
    end_boundary = Image.open(end_boundary_path).convert("RGB")
    end_last = Image.open(end_last_path).convert("RGB")
    
    # Combine all 4 images side by side
    images = [start_first, start_boundary, end_boundary, end_last]
    combined_width = sum(img.width for img in images)
    combined_height = max(img.height for img in images)
    combined_img = Image.new('RGB', (combined_width, combined_height))
    
    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a video sequence"
    
    query = f"""Look at these 4 frames from a video, left to right in time order.

The MIDDLE TWO frames show where we need to generate a transition.

Context from user: {base_prompt_text}

Write a short video prompt describing what happens between the middle frames. Include:
1. The motion or action
2. The visual style
3. Key details in the scene

Keep it under 50 words total. Just write the prompt, nothing else.

Example: "A cat jumps from table to chair. Realistic home video style. Sunny kitchen with wooden furniture."

Example: "Camera pans right across city skyline. Cinematic drone footage at sunset. Skyscrapers with orange light reflections."

Your prompt:"""

    system_prompt = "Write a short video description prompt. Under 50 words. No explanations, just the prompt."
    
    result = extender.extend_with_img(
        prompt=query,
        system_prompt=system_prompt,
        image=combined_img
    )
    
    return result.prompt.strip()


def _generate_vlm_prompts_for_joins(
    image_quads: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]],
    base_prompt: str,
    vlm_device: str,
    dprint
) -> List[Optional[str]]:
    """
    Generate VLM-enhanced prompts for ALL joins using 4-image quads.

    Args:
        image_quads: List of (start_first, start_boundary, end_boundary, end_last) tuples
        base_prompt: Base prompt to use as VLM context
        vlm_device: Device for VLM inference ('cuda' or 'cpu')
        dprint: Debug print function

    Returns:
        List of enhanced prompts (None for joins with missing image quads)
    """
    import sys
    from pathlib import Path
    import torch
    
    num_joins = len(image_quads)
    result = [None] * num_joins
    
    # Filter out invalid quads, track their indices
    valid_quads = []
    valid_indices = []
    
    for idx in range(num_joins):
        quad = image_quads[idx]
        # All 4 images must be present
        if all(path is not None for path in quad):
            valid_quads.append(quad)
            valid_indices.append(idx)
        else:
            dprint(f"[VLM_PROMPTS] Join {idx}: Skipping - missing image(s) in quad")
    
    if not valid_quads:
        dprint(f"[VLM_PROMPTS] No valid image quads for VLM processing")
        return result
    
    dprint(f"[VLM_PROMPTS] Processing {len(valid_quads)}/{num_joins} joins with VLM (4 images each)")
    dprint(f"[VLM_PROMPTS] Base prompt context: '{base_prompt[:80]}...'" if base_prompt else "[VLM_PROMPTS] No base prompt (VLM will infer from frames)")
    
    try:
        # Add Wan2GP to path for imports
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))
        
        from Wan2GP.shared.utils.prompt_extend import QwenPromptExpander
        from ..vlm_utils import download_qwen_vlm_if_needed
        
        # Log memory before loading (same as vlm_utils.py)
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_PROMPTS] GPU memory BEFORE VLM load: {gpu_mem_before:.2f} GB")
        
        # Initialize VLM ONCE for all quads (batch efficiency)
        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"
        dprint(f"[VLM_PROMPTS] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)
        
        dprint(f"[VLM_PROMPTS] Initializing Qwen2.5-VL-7B-Instruct...")
        extender = QwenPromptExpander(
            model_name=str(local_model_path),
            device=vlm_device,
            is_vl=True
        )
        dprint(f"[VLM_PROMPTS] Model loaded (initially on CPU, moves to {vlm_device} for inference)")
        
        # Process each quad
        for i, quad in enumerate(valid_quads):
            idx = valid_indices[i]
            start_first, start_boundary, end_boundary, end_last = quad
            try:
                dprint(f"[VLM_PROMPTS] Processing join {idx} ({i+1}/{len(valid_quads)})...")
                
                enhanced = _generate_join_transition_prompt(
                    start_first_path=start_first,
                    start_boundary_path=start_boundary,
                    end_boundary_path=end_boundary,
                    end_last_path=end_last,
                    base_prompt=base_prompt,
                    extender=extender,
                    dprint=dprint
                )
                
                result[idx] = enhanced
                dprint(f"[VLM_PROMPTS] Join {idx}: {enhanced[:100]}...")
                
            except Exception as e:
                dprint(f"[VLM_PROMPTS] Join {idx}: ERROR - {e}")
                # Continue with other quads
        
        # Cleanup VLM (same pattern as vlm_utils.py)
        # Log memory BEFORE cleanup
        if torch.cuda.is_available():
            gpu_mem_before_cleanup = torch.cuda.memory_allocated() / 1024**3
            dprint(f"[VLM_CLEANUP] GPU memory BEFORE cleanup: {gpu_mem_before_cleanup:.2f} GB")
        
        # Explicitly delete model and processor to free all memory
        dprint(f"[VLM_CLEANUP] Cleaning up VLM model and processor...")
        try:
            del extender.model
            del extender.processor
            del extender
            dprint(f"[VLM_CLEANUP] ✅ Successfully deleted VLM objects")
        except Exception as e:
            dprint(f"[VLM_CLEANUP] ⚠️  Error during deletion: {e}")
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        dprint(f"[VLM_CLEANUP] Garbage collected {collected} objects")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            gpu_freed = gpu_mem_before_cleanup - gpu_mem_after
            dprint(f"[VLM_CLEANUP] GPU memory AFTER cleanup: {gpu_mem_after:.2f} GB")
            dprint(f"[VLM_CLEANUP] GPU memory freed: {gpu_freed:.2f} GB")
        
        dprint(f"[VLM_CLEANUP] ✅ VLM cleanup complete")
        return result
        
    except Exception as e:
        dprint(f"[VLM_PROMPTS] ERROR in VLM processing: {e}")
        traceback.print_exc()
        return result


# =============================================================================
# SHARED CORE LOGIC - Used by both join_clips_orchestrator and edit_video_orchestrator
# =============================================================================

def calculate_min_clip_frames(gap_frame_count: int, context_frame_count: int, replace_mode: bool) -> int:
    """
    Calculate the minimum number of frames a clip must have to safely join.

    In REPLACE mode, we need:
        gap_frame_count + 2 × context_frame_count ≤ min_clip_frames

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
        # In REPLACE mode, we consume:
        # - gap_from_clip1 frames from end of clip1 (gap // 2)
        # - gap_from_clip2 frames from start of clip2 (gap - gap//2)
        # - context_frame_count frames as additional context from each side
        # Total consumed from a middle clip (both ends): gap + 2*context
        return gap_frame_count + 2 * context_frame_count
    else:
        # In INSERT mode, we only use context frames (no replacement)
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
    dprint(f"[VALIDATE_CLIPS]   (gap={gap_frame_count} + 2×context={context_frame_count}, replace_mode={replace_mode})")

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
            # Boundary clips need: context + gap_from_that_side
            if replace_mode:
                gap_from_side = gap_frame_count // 2 if is_first else (gap_frame_count - gap_frame_count // 2)
                required = context_frame_count + gap_from_side
            else:
                required = context_frame_count
        else:
            # Middle clips need full minimum
            required = min_frames

        if frames < required:
            violations.append({
                "idx": idx,
                "frames": frames,
                "required": required,
                "shortfall": required - frames
            })

    if violations:
        # Calculate what proportional reduction will produce
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
            + f"\n  Will reduce to approximately: gap≈{reduced_gap}, context≈{reduced_context} ({ratio:.0%} of original)"
            + f"\n  Transitions will be shorter but still generated successfully."
        )
        dprint(f"[VALIDATE_CLIPS] {warning_msg}")

        # Return True (valid) with warning - proportional reduction will handle it
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
    # Note: If use_input_video_resolution=True, the orchestrator will detect the actual
    # resolution from the first clip and override join_settings["resolution"] with it.
    # We initially set resolution=None here to avoid passing through a potentially wrong
    # client-provided value; the orchestrator will set the correct value after detection.
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
    import json

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
        # Need all transitions complete + final stitch complete
        if len(existing_joins) < num_joins:
            return None, None

        all_tasks = existing_joins + existing_final_stitch
        any_failed = any(is_terminal_failure(t) for t in all_tasks)

        if any_failed:
            failed_tasks = [t for t in all_tasks if is_terminal_failure(t)]
            error_msg = f"{len(failed_tasks)} task(s) failed/cancelled"
            dprint(f"[JOIN_CORE] FAILED: {error_msg}")
            return False, f"[ORCHESTRATOR_FAILED] {error_msg}"

        # Check if final stitch is complete
        if existing_final_stitch and is_complete(existing_final_stitch[0]):
            final_stitch = existing_final_stitch[0]
            final_output = final_stitch.get('output_location', 'Completed via idempotency')
            dprint(f"[JOIN_CORE] COMPLETE (parallel): Final stitch done, output: {final_output}")
            completion_data = json.dumps({"output_location": final_output, "thumbnail_url": ""})
            return True, f"[ORCHESTRATOR_COMPLETE]{completion_data}"

        # Still in progress
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
    dprint
) -> Tuple[bool, str]:
    """
    Core logic: Create chained join_clips_segment tasks (LEGACY - sequential pattern).

    DEPRECATED: Use _create_parallel_join_tasks for better quality (avoids re-encoding).

    This is the shared core function used by both:
    - join_clips_orchestrator: Provides clip_list directly
    - edit_video_orchestrator: Preprocesses source video into keeper clips first

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
            # Standardized field for completion handler
            "child_order": idx,
            # Skip generation creation - final_stitch handles the parent variant
            "skip_generation": True,

            # First join has explicit starting path, rest fetch from dependency
            "starting_video_path": clip_start.get("url") if idx == 0 else None,
            "ending_video_path": clip_end.get("url"),

            # Join settings
            **task_join_settings,

            # Output configuration
            "current_run_base_output_dir": str(current_run_output_dir.resolve()),
            "join_output_dir": str((current_run_output_dir / f"join_{idx}").resolve()),

            # Reference to full orchestrator payload
            "full_orchestrator_payload": orchestrator_payload,
        }

        dprint(f"[JOIN_CORE] Submitting join {idx} to database, depends_on={previous_join_task_id}")

        # Create task with dependency chain
        actual_db_row_id = db_ops.add_task_to_db(
            task_payload=join_payload,
            task_type_str="join_clips_segment",
            dependant_on=previous_join_task_id
        )

        dprint(f"[JOIN_CORE] Join {idx} created with DB ID: {actual_db_row_id}")

        # Update for next iteration
        previous_join_task_id = actual_db_row_id
        joins_created += 1

    return True, f"Successfully enqueued {joins_created} join tasks for run {run_id}"


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
    dprint
) -> Tuple[bool, str]:
    """
    Create parallel join tasks with a final stitch (NEW - parallel pattern).

    This pattern avoids quality loss from re-encoding:
    1. Create N-1 transition tasks in parallel (no dependencies between them)
       - Each task only generates the transition video (transition_only=True)
    2. Create a single join_final_stitch task that depends on ALL transition tasks
       - Stitches all original clips + transitions in one pass

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

        dprint(f"[JOIN_PARALLEL] Creating transition {idx}: {clip_start.get('name', 'clip')} → {clip_end.get('name', 'clip')}")

        # Merge global settings with per-join overrides
        task_join_settings = join_settings.copy()
        if idx < len(per_join_settings):
            task_join_settings.update(per_join_settings[idx])
            dprint(f"[JOIN_PARALLEL] Applied per-join overrides for transition {idx}")

        # Apply VLM-enhanced prompt if available (overrides base prompt)
        if idx < len(vlm_enhanced_prompts) and vlm_enhanced_prompts[idx] is not None:
            task_join_settings["prompt"] = vlm_enhanced_prompts[idx]
            dprint(f"[JOIN_PARALLEL] Transition {idx}: Using VLM-enhanced prompt")

        # Build transition payload - each has explicit clip paths, no dependency
        transition_payload = {
            "orchestrator_task_id_ref": orchestrator_task_id_str,
            "orchestrator_run_id": run_id,
            "project_id": orchestrator_project_id,
            "join_index": idx,
            "transition_index": idx,  # Used for ordering in final stitch
            "is_first_join": False,   # Not relevant for transition_only
            "is_last_join": False,    # Not relevant for transition_only
            # Standardized field for completion handler
            "child_order": idx,
            # Skip generation creation - final_stitch handles the parent variant
            "skip_generation": True,

            # Both clips are explicit (no dependency fetch)
            "starting_video_path": clip_start.get("url"),
            "ending_video_path": clip_end.get("url"),

            # CRITICAL: Enable transition_only mode
            "transition_only": True,

            # Join settings
            **task_join_settings,

            # Output configuration
            "current_run_base_output_dir": str(current_run_output_dir.resolve()),
            "join_output_dir": str((current_run_output_dir / f"transition_{idx}").resolve()),

            # Reference to full orchestrator payload
            "full_orchestrator_payload": orchestrator_payload,
        }

        dprint(f"[JOIN_PARALLEL] Submitting transition {idx} to database (no dependency)")

        # Create task WITHOUT dependency - all transitions run in parallel
        trans_task_id = db_ops.add_task_to_db(
            task_payload=transition_payload,
            task_type_str="join_clips_segment",
            dependant_on=None  # No dependency - parallel execution
        )

        dprint(f"[JOIN_PARALLEL] Transition {idx} created with DB ID: {trans_task_id}")
        transition_task_ids.append(trans_task_id)

    # === Phase 2: Create final stitch task that depends on ALL transitions ===
    dprint(f"[JOIN_PARALLEL] Creating final stitch task, depends on {len(transition_task_ids)} transitions")

    # Get settings for the final stitch payload
    # NOTE: gap_from_clip1/gap_from_clip2 are NOT passed here because:
    # 1. The orchestrator would calculate from raw gap_frame_count (before 4n+1 quantization)
    # 2. The segment tasks calculate from quantized gap_for_guide (after 4n+1 quantization)
    # 3. This mismatch caused a 1-frame alignment bug
    # 4. The final_stitch now reads these values from each transition's output_location (ground truth)
    context_frame_count = join_settings.get("context_frame_count", 8)

    final_stitch_payload = {
        "orchestrator_task_id_ref": orchestrator_task_id_str,
        "orchestrator_run_id": run_id,
        "project_id": orchestrator_project_id,

        # Original clips (for trimming and stitching)
        "clip_list": clip_list,

        # Transition task IDs to fetch outputs from
        "transition_task_ids": transition_task_ids,

        # Blending parameters (trim values come from per-transition output_location)
        "blend_frames": min(context_frame_count, 15),  # Safe blend limit
        "fps": join_settings.get("fps") or orchestrator_payload.get("fps", 16),

        # Audio (if provided)
        "audio_url": orchestrator_payload.get("audio_url"),

        # Output configuration
        "current_run_base_output_dir": str(current_run_output_dir.resolve()),
    }

    # Create final stitch task with multi-dependency (all transitions must complete)
    final_stitch_task_id = db_ops.add_task_to_db(
        task_payload=final_stitch_payload,
        task_type_str="join_final_stitch",
        dependant_on=transition_task_ids  # Multi-dependency: list of task IDs
    )

    dprint(f"[JOIN_PARALLEL] Final stitch task created with DB ID: {final_stitch_task_id}")
    dprint(f"[JOIN_PARALLEL] Complete: {num_joins} transitions + 1 final stitch = {num_joins + 1} total tasks")

    return True, f"Successfully enqueued {num_joins} parallel transitions + 1 final stitch for run {run_id}"


def _handle_join_clips_orchestrator_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    *,
    dprint
) -> Tuple[bool, str]:
    """
    Handle join_clips_orchestrator task - creates chained join_clips_child tasks.

    Args:
        task_params_from_db: Task parameters containing orchestrator_details
        main_output_dir_base: Base output directory
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        dprint: Debug print function

    Returns:
        (success: bool, message: str)
    """
    dprint(f"[JOIN_ORCHESTRATOR] Starting orchestrator task {orchestrator_task_id_str}")

    try:
        # === 1. PARSE ORCHESTRATOR PAYLOAD ===
        if 'orchestrator_details' not in task_params_from_db:
            dprint("[JOIN_ORCHESTRATOR] ERROR: orchestrator_details missing")
            return False, "orchestrator_details missing"

        orchestrator_payload = task_params_from_db['orchestrator_details']
        dprint(f"[JOIN_ORCHESTRATOR] Orchestrator payload keys: {list(orchestrator_payload.keys())}")

        # Extract required fields
        clip_list = orchestrator_payload.get("clip_list", [])
        run_id = orchestrator_payload.get("run_id")
        loop_first_clip = orchestrator_payload.get("loop_first_clip", False)

        # === DYNAMIC CLIP_LIST FROM SEGMENT TASKS ===
        # If segment_task_ids is provided (from travel_orchestrator with stitch_config),
        # fetch the output URLs from those completed tasks to build clip_list
        segment_task_ids = orchestrator_payload.get("segment_task_ids", [])
        if segment_task_ids and not clip_list:
            dprint(f"[JOIN_ORCHESTRATOR] Building clip_list from {len(segment_task_ids)} segment task outputs")

            built_clip_list = []
            for i, task_id in enumerate(segment_task_ids):
                # Fetch output URL from completed segment task
                output_url = db_ops.get_task_output_location_from_db(task_id)
                if not output_url:
                    return False, f"Segment task {task_id} has no output_location (may not be complete)"

                # Handle JSON output (some tasks return JSON with url inside)
                if output_url.startswith('{'):
                    try:
                        output_data = json.loads(output_url)
                        output_url = output_data.get("output_location") or output_data.get("url") or output_url
                    except json.JSONDecodeError:
                        pass  # Use as-is

                built_clip_list.append({
                    "url": output_url,
                    "name": f"Segment {i + 1}"
                })
                dprint(f"[JOIN_ORCHESTRATOR]   Segment {i}: {output_url[:80]}...")

            clip_list = built_clip_list
            dprint(f"[JOIN_ORCHESTRATOR] Built clip_list with {len(clip_list)} clips from segment outputs")

        if not run_id:
            return False, "run_id is required"

        # Handle loop_first_clip: reverse the first clip and use it as the second clip
        # This creates a "boomerang" effect where the clip plays forward then backward
        if loop_first_clip:
            dprint(f"[JOIN_ORCHESTRATOR] loop_first_clip=True - will reverse first clip to create looping effect")
            
            if not clip_list or len(clip_list) < 1:
                return False, "clip_list must contain at least 1 clip when loop_first_clip=True"
            
            # Create a temp directory for the reversed video
            loop_temp_dir = Path(main_output_dir_base) / f"join_clips_run_{run_id}" / "loop_temp"
            loop_temp_dir.mkdir(parents=True, exist_ok=True)
            
            first_clip = clip_list[0]
            first_clip_url = first_clip.get("url")
            first_clip_name = first_clip.get("name", "clip_0")
            
            if not first_clip_url:
                return False, "First clip in clip_list is missing 'url' field"
            
            # Download the first clip if it's a URL
            dprint(f"[JOIN_ORCHESTRATOR] Downloading first clip for reversal: {first_clip_url[:80]}...")
            local_first_clip_path = download_video_if_url(
                first_clip_url,
                download_target_dir=loop_temp_dir,
                task_id_for_logging=orchestrator_task_id_str,
                descriptive_name="first_clip_for_loop"
            )
            
            if not local_first_clip_path or not Path(local_first_clip_path).exists():
                return False, f"Failed to download first clip: {first_clip_url}"
            
            # Reverse the video
            reversed_clip_path = loop_temp_dir / f"{first_clip_name}_reversed.mp4"
            dprint(f"[JOIN_ORCHESTRATOR] Reversing first clip...")
            
            reversed_path = reverse_video(
                local_first_clip_path,
                reversed_clip_path,
                dprint=dprint
            )
            
            if not reversed_path or not reversed_path.exists():
                return False, f"Failed to reverse first clip: {local_first_clip_path}"
            
            dprint(f"[JOIN_ORCHESTRATOR] ✅ Created reversed clip locally: {reversed_path}")
            
            # Upload reversed clip to storage for cross-worker access
            reversed_filename = f"{first_clip_name}_reversed.mp4"
            reversed_url = upload_intermediate_file_to_storage(
                local_file_path=reversed_path,
                task_id=orchestrator_task_id_str,
                filename=reversed_filename,
                dprint=dprint
            )
            
            if not reversed_url:
                # Fallback to local path if upload fails (works for single-worker setups)
                dprint(f"[JOIN_ORCHESTRATOR] ⚠️  Upload failed, using local path (may fail on multi-worker)")
                reversed_url = str(reversed_path.resolve())
            else:
                dprint(f"[JOIN_ORCHESTRATOR] ✅ Reversed clip uploaded: {reversed_url}")
            
            # Build new clip_list: [original_first_clip, reversed_first_clip]
            # The reversed clip becomes the "second" clip
            reversed_clip_dict = {
                "url": reversed_url,
                "name": f"{first_clip_name}_reversed"
            }
            
            # Override clip_list with the two clips
            clip_list = [first_clip, reversed_clip_dict]
            dprint(f"[JOIN_ORCHESTRATOR] clip_list overridden: [{first_clip_name}, {reversed_clip_dict['name']}]")

        if not clip_list or len(clip_list) < 2:
            return False, "clip_list must contain at least 2 clips"

        num_joins = len(clip_list) - 1
        dprint(f"[JOIN_ORCHESTRATOR] Processing {len(clip_list)} clips = {num_joins} join tasks")

        # === EARLY IDEMPOTENCY CHECK (before expensive VLM work) ===
        # Check if children already exist - if so, skip all expensive processing
        idempotency_result, idempotency_message = _check_existing_join_tasks(
            orchestrator_task_id_str, num_joins, dprint
        )
        if idempotency_result is not None:
            return idempotency_result, idempotency_message

        # Extract join settings using shared helper
        join_settings = _extract_join_settings_from_payload(orchestrator_payload)
        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))

        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"join_clips_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"[JOIN_ORCHESTRATOR] Run output directory: {current_run_output_dir}")

        # === VALIDATE CLIP FRAME COUNTS (optional, enabled by default) ===
        skip_validation = orchestrator_payload.get("skip_frame_validation", False)
        if not skip_validation:
            validation_temp_dir = current_run_output_dir / "validation_temp"
            validation_temp_dir.mkdir(parents=True, exist_ok=True)

            is_valid, validation_message, frame_counts = validate_clip_frames_for_join(
                clip_list=clip_list,
                gap_frame_count=join_settings.get("gap_frame_count", 53),
                context_frame_count=join_settings.get("context_frame_count", 8),
                replace_mode=join_settings.get("replace_mode", False),
                temp_dir=validation_temp_dir,
                orchestrator_task_id=orchestrator_task_id_str,
                dprint=dprint
            )

            if not is_valid:
                # Actual failure (download error, missing URL, etc.)
                dprint(f"[JOIN_ORCHESTRATOR] VALIDATION FAILED: {validation_message}")
                return False, f"Clip frame validation failed: {validation_message}"

            if validation_message:
                # Warning about proportional reduction (will proceed anyway)
                dprint(f"[JOIN_ORCHESTRATOR] {validation_message}")

            dprint(f"[JOIN_ORCHESTRATOR] Clip frame validation complete, frame counts: {frame_counts}")
        else:
            dprint(f"[JOIN_ORCHESTRATOR] Frame validation skipped (skip_frame_validation=True)")

        # === DETECT RESOLUTION FROM INPUT VIDEO (when use_input_video_resolution=True) ===
        if join_settings.get("use_input_video_resolution", False):
            dprint(f"[JOIN_ORCHESTRATOR] use_input_video_resolution=True, detecting resolution from first clip...")

            # Download first clip to detect resolution
            first_clip_url = clip_list[0].get("url")
            if first_clip_url:
                resolution_temp_dir = current_run_output_dir / "resolution_temp"
                resolution_temp_dir.mkdir(parents=True, exist_ok=True)

                local_first_clip = download_video_if_url(
                    first_clip_url,
                    download_target_dir=resolution_temp_dir,
                    task_id_for_logging=orchestrator_task_id_str,
                    descriptive_name="detect_resolution"
                )

                if local_first_clip and Path(local_first_clip).exists():
                    detected_res = _get_video_resolution(local_first_clip, dprint=dprint)
                    if detected_res:
                        join_settings["resolution"] = list(detected_res)  # [width, height]
                        dprint(f"[JOIN_ORCHESTRATOR] ✓ Detected resolution from input video: {detected_res}")
                    else:
                        dprint(f"[JOIN_ORCHESTRATOR] ⚠ Could not detect resolution, segments will detect from frames")
                else:
                    dprint(f"[JOIN_ORCHESTRATOR] ⚠ Could not download first clip for resolution detection")

        # === VLM PROMPT ENHANCEMENT (optional) ===
        enhance_prompt = orchestrator_payload.get("enhance_prompt", False)
        vlm_enhanced_prompts: List[Optional[str]] = [None] * num_joins
        
        if enhance_prompt:
            dprint(f"[JOIN_ORCHESTRATOR] enhance_prompt=True, generating VLM-enhanced prompts for {num_joins} joins")
            
            vlm_device = orchestrator_payload.get("vlm_device", "cuda")
            vlm_temp_dir = current_run_output_dir / "vlm_temp"
            vlm_temp_dir.mkdir(parents=True, exist_ok=True)
            base_prompt = join_settings.get("prompt", "")

            try:
                # Step 1: Extract boundary frames from all clips (4 images per join)
                dprint(f"[JOIN_ORCHESTRATOR] Extracting 4 frames per join from {len(clip_list)} clips...")
                image_quads = _extract_boundary_frames_for_vlm(
                    clip_list=clip_list,
                    temp_dir=vlm_temp_dir,
                    orchestrator_task_id=orchestrator_task_id_str,
                    dprint=dprint
                )
                
                # Step 2: Generate VLM prompts for all joins
                dprint(f"[JOIN_ORCHESTRATOR] Running VLM batch on {len(image_quads)} quads (4 images each)...")
                vlm_enhanced_prompts = _generate_vlm_prompts_for_joins(
                    image_quads=image_quads,
                    base_prompt=base_prompt,
                    vlm_device=vlm_device,
                    dprint=dprint
                )
                
                valid_count = sum(1 for p in vlm_enhanced_prompts if p is not None)
                dprint(f"[JOIN_ORCHESTRATOR] VLM enhancement complete: {valid_count}/{num_joins} prompts generated")
                
            except Exception as vlm_error:
                dprint(f"[JOIN_ORCHESTRATOR] VLM enhancement failed, using base prompts: {vlm_error}")
                traceback.print_exc()
                vlm_enhanced_prompts = [None] * num_joins
        else:
            dprint(f"[JOIN_ORCHESTRATOR] enhance_prompt=False, using base prompt for all joins")

        # === CREATE JOIN TASKS ===
        # Choose between parallel pattern (new, better quality) and chain pattern (legacy)
        use_parallel = orchestrator_payload.get("use_parallel_joins", True)  # Default to parallel

        if use_parallel:
            dprint(f"[JOIN_ORCHESTRATOR] Using PARALLEL pattern (transitions in parallel + final stitch)")
            success, message = _create_parallel_join_tasks(
                clip_list=clip_list,
                run_id=run_id,
                join_settings=join_settings,
                per_join_settings=per_join_settings,
                vlm_enhanced_prompts=vlm_enhanced_prompts,
                current_run_output_dir=current_run_output_dir,
                orchestrator_task_id_str=orchestrator_task_id_str,
                orchestrator_project_id=orchestrator_project_id,
                orchestrator_payload=orchestrator_payload,
                dprint=dprint
            )
        else:
            dprint(f"[JOIN_ORCHESTRATOR] Using CHAIN pattern (legacy sequential)")
            success, message = _create_join_chain_tasks(
                clip_list=clip_list,
                run_id=run_id,
                join_settings=join_settings,
                per_join_settings=per_join_settings,
                vlm_enhanced_prompts=vlm_enhanced_prompts,
                current_run_output_dir=current_run_output_dir,
                orchestrator_task_id_str=orchestrator_task_id_str,
                orchestrator_project_id=orchestrator_project_id,
                orchestrator_payload=orchestrator_payload,
                dprint=dprint
            )

        dprint(f"[JOIN_ORCHESTRATOR] {message}")
        return success, message

    except Exception as e:
        msg = f"Failed during join orchestration: {e}"
        dprint(f"[JOIN_ORCHESTRATOR] ERROR: {msg}")
        traceback.print_exc()
        return False, msg
