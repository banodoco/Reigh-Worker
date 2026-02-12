"""Travel stitch task handler - stitches travel segments into final video."""

import shutil
import traceback
from pathlib import Path
import time
import uuid
from datetime import datetime

try:
    import cv2
    _COLOR_MATCH_DEPS_AVAILABLE = True
except ImportError:
    _COLOR_MATCH_DEPS_AVAILABLE = False

# Import structured logging
from ...core.log import travel_logger, safe_json_repr

from ... import db_operations as db_ops
from ...utils import (
    generate_unique_task_id,
    get_video_frame_count_and_fps,
    parse_resolution,
    prepare_output_path,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    wait_for_file_stable,
)
from .orchestrator import DEFAULT_SEED_BASE, UPSCALE_SEED_OFFSET

from ...media.video import (
    extract_frames_from_video,
    create_video_from_frames_list,
    cross_fade_overlap_frames,
)

from .debug_utils import debug_video_analysis, log_ram_usage
from .ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback

def _handle_travel_stitch_task(task_params_from_db: dict, main_output_dir_base: Path, stitch_task_id_str: str, *, dprint):
    travel_logger.essential(f"Starting travel stitch task", task_id=stitch_task_id_str)
    log_ram_usage("Stitch start", task_id=stitch_task_id_str)
    dprint(f"[IMMEDIATE DEBUG] _handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    dprint(f"[IMMEDIATE DEBUG] task_params_from_db keys: {list(task_params_from_db.keys())}")
    dprint(f"_handle_travel_stitch_task: Starting for {stitch_task_id_str}")
    # Safe logging: Use safe_json_repr to prevent hangs
    dprint(f"Stitch task_params_from_db: {safe_json_repr(task_params_from_db)}")
    stitch_params = task_params_from_db # This now contains full_orchestrator_payload
    stitch_success = False
    final_video_location_for_db = None
    
    try:
        # --- 1. Initialization & Parameter Extraction --- 
        orchestrator_task_id_ref = stitch_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = stitch_params.get("orchestrator_run_id")
        # Support both canonical (orchestrator_details) and legacy (full_orchestrator_payload) names
        full_orchestrator_payload = stitch_params.get("orchestrator_details") or stitch_params.get("full_orchestrator_payload")

        print(f"[IMMEDIATE DEBUG] orchestrator_run_id: {orchestrator_run_id}")
        print(f"[IMMEDIATE DEBUG] orchestrator_task_id_ref: {orchestrator_task_id_ref}")
        print(f"[IMMEDIATE DEBUG] full_orchestrator_payload present: {full_orchestrator_payload is not None}")

        if not all([orchestrator_task_id_ref, orchestrator_run_id, full_orchestrator_payload]):
            msg = f"Stitch task {stitch_task_id_str} missing critical orchestrator refs or orchestrator_details."
            travel_logger.error(msg, task_id=stitch_task_id_str)
            return False, msg

        project_id_for_stitch = stitch_params.get("project_id")
        current_run_base_output_dir_str = stitch_params.get("current_run_base_output_dir", 
                                                            full_orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve())))
        current_run_base_output_dir = Path(current_run_base_output_dir_str)
        
        # Use the base directory directly without creating stitch-specific subdirectories
        stitch_processing_dir = current_run_base_output_dir
        stitch_processing_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"Stitch Task {stitch_task_id_str}: Processing in {stitch_processing_dir.resolve()}")

        num_expected_new_segments = full_orchestrator_payload["num_new_segments_to_generate"]
        print(f"[IMMEDIATE DEBUG] num_expected_new_segments: {num_expected_new_segments}")
        
        # Parse resolution from payload - but DON'T snap to model grid yet!
        # The actual resolution will be determined from the input segment videos.
        # Snapping is only needed for generation, not for stitching existing videos.
        parsed_res_wh_str = full_orchestrator_payload["parsed_resolution_wh"]
        try:
            parsed_res_wh_from_payload = parse_resolution(parsed_res_wh_str)
            if parsed_res_wh_from_payload is None:
                raise ValueError(f"parse_resolution returned None for input: {parsed_res_wh_str}")
            # NOTE: We use this as a fallback only. Actual resolution comes from input videos.
        except (ValueError, KeyError, TypeError) as e_parse_res_stitch:
            msg = f"Stitch Task {stitch_task_id_str}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res_stitch}"
            print(f"[ERROR Task {stitch_task_id_str}]: {msg}"); return False, msg
        dprint(f"Stitch Task {stitch_task_id_str}: Payload resolution (w,h): {parsed_res_wh_from_payload} (will use actual video resolution)")
        
        # Placeholder - will be set from actual input video after loading
        parsed_res_wh = None

        final_fps = full_orchestrator_payload.get("fps_helpers", 16)
        # CRITICAL: Use stitch_params overlay settings, NOT the orchestrator's default!
        # For SVI mode, frame_overlap_settings_expanded contains [4, 4, ...] (SVI_STITCH_OVERLAP)
        # For VACE mode, it contains the configured overlap values
        # Fallback to orchestrator's frame_overlap_expanded only if not provided
        expanded_frame_overlaps = stitch_params.get("frame_overlap_settings_expanded") or full_orchestrator_payload.get("frame_overlap_expanded", [])
        dprint(f"[STITCH DEBUG] Using overlap settings from stitch_params: {expanded_frame_overlaps[:5]}... (len={len(expanded_frame_overlaps)})")
        crossfade_sharp_amt = full_orchestrator_payload.get("crossfade_sharp_amt", 0.3)
        initial_continued_video_path_str = full_orchestrator_payload.get("continue_from_video_resolved_path")

        # [OVERLAP DEBUG] Add detailed debug for overlap values
        dprint(f"[OVERLAP DEBUG] Stitch: expanded_frame_overlaps from payload: {expanded_frame_overlaps}")

        # Extract upscale parameters
        upscale_factor = full_orchestrator_payload.get("upscale_factor", 0.0) # Default to 0.0 if not present
        upscale_model_name = full_orchestrator_payload.get("upscale_model_name") # Default to None if not present

        # --- 2. Collect Paths to All Segment Videos --- 
        segment_video_paths_for_stitch = []
        if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists():
            dprint(f"Stitch: Prepending initial continued video: {initial_continued_video_path_str}")
            # Check the continue video properties (resolution comparison deferred until actual resolution is determined)
            cap = cv2.VideoCapture(str(initial_continued_video_path_str))
            if cap.isOpened():
                continue_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                continue_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                continue_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                dprint(f"Stitch: Continue video properties - Resolution: {continue_width}x{continue_height}, Frames: {continue_frame_count}")
                # Note: Resolution will be determined from first video in list (which will be this continue video)
            else:
                dprint(f"Stitch: ERROR - Could not open continue video for property check")
            segment_video_paths_for_stitch.append(str(Path(initial_continued_video_path_str).resolve()))
        
        # Fetch completed segments with a small retry loop to handle race conditions
        max_stitch_fetch_retries = 6  # Allow up to ~18s total wait
        completed_segment_outputs_from_db = []
        
        print(f"[IMMEDIATE DEBUG] About to start retry loop for run_id: {orchestrator_run_id}")
        
        for attempt in range(max_stitch_fetch_retries):
            print(f"[IMMEDIATE DEBUG] Stitch fetch attempt {attempt+1}/{max_stitch_fetch_retries} for run_id: {orchestrator_run_id}")
            dprint(f"[DEBUG] Stitch fetch attempt {attempt+1}/{max_stitch_fetch_retries} for run_id: {orchestrator_run_id}")
            
            try:
                completed_segment_outputs_from_db = db_ops.get_completed_segment_outputs_for_stitch(orchestrator_run_id, project_id=project_id_for_stitch) or []
                print(f"[IMMEDIATE DEBUG] DB query returned: {completed_segment_outputs_from_db}")
            except (RuntimeError, ValueError, OSError) as e_db_query:
                print(f"[IMMEDIATE DEBUG] DB query failed: {e_db_query}")
                completed_segment_outputs_from_db = []
            
            dprint(f"[DEBUG] Attempt {attempt+1} returned {len(completed_segment_outputs_from_db)} segments")
            print(f"[IMMEDIATE DEBUG] Attempt {attempt+1} returned {len(completed_segment_outputs_from_db)} segments")
            
            if len(completed_segment_outputs_from_db) >= num_expected_new_segments:
                dprint(f"[DEBUG] Expected {num_expected_new_segments} segment rows found on attempt {attempt+1}. Proceeding.")
                print(f"[IMMEDIATE DEBUG] Expected {num_expected_new_segments} segment rows found on attempt {attempt+1}. Proceeding.")
                break
            dprint(f"Stitch: No completed segment rows found (attempt {attempt+1}/{max_stitch_fetch_retries}). Waiting 3s and retrying...")
            print(f"[IMMEDIATE DEBUG] Insufficient segments found (attempt {attempt+1}/{max_stitch_fetch_retries}). Waiting 3s and retrying...")
            if attempt < max_stitch_fetch_retries - 1:  # Don't sleep after the last attempt
                time.sleep(3)
        dprint(f"Stitch Task {stitch_task_id_str}: Completed segments fetched: {completed_segment_outputs_from_db}")
        print(f"[IMMEDIATE DEBUG] Final completed_segment_outputs_from_db: {completed_segment_outputs_from_db}")

        # ------------------------------------------------------------------
        # 2b. Resolve each returned video path (relative path or URL)
        # ------------------------------------------------------------------
        dprint(f"[STITCH_DEBUG] Starting path resolution for {len(completed_segment_outputs_from_db)} segments")
        dprint(f"[STITCH_DEBUG] Raw DB results: {completed_segment_outputs_from_db}")
        for seg_idx, video_path_str_from_db in completed_segment_outputs_from_db:
            dprint(f"[STITCH_DEBUG] Processing segment {seg_idx} with path: {video_path_str_from_db}")
            resolved_video_path_for_stitch: Path | None = None

            if not video_path_str_from_db:
                dprint(f"[STITCH_DEBUG] WARNING: Segment {seg_idx} has empty video_path in DB; skipping.")
                continue

            # Case A: Relative path that starts with files/ - resolve from current working directory
            if video_path_str_from_db.startswith("files/") or video_path_str_from_db.startswith("public/files/"):
                dprint(f"[STITCH_DEBUG] Case A: Relative path detected for segment {seg_idx}")
                # Resolve relative to current working directory
                base_dir = Path.cwd()
                absolute_path_candidate = (base_dir / "public" / video_path_str_from_db.lstrip("public/")).resolve()
                print(f"[STITCH_DEBUG] Resolved relative path '{video_path_str_from_db}' to '{absolute_path_candidate}' for segment {seg_idx}")
                dprint(f"Stitch: Resolved relative path '{video_path_str_from_db}' to '{absolute_path_candidate}' for segment {seg_idx}")
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                    print(f"[STITCH_DEBUG] ✅ File exists at resolved path")
                else:
                    print(f"[STITCH_DEBUG] ❌ File missing at resolved path")
                    dprint(f"[WARNING] Stitch: Resolved absolute path '{absolute_path_candidate}' for segment {seg_idx} is missing.")

            # Case B: Remote public URL (Supabase storage)
            elif video_path_str_from_db.startswith("http"):
                print(f"[STITCH_DEBUG] Case B: Remote URL detected for segment {seg_idx}")
                try:
                    from ...utils import download_file as download_file
                    remote_url = video_path_str_from_db
                    local_filename = Path(remote_url).name
                    local_download_path = stitch_processing_dir / f"seg{seg_idx:02d}_{local_filename}"
                    print(f"[STITCH_DEBUG] Remote URL: {remote_url}")
                    print(f"[STITCH_DEBUG] Local download path: {local_download_path}")
                    dprint(f"[DEBUG] Remote URL detected, local download path: {local_download_path}")
                    
                    # Check if cached file exists and validate its frame count against orchestrator's expected values
                    need_download = True
                    if local_download_path.exists():
                        print(f"[STITCH_DEBUG] Local copy exists, validating frame count...")
                        try:
                            cached_frames, _ = get_video_frame_count_and_fps(str(local_download_path))
                            expected_segment_frames = full_orchestrator_payload["segment_frames_expanded"]
                            expected_frames = expected_segment_frames[seg_idx] if seg_idx < len(expected_segment_frames) else None
                            print(f"[STITCH_DEBUG] Cached file has {cached_frames} frames (expected: {expected_frames})")
                            
                            if expected_frames and cached_frames == expected_frames:
                                print(f"[STITCH_DEBUG] ✅ Cached file frame count matches expected ({cached_frames} frames)")
                                need_download = False
                            elif expected_frames:
                                print(f"[STITCH_DEBUG] ❌ Cached file frame count mismatch! Expected {expected_frames}, got {cached_frames}, will re-download")
                            else:
                                print(f"[STITCH_DEBUG] ❌ No expected frame count available for segment {seg_idx}, will re-download")
                        except (OSError, ValueError, RuntimeError, KeyError, IndexError) as e_validate:
                            print(f"[STITCH_DEBUG] Could not validate cached file: {e_validate}, will re-download")
                    
                    if need_download:
                        print(f"[STITCH_DEBUG] Downloading remote segment {seg_idx}...")
                        dprint(f"Stitch: Downloading remote segment {seg_idx} from {remote_url} to {local_download_path}")
                        # Remove stale cached file if it exists
                        if local_download_path.exists():
                            local_download_path.unlink()
                        download_file(remote_url, stitch_processing_dir, local_download_path.name)
                        print(f"[STITCH_DEBUG] ✅ Download completed for segment {seg_idx}")
                        dprint(f"[DEBUG] Download completed for segment {seg_idx}")
                    else:
                        print(f"[STITCH_DEBUG] ✅ Using validated cached file for segment {seg_idx}")
                        dprint(f"Stitch: Using validated cached file for segment {seg_idx} at {local_download_path}")
                    
                    resolved_video_path_for_stitch = local_download_path
                except (OSError, ValueError, RuntimeError) as e_dl:
                    print(f"[STITCH_DEBUG] ❌ Download failed for segment {seg_idx}: {e_dl}")
                    dprint(f"[WARNING] Stitch: Failed to download remote video for segment {seg_idx}: {e_dl}")

            # Case C: Provided absolute/local path
            else:
                print(f"[STITCH_DEBUG] Case C: Absolute/local path for segment {seg_idx}")
                absolute_path_candidate = Path(video_path_str_from_db).resolve()
                print(f"[STITCH_DEBUG] Treating as absolute path: {absolute_path_candidate}")
                dprint(f"[DEBUG] Treating as absolute path: {absolute_path_candidate}")
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                    print(f"[STITCH_DEBUG] ✅ Absolute path exists")
                    dprint(f"[DEBUG] Absolute path exists: {absolute_path_candidate}")
                else:
                    print(f"[STITCH_DEBUG] ❌ Absolute path missing or not a file")
                    dprint(f"[WARNING] Stitch: Absolute path '{absolute_path_candidate}' for segment {seg_idx} does not exist or is not a file.")

            if resolved_video_path_for_stitch is not None:
                segment_video_paths_for_stitch.append(str(resolved_video_path_for_stitch))
                print(f"[STITCH_DEBUG] ✅ Added video for segment {seg_idx}: {resolved_video_path_for_stitch}")
                dprint(f"Stitch: Added video for segment {seg_idx}: {resolved_video_path_for_stitch}")
                
                # Analyze the resolved video immediately
                debug_video_analysis(resolved_video_path_for_stitch, f"RESOLVED_Seg{seg_idx}", stitch_task_id_str)
            else: 
                print(f"[STITCH_DEBUG] ❌ Unable to resolve video for segment {seg_idx}; will be excluded from stitching.")
                dprint(f"[WARNING] Stitch: Unable to resolve video for segment {seg_idx}; will be excluded from stitching.")

        print(f"[STITCH_DEBUG] Path resolution complete")
        print(f"[STITCH_DEBUG] Final segment_video_paths_for_stitch: {segment_video_paths_for_stitch}")
        print(f"[STITCH_DEBUG] Total videos collected: {len(segment_video_paths_for_stitch)}")
        dprint(f"[DEBUG] Final segment_video_paths_for_stitch: {segment_video_paths_for_stitch}")
        dprint(f"[DEBUG] Total videos collected: {len(segment_video_paths_for_stitch)}")
        # [CRITICAL DEBUG] Log each video's frame count before stitching
        dprint(f"[CRITICAL DEBUG] About to stitch videos:")
        expected_segment_frames = full_orchestrator_payload["segment_frames_expanded"]
        for idx, video_path in enumerate(segment_video_paths_for_stitch):
            try:
                frame_count, fps = get_video_frame_count_and_fps(video_path)
                expected_frames = expected_segment_frames[idx] if idx < len(expected_segment_frames) else "unknown"
                dprint(f"[CRITICAL DEBUG] Video {idx}: {video_path} -> {frame_count} frames @ {fps} FPS (expected: {expected_frames})")
                if expected_frames != "unknown" and frame_count != expected_frames:
                    dprint(f"[CRITICAL DEBUG] ⚠️  FRAME COUNT MISMATCH! Expected {expected_frames}, got {frame_count}")
            except (OSError, ValueError, RuntimeError) as e_debug:
                dprint(f"[CRITICAL DEBUG] Video {idx}: {video_path} -> ERROR: {e_debug}")

        total_videos_for_stitch = (1 if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists() else 0) + num_expected_new_segments
        dprint(f"[DEBUG] Expected total videos: {total_videos_for_stitch}")
        if len(segment_video_paths_for_stitch) < total_videos_for_stitch:
            # This is a warning because some segments might have legitimately failed and been skipped by their handlers.
            # The stitcher should proceed with what it has, unless it has zero or one video when multiple were expected.
            dprint(f"[WARNING] Stitch: Expected {total_videos_for_stitch} videos for stitch, but found {len(segment_video_paths_for_stitch)}. Stitching with available videos.")
        
        if not segment_video_paths_for_stitch:
            dprint(f"[ERROR] Stitch: No valid segment videos found to stitch. DB returned {len(completed_segment_outputs_from_db)} segments, but none resolved to valid paths.")
            raise ValueError("Stitch: No valid segment videos found to stitch.")
        if len(segment_video_paths_for_stitch) == 1 and total_videos_for_stitch > 1:
            dprint(f"Stitch: Only one video segment found ({segment_video_paths_for_stitch[0]}) but {total_videos_for_stitch} were expected. Using this single video as the 'stitched' output.")
            # No actual stitching needed, just move/copy this single video to final dest.

        # --- 2c. Determine ACTUAL resolution from input videos ---
        # CRITICAL: Use the actual resolution of the input segment videos, not the snapped payload resolution.
        # This prevents dimension changes during stitching (e.g., 902x508 → 896x496).
        first_video_path = segment_video_paths_for_stitch[0]
        try:
            cap = cv2.VideoCapture(first_video_path)
            if cap.isOpened():
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                parsed_res_wh = (actual_width, actual_height)
                print(f"[STITCH_RESOLUTION] Using actual video resolution: {actual_width}x{actual_height}")
                dprint(f"Stitch: Using actual video resolution from first segment: {actual_width}x{actual_height}")
                
                # Log if there's a difference from payload resolution
                if parsed_res_wh_from_payload and parsed_res_wh != parsed_res_wh_from_payload:
                    print(f"[STITCH_RESOLUTION] ⚠️  Payload resolution was {parsed_res_wh_from_payload}, actual is {parsed_res_wh}")
                    dprint(f"Stitch: Resolution difference - payload: {parsed_res_wh_from_payload}, actual: {parsed_res_wh}")
            else:
                cap.release()
                raise ValueError(f"Could not open first video: {first_video_path}")
        except (OSError, ValueError, RuntimeError) as e_res:
            # Fallback to payload resolution (without snapping) if we can't read the video
            print(f"[STITCH_RESOLUTION] ⚠️  Could not read resolution from video, using payload: {e_res}")
            dprint(f"Stitch: Warning - could not get resolution from video ({e_res}), using payload resolution")
            parsed_res_wh = parsed_res_wh_from_payload

        # --- 3. Stitching (Crossfade or Concatenate) --- 
        current_stitched_video_path: Path | None = None # This will hold the path to the current version of the stitched video


        if len(segment_video_paths_for_stitch) == 1:
            # If only one video, copy it directly using prepare_output_path
            source_single_video_path = Path(segment_video_paths_for_stitch[0])
            single_video_filename = f"{stitch_task_id_str}_single_video{source_single_video_path.suffix}"

            current_stitched_video_path, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=single_video_filename,
                main_output_dir_base=main_output_dir_base,
                task_type="travel_stitch"
            )
            shutil.copy2(str(source_single_video_path), str(current_stitched_video_path))
            dprint(f"Stitch: Only one video found. Copied {source_single_video_path} to {current_stitched_video_path}")
        else: # More than one video, proceed with stitching logic
            num_stitch_points = len(segment_video_paths_for_stitch) - 1
            actual_overlaps_for_stitching = []
            if initial_continued_video_path_str: 
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points] 
            else: 
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points]
            
            # --- NEW OVERLAP DEBUG LOGGING ---
            dprint(f"[OVERLAP DEBUG] Number of videos: {len(segment_video_paths_for_stitch)} (expected stitch points: {num_stitch_points})")
            dprint(f"[OVERLAP DEBUG] actual_overlaps_for_stitching: {actual_overlaps_for_stitching}")
            if len(actual_overlaps_for_stitching) != num_stitch_points:
                dprint(f"[OVERLAP DEBUG] ⚠️  MISMATCH! We have {len(actual_overlaps_for_stitching)} overlaps for {num_stitch_points} joins")
            for join_idx, ov in enumerate(actual_overlaps_for_stitching):
                dprint(f"[OVERLAP DEBUG]   Join {join_idx} (video {join_idx} -> {join_idx+1}): overlap={ov}")
            # --- END NEW LOGGING ---
            
            any_positive_overlap = any(o > 0 for o in actual_overlaps_for_stitching)

            raw_stitched_video_filename = f"{stitch_task_id_str}_stitched_intermediate.mp4"
            path_for_raw_stitched_video, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=raw_stitched_video_filename,
                main_output_dir_base=main_output_dir_base,
                task_type="travel_stitch"
            )

            if any_positive_overlap:
                print(f"[CRITICAL DEBUG] Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")
                print(f"[STITCH_ANALYSIS] Cross-fade stitching analysis:")
                print(f"[STITCH_ANALYSIS]   Number of videos: {len(segment_video_paths_for_stitch)}")
                print(f"[STITCH_ANALYSIS]   Overlap values: {actual_overlaps_for_stitching}")
                print(f"[STITCH_ANALYSIS]   Expected stitch points: {num_stitch_points}")
                
                dprint(f"Stitch: Using cross-fade due to overlap values: {actual_overlaps_for_stitching}. Output to: {path_for_raw_stitched_video}")

                # Wait for all segment videos to be stable before extracting frames
                print(f"[CRITICAL DEBUG] Waiting for {len(segment_video_paths_for_stitch)} segment videos to be stable before frame extraction...")
                stable_paths = []
                for idx, video_path in enumerate(segment_video_paths_for_stitch):
                    dprint(f"Stitch: Checking file stability for segment {idx}: {video_path}")
                    if not Path(video_path).exists():
                        print(f"[CRITICAL DEBUG] Segment {idx} video file does not exist: {video_path}")
                        stable_paths.append(False)
                        continue

                    file_stable = wait_for_file_stable(video_path, checks=5, interval=1.0, dprint=dprint)
                    if file_stable:
                        print(f"[CRITICAL DEBUG] Segment {idx} video file is stable: {video_path}")
                        stable_paths.append(True)
                    else:
                        print(f"[CRITICAL DEBUG] Segment {idx} video file is NOT stable after waiting: {video_path}")
                        stable_paths.append(False)

                if not all(stable_paths):
                    unstable_indices = [i for i, stable in enumerate(stable_paths) if not stable]
                    raise ValueError(f"Stitch: One or more segment videos are not stable or missing: indices {unstable_indices}")

                print(f"[CRITICAL DEBUG] All segment videos are stable. Proceeding with frame extraction...")

                # Retry frame extraction with backoff and re-download for corrupted videos
                max_extraction_attempts = 3
                all_segment_frames_lists = None
                retry_log = []  # Track all retry attempts for detailed error reporting

                for attempt in range(max_extraction_attempts):
                    print(f"[CRITICAL DEBUG] Frame extraction attempt {attempt + 1}/{max_extraction_attempts}")
                    attempt_start_time = time.time()
                    all_segment_frames_lists = [extract_frames_from_video(p, dprint_func=dprint) for p in segment_video_paths_for_stitch]

                    # [CRITICAL DEBUG] Log frame extraction results
                    print(f"[CRITICAL DEBUG] Frame extraction results (attempt {attempt + 1}):")
                    failed_segments = []
                    successful_segments = []
                    for idx, frame_list in enumerate(all_segment_frames_lists):
                        if frame_list is not None and len(frame_list) > 0:
                            print(f"[CRITICAL DEBUG] Segment {idx}: {len(frame_list)} frames extracted")
                            successful_segments.append(idx)
                        else:
                            print(f"[CRITICAL DEBUG] Segment {idx}: FAILED to extract frames")
                            failed_segments.append(idx)

                    # Log this attempt
                    attempt_duration = time.time() - attempt_start_time
                    attempt_info = {
                        "attempt": attempt + 1,
                        "duration_seconds": round(attempt_duration, 2),
                        "successful_segments": successful_segments,
                        "failed_segments": failed_segments,
                        "redownloads": []
                    }

                    # Check if all extractions succeeded
                    if all(f_list is not None and len(f_list) > 0 for f_list in all_segment_frames_lists):
                        print(f"[CRITICAL DEBUG] All frame extractions successful on attempt {attempt + 1}")
                        retry_log.append(attempt_info)
                        break

                    # If not the last attempt, try to re-download corrupted videos before retry
                    if attempt < max_extraction_attempts - 1 and failed_segments:
                        wait_time = 3 + (attempt * 2)  # Progressive backoff: 3s, 5s, 7s
                        print(f"[CRITICAL DEBUG] Frame extraction failed for segments {failed_segments}. Attempting re-download and retry in {wait_time} seconds...")

                        # Try to re-download failed segments (only for remote URLs)
                        redownload_attempted = False
                        for failed_idx in failed_segments:
                            if failed_idx < len(completed_segment_outputs_from_db):
                                seg_output = completed_segment_outputs_from_db[failed_idx]
                                video_path_str_from_db = seg_output.get("video_file_path", "")

                                # Check if it's a remote URL that can be re-downloaded
                                if video_path_str_from_db.startswith("http"):
                                    try:
                                        failed_video_path = Path(segment_video_paths_for_stitch[failed_idx])
                                        print(f"[CRITICAL DEBUG] Re-downloading corrupted segment {failed_idx} from {video_path_str_from_db}")

                                        redownload_start = time.time()

                                        # Delete corrupted file
                                        if failed_video_path.exists():
                                            failed_video_path.unlink()
                                            print(f"[CRITICAL DEBUG] Deleted corrupted file: {failed_video_path}")

                                        # Re-download
                                        download_file(video_path_str_from_db, stitch_processing_dir, failed_video_path.name)

                                        # Wait for stability
                                        if wait_for_file_stable(failed_video_path, checks=5, interval=1.0, dprint=dprint):
                                            print(f"[CRITICAL DEBUG] Re-downloaded segment {failed_idx} successfully")
                                            redownload_duration = time.time() - redownload_start
                                            attempt_info["redownloads"].append({
                                                "segment_idx": failed_idx,
                                                "source_url": video_path_str_from_db,
                                                "duration_seconds": round(redownload_duration, 2),
                                                "success": True
                                            })
                                            redownload_attempted = True
                                        else:
                                            print(f"[CRITICAL DEBUG] Re-downloaded segment {failed_idx} not stable")
                                            redownload_duration = time.time() - redownload_start
                                            attempt_info["redownloads"].append({
                                                "segment_idx": failed_idx,
                                                "source_url": video_path_str_from_db,
                                                "duration_seconds": round(redownload_duration, 2),
                                                "success": False,
                                                "error": "File not stable after download"
                                            })

                                    except (OSError, ValueError, RuntimeError) as e_redownload:
                                        print(f"[CRITICAL DEBUG] Re-download failed for segment {failed_idx}: {e_redownload}")
                                        redownload_duration = time.time() - redownload_start
                                        attempt_info["redownloads"].append({
                                            "segment_idx": failed_idx,
                                            "source_url": video_path_str_from_db,
                                            "duration_seconds": round(redownload_duration, 2),
                                            "success": False,
                                            "error": str(e_redownload)
                                        })
                                else:
                                    print(f"[CRITICAL DEBUG] Segment {failed_idx} is not a remote URL, cannot re-download: {video_path_str_from_db}")
                                    attempt_info["redownloads"].append({
                                        "segment_idx": failed_idx,
                                        "source_url": video_path_str_from_db,
                                        "duration_seconds": 0,
                                        "success": False,
                                        "error": "Not a remote URL - cannot re-download"
                                    })

                        if redownload_attempted:
                            print(f"[CRITICAL DEBUG] Re-download completed. Waiting {wait_time} seconds before next extraction attempt...")
                        else:
                            print(f"[CRITICAL DEBUG] No re-downloads attempted. Waiting {wait_time} seconds before next extraction attempt...")

                        time.sleep(wait_time)

                    # Log this attempt (whether successful or failed)
                    retry_log.append(attempt_info)
                else:
                    # All attempts failed - generate detailed error report
                    failed_segments = [i for i, f_list in enumerate(all_segment_frames_lists) if not (f_list is not None and len(f_list) > 0)]

                    # Build detailed error message
                    error_details = []
                    error_details.append(f"Frame extraction failed for segments {failed_segments} after {max_extraction_attempts} attempts")
                    error_details.append(f"Total segments in stitch: {len(segment_video_paths_for_stitch)}")

                    # Add per-segment analysis
                    for idx, video_path in enumerate(segment_video_paths_for_stitch):
                        video_path_obj = Path(video_path)
                        status = "✅ SUCCESS" if idx not in failed_segments else "❌ FAILED"

                        if video_path_obj.exists():
                            try:
                                file_size = video_path_obj.stat().st_size
                                # Try to get basic video info
                                cap = cv2.VideoCapture(str(video_path))
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
                                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else -1
                                cap.release()

                                error_details.append(f"  Segment {idx} [{status}]: {video_path_obj.name} ({file_size:,} bytes, {frame_count} frames, {fps:.1f} fps)")
                            except (OSError, ValueError, RuntimeError):
                                file_size = video_path_obj.stat().st_size if video_path_obj.exists() else 0
                                error_details.append(f"  Segment {idx} [{status}]: {video_path_obj.name} ({file_size:,} bytes, properties unreadable)")
                        else:
                            error_details.append(f"  Segment {idx} [{status}]: {video_path} (FILE MISSING)")

                    # Add source information if available
                    if 'completed_segment_outputs_from_db' in locals():
                        error_details.append("Source URLs:")
                        for idx in failed_segments:
                            if idx < len(completed_segment_outputs_from_db):
                                seg_output = completed_segment_outputs_from_db[idx]
                                source_url = seg_output.get("video_file_path", "Unknown")
                                error_details.append(f"  Failed segment {idx} source: {source_url}")

                    # Add retry history
                    if retry_log:
                        error_details.append("Retry History:")
                        for log_entry in retry_log:
                            attempt_summary = f"  Attempt {log_entry['attempt']}: {log_entry['duration_seconds']}s, Success:{len(log_entry['successful_segments'])}, Failed:{len(log_entry['failed_segments'])}"
                            if log_entry['redownloads']:
                                redownload_summary = []
                                for rd in log_entry['redownloads']:
                                    status = "✅" if rd['success'] else "❌"
                                    redownload_summary.append(f"Seg{rd['segment_idx']}({status}{rd['duration_seconds']}s)")
                                attempt_summary += f", Redownloads:[{','.join(redownload_summary)}]"
                            error_details.append(attempt_summary)

                    # Before failing, try FFmpeg-based cross-fade as fallback
                    print(f"[CRITICAL DEBUG] Frame extraction failed completely. Attempting FFmpeg cross-fade fallback...")
                    try:
                        ffmpeg_result = attempt_ffmpeg_crossfade_fallback(
                            segment_video_paths_for_stitch,
                            actual_overlaps_for_stitching,
                            path_for_raw_stitched_video,
                            stitch_task_id_str,
                            dprint
                        )
                        if ffmpeg_result:
                            print(f"[CRITICAL DEBUG] FFmpeg cross-fade fallback succeeded!")
                            current_stitched_video_path = path_for_raw_stitched_video
                        else:
                            detailed_error = "Stitch: Both frame extraction and FFmpeg cross-fade fallback failed. " + " | ".join(error_details)
                            raise ValueError(detailed_error)
                    except (OSError, ValueError, RuntimeError) as e_ffmpeg:
                        detailed_error = f"Stitch: Frame extraction failed and FFmpeg fallback also failed ({str(e_ffmpeg)}). " + " | ".join(error_details)
                        raise ValueError(detailed_error)
                
                final_stitched_frames = []
                
                # Process each stitch point
                for i in range(num_stitch_points): 
                    frames_prev_segment = all_segment_frames_lists[i]
                    frames_curr_segment = all_segment_frames_lists[i+1]
                    current_overlap_val = actual_overlaps_for_stitching[i]

                    print(f"[CRITICAL DEBUG] Stitch point {i}: segments {i}->{i+1}, overlap={current_overlap_val}")
                    print(f"[CRITICAL DEBUG] Prev segment: {len(frames_prev_segment)} frames, Curr segment: {len(frames_curr_segment)} frames")

                    # --- NEW OVERLAP DETAIL LOG ---
                    if current_overlap_val > 0:
                        start_prev = len(frames_prev_segment) - current_overlap_val
                        end_prev = len(frames_prev_segment) - 1
                        start_curr = 0
                        end_curr = current_overlap_val - 1
                        print(
                            f"[OVERLAP_DETAIL] Join {i}: blending prev[{start_prev}:{end_prev}] with curr[{start_curr}:{end_curr}] (total {current_overlap_val} frames)"
                        )
                    # --- END OVERLAP DETAIL LOG ---

                    if i == 0:
                        # For the first stitch point, add frames from segment 0 up to the overlap
                        if current_overlap_val > 0:
                            # Add frames before the overlap region
                            frames_before_overlap = frames_prev_segment[:-current_overlap_val]
                            final_stitched_frames.extend(frames_before_overlap)
                            print(f"[CRITICAL DEBUG] Added {len(frames_before_overlap)} frames from segment 0 (before overlap)")
                        else:
                            # No overlap, add all frames from segment 0
                            final_stitched_frames.extend(frames_prev_segment)
                            print(f"[CRITICAL DEBUG] Added all {len(frames_prev_segment)} frames from segment 0 (no overlap)")
                    else:
                        pass

                    if current_overlap_val > 0:
                        # Check if we should regenerate anchor frames (skip blending the anchor)
                        regenerate_anchors = full_orchestrator_payload.get("regenerate_anchors", False)

                        if regenerate_anchors and current_overlap_val > 1:
                            # Regenerate anchor mode: crossfade all but the last frame (anchor)
                            # The anchor frame will be taken directly from current segment
                            crossfade_count = current_overlap_val - 1

                            # Remove the overlap frames (minus 1 for anchor) from accumulated
                            if i > 0:
                                frames_to_remove = min(crossfade_count, len(final_stitched_frames))
                                if frames_to_remove > 0:
                                    del final_stitched_frames[-frames_to_remove:]
                                    print(f"[CRITICAL DEBUG] [REGENERATE_ANCHORS] Removed {frames_to_remove} frames before cross-fade (keeping previous anchor)")

                            # Blend the non-anchor overlapping frames
                            frames_prev_for_fade = frames_prev_segment[-crossfade_count:] if crossfade_count > 0 else []
                            frames_curr_for_fade = frames_curr_segment[:crossfade_count]
                            faded_frames = cross_fade_overlap_frames(frames_prev_for_fade, frames_curr_for_fade, crossfade_count, "linear_sharp", crossfade_sharp_amt)
                            final_stitched_frames.extend(faded_frames)
                            print(f"[CRITICAL DEBUG] [REGENERATE_ANCHORS] Added {len(faded_frames)} cross-faded frames (skipping anchor)")

                            # Add the regenerated anchor frame directly (no blend)
                            anchor_frame = frames_curr_segment[crossfade_count]
                            final_stitched_frames.append(anchor_frame)
                            print(f"[CRITICAL DEBUG] [REGENERATE_ANCHORS] Added regenerated anchor frame directly (no blend)")

                            # Adjust start index for remaining frames
                            start_index_for_curr_tail = current_overlap_val
                        else:
                            # Normal crossfade mode: blend all overlap frames
                            # Remove the overlap frames already appended from the previous segment so that
                            # they can be replaced by the blended cross-fade frames for this stitch point.
                            if i > 0:
                                frames_to_remove = min(current_overlap_val, len(final_stitched_frames))
                                if frames_to_remove > 0:
                                    del final_stitched_frames[-frames_to_remove:]
                                    print(f"[CRITICAL DEBUG] Removed {frames_to_remove} duplicate overlap frames before cross-fade (stitch point {i})")
                            # Blend the overlapping frames
                            faded_frames = cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, "linear_sharp", crossfade_sharp_amt)
                            final_stitched_frames.extend(faded_frames)
                            print(f"[CRITICAL DEBUG] Added {len(faded_frames)} cross-faded frames")

                            # Normal start index for remaining frames
                            start_index_for_curr_tail = current_overlap_val
                    else:
                        start_index_for_curr_tail = 0

                    # Add the non-overlapping part of the current segment
                    if len(frames_curr_segment) > start_index_for_curr_tail:
                        frames_to_add = frames_curr_segment[start_index_for_curr_tail:]
                        final_stitched_frames.extend(frames_to_add)
                        print(f"[CRITICAL DEBUG] Added {len(frames_to_add)} frames from segment {i+1} (after overlap)")
                    
                    print(f"[CRITICAL DEBUG] Running total after stitch point {i}: {len(final_stitched_frames)} frames")
                
                if not final_stitched_frames: raise ValueError("Stitch: No frames produced after cross-fade logic.")
                
                # [CRITICAL DEBUG] Final calculation summary
                # With proper cross-fade: output = sum(all frames) - sum(overlaps)
                # Because overlapped frames are blended, not duplicated
                total_input_frames = sum(len(frames) for frames in all_segment_frames_lists)
                total_overlaps = sum(actual_overlaps_for_stitching)
                expected_output_frames = total_input_frames - total_overlaps
                actual_output_frames = len(final_stitched_frames)
                print(f"[CRITICAL DEBUG] FINAL CROSS-FADE SUMMARY:")
                print(f"[CRITICAL DEBUG] Total input frames: {total_input_frames}")
                print(f"[CRITICAL DEBUG] Total overlaps: {total_overlaps}")
                print(f"[CRITICAL DEBUG] Expected output: {expected_output_frames}")
                print(f"[CRITICAL DEBUG] Actual output: {actual_output_frames}")
                print(f"[CRITICAL DEBUG] Match: {expected_output_frames == actual_output_frames}")
                
                created_video_path_obj = create_video_from_frames_list(final_stitched_frames, path_for_raw_stitched_video, final_fps, parsed_res_wh)
                if created_video_path_obj and created_video_path_obj.exists():
                    current_stitched_video_path = created_video_path_obj
                else:
                    raise RuntimeError(f"Stitch: Cross-fade create_video_from_frames_list failed to produce video at {path_for_raw_stitched_video}")

            else: 
                dprint(f"Stitch: Using simple FFmpeg concatenation. Output to: {path_for_raw_stitched_video}")
                try:
                    from ...utils import stitch_videos_ffmpeg as stitch_videos_ffmpeg
                except ImportError:
                    print(f"[CRITICAL ERROR Task ID: {stitch_task_id_str}] Failed to import 'stitch_videos_ffmpeg'. Cannot proceed with stitching.")
                    raise

                if stitch_videos_ffmpeg(segment_video_paths_for_stitch, str(path_for_raw_stitched_video)):
                    current_stitched_video_path = path_for_raw_stitched_video
                else: 
                    raise RuntimeError(f"Stitch: Simple FFmpeg concatenation failed for output {path_for_raw_stitched_video}.")

        if not current_stitched_video_path or not current_stitched_video_path.exists():
            raise RuntimeError(f"Stitch: Stitching process failed, output video not found at {current_stitched_video_path}")
        
        video_path_after_optional_upscale = current_stitched_video_path

        if isinstance(upscale_factor, (float, int)) and upscale_factor > 1.0 and upscale_model_name:
            print(f"[STITCH UPSCALE] Starting upscale process: {upscale_factor}x using model {upscale_model_name}")
            dprint(f"Stitch: Upscaling (x{upscale_factor}) video {current_stitched_video_path.name} using model {upscale_model_name}")
            
            original_frames_count, original_fps = get_video_frame_count_and_fps(str(current_stitched_video_path))
            if original_frames_count is None or original_frames_count == 0:
                raise ValueError(f"Stitch: Cannot get frame count or 0 frames for video {current_stitched_video_path} before upscaling.")
            
            print(f"[STITCH UPSCALE] Input video: {original_frames_count} frames @ {original_fps} FPS")
            print(f"[STITCH UPSCALE] Target resolution: {int(parsed_res_wh[0] * upscale_factor)}x{int(parsed_res_wh[1] * upscale_factor)}")
            dprint(f"[DEBUG] Pre-upscale analysis: {original_frames_count} frames, {original_fps} FPS")

            target_width_upscaled = int(parsed_res_wh[0] * upscale_factor)
            target_height_upscaled = int(parsed_res_wh[1] * upscale_factor)
            
            upscale_sub_task_id = generate_unique_task_id(f"upscale_stitch_{orchestrator_run_id}_")
            
            upscale_payload = {
                "task_id": upscale_sub_task_id,
                "project_id": stitch_params.get("project_id"),
                "model": upscale_model_name,
                "video_source_path": str(current_stitched_video_path.resolve()), 
                "resolution": f"{target_width_upscaled}x{target_height_upscaled}",
                "frames": original_frames_count,
                "prompt": full_orchestrator_payload.get("original_task_args",{}).get("upscale_prompt", "cinematic, masterpiece, high detail, 4k"), 
                "seed": full_orchestrator_payload.get("seed_for_upscale", full_orchestrator_payload.get("seed_base", DEFAULT_SEED_BASE) + UPSCALE_SEED_OFFSET),
            }

            upscaler_engine_to_use = stitch_params.get("execution_engine_for_upscale", "wgp")
            
            db_ops.add_task_to_db(
                task_payload=upscale_payload, 
                task_type_str=upscaler_engine_to_use
            )
            print(f"[STITCH UPSCALE] Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            print(f"Stitch Task {stitch_task_id_str}: Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...")
            
            poll_interval_ups = full_orchestrator_payload.get("poll_interval", 15)
            poll_timeout_ups = full_orchestrator_payload.get("poll_timeout_upscale", full_orchestrator_payload.get("poll_timeout", 30 * 60) * 2)
            
            print(f"[STITCH UPSCALE] Polling for completion (timeout: {poll_timeout_ups}s, interval: {poll_interval_ups}s)")
            
            upscaled_video_db_location = db_ops.poll_task_status(
                task_id=upscale_sub_task_id, 
                poll_interval_seconds=poll_interval_ups, 
                timeout_seconds=poll_timeout_ups
            )
            print(f"[STITCH UPSCALE] Poll result: {upscaled_video_db_location}")
            dprint(f"Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} poll result: {upscaled_video_db_location}")

            if upscaled_video_db_location:
                # Path is already absolute (Supabase URL or absolute path)
                upscaled_video_abs_path: Path = Path(upscaled_video_db_location)

                if upscaled_video_abs_path.exists():
                    print(f"[STITCH UPSCALE] Upscale completed successfully: {upscaled_video_abs_path}")
                    dprint(f"Stitch: Upscale sub-task {upscale_sub_task_id} completed. Output: {upscaled_video_abs_path}")
                    
                    # Analyze upscaled result
                    try:
                        upscaled_frame_count, upscaled_fps = get_video_frame_count_and_fps(str(upscaled_video_abs_path))
                        print(f"[STITCH UPSCALE] Upscaled result: {upscaled_frame_count} frames @ {upscaled_fps} FPS")
                        dprint(f"[DEBUG] Post-upscale analysis: {upscaled_frame_count} frames, {upscaled_fps} FPS")
                        
                        # Compare frame counts
                        if upscaled_frame_count != original_frames_count:
                            print(f"[STITCH UPSCALE] Frame count changed during upscale: {original_frames_count} → {upscaled_frame_count}")
                    except (OSError, ValueError, RuntimeError) as e_post_upscale:
                        print(f"[WARNING] Could not analyze upscaled video: {e_post_upscale}")
                    
                    video_path_after_optional_upscale = upscaled_video_abs_path
                    
                    if not full_orchestrator_payload.get("skip_cleanup_enabled", False) and \
                       not full_orchestrator_payload.get("debug_mode_enabled", False) and \
                       current_stitched_video_path.exists() and current_stitched_video_path != video_path_after_optional_upscale:
                        try:
                            current_stitched_video_path.unlink()
                            dprint(f"Stitch: Removed non-upscaled video {current_stitched_video_path} after successful upscale.")
                        except OSError as e_del_non_upscaled:
                            dprint(f"Stitch: Warning - could not remove non-upscaled video {current_stitched_video_path}: {e_del_non_upscaled}")
                else: 
                    print(f"[STITCH UPSCALE] ERROR: Upscale output missing at {upscaled_video_abs_path}. Using non-upscaled video.")
                    print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} output missing ({upscaled_video_abs_path}). Using non-upscaled video.")
            else: 
                print(f"[STITCH UPSCALE] ERROR: Upscale sub-task failed or timed out. Using non-upscaled video.")
                print(f"[WARNING] Stitch Task {stitch_task_id_str}: Upscale sub-task {upscale_sub_task_id} failed or timed out. Using non-upscaled video.")

        elif upscale_factor > 1.0 and not upscale_model_name:
            print(f"[STITCH UPSCALE] Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")
            dprint(f"Stitch: Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.")
        else:
            print(f"[STITCH UPSCALE] No upscaling requested (factor: {upscale_factor})")
            dprint(f"Stitch: No upscaling (factor: {upscale_factor})")

        # Use consistent UUID-based naming for final video
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        if upscale_factor > 1.0:
            final_video_filename = f"{stitch_task_id_str}_upscaled_{upscale_factor:.1f}x_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        else:
            final_video_filename = f"{stitch_task_id_str}_output_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        
        final_video_path, initial_db_location = prepare_output_path_with_upload(
            task_id=stitch_task_id_str,
            filename=final_video_filename,
            main_output_dir_base=stitch_processing_dir,
            task_type="travel_stitch",
            dprint=dprint
        )
        
        # Move the video to final location if it's not already there
        if video_path_after_optional_upscale.resolve() != final_video_path.resolve():
            dprint(f"Stitch Task {stitch_task_id_str}: Moving {video_path_after_optional_upscale} to {final_video_path}")
            shutil.move(str(video_path_after_optional_upscale), str(final_video_path))
        else:
            dprint(f"Stitch Task {stitch_task_id_str}: Video already at final destination {final_video_path}")
        
        # Handle Supabase upload (if configured) and get final location for DB
        final_video_location_for_db = upload_and_get_final_output_location(
            final_video_path,
            final_video_filename,  # Pass only the filename to avoid redundant subfolder
            initial_db_location,
            dprint=dprint
        )
        
        travel_logger.info(f"Stitch complete: Final video saved to {final_video_path}", task_id=stitch_task_id_str)
        dprint(f"Stitch Task {stitch_task_id_str}: Final video saved to: {final_video_path} (DB location: {final_video_location_for_db})")
        
        # Analyze final result
        try:
            final_frame_count, final_fps = get_video_frame_count_and_fps(str(final_video_path))
            final_duration = final_frame_count / final_fps if final_fps > 0 else 0
            print(f"[STITCH FINAL] Final video: {final_frame_count} frames @ {final_fps} FPS = {final_duration:.2f}s")
            print(f"[STITCH_FINAL_ANALYSIS] Complete stitching analysis:")
            print(f"[STITCH_FINAL_ANALYSIS]   Input segments: {len(segment_video_paths_for_stitch)}")
            print(f"[STITCH_FINAL_ANALYSIS]   Overlap settings: {expanded_frame_overlaps}")
            # Calculate expected final length for analysis
            try:
                # Ground-truth expected length: compute from the actual decoded segment frame counts.
                # This avoids misleading results when orchestrator payload contains expanded/per-frame arrays.
                actual_segment_counts = []
                for p in segment_video_paths_for_stitch:
                    try:
                        fc, _fps = get_video_frame_count_and_fps(str(p))
                        if fc is None:
                            continue
                        actual_segment_counts.append(int(fc))
                    except (OSError, ValueError, RuntimeError):
                        continue

                if actual_segment_counts:
                    total_input_frames = sum(actual_segment_counts)
                    total_overlaps = sum(expanded_frame_overlaps) if expanded_frame_overlaps else 0
                    expected_final_length = total_input_frames - total_overlaps
                    print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames (from actual segments): {expected_final_length}")
                    print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
                    if final_frame_count != expected_final_length:
                        print(f"[STITCH_FINAL_ANALYSIS]   ⚠️  FINAL LENGTH MISMATCH! Expected {expected_final_length}, got {final_frame_count}")
                else:
                    print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: Not available (could not count input segments)")
                    print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
            except (OSError, ValueError, RuntimeError, TypeError) as e:
                print(f"[STITCH_FINAL_ANALYSIS]   Expected final frames: Not calculated ({e})")
                print(f"[STITCH_FINAL_ANALYSIS]   Actual final frames: {final_frame_count}")
            
            # Detailed analysis of the final video
            debug_video_analysis(final_video_path, "FINAL_STITCHED_VIDEO", stitch_task_id_str)
            
            dprint(f"[DEBUG] Final video analysis: {final_frame_count} frames, {final_fps} FPS, {final_duration:.2f}s duration")
        except (OSError, ValueError, RuntimeError) as e_final_analysis:
            print(f"[WARNING] Could not analyze final video: {e_final_analysis}")
        
        # Note: Individual segments already have banner overlays applied when show_input_images is enabled,
        # so the stitched video will automatically include them. No additional overlay needed here.
        
        stitch_success = True

        # --- Cleanup Downloaded Segment Files ---
        cleanup_enabled = (
            not full_orchestrator_payload.get("skip_cleanup_enabled", False) and
            not full_orchestrator_payload.get("debug_mode_enabled", False) and
            not db_ops.debug_mode
        )

        if cleanup_enabled:
            files_cleaned = 0
            total_size_cleaned = 0

            for video_path_str in segment_video_paths_for_stitch:
                video_path = Path(video_path_str)

                # Skip the initial continued video (not downloaded)
                if (initial_continued_video_path_str and
                    str(video_path.resolve()) == str(Path(initial_continued_video_path_str).resolve())):
                    continue

                # Only delete files in our processing directory (downloaded files)
                if video_path.exists() and stitch_processing_dir in video_path.parents:
                    try:
                        file_size = video_path.stat().st_size
                        video_path.unlink()
                        files_cleaned += 1
                        total_size_cleaned += file_size
                        dprint(f"Stitch: Cleaned up downloaded segment {video_path.name} ({file_size:,} bytes)")
                    except OSError as e_cleanup:
                        dprint(f"Stitch: Failed to clean up {video_path}: {e_cleanup}")

            if files_cleaned > 0:
                print(f"[STITCH_CLEANUP] Removed {files_cleaned} downloaded files ({total_size_cleaned:,} bytes)")
        else:
            print(f"[STITCH_CLEANUP] Skipping cleanup (debug mode or cleanup disabled)")

        # Note: Orchestrator will be marked complete by worker.py after stitch upload completes
        # This ensures the orchestrator gets the final storage URL, not a local path

        # Return the final video path so the stitch task itself gets uploaded via Edge Function
        log_ram_usage("Stitch end (success)", task_id=stitch_task_id_str)
        return stitch_success, str(final_video_path.resolve())

    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as e:
        travel_logger.error(f"Stitch: Unexpected error during stitching: {e}", task_id=stitch_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=stitch_task_id_str)
        
        # Notify orchestrator of stitch failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                error_msg = f"Stitch task failed: {str(e)[:200]}"
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg
                )
                dprint(f"Stitch: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to exception")
            except (RuntimeError, ValueError, OSError) as e_orch:
                dprint(f"Stitch: Warning - could not update orchestrator status: {e_orch}")

        log_ram_usage("Stitch end (error)", task_id=stitch_task_id_str)
        return False, f"Stitch task failed: {str(e)[:200]}"
