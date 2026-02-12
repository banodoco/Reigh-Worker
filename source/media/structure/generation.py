"""Structure guidance video generation and compositing."""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import traceback
import torch

from source.core.constants import BYTES_PER_MB
from source.media.structure.loading import load_structure_video_frames, _resample_frame_indices
from source.media.structure.preprocessors import process_structure_frames


def create_structure_motion_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    motion_strength: float,
    output_path: Path,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path:
    """
    Create a video of flow visualizations from the structure video.

    This is the orchestrator-level function that:
    1. Extracts optical flows from the structure video
    2. Generates colorful flow visualizations (rainbow images showing motion)
    3. Encodes them as an H.264 video

    The resulting video contains flow visualizations that segments can use as
    VACE guide videos for motion conditioning, matching the format used by
    FlowVisAnnotator in wgp.py.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate (total unguidanced frames)
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        motion_strength: Unused (kept for compatibility)
        output_path: Where to save the output video
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        dprint: Debug print function

    Returns:
        Path to the created video file

    Raises:
        ValueError: If structure video cannot be loaded or processed
    """
    dprint(f"[STRUCTURE_MOTION_VIDEO] Creating flow visualization video...")
    dprint(f"  Source: {structure_video_path}")
    dprint(f"  Frames: {max_frames_needed}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Treatment: {treatment}")

    try:
        # Step 1: Load structure video frames with treatment mode
        dprint(f"[STRUCTURE_MOTION_VIDEO] Loading structure video frames...")
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,  # Apply center cropping
            dprint=dprint
        )

        # Step 2: Extract optical flow visualizations
        dprint(f"[STRUCTURE_MOTION_VIDEO] Extracting optical flow visualizations...")
        flow_fields, flow_vis = extract_optical_flow_from_frames(structure_frames, dprint=dprint)

        if not flow_vis:
            raise ValueError("No optical flow visualizations extracted from structure video")

        dprint(f"[STRUCTURE_MOTION_VIDEO] Extracted {len(flow_vis)} flow visualizations")

        # Step 3: Duplicate first flow visualization to match frame count
        # This matches FlowVisAnnotator behavior: for N frames, return N visualizations
        # by duplicating the first one (flow_vis has N-1 items for N frames)
        dprint(f"[STRUCTURE_MOTION_VIDEO] Preparing flow visualization frames...")

        # Duplicate first visualization (matching FlowVisAnnotator pattern)
        motion_frames = [flow_vis[0]] + flow_vis

        dprint(f"[STRUCTURE_MOTION_VIDEO] Prepared {len(motion_frames)} flow visualization frames")

        # Step 4: Encode as video
        dprint(f"[STRUCTURE_MOTION_VIDEO] Encoding video to {output_path}")

        # Import video creation utilities
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from shared.utils.audio_video import save_video

        # Convert to torch tensor format expected by save_video
        # save_video expects [T, H, W, C] in range [0, 255]
        video_tensor = np.stack(motion_frames, axis=0)  # [T, H, W, C]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save video using WGP's video utilities
        # Note: save_video expects numpy array or torch tensor
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',  # Use libx264 with 8-bit encoding
            normalize=False,  # Already in [0, 255] uint8 range
            value_range=(0, 255)  # Specify value range for uint8 data
        )

        # Verify output exists
        if not output_path.exists():
            raise ValueError(f"Failed to create video at {output_path}")

        file_size_mb = output_path.stat().st_size / BYTES_PER_MB
        dprint(f"[STRUCTURE_MOTION_VIDEO] Created video: {output_path.name} ({file_size_mb:.2f} MB)")

        return output_path

    except (OSError, ValueError, RuntimeError) as e:
        dprint(f"[ERROR] Failed to create structure motion video: {e}")
        traceback.print_exc()
        raise


def create_structure_guidance_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    structure_type: str = "flow",
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path:
    """
    Create a video of preprocessed structure visualizations from the structure video.

    This is the NEW orchestrator-level function that:
    1. Loads and preprocesses frames from the structure video
    2. Applies the chosen preprocessor (flow, canny, depth, or raw)
    3. Encodes them as an H.264 video

    The resulting video contains structure visualizations that segments can use as
    VACE guide videos for structural conditioning, or raw frames for Uni3C.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate (total unguidanced frames)
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        output_path: Where to save the output video
        structure_type: Type of preprocessing:
            - "flow": Optical flow visualization (VACE)
            - "canny": Edge detection (VACE)
            - "depth": Depth map estimation (VACE)
            - "raw": No preprocessing - raw frames only (Uni3C)
            Default: "flow"
        motion_strength: Flow strength multiplier (only used for flow, also maps to uni3c_strength for raw)
        canny_intensity: Edge intensity multiplier (only used for canny)
        depth_contrast: Depth contrast adjustment (only used for depth)
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        dprint: Debug print function

    Returns:
        Path to the created video file

    Raises:
        ValueError: If structure video cannot be loaded or processed
    """
    dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Creating {structure_type} visualization video...")
    dprint(f"  Source: {structure_video_path}")
    dprint(f"  Type: {structure_type}")
    dprint(f"  Frames: {max_frames_needed}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Treatment: {treatment}")

    # Log active strength parameter
    if structure_type == "flow" and abs(motion_strength - 1.0) > 1e-6:
        dprint(f"  Motion strength: {motion_strength}")
    elif structure_type == "canny" and abs(canny_intensity - 1e-6) > 1e-6:
        dprint(f"  Canny intensity: {canny_intensity}")
    elif structure_type == "raw":
        dprint(f"  Raw frames: No preprocessing applied (Uni3C mode)")
        if abs(motion_strength - 1.0) > 1e-6:
            dprint(f"  Uni3C strength: {motion_strength} (from motion_strength)")
    elif structure_type == "depth" and abs(depth_contrast - 1.0) > 1e-6:
        dprint(f"  Depth contrast: {depth_contrast}")

    try:
        # Step 1: Load structure video frames with treatment mode
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Loading structure video frames...")
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,  # Apply center cropping
            dprint=dprint
        )

        # Step 2: Process frames with chosen preprocessor
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Processing with '{structure_type}' preprocessor...")
        processed_frames = process_structure_frames(
            structure_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
            dprint
        )

        if not processed_frames:
            raise ValueError(f"No {structure_type} visualizations extracted from structure video")

        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Processed {len(processed_frames)} frames")

        # Step 3: Encode as video
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Encoding video to {output_path}")

        # Import video creation utilities
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from shared.utils.audio_video import save_video

        # Convert to numpy array format expected by save_video
        # save_video expects [T, H, W, C] in range [0, 255]
        video_tensor = np.stack(processed_frames, axis=0)  # [T, H, W, C]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save video using WGP's video utilities
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',  # Use libx264 with 8-bit encoding
            normalize=False,  # Already in [0, 255] uint8 range
            value_range=(0, 255)  # Specify value range for uint8 data
        )

        # Verify output exists
        if not output_path.exists():
            raise ValueError(f"Failed to create video at {output_path}")

        file_size_mb = output_path.stat().st_size / BYTES_PER_MB
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Created video: {output_path.name} ({file_size_mb:.2f} MB)")

        # Clean up GPU memory
        dprint(f"[STRUCTURE_GUIDANCE_VIDEO] Cleaning up GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path

    except (OSError, ValueError, RuntimeError) as e:
        dprint(f"[ERROR] Failed to create structure guidance video: {e}")
        traceback.print_exc()
        raise


# =============================================================================
# Multi-Structure Video Compositing
# =============================================================================

def create_neutral_frame(structure_type: str, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Create a neutral frame for gaps in structure video coverage.

    These frames don't bias the generation - they represent "no guidance signal".

    IMPORTANT for Uni3C (structure_type="raw"):
        Black pixel frames are detected by _encode_uni3c_guide() in any2video.py
        and converted to zeros in latent space. This is critical because:
        - VAE-encoded black pixels != zero latents
        - Zero latents = true "no control"
        - VAE-encoded black = "control toward black output"

        The detection happens automatically during VAE encoding, so black frames
        created here will be properly handled as "no guidance".

    Args:
        structure_type: Type of structure preprocessing ("flow", "canny", "depth", "raw", or "uni3c")
        resolution: (width, height) tuple

    Returns:
        numpy array [H, W, C] uint8 RGB
    """
    w, h = resolution

    if structure_type == "flow":
        # Gray = center of HSV color wheel = no motion
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "canny":
        # Black = no edges detected
        return np.zeros((h, w, 3), dtype=np.uint8)
    elif structure_type == "depth":
        # Mid-gray = neutral depth (not close, not far)
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif structure_type == "raw":
        # Black = no structure signal for Uni3C
        # NOTE: These black frames will be detected during VAE encoding and
        # their latents will be zeroed for true "no control" behavior.
        return np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Default to black
        return np.zeros((h, w, 3), dtype=np.uint8)


def load_structure_video_frames_with_range(
    structure_video_path: str,
    target_frame_count: int,
    target_fps: int,
    target_resolution: Tuple[int, int],
    treatment: str = "adjust",
    crop_to_fit: bool = True,
    source_start_frame: int = 0,
    source_end_frame: Optional[int] = None,
    dprint: Callable = print
) -> List[np.ndarray]:
    """
    Load structure video frames with optional source range extraction.

    Extended version of load_structure_video_frames that supports extracting
    a specific range from the source video before applying treatment.

    Args:
        structure_video_path: Path to structure video
        target_frame_count: Number of output frames needed
        target_fps: Target FPS (used for clip mode)
        target_resolution: (width, height) tuple
        treatment: "adjust" (stretch/compress) or "clip" (temporal sample)
        crop_to_fit: Center-crop to match target aspect ratio
        source_start_frame: First frame to extract from source (default: 0)
        source_end_frame: Last frame (exclusive) to extract (default: None = end of video)
        dprint: Debug print function

    Returns:
        List of numpy uint8 arrays [H, W, C] in RGB format
    """
    import cv2
    from PIL import Image

    # Try decord first (faster), fall back to cv2
    use_decord = False
    try:
        import decord
        decord.bridge.set_bridge('torch')
        use_decord = True
    except ImportError:
        dprint("[STRUCTURE_VIDEO_RANGE] decord not available, using cv2 fallback")

    if use_decord:
        reader = decord.VideoReader(structure_video_path)
        video_fps = round(reader.get_avg_fps())
        total_video_frames = len(reader)
    else:
        cap = cv2.VideoCapture(structure_video_path)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolve source range
    actual_start = source_start_frame
    actual_end = source_end_frame if source_end_frame is not None else total_video_frames

    # Clamp to valid range
    actual_start = max(0, min(actual_start, total_video_frames - 1))
    actual_end = max(actual_start + 1, min(actual_end, total_video_frames))

    effective_source_frames = actual_end - actual_start

    dprint(f"[STRUCTURE_VIDEO_RANGE] Loading from source video:")
    dprint(f"  Total source frames: {total_video_frames} @ {video_fps}fps")
    dprint(f"  Source range: [{actual_start}, {actual_end}) = {effective_source_frames} frames")
    dprint(f"  Target frames needed: {target_frame_count}")
    dprint(f"  Treatment: {treatment}")

    # Load N+1 frames for flow processing (RAFT needs N frames to produce N-1 flows)
    frames_to_load = target_frame_count + 1

    # Calculate frame indices based on treatment mode
    if treatment == "adjust":
        # ADJUST: Stretch/compress source range to match needed count
        if effective_source_frames >= frames_to_load:
            # Compress: sample evenly
            frame_indices = [
                actual_start + int(i * (effective_source_frames - 1) / (frames_to_load - 1))
                for i in range(frames_to_load)
            ]
            dprint(f"  Adjust: Compressing {effective_source_frames} \u2192 {frames_to_load} frames")
        else:
            # Stretch: repeat frames
            frame_indices = [
                actual_start + int(i * (effective_source_frames - 1) / (frames_to_load - 1))
                for i in range(frames_to_load)
            ]
            dprint(f"  Adjust: Stretching {effective_source_frames} \u2192 {frames_to_load} frames")
    else:
        # CLIP: Temporal sampling from the source range
        # Calculate indices relative to source range
        frame_indices = _resample_frame_indices(
            video_fps=video_fps,
            video_frames_count=effective_source_frames,
            max_target_frames_count=frames_to_load,
            target_fps=target_fps,
            start_target_frame=0
        )
        # Offset to actual source positions
        frame_indices = [actual_start + idx for idx in frame_indices]

        # Handle if source range is too short
        if len(frame_indices) < frames_to_load:
            dprint(f"  Clip: Source range too short, looping to fill {frames_to_load} frames")
            while len(frame_indices) < frames_to_load:
                remaining = frames_to_load - len(frame_indices)
                frame_indices.extend(frame_indices[:remaining])

        dprint(f"  Clip: Extracted {len(frame_indices)} frame indices")

    if not frame_indices:
        raise ValueError(f"No frames could be extracted from range [{actual_start}, {actual_end})")

    # Extract frames
    if use_decord:
        frames = reader.get_batch(frame_indices)
        raw_frames = []
        for frame in frames:
            if hasattr(frame, 'cpu'):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = np.array(frame)
            raw_frames.append(frame_np)
    else:
        # cv2 fallback - read frames sequentially
        raw_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_frames.append(frame_rgb)
            else:
                # Fallback: use last frame or black
                if raw_frames:
                    raw_frames.append(raw_frames[-1].copy())
                else:
                    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    raw_frames.append(np.zeros((h_src, w_src, 3), dtype=np.uint8))
        cap.release()

    # Process frames to target resolution
    w, h = target_resolution
    target_aspect = w / h
    processed_frames = []

    for i, frame_np in enumerate(raw_frames):
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        frame_pil = Image.fromarray(frame_np)

        # Center crop if needed
        if crop_to_fit:
            src_w, src_h = frame_pil.size
            src_aspect = src_w / src_h

            if abs(src_aspect - target_aspect) > 0.01:
                if src_aspect > target_aspect:
                    new_w = int(src_h * target_aspect)
                    left = (src_w - new_w) // 2
                    frame_pil = frame_pil.crop((left, 0, left + new_w, src_h))
                else:
                    new_h = int(src_w / target_aspect)
                    top = (src_h - new_h) // 2
                    frame_pil = frame_pil.crop((0, top, src_w, top + new_h))

        frame_resized = frame_pil.resize((w, h), resample=Image.Resampling.LANCZOS)
        processed_frames.append(np.array(frame_resized))

    dprint(f"[STRUCTURE_VIDEO_RANGE] Loaded {len(processed_frames)} frames at {w}x{h}")

    return processed_frames


def validate_structure_video_configs(
    configs: List[dict],
    total_frames: int,
    dprint: Callable = print
) -> List[dict]:
    """
    Validate and normalize structure video configurations.

    Args:
        configs: List of structure video config dicts
        total_frames: Total frames in the output timeline
        dprint: Debug print function

    Returns:
        Sorted and validated configs

    Raises:
        ValueError: If configs are invalid or overlap
    """
    if not configs:
        return []

    # Sort by start_frame
    sorted_configs = sorted(configs, key=lambda c: c.get("start_frame", 0))

    # First pass: validate required fields and clip/filter configs
    valid_configs = []
    for i, config in enumerate(sorted_configs):
        # Check required fields
        if "path" not in config:
            raise ValueError(f"Structure video config {i} missing 'path'")
        if "start_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'start_frame'")
        if "end_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'end_frame'")

        start = config["start_frame"]
        end = config["end_frame"]

        # Validate range
        if start < 0:
            raise ValueError(f"Config {i}: start_frame {start} < 0")

        # Skip configs that start beyond video length
        if start >= total_frames:
            dprint(f"[COMPOSITE] \u26a0\ufe0f Config {i}: start_frame {start} >= total_frames {total_frames}, skipping this config")
            continue

        # Auto-clip end_frame to total_frames instead of failing
        if end > total_frames:
            dprint(f"[COMPOSITE] \u26a0\ufe0f Config {i}: end_frame {end} > total_frames {total_frames}, clipping to {total_frames}")
            end = total_frames
            config["end_frame"] = end

        if start >= end:
            raise ValueError(f"Config {i}: start_frame {start} >= end_frame {end}")

        valid_configs.append(config)

    # Second pass: check for overlaps on valid configs
    prev_end = -1
    for i, config in enumerate(valid_configs):
        start = config["start_frame"]
        end = config["end_frame"]

        if start < prev_end:
            raise ValueError(f"Config {i}: frame range [{start}, {end}) overlaps with previous config ending at {prev_end}")

        prev_end = end

    dprint(f"[COMPOSITE] Validated {len(valid_configs)} structure video configs (from {len(sorted_configs)} input)")
    for i, cfg in enumerate(valid_configs):
        dprint(f"  Config {i}: frames [{cfg['start_frame']}, {cfg['end_frame']}) from {Path(cfg['path']).name}")

    return valid_configs


def create_composite_guidance_video(
    structure_configs: List[dict],
    total_frames: int,
    structure_type: str,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    download_dir: Optional[Path] = None,
    dprint: Callable = print
) -> Path:
    """
    Create a single composite guidance video from multiple structure video sources.

    This function:
    1. Creates a timeline filled with neutral frames
    2. For each config, loads and processes source frames
    3. Places processed frames at the correct timeline positions
    4. Encodes everything as a single video

    Args:
        structure_configs: List of config dicts, each with:
            - path: Source video path/URL
            - start_frame: Start position in output timeline
            - end_frame: End position (exclusive) in output timeline
            - treatment: Optional "adjust" or "clip" (default: "adjust")
            - source_start_frame: Optional start frame in source video
            - source_end_frame: Optional end frame in source video
        total_frames: Total frames in the output timeline
        structure_type: "flow", "canny", "depth", "raw", or "uni3c"
        target_resolution: (width, height) tuple
        target_fps: Output video FPS
        motion_strength: Flow motion strength
        canny_intensity: Canny edge intensity
        depth_contrast: Depth map contrast
        download_dir: Directory for downloading source videos
        dprint: Debug print function

    Returns:
        Path to the created composite guidance video
    """
    dprint(f"[COMPOSITE] Creating composite guidance video...")
    dprint(f"  Total frames: {total_frames}")
    dprint(f"  Structure type: {structure_type}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Configs: {len(structure_configs)}")

    # Validate configs
    sorted_configs = validate_structure_video_configs(structure_configs, total_frames, dprint)

    if not sorted_configs:
        raise ValueError("No valid structure video configs provided")

    # Initialize timeline with neutral frames
    dprint(f"[COMPOSITE] Initializing {total_frames} neutral frames...")
    neutral_frame = create_neutral_frame(structure_type, target_resolution)
    composite_frames = [neutral_frame.copy() for _ in range(total_frames)]

    # Track which frame ranges are filled
    filled_ranges = []

    # Process each config
    for config_idx, config in enumerate(sorted_configs):
        source_path = config["path"]
        start_frame = config["start_frame"]
        end_frame = config["end_frame"]
        frames_needed = end_frame - start_frame
        treatment = config.get("treatment", "adjust")

        dprint(f"[COMPOSITE] Processing config {config_idx}: {Path(source_path).name}")
        dprint(f"  Timeline range: [{start_frame}, {end_frame}) = {frames_needed} frames")

        # Download if URL
        if source_path.startswith(("http://", "https://")):
            if download_dir is None:
                download_dir = Path("./temp_structure_downloads")
            download_dir.mkdir(parents=True, exist_ok=True)

            import requests
            local_filename = f"structure_src_{config_idx}_{Path(source_path).name}"
            local_path = download_dir / local_filename

            if not local_path.exists():
                dprint(f"  Downloading: {source_path}")
                response = requests.get(source_path, timeout=120)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                dprint(f"  Downloaded: {local_path.name} ({len(response.content) / BYTES_PER_MB:.2f} MB)")
            else:
                dprint(f"  Using cached: {local_path.name}")

            source_path = str(local_path)

        # Load source frames with optional range extraction
        source_frames = load_structure_video_frames_with_range(
            structure_video_path=source_path,
            target_frame_count=frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,
            source_start_frame=config.get("source_start_frame", 0),
            source_end_frame=config.get("source_end_frame"),
            dprint=dprint
        )

        # Process frames with chosen preprocessor
        dprint(f"  Processing with '{structure_type}' preprocessor...")
        processed_frames = process_structure_frames(
            source_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
            dprint
        )

        # Ensure we have exactly the frames needed
        if len(processed_frames) > frames_needed:
            processed_frames = processed_frames[:frames_needed]
            dprint(f"  Trimmed to {frames_needed} frames")
        elif len(processed_frames) < frames_needed:
            # Pad with last frame if short
            while len(processed_frames) < frames_needed:
                processed_frames.append(processed_frames[-1].copy())
            dprint(f"  Padded to {frames_needed} frames")

        # Place processed frames into composite
        for i, frame in enumerate(processed_frames):
            frame_idx = start_frame + i
            if 0 <= frame_idx < total_frames:
                composite_frames[frame_idx] = frame

        filled_ranges.append((start_frame, end_frame))
        dprint(f"  Placed {len(processed_frames)} frames at positions {start_frame}-{end_frame-1}")

    # Log coverage summary
    total_filled = sum(end - start for start, end in filled_ranges)
    total_neutral = total_frames - total_filled
    dprint(f"[COMPOSITE] Coverage summary:")
    dprint(f"  Filled frames: {total_filled} ({100*total_filled/total_frames:.1f}%)")
    dprint(f"  Neutral frames: {total_neutral} ({100*total_neutral/total_frames:.1f}%)")

    # Encode as video
    dprint(f"[COMPOSITE] Encoding composite video to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try WGP's save_video first, fall back to cv2
    try:
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from shared.utils.audio_video import save_video

        video_tensor = np.stack(composite_frames, axis=0)  # [T, H, W, C]

        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',
            normalize=False,
            value_range=(0, 255)
        )
    except ImportError:
        # cv2 fallback for encoding
        import cv2
        dprint("[COMPOSITE] Using cv2 fallback for video encoding")

        w, h = target_resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))

        for frame_rgb in composite_frames:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

    if not output_path.exists():
        raise ValueError(f"Failed to create composite video at {output_path}")

    file_size_mb = output_path.stat().st_size / BYTES_PER_MB
    dprint(f"[COMPOSITE] Created composite video: {output_path.name} ({file_size_mb:.2f} MB)")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def create_trimmed_structure_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    treatment: str = "adjust",
    dprint: Callable = print
) -> Path:
    """
    Create a trimmed/adjusted version of the structure video without applying any style transfer.
    This preserves the original video content but clips/stretches it to match the generation length.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        output_path: Where to save the output video
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)
        dprint: Debug print function

    Returns:
        Path to the created video file
    """
    dprint(f"[TRIMMED_STRUCTURE_VIDEO] Creating trimmed structure video...")
    dprint(f"  Source: {structure_video_path}")
    dprint(f"  Frames: {max_frames_needed}")
    dprint(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    dprint(f"  Treatment: {treatment}")

    try:
        # Step 1: Load structure video frames with treatment mode
        # This handles the trimming/adjusting logic
        frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,
            dprint=dprint
        )

        # Step 2: Encode as video directly (no style transfer)
        dprint(f"[TRIMMED_STRUCTURE_VIDEO] Encoding video to {output_path}")

        # Import video creation utilities
        wan_dir = Path(__file__).parent.parent.parent / "Wan2GP"
        if str(wan_dir) not in sys.path:
            sys.path.insert(0, str(wan_dir))

        from shared.utils.audio_video import save_video

        # Convert list of numpy arrays [H, W, C] to tensor [T, H, W, C]
        video_tensor = np.stack(frames, axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save video using WGP's video utilities
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',
            normalize=False,
            value_range=(0, 255)
        )

        return output_path

    except (OSError, ValueError, RuntimeError) as e:
        dprint(f"[TRIMMED_STRUCTURE_VIDEO] Error: {e}")
        raise
