"""FFmpeg-based video operations: creation, FPS conversion, frame extraction to video."""
import subprocess
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np

from source.media.video.video_info import (
    get_video_frame_count_and_fps,
    get_video_frame_count_ffprobe,
)

__all__ = [
    "ensure_video_fps",
    "extract_frame_range_to_video",
    "create_video_from_frames_list",
    "apply_saturation_to_video_ffmpeg",
]


def ensure_video_fps(
    video_path: str | Path,
    target_fps: float,
    output_dir: str | Path | None = None,
    fps_tolerance: float = 0.5,
    *,
    dprint_func=print
) -> Path | None:
    """
    Ensure a video is at the target FPS, resampling if necessary.

    This is important when frame indices are calculated for a specific FPS
    (e.g., from frontend) but the actual video file may be at a different FPS.

    Args:
        video_path: Path to source video
        target_fps: Desired FPS (e.g., 16)
        output_dir: Directory for resampled video (defaults to same dir as source)
        fps_tolerance: Maximum FPS difference before resampling (default 0.5)
        dprint_func: Debug print function

    Returns:
        Path to video at target FPS (original if already correct, resampled if not),
        or None on error

    Example:
        # Ensure video is at 16fps before frame-based extraction
        video_16fps = ensure_video_fps(downloaded_video, 16, work_dir)
        if video_16fps:
            extract_frame_range_to_video(video_16fps, output, 0, 252, 16)
    """
    video_path = Path(video_path)

    # Validate target_fps is not None
    if target_fps is None:
        dprint_func(f"[ENSURE_FPS_ERROR] target_fps is None, defaulting to 16")
        target_fps = 16

    if not video_path.exists():
        dprint_func(f"[ENSURE_FPS_ERROR] Video does not exist: {video_path}")
        return None

    # Get actual fps
    actual_frames, actual_fps = get_video_frame_count_and_fps(str(video_path))
    if not actual_fps:
        dprint_func(f"[ENSURE_FPS_ERROR] Could not determine video FPS: {video_path}")
        return None

    # Check if resampling is needed
    if abs(actual_fps - target_fps) <= fps_tolerance:
        dprint_func(f"[ENSURE_FPS] Video already at target FPS: {actual_fps} fps (target: {target_fps}, tolerance: \u00b1{fps_tolerance})")
        return video_path

    # Need to resample
    dprint_func(f"[ENSURE_FPS] \u26a0\ufe0f  FPS mismatch: actual={actual_fps}, target={target_fps}")
    dprint_func(f"[ENSURE_FPS] Resampling {video_path.name} to {target_fps} fps...")

    # Determine output path
    if output_dir is None:
        output_dir = video_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_path = output_dir / f"{video_path.stem}_resampled_{int(target_fps)}fps.mp4"

    # Use fps filter (not -r output option) for accurate frame-based resampling
    # The fps filter selects frames based on timestamps, ensuring frame N = time N/fps
    # This is critical for frame-accurate extraction where frame indices must match timestamps
    resample_cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f'fps={target_fps}',
        '-an',  # No audio for now
        str(resampled_path)
    ]

    try:
        result = subprocess.run(resample_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            dprint_func(f"[ENSURE_FPS_ERROR] FFmpeg resample failed: {result.stderr[:500]}")
            return None

        if not resampled_path.exists() or resampled_path.stat().st_size == 0:
            dprint_func(f"[ENSURE_FPS_ERROR] Resampled video missing or empty")
            return None

        # Verify result
        new_frames, new_fps = get_video_frame_count_and_fps(str(resampled_path))
        dprint_func(f"[ENSURE_FPS] \u2705 Resampled: {new_frames} frames @ {new_fps} fps")

        return resampled_path

    except subprocess.TimeoutExpired:
        dprint_func(f"[ENSURE_FPS_ERROR] FFmpeg resample timeout")
        return None
    except (subprocess.SubprocessError, OSError) as e:
        dprint_func(f"[ENSURE_FPS_ERROR] Exception: {e}")
        return None


def extract_frame_range_to_video(
    source_video: str | Path,
    output_path: str | Path,
    start_frame: int,
    end_frame: int | None,
    fps: float,
    *,
    dprint_func=print
) -> Path | None:
    """
    Extract a range of frames from a video to a new video file using FFmpeg.

    Uses FFmpeg's select filter with frame-accurate selection (not time-based).
    This is the canonical function for frame extraction - use this instead of
    inline FFmpeg commands.

    Args:
        source_video: Path to source video file
        output_path: Path for output video file
        start_frame: First frame to include (0-indexed, inclusive)
        end_frame: Last frame to include (0-indexed, inclusive), or None for all remaining frames
        fps: Output framerate
        dprint_func: Debug print function

    Returns:
        Path to extracted video, or None on failure

    Examples:
        # Extract frames 0-252 (253 frames)
        extract_frame_range_to_video(src, out, 0, 252, 16)

        # Extract frames from 13 onwards (skip first 13)
        extract_frame_range_to_video(src, out, 13, None, 16)
    """
    source_video = Path(source_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify source exists
    if not source_video.exists():
        dprint_func(f"[EXTRACT_RANGE_ERROR] Source video does not exist: {source_video}")
        return None

    # Get source properties - use ffprobe for accurate frame count
    source_frames = get_video_frame_count_ffprobe(str(source_video), dprint=dprint_func)
    _, source_fps = get_video_frame_count_and_fps(str(source_video))

    if not source_frames:
        # Fallback to OpenCV if ffprobe fails
        source_frames, source_fps = get_video_frame_count_and_fps(str(source_video))
        dprint_func(f"[EXTRACT_RANGE] \u26a0\ufe0f  Using OpenCV frame count (ffprobe failed): {source_frames}")

    if not source_frames:
        dprint_func(f"[EXTRACT_RANGE_ERROR] Could not determine source video frame count")
        return None

    # Validate range
    if start_frame < 0:
        dprint_func(f"[EXTRACT_RANGE_ERROR] start_frame cannot be negative: {start_frame}")
        return None

    if end_frame is not None and end_frame >= source_frames:
        dprint_func(f"[EXTRACT_RANGE_ERROR] end_frame {end_frame} >= source frames {source_frames}")
        return None

    # Build filter based on whether we have an end_frame
    if end_frame is not None:
        # Extract specific range: frames start_frame to end_frame (inclusive)
        filter_str = f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/FR/TB"
        expected_frames = end_frame - start_frame + 1
        range_desc = f"frames {start_frame}-{end_frame} ({expected_frames} frames)"
    else:
        # Extract from start_frame onwards (skip first start_frame frames)
        filter_str = f"select=gte(n\\,{start_frame}),setpts=N/FR/TB"
        expected_frames = source_frames - start_frame
        range_desc = f"frames {start_frame}-{source_frames-1} ({expected_frames} frames)"

    dprint_func(f"[EXTRACT_RANGE] Source: {source_video.name} ({source_frames} frames @ {source_fps} fps)")
    dprint_func(f"[EXTRACT_RANGE] Extracting: {range_desc}")

    cmd = [
        'ffmpeg', '-y',
        '-i', str(source_video),
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'slow',  # Better quality at same bitrate
        '-crf', '10',  # Near-lossless quality for intermediate files
        '-r', str(fps),
        '-an',  # No audio
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            dprint_func(f"[EXTRACT_RANGE_ERROR] FFmpeg failed: {result.stderr[:500]}")
            return None

        if not output_path.exists() or output_path.stat().st_size == 0:
            dprint_func(f"[EXTRACT_RANGE_ERROR] Output missing or empty: {output_path}")
            return None

        # Verify frame count (allow small tolerance for OpenCV/FFmpeg discrepancies)
        actual_frames, _ = get_video_frame_count_and_fps(str(output_path))
        if actual_frames is None:
            dprint_func(f"[EXTRACT_RANGE_ERROR] Could not verify output frame count")
            return None

        # Allow up to 3 frames difference - OpenCV and FFmpeg often disagree on frame counts
        # especially for end-of-video segments or variable framerate videos
        frame_diff = abs(actual_frames - expected_frames)
        if frame_diff > 3:
            dprint_func(f"[EXTRACT_RANGE_ERROR] Frame count mismatch too large: expected {expected_frames}, got {actual_frames} (diff: {frame_diff})")
            return None
        elif frame_diff > 0:
            dprint_func(f"[EXTRACT_RANGE] \u26a0\ufe0f  Minor frame count difference: expected {expected_frames}, got {actual_frames} (diff: {frame_diff}, within tolerance)")

        dprint_func(f"[EXTRACT_RANGE] \u2705 Extracted {actual_frames} frames to {output_path.name}")
        return output_path

    except subprocess.TimeoutExpired:
        dprint_func(f"[EXTRACT_RANGE_ERROR] FFmpeg timeout")
        return None
    except (subprocess.SubprocessError, OSError) as e:
        dprint_func(f"[EXTRACT_RANGE_ERROR] Exception: {e}")
        return None


def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int],
    *,
    dprint=None,
    standardize_colorspace: bool = False
) -> Path | None:
    """Creates a video from a list of NumPy BGR frames using FFmpeg subprocess.

    Uses streaming to pipe frames to FFmpeg incrementally, avoiding loading
    all frame data into memory at once. This is critical for large videos
    (e.g., 2000+ frames at 1080p would otherwise require 30+ GB RAM).

    Args:
        frames_list: List of BGR numpy arrays
        output_path: Output video path
        fps: Frames per second
        resolution: (width, height) tuple
        dprint: Optional logging function (defaults to print)
        standardize_colorspace: If True, adds BT.709 colorspace standardization

    Returns the Path object of the successfully written file, or None if failed.
    """
    # Use print if no dprint provided
    log = dprint if dprint else print

    output_path_obj = Path(output_path)
    output_path_mp4 = output_path_obj.with_suffix('.mp4')
    output_path_mp4.parent.mkdir(parents=True, exist_ok=True)

    log(f"[CREATE_VIDEO] Creating video: {output_path_mp4}")
    log(f"[CREATE_VIDEO] Input: {len(frames_list)} frames, resolution={resolution}, fps={fps}")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{resolution[0]}x{resolution[1]}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",  # Better quality at same bitrate
        "-crf", "10",  # Near-lossless quality for intermediate files
    ]

    # Add colorspace standardization if requested
    if standardize_colorspace:
        ffmpeg_cmd.extend([
            "-vf", "format=yuv420p,colorspace=bt709:iall=bt709:fast=1",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
        ])

    ffmpeg_cmd.append(str(output_path_mp4.resolve()))

    # Count valid frames first (without storing them)
    valid_count = 0
    skipped_none = 0
    skipped_invalid = 0
    for frame_np in frames_list:
        if frame_np is None or not isinstance(frame_np, np.ndarray):
            skipped_none += 1
        elif len(frame_np.shape) != 3 or frame_np.shape[2] != 3:
            skipped_invalid += 1
        else:
            valid_count += 1

    if skipped_none > 0:
        log(f"[CREATE_VIDEO] WARNING: Will skip {skipped_none} None/invalid frames")
    if skipped_invalid > 0:
        log(f"[CREATE_VIDEO] WARNING: Will skip {skipped_invalid} non-RGB frames")

    if valid_count == 0:
        log(f"[CREATE_VIDEO] ERROR: No valid frames to process! Input had {len(frames_list)} frames, all invalid")
        return None

    log(f"[CREATE_VIDEO] Streaming {valid_count} frames to FFmpeg (memory-efficient mode)")

    try:
        # Use Popen to stream frames incrementally instead of loading all into memory
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        frames_written = 0
        resize_warnings = 0

        for frame_idx, frame_np in enumerate(frames_list):
            # Skip invalid frames
            if frame_np is None or not isinstance(frame_np, np.ndarray):
                continue
            if len(frame_np.shape) != 3 or frame_np.shape[2] != 3:
                continue

            # Ensure uint8
            if frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)

            # Resize if needed
            if frame_np.shape[0] != resolution[1] or frame_np.shape[1] != resolution[0]:
                try:
                    frame_np = cv2.resize(frame_np, resolution, interpolation=cv2.INTER_AREA)
                except (OSError, ValueError, RuntimeError) as e:
                    if resize_warnings < 5:
                        log(f"[CREATE_VIDEO] WARNING: Failed to resize frame {frame_idx}: {e}")
                    resize_warnings += 1
                    continue

            # Write frame bytes directly to FFmpeg stdin
            try:
                proc.stdin.write(frame_np.tobytes())
                frames_written += 1
            except BrokenPipeError:
                log(f"[CREATE_VIDEO] ERROR: FFmpeg pipe broken after {frames_written} frames")
                break

        if resize_warnings > 5:
            log(f"[CREATE_VIDEO] WARNING: {resize_warnings} total resize failures (only first 5 logged)")

        # Close stdin to signal end of input
        proc.stdin.close()

        log(f"[CREATE_VIDEO] Wrote {frames_written} frames to FFmpeg, waiting for encoding...")

        # Read stderr in background thread to avoid blocking
        stderr_output = []
        def read_stderr():
            try:
                stderr_output.append(proc.stderr.read())
            except OSError as e:
                log(f"[CREATE_VIDEO] DEBUG: Failed to read FFmpeg stderr: {e}")

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.start()

        try:
            proc.wait(timeout=300)  # 5 minute timeout for encoding
        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"[CREATE_VIDEO] ERROR: FFmpeg timed out after 300 seconds")
            return None
        finally:
            stderr_thread.join(timeout=5)

        stderr = stderr_output[0] if stderr_output else b""

        if proc.returncode == 0:
            if output_path_mp4.exists() and output_path_mp4.stat().st_size > 0:
                log(f"[CREATE_VIDEO] SUCCESS: Created {output_path_mp4} ({output_path_mp4.stat().st_size} bytes)")
                return output_path_mp4
            log(f"[CREATE_VIDEO] ERROR: FFmpeg succeeded but output file missing or empty")
            return None
        else:
            stderr_str = stderr.decode('utf-8', errors='replace') if stderr else "no stderr"
            log(f"[CREATE_VIDEO] ERROR: FFmpeg failed with return code {proc.returncode}")
            log(f"[CREATE_VIDEO] FFmpeg stderr: {stderr_str[:500]}")
            if output_path_mp4.exists():
                try:
                    output_path_mp4.unlink()
                except OSError as e:
                    log(f"[CREATE_VIDEO] DEBUG: Failed to clean up partial output file {output_path_mp4}: {e}")
            return None

    except FileNotFoundError:
        log(f"[CREATE_VIDEO] ERROR: FFmpeg not found - is it installed?")
        return None
    except (subprocess.SubprocessError, OSError) as e:
        log(f"[CREATE_VIDEO] ERROR: Unexpected exception: {e}")
        traceback.print_exc()
        return None

    # If we exhausted all retries without returning, return None
    return None


def apply_saturation_to_video_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    saturation_level: float,
    preset: str = "medium"
) -> bool:
    """Applies a saturation adjustment to the full video using FFmpeg's eq filter.
    Returns: True if FFmpeg succeeds and the output file exists & is non-empty, else False.
    """
    inp = Path(input_video_path)
    outp = Path(output_video_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp.resolve()),
        "-vf", f"eq=saturation={saturation_level}",
        "-c:v", "libx264",
        "-crf", "10",  # Near-lossless quality for intermediate files
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",
        str(outp.resolve())
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        if outp.exists() and outp.stat().st_size > 0:
            return True
        return False
    except subprocess.CalledProcessError:
        return False
