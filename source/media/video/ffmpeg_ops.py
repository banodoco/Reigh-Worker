"""FFmpeg-based video operations: creation, FPS conversion, frame extraction to video."""
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from source.core.log import generation_logger
from source.media.video.video_info import (
    get_video_frame_count_and_fps,
    get_video_frame_count_ffprobe,
)

__all__ = [
    "ensure_video_fps",
    "extract_frame_range_to_video",
    "create_video_from_frames_list",
    "apply_saturation_to_video_ffmpeg",
    "video_has_audio",
    "mux_audio_from_segments",
]


def ensure_video_fps(
    input_video_path: str | Path,
    target_fps: float,
    output_dir: str | Path | None = None,
    fps_tolerance: float = 0.5,
) -> Path:
    """
    Ensure a video is at the target FPS, resampling if necessary.

    This is important when frame indices are calculated for a specific FPS
    (e.g., from frontend) but the actual video file may be at a different FPS.

    Args:
        input_video_path: Path to source video
        target_fps: Desired FPS (e.g., 16)
        output_dir: Directory for resampled video (defaults to same dir as source)
        fps_tolerance: Maximum FPS difference before resampling (default 0.5)

    Returns:
        Path to video at target FPS (original if already correct, resampled if not)

    Raises:
        OSError: If the source video does not exist or the resampled output is missing/empty
        RuntimeError: If FFmpeg fails or times out
        ValueError: If the video FPS cannot be determined

    Example:
        # Ensure video is at 16fps before frame-based extraction
        video_16fps = ensure_video_fps(downloaded_video, 16, work_dir)
        extract_frame_range_to_video(video_16fps, output, 0, 252, 16)
    """
    input_video_path = Path(input_video_path)

    # Validate target_fps is not None
    if target_fps is None:
        generation_logger.warning(f"[ENSURE_FPS] target_fps is None, defaulting to 16")
        target_fps = 16

    if not input_video_path.exists():
        raise OSError(f"Video does not exist: {input_video_path}")

    # Get actual fps
    actual_frames, actual_fps = get_video_frame_count_and_fps(str(input_video_path))
    if not actual_fps:
        raise ValueError(f"Could not determine video FPS: {input_video_path}")

    # Check if resampling is needed
    if abs(actual_fps - target_fps) <= fps_tolerance:
        generation_logger.debug(f"[ENSURE_FPS] Video already at target FPS: {actual_fps} fps (target: {target_fps}, tolerance: \u00b1{fps_tolerance})")
        return input_video_path

    # Need to resample
    generation_logger.debug(f"[ENSURE_FPS] FPS mismatch: actual={actual_fps}, target={target_fps}")
    generation_logger.debug(f"[ENSURE_FPS] Resampling {input_video_path.name} to {target_fps} fps...")

    # Determine output path
    if output_dir is None:
        output_dir = input_video_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_path = output_dir / f"{input_video_path.stem}_resampled_{int(target_fps)}fps.mp4"

    # Use fps filter (not -r output option) for accurate frame-based resampling
    # The fps filter selects frames based on timestamps, ensuring frame N = time N/fps
    # This is critical for frame-accurate extraction where frame indices must match timestamps
    resample_cmd = [
        'ffmpeg', '-y',
        '-i', str(input_video_path),
        '-vf', f'fps={target_fps}',
        '-an',  # No audio for now
        str(resampled_path)
    ]

    try:
        result = subprocess.run(resample_cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"FFmpeg resample timed out after 600s for {input_video_path}") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"FFmpeg resample failed for {input_video_path}: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg resample failed (rc={result.returncode}): {result.stderr[:500]}")

    if not resampled_path.exists() or resampled_path.stat().st_size == 0:
        raise OSError(f"Resampled video missing or empty: {resampled_path}")

    # Verify result
    new_frames, new_fps = get_video_frame_count_and_fps(str(resampled_path))
    generation_logger.debug(f"[ENSURE_FPS] Resampled: {new_frames} frames @ {new_fps} fps")

    return resampled_path


def extract_frame_range_to_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    start_frame: int,
    end_frame: int | None,
    fps: float,
) -> Path:
    """
    Extract a range of frames from a video to a new video file using FFmpeg.

    Uses FFmpeg's select filter with frame-accurate selection (not time-based).
    This is the canonical function for frame extraction - use this instead of
    inline FFmpeg commands.

    Args:
        input_video_path: Path to source video file
        output_video_path: Path for output video file
        start_frame: First frame to include (0-indexed, inclusive)
        end_frame: Last frame to include (0-indexed, inclusive), or None for all remaining frames
        fps: Output framerate

    Returns:
        Path to extracted video

    Raises:
        OSError: If source video does not exist or output is missing/empty
        ValueError: If frame range is invalid or frame count cannot be determined
        RuntimeError: If FFmpeg fails, times out, or output frame count is wrong

    Examples:
        # Extract frames 0-252 (253 frames)
        extract_frame_range_to_video(src, out, 0, 252, 16)

        # Extract frames from 13 onwards (skip first 13)
        extract_frame_range_to_video(src, out, 13, None, 16)
    """
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify source exists
    if not input_video_path.exists():
        raise OSError(f"Source video does not exist: {input_video_path}")

    # Get source properties - use ffprobe for accurate frame count
    source_frames = get_video_frame_count_ffprobe(str(input_video_path))
    _, source_fps = get_video_frame_count_and_fps(str(input_video_path))

    if not source_frames:
        # Fallback to OpenCV if ffprobe fails
        source_frames, source_fps = get_video_frame_count_and_fps(str(input_video_path))
        generation_logger.warning(f"[EXTRACT_RANGE] Using OpenCV frame count (ffprobe failed): {source_frames}")

    if not source_frames:
        raise ValueError(f"Could not determine source video frame count: {input_video_path}")

    # Validate range
    if start_frame < 0:
        raise ValueError(f"start_frame cannot be negative: {start_frame}")

    if end_frame is not None and end_frame >= source_frames:
        raise ValueError(f"end_frame {end_frame} >= source frames {source_frames}")

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

    generation_logger.debug(f"[EXTRACT_RANGE] Source: {input_video_path.name} ({source_frames} frames @ {source_fps} fps)")
    generation_logger.debug(f"[EXTRACT_RANGE] Extracting: {range_desc}")

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_video_path),
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'slow',  # Better quality at same bitrate
        '-crf', '10',  # Near-lossless quality for intermediate files
        '-r', str(fps),
        '-an',  # No audio
        str(output_video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"FFmpeg frame extraction timed out after 300s for {input_video_path}") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"FFmpeg frame extraction failed for {input_video_path}: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed (rc={result.returncode}): {result.stderr[:500]}")

    if not output_video_path.exists() or output_video_path.stat().st_size == 0:
        raise OSError(f"Output video missing or empty: {output_video_path}")

    # Verify frame count (allow small tolerance for OpenCV/FFmpeg discrepancies)
    actual_frames, _ = get_video_frame_count_and_fps(str(output_video_path))
    if actual_frames is None:
        raise RuntimeError(f"Could not verify output frame count for {output_video_path}")

    # Allow up to 3 frames difference - OpenCV and FFmpeg often disagree on frame counts
    # especially for end-of-video segments or variable framerate videos
    frame_diff = abs(actual_frames - expected_frames)
    if frame_diff > 3:
        raise RuntimeError(f"Frame count mismatch too large: expected {expected_frames}, got {actual_frames} (diff: {frame_diff})")
    elif frame_diff > 0:
        generation_logger.warning(f"[EXTRACT_RANGE] Minor frame count difference: expected {expected_frames}, got {actual_frames} (diff: {frame_diff}, within tolerance)")

    generation_logger.debug(f"[EXTRACT_RANGE] Extracted {actual_frames} frames to {output_video_path.name}")
    return output_video_path


def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int],
    *,
    standardize_colorspace: bool = False
) -> Path:
    """Creates a video from a list of NumPy BGR frames using FFmpeg subprocess.

    Uses streaming to pipe frames to FFmpeg incrementally, avoiding loading
    all frame data into memory at once. This is critical for large videos
    (e.g., 2000+ frames at 1080p would otherwise require 30+ GB RAM).

    Args:
        frames_list: List of BGR numpy arrays
        output_path: Output video path
        fps: Frames per second
        resolution: (width, height) tuple
        standardize_colorspace: If True, adds BT.709 colorspace standardization

    Returns:
        Path object of the successfully written file

    Raises:
        ValueError: If no valid frames are provided
        RuntimeError: If FFmpeg fails, times out, or output is missing/empty
        OSError: If FFmpeg is not installed or file system errors occur
    """

    output_path_obj = Path(output_path)
    output_path_mp4 = output_path_obj.with_suffix('.mp4')
    output_path_mp4.parent.mkdir(parents=True, exist_ok=True)

    generation_logger.debug(f"[CREATE_VIDEO] Creating video: {output_path_mp4}")
    generation_logger.debug(f"[CREATE_VIDEO] Input: {len(frames_list)} frames, resolution={resolution}, fps={fps}")

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
        generation_logger.warning(f"[CREATE_VIDEO] Will skip {skipped_none} None/invalid frames")
    if skipped_invalid > 0:
        generation_logger.warning(f"[CREATE_VIDEO] Will skip {skipped_invalid} non-RGB frames")

    if valid_count == 0:
        raise ValueError(f"No valid frames to process! Input had {len(frames_list)} frames, all invalid")

    generation_logger.debug(f"[CREATE_VIDEO] Streaming {valid_count} frames to FFmpeg (memory-efficient mode)")

    try:
        # Use Popen to stream frames incrementally instead of loading all into memory
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError as e:
        raise OSError("FFmpeg not found - is it installed?") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"Failed to start FFmpeg process: {e}") from e

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
                    generation_logger.warning(f"[CREATE_VIDEO] Failed to resize frame {frame_idx}: {e}")
                resize_warnings += 1
                continue

        # Write frame bytes directly to FFmpeg stdin
        try:
            proc.stdin.write(frame_np.tobytes())
            frames_written += 1
        except BrokenPipeError:
            generation_logger.error(f"[CREATE_VIDEO] FFmpeg pipe broken after {frames_written} frames")
            break

    if resize_warnings > 5:
        generation_logger.warning(f"[CREATE_VIDEO] {resize_warnings} total resize failures (only first 5 logged)")

    # Close stdin to signal end of input
    proc.stdin.close()

    generation_logger.debug(f"[CREATE_VIDEO] Wrote {frames_written} frames to FFmpeg, waiting for encoding...")

    # Read stderr in background thread to avoid blocking
    stderr_output = []
    def read_stderr():
        try:
            stderr_output.append(proc.stderr.read())
        except OSError as e:
            generation_logger.debug(f"[CREATE_VIDEO] Failed to read FFmpeg stderr: {e}")

    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.start()

    try:
        proc.wait(timeout=300)  # 5 minute timeout for encoding
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"FFmpeg timed out after 300 seconds creating video {output_path_mp4}")
    finally:
        stderr_thread.join(timeout=5)

    stderr = stderr_output[0] if stderr_output else b""

    if proc.returncode == 0:
        if output_path_mp4.exists() and output_path_mp4.stat().st_size > 0:
            generation_logger.debug(f"[CREATE_VIDEO] SUCCESS: Created {output_path_mp4} ({output_path_mp4.stat().st_size} bytes)")
            return output_path_mp4
        raise RuntimeError(f"FFmpeg succeeded but output file missing or empty: {output_path_mp4}")
    else:
        stderr_str = stderr.decode('utf-8', errors='replace') if stderr else "no stderr"
        if output_path_mp4.exists():
            try:
                output_path_mp4.unlink()
            except OSError as e:
                generation_logger.debug(f"[CREATE_VIDEO] Failed to clean up partial output file {output_path_mp4}: {e}")
        raise RuntimeError(f"FFmpeg failed (rc={proc.returncode}): {stderr_str[:500]}")


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
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", timeout=600)
        if outp.exists() and outp.stat().st_size > 0:
            return True
        return False
    except subprocess.CalledProcessError:
        return False


def video_has_audio(video_path: str | Path) -> bool:
    """Check whether a video file contains at least one audio stream.

    Uses ``ffprobe`` to inspect the container.  Returns ``False`` on any
    error (missing file, ffprobe not installed, etc.).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def mux_audio_from_segments(
    segment_video_paths: list[str | Path],
    stitched_video_path: str | Path,
    overlap_frames: list[int],
    fps: float,
    output_path: str | Path,
) -> Path | None:
    """Extract audio from segment videos, trim overlaps, concatenate, and mux onto the stitched video.

    The function:
    1. Filters segments that actually contain audio.
    2. For each audio-bearing segment, trims the overlap region (so audio
       does not double-up in crossfade zones).
    3. Concatenates the trimmed audio tracks.
    4. Muxes the result onto *stitched_video_path*, writing to *output_path*.

    Returns *output_path* on success, or ``None`` if no audio was found or
    the mux failed.
    """
    import tempfile

    segment_video_paths = [Path(p) for p in segment_video_paths]
    stitched_video_path = Path(stitched_video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Identify segments with audio
    audio_segments: list[tuple[int, Path]] = []
    for idx, seg_path in enumerate(segment_video_paths):
        if video_has_audio(seg_path):
            audio_segments.append((idx, seg_path))

    if not audio_segments:
        generation_logger.debug("[MUX_AUDIO] No segments contain audio; skipping mux")
        return None

    generation_logger.debug(f"[MUX_AUDIO] {len(audio_segments)}/{len(segment_video_paths)} segments have audio")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            trimmed_audio_files: list[Path] = []

            for seg_idx, seg_path in audio_segments:
                # Compute trim: remove overlap_frames at the start for segments after the first
                trim_start_s = 0.0
                if seg_idx > 0 and seg_idx - 1 < len(overlap_frames):
                    trim_start_s = overlap_frames[seg_idx - 1] / fps if fps > 0 else 0.0

                audio_out = tmpdir / f"audio_{seg_idx:03d}.aac"
                cmd = ["ffmpeg", "-y", "-i", str(seg_path)]
                if trim_start_s > 0:
                    cmd.extend(["-ss", f"{trim_start_s:.4f}"])
                cmd.extend(["-vn", "-acodec", "aac", "-b:a", "192k", str(audio_out)])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode != 0 or not audio_out.exists():
                    generation_logger.debug(f"[MUX_AUDIO] Failed to extract audio from segment {seg_idx}")
                    continue
                trimmed_audio_files.append(audio_out)

            if not trimmed_audio_files:
                generation_logger.debug("[MUX_AUDIO] No audio tracks extracted successfully")
                return None

            # Concatenate audio files
            concat_list = tmpdir / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{f}'" for f in trimmed_audio_files)
            )
            concat_audio = tmpdir / "concat_audio.aac"
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy", str(concat_audio),
            ]
            result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not concat_audio.exists():
                generation_logger.debug("[MUX_AUDIO] Audio concatenation failed")
                return None

            # Mux audio onto stitched video
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", str(stitched_video_path),
                "-i", str(concat_audio),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_path),
            ]
            result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
                generation_logger.debug(f"[MUX_AUDIO] Final mux failed: {result.stderr[:300]}")
                return None

        generation_logger.debug(f"[MUX_AUDIO] Successfully muxed audio onto {output_path.name}")
        return output_path

    except (subprocess.SubprocessError, OSError) as e:
        generation_logger.debug(f"[MUX_AUDIO] Audio mux error: {e}")
        return None
