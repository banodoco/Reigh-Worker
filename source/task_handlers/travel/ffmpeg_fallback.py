"""FFmpeg-based cross-fade fallback for travel stitching."""

from pathlib import Path

from ...core.log import travel_logger

def attempt_ffmpeg_crossfade_fallback(segment_video_paths: list[str], overlaps: list[int], output_path: Path, task_id: str) -> bool:
    """
    Fallback cross-fade implementation using FFmpeg's xfade filter.
    Achieves the same visual effect as frame-based cross-fade without frame extraction.

    Args:
        segment_video_paths: List of video file paths to stitch
        overlaps: List of overlap frame counts between segments
        output_path: Path for output video
        task_id: Task ID for logging

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import subprocess
        import cv2

        if len(segment_video_paths) < 2:
            travel_logger.debug(f"FFmpeg Fallback: Not enough videos to cross-fade ({len(segment_video_paths)})")
            return False

        # Get video properties from first segment to calculate timing
        cap = cv2.VideoCapture(segment_video_paths[0])
        if not cap.isOpened():
            travel_logger.debug(f"FFmpeg Fallback: Cannot read first video properties")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            travel_logger.debug(f"FFmpeg Fallback: Invalid FPS ({fps})")
            return False

        travel_logger.essential(f"FFmpeg fallback: Creating cross-fade with {len(segment_video_paths)} videos at {fps} FPS", task_id=task_id)

        # Build FFmpeg command for cross-fade stitching
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output

        # Add all input videos
        for video_path in segment_video_paths:
            cmd.extend(["-i", str(video_path)])

        # Build complex filter for cross-fade transitions
        filter_parts = []
        current_label = "[0:v]"

        for i, overlap_frames in enumerate(overlaps):
            if i >= len(segment_video_paths) - 1:
                break

            next_input_idx = i + 1
            next_label = f"[{next_input_idx}:v]"
            output_label = f"[fade{i}]" if i < len(overlaps) - 1 else ""

            # Convert overlap frames to duration in seconds
            overlap_duration = overlap_frames / fps

            # Get duration of current video to calculate offset
            cap = cv2.VideoCapture(segment_video_paths[i])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            video_duration = total_frames / fps
            offset_time = video_duration - overlap_duration

            # Create xfade filter
            xfade_filter = f"{current_label}{next_label}xfade=transition=fade:duration={overlap_duration:.3f}:offset={offset_time:.3f}{output_label}"
            filter_parts.append(xfade_filter)

            current_label = f"[fade{i}]"

        if filter_parts:
            filter_complex = ";".join(filter_parts)
            cmd.extend(["-filter_complex", filter_complex])

        # Add output parameters
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "10",  # Near-lossless quality for intermediate files
            "-pix_fmt", "yuv420p",
            str(output_path)
        ])

        travel_logger.debug(f"FFmpeg fallback: Running command: {' '.join(cmd[:10])}...", task_id=task_id)
        travel_logger.debug(f"FFmpeg Fallback: Running cross-fade command")

        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            travel_logger.essential(f"FFmpeg fallback success! Output: {output_path} ({output_path.stat().st_size:,} bytes)", task_id=task_id)
            travel_logger.debug(f"FFmpeg Fallback: Success - created {output_path}")
            return True
        else:
            travel_logger.error(f"FFmpeg fallback failed with return code {result.returncode}", task_id=task_id)
            if result.stderr:
                travel_logger.error(f"FFmpeg fallback error: {result.stderr[:200]}...", task_id=task_id)
            travel_logger.debug(f"FFmpeg Fallback: Failed - {result.stderr[:100] if result.stderr else 'Unknown error'}")
            return False

    except subprocess.TimeoutExpired:
        travel_logger.error(f"FFmpeg fallback timeout after 300 seconds", task_id=task_id)
        travel_logger.debug(f"FFmpeg Fallback: Timeout")
        return False
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        travel_logger.error(f"FFmpeg fallback exception: {e}", task_id=task_id)
        travel_logger.debug(f"FFmpeg Fallback: Exception - {e}")
        return False
