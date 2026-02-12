"""FFmpeg-based cross-fade fallback for travel stitching."""

from pathlib import Path


def attempt_ffmpeg_crossfade_fallback(segment_video_paths: list[str], overlaps: list[int], output_path: Path, task_id: str, dprint) -> bool:
    """
    Fallback cross-fade implementation using FFmpeg's xfade filter.
    Achieves the same visual effect as frame-based cross-fade without frame extraction.

    Args:
        segment_video_paths: List of video file paths to stitch
        overlaps: List of overlap frame counts between segments
        output_path: Path for output video
        task_id: Task ID for logging
        dprint: Debug print function

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import subprocess
        import cv2

        if len(segment_video_paths) < 2:
            dprint(f"FFmpeg Fallback: Not enough videos to cross-fade ({len(segment_video_paths)})")
            return False

        # Get video properties from first segment to calculate timing
        cap = cv2.VideoCapture(segment_video_paths[0])
        if not cap.isOpened():
            dprint(f"FFmpeg Fallback: Cannot read first video properties")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            dprint(f"FFmpeg Fallback: Invalid FPS ({fps})")
            return False

        print(f"[FFMPEG FALLBACK] Creating cross-fade with {len(segment_video_paths)} videos at {fps} FPS")

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

        print(f"[FFMPEG FALLBACK] Running command: {' '.join(cmd[:10])}...")  # Don't log full command (too long)
        dprint(f"FFmpeg Fallback: Running cross-fade command")

        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            print(f"[FFMPEG FALLBACK] Success! Output: {output_path} ({output_path.stat().st_size:,} bytes)")
            dprint(f"FFmpeg Fallback: Success - created {output_path}")
            return True
        else:
            print(f"[FFMPEG FALLBACK] Failed with return code {result.returncode}")
            if result.stderr:
                print(f"[FFMPEG FALLBACK] Error: {result.stderr[:200]}...")
            dprint(f"FFmpeg Fallback: Failed - {result.stderr[:100] if result.stderr else 'Unknown error'}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[FFMPEG FALLBACK] Timeout after 300 seconds")
        dprint(f"FFmpeg Fallback: Timeout")
        return False
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        print(f"[FFMPEG FALLBACK] Exception: {e}")
        dprint(f"FFmpeg Fallback: Exception - {e}")
        return False
