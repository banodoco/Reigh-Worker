"""Mask video creation utilities."""

from pathlib import Path

import numpy as np

from source.utils.frame_utils import create_video_from_frames_list


def create_mask_video_from_inactive_indices(
    total_frames: int,
    resolution_wh: tuple[int, int],
    inactive_frame_indices: set[int] | list[int],
    output_path: Path | str,
    fps: int = 16,
    task_id_for_logging: str = "unknown",
    *, dprint = print
) -> Path | None:
    """
    Create a mask video where:
    - Black frames (0) = inactive/keep original - don't edit these frames
    - White frames (255) = active/generate - model should generate these frames

    Args:
        total_frames: Total number of frames in the video
        resolution_wh: (width, height) tuple for video resolution
        inactive_frame_indices: Set or list of frame indices that should be black (inactive)
        output_path: Where to save the mask video
        fps: Frames per second for the output video
        task_id_for_logging: Task ID for debug logging
        dprint: Print function for logging

    Returns:
        Path to created mask video, or None if creation failed
    """
    try:
        if total_frames <= 0:
            dprint(f"[WARNING] Task {task_id_for_logging}: Cannot create mask video with {total_frames} frames")
            return None

        h, w = resolution_wh[1], resolution_wh[0]  # height, width
        inactive_set = set(inactive_frame_indices) if not isinstance(inactive_frame_indices, set) else inactive_frame_indices

        dprint(f"Task {task_id_for_logging}: Creating mask video - total_frames={total_frames}, "
               f"inactive_indices={sorted(list(inactive_set))[:10]}{'...' if len(inactive_set) > 10 else ''}")

        # Create mask frames: 0 (black) for inactive, 255 (white) for active
        mask_frames_buf: list[np.ndarray] = [
            np.full((h, w, 3), 0 if idx in inactive_set else 255, dtype=np.uint8)
            for idx in range(total_frames)
        ]

        created_mask_video = create_video_from_frames_list(
            mask_frames_buf,
            Path(output_path),
            fps,
            resolution_wh
        )

        if created_mask_video and created_mask_video.exists():
            dprint(f"Task {task_id_for_logging}: Mask video created successfully at {created_mask_video}")
            return created_mask_video
        else:
            dprint(f"[WARNING] Task {task_id_for_logging}: Failed to create mask video at {output_path}")
            return None

    except (OSError, ValueError, RuntimeError) as e:
        dprint(f"[ERROR] Task {task_id_for_logging}: Mask video creation failed: {e}")
        return None


def create_simple_first_frame_mask_video(
    total_frames: int,
    resolution_wh: tuple[int, int],
    output_path: Path | str,
    fps: int = 16,
    task_id_for_logging: str = "unknown",
    *, dprint = print
) -> Path | None:
    """
    Convenience function to create a mask video where only the first frame is inactive (black).
    This is useful for workflows where you want to keep the first frame unchanged
    and generate the rest.

    Returns:
        Path to created mask video, or None if creation failed
    """
    return create_mask_video_from_inactive_indices(
        total_frames=total_frames,
        resolution_wh=resolution_wh,
        inactive_frame_indices={0},  # Only first frame is inactive
        output_path=output_path,
        fps=fps,
        task_id_for_logging=task_id_for_logging,
        dprint=dprint
    )
