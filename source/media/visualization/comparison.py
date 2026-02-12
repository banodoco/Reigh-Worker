"""Comparison and travel visualization creation."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

from source.media.visualization.timeline import _apply_video_treatment
from source.media.visualization.layouts import (
    _create_side_by_side_layout,
    _create_triple_layout,
    _create_grid_layout,
    _create_vertical_layout,
)


def create_travel_visualization(
    output_video_path: str,
    structure_video_path: str,
    guidance_video_path: Optional[str],
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]] = None,
    viz_output_path: Optional[str] = None,
    layout: str = "side_by_side",
    fps: int = 16,
    show_guidance: bool = True,
    structure_video_treatment: str = "adjust",
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
) -> str:
    """
    Create a visualization collage showing the generation process.

    Args:
        output_video_path: Path to final output video
        structure_video_path: Path to structure/flow video
        guidance_video_path: Path to guidance video (optional)
        input_image_paths: List of input image paths
        segment_frames: List of frame counts per segment (raw, before overlap subtraction)
        segment_prompts: Optional list of prompts per segment
        viz_output_path: Where to save visualization (default: adds _viz suffix)
        layout: Layout type - "side_by_side", "triple", "grid", or "vertical"
        fps: FPS for output video
        show_guidance: Whether to include guidance video
        structure_video_treatment: "adjust" (stretch/compress) or "clip" (temporal sample)
        frame_overlaps: Optional list of overlap counts between segments
        structure_video_type: Optional structure video type label for overlay
        structure_video_strength: Optional structure video strength value for overlay

    Returns:
        Path to created visualization video
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "MoviePy is required for visualization. Install with: pip install moviepy"
        )

    if viz_output_path is None:
        output_path = Path(output_video_path)
        viz_output_path = str(output_path.parent / f"{output_path.stem}_viz.mp4")

    print(f"\U0001f4f9 Loading videos for visualization...")
    output_clip = VideoFileClip(output_video_path)
    structure_clip = VideoFileClip(structure_video_path)

    if guidance_video_path and show_guidance:
        guidance_clip = VideoFileClip(guidance_video_path)
    else:
        guidance_clip = None

    print(f"\U0001f4d0 Structure video treatment: {structure_video_treatment}")
    structure_clip = _apply_video_treatment(
        structure_clip,
        target_duration=output_clip.duration,
        target_fps=fps,
        treatment=structure_video_treatment,
        video_name="structure"
    )

    if guidance_clip:
        guidance_clip = _apply_video_treatment(
            guidance_clip,
            target_duration=output_clip.duration,
            target_fps=fps,
            treatment=structure_video_treatment,
            video_name="guidance"
        )

    layout_kwargs = dict(
        output_clip=output_clip,
        structure_clip=structure_clip,
        guidance_clip=guidance_clip,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps,
        structure_video_type=structure_video_type,
        structure_video_strength=structure_video_strength,
    )

    if layout == "side_by_side":
        result = _create_side_by_side_layout(**layout_kwargs)
    elif layout == "triple":
        result = _create_triple_layout(**layout_kwargs)
    elif layout == "grid":
        result = _create_grid_layout(**layout_kwargs)
    elif layout == "vertical":
        result = _create_vertical_layout(**layout_kwargs)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    print(f"\U0001f4be Writing visualization to: {viz_output_path}")
    result.write_videofile(
        viz_output_path,
        fps=fps,
        codec='libx264',
        audio=False,
        preset='slow',
        ffmpeg_params=['-crf', '10', '-pix_fmt', 'yuv420p'],
        logger=None
    )

    output_clip.close()
    structure_clip.close()
    if guidance_clip:
        guidance_clip.close()
    result.close()

    print(f"\u2705 Visualization saved: {viz_output_path}")
    return viz_output_path


def create_simple_comparison(
    video1_path: str,
    video2_path: str,
    output_path: str,
    labels: Optional[Tuple[str, str]] = None,
    orientation: str = "horizontal"
) -> str:
    """
    Create a simple side-by-side or stacked comparison of two videos.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Where to save comparison
        labels: Optional tuple of (label1, label2)
        orientation: "horizontal" or "vertical"

    Returns:
        Path to created comparison video
    """
    try:
        from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
    except ImportError:
        raise ImportError("MoviePy is required. Install with: pip install moviepy")

    print(f"\U0001f4f9 Creating comparison video...")

    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)

    if labels:
        try:
            from moviepy.editor import TextClip
            label1 = TextClip(labels[0], fontsize=24, color='white', bg_color='black')
            label1 = label1.set_duration(clip1.duration).set_position(("center", "top"))
            clip1 = CompositeVideoClip([clip1, label1])

            label2 = TextClip(labels[1], fontsize=24, color='white', bg_color='black')
            label2 = label2.set_duration(clip2.duration).set_position(("center", "top"))
            clip2 = CompositeVideoClip([clip2, label2])
        except (OSError, ValueError, RuntimeError):
            print("Warning: Could not add labels (text rendering failed)")

    if orientation == "horizontal":
        final = clips_array([[clip1, clip2]])
    else:
        final = clips_array([[clip1], [clip2]])

    print(f"\U0001f4be Writing comparison to: {output_path}")
    final.write_videofile(
        output_path,
        codec='libx264',
        audio=False,
        preset='slow',
        ffmpeg_params=['-crf', '10', '-pix_fmt', 'yuv420p'],
        logger=None
    )

    clip1.close()
    clip2.close()
    final.close()

    print(f"\u2705 Comparison saved: {output_path}")
    return output_path


def create_opencv_side_by_side(
    video1_path: str,
    video2_path: str,
    output_path: str,
    fps: Optional[int] = None
) -> str:
    """
    Create side-by-side comparison using only OpenCV (faster, no MoviePy needed).

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Where to save comparison
        fps: Output FPS (defaults to video1 FPS)

    Returns:
        Path to created comparison video
    """
    print(f"\U0001f4f9 Creating OpenCV side-by-side comparison...")

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if fps is None:
        fps = int(cap1.get(cv2.CAP_PROP_FPS))

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_height = min(height1, height2)
    target_width1 = int(width1 * target_height / height1)
    target_width2 = int(width2 * target_height / height2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (target_width1 + target_width2, target_height)
    )

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1 = cv2.resize(frame1, (target_width1, target_height))
        frame2 = cv2.resize(frame2, (target_width2, target_height))

        combined = np.hstack([frame1, frame2])

        out.write(combined)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    cap1.release()
    cap2.release()
    out.release()

    print(f"\u2705 Comparison saved: {output_path} ({frame_count} frames)")
    return output_path
