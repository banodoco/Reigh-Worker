"""Layout functions for visualization collages."""

from typing import List, Optional

from source.core.log import headless_logger
from source.media.visualization.timeline import _create_timeline_clip


def _create_multi_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None,
    *,
    include_guidance: bool = False,
    timeline_position: str = "top",
    timeline_height: int = 150,
    composite_extra_height: int = 150,
):
    """
    Shared helper for side-by-side and triple-view layouts.

    Args:
        include_guidance: When True, include guidance_clip as a third column.
        timeline_position: "top" or "bottom".
        timeline_height: Pixel height of the timeline strip.
        composite_extra_height: Extra vertical pixels added for the timeline area.
    """
    from moviepy.editor import clips_array, CompositeVideoClip

    target_height = 400
    structure_resized = structure_clip.resize(height=target_height)
    output_resized = output_clip.resize(height=target_height)

    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip
        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"
        try:
            text_clip = TextClip(
                overlay_text, fontsize=16, color='white',
                font='DejaVu-Sans-Bold', bg_color='black',
                method='caption', size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))
            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except (OSError, ValueError, RuntimeError) as e:
            headless_logger.warning(f"Could not add structure overlay: {e}")

    if include_guidance and guidance_clip:
        guidance_resized = guidance_clip.resize(height=target_height)
        video_array = clips_array([[structure_resized, guidance_resized, output_resized]])
    else:
        video_array = clips_array([[structure_resized, output_resized]])

    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration, width=video_array.w,
        height=timeline_height, input_image_paths=input_image_paths,
        segment_frames=segment_frames, segment_prompts=segment_prompts,
        fps=fps, frame_overlaps=frame_overlaps,
    )

    if timeline_position == "top":
        timeline_clip = timeline_clip.set_position(("center", 0))
    else:
        timeline_clip = timeline_clip.set_position(("center", video_array.h))

    final = CompositeVideoClip(
        [video_array, timeline_clip],
        size=(video_array.w, video_array.h + composite_extra_height)
    )
    return final


def _create_side_by_side_layout(
    output_clip, structure_clip, guidance_clip,
    input_image_paths: List[str], segment_frames: List[int],
    segment_prompts: Optional[List[str]], fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """Side-by-side layout with timeline on top."""
    return _create_multi_layout(
        output_clip, structure_clip, guidance_clip,
        input_image_paths, segment_frames, segment_prompts, fps,
        frame_overlaps=frame_overlaps,
        structure_video_type=structure_video_type,
        structure_video_strength=structure_video_strength,
        include_guidance=False, timeline_position="top",
        timeline_height=150, composite_extra_height=150,
    )


def _create_triple_layout(
    output_clip, structure_clip, guidance_clip,
    input_image_paths: List[str], segment_frames: List[int],
    segment_prompts: Optional[List[str]], fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """Triple view layout with timeline on bottom."""
    return _create_multi_layout(
        output_clip, structure_clip, guidance_clip,
        input_image_paths, segment_frames, segment_prompts, fps,
        frame_overlaps=frame_overlaps,
        structure_video_type=structure_video_type,
        structure_video_strength=structure_video_strength,
        include_guidance=True, timeline_position="bottom",
        timeline_height=200, composite_extra_height=100,
    )


def _create_grid_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """
    Create 2x2 grid layout.

    Layout:
    +----------------+----------------+
    |  Structure     |   Guidance     |
    +----------------+----------------+
    |   Output       |   Timeline     |
    +----------------+----------------+
    """
    from moviepy.editor import clips_array

    target_height = 300
    target_width = structure_clip.w * target_height // structure_clip.h

    structure_resized = structure_clip.resize(height=target_height)
    output_resized = output_clip.resize(height=target_height)

    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip

        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"

        try:
            text_clip = TextClip(
                overlay_text,
                fontsize=16,
                color='white',
                font='DejaVu-Sans-Bold',
                bg_color='black',
                method='caption',
                size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))

            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except (OSError, ValueError, RuntimeError) as e:
            headless_logger.warning(f"Could not add structure overlay: {e}")

    if guidance_clip:
        guidance_resized = guidance_clip.resize(height=target_height)
    else:
        from moviepy.editor import ColorClip
        guidance_resized = ColorClip(
            size=(target_width, target_height),
            color=(0, 0, 0),
            duration=output_clip.duration
        )

    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration,
        width=target_width,
        height=target_height,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps
    )

    video_array = clips_array([
        [structure_resized, guidance_resized],
        [output_resized, timeline_clip]
    ])

    return video_array


def _create_vertical_layout(
    output_clip,
    structure_clip,
    guidance_clip,
    input_image_paths: List[str],
    segment_frames: List[int],
    segment_prompts: Optional[List[str]],
    fps: int,
    frame_overlaps: Optional[List[int]] = None,
    structure_video_type: Optional[str] = None,
    structure_video_strength: Optional[float] = None
):
    """
    Create vertical layout with images on left, videos stacked on right.

    Layout:
    +----------+-----------------+
    | Img 1    |                 |
    | Img 2    |  Structure      |
    | Img 3    |     Video       |
    | Img 4    |                 |
    | Img 5    |-----------------+
    |  ...     |                 |
    |          |  Output         |
    |          |    Video        |
    |          |                 |
    +----------+-----------------+
    """
    from moviepy.editor import clips_array, CompositeVideoClip

    video_target_height = 300

    structure_resized = structure_clip.resize(height=video_target_height)
    output_resized = output_clip.resize(height=video_target_height)

    def ensure_even_dimensions(clip):
        w, h = clip.w, clip.h
        if w % 2 != 0:
            w += 1
        if h % 2 != 0:
            h += 1
        if w != clip.w or h != clip.h:
            return clip.resize(width=w, height=h)
        return clip

    structure_resized = ensure_even_dimensions(structure_resized)
    output_resized = ensure_even_dimensions(output_resized)

    headless_logger.debug(f"  structure_resized: {structure_resized.w}x{structure_resized.h}")
    headless_logger.debug(f"  output_resized: {output_resized.w}x{output_resized.h}")

    if structure_video_type and structure_video_strength is not None:
        from moviepy.editor import TextClip, CompositeVideoClip

        overlay_text = f"{structure_video_type} | {structure_video_strength:.1f}"

        try:
            text_clip = TextClip(
                overlay_text,
                fontsize=16,
                color='white',
                font='DejaVu-Sans-Bold',
                bg_color='black',
                method='caption',
                size=(None, None)
            ).set_duration(structure_resized.duration).set_position(('left', 'top'))

            structure_resized = CompositeVideoClip([structure_resized, text_clip])
        except (OSError, ValueError, RuntimeError) as e:
            headless_logger.warning(f"Could not add structure overlay: {e}")

    video_stack = clips_array([[structure_resized], [output_resized]])

    timeline_height = video_stack.h
    timeline_width = 200
    _margin = 20

    video_stack_width = video_stack.w
    total_width = timeline_width + video_stack_width

    if total_width % 2 != 0:
        timeline_width += 1

    if video_stack.h % 2 != 0:
        timeline_height = video_stack.h + 1
        video_stack = video_stack.resize(height=timeline_height)

    timeline_clip = _create_timeline_clip(
        duration=output_clip.duration,
        width=timeline_width,
        height=timeline_height,
        input_image_paths=input_image_paths,
        segment_frames=segment_frames,
        segment_prompts=segment_prompts,
        fps=fps,
        frame_overlaps=frame_overlaps,
        vertical=True
    )

    final = clips_array([[timeline_clip, video_stack]])

    final_width = final.w
    final_height = final.h
    if final_width % 2 != 0 or final_height % 2 != 0:
        if final_width % 2 != 0:
            final_width += 1
        if final_height % 2 != 0:
            final_height += 1
        final = final.resize(width=final_width, height=final_height)

    return final
