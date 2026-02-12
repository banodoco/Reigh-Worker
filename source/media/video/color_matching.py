"""Color transfer and matching functions for video processing."""
from pathlib import Path

import cv2
import numpy as np

from source.media.video.frame_extraction import extract_frames_from_video
from source.media.video.video_info import get_video_frame_count_and_fps
from source.media.video.ffmpeg_ops import create_video_from_frames_list

__all__ = [
    "_cm_enhance_saturation",
    "_cm_transfer_mean_std_lab",
    "apply_color_matching_to_video",
]

# Flag for dependency availability
_COLOR_MATCH_DEPS_AVAILABLE = True


def _cm_enhance_saturation(image_bgr, saturation_factor=0.5):
    """
    Adjust saturation of an image by the given factor.
    saturation_factor: 1.0 = no change, 0.5 = 50% reduction, 1.3 = 30% increase, etc.
    """
    if not _COLOR_MATCH_DEPS_AVAILABLE: return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_float = hsv.astype(np.float32)
    h, s, v = cv2.split(hsv_float)
    s_adjusted = s * saturation_factor
    s_adjusted = np.clip(s_adjusted, 0, 255)
    hsv_adjusted = cv2.merge([h, s_adjusted, v])
    hsv_adjusted_uint8 = hsv_adjusted.astype(np.uint8)
    adjusted_bgr = cv2.cvtColor(hsv_adjusted_uint8, cv2.COLOR_HSV2BGR)
    return adjusted_bgr

def _cm_transfer_mean_std_lab(source_bgr, target_bgr):
    if not _COLOR_MATCH_DEPS_AVAILABLE: return source_bgr
    MIN_ALLOWED_STD_RATIO_FOR_LUMINANCE = 0.1
    MIN_ALLOWED_STD_RATIO_FOR_COLOR = 0.4
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)

    source_lab_float = source_lab.astype(np.float32)
    target_lab_float = target_lab.astype(np.float32)

    s_l, s_a, s_b = cv2.split(source_lab_float)
    t_l, t_a, t_b = cv2.split(target_lab_float)

    channels_out = []
    for i, (s_chan, t_chan) in enumerate(zip([s_l, s_a, s_b], [t_l, t_a, t_b])):
        s_mean_val, s_std_val = cv2.meanStdDev(s_chan)
        t_mean_val, t_std_val = cv2.meanStdDev(t_chan)
        s_mean, s_std = s_mean_val[0][0], s_std_val[0][0]
        t_mean, t_std = t_mean_val[0][0], t_std_val[0][0]

        std_ratio = t_std / s_std if s_std > 1e-5 else 1.0

        min_ratio = MIN_ALLOWED_STD_RATIO_FOR_LUMINANCE if i == 0 else MIN_ALLOWED_STD_RATIO_FOR_COLOR
        effective_std_ratio = max(std_ratio, min_ratio)

        if s_std > 1e-5:
            transformed_chan = (s_chan - s_mean) * effective_std_ratio + t_mean
        else:
            transformed_chan = np.full_like(s_chan, t_mean)

        channels_out.append(transformed_chan)

    result_lab_float = cv2.merge(channels_out)
    result_lab_clipped = np.clip(result_lab_float, 0, 255)
    result_lab_uint8 = result_lab_clipped.astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab_uint8, cv2.COLOR_LAB2BGR)
    return result_bgr

def apply_color_matching_to_video(video_path: str, start_ref_path: str, end_ref_path: str, output_path: str, dprint):
    if not all([_COLOR_MATCH_DEPS_AVAILABLE, Path(video_path).exists(), Path(start_ref_path).exists(), Path(end_ref_path).exists()]):
        dprint(f"Color Matching: Skipping due to missing deps or files. Deps:{_COLOR_MATCH_DEPS_AVAILABLE}, Video:{Path(video_path).exists()}, Start:{Path(start_ref_path).exists()}, End:{Path(end_ref_path).exists()}")
        return None

    frames = extract_frames_from_video(video_path)
    frame_count, fps = get_video_frame_count_and_fps(video_path)
    if not frames or not frame_count or not fps:
        dprint("Color Matching: Frame extraction or metadata retrieval failed.")
        return None

    # Get resolution from the first frame
    h, w, _ = frames[0].shape
    resolution = (w, h)

    start_ref_bgr = cv2.imread(start_ref_path)
    end_ref_bgr = cv2.imread(end_ref_path)
    start_ref_resized = cv2.resize(start_ref_bgr, resolution)
    end_ref_resized = cv2.resize(end_ref_bgr, resolution)

    total_frames = len(frames)
    accumulated_frames = []

    for i, frame_bgr in enumerate(frames):
        frame_bgr_desaturated = _cm_enhance_saturation(frame_bgr, saturation_factor=0.5)

        corrected_start_bgr = _cm_transfer_mean_std_lab(frame_bgr_desaturated, start_ref_resized)
        corrected_end_bgr = _cm_transfer_mean_std_lab(frame_bgr_desaturated, end_ref_resized)

        t = i / (total_frames - 1) if total_frames > 1 else 1.0
        w_original = (0.5 * t) if t < 0.5 else (0.5 - 0.5 * t)
        w_correct = 1.0 - w_original
        w_start = (1.0 - t) * w_correct
        w_end = t * w_correct

        blend_float = (w_start * corrected_start_bgr.astype(np.float32) +
                       w_end * corrected_end_bgr.astype(np.float32) +
                       w_original * frame_bgr.astype(np.float32))

        blended_frame_bgr = np.clip(blend_float, 0, 255).astype(np.uint8)
        accumulated_frames.append(blended_frame_bgr)

    if accumulated_frames:
        created_video_path = create_video_from_frames_list(accumulated_frames, output_path, fps, resolution)
        dprint(f"Color Matching: Successfully created color matched video at {created_video_path}")
        return created_video_path

    dprint("Color Matching: Failed to produce any frames.")
    return None
