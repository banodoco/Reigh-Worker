"""Skeleton/keypoint drawing, pose interpolation, and debug video for poses."""

import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from source.utils.prompt_utils import dprint, DEBUG_MODE
from source.utils.frame_utils import (
    _image_to_frame_simple,
    create_color_frame,
    create_video_from_frames_list,
)


# --- Pose color and skeleton constants ---
body_colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]
face_color = [255, 255, 255]
hand_keypoint_color = [0, 0, 255]
hand_limb_colors = [
    [255,0,0],[255,60,0],[255,120,0],[255,180,0], [180,255,0],[120,255,0],[60,255,0],[0,255,0],
    [0,255,60],[0,255,120],[0,255,180],[0,180,255], [0,120,255],[0,60,255],[0,0,255],[60,0,255],
    [120,0,255],[180,0,255],[255,0,180],[255,0,120]
]

# MediaPipe Pose connections (33 landmarks, indices 0-32)
# Based on mp.solutions.pose.POSE_CONNECTIONS
body_skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to Left Eye to Left Ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Nose to Right Eye to Right Ear
    (9, 10),  # Mouth
    (11, 12), # Shoulders
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 19), (15, 21), # Left Arm and simplified Left Hand (wrist to fingers)
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 20), (16, 22), # Right Arm and simplified Right Hand (wrist to fingers)
    (11, 23), (12, 24), # Connect shoulders to Hips
    (23, 24), # Hip connection
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Left Leg and Foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # Right Leg and Foot
]

face_skeleton = [] # Draw face dots only, no connections

# MediaPipe Hand connections (21 landmarks per hand, indices 0-20)
hand_skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]


def draw_keypoints_and_skeleton(image, keypoints_data, skeleton_connections, colors_config, confidence_threshold=0.1, point_radius=3, line_thickness=2, is_face=False, is_hand=False):
    if not keypoints_data:
        return

    tri_tuples = []
    if isinstance(keypoints_data, list) and len(keypoints_data) > 0 and isinstance(keypoints_data[0], (int, float)) and len(keypoints_data) % 3 == 0:
        for i in range(0, len(keypoints_data), 3):
            tri_tuples.append(keypoints_data[i:i+3])
    else:
        dprint(f"draw_keypoints_and_skeleton: Unexpected keypoints_data format or length not divisible by 3. Data: {keypoints_data}")
        return

    if skeleton_connections:
        for i, (joint_idx_a, joint_idx_b) in enumerate(skeleton_connections):
            if joint_idx_a >= len(tri_tuples) or joint_idx_b >= len(tri_tuples):
                continue
            a_x, a_y, a_confidence = tri_tuples[joint_idx_a]
            b_x, b_y, b_confidence = tri_tuples[joint_idx_b]

            if a_confidence >= confidence_threshold and b_confidence >= confidence_threshold:
                limb_color = None
                if is_hand:
                    limb_color_list = colors_config['limbs']
                    limb_color = limb_color_list[i % len(limb_color_list)]
                else:
                    limb_color_list = colors_config if isinstance(colors_config, list) else [colors_config]
                    limb_color = limb_color_list[i % len(limb_color_list)]
                if limb_color is not None:
                    cv2.line(image, (int(a_x), int(a_y)), (int(b_x), int(b_y)), limb_color, line_thickness)

    for i, (x, y, confidence) in enumerate(tri_tuples):
        if confidence >= confidence_threshold:
            point_color = None
            current_radius = point_radius
            if is_hand:
                point_color = colors_config['points']
            elif is_face:
                point_color = colors_config
                current_radius = 2
            else:
                point_color_list = colors_config
                point_color = point_color_list[i % len(point_color_list)]
            if point_color is not None:
                cv2.circle(image, (int(x), int(y)), current_radius, point_color, -1)


def gen_skeleton_with_face_hands(pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d,
                                 canvas_width, canvas_height, landmarkType, confidence_threshold=0.1):
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    def scale_keypoints(keypoints, target_w, target_h, input_is_normalized):
        if not keypoints: return []
        scaled = []
        if not isinstance(keypoints, list) or (keypoints and not isinstance(keypoints[0], (int, float))):
            dprint(f"scale_keypoints: Unexpected keypoints format: {type(keypoints)}. Expecting flat list of numbers.")
            return []

        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            if input_is_normalized: scaled.extend([x * target_w, y * target_h, conf])
            else: scaled.extend([x, y, conf])
        return scaled

    input_is_normalized = (landmarkType == "OpenPose") # This might need adjustment based on actual landmarkType usage

    scaled_pose = scale_keypoints(pose_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    scaled_face = scale_keypoints(face_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    scaled_hand_left = scale_keypoints(hand_left_keypoints_2d, canvas_width, canvas_height, input_is_normalized)
    scaled_hand_right = scale_keypoints(hand_right_keypoints_2d, canvas_width, canvas_height, input_is_normalized)

    draw_keypoints_and_skeleton(image, scaled_pose, body_skeleton, body_colors, confidence_threshold, point_radius=6, line_thickness=4)
    if scaled_face:
        draw_keypoints_and_skeleton(image, scaled_face, face_skeleton, face_color, confidence_threshold, point_radius=2, line_thickness=1, is_face=True)
    hand_colors_config = {'limbs': hand_limb_colors, 'points': hand_keypoint_color}
    if scaled_hand_left:
        draw_keypoints_and_skeleton(image, scaled_hand_left, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    if scaled_hand_right:
        draw_keypoints_and_skeleton(image, scaled_hand_right, hand_skeleton, hand_colors_config, confidence_threshold, point_radius=3, line_thickness=2, is_hand=True)
    return image


def transform_all_keypoints(keypoints_1_dict, keypoints_2_dict, frames, interpolation="linear"):
    def interpolate_keypoint_set(kp1_list, kp2_list, num_frames, interp_method):
        if not kp1_list and not kp2_list: return [[] for _ in range(num_frames)]

        len1 = len(kp1_list) if kp1_list else 0
        len2 = len(kp2_list) if kp2_list else 0

        if not kp1_list: kp1_list = [0.0] * len2
        if not kp2_list: kp2_list = [0.0] * len1

        if len(kp1_list) != len(kp2_list) or not kp1_list or len(kp1_list) % 3 != 0:
             dprint(f"interpolate_keypoint_set: Mismatched, empty, or non-triplet keypoint lists after padding. KP1 len: {len(kp1_list)}, KP2 len: {len(kp2_list)}. Returning empty sequences.")
             return [[] for _ in range(num_frames)]

        tri_tuples_1 = [kp1_list[i:i + 3] for i in range(0, len(kp1_list), 3)]
        tri_tuples_2 = [kp2_list[i:i + 3] for i in range(0, len(kp2_list), 3)]

        keypoints_sequence = []
        for j in range(num_frames):
            interpolated_kps_for_frame = []
            t = j / float(num_frames - 1) if num_frames > 1 else 0.0

            interp_factor = t
            if interp_method == "ease-in": interp_factor = t * t
            elif interp_method == "ease-out": interp_factor = 1 - (1 - t) * (1 - t)
            elif interp_method == "ease-in-out":
                if t < 0.5: interp_factor = 2 * t * t
                else: interp_factor = 1 - pow(-2 * t + 2, 2) / 2

            for i in range(len(tri_tuples_1)):
                x1, y1, c1 = tri_tuples_1[i]
                x2, y2, c2 = tri_tuples_2[i]
                new_x, new_y, new_c = 0.0, 0.0, 0.0

                if c1 > 0.05 and c2 > 0.05:
                    new_x = x1 + (x2 - x1) * interp_factor
                    new_y = y1 + (y2 - y1) * interp_factor
                    new_c = c1 + (c2 - c1) * interp_factor
                elif c1 > 0.05 and c2 <= 0.05:
                    new_x, new_y = x1, y1
                    new_c = c1 * (1.0 - interp_factor)
                elif c1 <= 0.05 and c2 > 0.05:
                    new_x, new_y = x2, y2
                    new_c = c2 * interp_factor
                interpolated_kps_for_frame.extend([new_x, new_y, new_c])
            keypoints_sequence.append(interpolated_kps_for_frame)
        return keypoints_sequence

    pose_1 = keypoints_1_dict.get('pose_keypoints_2d', [])
    face_1 = keypoints_1_dict.get('face_keypoints_2d', [])
    hand_left_1 = keypoints_1_dict.get('hand_left_keypoints_2d', [])
    hand_right_1 = keypoints_1_dict.get('hand_right_keypoints_2d', [])

    pose_2 = keypoints_2_dict.get('pose_keypoints_2d', [])
    face_2 = keypoints_2_dict.get('face_keypoints_2d', [])
    hand_left_2 = keypoints_2_dict.get('hand_left_keypoints_2d', [])
    hand_right_2 = keypoints_2_dict.get('hand_right_keypoints_2d', [])

    pose_sequence = interpolate_keypoint_set(pose_1, pose_2, frames, interpolation)
    face_sequence = interpolate_keypoint_set(face_1, face_2, frames, interpolation)
    hand_left_sequence = interpolate_keypoint_set(hand_left_1, hand_left_2, frames, interpolation)
    hand_right_sequence = interpolate_keypoint_set(hand_right_1, hand_right_2, frames, interpolation)

    combined_sequence = []
    for i in range(frames):
        combined_frame_data = {
            'pose_keypoints_2d': pose_sequence[i] if i < len(pose_sequence) else [],
            'face_keypoints_2d': face_sequence[i] if i < len(face_sequence) else [],
            'hand_left_keypoints_2d': hand_left_sequence[i] if i < len(hand_left_sequence) else [],
            'hand_right_keypoints_2d': hand_right_sequence[i] if i < len(hand_right_sequence) else []
        }
        combined_sequence.append(combined_frame_data)
    return combined_sequence


def extract_pose_keypoints(image_path: str | Path, include_face=True, include_hands=True, resolution: tuple[int,int]=(640,480)) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = resolution[1], resolution[0] # For scaling output coords

    mp_holistic = mp.solutions.holistic
    holistic_instance = mp_holistic.Holistic(static_image_mode=True,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
    try:
        # Convert BGR image to RGB for MediaPipe
        results = holistic_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        holistic_instance.close()

    keypoints = {}
    pose_kps = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose_kps.extend([lm.x * width, lm.y * height, lm.visibility])
    keypoints['pose_keypoints_2d'] = pose_kps

    face_kps = []
    if include_face and results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            face_kps.extend([lm.x * width, lm.y * height, lm.visibility if hasattr(lm, 'visibility') else 1.0])
    keypoints['face_keypoints_2d'] = face_kps

    left_hand_kps = []
    if include_hands and results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            left_hand_kps.extend([lm.x * width, lm.y * height, lm.visibility])
    keypoints['hand_left_keypoints_2d'] = left_hand_kps

    right_hand_kps = []
    if include_hands and results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            right_hand_kps.extend([lm.x * width, lm.y * height, lm.visibility])
    keypoints['hand_right_keypoints_2d'] = right_hand_kps

    return keypoints


def create_pose_interpolated_guide_video(output_video_path: str | Path, resolution: tuple[int, int], total_frames: int,
                                           start_image_path: str | Path, end_image_path: str | Path,
                                           interpolation="linear", confidence_threshold=0.1,
                                           include_face=True, include_hands=True, fps=25):
    dprint(f"Creating pose interpolated guide: {output_video_path} from '{Path(start_image_path).name}' to '{Path(end_image_path).name}' ({total_frames} frames). First frame will be actual start image.")

    if total_frames <= 0:
        dprint(f"Video creation skipped for {output_video_path} as total_frames is {total_frames}.")
        return

    frames_list = []
    canvas_width, canvas_height = resolution

    first_visual_frame_np = _image_to_frame_simple(start_image_path, resolution)
    if first_visual_frame_np is None:
        print(f"Error loading start image {start_image_path} for guide video frame 0. Using black frame.")
        traceback.print_exc()
        first_visual_frame_np = create_color_frame(resolution, (0,0,0))
    frames_list.append(first_visual_frame_np)

    if total_frames > 1:
        try:
            # Pass the target resolution for keypoint scaling
            keypoints_from = extract_pose_keypoints(start_image_path, include_face, include_hands, resolution)
            keypoints_to = extract_pose_keypoints(end_image_path, include_face, include_hands, resolution)
        except (OSError, ValueError, RuntimeError) as e_extract:
            print(f"Error extracting keypoints for pose interpolation: {e_extract}. Filling remaining guide frames with black.")
            traceback.print_exc()
            black_frame = create_color_frame(resolution, (0,0,0))
            for _ in range(total_frames - 1):
                frames_list.append(black_frame)
            create_video_from_frames_list(frames_list, output_video_path, fps, resolution)
            return

        interpolated_sequence = transform_all_keypoints(keypoints_from, keypoints_to, total_frames, interpolation)

        # landmarkType for gen_skeleton_with_face_hands should indicate absolute coordinates
        # as extract_pose_keypoints now returns absolute coordinates scaled to 'resolution'
        landmark_type_for_gen = "AbsoluteCoords"

        for i in range(1, total_frames):
            if i < len(interpolated_sequence):
                frame_data = interpolated_sequence[i]
                pose_kps = frame_data.get('pose_keypoints_2d', [])
                face_kps = frame_data.get('face_keypoints_2d', []) if include_face else []
                hand_left_kps = frame_data.get('hand_left_keypoints_2d', []) if include_hands else []
                hand_right_kps = frame_data.get('hand_right_keypoints_2d', []) if include_hands else []

                img = gen_skeleton_with_face_hands(
                    pose_kps, face_kps, hand_left_kps, hand_right_kps,
                    canvas_width, canvas_height,
                    landmark_type_for_gen, # Keypoints are already absolute
                    confidence_threshold
                )
                frames_list.append(img)
            else:
                dprint(f"Warning: Interpolated sequence too short at index {i} for {output_video_path}. Appending black frame.")
                frames_list.append(create_color_frame(resolution, (0,0,0)))

    if len(frames_list) != total_frames:
        dprint(f"Warning: Generated {len(frames_list)} frames for {output_video_path}, expected {total_frames}. Adjusting.")
        if len(frames_list) < total_frames:
            last_frame = frames_list[-1] if frames_list else create_color_frame(resolution, (0,0,0))
            frames_list.extend([last_frame.copy() for _ in range(total_frames - len(frames_list))])
        else:
            frames_list = frames_list[:total_frames]

    if not frames_list:
        dprint(f"Error: No frames for video {output_video_path}. Skipping creation.")
        return

    create_video_from_frames_list(frames_list, output_video_path, fps, resolution)


# --- Debug Summary Video Helpers ---
def get_resized_frame(video_path_str: str, target_size: tuple[int, int], frame_ratio: float = 0.5) -> np.ndarray | None:
    """Extracts a frame (by ratio, e.g., 0.5 for middle) from a video and resizes it."""
    video_path = Path(video_path_str)
    if not video_path.exists() or video_path.stat().st_size == 0:
        dprint(f"GET_RESIZED_FRAME: Video not found or empty: {video_path_str}")
        placeholder = create_color_frame(target_size, (10, 10, 10)) # Dark grey
        cv2.putText(placeholder, "Not Found", (10, target_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return placeholder

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            dprint(f"GET_RESIZED_FRAME: Could not open video: {video_path_str}")
            return create_color_frame(target_size, (20,20,20))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            dprint(f"GET_RESIZED_FRAME: Video has 0 frames: {video_path_str}")
            return create_color_frame(target_size, (30,30,30))

        frame_to_get = int(total_frames * frame_ratio)
        frame_to_get = max(0, min(frame_to_get, total_frames - 1)) # Clamp

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_to_get))
        ret, frame = cap.read()
        if not ret or frame is None:
            dprint(f"GET_RESIZED_FRAME: Could not read frame {frame_to_get} from: {video_path_str}")
            return create_color_frame(target_size, (40,40,40))

        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    except (OSError, ValueError, RuntimeError) as e:
        dprint(f"GET_RESIZED_FRAME: Exception processing {video_path_str}: {e}")
        return create_color_frame(target_size, (50,50,50)) # Error color
    finally:
        if cap: cap.release()


def draw_multiline_text(image, text_lines, start_pos, font, font_scale, color, thickness, line_spacing):
    x, y = start_pos
    for i, line in enumerate(text_lines):
        line_y = y + (i * (cv2.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing))
        cv2.putText(image, line, (x, line_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def generate_debug_summary_video(segments_data: list[dict], output_path: str | Path, fps: int,
                                 num_frames_for_collage: int,
                                 target_thumb_size: tuple[int, int] = (320, 180)):
    if not DEBUG_MODE: return # Only run if debug mode is on
    if not segments_data:
        dprint("GENERATE_DEBUG_SUMMARY_VIDEO: No segment data provided.")
        return

    dprint(f"Generating animated debug collage with {num_frames_for_collage} frames, at {fps} FPS.")

    thumb_w, thumb_h = target_thumb_size
    padding = 10
    header_h = 50
    text_line_h_approx = 20
    max_settings_lines = 6
    settings_area_h = (text_line_h_approx * max_settings_lines) + padding

    num_segments = len(segments_data)
    col_w = thumb_w + (2 * padding)
    canvas_w = num_segments * col_w
    canvas_h = header_h + (thumb_h * 2) + (padding * 3) + settings_area_h + padding

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_small = 0.4
    font_scale_title = 0.6
    text_color = (230, 230, 230)
    title_color = (255, 255, 255)
    line_thickness = 1

    overall_static_template_canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 30), dtype=np.uint8)
    for idx, seg_data in enumerate(segments_data):
        col_x_start = idx * col_w
        center_x_col = col_x_start + col_w // 2
        title_text = f"Segment {seg_data['segment_index']}"
        (tw, th), _ = cv2.getTextSize(title_text, font, font_scale_title, line_thickness)
        cv2.putText(overall_static_template_canvas, title_text, (center_x_col - tw//2, header_h - padding), font, font_scale_title, title_color, line_thickness, cv2.LINE_AA)

        y_offset = header_h
        cv2.putText(overall_static_template_canvas, "Input Guide", (col_x_start + padding, y_offset + text_line_h_approx), font, font_scale_small, text_color, line_thickness)
        y_offset += thumb_h + padding
        cv2.putText(overall_static_template_canvas, "Headless Output", (col_x_start + padding, y_offset + text_line_h_approx), font, font_scale_small, text_color, line_thickness)
        y_offset += thumb_h + padding

        settings_y_start = y_offset
        cv2.putText(overall_static_template_canvas, "Settings:", (col_x_start + padding, settings_y_start + text_line_h_approx), font, font_scale_small, text_color, line_thickness)
        settings_text_lines = []
        payload = seg_data.get("task_payload", {})
        settings_text_lines.append(f"Task ID: {payload.get('task_id', 'N/A')[:10]}...")
        prompt_short = payload.get('prompt', 'N/A')[:35] + ("..." if len(payload.get('prompt', '')) > 35 else "")
        settings_text_lines.append(f"Prompt: {prompt_short}")
        settings_text_lines.append(f"Seed: {payload.get('seed', 'N/A')}, Frames: {payload.get('frames', 'N/A')}")
        settings_text_lines.append(f"Resolution: {payload.get('resolution', 'N/A')}")
        draw_multiline_text(overall_static_template_canvas, settings_text_lines[:max_settings_lines],
                            (col_x_start + padding, settings_y_start + text_line_h_approx + padding),
                            font, font_scale_small, text_color, line_thickness, 5)

    error_placeholder_frame = create_color_frame(target_thumb_size, (50, 0, 0))
    cv2.putText(error_placeholder_frame, "ERR", (10, target_thumb_size[1]//2), font, 0.8, (255,255,255), 1)
    not_found_placeholder_frame = create_color_frame(target_thumb_size, (0, 50, 0))
    cv2.putText(not_found_placeholder_frame, "N/A", (10, target_thumb_size[1]//2), font, 0.8, (255,255,255), 1)
    static_thumbs_cache = {}
    for seg_idx_cache, seg_data_cache in enumerate(segments_data):
        guide_thumb = get_resized_frame(seg_data_cache["guide_video_path"], target_thumb_size, frame_ratio=0.5)
        output_thumb = get_resized_frame(seg_data_cache["raw_headless_output_path"], target_thumb_size, frame_ratio=0.5)

        static_thumbs_cache[seg_idx_cache] = {
            'guide': guide_thumb if guide_thumb is not None else not_found_placeholder_frame,
            'output': output_thumb if output_thumb is not None else not_found_placeholder_frame
        }

    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (canvas_w, canvas_h))
        if not writer.isOpened():
            dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Failed to open VideoWriter for {output_path}")
            return

        dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Writing sequentially animated collage to {output_path}")

        for active_seg_idx in range(num_segments):
            dprint(f"Animating segment {active_seg_idx} in collage...")
            caps_for_active_segment = {'guide': None, 'output': None, 'last_frames': {}}
            video_paths_to_load = {
                'guide': segments_data[active_seg_idx]["guide_video_path"],
                'output': segments_data[active_seg_idx]["raw_headless_output_path"]
            }
            for key, path_str in video_paths_to_load.items():
                p = Path(path_str)
                if p.exists() and p.stat().st_size > 0:
                    cap_video = cv2.VideoCapture(str(p))
                    if cap_video.isOpened():
                        caps_for_active_segment[key] = cap_video
                        ret, frame = cap_video.read();
                        caps_for_active_segment['last_frames'][key] = cv2.resize(frame, target_thumb_size, cv2.INTER_AREA) if ret and frame is not None else error_placeholder_frame
                        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    else: caps_for_active_segment['last_frames'][key] = error_placeholder_frame
                else: caps_for_active_segment['last_frames'][key] = not_found_placeholder_frame

            for frame_num in range(num_frames_for_collage):
                current_frame_canvas = overall_static_template_canvas.copy()

                for display_seg_idx in range(num_segments):
                    col_x_start = display_seg_idx * col_w
                    current_y_pos = header_h

                    videos_to_composite = [None, None] # guide, output

                    if display_seg_idx == active_seg_idx:
                        if caps_for_active_segment['guide']:
                            ret, frame = caps_for_active_segment['guide'].read()
                            if ret and frame is not None: videos_to_composite[0] = cv2.resize(frame, target_thumb_size); caps_for_active_segment['last_frames']['guide'] = videos_to_composite[0]
                            else: videos_to_composite[0] = caps_for_active_segment['last_frames'].get('guide', error_placeholder_frame)
                        else: videos_to_composite[0] = caps_for_active_segment['last_frames'].get('guide', not_found_placeholder_frame)
                        if caps_for_active_segment['output']:
                            ret, frame = caps_for_active_segment['output'].read()
                            if ret and frame is not None: videos_to_composite[1] = cv2.resize(frame, target_thumb_size); caps_for_active_segment['last_frames']['output'] = videos_to_composite[1]
                            else: videos_to_composite[1] = caps_for_active_segment['last_frames'].get('output', error_placeholder_frame)
                        else: videos_to_composite[1] = caps_for_active_segment['last_frames'].get('output', not_found_placeholder_frame)
                    else:
                        videos_to_composite[0] = static_thumbs_cache[display_seg_idx]['guide']
                        videos_to_composite[1] = static_thumbs_cache[display_seg_idx]['output']

                    current_frame_canvas[current_y_pos : current_y_pos + thumb_h, col_x_start + padding : col_x_start + padding + thumb_w] = videos_to_composite[0]
                    current_y_pos += thumb_h + padding
                    current_frame_canvas[current_y_pos : current_y_pos + thumb_h, col_x_start + padding : col_x_start + padding + thumb_w] = videos_to_composite[1]

                writer.write(current_frame_canvas)

            if caps_for_active_segment['guide']: caps_for_active_segment['guide'].release()
            if caps_for_active_segment['output']: caps_for_active_segment['output'].release()
            dprint(f"Finished animating segment {active_seg_idx} in collage.")

        dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Finished writing sequentially animated debug collage.")

    except (OSError, ValueError, RuntimeError) as e:
        dprint(f"GENERATE_DEBUG_SUMMARY_VIDEO: Exception during video writing: {e} - {traceback.format_exc()}")
    finally:
        if writer: writer.release()
        dprint("GENERATE_DEBUG_SUMMARY_VIDEO: Video writer released.")
