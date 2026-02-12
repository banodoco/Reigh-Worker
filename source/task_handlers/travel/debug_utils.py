"""Debug and monitoring utilities for travel between images tasks."""

import os
from pathlib import Path

# Import structured logging
from ...core.log import travel_logger

# RAM monitoring
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    travel_logger.warning("psutil not available - RAM monitoring disabled")

try:
    import cv2
    _COLOR_MATCH_DEPS_AVAILABLE = True
except ImportError:
    _COLOR_MATCH_DEPS_AVAILABLE = False

from ...utils import get_video_frame_count_and_fps
from .svi_config import BYTES_PER_MB, MB_PER_GB, BYTES_PER_GB


# Add debugging helper function
def debug_video_analysis(video_path: str | Path, label: str, task_id: str = "unknown") -> dict:
    """Analyze a video file and return comprehensive debug info"""
    try:
        path_obj = Path(video_path)
        if not path_obj.exists():
            travel_logger.debug(f"{label}: FILE MISSING - {video_path}", task_id=task_id)
            return {"exists": False, "path": str(video_path)}

        frame_count, fps = get_video_frame_count_and_fps(str(path_obj))
        file_size = path_obj.stat().st_size
        duration = frame_count / fps if fps and fps > 0 else 0

        debug_info = {
            "exists": True,
            "path": str(path_obj.resolve()),
            "frame_count": frame_count,
            "fps": fps,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / BYTES_PER_MB, 2)
        }

        travel_logger.debug(f"{label}: {debug_info['frame_count']} frames, {debug_info['fps']} fps, {debug_info['duration_seconds']:.2f}s, {debug_info['file_size_mb']} MB", task_id=task_id)

        # Lightweight color diagnostic: sample a few frames and compute mean BGR.
        # Helps detect "brown tint" / channel or range issues without decoding entire video.
        try:
            if frame_count and frame_count > 0:
                cap = cv2.VideoCapture(str(path_obj))
                if cap.isOpened():
                    sample_idxs = [0, max(0, int(frame_count // 2)), max(0, int(frame_count - 1))]
                    samples = {}
                    for idx in sample_idxs:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                        ok, fr = cap.read()
                        if not ok or fr is None:
                            samples[str(idx)] = None
                            continue
                        # OpenCV frames are BGR uint8; mean per channel is robust for tint detection.
                        mean_bgr = tuple(float(x) for x in fr.mean(axis=(0, 1)))
                        samples[str(idx)] = {
                            "mean_bgr": mean_bgr,
                            "shape": tuple(int(x) for x in fr.shape),
                        }
                    cap.release()
                    debug_info["frame_color_samples"] = samples
                    travel_logger.debug(f"[FrameBrowningIssue] {label}: frame_color_samples={samples}", task_id=task_id)
        except (OSError, ValueError, RuntimeError) as e_color:
            # Never fail the pipeline due to debug sampling.
            travel_logger.debug(f"[FrameBrowningIssue] {label}: color sample failed: {e_color}", task_id=task_id)

        return debug_info

    except (OSError, ValueError, RuntimeError) as e:
        travel_logger.debug(f"{label}: ERROR analyzing video - {e}", task_id=task_id)
        return {"exists": False, "error": str(e), "path": str(video_path)}

# RAM monitoring helper function
def log_ram_usage(label: str, task_id: str = "unknown", logger=None) -> dict:
    """
    Log current RAM usage with a descriptive label.
    Returns dict with RAM metrics for programmatic use.
    """
    if logger is None:
        logger = travel_logger

    if not _PSUTIL_AVAILABLE:
        return {"available": False}

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / BYTES_PER_MB
        rss_gb = rss_mb / MB_PER_GB

        # Get system-wide memory stats
        sys_mem = psutil.virtual_memory()
        sys_total_gb = sys_mem.total / BYTES_PER_GB
        sys_available_gb = sys_mem.available / BYTES_PER_GB
        sys_used_percent = sys_mem.percent

        logger.info(
            f"[RAM] {label}: Process={rss_mb:.0f}MB ({rss_gb:.2f}GB) | "
            f"System={sys_used_percent:.1f}% used, {sys_available_gb:.1f}GB/{sys_total_gb:.1f}GB available",
            task_id=task_id
        )

        return {
            "available": True,
            "process_rss_mb": rss_mb,
            "process_rss_gb": rss_gb,
            "system_total_gb": sys_total_gb,
            "system_available_gb": sys_available_gb,
            "system_used_percent": sys_used_percent
        }

    except (ProcessLookupError, OSError) as e:
        logger.warning(f"[RAM] Failed to get RAM usage: {e}", task_id=task_id)
        return {"available": False, "error": str(e)}
