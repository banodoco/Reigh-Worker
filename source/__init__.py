# Explicit imports for modules used via deferred/dynamic loading.
# This makes static analysis aware of the dependency graph.
from source.models.wgp import wgp_patches  # noqa: F401  — used by headless_wgp
from source.media.video import vace_frame_utils  # noqa: F401  — used by inpaint_frames, join_clips
from source.media.video import hires_utils  # noqa: F401  — used by Wan2GP/models/qwen/qwen_main
from source.models.lora import lora_utils  # noqa: F401  — used by headless_wgp, task_conversion
from source.task_handlers.worker import heartbeat_utils  # noqa: F401  — used by worker.py
from source.task_handlers.worker import fatal_error_handler  # noqa: F401  — used by headless_model_management
