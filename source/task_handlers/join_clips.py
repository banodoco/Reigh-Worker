"""Join clips task handlers - re-export facade for backward compatibility."""
from source.task_handlers.join.generation import *  # noqa: F401,F403
from source.task_handlers.join.final_stitch import *  # noqa: F401,F403
from source.task_handlers.join.vace_quantization import *  # noqa: F401,F403

# Explicit re-exports for underscore-prefixed names (star import skips these)
from source.task_handlers.join.generation import _handle_join_clips_task  # noqa: F401
from source.task_handlers.join.final_stitch import _handle_join_final_stitch  # noqa: F401
from source.task_handlers.join.vace_quantization import _calculate_vace_quantization  # noqa: F401
