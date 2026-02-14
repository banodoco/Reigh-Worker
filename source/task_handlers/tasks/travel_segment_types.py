"""TypedDict definitions for travel segment task payloads.

These provide IDE autocomplete and type checking for the dict-heavy
task_registry.py code. All fields are optional (total=False) since payloads
arrive from the database and may have missing keys.
"""
from __future__ import annotations

from typing import List, TypedDict


class IndividualSegmentParams(TypedDict, total=False):
    """Per-segment override parameters.

    Stored under ``task_params_dict["individual_segment_params"]`` and given
    highest priority when resolving prompt, seed, frame count, images, LoRAs,
    and phase config for a single segment.
    """

    # --- Prompts ---
    enhanced_prompt: str
    base_prompt: str
    negative_prompt: str

    # --- Frame count ---
    num_frames: int

    # --- Images ---
    start_image_url: str
    end_image_url: str
    input_image_paths_resolved: List[str]

    # --- Generation tuning ---
    seed_to_use: int
    num_inference_steps: int
    guidance_scale: float

    # --- Phase config (per-segment override) ---
    phase_config: dict

    # --- LoRA (per-segment override) ---
    segment_loras: List[dict]
