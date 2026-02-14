"""
Qwen prompt and system-instruction formatting utilities.

Centralises system prompt selection and prompt modification logic
used across the various Qwen task handlers.

Extracted from qwen_handler.py.
"""

from typing import Any, Dict


# ── Default system prompts per task category ────────────────────────────

SYSTEM_PROMPT_IMAGE_EDIT = (
    "You are a professional image editor. Analyze the input image carefully, "
    "then apply the requested modifications precisely while maintaining visual "
    "coherence and image quality."
)

SYSTEM_PROMPT_INPAINT = (
    "You are an expert at inpainting. The green areas indicate regions to fill. "
    "Analyze the context and generate natural content based on the description."
)

SYSTEM_PROMPT_ANNOTATED_EDIT = (
    "You are an expert at interpreting visual annotations. Analyze the green "
    "annotations and modify the marked areas according to instructions."
)

SYSTEM_PROMPT_IMAGE_GEN = (
    "You are a professional image generator. Create high-quality, detailed "
    "images based on the description provided."
)

SYSTEM_PROMPT_IMAGE_HIRES = (
    "You are a professional image generator. Create high-quality, detailed "
    "images based on the description."
)

SYSTEM_PROMPT_IMAGE_2512 = (
    "You are an expert image generator specializing in photorealistic images "
    "with accurate text rendering. Pay careful attention to any text, typography, "
    "or lettering in the prompt and render it clearly and legibly."
)

SYSTEM_PROMPT_TURBO = "Generate an image based on the description."

SYSTEM_PROMPT_IMG2IMG = "You are an expert at image-to-image generation."


# ── Style-task system prompt selection ──────────────────────────────────

def select_style_system_prompt(
    has_subject: bool,
    has_style: bool,
    has_scene: bool,
) -> str:
    """Pick the most specific system prompt for a style task based on active references."""
    if has_subject and has_style and has_scene:
        return "You are an expert at creating images with consistent subjects, styles, and scenes."
    if has_subject and has_style:
        return "You are an expert at creating images with consistent subjects and styles."
    if has_style:
        return "You are an expert at applying artistic styles consistently."
    return SYSTEM_PROMPT_IMG2IMG


# ── Shared helpers ──────────────────────────────────────────────────────

def apply_system_prompt(
    db_task_params: Dict[str, Any],
    generation_params: Dict[str, Any],
    default_prompt: str,
) -> None:
    """
    Set generation_params["system_prompt"] from the task params if supplied,
    otherwise fall back to *default_prompt*.
    """
    custom = db_task_params.get("system_prompt")
    if custom:
        generation_params["system_prompt"] = custom
    else:
        generation_params["system_prompt"] = default_prompt


def build_style_prompt(
    original_prompt: str,
    style_strength: float,
    subject_strength: float,
    subject_description: str,
    in_this_scene: bool,
) -> str:
    """
    Build the modified prompt for style-transfer tasks.

    Prepends style / subject preamble tokens depending on which strengths
    are active, then appends the original user prompt.

    Returns the (possibly modified) prompt string.
    """
    parts: list[str] = []
    has_style_prefix = False

    if style_strength > 0.0:
        parts.append("In the style of this image,")
        has_style_prefix = True

    if subject_strength > 0.0 and subject_description:
        make_word = "make" if has_style_prefix else "Make"
        if in_this_scene:
            parts.append(f"{make_word} an image of this {subject_description} in this scene:")
        else:
            parts.append(f"{make_word} an image of this {subject_description}:")

    if parts:
        return " ".join(parts) + " " + original_prompt
    return original_prompt
