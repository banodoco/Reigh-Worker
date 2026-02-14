# source/lora_paths.py
"""
Centralized LoRA directory path definitions for the Headless-Wan2GP system.

This module provides the single source of truth for LoRA directory locations.
Import from here instead of hardcoding paths across modules.
"""
from pathlib import Path
from typing import List


def get_lora_search_dirs(wan_dir: Path, repo_root: Path = None) -> List[Path]:
    """
    Get all standard LoRA search directories.

    Args:
        wan_dir: Path to the Wan2GP directory
        repo_root: Optional path to the repo root (for checking stray files)

    Returns:
        List of Path objects for all LoRA directories to search
    """
    dirs = [
        # Wan2GP LoRA directories
        wan_dir / "loras",
        wan_dir / "loras" / "wan",
        wan_dir / "loras_i2v",
        wan_dir / "loras_hunyuan",
        wan_dir / "loras_hunyuan" / "1.5",
        wan_dir / "loras_hunyuan_i2v",
        wan_dir / "loras_flux",
        wan_dir / "loras_qwen",
        wan_dir / "loras_ltxv",
        wan_dir / "loras_kandinsky5",
    ]

    # Also check repo root for any stray files from previous bugs
    if repo_root:
        dirs.extend([
            repo_root / "loras",
            repo_root / "loras" / "wan",
        ])

    return dirs


def get_lora_dir_for_model(model_type: str, wan_dir: Path) -> Path:
    """
    Get the appropriate LoRA directory for a given model type.

    Args:
        model_type: The model type identifier (e.g., "vace", "hunyuan", "flux")
        wan_dir: Path to the Wan2GP directory

    Returns:
        Path to the appropriate LoRA directory
    """
    if not model_type:
        return wan_dir / "loras"

    model_lower = model_type.lower()

    # Wan 2.x / VACE models -> loras/wan
    if "wan" in model_lower or "vace" in model_lower:
        return wan_dir / "loras" / "wan"

    # Hunyuan models - check I2V first (more specific), then 1.5, then general
    if "hunyuan" in model_lower:
        if "i2v" in model_lower:
            return wan_dir / "loras_hunyuan_i2v"
        elif "1_5" in model_lower or "1.5" in model_lower:
            return wan_dir / "loras_hunyuan" / "1.5"
        else:
            return wan_dir / "loras_hunyuan"

    # Flux models
    if "flux" in model_lower:
        return wan_dir / "loras_flux"

    # Qwen models
    if "qwen" in model_lower:
        return wan_dir / "loras_qwen"

    # LTXV models
    if "ltxv" in model_lower:
        return wan_dir / "loras_ltxv"

    # Kandinsky models
    if "kandinsky" in model_lower:
        return wan_dir / "loras_kandinsky5"

    # Default fallback
    return wan_dir / "loras"
