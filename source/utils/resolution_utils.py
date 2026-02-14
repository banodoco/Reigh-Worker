"""Resolution parsing and snapping utilities."""

__all__ = [
    "snap_resolution_to_model_grid",
    "parse_resolution",
    "get_model_grid_size",
]


def get_model_grid_size(model_name: str) -> int:
    """Return the resolution grid size for a model.

    LTX-2 requires multiples of 64; Wan and other models use 16.
    """
    if "ltx2" in model_name.lower():
        return 64
    return 16


def snap_resolution_to_model_grid(parsed_res: tuple[int, int], *, grid_size: int = 16) -> tuple[int, int]:
    """
    Snaps resolution to model grid requirements (multiples of *grid_size*).

    Args:
        parsed_res: (width, height) tuple
        grid_size: Grid multiple to snap to (default 16, LTX-2 uses 64)

    Returns:
        (width, height) tuple snapped to nearest valid values
    """
    width, height = parsed_res
    width = (width // grid_size) * grid_size
    height = (height // grid_size) * grid_size
    return width, height


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parses 'WIDTHxHEIGHT' string to (width, height) tuple."""
    try:
        w, h = map(int, res_str.split('x'))
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive.")
        return w, h
    except ValueError as e:
        raise ValueError(f"Resolution string must be in WIDTHxHEIGHT format with positive integers (e.g., '960x544'), got {res_str}. Error: {e}") from e
