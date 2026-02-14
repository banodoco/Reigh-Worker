"""Resolution parsing and snapping utilities."""

__all__ = [
    "snap_resolution_to_model_grid",
    "parse_resolution",
]


def snap_resolution_to_model_grid(parsed_res: tuple[int, int]) -> tuple[int, int]:
    """
    Snaps resolution to model grid requirements (multiples of 16).

    Args:
        parsed_res: (width, height) tuple

    Returns:
        (width, height) tuple snapped to nearest valid values
    """
    width, height = parsed_res
    # Ensure resolution is compatible with model requirements (multiples of 16)
    width = (width // 16) * 16
    height = (height // 16) * 16
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
