"""VLM image preparation utilities."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_framed_vlm_image(start_img: Image.Image, end_img: Image.Image, border_width: int = 6) -> Image.Image:
    """
    Create a combined image with colored borders for VLM to see.

    Green border on left (start), red border on right (end).
    This helps VLM understand which image is which.

    Args:
        start_img: The starting image (will get green border)
        end_img: The ending image (will get red border)
        border_width: Width of the colored border

    Returns:
        Combined image with colored borders
    """
    from PIL import ImageOps

    # Colors
    start_color = (46, 139, 87)   # Sea green for start
    end_color = (178, 34, 34)     # Firebrick red for end

    # Add borders to each image
    start_bordered = ImageOps.expand(start_img, border=border_width, fill=start_color)
    end_bordered = ImageOps.expand(end_img, border=border_width, fill=end_color)

    # Small gap between images
    gap = 4

    # Combine side by side
    combined_width = start_bordered.width + end_bordered.width + gap
    combined_height = max(start_bordered.height, end_bordered.height)
    combined = Image.new('RGB', (combined_width, combined_height), (40, 40, 40))
    combined.paste(start_bordered, (0, 0))
    combined.paste(end_bordered, (start_bordered.width + gap, 0))

    return combined


def create_labeled_debug_image(start_img: Image.Image, end_img: Image.Image, pair_index: int = 0) -> Image.Image:
    """
    Create a labeled debug image showing start and end images side by side with clear labels and frames.

    This is for human inspection - NOT what VLM sees (VLM sees raw side-by-side without labels).

    Args:
        start_img: The starting image (left side)
        end_img: The ending image (right side)
        pair_index: The pair number for the title

    Returns:
        Combined image with labels and frames
    """
    # Settings
    border_width = 8
    label_height = 60
    gap_between = 30
    title_height = 70
    padding = 15

    # Colors
    start_color = (46, 139, 87)   # Sea green for start
    end_color = (178, 34, 34)     # Firebrick red for end
    bg_color = (40, 40, 40)       # Dark gray background
    text_color = (255, 255, 255)  # White text

    # Calculate dimensions
    img_width = max(start_img.width, end_img.width)
    img_height = max(start_img.height, end_img.height)

    # Total canvas size
    total_width = (img_width + border_width * 2) * 2 + gap_between + padding * 2
    total_height = title_height + label_height + img_height + border_width * 2 + padding * 2

    # Create canvas
    canvas = Image.new('RGB', (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fall back to default
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
        font = None
        for fp in font_paths:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, 36)
                title_font = ImageFont.truetype(fp, 42)
                break
        if font is None:
            font = ImageFont.load_default()
            title_font = font
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    title = f"VLM Debug - Pair {pair_index} (LEFT=Start \u2192 RIGHT=End)"
    try:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = bbox[2] - bbox[0]
    except (AttributeError, TypeError):
        title_width = len(title) * 10
    title_x = (total_width - title_width) // 2
    draw.text((title_x, 10), title, fill=text_color, font=title_font)

    # Calculate positions
    left_x = padding
    right_x = padding + img_width + border_width * 2 + gap_between
    images_y = title_height + label_height

    # Draw "Starting Image" label and frame
    draw.rectangle(
        [left_x, images_y, left_x + img_width + border_width * 2, images_y + img_height + border_width * 2],
        outline=start_color,
        width=border_width
    )
    start_label = "\u2190 STARTING IMAGE"
    try:
        bbox = draw.textbbox((0, 0), start_label, font=font)
        label_width = bbox[2] - bbox[0]
    except (AttributeError, TypeError):
        label_width = len(start_label) * 8
    label_x = left_x + (img_width + border_width * 2 - label_width) // 2
    draw.text((label_x, title_height + 5), start_label, fill=start_color, font=font)

    # Draw "Ending Image" label and frame
    draw.rectangle(
        [right_x, images_y, right_x + img_width + border_width * 2, images_y + img_height + border_width * 2],
        outline=end_color,
        width=border_width
    )
    end_label = "ENDING IMAGE \u2192"
    try:
        bbox = draw.textbbox((0, 0), end_label, font=font)
        label_width = bbox[2] - bbox[0]
    except (AttributeError, TypeError):
        label_width = len(end_label) * 8
    label_x = right_x + (img_width + border_width * 2 - label_width) // 2
    draw.text((label_x, title_height + 5), end_label, fill=end_color, font=font)

    # Paste images (centered within their frames if smaller)
    start_paste_x = left_x + border_width + (img_width - start_img.width) // 2
    start_paste_y = images_y + border_width + (img_height - start_img.height) // 2
    canvas.paste(start_img, (start_paste_x, start_paste_y))

    end_paste_x = right_x + border_width + (img_width - end_img.width) // 2
    end_paste_y = images_y + border_width + (img_height - end_img.height) // 2
    canvas.paste(end_img, (end_paste_x, end_paste_y))

    return canvas
