"""Reference parameters for the LTX-2 multi-frame guided video workflow.

Derived from tests/Multi-frame video Ltx2.json ComfyUI workflow which uses
LTXVAddGuideMulti to place 4 guide images at different frame positions.
"""

LTX2_MULTIFRAME_PARAMS = {
    "prompt": "3d animated cartoon character is reading a book",
    "negative_prompt": "blurry, low quality, watermark",
    "resolution": "768x512",
    "video_length": 105,
    "seed": 10,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guide_images": [
        {"image": "img1.png", "frame_idx": 0, "strength": 1.0},
        {"image": "img2.png", "frame_idx": 40, "strength": 1.0},
        {"image": "img3.png", "frame_idx": 80, "strength": 1.0},
        {"image": "img4.png", "frame_idx": -1, "strength": 1.0},
    ],
}
