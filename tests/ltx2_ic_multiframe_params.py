"""Reference parameters for combined IC LoRA + multi-frame guide images workflow.

Combines structural control (pose via IC LoRA) with multiple guide images
placed at specific frame positions via latent injection.
"""

LTX2_IC_MULTIFRAME_PARAMS = {
    "prompt": "a young woman dancing in her room, cinematic lighting",
    "negative_prompt": "blurry, low quality, watermark",
    "resolution": "768x512",
    "video_length": 97,
    "seed": 42,
    "num_inference_steps": 20,
    "guidance_scale": 4.0,

    # IC LoRA control
    "video_prompt_type": "PVG",
    "control_net_weight": 0.8,

    # Multi-frame guide images at specific positions
    "guide_images": [
        {"image": "guide_0.png", "frame_idx": 0, "strength": 1.0},
        {"image": "guide_32.png", "frame_idx": 32, "strength": 1.0},
        {"image": "guide_64.png", "frame_idx": 64, "strength": 1.0},
        {"image": "guide_last.png", "frame_idx": -1, "strength": 1.0},
    ],
}
