"""Reference parameters for the unified ltx2_controlled task type.

Covers all combinations of IC-LoRA, standard LoRA, and image guide controls.
"""

# IC-LoRA only (pose + guide video)
LTX2_CONTROLLED_IC_ONLY = {
    "prompt": "a woman performing yoga poses, smooth motion, cinematic",
    "negative_prompt": "blurry, low quality",
    "resolution": "768x512",
    "video_length": 97,
    "seed": 42,
    "num_inference_steps": 20,
    "guidance_scale": 4.0,
    "ic_loras": [
        {
            "type": "pose",
            "weight": 0.8,
            "guide_video": "pose_guide.mp4",
        }
    ],
}

# Standard LoRA only
LTX2_CONTROLLED_LORA_ONLY = {
    "prompt": "a cat sitting on a windowsill, golden hour light",
    "resolution": "1024x576",
    "video_length": 121,
    "seed": 7,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "loras": [
        {"path": "https://huggingface.co/user/style_lora.safetensors", "weight": 1.0},
    ],
}

# Image guides only (multiple anchors per image)
LTX2_CONTROLLED_GUIDES_ONLY = {
    "prompt": "3d animated cartoon character is reading a book",
    "resolution": "768x512",
    "video_length": 105,
    "seed": 10,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "image_guides": [
        {
            "image": "img_start.png",
            "anchors": [
                {"frame": 0, "weight": 1.0},
            ],
        },
        {
            "image": "img_mid.png",
            "anchors": [
                {"frame": 50, "weight": 0.9},
            ],
        },
        {
            "image": "img_end.png",
            "anchors": [
                {"frame": -1, "weight": 1.0},
            ],
        },
    ],
}

# All three combined: IC-LoRA + standard LoRA + image guides
LTX2_CONTROLLED_ALL_COMBINED = {
    "prompt": "a young woman dancing in her room, cinematic lighting",
    "negative_prompt": "blurry, low quality, watermark",
    "resolution": "768x512",
    "video_length": 97,
    "seed": 42,
    "num_inference_steps": 30,
    "guidance_scale": 4.0,
    "ic_loras": [
        {
            "type": "pose",
            "weight": 0.8,
            "guide_video": "pose_guide.mp4",
        }
    ],
    "loras": [
        {"path": "https://huggingface.co/user/style_lora.safetensors", "weight": 0.7},
    ],
    "image_guides": [
        {
            "image": "guide_start.png",
            "anchors": [
                {"frame": 0, "weight": 1.0},
                {"frame": 48, "weight": 0.8},
            ],
        },
        {
            "image": "guide_end.png",
            "anchors": [
                {"frame": -1, "weight": 1.0},
            ],
        },
    ],
}

# Edge case: multiple IC-LoRA types (pose + depth)
LTX2_CONTROLLED_MULTI_IC = {
    "prompt": "a dancer performing contemporary dance, studio lighting",
    "resolution": "768x512",
    "video_length": 97,
    "seed": 100,
    "num_inference_steps": 25,
    "guidance_scale": 4.0,
    "ic_loras": [
        {"type": "pose", "weight": 0.8, "guide_video": "pose_guide.mp4"},
        {"type": "depth", "weight": 0.6},
    ],
}

# Edge case: single anchor at last frame
LTX2_CONTROLLED_SINGLE_ANCHOR_LAST = {
    "prompt": "a timelapse of a flower blooming",
    "resolution": "1024x576",
    "video_length": 121,
    "seed": -1,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "image_guides": [
        {
            "image": "flower_bloom.png",
            "anchors": [
                {"frame": -1, "weight": 1.0},
            ],
        },
    ],
}
