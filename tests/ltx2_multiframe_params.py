"""LTX-2 Multi-Frame workflow reference parameters.

Extracted from the ComfyUI workflow (tests/Multi-frame video Ltx2.json)
embedded API format. Used by smoke tests to verify parameter correctness.
"""

LTX2_MULTIFRAME_WORKFLOW_PARAMS = {
    # Core generation
    "prompt": "3d animated cartoon character is reading a book",
    "negative_prompt": "blurry, low quality, watermark",
    "width": 768,
    "height": 512,
    "num_frames": 105,
    "fps": 25.0,
    "steps": 20,
    "seed": 10,
    "sampler": "euler",

    # Scheduler
    "max_shift": 2.05,
    "base_shift": 0.95,
    "terminal": 0.1,

    # Guider CFG
    "video_cfg": 3.0,
    "audio_cfg": 7.0,

    # Guide frame positions (images 1-3)
    "frame_indices": [40, 80, -1],
    "guide_strengths": [1.0, 1.0, 1.0],

    # Image 4 preprocessing
    "img_compression": 35,
    "img4_strength": 1.0,

    # Models
    "ckpt_name": "ltx-av-step-1751000_vocoder_24K.safetensors",
    "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized_readout_proj/model/model.safetensors",
}
