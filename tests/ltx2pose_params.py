"""ltx2pose.json ComfyUI workflow → WanOrchestrator.generate() parameter mapping.

This module provides the reference parameter dictionary extracted from the
ComfyUI workflow (tests/ltx2pose.json) translated to Wan2GP pipeline arguments.
Used by smoke tests to verify parameter correctness.
"""

LTX2_POSE_WORKFLOW_PARAMS = {
    # Core generation parameters
    "prompt": "a young woman dancing in her room",
    "negative_prompt": "blurry, low quality, still frame, frames, watermark, overlay, titles",
    "resolution": "768x512",
    "video_length": 97,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,       # CFG=1 (distilled mode, no CFG)
    "flow_shift": 2.05,          # LTXVScheduler sigma shift
    "seed": 42320326744877,

    # IC-LoRA Control
    "video_prompt_type": "PVG",  # P=pose, V=video guide, G=guidance
    "control_net_weight": 0.95,  # end_percent from LTXAddVideoICLoRAGuide
    "denoising_strength": 1.0,   # Full denoise (IC-LoRA controls via guide, not noise)

    # LoRAs
    # Union: ltx-2-19b-ic-lora-union-control.safetensors (strength=1.0)
    # Distill: ltx-2-19b-distilled-lora-384.safetensors (strength=0.7)
    "activated_loras": ["ltx-2-19b-distilled-lora-384.safetensors"],
    "loras_multipliers": "0.7",

    # Notes:
    # - Second stage upscaling: Wan2GP uses its own spatial upsampler (not Flux2)
    # - Frame rate: workflow uses 25fps but video is 24fps - use 24 for consistency
}
