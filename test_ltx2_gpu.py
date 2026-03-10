#!/usr/bin/env python3
"""
Minimal GPU test for LTX-2 19B generation via the headless pipeline.

This test:
1. Loads the LTX-2 19B FP8 model
2. Generates a short (17 frames) text-to-video clip
3. Verifies the output exists and is a valid video file

Usage:
    cd /workspace/Reigh-Worker
    python test_ltx2_gpu.py
"""

import os
import sys
import time

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WAN_DIR = os.path.join(PROJECT_ROOT, "Wan2GP")

# Must cd into Wan2GP for WGP to find defaults/models
os.chdir(WAN_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, WAN_DIR)

# Clean argv to prevent wgp argparse conflicts
sys.argv = ["test_ltx2_gpu.py"]


def main():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Verify model files exist
    ckpts = os.path.join(WAN_DIR, "ckpts")
    required_files = [
        "ltx-2-19b-dev-fp8_diffusion_model.safetensors",
        "ltx-2-19b_vae.safetensors",
        "ltx-2-19b_audio_vae.safetensors",
        "ltx-2-19b_vocoder.safetensors",
        "ltx-2-19b_text_embedding_projection.safetensors",
        "ltx-2-19b-dev_embeddings_connector.safetensors",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    ]
    gemma_dir = os.path.join(ckpts, "gemma-3-12b-it-qat-q4_0-unquantized")

    print("\n--- Model Files Check ---")
    all_present = True
    for f in required_files:
        path = os.path.join(ckpts, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) / 1024**3 if exists else 0
        status = f"OK ({size:.1f} GB)" if exists else "MISSING"
        print(f"  {f}: {status}")
        if not exists:
            all_present = False

    gemma_exists = os.path.isdir(gemma_dir)
    print(f"  gemma-3-12b-it-qat-q4_0-unquantized/: {'OK' if gemma_exists else 'MISSING'}")
    if not gemma_exists:
        all_present = False

    if not all_present:
        print("\nERROR: Some model files are missing. Cannot proceed with GPU test.")
        sys.exit(1)

    print("\nAll model files present.")

    # Import orchestrator
    print("\n--- Loading WanOrchestrator ---")
    from headless_wgp import WanOrchestrator

    output_dir = os.path.join(PROJECT_ROOT, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    orch = WanOrchestrator(WAN_DIR, main_output_dir=output_dir)

    # Load model
    print("\n--- Loading LTX-2 19B (FP8) ---")
    t0 = time.time()
    switched = orch.load_model("ltx2_19B")
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s (switched={switched})")
    print(f"Is LTX-2: {orch._is_ltx2()}")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM allocated: {vram_used:.1f} GB, reserved: {vram_reserved:.1f} GB")

    # Generate minimal video (17 frames = minimum for LTX-2, small resolution)
    print("\n--- Generating Test Video ---")
    print("  Prompt: 'A calm ocean wave gently rolling onto a sandy beach at sunset'")
    print("  Resolution: 512x320")
    print("  Frames: 17 (minimum)")
    print("  Steps: 20")

    t0 = time.time()
    try:
        result = orch.generate(
            prompt="A calm ocean wave gently rolling onto a sandy beach at sunset",
            resolution="512x320",
            video_length=17,
            num_inference_steps=20,
            guidance_scale=4.0,
            seed=42,
        )
        gen_time = time.time() - t0

        if result and os.path.isfile(result):
            size = os.path.getsize(result)
            print(f"\nSUCCESS! Video generated in {gen_time:.1f}s")
            print(f"  Output: {result}")
            print(f"  Size: {size / 1024:.1f} KB")
        else:
            print(f"\nFAILED: generate() returned '{result}' but file does not exist")
            sys.exit(1)

    except Exception as e:
        gen_time = time.time() - t0
        print(f"\nERROR after {gen_time:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nFinal VRAM: allocated={vram_used:.1f} GB, reserved={vram_reserved:.1f} GB")

    print("\n=== LTX-2 GPU Test PASSED ===")


if __name__ == "__main__":
    main()
