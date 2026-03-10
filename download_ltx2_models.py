#!/usr/bin/env python3
"""Download all LTX-2 19B model files from HuggingFace."""

import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "DeepBeepMeep/LTX-2"
CKPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP", "ckpts")

# Individual safetensors files to download into ckpts/
FILES = [
    "ltx-2-19b-dev-fp8_diffusion_model.safetensors",
    "ltx-2-19b_vae.safetensors",
    "ltx-2-19b_audio_vae.safetensors",
    "ltx-2-19b_vocoder.safetensors",
    "ltx-2-19b_text_embedding_projection.safetensors",
    "ltx-2-19b-dev_embeddings_connector.safetensors",
    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
]

# Gemma subfolder to download
GEMMA_SUBFOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"


def main():
    os.makedirs(CKPTS_DIR, exist_ok=True)
    print(f"Downloading LTX-2 19B models to {CKPTS_DIR}\n")

    # Download individual files
    for filename in FILES:
        dest = os.path.join(CKPTS_DIR, filename)
        if os.path.isfile(dest):
            size_gb = os.path.getsize(dest) / 1024**3
            print(f"  SKIP (exists, {size_gb:.1f} GB): {filename}")
            continue
        print(f"  Downloading: {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=CKPTS_DIR,
            local_dir_use_symlinks=False,
        )
        size_gb = os.path.getsize(dest) / 1024**3
        print(f"  OK ({size_gb:.1f} GB): {filename}")

    # Download gemma subfolder
    gemma_dir = os.path.join(CKPTS_DIR, GEMMA_SUBFOLDER)
    if os.path.isdir(gemma_dir) and os.listdir(gemma_dir):
        print(f"\n  SKIP (exists): {GEMMA_SUBFOLDER}/")
    else:
        print(f"\n  Downloading: {GEMMA_SUBFOLDER}/ ...")
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{GEMMA_SUBFOLDER}/*",
            local_dir=CKPTS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"  OK: {GEMMA_SUBFOLDER}/")

    # Also download preload files (lora control nets) from config
    preload_files = [
        "ltx-2-19b-ic-lora-pose-control.safetensors",
        "ltx-2-19b-ic-lora-depth-control.safetensors",
        "ltx-2-19b-ic-lora-canny-control.safetensors",
        "ltx-2-19b-distilled-lora-384.safetensors",
    ]
    print("\n--- Preload / LoRA files ---")
    for filename in preload_files:
        dest = os.path.join(CKPTS_DIR, filename)
        if os.path.isfile(dest):
            size_gb = os.path.getsize(dest) / 1024**3
            print(f"  SKIP (exists, {size_gb:.1f} GB): {filename}")
            continue
        print(f"  Downloading: {filename} ...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=CKPTS_DIR,
                local_dir_use_symlinks=False,
            )
            size_gb = os.path.getsize(dest) / 1024**3
            print(f"  OK ({size_gb:.1f} GB): {filename}")
        except Exception as e:
            print(f"  WARNING: Could not download {filename}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
