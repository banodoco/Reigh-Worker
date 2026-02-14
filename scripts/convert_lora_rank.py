#!/usr/bin/env python3
"""
Convert high-rank LoRA files to lower rank using SVD truncation.

The lightx2v Wan2.2-Distill-Loras use rank64 with .diff weights, making
inference ~3-5x slower than rank16/32 standard LoRAs. This script:
1. Recomposes each lora_A/lora_B pair into the full update matrix
2. SVD decomposes and truncates to the target rank
3. Strips non-standard keys (diff_m, norm_k_img) 
4. Keeps .diff weights for norm layers (1D, negligible cost)
5. Saves a new safetensors file in standard LoRA format

Usage:
    python convert_lora_rank.py <input.safetensors> <output.safetensors> [--rank 32]
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file


def extract_module_pairs(state_dict):
    """Group LoRA keys by module name."""
    modules = defaultdict(dict)
    other_keys = {}
    
    for k, v in state_dict.items():
        # Strip common prefixes
        k_stripped = k
        for pfx in ("diffusion_model.", "transformer."):
            if k_stripped.startswith(pfx):
                k_stripped = k_stripped[len(pfx):]
        
        if k_stripped.endswith(".lora_A.weight") or k_stripped.endswith(".lora_down.weight"):
            module = k_stripped.rsplit(".", 2)[0]
            modules[module]["lora_A"] = (k, v)
        elif k_stripped.endswith(".lora_B.weight") or k_stripped.endswith(".lora_up.weight"):
            module = k_stripped.rsplit(".", 2)[0]
            modules[module]["lora_B"] = (k, v)
        elif k_stripped.endswith(".alpha"):
            module = k_stripped[:-6]  # strip ".alpha"
            modules[module]["alpha"] = (k, v)
        else:
            other_keys[k] = v
    
    return modules, other_keys


def svd_truncate(lora_A, lora_B, target_rank, alpha=None):
    """
    Recompose and re-decompose LoRA at a lower rank via SVD.
    
    lora_A: [current_rank, in_features]
    lora_B: [out_features, current_rank]
    
    Returns new_lora_A, new_lora_B, new_alpha at target_rank.
    """
    current_rank = lora_A.shape[0]
    
    if current_rank <= target_rank:
        # Already at or below target rank, keep as-is
        return lora_A, lora_B, alpha
    
    # Compute effective scale
    scale = 1.0
    if alpha is not None:
        scale = alpha / current_rank
    
    # Recompose: W_delta = scale * lora_B @ lora_A
    W = (lora_B.float() @ lora_A.float()) * scale
    
    # SVD decompose
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    # Truncate to target rank
    U_k = U[:, :target_rank]        # [out_features, target_rank]
    S_k = S[:target_rank]            # [target_rank]
    Vt_k = Vt[:target_rank, :]       # [target_rank, in_features]
    
    # Distribute singular values: new_B = U * sqrt(S), new_A = sqrt(S) * Vt
    sqrt_S = torch.sqrt(S_k)
    new_lora_B = (U_k * sqrt_S.unsqueeze(0)).to(lora_B.dtype)   # [out_features, target_rank]
    new_lora_A = (sqrt_S.unsqueeze(1) * Vt_k).to(lora_A.dtype)  # [target_rank, in_features]
    
    # New alpha = target_rank (so effective scale = target_rank/target_rank = 1.0,
    # and the actual scaling is baked into the matrices)
    new_alpha = float(target_rank)
    
    # Compute reconstruction error
    W_reconstructed = new_lora_B.float() @ new_lora_A.float()
    error = torch.norm(W - W_reconstructed) / torch.norm(W)
    
    return new_lora_A, new_lora_B, new_alpha, error.item()


def convert_lora(input_path, output_path, target_rank=32, strip_diff_m=True, strip_norm_k_img=True):
    """Convert a LoRA file to a lower rank."""
    print(f"Loading {input_path}...")
    sd = load_file(input_path)
    
    print(f"Total keys: {len(sd)}")
    
    # Extract module pairs
    modules, other_keys = extract_module_pairs(sd)
    
    print(f"Found {len(modules)} LoRA modules")
    
    # Categorize other keys
    diff_keys = {k: v for k, v in other_keys.items() if k.endswith(".diff")}
    diff_b_keys = {k: v for k, v in other_keys.items() if k.endswith(".diff_b")}
    diff_m_keys = {k: v for k, v in other_keys.items() if k.endswith(".diff_m")}
    
    remaining = {k: v for k, v in other_keys.items() 
                 if not k.endswith(".diff") and not k.endswith(".diff_b") and not k.endswith(".diff_m")}
    
    print(f"  .diff weights (norms etc): {len(diff_keys)}")
    print(f"  .diff_b weights (biases): {len(diff_b_keys)}")
    print(f"  .diff_m weights (to strip): {len(diff_m_keys)}")
    print(f"  Other keys: {len(remaining)}")
    
    # Process LoRA modules
    new_sd = {}
    errors = []
    skipped = 0
    converted = 0
    
    for module_name, parts in modules.items():
        if "lora_A" not in parts or "lora_B" not in parts:
            # Incomplete pair, keep as-is
            for key_type, (orig_key, tensor) in parts.items():
                new_sd[orig_key] = tensor
            skipped += 1
            continue
        
        lora_A_key, lora_A = parts["lora_A"]
        lora_B_key, lora_B = parts["lora_B"]
        alpha_val = None
        alpha_key = None
        if "alpha" in parts:
            alpha_key, alpha_tensor = parts["alpha"]
            alpha_val = float(alpha_tensor.item()) if torch.is_tensor(alpha_tensor) else float(alpha_tensor)
        
        current_rank = lora_A.shape[0]
        
        if current_rank <= target_rank:
            # Already at or below target, keep as-is
            new_sd[lora_A_key] = lora_A
            new_sd[lora_B_key] = lora_B
            if alpha_key:
                new_sd[alpha_key] = parts["alpha"][1]
            skipped += 1
            continue
        
        # Convert
        result = svd_truncate(lora_A, lora_B, target_rank, alpha_val)
        new_A, new_B, new_alpha, error = result
        
        new_sd[lora_A_key] = new_A
        new_sd[lora_B_key] = new_B
        if alpha_key:
            new_sd[alpha_key] = torch.tensor(new_alpha)
        
        errors.append(error)
        converted += 1
    
    print(f"\nConverted {converted} modules from rank {current_rank} -> {target_rank}")
    print(f"Kept {skipped} modules unchanged")
    
    if errors:
        avg_err = sum(errors) / len(errors)
        max_err = max(errors)
        print(f"Reconstruction error: avg={avg_err:.6f}, max={max_err:.6f}")
    
    # Add .diff weights (keep norms, they're tiny and fast)
    kept_diff = 0
    stripped_diff = 0
    for k, v in diff_keys.items():
        # Strip norm_k_img keys (module doesn't exist in WAN 2.2 I2V)
        if strip_norm_k_img and "norm_k_img" in k:
            stripped_diff += 1
            continue
        new_sd[k] = v
        kept_diff += 1
    
    print(f"\n.diff weights: kept {kept_diff}, stripped {stripped_diff} (norm_k_img)")
    
    # Add .diff_b weights (bias diffs)
    for k, v in diff_b_keys.items():
        if strip_norm_k_img and "norm_k_img" in k:
            continue
        new_sd[k] = v
    
    # Strip diff_m
    if strip_diff_m:
        print(f"Stripped {len(diff_m_keys)} .diff_m keys")
    else:
        new_sd.update(diff_m_keys)
    
    # Add any remaining keys
    new_sd.update(remaining)
    
    print(f"\nOutput: {len(new_sd)} keys (was {len(sd)})")
    
    # Compute size comparison
    BYTES_PER_MB = 1024 * 1024
    input_size = os.path.getsize(input_path) / BYTES_PER_MB

    # Save
    print(f"Saving to {output_path}...")
    save_file(new_sd, output_path)

    output_size = os.path.getsize(output_path) / BYTES_PER_MB
    print(f"Size: {input_size:.1f}MB -> {output_size:.1f}MB ({output_size/input_size*100:.0f}%)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert LoRA rank via SVD truncation")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("output", help="Output safetensors file")
    parser.add_argument("--rank", type=int, default=32, help="Target rank (default: 32)")
    parser.add_argument("--keep-diff-m", action="store_true", help="Keep .diff_m keys")
    parser.add_argument("--keep-norm-k-img", action="store_true", help="Keep norm_k_img keys")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    convert_lora(
        args.input, 
        args.output, 
        target_rank=args.rank,
        strip_diff_m=not args.keep_diff_m,
        strip_norm_k_img=not args.keep_norm_k_img,
    )


if __name__ == "__main__":
    main()
