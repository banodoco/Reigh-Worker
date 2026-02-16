# Test Files Guide

19 Python files: 12 headless tests, 5 GPU tests, 2 parameter modules.

## Quick Run

```bash
# Headless (no GPU required)
python -m pytest tests/test_service_health.py tests/test_all_services_headless.py tests/test_task_conversion_headless.py tests/test_ltx2_headless.py tests/test_ltx2_pose_smoke.py tests/test_ltx2_multiframe_smoke.py tests/test_ltx_headless.py tests/test_wan_headless.py tests/test_lora_flow.py tests/test_lora_formats_baseline.py tests/test_travel_between_images.py tests/test_multi_structure_video.py -v

# GPU (requires CUDA + model weights)
python -m pytest tests/test_ic_lora_gpu.py tests/test_ltx2_multiframe_gpu.py tests/test_travel_real_gpu.py tests/test_travel_ltx2_lora_gpu.py -v -s
```

**GPU test prerequisites:** Place `vid1.mp4` and `img1.png` in the `Wan2GP/` directory. Model weights download automatically on first run via `preload_URLs`.

## File List

### Headless Tests (no GPU required)

| File | What It Tests |
|------|--------------|
| `test_service_health.py` | Task type set sizes, handler coverage, JSON config validity |
| `test_all_services_headless.py` | 24 service type registration, model detection, helper functions |
| `test_task_conversion_headless.py` | DB task → GenerationTask pipeline, phase config parsing |
| `test_ltx2_headless.py` | LTX-2 model detection, v10.x param passthrough, image/audio bridging |
| `test_ltx2_pose_smoke.py` | IC-LoRA pose workflow, MediaPipe pose extraction, control signal |
| `test_ltx2_multiframe_smoke.py` | Multi-frame guide image param flow (task_types → ltx2.generate) |
| `test_ltx_headless.py` | LTXv detection, cross-model switching (LTX2 ↔ LTXv ↔ T2V) |
| `test_wan_headless.py` | T2V/I2V/VACE/Flux/Hunyuan smoke, model switching rotation |
| `test_lora_flow.py` | LoRA URL resolution: parse_phase_config → download → to_wgp_format |
| `test_lora_formats_baseline.py` | LoRA format resolution (legacy WGP, CSV, dict formats) |
| `test_travel_between_images.py` | Travel-between-images segment assignment, frame quantization, stitch |
| `test_multi_structure_video.py` | Multi-structure video compositing, neutral frame, segment stitching |

### GPU Tests (requires CUDA)

| File | What It Tests |
|------|--------------|
| `test_ic_lora_gpu.py` | IC-LoRA (depth/pose/canny) end-to-end, union control LoRA |
| `test_ltx2_multiframe_gpu.py` | Multi-frame guide image injection, different frame positions |
| `test_travel_real_gpu.py` | LTX-2 travel-between-images real GPU inference |
| `test_travel_ltx2_lora_gpu.py` | LTX-2 travel + Deforum Evolution LoRA, 5 test image transitions |
| `run_multi_frame_injection.py` | Standalone: 105-frame video generation with 4 guide images |

### Parameter Files (not tests, reference only)

| File | Contents |
|------|----------|
| `ltx2_multiframe_params.py` | ComfyUI multi-frame workflow reference parameters |
| `ltx2pose_params.py` | ComfyUI IC-LoRA pose workflow reference parameters |

## Multi-Frame Guide Image Injection

`run_multi_frame_injection.py` — LTX-2 multi-frame guide image end-to-end test.

Injects multiple reference images at arbitrary frame positions during video generation. Each guide image is latent-encoded and conditioned at the specified frame index.

```bash
python tests/run_multi_frame_injection.py
```

**What was fixed:**

1. **`generate_video()` signature** (`wgp.py`): Added `guide_images` parameter so `_filter_wgp_params()` (which uses `inspect` to read the signature) no longer drops it.
2. **`wan_model.generate()` call** (`wgp.py`): Added `guide_images=window_guide_images` to forward the data to the LTX-2 model.
3. **Sliding window filtering** (`wgp.py`): Guide images are filtered per window — only images within the current window's frame range are passed, with frame indices adjusted to be window-relative. Handles `frame_idx=-1` (last frame) by including it only in the final window.

**Data flow:**

```
User kwargs: [{"image": path, "frame_idx": 0, "strength": 1.0}, ...]
  → orchestrator.py LTX2_BRIDGE: dict → PIL tuples
  → wgp_params.py build_normal_params: wgp_params["guide_images"]
  → _filter_wgp_params: passes through (now in signature)
  → generate_video(guide_images=...): filters per sliding window
  → wan_model.generate(guide_images=...): ltx2.py latent injection
```

**Output:** `/tmp/reigh_output/*.mp4`

## Bug Fixes Found During Testing

| Bug | Fix | File |
|-----|-----|------|
| `apply_changes` import crash | Replaced with `get_default_settings` + `set_model_settings` API | `orchestrator.py` |
| 12 missing v10.x params (`alt_prompt`, `duration_seconds`, `audio_scale`, `self_refiner_*`, etc.) | Added to both passthrough and normal mode builders | `wgp_params.py` |
| Union control LoRA 404 (wrong filename) | Corrected to `ltx-2-19b-ic-lora-union-control-ref0.5.safetensors` | `ltx2_19B.json` |
| `NoneType / int` crash on control_net_weight2 | Moved default assignment outside `if is_vace:` block | `orchestrator.py` |

## IC-LoRA Pipeline Insight

IC-LoRA weights (pose, depth, canny, union) load as standard LoRA adapters and work. However the dedicated `ICLoraPipeline` — with reference downscale and video conditioning guide injection (ComfyUI equivalent: `LTXICLoRALoaderModelOnly` + `LTXAddVideoICLoRAGuide`) — is **not active**. The config defaults to `two_stage` pipeline.

To enable: add `"ltx2_pipeline": "ic_lora"` in `Wan2GP/defaults/ltx2_19B.json` model definition.

## Key Source Files

| File | Role |
|------|------|
| `source/models/wgp/orchestrator.py` | WanOrchestrator — model init, generate() dispatch |
| `source/models/wgp/generators/wgp_params.py` | Parameter dict builders (passthrough + normal mode) |
| `source/task_types.py` | Task type sets, model mappings |
| `source/task_conversion.py` | DB task → GenerationTask, phase config parsing |
| `Wan2GP/defaults/ltx2_19B.json` | LTX-2 19B model config (URLs, LoRAs, preloads) |
| `Wan2GP/models/ltx2/ltx2.py` | LTX-2 model class, pipeline selection logic |
