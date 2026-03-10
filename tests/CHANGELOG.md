# Changelog

## 2026-02-20 — Unified `ltx2_controlled` Task Type

Adds a single task type that accepts any combination of IC-LoRA, standard LoRA, and image guide controls. Replaces the need to pick between `ltx2`, `ltx2_multiframe`, and `ltx2_ic_multiframe` on the frontend.

### New Files

| File | Description |
|------|-------------|
| `tests/ltx2_controlled_params.py` | 6 reference parameter sets covering all control combinations |

### Modified Files

| File | Change |
|------|--------|
| `source/task_handlers/tasks/task_types.py` | Registered `ltx2_controlled` in `WGP_TASK_TYPES`, `DIRECT_QUEUE_TASK_TYPES`, and `TASK_TYPE_TO_MODEL` (→ `ltx2_19B`) |
| `source/task_handlers/tasks/task_conversion.py` | Added `_download_to_temp()` helper; added `ic_loras`, `image_guides` to param whitelist; added `ltx2_controlled` handler block |

### Handler Block Details (`task_conversion.py`)

The `ltx2_controlled` handler has four stages:

1. **IC-LoRA parsing** — Validates type (pose/depth/canny → P/D/E flags), builds `video_prompt_type`, sets `control_net_weight`/`control_net_weight2`, downloads `guide_video` if URL
2. **Standard LoRA parsing** — Converts `loras` array to `activated_loras` + `loras_multipliers`, merges with existing entries
3. **Image guide parsing** — Validates `image`/`anchors` fields, downloads URLs, expands anchors into `guide_images` format with frame range validation
4. **Metadata** — Stores `_applied_controls_metadata` for debugging/output tracking

### Unified Payload Format

```json
{
  "task_type": "ltx2_controlled",
  "prompt": "...",
  "resolution": "1024x576",
  "video_length": 121,

  "ic_loras": [
    { "type": "pose", "weight": 0.8, "guide_video": "https://..." }
  ],

  "loras": [
    { "path": "https://huggingface.co/.../lora.safetensors", "weight": 1.0 }
  ],

  "image_guides": [
    {
      "image": "https://...",
      "anchors": [
        { "frame": 0, "weight": 1.0 },
        { "frame": -1, "weight": 0.8 }
      ]
    }
  ]
}
```

All three control sections are optional and composable.

### Validation Rules

| Check | Error |
|-------|-------|
| IC-LoRA type not in {pose, depth, canny} | `Invalid IC-LoRA type '{type}'. Must be one of: [pose, depth, canny]` |
| Anchor frame out of range | `Anchor frame {N} out of range [0, {max}]. Use -1 for last frame.` |
| Image guide missing `image` | `Each image_guide must have an 'image' field` |
| Image guide missing `anchors` | `Each image_guide must have at least one anchor` |
| IC-LoRA without `guide_video` | Warning log (not error) |

### Test Parameter Sets (`ltx2_controlled_params.py`)

| Variable | Coverage |
|----------|----------|
| `LTX2_CONTROLLED_IC_ONLY` | IC-LoRA pose + guide video |
| `LTX2_CONTROLLED_LORA_ONLY` | Single standard LoRA via URL |
| `LTX2_CONTROLLED_GUIDES_ONLY` | 3 image guides with single anchors, includes frame -1 |
| `LTX2_CONTROLLED_ALL_COMBINED` | IC-LoRA + standard LoRA + image guides (multi-anchor) |
| `LTX2_CONTROLLED_MULTI_IC` | Two IC-LoRA types (pose + depth) |
| `LTX2_CONTROLLED_SINGLE_ANCHOR_LAST` | Single image guide anchored at last frame only |

### What Was NOT Changed

- `Wan2GP/models/ltx2/ltx2.py` — Pipeline already composes all controls
- `source/models/wgp/orchestrator.py` — Bridge already converts `guide_images` paths to PIL
- `source/task_handlers/tasks/task_registry.py` — `ltx2_controlled` routes through `DIRECT_QUEUE_TASK_TYPES` automatically
- Existing `ltx2`, `ltx2_multiframe`, `ltx2_ic_multiframe` task types — fully backward compatible
