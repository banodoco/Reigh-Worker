# Project Structure

Queue-based video generation system built on [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).

## Core Files

```
├── worker.py                    # Main worker - polls DB, claims tasks, routes to handlers
├── headless_model_management.py # HeadlessTaskQueue - GPU/model management
├── headless_wgp.py              # WanOrchestrator - WGP wrapper with parameter mapping
├── debug.py                     # CLI tool for investigating tasks/workers
├── create_test_task.py          # Create test tasks for debugging
```

## source/ Package

### Task System
- `task_types.py` - Task type definitions (single source of truth)
- `task_registry.py` - Task routing and dispatch
- `task_conversion.py` - DB params → GenerationTask conversion
- `db_operations.py` - Database operations (SQLite/Supabase)

### Task Handlers (source/task_handlers/)
- `travel_between_images.py` - Multi-image travel video orchestration
- `join_clips.py` / `join_clips_orchestrator.py` - Video clip joining with transitions
- `edit_video_orchestrator.py` - Selective video portion regeneration
- `inpaint_frames.py` - Frame regeneration
- `magic_edit.py` - Replicate API integration
- `create_visualization.py` - Debug visualizations

### Parameters (source/params/)
- `task.py` - TaskConfig combining all param groups
- `lora.py` - LoRA configuration
- `vace.py` - VACE video guide/mask params
- `generation.py` - Core generation params
- `phase.py` - Phase config wrapper

### Utilities
- `common_utils.py` - Shared helpers (download, resize, ffmpeg)
- `video_utils.py` - Video processing (crossfade, extraction, color matching)
- `lora_utils.py` - LoRA download and resolution
- `lora_paths.py` - LoRA directory configuration
- `logging_utils.py` - Structured logging
- `param_aliases.py` - Parameter name normalization
- `platform_utils.py` - Platform-specific utilities
- `wgp_patches.py` - WGP monkeypatches for headless mode
- `specialized_handlers.py` - OpenPose, RIFE handlers
- `comfy_handler.py` / `comfy_utils.py` - ComfyUI integration

### Model Handlers (source/model_handlers/)
- `qwen_handler.py` - Qwen image editing (5 task types)

## External

- `Wan2GP/` - Upstream video generation engine (submodule)
- `supabase/functions/` - Edge Functions for task lifecycle

## Data Flow

```
DB → worker.py → HeadlessTaskQueue → WanOrchestrator → wgp.py → Files
```

1. **worker.py** polls tasks table, claims work
2. **HeadlessTaskQueue** manages model loading and task processing
3. **WanOrchestrator** maps parameters, calls WGP
4. **wgp.py** performs generation, writes to outputs/
5. Results flow back through the chain to update DB

## Database

| Column | Purpose |
|--------|---------|
| `id` | UUID primary key |
| `task_type` | e.g., `vace`, `travel_orchestrator`, `t2v` |
| `dependant_on` | Optional FK forming execution DAG |
| `params` | JSON payload |
| `status` | `Queued` → `In Progress` → `Complete`/`Failed` |
| `output_location` | Final output path/URL |

## Configuration

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Service role key |
| `--db-type` | `sqlite` (default) or `supabase` |
| `--debug` | Keep temp folders, extra logs |
