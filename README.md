# Reigh-Worker

GPU worker for [Reigh](https://github.com/banodoco/Reigh) — processes video generation tasks using [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).

## Quick Start

```bash
# 1. Create venv
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r Wan2GP/requirements.txt
pip install -r requirements.txt

# 3. Run worker
SUPABASE_SERVICE_ROLE_KEY="your-key" python worker.py \
    --supabase-url "https://your-project.supabase.co" \
    --worker "my-worker-001"
```

Get credentials from [reigh.art](https://reigh.art/).

## Standalone Usage

Use the generation engine without Reigh for local testing or custom pipelines:

```bash
# Join two video clips with AI-generated transition
python examples/join_clips_example.py \
    --clip1 scene1.mp4 --clip2 scene2.mp4 \
    --output transition.mp4 --prompt "smooth camera glide"

# Regenerate corrupted frames
python examples/inpaint_frames_example.py \
    --video my_video.mp4 --start-frame 45 --end-frame 61 \
    --output fixed.mp4 --prompt "smooth motion"
```

### Using HeadlessTaskQueue Directly

```python
from headless_model_management import HeadlessTaskQueue, GenerationTask
from pathlib import Path

queue = HeadlessTaskQueue(wan_dir=str(Path(__file__).parent / "Wan2GP"), max_workers=1)
queue.start()

task = GenerationTask(
    id="my_task",
    model="wan_2_2_vace_lightning_baseline_2_2_2",
    prompt="a cat walking through a garden",
    parameters={"video_length": 81, "resolution": "896x512", "seed": 42}
)

queue.submit_task(task)
result = queue.wait_for_completion(task.id, timeout=600)
print(f"Output: {result.get('output_path')}" if result.get("success") else f"Error: {result.get('error')}")

queue.stop()
```

## Debugging

```bash
python -m debug task <task_id>          # Investigate a task
python -m debug tasks --status Failed   # List recent failures
```

## Project Structure

See [STRUCTURE.md](STRUCTURE.md) for detailed project layout.

## Powered By

[Wan2GP](https://github.com/deepbeepmeep/Wan2GP) by [deepbeepmeep](https://github.com/deepbeepmeep) — the `Wan2GP/` directory contains the upstream engine.
