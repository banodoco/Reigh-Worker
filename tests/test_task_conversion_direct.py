from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_db_task_to_generation_task_direct_qwen_style_sets_prompt_and_model(monkeypatch):
    import source.task_handlers.tasks.task_conversion as tc

    class _DummyQwen:
        def __init__(self, **_kwargs):
            pass

        def handle_qwen_image_style(self, _db, generation_params):
            generation_params["prompt"] = "styled prompt"

    monkeypatch.setattr(tc, "QwenHandler", _DummyQwen)
    task = tc.db_task_to_generation_task(
        db_task_params={"prompt": "hello", "model": "base_model", "resolution": "512x512"},
        task_id="t1",
        task_type="qwen_image_style",
        wan2gp_path=str(REPO_ROOT / "Wan2GP"),
    )

    assert task.model == "qwen_image_edit_20B"
    assert task.prompt == "styled prompt"
    assert task.parameters["resolution"] == "512x512"


def test_db_task_to_generation_task_direct_z_image_i2i_download(monkeypatch):
    import source.task_handlers.tasks.task_conversion as tc
    from PIL import Image

    class _DummyQwen:
        def __init__(self, **_kwargs):
            pass

    image = Image.new("RGB", (640, 360), "white")
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")

    fake_requests = SimpleNamespace(
        get=lambda *_args, **_kwargs: SimpleNamespace(content=img_bytes.getvalue(), raise_for_status=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(tc, "QwenHandler", _DummyQwen)

    task = tc.db_task_to_generation_task(
        db_task_params={
            "prompt": "",
            "image_url": "https://example.com/input.png",
            "denoise_strength": 0.55,
            "num_inference_steps": 9,
        },
        task_id="i2i1",
        task_type="z_image_turbo_i2i",
        wan2gp_path=str(REPO_ROOT / "Wan2GP"),
    )

    assert task.model == "z_image_img2img"
    assert task.parameters["denoising_strength"] == 0.55
    assert task.parameters["num_inference_steps"] == 9
    assert "x" in task.parameters["resolution"]
    image_start = task.parameters["image_start"]
    assert Path(image_start).exists()
    Path(image_start).unlink(missing_ok=True)
