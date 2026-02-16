"""Smoke tests for LTX-2 multi-frame workflow (no GPU/ComfyUI required)."""

import pytest

from source.models.comfy.ltx2_multiframe_workflow import build_ltx2_multiframe_workflow
from tests.ltx2_multiframe_params import LTX2_MULTIFRAME_WORKFLOW_PARAMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_workflow(**overrides):
    """Build a workflow with default test filenames."""
    kwargs = {
        "image_1_filename": "img1.png",
        "image_2_filename": "img2.png",
        "image_3_filename": "img3.png",
        "image_4_filename": "img4.png",
        "prompt": "test prompt",
    }
    kwargs.update(overrides)
    return build_ltx2_multiframe_workflow(**kwargs)


# ---------------------------------------------------------------------------
# TestWorkflowConstruction
# ---------------------------------------------------------------------------

class TestWorkflowConstruction:
    """Verify the API-format workflow structure."""

    def test_build_returns_valid_api_format(self):
        wf = _default_workflow()
        assert isinstance(wf, dict)
        for node_id, node in wf.items():
            assert isinstance(node_id, str), f"Node ID {node_id} must be string"
            assert "class_type" in node, f"Node {node_id} missing class_type"
            assert "inputs" in node, f"Node {node_id} missing inputs"

    def test_has_all_required_node_types(self):
        wf = _default_workflow()
        types = {n["class_type"] for n in wf.values()}
        required = {
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "LTXVAddGuideMulti",
            "LoadImage",
            "ImageResizeKJv2",
            "LTXVPreprocess",
            "LTXVImgToVideoInplace",
            "MultimodalGuider",
            "SamplerCustomAdvanced",
            "EmptyLTXVLatentVideo",
            "LTXVScheduler",
            "RandomNoise",
            "VHS_VideoCombine",
            "LTXVConcatAVLatent",
        }
        missing = required - types
        assert not missing, f"Missing node types: {missing}"

    def test_load_image_count(self):
        wf = _default_workflow()
        load_image_nodes = [n for n in wf.values() if n["class_type"] == "LoadImage"]
        assert len(load_image_nodes) == 4

    def test_image_filenames_injected(self):
        wf = _default_workflow(
            image_1_filename="guide_a.png",
            image_2_filename="guide_b.jpg",
            image_3_filename="guide_c.webp",
            image_4_filename="inplace_d.png",
        )
        filenames = {
            n["inputs"]["image"]
            for n in wf.values()
            if n["class_type"] == "LoadImage"
        }
        assert filenames == {"guide_a.png", "guide_b.jpg", "guide_c.webp", "inplace_d.png"}

    def test_prompt_injected(self):
        wf = _default_workflow(prompt="a cat playing piano")
        # Node "3" is positive CLIPTextEncode
        assert wf["3"]["inputs"]["text"] == "a cat playing piano"

    def test_negative_prompt_injected(self):
        wf = _default_workflow(negative_prompt="ugly, deformed")
        assert wf["4"]["inputs"]["text"] == "ugly, deformed"

    def test_seed_injected(self):
        wf = _default_workflow(seed=42)
        assert wf["11"]["inputs"]["noise_seed"] == 42

    def test_resolution_injected(self):
        wf = _default_workflow(width=1024, height=576)
        assert wf["43"]["inputs"]["width"] == 1024
        assert wf["43"]["inputs"]["height"] == 576

    def test_frame_count_injected(self):
        wf = _default_workflow(num_frames=200)
        assert wf["27"]["inputs"]["value"] == 200

    def test_guide_frame_indices(self):
        wf = _default_workflow(frame_idx_1=10, frame_idx_2=50, frame_idx_3=-1)
        guide = wf["130"]["inputs"]
        assert guide["frame_idx_1"] == 10
        assert guide["frame_idx_2"] == 50
        assert guide["frame_idx_3"] == -1

    def test_no_ui_format_keys(self):
        wf = _default_workflow()
        assert "nodes" not in wf, "Workflow should be API format, not UI format"
        assert "links" not in wf, "Workflow should be API format, not UI format"

    def test_all_node_refs_point_to_existing_nodes(self):
        wf = _default_workflow()
        node_ids = set(wf.keys())
        for node_id, node in wf.items():
            for key, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2:
                    ref_id, slot = value
                    if isinstance(ref_id, str) and isinstance(slot, int):
                        assert ref_id in node_ids, (
                            f"Node {node_id} input '{key}' references "
                            f"non-existent node {ref_id}"
                        )


# ---------------------------------------------------------------------------
# TestLoRAChain
# ---------------------------------------------------------------------------

class TestLoRAChain:
    """Verify LoRA insertion between checkpoint and MultiGPU patcher."""

    def test_no_loras_direct_connection(self):
        wf = _default_workflow(loras=None)
        # Node "44" (MultiGPU patcher) should get model directly from "1" (checkpoint)
        assert wf["44"]["inputs"]["model"] == ["1", 0]

    def test_loras_wired_in_sequence(self):
        loras = [
            {"name": "lora_a.safetensors", "strength": 0.8},
            {"name": "lora_b.safetensors", "strength": 0.5},
        ]
        wf = _default_workflow(loras=loras)

        # First LoRA gets model from checkpoint
        assert wf["140"]["inputs"]["model"] == ["1", 0]
        assert wf["140"]["inputs"]["lora_name"] == "lora_a.safetensors"
        assert wf["140"]["inputs"]["strength_model"] == 0.8

        # Second LoRA gets model from first LoRA
        assert wf["141"]["inputs"]["model"] == ["140", 0]
        assert wf["141"]["inputs"]["lora_name"] == "lora_b.safetensors"
        assert wf["141"]["inputs"]["strength_model"] == 0.5

        # MultiGPU patcher gets model from last LoRA
        assert wf["44"]["inputs"]["model"] == ["141", 0]

    def test_single_lora(self):
        loras = [{"name": "cam.safetensors", "strength": 1.0}]
        wf = _default_workflow(loras=loras)

        assert wf["140"]["inputs"]["model"] == ["1", 0]
        assert wf["44"]["inputs"]["model"] == ["140", 0]


# ---------------------------------------------------------------------------
# TestParameterMapping
# ---------------------------------------------------------------------------

class TestParameterMapping:
    """Verify parameters match the reference workflow."""

    def test_workflow_params_match_reference(self):
        ref = LTX2_MULTIFRAME_WORKFLOW_PARAMS
        wf = _default_workflow(
            seed=ref["seed"],
            width=ref["width"],
            height=ref["height"],
            num_frames=ref["num_frames"],
            fps=ref["fps"],
            steps=ref["steps"],
            max_shift=ref["max_shift"],
            base_shift=ref["base_shift"],
            terminal=ref["terminal"],
            video_cfg=ref["video_cfg"],
            audio_cfg=ref["audio_cfg"],
            frame_idx_1=ref["frame_indices"][0],
            frame_idx_2=ref["frame_indices"][1],
            frame_idx_3=ref["frame_indices"][2],
            img_compression=ref["img_compression"],
            ckpt_name=ref["ckpt_name"],
            gemma_path=ref["gemma_path"],
        )

        # Check checkpoint
        assert wf["1"]["inputs"]["ckpt_name"] == ref["ckpt_name"]
        # Check resolution
        assert wf["43"]["inputs"]["width"] == ref["width"]
        assert wf["43"]["inputs"]["height"] == ref["height"]
        # Check frames
        assert wf["27"]["inputs"]["value"] == ref["num_frames"]
        # Check seed
        assert wf["11"]["inputs"]["noise_seed"] == ref["seed"]
        # Check sampler
        assert wf["8"]["inputs"]["sampler_name"] == ref["sampler"]

    def test_scheduler_params(self):
        wf = _default_workflow(max_shift=2.05, base_shift=0.95, terminal=0.1, steps=20)
        sched = wf["9"]["inputs"]
        assert sched["max_shift"] == 2.05
        assert sched["base_shift"] == 0.95
        assert sched["terminal"] == 0.1
        assert sched["steps"] == 20

    def test_guider_params(self):
        wf = _default_workflow(video_cfg=3.0, audio_cfg=7.0)
        assert wf["18"]["inputs"]["cfg"] == 3.0
        assert wf["19"]["inputs"]["cfg"] == 7.0


# ---------------------------------------------------------------------------
# TestTaskRegistration
# ---------------------------------------------------------------------------

class TestTaskRegistration:
    """Verify task type is properly registered."""

    def test_task_type_in_model_mapping(self):
        from source.task_handlers.tasks.task_types import TASK_TYPE_TO_MODEL
        assert "ltx2_multiframe" in TASK_TYPE_TO_MODEL
        assert TASK_TYPE_TO_MODEL["ltx2_multiframe"] == "ltx2_19B"

    def test_task_type_not_in_direct_queue(self):
        from source.task_handlers.tasks.task_types import DIRECT_QUEUE_TASK_TYPES
        assert "ltx2_multiframe" not in DIRECT_QUEUE_TASK_TYPES

    def test_handler_importable(self):
        from source.models.comfy.ltx2_multiframe_handler import handle_ltx2_multiframe_task
        assert callable(handle_ltx2_multiframe_task)


# ---------------------------------------------------------------------------
# TestWiringIntegrity
# ---------------------------------------------------------------------------

class TestWiringIntegrity:
    """Verify critical node connections."""

    def test_multimodal_guider_uses_guide_multi_output(self):
        wf = _default_workflow()
        guider = wf["17"]["inputs"]
        assert guider["positive"] == ["130", 0], "Guider positive should come from LTXVAddGuideMulti"
        assert guider["negative"] == ["130", 1], "Guider negative should come from LTXVAddGuideMulti"

    def test_concat_av_uses_guide_multi_latent(self):
        wf = _default_workflow()
        concat = wf["28"]["inputs"]
        assert concat["video_latent"] == ["130", 2], "ConcatAV video_latent should come from LTXVAddGuideMulti"

    def test_guide_multi_uses_conditioning_from_ltxv(self):
        wf = _default_workflow()
        guide = wf["130"]["inputs"]
        assert guide["positive"] == ["22", 0]
        assert guide["negative"] == ["22", 1]

    def test_img_to_video_inplace_chain(self):
        wf = _default_workflow()
        # LoadImage "103" → LTXVPreprocess "120" → LTXVImgToVideoInplace "121"
        assert wf["120"]["inputs"]["image"] == ["103", 0]
        assert wf["121"]["inputs"]["image"] == ["120", 0]
        assert wf["121"]["inputs"]["latent"] == ["43", 0]
        # LTXVImgToVideoInplace → LTXVAddGuideMulti latent
        assert wf["130"]["inputs"]["latent"] == ["121", 0]

    def test_resize_nodes_use_correct_sources(self):
        wf = _default_workflow()
        assert wf["110"]["inputs"]["image"] == ["100", 0]
        assert wf["111"]["inputs"]["image"] == ["101", 0]
        assert wf["112"]["inputs"]["image"] == ["102", 0]

    def test_guide_multi_uses_resized_images(self):
        wf = _default_workflow()
        guide = wf["130"]["inputs"]
        assert guide["image_1"] == ["110", 0]
        assert guide["image_2"] == ["111", 0]
        assert guide["image_3"] == ["112", 0]
