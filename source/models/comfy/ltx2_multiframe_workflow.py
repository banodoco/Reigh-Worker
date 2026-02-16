"""
LTX-2 Multi-Frame Video Workflow Builder

Constructs a ComfyUI API-format workflow for multi-frame guided video generation.
The base pipeline (nodes 1-44) handles checkpoint loading, text encoding, scheduling,
sampling, and output. On top of these, guide image nodes are added:
  - 4x LoadImage nodes (IDs "100"-"103")
  - 3x ImageResizeKJv2 nodes (IDs "110"-"112") for guide images 1-3
  - 1x LTXVPreprocess (ID "120") + 1x LTXVImgToVideoInplace (ID "121") for image 4
  - 1x LTXVAddGuideMulti (ID "130") combining all guides
  - Optional LoraLoaderModelOnly nodes (IDs "140"+)
"""

import copy
from typing import Dict, List, Optional


def _build_base_nodes(
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    steps: int,
    max_shift: float,
    base_shift: float,
    terminal: float,
    video_cfg: float,
    audio_cfg: float,
    ckpt_name: str,
    gemma_path: str,
) -> Dict:
    """Build the base pipeline nodes (1-44) from the embedded API format."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "inputs": {
                "ltxv_path": ckpt_name,
                "gemma_path": gemma_path,
                "max_length": 1024,
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "9": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "stretch": True,
                "latent": ["28", 0],
                "max_shift": max_shift,
                "terminal": terminal,
                "steps": steps,
                "base_shift": base_shift,
            },
        },
        "11": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"vae": ["1", 2], "samples": ["29", 0]},
        },
        "13": {
            "class_type": "LTXVAudioVAELoader",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "14": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"audio_vae": ["13", 0], "samples": ["29", 1]},
        },
        "15": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "save_output": True,
                "filename_prefix": "AnimateDiff",
                "images": ["12", 0],
                "loop_count": 0,
                "pix_fmt": "yuv420p",
                "save_metadata": True,
                "crf": 19,
                "trim_to_audio": False,
                "format": "video/h264-mp4",
                "audio": ["14", 0],
                "frame_rate": ["23", 0],
                "pingpong": False,
            },
        },
        "17": {
            "class_type": "MultimodalGuider",
            "inputs": {
                "skip_blocks": "29",
                # Rewired below to LTXVAddGuideMulti ("130") outputs
                "negative": ["130", 1],
                "model": ["28", 1],
                "positive": ["130", 0],
                "parameters": ["18", 0],
            },
        },
        "18": {
            "class_type": "GuiderParameters",
            "inputs": {
                "modality": "VIDEO",
                "cfg": video_cfg,
                "rescale": 0,
                "stg": 0,
                "parameters": ["19", 0],
                "modality_scale": 3,
            },
        },
        "19": {
            "class_type": "GuiderParameters",
            "inputs": {
                "modality": "AUDIO",
                "cfg": audio_cfg,
                "rescale": 0,
                "stg": 0,
                "modality_scale": 3,
            },
        },
        "21": {
            "class_type": "PreviewAudio",
            "inputs": {"audio": ["14", 0], "audioUI": ""},
        },
        "22": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "negative": ["4", 0],
                "positive": ["3", 0],
                "frame_rate": ["23", 0],
            },
        },
        "23": {
            "class_type": "FloatConstant",
            "inputs": {"value": fps},
        },
        "26": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "batch_size": 1,
                "frame_rate": ["42", 0],
                "frames_number": ["27", 0],
            },
        },
        "27": {
            "class_type": "INTConstant",
            "inputs": {"value": num_frames},
        },
        "28": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {
                "audio_latent": ["26", 0],
                # Rewired: video_latent from LTXVAddGuideMulti latent output
                "video_latent": ["130", 2],
                "model": ["44", 0],
            },
        },
        "29": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": ["41", 0], "model": ["28", 1]},
        },
        "41": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "guider": ["17", 0],
                "latent_image": ["28", 0],
                "noise": ["11", 0],
                "sigmas": ["9", 0],
                "sampler": ["8", 0],
            },
        },
        "42": {
            "class_type": "CM_FloatToInt",
            "inputs": {"a": ["23", 0]},
        },
        "43": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "batch_size": 1,
                "width": width,
                "length": ["27", 0],
                "height": height,
            },
        },
        "44": {
            "class_type": "LTXVSequenceParallelMultiGPUPatcher",
            "inputs": {
                "disable_backup": False,
                "torch_compile": True,
                "model": ["1", 0],
            },
        },
    }


def _build_image_load_nodes(
    image_1_filename: str,
    image_2_filename: str,
    image_3_filename: str,
    image_4_filename: str,
) -> Dict:
    """Build LoadImage nodes for the 4 guide images."""
    nodes = {}
    filenames = [
        ("100", image_1_filename),
        ("101", image_2_filename),
        ("102", image_3_filename),
        ("103", image_4_filename),
    ]
    for node_id, filename in filenames:
        nodes[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
        }
    return nodes


def _build_image_resize_nodes(width: int, height: int) -> Dict:
    """Build ImageResizeKJv2 nodes for guide images 1-3."""
    nodes = {}
    # Images 1-3 are resized to match video dimensions
    for i, (node_id, source_id) in enumerate([
        ("110", "100"),  # image_1
        ("111", "101"),  # image_2
        ("112", "102"),  # image_3
    ]):
        nodes[node_id] = {
            "class_type": "ImageResizeKJv2",
            "inputs": {
                "image": [source_id, 0],
                "width": width,
                "height": height,
                "interpolation": "lanczos",
                "method": "crop",
                "divisor": 2,
            },
        }
    return nodes


def _build_image4_processing_nodes(
    img_compression: int,
    img4_strength: float,
) -> Dict:
    """Build LTXVPreprocess and LTXVImgToVideoInplace for image 4."""
    return {
        "120": {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "image": ["103", 0],
                "img_compression": img_compression,
            },
        },
        "121": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["1", 2],
                "image": ["120", 0],
                "latent": ["43", 0],
                "strength": img4_strength,
                "bypass": False,
            },
        },
    }


def _build_guide_multi_node(
    frame_idx_1: int,
    frame_idx_2: int,
    frame_idx_3: int,
    strength_1: float,
    strength_2: float,
    strength_3: float,
) -> Dict:
    """Build the LTXVAddGuideMulti node."""
    return {
        "130": {
            "class_type": "LTXVAddGuideMulti",
            "inputs": {
                "positive": ["22", 0],
                "negative": ["22", 1],
                "vae": ["1", 2],
                "latent": ["121", 0],
                "image_1": ["110", 0],
                "image_2": ["111", 0],
                "image_3": ["112", 0],
                "frame_idx_1": frame_idx_1,
                "strength_1": strength_1,
                "frame_idx_2": frame_idx_2,
                "strength_2": strength_2,
                "frame_idx_3": frame_idx_3,
                "strength_3": strength_3,
            },
        }
    }


def _insert_lora_chain(
    workflow: Dict,
    loras: List[Dict],
) -> None:
    """Insert LoRA chain between checkpoint and MultiGPU patcher.

    Modifies workflow in-place. Creates nodes "140", "141", etc.
    """
    if not loras:
        return

    prev_model_ref = ["1", 0]  # Start from checkpoint model output

    for i, lora in enumerate(loras):
        node_id = str(140 + i)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": prev_model_ref,
                "lora_name": lora["name"],
                "strength_model": lora.get("strength", 1.0),
            },
        }
        prev_model_ref = [node_id, 0]

    # Wire last LoRA output to MultiGPU patcher
    workflow["44"]["inputs"]["model"] = prev_model_ref


def build_ltx2_multiframe_workflow(
    image_1_filename: str,
    image_2_filename: str,
    image_3_filename: str,
    image_4_filename: str,
    prompt: str,
    negative_prompt: str = "blurry, low quality, watermark",
    seed: int = 10,
    width: int = 768,
    height: int = 512,
    num_frames: int = 105,
    fps: float = 25.0,
    steps: int = 20,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    terminal: float = 0.1,
    frame_idx_1: int = 40,
    frame_idx_2: int = 80,
    frame_idx_3: int = -1,
    strength_1: float = 1.0,
    strength_2: float = 1.0,
    strength_3: float = 1.0,
    video_cfg: float = 3.0,
    audio_cfg: float = 7.0,
    loras: Optional[List[Dict]] = None,
    ckpt_name: str = "ltx-av-step-1751000_vocoder_24K.safetensors",
    gemma_path: str = "gemma-3-12b-it-qat-q4_0-unquantized_readout_proj/model/model.safetensors",
    img_compression: int = 35,
    img4_strength: float = 1.0,
) -> Dict:
    """Build a complete ComfyUI API-format workflow for LTX-2 multi-frame video.

    Image 4 is the "inplace" image encoded directly into the latent at frame 0.
    Images 1-3 are guide images placed at frame_idx_1/2/3 respectively.

    Args:
        image_1_filename: Uploaded filename for guide image at frame_idx_1.
        image_2_filename: Uploaded filename for guide image at frame_idx_2.
        image_3_filename: Uploaded filename for guide image at frame_idx_3.
        image_4_filename: Uploaded filename for inplace image at frame 0.
        prompt: Positive text prompt.
        negative_prompt: Negative text prompt.
        seed: Random seed.
        width: Video width in pixels.
        height: Video height in pixels.
        num_frames: Total number of frames.
        fps: Frames per second.
        steps: Number of diffusion steps.
        max_shift: LTXVScheduler max_shift parameter.
        base_shift: LTXVScheduler base_shift parameter.
        terminal: LTXVScheduler terminal parameter.
        frame_idx_1: Frame index for guide image 1.
        frame_idx_2: Frame index for guide image 2.
        frame_idx_3: Frame index for guide image 3 (-1 = last frame).
        strength_1: Guide strength for image 1.
        strength_2: Guide strength for image 2.
        strength_3: Guide strength for image 3.
        video_cfg: Video CFG scale.
        audio_cfg: Audio CFG scale.
        loras: Optional list of LoRA dicts with "name" and "strength" keys.
        ckpt_name: Checkpoint filename.
        gemma_path: Gemma text encoder path.
        img_compression: LTXVPreprocess compression for image 4.
        img4_strength: LTXVImgToVideoInplace strength for image 4.

    Returns:
        Complete API-format workflow dict with string node IDs.
    """
    workflow = _build_base_nodes(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        max_shift=max_shift,
        base_shift=base_shift,
        terminal=terminal,
        video_cfg=video_cfg,
        audio_cfg=audio_cfg,
        ckpt_name=ckpt_name,
        gemma_path=gemma_path,
    )

    # Add image loading nodes
    workflow.update(_build_image_load_nodes(
        image_1_filename, image_2_filename,
        image_3_filename, image_4_filename,
    ))

    # Add image resize nodes for images 1-3
    workflow.update(_build_image_resize_nodes(width, height))

    # Add image 4 preprocessing (LTXVPreprocess + ImgToVideoInplace)
    workflow.update(_build_image4_processing_nodes(img_compression, img4_strength))

    # Add LTXVAddGuideMulti
    workflow.update(_build_guide_multi_node(
        frame_idx_1=frame_idx_1,
        frame_idx_2=frame_idx_2,
        frame_idx_3=frame_idx_3,
        strength_1=strength_1,
        strength_2=strength_2,
        strength_3=strength_3,
    ))

    # Insert LoRA chain if provided
    _insert_lora_chain(workflow, loras or [])

    return workflow
