"""
LTX-2 Multi-Frame Video Task Handler

Handles multi-frame guided video generation tasks via ComfyUI.
Downloads guide images, uploads to ComfyUI, builds the workflow, and executes.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Tuple, Optional
import httpx

from source.core.log import headless_logger, model_logger
from source.models.comfy.comfy_utils import ComfyUIClient
from source.models.comfy.comfy_handler import _ensure_comfy_running
from source.models.comfy.ltx2_multiframe_workflow import build_ltx2_multiframe_workflow


def handle_ltx2_multiframe_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str,
) -> Tuple[bool, Optional[str]]:
    """Handle LTX-2 multi-frame video generation tasks.

    Args:
        task_params_from_db: Task parameters from database. Expected keys:
            - prompt (str): Required positive text prompt.
            - negative_prompt (str): Optional negative prompt.
            - image_urls (list[str]): Required, 4 image URLs or local paths.
            - seed (int): Optional random seed.
            - width (int): Optional video width.
            - height (int): Optional video height.
            - num_frames (int): Optional frame count.
            - steps (int): Optional diffusion steps.
            - frame_indices (list[int]): Optional [idx1, idx2, idx3] for images 1-3.
            - guide_strengths (list[float]): Optional [s1, s2, s3] for images 1-3.
            - loras (list[dict]): Optional LoRAs with "name" and "strength".
            - video_cfg (float): Optional video CFG scale.
            - audio_cfg (float): Optional audio CFG scale.
            - img_compression (int): Optional LTXVPreprocess compression.
            - img4_strength (float): Optional inplace image strength.
            - fps (float): Optional frames per second.
            - max_shift (float): Optional scheduler max_shift.
            - base_shift (float): Optional scheduler base_shift.
            - terminal (float): Optional scheduler terminal.
            - ckpt_name (str): Optional checkpoint filename.
            - gemma_path (str): Optional Gemma model path.
        main_output_dir_base: Base output directory.
        task_id: Task identifier.

    Returns:
        (success, output_path_or_error) tuple.
    """
    model_logger.debug(f"Processing LTX-2 multiframe task {task_id}")
    headless_logger.info("Processing LTX-2 multiframe task", task_id=task_id)

    try:
        params = task_params_from_db
        if isinstance(params, str):
            params = json.loads(params)

        # Validate required params
        prompt = params.get("prompt")
        if not prompt:
            return False, "Missing required parameter: prompt"

        image_urls = params.get("image_urls")
        if not image_urls or len(image_urls) != 4:
            return False, "Missing or invalid image_urls: exactly 4 image URLs/paths required"

        # Run async processing
        async def _process():
            if not await _ensure_comfy_running():
                raise RuntimeError(
                    "ComfyUI is not available on this worker. "
                    "Ensure ComfyUI is installed at COMFY_PATH."
                )

            comfy_client = ComfyUIClient()

            async with httpx.AsyncClient(timeout=300.0) as client:
                # Download and upload 4 images
                uploaded_filenames = []
                for idx, image_source in enumerate(image_urls):
                    image_bytes = await _get_image_bytes(client, image_source)
                    ext = _get_extension(image_source)
                    upload_name = f"{task_id}_{idx}.{ext}"

                    uploaded = await comfy_client.upload_image(
                        client, image_bytes, upload_name
                    )
                    uploaded_filenames.append(uploaded)
                    model_logger.debug(f"Uploaded image {idx}: {uploaded}")

                headless_logger.info(
                    f"Uploaded {len(uploaded_filenames)} images",
                    task_id=task_id,
                )

                # Build workflow kwargs
                frame_indices = params.get("frame_indices", [40, 80, -1])
                guide_strengths = params.get("guide_strengths", [1.0, 1.0, 1.0])

                wf_kwargs = {
                    "image_1_filename": uploaded_filenames[0],
                    "image_2_filename": uploaded_filenames[1],
                    "image_3_filename": uploaded_filenames[2],
                    "image_4_filename": uploaded_filenames[3],
                    "prompt": prompt,
                    "negative_prompt": params.get(
                        "negative_prompt", "blurry, low quality, watermark"
                    ),
                    "frame_idx_1": frame_indices[0] if len(frame_indices) > 0 else 40,
                    "frame_idx_2": frame_indices[1] if len(frame_indices) > 1 else 80,
                    "frame_idx_3": frame_indices[2] if len(frame_indices) > 2 else -1,
                    "strength_1": guide_strengths[0] if len(guide_strengths) > 0 else 1.0,
                    "strength_2": guide_strengths[1] if len(guide_strengths) > 1 else 1.0,
                    "strength_3": guide_strengths[2] if len(guide_strengths) > 2 else 1.0,
                }

                # Optional params with defaults matching workflow builder
                optional_keys = [
                    "seed", "width", "height", "num_frames", "fps", "steps",
                    "max_shift", "base_shift", "terminal",
                    "video_cfg", "audio_cfg", "loras",
                    "ckpt_name", "gemma_path",
                    "img_compression", "img4_strength",
                ]
                for key in optional_keys:
                    if key in params:
                        wf_kwargs[key] = params[key]

                # Build and submit workflow
                workflow = build_ltx2_multiframe_workflow(**wf_kwargs)
                model_logger.debug("Built multiframe workflow")

                prompt_id = await comfy_client.queue_workflow(client, workflow)
                model_logger.debug(f"Workflow queued: {prompt_id}")
                headless_logger.info(f"Workflow queued: {prompt_id}", task_id=task_id)

                # Wait for completion
                history = await comfy_client.wait_for_completion(client, prompt_id)
                model_logger.debug("Workflow completed")
                headless_logger.info("Workflow completed", task_id=task_id)

                # Download outputs
                outputs = await comfy_client.download_output(client, history)
                if not outputs:
                    raise RuntimeError("No outputs generated by workflow")

                model_logger.debug(f"Downloaded {len(outputs)} output(s)")
                return outputs[0]

        output = asyncio.run(_process())

        # Save output
        output_dir = main_output_dir_base / "ltx2_multiframe"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{output['filename']}"

        with open(output_path, 'wb') as f:
            f.write(output['content'])

        model_logger.debug(f"Saved output to: {output_path}")
        headless_logger.info(f"Saved output: {output_path}", task_id=task_id)

        return True, str(output_path)

    except (httpx.HTTPError, OSError, json.JSONDecodeError, ValueError,
            RuntimeError, TimeoutError) as e:
        error_msg = f"LTX-2 multiframe task failed: {str(e)}"
        model_logger.debug(error_msg)
        headless_logger.error(error_msg, task_id=task_id, exc_info=True)
        return False, error_msg


async def _get_image_bytes(client: httpx.AsyncClient, source: str) -> bytes:
    """Get image bytes from URL or local path."""
    if source.startswith(("http://", "https://")):
        response = await client.get(source)
        response.raise_for_status()
        return response.content
    else:
        return Path(source).read_bytes()


def _get_extension(source: str) -> str:
    """Extract file extension from URL or path, defaulting to png."""
    # Strip query params for URLs
    clean = source.split("?")[0].split("#")[0]
    if "." in clean.split("/")[-1]:
        ext = clean.rsplit(".", 1)[-1].lower()
        if ext in ("png", "jpg", "jpeg", "webp", "bmp"):
            return ext
    return "png"
