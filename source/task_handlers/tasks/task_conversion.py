import os

from source.core.log import headless_logger
from source.core.params.phase_config_parser import parse_phase_config


# Target megapixel count for auto-scaling img2img input images
IMG2IMG_TARGET_MEGAPIXELS = 1024 * 1024

# Default resolution string for image generation tasks
DEFAULT_IMAGE_RESOLUTION = "1024x1024"
from source.models.model_handlers.qwen_handler import QwenHandler
from source.utils import extract_orchestrator_parameters
from headless_model_management import GenerationTask

def _download_to_temp(url: str, suffix: str = ".png", timeout: int = 30) -> str:
    """Download a URL to a temporary file and return the local path."""
    import tempfile
    import requests
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name


def db_task_to_generation_task(db_task_params: dict, task_id: str, task_type: str, wan2gp_path: str, debug_mode: bool = False) -> GenerationTask:
    """
    Convert a database task row to a GenerationTask object for the queue system.
    """
    headless_logger.debug(f"Converting DB task to GenerationTask", task_id=task_id)

    prompt = db_task_params.get("prompt", "")

    # For img2img tasks, empty prompt is acceptable (will use minimal changes)
    # Provide a minimal default prompt to avoid errors
    img2img_task_types = {"z_image_turbo_i2i", "qwen_image_edit", "qwen_image_style", "image_inpaint"}
    if not prompt:
        if task_type in img2img_task_types:
            prompt = " "  # Minimal prompt for img2img
            headless_logger.debug(f"Task {task_id}: Using minimal prompt for img2img task", task_id=task_id)
        else:
            raise ValueError(f"Task {task_id}: prompt is required")
    
    model = db_task_params.get("model")
    if not model:
        from source.task_handlers.tasks.task_types import get_default_model
        model = get_default_model(task_type)
    
    generation_params = {}
    
    param_whitelist = {
        "negative_prompt", "resolution", "video_length", "num_inference_steps",
        "guidance_scale", "seed", "embedded_guidance_scale", "flow_shift",
        "audio_guidance_scale", "repeat_generation", "multi_images_gen_type",
        "guidance2_scale", "guidance3_scale", "guidance_phases", "switch_threshold", "switch_threshold2", "model_switch_phase",
        "video_guide", "video_mask",
        "video_prompt_type", "control_net_weight", "control_net_weight2",
        "keep_frames_video_guide", "video_guide_outpainting", "mask_expand",
        "image_prompt_type", "image_start", "image_end", "image_refs",
        "frames_positions", "image_guide", "image_mask",
        "model_mode", "video_source", "keep_frames_video_source",
        "audio_guide", "audio_guide2", "audio_source", "audio_prompt_type", "speakers_locations",
        "activated_loras", "loras_multipliers", "additional_loras", "loras",
        "tea_cache_setting", "tea_cache_start_step_perc", "RIFLEx_setting",
        "slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc",
        "cfg_star_switch", "cfg_zero_step", "prompt_enhancer",
        # Hires fix parameters
        "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_method", "lightning_lora_strength",
        "sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise",
        "sliding_window_discard_last_frames", "latent_noise_mask_strength",
        "vid2vid_init_video", "vid2vid_init_strength",
        "remove_background_images_ref", "temporal_upsampling", "spatial_upsampling",
        "film_grain_intensity", "film_grain_saturation",
        "image_refs_relative_size",
        "output_dir", "custom_output_dir",
        "override_profile",
        "image", "image_url", "mask_url",
        "style_reference_image", "subject_reference_image",
        "style_reference_strength", "subject_strength",
        "subject_description", "in_this_scene",
        "output_format", "enable_base64_output", "enable_sync_mode",
        # Uni3C motion guidance parameters
        "use_uni3c", "uni3c_guide_video", "uni3c_strength",
        "uni3c_start_percent", "uni3c_end_percent",
        "uni3c_keep_on_gpu", "uni3c_frame_policy",
        "uni3c_zero_empty_frames", "uni3c_blackout_last_frame",
        # Image-to-image parameters
        "denoising_strength",
        # Multi-frame guide images
        "guide_images",
        # Unified controlled params
        "ic_loras", "image_guides",
    }
    
    for param in param_whitelist:
        if param in db_task_params:
            generation_params[param] = db_task_params[param]
    
    # Layer 1 Uni3C logging - detect whitelist failures early
    if "use_uni3c" in generation_params:
        headless_logger.info(
            f"[UNI3C] Task {task_id}: use_uni3c={generation_params.get('use_uni3c')}, "
            f"guide_video={generation_params.get('uni3c_guide_video', 'NOT_SET')}, "
            f"strength={generation_params.get('uni3c_strength', 'NOT_SET')}"
        )
    elif db_task_params.get("use_uni3c"):
        # CRITICAL: Detect when whitelist is missing the param
        headless_logger.warning(
            f"[UNI3C] Task {task_id}: ⚠️ use_uni3c was in db_task_params but NOT in generation_params! "
            f"Check param_whitelist in task_conversion.py"
        )
    
    # Extract orchestrator parameters
    extracted_params = extract_orchestrator_parameters(db_task_params, task_id)

    if "phase_config" in extracted_params:
        db_task_params["phase_config"] = extracted_params["phase_config"]

    # Copy additional_loras from extracted params if not already in generation_params
    if "additional_loras" in extracted_params and "additional_loras" not in generation_params:
        generation_params["additional_loras"] = extracted_params["additional_loras"]
    
    if "steps" in db_task_params and "num_inference_steps" not in generation_params:
        generation_params["num_inference_steps"] = db_task_params["steps"]
    
    # Note: LoRA resolution is handled centrally in HeadlessTaskQueue._convert_to_wgp_task()
    # via TaskConfig/LoRAConfig which detects URLs and downloads them automatically

    # Qwen Task Handlers
    qwen_handler = QwenHandler(
        wan_root=wan2gp_path,
        task_id=task_id)

    if task_type == "qwen_image_edit":
        qwen_handler.handle_qwen_image_edit(db_task_params, generation_params)
    elif task_type == "qwen_image_hires":
        qwen_handler.handle_qwen_image_hires(db_task_params, generation_params)
    elif task_type == "image_inpaint":
        qwen_handler.handle_image_inpaint(db_task_params, generation_params)
    elif task_type == "annotated_image_edit":
        qwen_handler.handle_annotated_image_edit(db_task_params, generation_params)
    elif task_type == "qwen_image_style":
        qwen_handler.handle_qwen_image_style(db_task_params, generation_params)
        prompt = generation_params.get("prompt", prompt)
        model = "qwen_image_edit_20B"
    elif task_type == "qwen_image":
        qwen_handler.handle_qwen_image(db_task_params, generation_params)
        model = "qwen_image_edit_20B"
    elif task_type == "qwen_image_2512":
        qwen_handler.handle_qwen_image_2512(db_task_params, generation_params)
        model = "qwen_image_2512_20B"
    elif task_type == "z_image_turbo":
        # Z-Image turbo - fast text-to-image generation
        generation_params.setdefault("video_prompt_type", "")  # No input image
        generation_params.setdefault("video_length", 1)  # Single image output
        generation_params.setdefault("guidance_scale", 0)  # Z-Image uses guidance_scale=0
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 8)))

        # Resolution handling
        if "resolution" in db_task_params:
            generation_params["resolution"] = db_task_params["resolution"]

        # Override model to use Z-Image (user might pass "z-image" with hyphen)
        model = "z_image"

    elif task_type == "z_image_turbo_i2i":
        # Z-Image turbo img2img - fast image-to-image generation
        import tempfile
        from PIL import Image

        # Get image URL
        image_url = db_task_params.get("image") or db_task_params.get("image_url")
        if not image_url:
            raise ValueError(f"Task {task_id}: 'image' or 'image_url' required for z_image_turbo_i2i")

        # Download image to local file (required for WGP)
        local_image_path = None
        try:
            import requests
            headless_logger.debug(f"Downloading image for img2img: {image_url}", task_id=task_id)
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Save to temp file and keep it for WGP to use
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(response.content)
                local_image_path = tmp_file.name

            headless_logger.info(f"[Z_IMAGE_I2I] Downloaded image to: {local_image_path}", task_id=task_id)

            # Handle resolution - auto-detect from image or use provided
            image_size = db_task_params.get("image_size", "auto")
            if image_size == "auto" or not db_task_params.get("resolution"):
                with Image.open(local_image_path) as img:
                    width, height = img.size

                    # Scale to approximately 1 megapixel (1024x1024) while preserving aspect ratio
                    target_pixels = IMG2IMG_TARGET_MEGAPIXELS
                    current_pixels = width * height

                    if current_pixels > 0:
                        # Calculate scale factor to reach target pixels
                        import math
                        scale = math.sqrt(target_pixels / current_pixels)

                        # Apply scale to both dimensions
                        scaled_width = int(round(width * scale))
                        scaled_height = int(round(height * scale))

                        # Round to nearest multiple of 8 (Z-Image requirement)
                        scaled_width = (scaled_width // 8) * 8
                        scaled_height = (scaled_height // 8) * 8

                        # Ensure minimum size
                        scaled_width = max(8, scaled_width)
                        scaled_height = max(8, scaled_height)

                        generation_params["resolution"] = f"{scaled_width}x{scaled_height}"
                        headless_logger.info(
                            f"Scaled resolution: {width}x{height} ({current_pixels:,} px) → "
                            f"{scaled_width}x{scaled_height} ({scaled_width*scaled_height:,} px, scale={scale:.3f})",
                            task_id=task_id
                        )
                    else:
                        generation_params["resolution"] = DEFAULT_IMAGE_RESOLUTION
            elif "resolution" in db_task_params:
                generation_params["resolution"] = db_task_params["resolution"]
            else:
                generation_params["resolution"] = DEFAULT_IMAGE_RESOLUTION

        except (OSError, ValueError, RuntimeError) as e:
            # If download fails, clean up and raise error
            if local_image_path:
                try:
                    os.unlink(local_image_path)
                except OSError as e_cleanup:
                    headless_logger.debug(f"Failed to clean up temp image file after download error: {e_cleanup}", task_id=task_id)
            raise ValueError(f"Task {task_id}: Failed to download image for img2img: {e}") from e

        # CRITICAL: Pass local file path (not URL) to WGP
        generation_params["image_start"] = local_image_path

        # Set img2img parameters
        generation_params.setdefault("video_prompt_type", "")  # Image input handled via image_start
        generation_params.setdefault("video_length", 1)  # Single image output
        generation_params.setdefault("guidance_scale", 0)  # Z-Image uses guidance_scale=0
        generation_params.setdefault("num_inference_steps", int(db_task_params.get("num_inference_steps", 12)))

        # CRITICAL: Add denoising_strength to generation_params so it gets passed through
        actual_strength = db_task_params.get("denoising_strength") or db_task_params.get("denoise_strength") or db_task_params.get("strength", 0.7)
        generation_params["denoising_strength"] = actual_strength

        headless_logger.info(
            f"[Z_IMAGE_I2I] Setup complete - local_image={local_image_path}, resolution={generation_params['resolution']}, "
            f"strength={actual_strength}, steps={generation_params['num_inference_steps']}",
            task_id=task_id
        )

        # Override model to use Z-Image img2img
        model = "z_image_img2img"

    elif task_type in ("ltx2_multiframe", "ltx2_ic_multiframe"):
        # Download guide images from URLs to local paths
        guide_images_raw = db_task_params.get("guide_images", [])
        if guide_images_raw:
            import tempfile
            import requests
            resolved_guides = []
            for i, entry in enumerate(guide_images_raw):
                img_source = entry.get("image") or entry.get("image_url")
                if img_source and img_source.startswith(("http://", "https://")):
                    response = requests.get(img_source, timeout=30)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(response.content)
                        local_path = tmp.name
                else:
                    local_path = img_source
                resolved_guides.append({
                    "image": local_path,
                    "frame_idx": entry.get("frame_idx", 0),
                    "strength": entry.get("strength", 1.0),
                })
            generation_params["guide_images"] = resolved_guides
            headless_logger.info(
                f"[{task_type.upper()}] Resolved {len(resolved_guides)} guide images",
                task_id=task_id
            )

    elif task_type == "ltx2_controlled":
        # Unified LTX2 controlled task type: any combination of IC-LoRAs, standard LoRAs, and image guides

        # 2a. Parse and validate ic_loras
        ic_loras = db_task_params.get("ic_loras", [])
        if ic_loras:
            VALID_IC_TYPES = {"pose": "P", "depth": "D", "canny": "E"}
            vpt_flags = []
            for entry in ic_loras:
                ic_type = entry.get("type", "").lower()
                if ic_type not in VALID_IC_TYPES:
                    raise ValueError(f"Invalid IC-LoRA type '{ic_type}'. Must be one of: {list(VALID_IC_TYPES.keys())}")
                vpt_flags.append(VALID_IC_TYPES[ic_type])

            # Build video_prompt_type: IC flags + V (video conditioning)
            # "G" (guide mode) is added later only if image_guides are present
            generation_params["video_prompt_type"] = "".join(vpt_flags) + "V"

            # Weight from first IC-LoRA (primary control weight)
            generation_params["control_net_weight"] = ic_loras[0].get("weight", 1.0)
            if len(ic_loras) > 1:
                generation_params["control_net_weight2"] = ic_loras[1].get("weight", 1.0)

            # Download guide_video from the IC-LoRA entry
            guide_video_source = ic_loras[0].get("guide_video")
            if guide_video_source:
                if guide_video_source.startswith(("http://", "https://")):
                    local_path = _download_to_temp(guide_video_source, suffix=".mp4")
                    generation_params["video_guide"] = local_path
                else:
                    if not os.path.exists(guide_video_source):
                        raise ValueError(f"guide_video local path does not exist: {guide_video_source}")
                    generation_params["video_guide"] = guide_video_source
            else:
                headless_logger.warning(
                    f"[LTX2_CONTROLLED] IC-LoRA entry has no guide_video",
                    task_id=task_id
                )

            headless_logger.info(
                f"[LTX2_CONTROLLED] IC-LoRAs: types={[e.get('type') for e in ic_loras]}, "
                f"vpt={generation_params['video_prompt_type']}",
                task_id=task_id
            )

        # 2b. Parse and validate loras (standard LoRAs)
        loras_list = db_task_params.get("loras", [])
        if loras_list:
            activated = []
            multipliers = []
            for lora_entry in loras_list:
                path = lora_entry.get("path", "")
                weight = lora_entry.get("weight", 1.0)
                if path:
                    activated.append(path)
                    multipliers.append(str(weight))
            if activated:
                existing = generation_params.get("activated_loras", [])
                existing_mults = generation_params.get("loras_multipliers", "").split()
                generation_params["activated_loras"] = existing + activated
                generation_params["loras_multipliers"] = " ".join(existing_mults + multipliers)
                headless_logger.info(
                    f"[LTX2_CONTROLLED] Standard LoRAs: {len(activated)} loaded",
                    task_id=task_id
                )

        # 2c. Parse and validate image_guides
        image_guides = db_task_params.get("image_guides", [])
        video_length = db_task_params.get("video_length", 121)

        if image_guides:
            resolved_guides = []
            for guide_entry in image_guides:
                img_source = guide_entry.get("image") or guide_entry.get("image_url")
                anchors = guide_entry.get("anchors", [])

                if not img_source:
                    raise ValueError("Each image_guide must have an 'image' field")
                if not anchors:
                    raise ValueError("Each image_guide must have at least one anchor")

                # Download image if URL
                if img_source.startswith(("http://", "https://")):
                    local_path = _download_to_temp(img_source, suffix=".png")
                else:
                    if not os.path.exists(img_source):
                        raise ValueError(f"image_guides local path does not exist: {img_source}")
                    local_path = img_source

                # Expand anchors: one image can map to multiple frame positions
                for anchor in anchors:
                    frame = anchor.get("frame", 0)
                    weight = anchor.get("weight", 1.0)

                    if frame != -1 and (frame < 0 or frame >= video_length):
                        raise ValueError(
                            f"Anchor frame {frame} out of range [0, {video_length - 1}]. "
                            f"Use -1 for the last frame."
                        )

                    resolved_guides.append({
                        "image": local_path,
                        "frame_idx": frame,
                        "strength": weight,
                    })

            generation_params["guide_images"] = resolved_guides

            # Add "G" flag to video_prompt_type if IC-LoRAs set it
            if "video_prompt_type" in generation_params:
                generation_params["video_prompt_type"] += "G"

            headless_logger.info(
                f"[LTX2_CONTROLLED] Resolved {len(resolved_guides)} guide image anchors "
                f"from {len(image_guides)} source images",
                task_id=task_id
            )

        # 2d. Store applied controls metadata
        applied_controls = {}
        if ic_loras:
            applied_controls["ic_loras"] = [
                {"type": e.get("type"), "weight": e.get("weight", 1.0)} for e in ic_loras
            ]
        if loras_list:
            applied_controls["loras"] = [
                {"path": e.get("path"), "weight": e.get("weight", 1.0)} for e in loras_list
            ]
        if image_guides:
            applied_controls["image_guides"] = [
                {"anchors": e.get("anchors", []), "image": e.get("image")} for e in image_guides
            ]
        if applied_controls:
            generation_params["_applied_controls_metadata"] = applied_controls

    # Defaults
    essential_defaults = {
        "seed": -1,
        "negative_prompt": "",
    }
    for param, default_value in essential_defaults.items():
        if param not in generation_params:
            generation_params[param] = default_value
    
    # Phase Config Override
    if "phase_config" in db_task_params:
        try:
            steps_per_phase = db_task_params["phase_config"].get("steps_per_phase", [2, 2, 2])
            phase_config_steps = sum(steps_per_phase)

            parsed_phase_config = parse_phase_config(
                phase_config=db_task_params["phase_config"],
                num_inference_steps=phase_config_steps,
                task_id=task_id,
                model_name=model,
                debug_mode=debug_mode
            )

            generation_params["num_inference_steps"] = phase_config_steps

            for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                       "guidance_scale", "guidance2_scale", "guidance3_scale",
                       "flow_shift", "sample_solver", "model_switch_phase",
                       "lora_names", "lora_multipliers", "additional_loras"]:
                if key in parsed_phase_config and parsed_phase_config[key] is not None:
                    generation_params[key] = parsed_phase_config[key]

            if "lora_names" in parsed_phase_config:
                generation_params["activated_loras"] = parsed_phase_config["lora_names"]
            if "lora_multipliers" in parsed_phase_config:
                generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])

            if "_patch_config" in parsed_phase_config:
                generation_params["_parsed_phase_config"] = parsed_phase_config
                generation_params["_phase_config_model_name"] = model
            
            # Note: LoRA URL resolution is handled centrally in HeadlessTaskQueue._convert_to_wgp_task()
            # URLs in activated_loras are detected by LoRAConfig.from_params() and downloaded there
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Task {task_id}: Invalid phase_config: {e}") from e

    priority = db_task_params.get("priority", 0)
    if task_type.endswith("_orchestrator"):
        priority = max(priority, 10)
    
    generation_task = GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=generation_params,
        priority=priority
    )
    
    return generation_task

