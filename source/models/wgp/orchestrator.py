"""
WanGP Content Generation Orchestrator

A thin wrapper around wgp.py's generate_video() function for programmatic use.
Supports T2V, VACE, and Flux generation without running the Gradio UI.
"""

import os
import sys
import traceback
from typing import Optional, List, TYPE_CHECKING

# Import structured logging
from source.core.log import (
    orchestrator_logger, model_logger, generation_logger,
)

# Type hints for TaskConfig (avoid circular import)
if TYPE_CHECKING:
    from source.core.params import TaskConfig

# ---------------------------------------------------------------------------
# Delegated modules (source/models/wgp/ package)
# ---------------------------------------------------------------------------
from source.models.wgp.error_extraction import _extract_wgp_error, LOG_TAIL_MAX_CHARS
from source.models.wgp.param_resolution import resolve_parameters as _resolve_parameters_impl
from source.models.wgp.lora_setup import setup_loras_for_model as _setup_loras_for_model_impl
from source.models.wgp.model_ops import (
    load_missing_model_definition as _load_missing_model_definition_impl,
    load_model_impl as _load_model_impl,
    unload_model_impl as _unload_model_impl,
    get_or_load_uni3c_controlnet as _get_or_load_uni3c_controlnet_impl,
)
from source.models.wgp.generation_helpers import (
    verify_wgp_directory as _verify_wgp_directory,
    create_vace_fixed_generate_video as _create_vace_fixed_generate_video_impl,
    load_image as _load_image_impl,
    resolve_media_path as _resolve_media_path_impl,
    is_vace as _is_vace_impl,
    is_model_vace as _is_model_vace_impl,
    is_flux as _is_flux_impl,
    is_t2v as _is_t2v_impl,
    is_qwen as _is_qwen_impl,
)

# Import debug mode flag
try:
    from worker import debug_mode
except ImportError:
    debug_mode = False


class WanOrchestrator:
    """Thin adapter around `wgp.generate_video` for easier programmatic use."""

    def __init__(self, wan_root: str, main_output_dir: Optional[str] = None):
        """Initialize orchestrator with WanGP directory.

        Args:
            wan_root: Path to WanGP repository root directory (MUST be absolute path to Wan2GP/)
            main_output_dir: Optional path for output directory. If not provided, defaults to
                           'outputs' directory next to wan_root (preserves backwards compatibility)

        IMPORTANT: Caller MUST have already changed to wan_root directory before calling this.
        wgp.py uses relative paths and expects to run from Wan2GP/.
        """
        import logging
        _init_logger = logging.getLogger('HeadlessQueue')

        # Store the wan_root (should match current directory)
        self.wan_root = os.path.abspath(wan_root)
        current_dir = os.getcwd()

        if debug_mode:
            _init_logger.info(f"[INIT_DEBUG] WanOrchestrator.__init__ called with wan_root: {self.wan_root}")
            _init_logger.info(f"[INIT_DEBUG] Current working directory: {current_dir}")

        # CRITICAL CHECK: Verify caller already changed to the correct directory
        # wgp.py will import and execute module-level code that uses relative paths like "defaults/*.json"
        # If we're in the wrong directory, wgp will load 0 models and fail mysteriously
        if current_dir != self.wan_root:
            error_msg = (
                f"CRITICAL: WanOrchestrator must be initialized from Wan2GP directory!\n"
                f"  Current directory: {current_dir}\n"
                f"  Expected directory: {self.wan_root}\n"
                f"  Caller must chdir() before creating WanOrchestrator instance."
            )
            if debug_mode:
                _init_logger.error(f"[INIT_DEBUG] {error_msg}")
            raise RuntimeError(error_msg)

        # Verify Wan2GP structure
        if not os.path.isdir("defaults"):
            raise RuntimeError(
                f"defaults/ directory not found in {current_dir}. "
                f"This doesn't appear to be a valid Wan2GP directory!"
            )
        if not os.path.isdir("models"):
            if debug_mode:
                _init_logger.warning(f"[INIT_DEBUG] models/ directory not found in {current_dir}")

        # Ensure Wan2GP is first in sys.path so wgp.py imports correctly
        if self.wan_root in sys.path:
            sys.path.remove(self.wan_root)
        sys.path.insert(0, self.wan_root)
        if debug_mode:
            _init_logger.info(f"[INIT_DEBUG] Added {self.wan_root} to sys.path[0]")
        
        # Optional smoke/CPU-only modes
        self.smoke_mode = bool(os.environ.get("HEADLESS_WAN2GP_SMOKE", ""))
        force_cpu = os.environ.get("HEADLESS_WAN2GP_FORCE_CPU", "0") == "1"

        # Force CPU if requested and guard CUDA capability queries before importing WGP
        if force_cpu and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import torch  # type: ignore
            # If CUDA isn't available, stub capability query used by upstream on import
            if force_cpu or not torch.cuda.is_available():
                try:
                    def _safe_get_device_capability(device=None):
                        return (8, 0)
                    torch.cuda.get_device_capability = _safe_get_device_capability  # type: ignore
                except (RuntimeError, AttributeError) as e:
                    _init_logger.debug("Failed to stub torch.cuda.get_device_capability: %s", e)
        except ImportError as e:
            _init_logger.debug("Failed to import torch for CPU/CUDA guards: %s", e)

        # Force a headless Matplotlib backend to avoid Tkinter requirements during upstream imports
        if not os.environ.get("MPLBACKEND"):
            os.environ["MPLBACKEND"] = "Agg"
        try:
            import matplotlib  # type: ignore
            _orig_use = matplotlib.use  # type: ignore
            def _force_agg(_backend=None, *args, **kwargs):
                try:
                    return _orig_use('Agg', force=True)
                except (RuntimeError, ValueError) as e:
                    return None
            matplotlib.use = _force_agg  # type: ignore
        except ImportError as e:
            _init_logger.debug("Failed to force matplotlib Agg backend: %s", e)

        # Pre-import config hygiene for upstream wgp.py
        # Some older configs may store preload_model_policy as an int, but upstream now expects a list
        try:
            cfg_path = os.path.join(self.wan_root, "wgp_config.json")
            if os.path.isfile(cfg_path):
                import json as _json
                with open(cfg_path, "r", encoding="utf-8") as _r:
                    _cfg = _json.load(_r)
                changed = False
                pmp = _cfg.get("preload_model_policy", [])
                if isinstance(pmp, int):
                    _cfg["preload_model_policy"] = []  # disable preloading and fix type
                    changed = True
                # Ensure save paths exist and are strings
                if not isinstance(_cfg.get("save_path", ""), str):
                    _cfg["save_path"] = "outputs"
                    changed = True
                if not isinstance(_cfg.get("image_save_path", ""), str):
                    _cfg["image_save_path"] = "outputs"
                    changed = True
                if changed:
                    with open(cfg_path, "w", encoding="utf-8") as _w:
                        _json.dump(_cfg, _w, indent=4)
        except (OSError, ValueError, KeyError, TypeError):
            # Config hygiene should not block import
            pass

        # Import WGP components after path setup (skip entirely in smoke mode)
        if not self.smoke_mode:
            try:
                if debug_mode:
                    _init_logger.info(f"[INIT_DEBUG] About to import wgp module")
                current_dir = os.getcwd()
                if debug_mode:
                    _init_logger.info(f"[INIT_DEBUG] Current directory: {current_dir}")
                    _init_logger.info(f"[INIT_DEBUG] sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
                    _init_logger.info(f"[INIT_DEBUG] wgp already in sys.modules: {'wgp' in sys.modules}")

                # Double-check we're in the right directory before importing
                if current_dir != self.wan_root:
                    if debug_mode:
                        _init_logger.error(f"[INIT_DEBUG] CRITICAL: Current directory {current_dir} != expected {self.wan_root}")
                    raise RuntimeError(f"Directory changed unexpectedly before wgp import: {current_dir} != {self.wan_root}")

                # If wgp was previously imported from wrong directory, remove it so it reimports
                if 'wgp' in sys.modules:
                    if debug_mode:
                        _init_logger.warning(f"[INIT_DEBUG] wgp already in sys.modules - removing to force reimport from correct directory")
                    del sys.modules['wgp']

                _saved_argv = list(sys.argv)
                sys.argv = ["headless_wgp.py"]
                from wgp import (
                    generate_video, get_base_model_type, get_model_family,
                    test_vace_module, apply_changes
                )

                # Verify directory didn't change during wgp import
                _verify_wgp_directory(_init_logger, "after importing wgp module")

                # Apply VACE fix wrapper to generate_video
                self._generate_video = _create_vace_fixed_generate_video_impl(generate_video)
                self._get_base_model_type = get_base_model_type
                self._get_model_family = get_model_family
                self._test_vace_module = test_vace_module
                self._apply_changes = apply_changes

                # Apply WGP monkeypatches for headless operation (Qwen support, LoRA fixes, etc.)
                import wgp as wgp
                from source.models.wgp.wgp_patches import apply_all_wgp_patches
                apply_all_wgp_patches(wgp, self.wan_root)
                
                # Initialize WGP global state (normally done by UI)
                import wgp
                # Set absolute output path to avoid issues when working directory changes
                # Use main_output_dir if provided, otherwise default to 'outputs' next to wan_root
                if main_output_dir is not None:
                    absolute_outputs_path = os.path.abspath(main_output_dir)
                    model_logger.debug(f"[OUTPUT_DIR] Using provided main_output_dir: {absolute_outputs_path}")
                else:
                    absolute_outputs_path = os.path.abspath(os.path.join(os.path.dirname(self.wan_root), 'outputs'))
                    model_logger.debug(f"[OUTPUT_DIR] Using default output directory: {absolute_outputs_path}")

                # Initialize attributes that don't exist yet
                for attr, default in {
                    'wan_model': None, 'offloadobj': None, 'reload_needed': True,
                    'transformer_type': None
                }.items():
                    if not hasattr(wgp, attr):
                        setattr(wgp, attr, default)

                # CRITICAL: wgp.py has BOTH a server_config dictionary AND module-level variables
                # Lines 2347-2348 of wgp.py:
                #   save_path = server_config.get("save_path", ...)
                #   image_save_path = server_config.get("image_save_path", ...)
                # We must update BOTH the dictionary AND the module-level variables!
                if not hasattr(wgp, 'server_config'):
                    wgp.server_config = {}

                # Store old values for debugging
                _old_dict_save_path = wgp.server_config.get('save_path', 'NOT_SET')
                _old_dict_image_save_path = wgp.server_config.get('image_save_path', 'NOT_SET')
                _old_module_save_path = getattr(wgp, 'save_path', 'NOT_SET')
                _old_module_image_save_path = getattr(wgp, 'image_save_path', 'NOT_SET')

                # Update the dictionary (for consistency)
                wgp.server_config['save_path'] = absolute_outputs_path
                wgp.server_config['image_save_path'] = absolute_outputs_path

                # Update the module-level variables (this is what wgp.py actually uses!)
                wgp.save_path = absolute_outputs_path
                wgp.image_save_path = absolute_outputs_path

                model_logger.info(f"[OUTPUT_DIR] BEFORE: dict={{save_path={_old_dict_save_path}, image_save_path={_old_dict_image_save_path}}}, module={{save_path={_old_module_save_path}, image_save_path={_old_module_image_save_path}}}")
                model_logger.info(f"[OUTPUT_DIR] AFTER:  dict={{save_path={wgp.server_config['save_path']}, image_save_path={wgp.server_config['image_save_path']}}}, module={{save_path={wgp.save_path}, image_save_path={wgp.image_save_path}}}")
                
                # Debug: Check if model definitions are loaded
                if debug_mode:
                    model_logger.info(f"[INIT_DEBUG] Available models after WGP import: {list(wgp.models_def.keys())}")
                _verify_wgp_directory(model_logger, "after wgp setup and monkeypatching")

                if not wgp.models_def:
                    error_msg = (
                        f"CRITICAL: No model definitions found after importing wgp! "
                        f"Current directory: {os.getcwd()}, "
                        f"Expected: {self.wan_root}, "
                        f"defaults/ exists: {os.path.exists('defaults')}, "
                        f"finetunes/ exists: {os.path.exists('finetunes')}"
                    )
                    model_logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except ImportError as e:
                raise ImportError(f"Failed to import wgp module. Ensure {wan_root} contains wgp.py: {e}")
            finally:
                try:
                    sys.argv = _saved_argv
                except (TypeError, ValueError) as e:
                    _init_logger.debug("Failed to restore sys.argv after wgp import: %s", e)

        # Initialize state object (mimics UI state)
        self.state = {
            "generation": {},
            "model_type": None,
            "model_filename": "",
            "advanced": False,
            "last_model_per_family": {},
            "last_resolution_per_group": {},
            "gen": {
                "queue": [],
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "prompt_no": 1,
                "prompts_max": 1
            },
            "loras": [],
            "loras_names": [],
            "loras_presets": {},
            # Additional state properties to prevent future KeyErrors
            "validate_success": 1,      # Required for validation checks
            "apply_success": 1,         # Required for settings application  
            "refresh": None,            # Required for UI refresh operations
            "all_settings": {},        # Required for settings persistence
            "image_mode_tab": 0,       # Required for image/video mode switching
            "prompt": ""               # Required for prompt handling
        }
        
        # Apply sensible defaults (mirrors typical UI defaults)
        if not self.smoke_mode:
            # Ensure model_type is set before applying config, as upstream generate_header/get_overridden_attention
            # will look up the current model's definition via state["model_type"].
            try:
                import wgp as _w
                default_model_type = getattr(_w, "transformer_type", None) or "t2v"
            except (ImportError, AttributeError):
                default_model_type = "t2v"
            self.state["model_type"] = default_model_type

            # Upstream apply_changes signature accepts a single save_path_choice
            outputs_dir = "outputs/"
            try:
                orchestrator_logger.debug("Calling wgp.apply_changes() to initialize defaults...")
                self._apply_changes(
                    self.state,
                    transformer_types_choices=["t2v"],
                    transformer_dtype_policy_choice="auto",
                    text_encoder_quantization_choice="bf16",
                    VAE_precision_choice="fp32",
                    mixed_precision_choice=0,
                    save_path_choice=outputs_dir,
                    image_save_path_choice=outputs_dir,
                    attention_choice="auto",
                    compile_choice="",
                    profile_choice=1,  # Profile 1 for 24GB+ VRAM (4090/3090)
                    vae_config_choice="default",
                    metadata_choice="none",
                    quantization_choice="int8",
                    preload_model_policy_choice=[]
                )
                orchestrator_logger.debug("wgp.apply_changes() completed successfully")
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                orchestrator_logger.error(f"❌ FATAL: wgp.apply_changes() failed: {e}\n{traceback.format_exc()}")
                raise RuntimeError(f"Failed to apply WGP defaults during orchestrator init: {e}") from e

            # Verify directory after apply_changes (it may have done file operations)
            _verify_wgp_directory(orchestrator_logger, "after apply_changes()")

            # CRITICAL: apply_changes() reads from server_config dict back into module-level vars
            # So we must set our output paths AGAIN after apply_changes() completes!
            orchestrator_logger.info(f"[OUTPUT_DIR] Reapplying output directory configuration after apply_changes()...")
            wgp.server_config['save_path'] = absolute_outputs_path
            wgp.server_config['image_save_path'] = absolute_outputs_path
            wgp.save_path = absolute_outputs_path
            wgp.image_save_path = absolute_outputs_path
            orchestrator_logger.info(f"[OUTPUT_DIR] Final check: dict={{save_path={wgp.server_config['save_path']}, image_save_path={wgp.server_config['image_save_path']}}}, module={{save_path={wgp.save_path}, image_save_path={wgp.image_save_path}}}")

        else:
            # Provide stubbed helpers for smoke mode
            self._get_base_model_type = lambda model_key: ("t2v" if "flux" not in (model_key or "") else "flux")
            self._get_model_family = lambda model_key, for_ui=False: ("VACE" if "vace" in (model_key or "") else ("Flux" if "flux" in (model_key or "") else "T2V"))
            self._test_vace_module = lambda model_name: ("vace" in (model_name or ""))
        self.current_model = None
        self.offloadobj = None  # Store WGP's offload object
        self.passthrough_mode = False  # Flag for explicit passthrough mode
        self._cached_uni3c_controlnet = None  # Cached Uni3C ControlNet to avoid reloading from disk

        orchestrator_logger.success(f"WanOrchestrator initialized with WGP at {wan_root}")

        # Final notice for smoke mode
        if self.smoke_mode:
            orchestrator_logger.warning("HEADLESS_WAN2GP_SMOKE enabled: generation will return sample outputs only")
        
    def _load_missing_model_definition(self, model_key: str, json_path: str):
        """Delegate to source.models.wgp.model_ops."""
        return _load_missing_model_definition_impl(self, model_key, json_path)

    def load_model(self, model_key: str) -> bool:
        """Load and validate a model type using WGP's exact generation-time pattern.

        Delegates to source.models.wgp.model_ops.load_model_impl.
        """
        return _load_model_impl(self, model_key)

    def unload_model(self):
        """Unload the current model. Delegates to source.models.wgp.model_ops."""
        return _unload_model_impl(self)

    def _setup_loras_for_model(self, model_type: str):
        """Initialize LoRA discovery. Delegates to source.models.wgp.lora_setup."""
        return _setup_loras_for_model_impl(self, model_type)

    def _create_vace_fixed_generate_video(self, original_generate_video):
        """Delegate to source.models.wgp.generation_helpers."""
        return _create_vace_fixed_generate_video_impl(original_generate_video)

    def _is_vace(self) -> bool:
        """Check if current model is a VACE model. Delegates to source.models.wgp.generation_helpers."""
        return _is_vace_impl(self)

    def is_model_vace(self, model_name: str) -> bool:
        """Check if a given model name is a VACE model (model-agnostic). Delegates to source.models.wgp.generation_helpers."""
        return _is_model_vace_impl(self, model_name)

    def _is_flux(self) -> bool:
        """Check if current model is a Flux model. Delegates to source.models.wgp.generation_helpers."""
        return _is_flux_impl(self)

    def _is_t2v(self) -> bool:
        """Check if current model is a T2V model. Delegates to source.models.wgp.generation_helpers."""
        return _is_t2v_impl(self)

    def _is_qwen(self) -> bool:
        """Check if current model is a Qwen image model. Delegates to source.models.wgp.generation_helpers."""
        return _is_qwen_impl(self)

    def _get_or_load_uni3c_controlnet(self):
        """Get cached Uni3C controlnet. Delegates to source.models.wgp.model_ops."""
        return _get_or_load_uni3c_controlnet_impl(self)

    def _load_image(self, path: Optional[str], mask: bool = False):
        """Load image from path. Delegates to source.models.wgp.generation_helpers."""
        return _load_image_impl(self, path, mask=mask)

    def _resolve_media_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve media paths. Delegates to source.models.wgp.generation_helpers."""
        return _resolve_media_path_impl(self, path)

    def _resolve_parameters(self, model_type: str, task_params: dict) -> dict:
        """Resolve generation parameters. Delegates to source.models.wgp.param_resolution."""
        return _resolve_parameters_impl(self, model_type, task_params)

    def generate(self, 
                prompt: str,
                model_type: str = None,
                # Common parameters - None means "use model/system defaults"
                resolution: Optional[str] = None,
                video_length: Optional[int] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                seed: Optional[int] = None,
                # VACE parameters
                video_guide: Optional[str] = None,
                video_mask: Optional[str] = None,
                video_prompt_type: Optional[str] = None,
                control_net_weight: Optional[float] = None,
                control_net_weight2: Optional[float] = None,
                # Flux parameters
                embedded_guidance_scale: Optional[float] = None,
                # LoRA parameters
                lora_names: Optional[List[str]] = None,
                lora_multipliers: Optional[List] = None,  # Can be List[float] or List[str] for phase-config
                # Other parameters
                negative_prompt: Optional[str] = None,
                batch_size: Optional[int] = None,
                **kwargs) -> str:
        """Generate content using the loaded model.
        
        Args:
            prompt: Text prompt for generation
            resolution: Output resolution (e.g., "1280x720", "1024x1024")
            video_length: Number of frames for video or images for Flux
            num_inference_steps: Denoising steps
            guidance_scale: CFG guidance strength
            seed: Random seed for reproducibility
            video_guide: Path to control video (required for VACE)
            video_mask: Path to mask video (optional)
            video_prompt_type: VACE encoding type (e.g., "VP", "VPD", "VPDA")
            control_net_weight: Strength for first VACE encoding
            control_net_weight2: Strength for second VACE encoding
            embedded_guidance_scale: Flux-specific guidance
            lora_names: List of LoRA filenames
            lora_multipliers: List of LoRA strength multipliers
            negative_prompt: Negative prompt text
            batch_size: Batch size for generation
            **kwargs: Additional parameters passed to generate_video
            
        Returns:
            Path to generated output file(s)
            
        Raises:
            RuntimeError: If no model is loaded
            ValueError: If required parameters are missing for model type
        """
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # [DEBUG_KWARGS] Trace image_start presence
        generation_logger.info(f"[DEBUG_KWARGS] generate received kwargs keys: {list(kwargs.keys())}")
        if "image_start" in kwargs:
            generation_logger.info(f"[DEBUG_KWARGS] image_start type: {type(kwargs['image_start'])}, value: {kwargs['image_start']}")
        else:
            generation_logger.info(f"[DEBUG_KWARGS] image_start NOT in kwargs")

        # ------------------------------------------------------------------
        # SVI / Image-Refs Path Bridging (GROUND TRUTH FIX)
        # ------------------------------------------------------------------
        # Our task pipeline often passes `image_refs_paths` (list[str]) so we can
        # keep tasks JSON-serializable. Wan2GP/WGP expects `image_refs` as a list
        # of PIL.Image objects.
        #
        # If we don't convert here, logs will show `image_refs: None` and SVI's
        # anchor reference won't actually be used (can cause odd end-frame behavior).
        try:
            if (
                "image_refs_paths" in kwargs
                and kwargs.get("image_refs") in (None, "", [])
                and isinstance(kwargs["image_refs_paths"], (list, tuple))
                and len(kwargs["image_refs_paths"]) > 0
            ):
                from PIL import Image
                from PIL import ImageOps

                refs: list = []
                for p in kwargs["image_refs_paths"]:
                    if not p:
                        continue
                    try:
                        img = Image.open(str(p)).convert("RGB")
                        img = ImageOps.exif_transpose(img)
                        refs.append(img)
                    except (OSError, ValueError, RuntimeError) as e_img:
                        if debug_mode:
                            generation_logger.warning(f"[SVI_GROUND_TRUTH] Failed to load image_ref path '{p}': {e_img}")
                if refs:
                    kwargs["image_refs"] = refs
                    if debug_mode:
                        generation_logger.info(f"[SVI_GROUND_TRUTH] Converted image_refs_paths -> image_refs (count={len(refs)})")
                else:
                    if debug_mode:
                        generation_logger.warning("[SVI_GROUND_TRUTH] image_refs_paths provided but no images could be loaded")
        except (OSError, ValueError, RuntimeError, TypeError) as e_refs:
            if debug_mode:
                generation_logger.warning(f"[SVI_GROUND_TRUTH] Exception while converting image_refs_paths -> image_refs: {e_refs}")

        # Smoke-mode short-circuit: create a sample output and return its path
        if self.smoke_mode:
            from pathlib import Path
            import shutil
            out_dir = Path(os.getcwd()) / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            _project_root = Path(__file__).parent.parent.parent.parent
            sample_src = Path(os.path.abspath(os.path.join(_project_root, "samples", "test.mp4")))
            if not sample_src.exists():
                # Fallback to project-level samples directory
                sample_src = Path(os.path.abspath(os.path.join(_project_root, "samples", "video.mp4")))
            ts = __import__("time").strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"smoke_{self.current_model}_{ts}.mp4"
            try:
                shutil.copyfile(str(sample_src), str(out_path))
            except OSError as e:
                # If copy fails for any reason, create an empty file as placeholder
                generation_logger.debug("[SMOKE] Failed to copy sample file %s, creating empty placeholder: %s", sample_src, e)
                out_path.write_bytes(b"")
            # Mimic WGP state behavior
            try:
                self.state["gen"]["file_list"].append(str(out_path))
            except (KeyError, TypeError) as e:
                generation_logger.debug("[SMOKE] Failed to update state gen file_list: %s", e)
            generation_logger.info(f"[SMOKE] Generated placeholder output at: {out_path}")
            return str(out_path)

        # Use provided model_type or current loaded model
        effective_model_type = model_type or self.current_model
        
        # Build task explicit parameters: only non-None values are considered "explicit"
        # This ensures method signature defaults don't override model/system defaults
        task_explicit_params = {"prompt": prompt}  # prompt is always required
        
        # Add all non-None parameters from method signature
        param_values = {
            "resolution": resolution,
            "video_length": video_length,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "video_guide": self._resolve_media_path(video_guide),
            "video_mask": self._resolve_media_path(video_mask),
            "video_prompt_type": video_prompt_type,
            "control_net_weight": control_net_weight,
            "control_net_weight2": control_net_weight2,
            "embedded_guidance_scale": embedded_guidance_scale,
            "lora_names": lora_names,
            "lora_multipliers": lora_multipliers,
            "negative_prompt": negative_prompt,
            "batch_size": batch_size,
        }
        
        # Only include explicitly passed parameters (non-None values)
        for param, value in param_values.items():
            if value is not None:
                task_explicit_params[param] = value
        
        # Add any additional kwargs
        task_explicit_params.update(kwargs)

        # Initialize LoRA variables (needed for both modes)
        activated_loras = []
        loras_multipliers_str = ""

        # Resolve final parameters with proper precedence (skip in passthrough mode)
        if self.passthrough_mode:
            # In passthrough mode, use task parameters directly without any resolution
            resolved_params = task_explicit_params.copy()
            generation_logger.info(f"[PASSTHROUGH] Using task parameters directly without resolution: {len(resolved_params)} params")
        else:
            resolved_params = self._resolve_parameters(effective_model_type, task_explicit_params)
            generation_logger.info(f"[PARAM_RESOLUTION] Final parameters for '{effective_model_type}': num_inference_steps={resolved_params.get('num_inference_steps')}, guidance_scale={resolved_params.get('guidance_scale')}")

        # Determine model types for generation
        _base_model_type = self._get_base_model_type(self.current_model)
        _vace_test_result = self._test_vace_module(self.current_model)
        
        is_vace = self._is_vace()
        is_flux = self._is_flux()
        is_qwen = self._is_qwen()
        is_t2v = self._is_t2v()
        
        generation_logger.debug(f"Model detection - VACE: {is_vace}, Flux: {is_flux}, T2V: {is_t2v}")
        generation_logger.debug(f"Generation parameters - prompt: '{prompt[:50]}...', resolution: {resolution}, length: {video_length}")
        
        if is_vace:
            # Ensure sane defaults for required VACE controls
            if not video_prompt_type:
                video_prompt_type = "VP"
            if control_net_weight is None:
                control_net_weight = 1.0
            if control_net_weight2 is None:
                control_net_weight2 = 1.0
            generation_logger.debug(f"VACE parameters - guide: {video_guide}, type: {video_prompt_type}, weights: {control_net_weight}/{control_net_weight2}")

        # Validate model-specific requirements
        # Resolve media paths before validation to ensure correct location is used
        video_guide = self._resolve_media_path(video_guide)
        video_mask = self._resolve_media_path(video_mask)

        if is_vace and not video_guide:
            raise ValueError("VACE models require video_guide parameter")

        # Extract resolved parameters
        final_video_length = resolved_params.get("video_length", 49)
        final_batch_size = resolved_params.get("batch_size", 1)
        final_guidance_scale = resolved_params.get("guidance_scale", 7.5)
        final_embedded_guidance = resolved_params.get("embedded_guidance_scale", 3.0)

        # JSON parameters pass through directly to WGP - no processing needed
        
        # Configure model-specific parameters
        if is_flux:
            image_mode = 1
            # For Flux, video_length means number of images
            actual_video_length = 1
            actual_batch_size = final_video_length
            # Use embedded guidance for Flux
            actual_guidance = final_embedded_guidance
        elif is_qwen:
            # Qwen is an image model
            image_mode = 1
            actual_video_length = 1
            actual_batch_size = resolved_params.get("batch_size", 1)
            actual_guidance = final_guidance_scale
        else:
            image_mode = 0
            actual_video_length = final_video_length
            
            # Safety check: Wan models with latent_size=4 crash if video_length < 4
            # This happens because (len // 4) * 4 + 1 results in 1 frame,
            # and I2V logic needs at least 2 frames (start + end logic)
            if actual_video_length < 5:
                generation_logger.warning(f"[SAFETY] Boosting video_length from {actual_video_length} to 5 to prevent quantization crash")
                actual_video_length = 5
            
            actual_batch_size = final_batch_size
            actual_guidance = final_guidance_scale

        # Set up VACE parameters
        # Only disable VACE if model doesn't support it AND no VACE params were provided
        # BUT preserve video_prompt_type if caller explicitly set it (e.g., "I" for SVI image_refs)
        if not is_vace and not video_guide and not video_mask:
            video_guide = None
            video_mask = None
            # Only set to "disabled" if caller didn't explicitly request a prompt type
            # "I" is used for SVI to enable image_refs passthrough
            if video_prompt_type is None or video_prompt_type == "":
                video_prompt_type = "disabled"
            control_net_weight = 0.0
            control_net_weight2 = 0.0

        # Check if we're in JSON passthrough mode - don't process anything, just pass the JSON
        # This happens when --passthrough flag is used or when we have a full model definition
        is_passthrough_mode = self.passthrough_mode

        if is_passthrough_mode:
            # Complete passthrough mode: bypass ALL parameter processing
            generation_logger.info("Using JSON passthrough mode - ALL parameters pass through directly from JSON")

            # Don't set LoRA parameters - let WGP extract them from model JSON via get_transformer_loras()
            # WGP will call get_transformer_loras(model_type) to get LoRAs from the temp model file
        else:
            # Normal mode: LoRAs are already validated paths from LoRAConfig
            # (task_registry.py downloaded URLs and resolved paths before passing to us)
            # We just need to format them for WGP's expected string format
            
            # Check both function param (lora_names) AND kwargs (activated_loras) since callers
            # may pass LoRAs using either key depending on the code path
            activated_loras = lora_names or kwargs.get('activated_loras') or []
            
            # Format multipliers as space-separated string for WGP
            # Check both lora_multipliers (function param) and kwargs variants
            # Handles both regular floats (1.0, 0.8) and phase-config strings ("1.0;0;0")
            eff_multipliers = lora_multipliers or kwargs.get('lora_multipliers') or kwargs.get('loras_multipliers')
            if eff_multipliers:
                if isinstance(eff_multipliers, str):
                    # WGP splits on spaces; replace commas defensively
                    loras_multipliers_str = eff_multipliers.replace(',', ' ')
                else:
                    loras_multipliers_str = " ".join(str(m) for m in eff_multipliers)
            else:
                loras_multipliers_str = ""
            
            if activated_loras:
                generation_logger.info(f"Using {len(activated_loras)} LoRAs from {'lora_names' if lora_names else 'kwargs.activated_loras'}")
                generation_logger.debug(f"LoRA paths: {[os.path.basename(p) if isinstance(p, str) else str(p) for p in activated_loras]}")
            else:
                generation_logger.debug("No LoRAs to apply")

        # Create minimal task and callback objects (needed for wgp_params)
        task = {"id": 1, "params": {}, "repeats": 1}
        
        def send_cmd(cmd: str, data=None):
            if cmd == "status":
                generation_logger.debug(f"Status: {data}")
            elif cmd == "progress":
                if isinstance(data, list) and len(data) >= 2:
                    progress, status = data[0], data[1]
                    generation_logger.debug(f"Progress: {progress}% - {status}")
                else:
                    generation_logger.debug(f"Progress: {data}")
            elif cmd == "output":
                generation_logger.essential("Output generated")
            elif cmd == "exit":
                generation_logger.essential("Generation completed")
            elif cmd == "error":
                generation_logger.error(f"Error: {data}")
            elif cmd == "info":
                generation_logger.debug(f"Info: {data}")
            elif cmd == "preview":
                generation_logger.debug("Preview updated")

        if is_passthrough_mode:
            # COMPLETE PASSTHROUGH MODE: Pass ALL parameters from JSON with required defaults
            wgp_params = {
                # Core parameters (fixed, not overridable)
                'task': task,
                'send_cmd': send_cmd,
                'state': self.state,
                'model_type': self.current_model,
                'image_mode': image_mode,

                # Required parameters with defaults
                'prompt': resolved_params.get('prompt', ''),
                'negative_prompt': resolved_params.get('negative_prompt', ''),
                'resolution': resolved_params.get('resolution', '1280x720'),
                'video_length': resolved_params.get('video_length', 81),
                'batch_size': resolved_params.get('batch_size', 1),
                'seed': resolved_params.get('seed', 42),
                'force_fps': 'auto',

                # VACE control parameters
                'video_guide': video_guide,
                'video_mask': video_mask,
                'video_guide2': None,
                'video_mask2': None,
                'video_prompt_type': video_prompt_type or 'VM',
                'control_net_weight': control_net_weight or 1.0,
                'control_net_weight2': control_net_weight2 or 1.0,
                'denoising_strength': 1.0,

                # LoRA parameters
                'activated_loras': [],
                'loras_multipliers': '',

                # Audio parameters
                'audio_guidance_scale': 1.0,
                'embedded_guidance_scale': resolved_params.get('embedded_guidance_scale', 0.0),
                'repeat_generation': 1,
                'multi_prompts_gen_type': 0,
                'multi_images_gen_type': 0,
                'skip_steps_cache_type': '',
                'skip_steps_multiplier': 1.0,
                'skip_steps_start_step_perc': 0.0,

                # Image parameters
                'image_prompt_type': 'disabled',
                'image_start': None,
                'image_end': None,
                'model_mode': 0,
                'video_source': None,
                'keep_frames_video_source': '',
                'image_refs': None,
                'frames_positions': '',
                'image_guide': None,
                'keep_frames_video_guide': '',
                'video_guide_outpainting': '0 0 0 0',
                'image_mask': None,
                'mask_expand': 0,

                # Audio parameters
                'audio_guide': None,
                'audio_guide2': None,
                'audio_source': None,
                'audio_prompt_type': '',
                'speakers_locations': '',

                # Sliding window parameters
                'sliding_window_size': 129,
                'sliding_window_overlap': 0,
                'sliding_window_color_correction_strength': 0.0,
                'sliding_window_overlap_noise': 0.1,
                'sliding_window_discard_last_frames': 0,
                'image_refs_relative_size': 50,

                # Post-processing parameters
                'remove_background_images_ref': 0,
                'temporal_upsampling': '',
                'spatial_upsampling': '',
                'film_grain_intensity': 0.0,
                'film_grain_saturation': 0.0,
                'MMAudio_setting': 0,
                'MMAudio_prompt': '',
                'MMAudio_neg_prompt': '',

                # Advanced parameters
                'RIFLEx_setting': 0,
                'NAG_scale': 0.0,
                'NAG_tau': 1.0,
                'NAG_alpha': 0.0,
                'slg_switch': 0,
                'slg_layers': '',
                'slg_start_perc': 0.0,
                'slg_end_perc': 100.0,
                'apg_switch': 0,
                'cfg_star_switch': 0,
                'cfg_zero_step': 0,
                'prompt_enhancer': 0,
                'min_frames_if_references': 9,
                'override_profile': -1,

                # New v9.1 required parameters
                'alt_guidance_scale': 0.0,
                'masking_strength': 1.0,
                'control_net_weight_alt': 1.0,
                'motion_amplitude': 1.0,
                'custom_guide': None,
                'pace': 0.5,
                'exaggeration': 0.5,
                'temperature': 1.0,
                'output_filename': '',

                # Hires config for two-pass generation (qwen models)
                'hires_config': resolved_params.get('hires_config', None),
                'system_prompt': resolved_params.get('system_prompt', ''),

                # Mode and filename
                'mode': 'generate',
                'model_filename': '',
            }

            # Override with ALL parameters from JSON (this preserves your exact JSON values)
            for param_key, param_value in resolved_params.items():
                if param_key not in ['task', 'send_cmd', 'state', 'model_type']:  # Don't override core system params
                    wgp_params[param_key] = param_value
                    if param_key == 'guidance2_scale':
                        generation_logger.info(f"[PASSTHROUGH_DEBUG] Setting {param_key} = {param_value} from JSON")

            # Debug: Check final guidance2_scale value before WGP call
            generation_logger.info(f"[PASSTHROUGH_DEBUG] Final wgp_params guidance2_scale = {wgp_params.get('guidance2_scale', 'NOT_SET')}")

        else:
            # Build parameter dictionary from resolved parameters
            # Supply defaults for required WGP args that may be unused depending on phases/model
            guidance3_scale_value = resolved_params.get(
                "guidance3_scale",
                resolved_params.get("guidance2_scale", actual_guidance),
            )
            switch_threshold2_value = resolved_params.get("switch_threshold2", 0)
            guidance_phases_value = resolved_params.get("guidance_phases", 1)
            model_switch_phase_value = resolved_params.get("model_switch_phase", 1)
            image_refs_relative_size_value = resolved_params.get("image_refs_relative_size", 50)
            override_profile_value = resolved_params.get("override_profile", -1)

            wgp_params = {
                # Core parameters (fixed, not overridable)
                'task': task,
                'send_cmd': send_cmd,
                'state': self.state,
                'model_type': self.current_model,
                'prompt': resolved_params.get("prompt", prompt),
                'negative_prompt': resolved_params.get("negative_prompt", ""),
                'resolution': resolved_params.get("resolution", "1280x720"),
                'video_length': actual_video_length,
                'batch_size': actual_batch_size,
                'seed': resolved_params.get("seed", 42),
                'force_fps': "auto",
                'image_mode': image_mode,

                # VACE control parameters (only pass supported fields upstream)
                'video_guide': video_guide,
                'video_mask': video_mask,
                'video_guide2': None,
                'video_mask2': None,
                'video_prompt_type': video_prompt_type,
                'control_net_weight': control_net_weight,
                'control_net_weight2': control_net_weight2,
                'denoising_strength': 1.0,

                # LoRA parameters in normal mode
                'activated_loras': activated_loras,
                'loras_multipliers': loras_multipliers_str,

                # Overridable parameters from resolved configuration
                'num_inference_steps': resolved_params.get("num_inference_steps", 25),
                'guidance_scale': actual_guidance,
                'guidance2_scale': resolved_params.get("guidance2_scale", actual_guidance),
                'guidance3_scale': guidance3_scale_value,
                'switch_threshold': resolved_params.get("switch_threshold", 500),
                'switch_threshold2': switch_threshold2_value,
                'guidance_phases': guidance_phases_value,
                'model_switch_phase': model_switch_phase_value,
                'embedded_guidance_scale': final_embedded_guidance if is_flux else 0.0,
                'flow_shift': resolved_params.get("flow_shift", 7.0),
                'sample_solver': resolved_params.get("sample_solver", "euler"),

                # New v9.1 required parameters
                'alt_guidance_scale': resolved_params.get("alt_guidance_scale", 0.0),
                'masking_strength': resolved_params.get("masking_strength", 1.0),
                'control_net_weight_alt': resolved_params.get("control_net_weight_alt", 1.0),
                'motion_amplitude': resolved_params.get("motion_amplitude", 1.0),
                'custom_guide': resolved_params.get("custom_guide", None),
                'pace': resolved_params.get("pace", 0.5),
                'exaggeration': resolved_params.get("exaggeration", 0.5),
                'temperature': resolved_params.get("temperature", 1.0),
                'output_filename': resolved_params.get("output_filename", ""),

                # Hires config for two-pass generation (qwen models)
                'hires_config': resolved_params.get("hires_config", None),
                'system_prompt': resolved_params.get("system_prompt", ""),
            }

            # Standard defaults for other parameters - extend the dictionary
            wgp_params.update({
            'audio_guidance_scale': 1.0,
            'repeat_generation': 1,
            'multi_prompts_gen_type': 0,
            'multi_images_gen_type': 0,
            'skip_steps_cache_type': "",
            'skip_steps_multiplier': 1.0,
            'skip_steps_start_step_perc': 0.0,
            
            # Image parameters
            'image_prompt_type': resolved_params.get("image_prompt_type", "disabled"),
            'image_start': resolved_params.get("image_start"),
            'image_end': resolved_params.get("image_end"),
            'image_refs': resolved_params.get("image_refs"),
            'frames_positions': resolved_params.get("frames_positions", ""),
            'image_guide': resolved_params.get("image_guide"),
            'image_mask': resolved_params.get("image_mask"),
            
            # Video parameters
            'model_mode': 0,
            'video_source': None,
            'keep_frames_video_source': "",
            'keep_frames_video_guide': "",
            'video_guide_outpainting': "0 0 0 0",
            'mask_expand': 0,
            
            # Audio parameters
            'audio_guide': None,
            'audio_guide2': None,
            'audio_source': None,
            'audio_prompt_type': "",
            'speakers_locations': "",
            
            # Sliding window
            'sliding_window_size': 129,
            'sliding_window_overlap': 0,
            'sliding_window_color_correction_strength': 0.0,
            'sliding_window_overlap_noise': 0.1,
            'sliding_window_discard_last_frames': 0,
            'latent_noise_mask_strength': 0.0,  # 0.0 = disabled, 1.0 = full latent noise masking
            'vid2vid_init_video': None,  # Path to video for vid2vid initialization
            'vid2vid_init_strength': 0.7,  # 0.0 = keep original, 1.0 = random noise (default)
            'image_refs_relative_size': image_refs_relative_size_value,

            # Post-processing
            'remove_background_images_ref': 0,
            'temporal_upsampling': "",
            'spatial_upsampling': "",
            'film_grain_intensity': 0.0,
            'film_grain_saturation': 0.0,
            'MMAudio_setting': 0,
            'MMAudio_prompt': "",
            'MMAudio_neg_prompt': "",
            
            # Advanced parameters
            'RIFLEx_setting': 0,
            'NAG_scale': 0.0,
            'NAG_tau': 1.0,
            'NAG_alpha': 0.0,
            'slg_switch': 0,
            'slg_layers': "",
            'slg_start_perc': 0.0,
            'slg_end_perc': 100.0,
            'apg_switch': 0,
            'cfg_star_switch': 0,
            'cfg_zero_step': 0,
            'prompt_enhancer': 0,
            'min_frames_if_references': 9,
            'override_profile': override_profile_value,
            
            # Mode and filename
            'mode': "generate",
            'model_filename': "",
        })
        


        # Override ANY parameter provided in kwargs
        # This allows complete customization of generation parameters
        for key, value in kwargs.items():
            if key in wgp_params:
                generation_logger.debug(f"Overriding parameter from kwargs: {key}={wgp_params[key]} -> {value}")
                wgp_params[key] = value
            else:
                generation_logger.debug(f"Adding new parameter from kwargs: {key}={value}")
                wgp_params[key] = value

        # Generate content type description
        content_type = "images" if (is_flux or is_qwen) else "video"
        model_type_desc = (
            "Flux" if is_flux else ("Qwen" if is_qwen else ("VACE" if is_vace else "T2V"))
        )
        count_desc = f"{video_length} {'images' if is_flux else 'frames'}"
        
        generation_logger.essential(f"Generating {model_type_desc} {content_type}: {resolution}, {count_desc}")
        if is_vace:
            encodings = [c for c in video_prompt_type if c in "VPDSLCMUA"]
            generation_logger.debug(f"VACE encodings: {encodings}")
        if activated_loras:
            generation_logger.debug(f"LoRAs: {activated_loras}")

        # Initialize capture variables at outer scope for exception handling
        captured_stdout = None
        captured_stderr = None
        captured_logs = []
        
        try:
            generation_logger.debug("Calling WGP generate_video with VACE module support")
            
            # Log critical parameters being passed to WGP
            if is_vace:
                generation_logger.debug(f"VACE parameters to WGP - model: {self.current_model}, guide: {video_guide}, type: {video_prompt_type}")
            else:
                generation_logger.debug(f"Standard parameters to WGP - model: {self.current_model}")
            
            # [CausVidDebugTrace] Log parameters being passed to generate_video
            generation_logger.info(f"[CausVidDebugTrace] WanOrchestrator.generate calling _generate_video with:")
            generation_logger.info(f"[CausVidDebugTrace]   model_type: {self.current_model}")
            generation_logger.info(f"[CausVidDebugTrace]   num_inference_steps: {num_inference_steps}")
            generation_logger.info(f"[CausVidDebugTrace]   guidance_scale: {actual_guidance}")
            if is_passthrough_mode:
                generation_logger.info(f"[CausVidDebugTrace]   guidance2_scale: {wgp_params.get('guidance2_scale', 'NOT_SET')} (passthrough)")
            else:
                generation_logger.info(f"[CausVidDebugTrace]   guidance2_scale: {wgp_params.get('guidance2_scale', 'NOT_SET')} (normal)")
            generation_logger.info(f"[CausVidDebugTrace]   activated_loras: {activated_loras}")
            generation_logger.info(f"[CausVidDebugTrace]   loras_multipliers_str: {loras_multipliers_str}")
            
            # ARCHITECTURAL FIX: Pre-populate WGP UI state for LoRA compatibility
            # WGP was designed for UI usage where state["loras"] gets populated through UI
            # In headless mode, we need to pre-populate this state to ensure LoRA loading works
            original_loras = self.state.get("loras", [])
            if activated_loras and len(activated_loras) > 0:
                generation_logger.info(f"[CausVidDebugTrace] WanOrchestrator: Pre-populating WGP state with {len(activated_loras)} LoRAs")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] ═══════════════════════════════════════════════════════")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] COMPLETE LoRA APPLICATION BREAKDOWN:")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] ═══════════════════════════════════════════════════════")

                # Parse multipliers string to show per-LoRA breakdown
                multiplier_list = loras_multipliers_str.split() if loras_multipliers_str else []

                for idx, lora_path in enumerate(activated_loras):
                    # Extract filename from path (handles both Unix and Windows paths)
                    lora_filename = str(lora_path).split('/')[-1].split('\\')[-1]
                    mult_str = multiplier_list[idx] if idx < len(multiplier_list) else "1.0"
                    phases = mult_str.split(";")

                    generation_logger.info(f"[LORA_APPLICATION_TRACE]")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE] LoRA #{idx+1}: {lora_filename}")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE]   Full path: {lora_path}")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE]   Multiplier string: {mult_str}")
                    generation_logger.info(f"[LORA_APPLICATION_TRACE]   Phase breakdown:")

                    if len(phases) == 1:
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - All phases: {phases[0]} (constant strength)")
                    else:
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - Phase 1 (steps 0-1): {phases[0] if len(phases) > 0 else '1.0'}")
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - Phase 2 (steps 2-3): {phases[1] if len(phases) > 1 else '1.0'}")
                        generation_logger.info(f"[LORA_APPLICATION_TRACE]     - Phase 3 (steps 4-5): {phases[2] if len(phases) > 2 else '1.0'}")

                generation_logger.info(f"[LORA_APPLICATION_TRACE]")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] SUMMARY:")
                generation_logger.info(f"[LORA_APPLICATION_TRACE]   Total LoRAs: {len(activated_loras)}")
                generation_logger.info(f"[LORA_APPLICATION_TRACE]   Model config LoRAs: {len(activated_loras) - 1 if lora_names and len(lora_names) > 0 else len(activated_loras)}")
                generation_logger.info(f"[LORA_APPLICATION_TRACE]   amount_of_motion LoRA: {'Yes' if lora_names and len(lora_names) > 0 else 'No'}")
                generation_logger.info(f"[LORA_APPLICATION_TRACE] ═══════════════════════════════════════════════════════")

                self.state["loras"] = activated_loras.copy()  # Populate UI state for WGP compatibility
                generation_logger.debug(f"[CausVidDebugTrace] WanOrchestrator: state['loras'] = {self.state['loras']}")
            
            try:
                # Call the VACE-fixed generate_video with the unified parameter dictionary
                # IMPORTANT: Do not pass unsupported keys upstream
                # Pre-initialize WGP process status to avoid None in early callback
                try:
                    if isinstance(self.state.get("gen"), dict):
                        self.state["gen"]["process_status"] = "process:main"
                except (KeyError, TypeError):
                    pass
                # Load I2V input images (image_start/image_end) if they are paths
                # This is required for I2V models which expect PIL images, not paths
                # Extract target resolution for resizing images to match structure video
                target_width, target_height = None, None
                if wgp_params.get('resolution'):
                    try:
                        w_str, h_str = wgp_params['resolution'].split('x')
                        target_width, target_height = int(w_str), int(h_str)
                        generation_logger.info(f"[PREFLIGHT] Target resolution for image resizing: {target_width}x{target_height}")
                    except (ValueError, TypeError, AttributeError) as e_res:
                        generation_logger.warning(f"[PREFLIGHT] Could not parse resolution '{wgp_params.get('resolution')}': {e_res}")

                for img_param in ['image_start', 'image_end']:
                    val = wgp_params.get(img_param)
                    if val:
                        if isinstance(val, str):
                            generation_logger.info(f"[PREFLIGHT] Loading {img_param} from path: {val}")
                            img = self._load_image(val, mask=False)
                            # Resize to exact target resolution to match structure video
                            if img and target_width and target_height:
                                from PIL import Image
                                if img.size != (target_width, target_height):
                                    generation_logger.info(f"[PREFLIGHT] Resizing {img_param} from {img.size[0]}x{img.size[1]} to {target_width}x{target_height}")
                                    img = img.resize((target_width, target_height), Image.LANCZOS)
                            wgp_params[img_param] = img
                        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
                            generation_logger.info(f"[PREFLIGHT] Loading {img_param} from list of paths")
                            loaded_imgs = []
                            for p in val:
                                img = self._load_image(p, mask=False)
                                # Resize to exact target resolution to match structure video
                                if img and target_width and target_height:
                                    from PIL import Image
                                    if img.size != (target_width, target_height):
                                        generation_logger.info(f"[PREFLIGHT] Resizing {img_param} image from {img.size[0]}x{img.size[1]} to {target_width}x{target_height}")
                                        img = img.resize((target_width, target_height), Image.LANCZOS)
                                loaded_imgs.append(img)
                            wgp_params[img_param] = loaded_imgs

                # For image-based models, load PIL images instead of passing paths
                if is_qwen or image_mode == 1:
                    if wgp_params.get('image_guide') and isinstance(wgp_params['image_guide'], str):
                        wgp_params['image_guide'] = self._load_image(wgp_params['image_guide'], mask=False)
                    if wgp_params.get('image_mask') and isinstance(wgp_params['image_mask'], str):
                        wgp_params['image_mask'] = self._load_image(wgp_params['image_mask'], mask=True)

                    # Ensure proper parameter coordination for Qwen models
                    if is_qwen:
                        # Ensure image_mask is truly None if not provided, not empty string or other falsy value
                        if not wgp_params.get('image_mask'):
                            wgp_params['image_mask'] = None
                            generation_logger.debug("[PREFLIGHT] Ensured image_mask=None for Qwen regular generation")
                        else:
                            # If there IS a mask, set model_mode=1 for inpainting
                            wgp_params['model_mode'] = 1
                            generation_logger.info("[PREFLIGHT] Set model_mode=1 for Qwen inpainting (image_mask present)")
                    # Preflight logs for image models
                    try:
                        ig = wgp_params.get('image_guide')
                        if ig is not None:
                            from PIL import Image as _PILImage  # type: ignore
                            if isinstance(ig, _PILImage.Image):
                                generation_logger.info("[PREFLIGHT] image_guide resolved to PIL.Image with size %sx%s and mode %s" % (ig.size[0], ig.size[1], ig.mode))
                            else:
                                generation_logger.warning(f"[PREFLIGHT] image_guide is not a PIL image (type={type(ig)})")
                        else:
                            generation_logger.warning("[PREFLIGHT] image_guide is None for image model")
                    except (AttributeError, TypeError) as _e:
                        generation_logger.warning(f"[PREFLIGHT] Could not inspect image_guide: {_e}")
                # Sanitize image_refs: WGP expects None when there are no refs
                try:
                    if isinstance(wgp_params.get('image_refs'), list) and len(wgp_params['image_refs']) == 0:
                        wgp_params['image_refs'] = None
                except (TypeError, KeyError):
                    pass

                # Inject cached Uni3C controlnet if use_uni3c is enabled
                # This avoids the ~2 minute disk load time for the 1.9GB checkpoint on each generation
                if wgp_params.get('use_uni3c') and not wgp_params.get('uni3c_controlnet'):
                    cached_controlnet = self._get_or_load_uni3c_controlnet()
                    if cached_controlnet is not None:
                        wgp_params['uni3c_controlnet'] = cached_controlnet

                # Filter out unsupported parameters to match upstream signature exactly
                try:
                    import inspect as _inspect
                    import wgp as _wgp  # use the upstream function, not our wrapper
                    _sig = _inspect.signature(_wgp.generate_video)
                    _allowed = set(_sig.parameters.keys())
                    _filtered_params = {k: v for k, v in wgp_params.items() if k in _allowed}
                    _dropped = sorted(set(wgp_params.keys()) - _allowed)
                    if _dropped:
                        generation_logger.debug(f"[PARAM_SANITIZE] Dropping unsupported params: {_dropped}")
                except (ValueError, TypeError, RuntimeError) as _e:
                    # On any error, fall back to original params
                    _filtered_params = wgp_params

                # COMPREHENSIVE LOGGING: Show all final parameters being sent to WGP
                generation_logger.info("=" * 80)
                generation_logger.info("FINAL PARAMETERS BEING SENT TO WGP generate_video():")
                generation_logger.info("=" * 80)
                for key, value in sorted(_filtered_params.items()):
                    # Skip or truncate extremely verbose parameters
                    if key == 'state':
                        # State dict is huge - just show key info
                        state_summary = {
                            'model_type': value.get('model_type'),
                            'gen_file_count': len(value.get('gen', {}).get('file_list', [])),
                            'loras_count': len(value.get('loras', []))
                        }
                        generation_logger.info(f"[FINAL_PARAMS] {key}: {state_summary} (truncated)")
                    elif key in ['guidance_scale', 'guidance2_scale', 'guidance3_scale', 'num_inference_steps', 'switch_threshold', 'switch_threshold2']:
                        generation_logger.info(f"[FINAL_PARAMS] {key}: {value} ⭐")  # Star important guidance params
                    else:
                        generation_logger.info(f"[FINAL_PARAMS] {key}: {value}")
                generation_logger.info("=" * 80)

                # Comprehensive capture of all WGP output: stdout, stderr, and Python logging
                # NOTE: This monkeypatches sys.stdout/sys.stderr for the duration of a single generation.
                # That is process-global and can capture output from other threads. In this repo, generation
                # is effectively single-task-at-a-time per worker process, so this is acceptable.
                import logging as py_logging
                from collections import deque

                # Keep tails only (avoid unbounded memory growth)
                _CAPTURE_STDOUT_CHARS = 20_000
                _CAPTURE_STDERR_CHARS = 20_000
                _CAPTURE_PYLOG_RECORDS = 1000  # keep last N log records (all levels)

                class TailBuffer:
                    def __init__(self, max_chars: int):
                        self.max_chars = max_chars
                        self._buf = ""

                    def write(self, text: str):
                        if not text:
                            return
                        try:
                            self._buf += str(text)
                            if len(self._buf) > self.max_chars:
                                self._buf = self._buf[-self.max_chars :]
                        except (TypeError, ValueError, MemoryError):
                            # Never let logging capture break generation
                            pass

                    def getvalue(self) -> str:
                        return self._buf

                captured_stdout = TailBuffer(_CAPTURE_STDOUT_CHARS)
                captured_stderr = TailBuffer(_CAPTURE_STDERR_CHARS)
                captured_logs = deque(maxlen=_CAPTURE_PYLOG_RECORDS)

                # TeeWriter: capture while still printing to console.
                # Must proxy common file-like attributes (encoding/isatty/fileno/etc) to avoid breaking libs.
                class TeeWriter:
                    def __init__(self, original, capture):
                        self._original = original
                        self._capture = capture

                    def write(self, text):
                        try:
                            self._original.write(text)
                        except (OSError, ValueError):
                            pass
                        try:
                            self._capture.write(text)
                        except (OSError, ValueError):
                            pass

                    def writelines(self, lines):
                        for line in lines:
                            self.write(line)

                    def flush(self):
                        try:
                            self._original.flush()
                        except (OSError, ValueError):
                            pass

                    def isatty(self):
                        try:
                            return self._original.isatty()
                        except (OSError, ValueError):
                            return False

                    def fileno(self):
                        return self._original.fileno()

                    @property
                    def encoding(self):
                        return getattr(self._original, "encoding", None)

                    def __getattr__(self, name):
                        # Proxy everything else to the underlying stream
                        return getattr(self._original, name)

                # Custom logging handler to capture Python logging (all levels)
                class CaptureHandler(py_logging.Handler):
                    def __init__(self, log_deque):
                        super().__init__(level=py_logging.DEBUG)
                        self._log_deque = log_deque
                        self._dedupe = deque(maxlen=200)  # avoid spamming duplicates

                    def emit(self, record):
                        try:
                            msg = self.format(record)
                            key = (record.levelname, record.name, msg)
                            if key in self._dedupe:
                                return
                            self._dedupe.append(key)
                            self._log_deque.append({
                                "level": record.levelname,
                                "name": record.name,
                                "message": msg,
                            })
                        except (ValueError, TypeError, RuntimeError):
                            pass

                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                # Add capture handler to root logger and any libraries that don't propagate.
                capture_handler = CaptureHandler(captured_logs)
                capture_handler.setFormatter(py_logging.Formatter("%(levelname)s:%(name)s: %(message)s"))

                root_logger = py_logging.getLogger()
                candidate_loggers = [
                    py_logging.getLogger("diffusers"),
                    py_logging.getLogger("transformers"),
                    py_logging.getLogger("torch"),
                    py_logging.getLogger("PIL"),
                ]

                loggers_to_capture = [root_logger]
                for lg in candidate_loggers:
                    # Only attach to non-propagating loggers to avoid duplicates (root already captures propagated logs)
                    if getattr(lg, "propagate", True) is False:
                        loggers_to_capture.append(lg)
                
                # Store original levels and add handler
                original_levels = {}
                for logger in loggers_to_capture:
                    original_levels[logger.name] = logger.level
                    logger.addHandler(capture_handler)
                    # Ensure we capture all logs even if logger is set to a higher level
                    if logger.level > py_logging.DEBUG:
                        logger.setLevel(py_logging.DEBUG)
                
                try:
                    sys.stdout = TeeWriter(original_stdout, captured_stdout)
                    sys.stderr = TeeWriter(original_stderr, captured_stderr)
                    
                    self._generate_video(**_filtered_params)
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    
                    # Remove capture handler and restore levels
                    for logger in loggers_to_capture:
                        logger.removeHandler(capture_handler)
                        if logger.name in original_levels:
                            logger.setLevel(original_levels[logger.name])

                # Verify directory after generation (wgp may have changed it during file operations)
                _verify_wgp_directory(generation_logger, "after wgp.generate_video()")

            finally:
                # ARCHITECTURAL FIX: Restore original UI state after generation
                if activated_loras and len(activated_loras) > 0:
                    self.state["loras"] = original_loras  # Restore original state
                    generation_logger.debug(f"[CausVidDebugTrace] WanOrchestrator: Restored original state['loras'] = {self.state['loras']}")

            # WGP doesn't return the path, but stores it in state["gen"]["file_list"]
            output_path = None
            try:
                file_list = self.state["gen"]["file_list"]
                if file_list:
                    output_path = file_list[-1]  # Get the most recently generated file
                    generation_logger.success(f"{model_type_desc} generation completed")
                    generation_logger.essential(f"Output saved to: {output_path}")
                else:
                    generation_logger.warning(f"{model_type_desc} generation completed but no output path found in file_list")
                    # Log captured output to help debug silent failures
                    stdout_content = captured_stdout.getvalue() if captured_stdout is not None else ""
                    stderr_content = captured_stderr.getvalue() if captured_stderr is not None else ""

                    # Log captured Python logs from diffusers/transformers/torch
                    if captured_logs:
                        all_logs = list(captured_logs)
                        # Show errors/warnings first (most important)
                        error_logs = [log for log in all_logs if log['level'] in ['ERROR', 'CRITICAL', 'WARNING']]
                        if error_logs:
                            log_summary = '\n'.join([f"  [{log['level']}] {log['name']}: {log['message'][:200]}" for log in error_logs[-30:]])
                            generation_logger.error(f"[WGP_PYLOG] Python errors/warnings ({len(error_logs)} total):\n{log_summary}")

                        # Show all recent logs for context
                        recent_logs = all_logs[-50:]  # Last 50 logs of any level
                        log_summary = '\n'.join([f"  [{log['level']}] {log['name']}: {log['message'][:150]}" for log in recent_logs])
                        generation_logger.error(f"[WGP_PYLOG] Recent Python logs ({len(all_logs)} total):\n{log_summary}")

                    if stderr_content:
                        # Log last 2000 chars of stderr (most likely to contain the error)
                        stderr_tail = stderr_content[-LOG_TAIL_MAX_CHARS:] if len(stderr_content) > LOG_TAIL_MAX_CHARS else stderr_content
                        generation_logger.error(f"[WGP_STDERR] Generation produced no output. Captured stderr:\n{stderr_tail}")
                    if stdout_content:
                        # Check for error patterns in stdout
                        stdout_lower = stdout_content.lower()
                        if any(err in stdout_lower for err in ['error', 'exception', 'traceback', 'failed', 'cuda', 'oom', 'out of memory']):
                            stdout_tail = stdout_content[-LOG_TAIL_MAX_CHARS:] if len(stdout_content) > LOG_TAIL_MAX_CHARS else stdout_content
                            generation_logger.error(f"[WGP_STDOUT] Error patterns found in stdout:\n{stdout_tail}")

                    # CRITICAL: Extract actual error from WGP output and raise it.
                    # This allows the retry system to properly classify errors like OOM
                    # instead of just seeing "No output generated".
                    actual_error = _extract_wgp_error(stdout_content, stderr_content)
                    if actual_error:
                        generation_logger.error(f"[WGP_ERROR] Extracted error from WGP output: {actual_error}")
                        raise RuntimeError(f"WGP generation failed: {actual_error}")
                    else:
                        # No specific error found - raise generic error
                        # (will be classified as generation_no_output with 2 retries)
                        raise RuntimeError("No output generated")
            except RuntimeError:
                # Re-raise RuntimeError (our intentional errors) to propagate to caller
                raise
            except (KeyError, TypeError, IndexError) as e:
                # Only catch unexpected errors accessing state
                generation_logger.warning(f"Could not retrieve output path from state: {e}")

            # Memory monitoring
            try:
                import torch
                import psutil

                # RAM usage
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024**3)
                ram_total_gb = ram.total / (1024**3)
                ram_percent = ram.percent

                # VRAM usage
                if torch.cuda.is_available():
                    vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    vram_percent = (vram_reserved / vram_total) * 100

                    generation_logger.essential(f"💾 RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.0f}%) | VRAM: {vram_reserved:.1f}GB / {vram_total:.1f}GB ({vram_percent:.0f}%) [Allocated: {vram_allocated:.1f}GB]")
                else:
                    generation_logger.essential(f"💾 RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.0f}%)")
            except (RuntimeError, OSError, ImportError) as e:
                generation_logger.debug(f"Could not retrieve memory stats: {e}")

            return output_path
            
        except (RuntimeError, ValueError, OSError, TypeError) as e:
            generation_logger.error(f"Generation failed: {e}")
            # Try to log any captured output that might help debug
            try:
                # Log captured Python logs (from diffusers/transformers/torch)
                if captured_logs:
                    all_logs = list(captured_logs)
                    # Show errors/warnings first
                    error_logs = [log for log in all_logs if log['level'] in ['ERROR', 'CRITICAL', 'WARNING']]
                    if error_logs:
                        log_summary = '\n'.join([f"  [{log['level']}] {log['name']}: {log['message'][:200]}" for log in error_logs[-30:]])
                        generation_logger.error(f"[WGP_PYLOG] Python errors/warnings ({len(error_logs)} total):\n{log_summary}")
                    
                    # Show all recent logs for context
                    recent_logs = all_logs[-50:]
                    log_summary = '\n'.join([f"  [{log['level']}] {log['name']}: {log['message'][:150]}" for log in recent_logs])
                    generation_logger.error(f"[WGP_PYLOG] Recent Python logs ({len(all_logs)} total):\n{log_summary}")
                
                if captured_stderr is not None:
                    stderr_content = captured_stderr.getvalue()
                    if stderr_content:
                        stderr_tail = stderr_content[-LOG_TAIL_MAX_CHARS:] if len(stderr_content) > LOG_TAIL_MAX_CHARS else stderr_content
                        generation_logger.error(f"[WGP_STDERR] Exception occurred. Captured stderr:\n{stderr_tail}")
                if captured_stdout is not None:
                    stdout_content = captured_stdout.getvalue()
                    if stdout_content:
                        stdout_lower = stdout_content.lower()
                        if any(err in stdout_lower for err in ['error', 'exception', 'traceback', 'failed', 'cuda', 'oom', 'out of memory']):
                            stdout_tail = stdout_content[-LOG_TAIL_MAX_CHARS:] if len(stdout_content) > LOG_TAIL_MAX_CHARS else stdout_content
                            generation_logger.error(f"[WGP_STDOUT] Error context from stdout:\n{stdout_tail}")
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass  # Don't let logging errors mask the original exception
            raise

    # Convenience methods for specific generation types
    
    def generate_t2v(self, prompt: str, model_type: str = None, **kwargs) -> str:
        """Generate text-to-video content."""
        if not self._is_t2v():
            generation_logger.warning(f"Current model {self.current_model} may not be optimized for T2V")
        return self.generate(prompt=prompt, model_type=model_type, **kwargs)
    
    def generate_with_config(self, config: 'TaskConfig') -> str:
        """
        Generate content using a typed TaskConfig object.
        
        This is the new typed API that provides cleaner parameter handling:
        1. All parameters are already parsed and validated in TaskConfig
        2. LoRAs are already resolved (downloaded if needed)
        3. Phase config is already applied
        
        Args:
            config: TaskConfig object with all generation parameters
            
        Returns:
            Path to generated output file(s)
            
        Example:
            from source.core.params import TaskConfig
            
            config = TaskConfig.from_db_task(task.parameters, task_id=task.id)
            result = orchestrator.generate_with_config(config)
        """
        # Import here to avoid circular import
        from source.core.params import TaskConfig
        
        if not isinstance(config, TaskConfig):
            raise TypeError(f"Expected TaskConfig, got {type(config)}")
        
        if not self.current_model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Log what we're generating
        generation_logger.info(f"[GENERATE_CONFIG] Task {config.task_id}: Starting generation with typed config")
        if not config.lora.is_empty():
            generation_logger.info(f"[GENERATE_CONFIG] Task {config.task_id}: LoRAs: {config.lora.filenames}")
        if not config.phase.is_empty():
            generation_logger.info(f"[GENERATE_CONFIG] Task {config.task_id}: Phases: {config.phase.num_phases}")
        
        # Convert to WGP format
        wgp_params = config.to_wgp_format()
        
        # Ensure prompt is set
        prompt = wgp_params.pop('prompt', config.generation.prompt)
        if not prompt:
            raise ValueError("No prompt provided in TaskConfig")
        
        # Determine model type
        model_type = wgp_params.pop('model', None) or config.model or self.current_model
        
        # Call the standard generate method with unpacked params
        return self.generate(
            prompt=prompt,
            model_type=model_type,
            **wgp_params
        )
    
    def generate_vace(self, 
                     prompt: str, 
                     video_guide: str,
                     model_type: str = None,
                     video_mask: Optional[str] = None,
                     video_prompt_type: str = "VP",
                     control_net_weight: float = 1.0,
                     control_net_weight2: float = 1.0,
                     **kwargs) -> str:
        """Generate VACE controlled video content.
        
        Args:
            prompt: Text prompt for generation
            video_guide: Path to primary control video (required)
            video_mask: Path to primary mask video (optional)
            video_prompt_type: VACE encoding type (e.g., "VP", "VPD", "VPDA")
            control_net_weight: Strength for first VACE encoding
            control_net_weight2: Strength for second VACE encoding
            **kwargs: Additional parameters
            
        Returns:
            Path to generated video file
        """
        if not self._is_vace():
            generation_logger.warning(f"Current model {self.current_model} may not be a VACE model")
        
        # [CausVidDebugTrace] Log LoRA parameters at VACE level
        generation_logger.info(f"[CausVidDebugTrace] generate_vace received kwargs: {list(kwargs.keys())}")
        if "lora_names" in kwargs:
            generation_logger.info(f"[CausVidDebugTrace] generate_vace lora_names: {kwargs['lora_names']}")
        if "lora_multipliers" in kwargs:
            generation_logger.info(f"[CausVidDebugTrace] generate_vace lora_multipliers: {kwargs['lora_multipliers']}")
        
        return self.generate(
            prompt=prompt,
            model_type=model_type,
            video_guide=video_guide,
            video_mask=video_mask,
            video_prompt_type=video_prompt_type,
            control_net_weight=control_net_weight,
            control_net_weight2=control_net_weight2,
            **kwargs  # Any additional parameters including switch_threshold
        )

    def generate_flux(self, prompt: str, images: int = 4, model_type: str = None, **kwargs) -> str:
        """Generate Flux images.
        
        Args:
            prompt: Text prompt for generation
            images: Number of images to generate (uses video_length parameter)
            **kwargs: Additional parameters
            
        Returns:
            Path to generated image(s)
        """
        if not self._is_flux():
            generation_logger.warning(f"Current model {self.current_model} may not be a Flux model")
        
        return self.generate(
            prompt=prompt,
            model_type=model_type,
            video_length=images,  # For Flux, video_length = number of images
            **kwargs
        )

# Backward compatibility
WanContentOrchestrator = WanOrchestrator
