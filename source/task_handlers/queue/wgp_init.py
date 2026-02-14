"""
WGP/CUDA initialization mixin for HeadlessTaskQueue.

Handles CUDA warmup, diagnostics, and lazy WanOrchestrator initialization.
Separated from the main queue coordinator to isolate the complex
startup/initialization logic.
"""

import os
import sys
import traceback

from source.core.log import is_debug_enabled


class WgpInitMixin:
    """Mixin providing CUDA warmup and WanOrchestrator lazy initialization."""

    def _ensure_orchestrator(self):
        """
        Lazily initialize orchestrator on first use to avoid CUDA init during queue setup.

        The orchestrator imports wgp, which triggers deep module imports (models/wan/modules/t5.py)
        that call torch.cuda.current_device() at class definition time. We defer this until
        the first task is actually processed, when CUDA is guaranteed to be ready.
        """
        if self.orchestrator is not None:
            return  # Already initialized

        if self._orchestrator_init_attempted:
            raise RuntimeError("Orchestrator initialization failed previously")

        self._orchestrator_init_attempted = True

        try:
            if is_debug_enabled():
                self.logger.info("[LAZY_INIT] Initializing WanOrchestrator (first use)...")
                self.logger.info("[LAZY_INIT] Warming up CUDA before importing wgp...")

            # Warm up CUDA before importing wgp (upstream T5EncoderModel has torch.cuda.current_device()
            # as a default arg, which is evaluated at module import time)
            import torch

            # Detailed CUDA diagnostics
            if is_debug_enabled():
                self.logger.info("[CUDA_DEBUG] ========== CUDA DIAGNOSTICS ==========")
                self.logger.info(f"[CUDA_DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                if is_debug_enabled():
                    _log_cuda_available_diagnostics(self.logger, torch)
            else:
                if is_debug_enabled():
                    _log_cuda_unavailable_diagnostics(self.logger, torch)

            if is_debug_enabled():
                self.logger.info("[CUDA_DEBUG] ===========================================")

            self._import_and_create_orchestrator()

            if is_debug_enabled():
                self.logger.info("[LAZY_INIT] WanOrchestrator initialized successfully")

            # Now that orchestrator exists, complete wgp integration
            self._init_wgp_integration()

        except Exception as e:
            # Always log orchestrator init failures - this is critical for debugging!
            self.logger.error(f"[LAZY_INIT] Failed to initialize WanOrchestrator: {e}")
            if is_debug_enabled():
                self.logger.error(f"[LAZY_INIT] Traceback:\n{traceback.format_exc()}")
            raise

    def _import_and_create_orchestrator(self):
        """
        Import WanOrchestrator and create instance, handling sys.argv and cwd protection.

        wgp.py parses sys.argv and uses relative paths, so we must protect both
        before import.
        """
        if is_debug_enabled():
            self.logger.info("[LAZY_INIT] Importing WanOrchestrator (this imports wgp and model modules)...")

        # Protect sys.argv and working directory before importing headless_wgp which imports wgp
        # wgp.py will try to parse sys.argv and will fail if it contains Supabase/database arguments
        # wgp.py also uses relative paths for model loading and needs to run from Wan2GP directory
        # CRITICAL: wgp.py loads models at MODULE-LEVEL (line 2260), so we MUST chdir BEFORE import
        _saved_argv_for_import = sys.argv[:]
        try:
            sys.argv = ["headless_model_management.py"]  # Clean argv for wgp import

            # CRITICAL: Change to Wan2GP directory BEFORE importing/initializing WanOrchestrator
            # wgp.py uses relative paths (defaults/*.json) and expects to run from Wan2GP/
            if is_debug_enabled():
                self.logger.info(f"[LAZY_INIT] Changing to Wan2GP directory: {self.wan_dir}")
                self.logger.info(f"[LAZY_INIT] Current directory before chdir: {os.getcwd()}")

            os.chdir(self.wan_dir)

            actual_cwd = os.getcwd()
            if is_debug_enabled():
                self.logger.info(f"[LAZY_INIT] Changed directory to: {actual_cwd}")

            # Verify the change worked
            if actual_cwd != self.wan_dir:
                raise RuntimeError(
                    f"Directory change failed! Expected {self.wan_dir}, got {actual_cwd}"
                )

            # Verify critical structure exists
            if not os.path.isdir("defaults"):
                raise RuntimeError(
                    f"defaults/ directory not found in {actual_cwd}. "
                    f"Cannot proceed without model definitions!"
                )

            if is_debug_enabled():
                self.logger.info(f"[LAZY_INIT] Now in Wan2GP directory, importing WanOrchestrator...")

            from headless_wgp import WanOrchestrator

            # Set mmgp verbose level for debug logging in any2video.py SVI path etc
            if is_debug_enabled():
                try:
                    from mmgp import offload
                    offload.default_verboseLevel = 2
                    self.logger.info("[LAZY_INIT] Set offload.default_verboseLevel=2 for debug logging")
                except ImportError:
                    pass

            self.orchestrator = WanOrchestrator(self.wan_dir, main_output_dir=self.main_output_dir)
        finally:
            sys.argv = _saved_argv_for_import  # Restore original arguments
            # NOTE: We do NOT restore the working directory - WGP expects to stay in Wan2GP/
            # This ensures model downloads, file operations, etc. use correct paths

    def _init_wgp_integration(self):
        """
        Initialize integration with wgp.py's existing systems.

        This reuses wgp.py's state management, queue handling, and model persistence
        rather than reimplementing it.

        Called after orchestrator is lazily initialized.
        """
        if self.orchestrator is None:
            self.logger.warning("Skipping wgp integration - orchestrator not initialized yet")
            return

        # Core integration: reuse orchestrator's state management
        self.wgp_state = self.orchestrator.state

        self.logger.info("WGP integration initialized")


def _log_cuda_available_diagnostics(logger, torch):
    """Log detailed CUDA diagnostics when CUDA is available."""
    try:
        device_count = torch.cuda.device_count()
        logger.info(f"[CUDA_DEBUG] Device count: {device_count}")

        for i in range(device_count):
            logger.info(f"[CUDA_DEBUG] Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"[CUDA_DEBUG]   - Properties: {torch.cuda.get_device_properties(i)}")

        # Try to get CUDA version info
        try:
            logger.info(f"[CUDA_DEBUG] CUDA version (torch): {torch.version.cuda}")
        except Exception as e:
            logger.debug(f"[CUDA_DEBUG] Could not retrieve CUDA version: {e}")

        # Try to initialize current device
        try:
            current_dev = torch.cuda.current_device()
            logger.info(f"[CUDA_DEBUG] Current device: {current_dev}")

            # Try a simple tensor operation
            test_tensor = torch.tensor([1.0], device='cuda')
            logger.info(f"[CUDA_DEBUG] Successfully created tensor on CUDA: {test_tensor.device}")

        except Exception as e:
            logger.error(f"[CUDA_DEBUG] Failed to initialize current device: {e}")
            raise

    except Exception as e:
        logger.error(f"[CUDA_DEBUG] Error during CUDA diagnostics: {e}\n{traceback.format_exc()}")
        raise


def _log_cuda_unavailable_diagnostics(logger, torch):
    """Log diagnostics when CUDA is not available to help debug the issue."""
    logger.warning("[CUDA_DEBUG] torch.cuda.is_available() returned False")
    logger.warning("[CUDA_DEBUG] Checking why CUDA is not available...")

    # Check if CUDA was built with torch
    logger.info(f"[CUDA_DEBUG] torch.version.cuda: {torch.version.cuda}")
    logger.info(f"[CUDA_DEBUG] torch.backends.cudnn.version(): {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

    # Try to import pynvml for driver info
    try:
        import pynvml
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        logger.info(f"[CUDA_DEBUG] NVIDIA driver version: {driver_version}")
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"[CUDA_DEBUG] NVML device count: {device_count}")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            logger.info(f"[CUDA_DEBUG] NVML Device {i}: {name}")
    except Exception as e:
        logger.warning(f"[CUDA_DEBUG] Could not get NVML info: {e}")
