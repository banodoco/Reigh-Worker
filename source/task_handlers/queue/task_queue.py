#!/usr/bin/env python3
"""
WanGP Headless Task Queue Manager

A persistent service that maintains model state and processes generation tasks
via a queue system. This script keeps models loaded in memory and delegates
actual generation to headless_wgp.py while managing the queue, persistence,
and task scheduling.

Key Features:
- Persistent model state (models stay loaded until switched)
- Task queue with priority support
- Auto model switching and memory management
- Status monitoring and progress tracking
- Uses wgp.py's native queue and state management
- Hot-swappable task processing

Usage:
    # Start the headless service
    python headless_model_management.py --wan-dir /path/to/WanGP --port 8080
    
    # Submit tasks via API or queue files
    curl -X POST http://localhost:8080/generate \
         -H "Content-Type: application/json" \
         -d '{"model": "vace_14B", "prompt": "mystical forest", "video_guide": "input.mp4"}'
"""

import os
import sys
import time
import traceback

# Import debug print function from worker
try:
    from worker import dprint
except ImportError:
    def dprint(msg):
        if os.environ.get('DEBUG'):
            print(msg)
import threading
import queue
import argparse
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from source.core.log import queue_logger
from source.task_handlers.queue.download_ops import switch_model_impl, convert_to_wgp_task_impl
from source.task_handlers.queue.worker_thread import (
    worker_loop,
    process_task_impl,
    execute_generation_impl,
    cleanup_memory_after_task,
)
from source.task_handlers.queue.queue_lifecycle import start_queue, stop_queue, submit_task_impl

from source.core.constants import BYTES_PER_GB as BYTES_PER_GIB

# Add WanGP to path for imports
def setup_wgp_path(wan_dir: str):
    """Setup WanGP path and imports."""
    wan_dir = os.path.abspath(wan_dir)
    if wan_dir not in sys.path:
        sys.path.insert(0, wan_dir)
    return wan_dir

# Task definitions
@dataclass
class GenerationTask:
    """Represents a single generation task."""
    id: str
    model: str
    prompt: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: str = None
    status: str = "pending"  # pending, processing, completed, failed
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass 
class QueueStatus:
    """Current queue status information."""
    pending_tasks: int
    processing_task: Optional[str]
    completed_tasks: int
    failed_tasks: int
    current_model: Optional[str]
    uptime: float
    memory_usage: Dict[str, Any]


class HeadlessTaskQueue:
    """
    Main task queue manager that integrates with wgp.py's existing queue system.
    
    This class leverages wgp.py's built-in task management and state persistence
    while providing a clean API for headless operation.
    """
    
    def __init__(self, wan_dir: str, max_workers: int = 1, debug_mode: bool = False, main_output_dir: Optional[str] = None):
        """
        Initialize the headless task queue.

        Args:
            wan_dir: Path to WanGP directory
            max_workers: Number of concurrent generation workers (recommend 1 for GPU)
            debug_mode: Enable verbose debug logging (should match worker's --debug flag)
            main_output_dir: Optional path for output directory. If not provided, defaults to
                           'outputs' directory next to wan_dir (preserves backwards compatibility)
        """
        self.wan_dir = setup_wgp_path(wan_dir)
        self.max_workers = max_workers
        self.main_output_dir = main_output_dir
        self.running = False
        self.start_time = time.time()
        self.debug_mode = debug_mode  # Now controlled by caller
        
        # Import wgp after path setup (protect sys.argv to prevent argument conflicts)
        _saved_argv = sys.argv[:]
        sys.argv = ["headless_wgp.py"]
        # Headless stubs to avoid optional UI deps (tkinter/matanyone) during import
        try:
            import types
            # Stub tkinter if not available
            if 'tkinter' not in sys.modules:
                sys.modules['tkinter'] = types.ModuleType('tkinter')
            # Stub preprocessing.matanyone.app with minimal interface
            dummy_pkg = types.ModuleType('preprocessing')
            dummy_matanyone = types.ModuleType('preprocessing.matanyone')
            dummy_app = types.ModuleType('preprocessing.matanyone.app')
            def _noop_handler():
                class _Dummy:
                    def __getattr__(self, _):
                        return None
                return _Dummy()
            dummy_app.get_vmc_event_handler = _noop_handler  # type: ignore
            sys.modules['preprocessing'] = dummy_pkg
            sys.modules['preprocessing.matanyone'] = dummy_matanyone
            sys.modules['preprocessing.matanyone.app'] = dummy_app
        except Exception as e:
            logging.getLogger('HeadlessQueue').debug(f"Failed to set up headless UI stubs: {e}")
        # Don't import wgp during initialization to avoid CUDA/argparse conflicts
        # wgp will be imported lazily when needed (e.g., in _apply_sampler_cfg_preset)
        # This allows the queue to initialize even if CUDA isn't ready yet
        self.wgp = None
        
        # Restore sys.argv immediately (no wgp import, so no need for protection)
        try:
            sys.argv = _saved_argv
        except Exception as e:
            logging.getLogger('HeadlessQueue').debug(f"Failed to restore sys.argv after init: {e}")
        
        # Defer orchestrator initialization to avoid CUDA init during queue setup
        # Orchestrator imports wgp, which triggers deep imports that call torch.cuda
        # We'll initialize it lazily when first needed
        self.orchestrator = None
        self._orchestrator_init_attempted = False
        logging.getLogger('HeadlessQueue').info(f"HeadlessTaskQueue created (orchestrator will initialize on first use)")
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.task_history: Dict[str, GenerationTask] = {}
        self.current_task: Optional[GenerationTask] = None
        self.current_model: Optional[str] = None
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.queue_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "model_switches": 0,
            "total_generation_time": 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
        # Initialize wgp state (reuse existing state management)
        self._init_wgp_integration()
        
        self.logger.info(f"HeadlessTaskQueue initialized with WanGP at {wan_dir}")
    
    def _setup_logging(self):
        """Setup structured logging that goes to Supabase via the log interceptor."""
        # Keep Python's basic logging for local file backup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('headless.log')
            ]
        )
        self._file_logger = logging.getLogger('HeadlessQueue')
        
        # Use queue_logger (ComponentLogger) as main logger - this goes to Supabase
        # ComponentLogger has compatible interface: .info(), .error(), .warning(), .debug()
        self.logger = queue_logger
    
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
            if self.debug_mode:
                self.logger.info("[LAZY_INIT] Initializing WanOrchestrator (first use)...")
                self.logger.info("[LAZY_INIT] Warming up CUDA before importing wgp...")

            # Warm up CUDA before importing wgp (upstream T5EncoderModel has torch.cuda.current_device()
            # as a default arg, which is evaluated at module import time)
            import torch

            # Detailed CUDA diagnostics
            if self.debug_mode:
                self.logger.info("[CUDA_DEBUG] ========== CUDA DIAGNOSTICS ==========")
                self.logger.info(f"[CUDA_DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                if self.debug_mode:
                    try:
                        device_count = torch.cuda.device_count()
                        self.logger.info(f"[CUDA_DEBUG] Device count: {device_count}")

                        for i in range(device_count):
                            self.logger.info(f"[CUDA_DEBUG] Device {i}: {torch.cuda.get_device_name(i)}")
                            self.logger.info(f"[CUDA_DEBUG]   - Properties: {torch.cuda.get_device_properties(i)}")

                        # Try to get CUDA version info
                        try:
                            self.logger.info(f"[CUDA_DEBUG] CUDA version (torch): {torch.version.cuda}")
                        except Exception as e:
                            self.logger.debug(f"[CUDA_DEBUG] Could not retrieve CUDA version: {e}")

                        # Try to initialize current device
                        try:
                            current_dev = torch.cuda.current_device()
                            self.logger.info(f"[CUDA_DEBUG] Current device: {current_dev}")

                            # Try a simple tensor operation
                            test_tensor = torch.tensor([1.0], device='cuda')
                            self.logger.info(f"[CUDA_DEBUG] ✅ Successfully created tensor on CUDA: {test_tensor.device}")

                        except Exception as e:
                            self.logger.error(f"[CUDA_DEBUG] ❌ Failed to initialize current device: {e}")
                            raise

                    except Exception as e:
                        self.logger.error(f"[CUDA_DEBUG] ❌ Error during CUDA diagnostics: {e}\n{traceback.format_exc()}")
                        raise

            else:
                if self.debug_mode:
                    self.logger.warning("[CUDA_DEBUG] ⚠️  torch.cuda.is_available() returned False")
                    self.logger.warning("[CUDA_DEBUG] Checking why CUDA is not available...")

                    # Check if CUDA was built with torch
                    self.logger.info(f"[CUDA_DEBUG] torch.version.cuda: {torch.version.cuda}")
                    self.logger.info(f"[CUDA_DEBUG] torch.backends.cudnn.version(): {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

                    # Try to import pynvml for driver info
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        driver_version = pynvml.nvmlSystemGetDriverVersion()
                        self.logger.info(f"[CUDA_DEBUG] NVIDIA driver version: {driver_version}")
                        device_count = pynvml.nvmlDeviceGetCount()
                        self.logger.info(f"[CUDA_DEBUG] NVML device count: {device_count}")
                        for i in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            name = pynvml.nvmlDeviceGetName(handle)
                            self.logger.info(f"[CUDA_DEBUG] NVML Device {i}: {name}")
                    except Exception as e:
                        self.logger.warning(f"[CUDA_DEBUG] Could not get NVML info: {e}")

            if self.debug_mode:
                self.logger.info("[CUDA_DEBUG] ===========================================")

            if self.debug_mode:
                self.logger.info("[LAZY_INIT] Importing WanOrchestrator (this imports wgp and model modules)...")
            # Protect sys.argv and working directory before importing headless_wgp which imports wgp
            # wgp.py will try to parse sys.argv and will fail if it contains Supabase/database arguments
            # wgp.py also uses relative paths for model loading and needs to run from Wan2GP directory
            # CRITICAL: wgp.py loads models at MODULE-LEVEL (line 2260), so we MUST chdir BEFORE import
            _saved_argv_for_import = sys.argv[:]
            _saved_cwd = os.getcwd()
            try:
                sys.argv = ["headless_model_management.py"]  # Clean argv for wgp import

                # CRITICAL: Change to Wan2GP directory BEFORE importing/initializing WanOrchestrator
                # wgp.py uses relative paths (defaults/*.json) and expects to run from Wan2GP/
                if self.debug_mode:
                    self.logger.info(f"[LAZY_INIT] Changing to Wan2GP directory: {self.wan_dir}")
                    self.logger.info(f"[LAZY_INIT] Current directory before chdir: {os.getcwd()}")

                os.chdir(self.wan_dir)

                actual_cwd = os.getcwd()
                if self.debug_mode:
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

                if self.debug_mode:
                    self.logger.info(f"[LAZY_INIT] ✅ Now in Wan2GP directory, importing WanOrchestrator...")

                from headless_wgp import WanOrchestrator
                
                # Set mmgp verbose level for debug logging in any2video.py SVI path etc
                if self.debug_mode:
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

            if self.debug_mode:
                self.logger.info("[LAZY_INIT] ✅ WanOrchestrator initialized successfully")

            # Now that orchestrator exists, complete wgp integration
            self._init_wgp_integration()

        except Exception as e:
            # Always log orchestrator init failures - this is critical for debugging!
            self.logger.error(f"[LAZY_INIT] ❌ Failed to initialize WanOrchestrator: {e}")
            if self.debug_mode:
                self.logger.error(f"[LAZY_INIT] Traceback:\n{traceback.format_exc()}")
            raise
    
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

    def _cleanup_memory_after_task(self, task_id: str):
        """Clean up memory after task completion WITHOUT unloading models."""
        return cleanup_memory_after_task(self, task_id)
    
    def start(self, preload_model: Optional[str] = None):
        """Start the task queue processing service."""
        return start_queue(self, preload_model=preload_model)
    
    def stop(self, timeout: float = 30.0):
        """Stop the task queue processing service."""
        return stop_queue(self, timeout=timeout)
    
    def submit_task(self, task: GenerationTask) -> str:
        """Submit a new generation task to the queue."""
        return submit_task_impl(self, task)
    
    def get_task_status(self, task_id: str) -> Optional[GenerationTask]:
        """Get status of a specific task."""
        return self.task_history.get(task_id)
    
    def wait_for_completion(self, task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """
        Wait for a task to complete and return the result.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary with 'success', 'output_path', and optional 'error' keys
        """
        import time
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            task_status = self.get_task_status(task_id)
            
            if task_status is None:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found in queue"
                }
            
            if task_status.status == "completed":
                return {
                    "success": True,
                    "output_path": task_status.result_path
                }
            elif task_status.status == "failed":
                return {
                    "success": False,
                    "error": task_status.error_message or "Task failed with unknown error"
                }
            
            # Task is still pending or processing, wait a bit
            time.sleep(1.0)
        
        # Timeout reached
        return {
            "success": False,
            "error": f"Task {task_id} did not complete within {timeout} seconds"
        }
    
    def get_queue_status(self) -> QueueStatus:
        """Get current queue status."""
        with self.queue_lock:
            return QueueStatus(
                pending_tasks=self.task_queue.qsize(),
                processing_task=self.current_task.id if self.current_task else None,
                completed_tasks=self.stats["tasks_completed"],
                failed_tasks=self.stats["tasks_failed"],
                current_model=self.current_model,
                uptime=time.time() - self.start_time,
                memory_usage=self._get_memory_usage()
            )
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        return worker_loop(self)
    
    def _process_task(self, task: GenerationTask, worker_name: str):
        """Process a single generation task."""
        return process_task_impl(self, task, worker_name)
    
    def _switch_model(self, model_key: str, worker_name: str) -> bool:
        """Ensure the correct model is loaded using wgp.py's model management."""
        return switch_model_impl(self, model_key, worker_name)
    
    def _execute_generation(self, task: GenerationTask, worker_name: str) -> str:
        """Execute the actual generation using headless_wgp.py."""
        return execute_generation_impl(self, task, worker_name)

    def _model_supports_vace(self, model_key: str) -> bool:
        """
        Check if a model supports VACE features (video guides, masks, etc.).
        """
        # Ensure orchestrator is initialized before checking model support
        self._ensure_orchestrator()
        
        try:
            # Use orchestrator's VACE detection with model key
            if hasattr(self.orchestrator, 'is_model_vace'):
                return self.orchestrator.is_model_vace(model_key)
            elif hasattr(self.orchestrator, '_is_vace'):
                # Fallback: load model and check (less efficient)
                current_model = self.current_model
                if current_model != model_key:
                    # Would need to load model to check - use name-based detection as fallback
                    return "vace" in model_key.lower()
                return self.orchestrator._is_vace()
            else:
                # Ultimate fallback: name-based detection
                return "vace" in model_key.lower()
        except Exception as e:
            self.logger.warning(f"Could not determine VACE support for model '{model_key}': {e}")
            return "vace" in model_key.lower()
    
    def _is_single_image_task(self, task: GenerationTask) -> bool:
        """
        Check if this is a single image task that should be converted from video to PNG.
        """
        # Check if video_length is 1 (single frame) and this looks like an image task
        video_length = task.parameters.get("video_length", 0)
        return video_length == 1
    
    def _convert_single_frame_video_to_png(self, task: GenerationTask, video_path: str, worker_name: str) -> str:
        """
        Convert a single-frame video to PNG format for single image tasks.
        
        This restores the functionality that was in the original single_image.py handler
        where single-frame videos were converted to PNG files.
        """
        try:
            import cv2
            
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                self.logger.error(f"Video file does not exist for PNG conversion: {video_path}")
                return video_path  # Return original path if conversion fails
            
            # Create PNG output path with sanitized filename to prevent upload issues
            original_filename = video_path_obj.stem
            
            # Sanitize the filename for storage compatibility
            try:
                from source.utils import sanitize_filename_for_storage

                sanitized_filename = sanitize_filename_for_storage(original_filename)
                if not sanitized_filename:
                    sanitized_filename = "generated_image"

            except ImportError:
                # Fallback sanitization if import fails
                import re
                sanitized_filename = re.sub(r'[§®©™@·º½¾¿¡~\x00-\x1F\x7F-\x9F<>:"/\\|?*,]', '', original_filename)
                sanitized_filename = re.sub(r'\s+', '_', sanitized_filename.strip())
                if not sanitized_filename:
                    sanitized_filename = "generated_image"
            
            # Create PNG path with sanitized filename
            png_path = video_path_obj.parent / f"{sanitized_filename}.png"
            
            # Log sanitization if filename changed
            if sanitized_filename != original_filename:
                self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Sanitized filename '{original_filename}' -> '{sanitized_filename}'")
            
            self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Converting {video_path_obj.name} to {png_path.name}")
            
            # Extract the first frame using OpenCV
            cap = cv2.VideoCapture(str(video_path_obj))
            try:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Save the frame as PNG
                        success = cv2.imwrite(str(png_path), frame)
                        if success and png_path.exists():
                            self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Successfully saved PNG to {png_path}")
                            
                            # Clean up the original video file
                            try:
                                video_path_obj.unlink()
                                self.logger.info(f"[PNG_CONVERSION] Task {task.id}: Removed original video file")
                            except Exception as e_cleanup:
                                self.logger.warning(f"[PNG_CONVERSION] Task {task.id}: Could not remove original video: {e_cleanup}")
                            
                            return str(png_path)
                        else:
                            self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Failed to save PNG to {png_path}")
                    else:
                        self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Failed to read frame from video")
                else:
                    self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Failed to open video file")
            finally:
                cap.release()
                
        except ImportError:
            self.logger.warning(f"[PNG_CONVERSION] Task {task.id}: OpenCV not available, keeping video format")
        except Exception as e:
            self.logger.error(f"[PNG_CONVERSION] Task {task.id}: Error during conversion: {e}")
        
        # Return original video path if conversion failed
        return video_path
    
    def _monitor_loop(self):
        """Background monitoring and maintenance loop."""
        from source.task_handlers.queue.worker_thread import _monitor_loop as _monitor_loop_impl
        return _monitor_loop_impl(self)
    
    def _convert_to_wgp_task(self, task: GenerationTask) -> Dict[str, Any]:
        """Convert task to WGP parameters using typed TaskConfig."""
        return convert_to_wgp_task_impl(self, task)
    
    def _apply_sampler_cfg_preset(self, model_key: str, sample_solver: str, wgp_params: Dict[str, Any]):
        """Apply sampler-specific CFG and flow_shift settings from model configuration."""
        try:
            # Import WGP to get model definition
            # Protect sys.argv in case wgp hasn't been imported yet
            _saved_argv = sys.argv[:]
            try:
                sys.argv = ["headless_model_management.py"]
                import wgp
            finally:
                sys.argv = _saved_argv

            model_def = wgp.get_model_def(model_key)
            
            # Check if model has sampler-specific presets
            sampler_presets = model_def.get("sampler_cfg_presets", {})
            if sample_solver in sampler_presets:
                preset = sampler_presets[sample_solver]
                
                # Apply preset settings, but allow task parameters to override
                applied_params = {}
                for param, value in preset.items():
                    if param not in wgp_params:  # Only apply if not explicitly set in task
                        wgp_params[param] = value
                        applied_params[param] = value
                        
                self.logger.info(f"Applied sampler '{sample_solver}' CFG preset: {applied_params}")
            else:
                self.logger.debug(f"No CFG preset found for sampler '{sample_solver}' in model '{model_key}'")
                
        except Exception as e:
            self.logger.warning(f"Failed to apply sampler CFG preset: {e}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics. Returns stub data for now."""
        return {
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "system_memory_used": 0,
            "model_memory_usage": 0
        }


def create_sample_task(task_id: str, model: str, prompt: str, **params) -> GenerationTask:
    """Helper to create sample tasks for testing."""
    return GenerationTask(
        id=task_id,
        model=model,
        prompt=prompt,
        parameters=params
    )


def main():
    """Main entry point for the headless service."""
    parser = argparse.ArgumentParser(description="WanGP Headless Task Queue")
    parser.add_argument("--wan-dir", required=True, help="Path to WanGP directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize queue
    task_queue = HeadlessTaskQueue(args.wan_dir, max_workers=args.workers)

    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal, stopping...")
        task_queue.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start services
        task_queue.start()

        print(f"Headless queue started")
        print(f"WanGP directory: {args.wan_dir}")
        print(f"Workers: {args.workers}")
        print("Press Ctrl+C to stop...")

        # Example: Submit some test tasks
        if args.debug:
            print("\nSubmitting test tasks...")

            # Test T2V task
            t2v_task = create_sample_task(
                "test-t2v-1",
                "t2v",
                "a mystical forest with glowing trees",
                resolution="1280x720",
                video_length=49,
                seed=42
            )
            task_queue.submit_task(t2v_task)

        # Keep running until shutdown
        while task_queue.running:
            time.sleep(1.0)

            # Print periodic status
            if args.debug:
                status = task_queue.get_queue_status()
                print(f"Queue: {status.pending_tasks} pending, "
                      f"{status.completed_tasks} completed, "
                      f"{status.failed_tasks} failed")

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    finally:
        task_queue.stop()


if __name__ == "__main__":
    main()
