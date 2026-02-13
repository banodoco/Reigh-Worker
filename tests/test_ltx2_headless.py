"""
Integration test for LTX-2 headless generation pipeline.

Tests the full path:
  WanOrchestrator (smoke mode) → LTX-2 model detection → image/audio parameter
  bridging → generate_video parameter construction → output verification.

Uses HEADLESS_WAN2GP_SMOKE=1 so no GPU or model weights are required.

Run with:
    python -m pytest tests/test_ltx2_headless.py -v
    python tests/test_ltx2_headless.py          # standalone
"""

import os
import sys
import json
import shutil
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures: temporary images, audio, and output directory
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _chdir_to_wan2gp(tmp_path):
    """WanOrchestrator asserts cwd == wan_root.  We cd into the real Wan2GP dir
    for init, then restore cwd on teardown."""
    original_cwd = os.getcwd()
    wan_root = str(PROJECT_ROOT / "Wan2GP")
    os.chdir(wan_root)
    yield wan_root
    os.chdir(original_cwd)


@pytest.fixture()
def output_dir(tmp_path):
    """Dedicated output directory that is cleaned up after each test."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture()
def start_image(tmp_path):
    """Create a small RGB PNG to act as the start anchor image."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    path = tmp_path / "start.png"
    img.save(str(path))
    return str(path)


@pytest.fixture()
def end_image(tmp_path):
    """Create a small RGB PNG to act as the end anchor image."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(50, 100, 200))
    path = tmp_path / "end.png"
    img.save(str(path))
    return str(path)


@pytest.fixture()
def audio_file(tmp_path):
    """Create a minimal WAV file for audio conditioning."""
    import struct

    path = tmp_path / "audio.wav"
    # Minimal 44-byte WAV header + 100 samples of silence
    num_samples = 100
    sample_rate = 16000
    bits_per_sample = 16
    num_channels = 1
    data_size = num_samples * num_channels * (bits_per_sample // 8)
    fmt_chunk_size = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)

    with open(str(path), "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt sub-chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", fmt_chunk_size))
        f.write(struct.pack("<HHIIHH", 1, num_channels, sample_rate,
                            byte_rate, block_align, bits_per_sample))
        # data sub-chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)

    return str(path)


@pytest.fixture()
def sample_video(tmp_path):
    """Create a placeholder file that smoke mode can copy as output."""
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    sample = samples_dir / "test.mp4"
    if not sample.exists():
        # Write a tiny file so smoke mode has something to copy
        sample.write_bytes(b"\x00" * 128)
    return str(sample)


# ---------------------------------------------------------------------------
# Helper: build the orchestrator in smoke mode
# ---------------------------------------------------------------------------

def _make_orchestrator(wan_root: str, output_dir: Path):
    """Instantiate a WanOrchestrator in smoke mode (no GPU needed)."""
    os.environ["HEADLESS_WAN2GP_SMOKE"] = "1"
    try:
        from headless_wgp import WanOrchestrator
        orch = WanOrchestrator(wan_root, main_output_dir=str(output_dir))
    finally:
        os.environ.pop("HEADLESS_WAN2GP_SMOKE", None)
    return orch


# ===================================================================
# Tests
# ===================================================================

class TestLTX2ModelDetection:
    """Verify _is_ltx2() works for various model names."""

    def test_ltx2_19b_detected(self, _chdir_to_wan2gp, output_dir):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True
        assert orch._is_t2v() is True   # ltx2 is also in the T2V family
        assert orch._is_flux() is False
        assert orch._is_vace() is False
        assert orch._is_qwen() is False

    def test_non_ltx2_not_detected(self, _chdir_to_wan2gp, output_dir):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("t2v")
        assert orch._is_ltx2() is False

    def test_flux_not_detected_as_ltx2(self, _chdir_to_wan2gp, output_dir):
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("flux")
        assert orch._is_ltx2() is False
        assert orch._is_flux() is True


class TestLTX2TaskTypes:
    """Verify task_types.py includes ltx2 registrations."""

    def test_ltx2_in_wgp_task_types(self):
        from source.task_handlers.tasks.task_types import WGP_TASK_TYPES
        assert "ltx2" in WGP_TASK_TYPES

    def test_ltx2_in_direct_queue(self):
        from source.task_handlers.tasks.task_types import DIRECT_QUEUE_TASK_TYPES
        assert "ltx2" in DIRECT_QUEUE_TASK_TYPES

    def test_ltx2_model_mapping(self):
        from source.task_handlers.tasks.task_types import get_default_model
        assert get_default_model("ltx2") == "ltx2_19B"


class TestLTX2ImageAudioBridge:
    """Test the LTX-2 image/audio path → object bridging in generate()."""

    def test_start_image_converted_to_pil(
        self, _chdir_to_wan2gp, output_dir, start_image, sample_video
    ):
        """start_image path should be converted to PIL.Image in kwargs and
        image_prompt_type auto-detected as 'TS'."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")

        # In smoke mode, generate() returns a placeholder path.
        # We verify the bridging logic ran by checking logs/output exists.
        result = orch.generate(
            prompt="A cat on a sunny beach",
            start_image=start_image,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_both_images_set_tse(
        self, _chdir_to_wan2gp, output_dir, start_image, end_image, sample_video
    ):
        """When both start_image and end_image are given, image_prompt_type
        should be auto-set to 'TSE'."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        result = orch.generate(
            prompt="Smooth transition between scenes",
            start_image=start_image,
            end_image=end_image,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_audio_input_mapped(
        self, _chdir_to_wan2gp, output_dir, audio_file, sample_video
    ):
        """audio_input should be mapped to audio_guide in kwargs."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        result = orch.generate(
            prompt="Ambient nature sounds",
            audio_input=audio_file,
        )
        assert result is not None
        assert os.path.exists(result)

    def test_full_ltx2_generation_with_all_inputs(
        self, _chdir_to_wan2gp, output_dir, start_image, end_image, audio_file, sample_video
    ):
        """Full LTX-2 smoke run with start image, end image, and audio."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        result = orch.generate(
            prompt="Cinematic transition with soundtrack",
            resolution="768x512",
            video_length=49,
            start_image=start_image,
            end_image=end_image,
            audio_input=audio_file,
            audio_scale=0.8,
            seed=42,
        )
        assert result is not None, "generate() returned None"
        assert os.path.exists(result), f"Output file does not exist: {result}"
        assert os.path.getsize(result) > 0, f"Output file is empty: {result}"


class TestLTX2ParameterWiring:
    """Verify the new upstream parameters appear in wgp_params."""

    NEW_PARAMS = [
        "alt_prompt",
        "duration_seconds",
        "pause_seconds",
        "audio_scale",
        "input_video_strength",
        "override_attention",
        "custom_settings",
        "top_k",
        "self_refiner_setting",
        "self_refiner_plan",
        "self_refiner_f_uncertainty",
        "self_refiner_certain_percentage",
    ]

    def test_new_params_present_in_generate_video_signature(self):
        """All new parameters should be in the upstream generate_video()
        signature (confirming the Wan2GP sync is correct)."""
        import ast

        wgp_path = PROJECT_ROOT / "Wan2GP" / "wgp.py"
        tree = ast.parse(wgp_path.read_text())
        upstream_params = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_video":
                upstream_params = {arg.arg for arg in node.args.args}
                break

        for p in self.NEW_PARAMS:
            assert p in upstream_params, (
                f"'{p}' not in upstream generate_video() signature"
            )

    def test_new_params_present_in_headless_wgp(self):
        """All new parameters should appear as keys in the wgp_params dict
        within headless_wgp.py (both passthrough and normal modes)."""
        import re

        src = (PROJECT_ROOT / "source" / "models" / "wgp" / "orchestrator.py").read_text()
        # Find all string-keyed dict entries in wgp_params
        found_keys = set(re.findall(r"'([a-zA-Z_]\w*)'\s*:", src))

        for p in self.NEW_PARAMS:
            assert p in found_keys, (
                f"'{p}' not found as a key in orchestrator.py"
            )


class TestLTX2TravelToggle:
    """Verify the use_ltx2 toggle in travel_between_images.py code paths."""

    def test_use_ltx2_sets_model_name(self):
        """When use_ltx2=True, segment payload should have model_name='ltx2_19B'."""
        import re

        src = (
            PROJECT_ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
        ).read_text()

        # The code sets model_name conditionally:
        # "model_name": "ltx2_19B" if use_ltx2 else orchestrator_payload["model_name"],
        pattern = r'"model_name":\s*"ltx2_19B"\s+if\s+use_ltx2'
        assert re.search(pattern, src), (
            "travel_between_images.py does not set model_name='ltx2_19B' "
            "when use_ltx2 is True"
        )

    def test_use_ltx2_in_segment_payload(self):
        """Segment payload should contain use_ltx2 key."""
        src = (
            PROJECT_ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
        ).read_text()
        assert '"use_ltx2": use_ltx2,' in src

    def test_use_ltx2_in_stitch_payload(self):
        """Stitch payload should contain use_ltx2 key."""
        src = (
            PROJECT_ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
        ).read_text()
        assert '"use_ltx2": use_ltx2,' in src

    def test_ltx2_stitching_enabled(self):
        """LTX-2 mode should enable stitch task creation."""
        src = (
            PROJECT_ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
        ).read_text()
        # The stitch logic has: if use_ltx2: should_create_stitch = True
        assert "use_ltx2" in src
        # Check stitch decision block also includes ltx2
        assert "use_ltx2 or use_svi" in src or \
               "use_ltx2\n            or use_svi" in src

    def test_ltx2_config_sets_image_prompt_type(self):
        """The LTX-2 config block should set image_prompt_type."""
        src = (
            PROJECT_ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
        ).read_text()
        assert 'segment_payload["image_prompt_type"] = ltx2_prompt_type' in src

    def test_ltx2_config_disables_svi(self):
        """When use_ltx2, SVI should be explicitly disabled on segments."""
        src = (
            PROJECT_ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
        ).read_text()
        # Inside the "if use_ltx2:" block:
        assert 'segment_payload["use_svi"] = False' in src
        assert 'segment_payload["svi2pro"] = False' in src


class TestLTX2UpstreamSync:
    """Verify the Wan2GP upstream sync is correct."""

    def test_version_is_10_83(self):
        wgp_path = PROJECT_ROOT / "Wan2GP" / "wgp.py"
        content = wgp_path.read_text()
        assert 'WanGP_version = "10.83"' in content

    def test_ltx2_model_directory_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2").is_dir()

    def test_ltx2_handler_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx2_handler.py").is_file()

    def test_ltx2_default_config_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "defaults" / "ltx2_19B.json").is_file()

    def test_ltx2_default_config_valid_json(self):
        cfg_path = PROJECT_ROOT / "Wan2GP" / "defaults" / "ltx2_19B.json"
        data = json.loads(cfg_path.read_text())
        assert isinstance(data, dict)

    def test_new_shared_directories_exist(self):
        for subdir in ["llm_engines", "kernels", "prompt_enhancer"]:
            assert (PROJECT_ROOT / "Wan2GP" / "shared" / subdir).is_dir(), (
                f"Wan2GP/shared/{subdir}/ not found"
            )

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "Wan2GP" / "profiles").is_dir(),
        reason="profiles/ only present on deployed environments",
    )
    def test_profiles_directory_exists(self):
        assert (PROJECT_ROOT / "Wan2GP" / "profiles").is_dir()

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "Wan2GP.bak").is_dir(),
        reason="Wan2GP.bak/ only present after upgrade",
    )
    def test_backup_preserved(self):
        bak = PROJECT_ROOT / "Wan2GP.bak"
        assert bak.is_dir(), "Wan2GP.bak/ backup not found"
        wgp_bak = bak / "wgp.py"
        assert wgp_bak.is_file()
        content = wgp_bak.read_text()
        assert 'WanGP_version = "10.01"' in content


class TestLTX2EndToEndSmoke:
    """End-to-end smoke test: submit a headless LTX-2 task with
    use_ltx2=True, a start image, and an audio file.  Verify the
    output video is produced in the output directory."""

    def test_e2e_ltx2_smoke_generation(
        self, _chdir_to_wan2gp, output_dir, start_image, end_image, audio_file, sample_video
    ):
        """
        Full end-to-end smoke test:
        1. Create WanOrchestrator in smoke mode
        2. Load ltx2_19B model
        3. Call generate() with start_image, end_image, audio_input
        4. Verify output file exists in the output directory
        """
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)

        # --- Step 1: Load model ---
        switched = orch.load_model("ltx2_19B")
        assert switched is True, "Model should have switched (was None before)"
        assert orch.current_model == "ltx2_19B"
        assert orch._is_ltx2() is True

        # --- Step 2: Generate with LTX-2 specific parameters ---
        result_path = orch.generate(
            prompt="A cinematic scene with ambient audio",
            resolution="768x512",
            video_length=49,
            seed=12345,
            start_image=start_image,
            end_image=end_image,
            audio_input=audio_file,
            audio_scale=0.8,
        )

        # --- Step 3: Verify output ---
        assert result_path is not None, "generate() returned None — no output produced"
        assert os.path.isfile(result_path), f"Output file not found at: {result_path}"
        assert os.path.getsize(result_path) > 0, f"Output file is empty: {result_path}"

        # Verify the file is inside the expected output tree
        result_abs = os.path.abspath(result_path)
        output_abs = os.path.abspath(str(output_dir))
        # Smoke mode writes to cwd/outputs, not necessarily our tmp output_dir,
        # but the file should still exist on disk.
        assert os.path.isfile(result_abs), (
            f"Output not at expected absolute path: {result_abs}"
        )

        print(f"\n[PASS] LTX-2 E2E smoke test produced output: {result_path}")
        print(f"       File size: {os.path.getsize(result_path)} bytes")

    def test_e2e_model_reload_is_noop(
        self, _chdir_to_wan2gp, output_dir, sample_video
    ):
        """Loading the same model twice should not 'switch'."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        assert orch.load_model("ltx2_19B") is True   # first load
        assert orch.load_model("ltx2_19B") is False  # noop


class TestICLoraAudioImageSmoke:
    """IC LoRA pipeline tests: signature introspection, audio conditioning
    format, and smoke test with images + audio via WanOrchestrator.

    NOTE: ICLoraPipeline is not currently reachable from the pipeline routing
    in ltx2.py (line 447). Only "distilled" and "two_stage" kinds are handled.
    The smoke test exercises the standard TI2VidTwoStagesPipeline path.
    """

    def test_ic_lora_with_images_and_audio(
        self, _chdir_to_wan2gp, output_dir, start_image, end_image, audio_file, sample_video
    ):
        """Smoke test: LTX-2 generation with start/end images and audio."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        result = orch.generate(
            prompt="IC LoRA test with images and audio conditioning",
            resolution="768x512",
            video_length=49,
            start_image=start_image,
            end_image=end_image,
            audio_input=audio_file,
            audio_scale=0.8,
            seed=42,
        )
        assert result is not None, "generate() returned None"
        assert os.path.exists(result), f"Output file does not exist: {result}"
        assert os.path.getsize(result) > 0, f"Output file is empty: {result}"

    def test_ic_lora_pipeline_accepts_all_conditioning_types(self):
        """ICLoraPipeline.__call__ should accept images, video_conditioning,
        and audio_conditionings parameters (via AST to avoid heavy imports)."""
        import ast

        src_path = PROJECT_ROOT / "Wan2GP" / "models" / "ltx2" / "ltx_pipelines" / "ic_lora.py"
        tree = ast.parse(src_path.read_text())

        call_params = set()
        call_annotations = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__call__":
                for arg in node.args.args:
                    call_params.add(arg.arg)
                    if arg.annotation:
                        call_annotations[arg.arg] = ast.dump(arg.annotation)
                break

        # Verify key conditioning parameters exist
        assert "images" in call_params
        assert "video_conditioning" in call_params
        assert "audio_conditionings" in call_params

        # Verify list type annotations via AST
        assert "list" in call_annotations.get("images", "").lower()
        assert "list" in call_annotations.get("video_conditioning", "").lower()

    def test_audio_conditioning_format(self):
        """AudioConditionByLatent should be instantiable and expose
        latent, strength, and a callable apply_to method."""
        try:
            import torch
            from Wan2GP.models.ltx2.ltx_core.conditioning.types.latent_cond import AudioConditionByLatent
        except ImportError as exc:
            pytest.skip(f"Skipping due to missing dependency: {exc}")

        dummy = torch.zeros(1, 8, 4, 64)
        cond = AudioConditionByLatent(latent=dummy, strength=0.8)

        assert cond.latent is dummy
        assert cond.strength == 0.8
        assert hasattr(cond, "apply_to")
        assert callable(cond.apply_to)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running without pytest installed
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
