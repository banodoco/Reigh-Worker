"""Tests for LTX2 controlled task type: IC-LoRAs, standard LoRAs, image guides.

Smoke tests run without GPU. GPU tests are skipped when CUDA is unavailable.
"""

import inspect
import os
import sys
import tempfile

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
WAN2GP_PATH = os.path.join(PROJECT_ROOT, "Wan2GP")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _convert(db_task_params, task_id="test-ctrl"):
    """Shortcut to call db_task_to_generation_task for ltx2_controlled."""
    from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
    return db_task_to_generation_task(
        db_task_params=db_task_params,
        task_id=task_id,
        task_type="ltx2_controlled",
        wan2gp_path=WAN2GP_PATH,
    )


def _base_params(**overrides):
    """Return minimal valid db_task_params for ltx2_controlled."""
    params = {
        "prompt": "a test video",
        "video_length": 121,
    }
    params.update(overrides)
    return params


def _make_temp_image():
    """Create a temporary file that acts as a stand-in image."""
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # minimal PNG-ish header
    f.close()
    return f.name


def _make_temp_video():
    """Create a temporary file that acts as a stand-in video."""
    f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    f.write(b"\x00" * 64)
    f.close()
    return f.name


# ===================================================================
# 1) Registration
# ===================================================================
class TestRegistration:
    """Verify ltx2_controlled is properly registered."""

    def test_task_type_in_model_mapping(self):
        from source.task_handlers.tasks.task_types import TASK_TYPE_TO_MODEL
        assert "ltx2_controlled" in TASK_TYPE_TO_MODEL
        assert TASK_TYPE_TO_MODEL["ltx2_controlled"] == "ltx2_19B"

    def test_task_type_in_direct_queue(self):
        from source.task_handlers.tasks.task_types import DIRECT_QUEUE_TASK_TYPES
        assert "ltx2_controlled" in DIRECT_QUEUE_TASK_TYPES

    def test_task_type_in_wgp(self):
        from source.task_handlers.tasks.task_types import WGP_TASK_TYPES
        assert "ltx2_controlled" in WGP_TASK_TYPES

    def test_whitelist_has_controlled_params(self):
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
        src = inspect.getsource(db_task_to_generation_task)
        assert "ic_loras" in src, "ic_loras missing from whitelist"
        assert "image_guides" in src, "image_guides missing from whitelist"


# ===================================================================
# 2) IC-LoRA handling
# ===================================================================
class TestICLora:
    """Test IC-LoRA parsing and video_prompt_type flag construction."""

    def test_pose_only_gives_PV_no_G(self):
        """pose IC-LoRA without image_guides → 'PV' (no G)."""
        video_path = _make_temp_video()
        try:
            params = _base_params(ic_loras=[
                {"type": "pose", "weight": 0.8, "guide_video": video_path}
            ])
            task = _convert(params)
            assert task.parameters["video_prompt_type"] == "PV"
        finally:
            os.unlink(video_path)

    def test_depth_canny_gives_DEV(self):
        """depth + canny → 'DEV'."""
        v1 = _make_temp_video()
        v2 = _make_temp_video()
        try:
            params = _base_params(ic_loras=[
                {"type": "depth", "weight": 1.0, "guide_video": v1},
                {"type": "canny", "weight": 0.5, "guide_video": v2},
            ])
            task = _convert(params)
            assert task.parameters["video_prompt_type"] == "DEV"
            assert task.parameters["control_net_weight"] == 1.0
            assert task.parameters["control_net_weight2"] == 0.5
        finally:
            os.unlink(v1)
            os.unlink(v2)

    def test_invalid_ic_type_raises(self):
        """Unknown IC-LoRA type must raise ValueError."""
        video_path = _make_temp_video()
        try:
            params = _base_params(ic_loras=[
                {"type": "unknown_type", "guide_video": video_path}
            ])
            with pytest.raises(ValueError, match="Invalid IC-LoRA type"):
                _convert(params)
        finally:
            os.unlink(video_path)

    def test_ic_lora_with_image_guides_adds_G(self):
        """IC-LoRA + image_guides → 'G' is appended."""
        video_path = _make_temp_video()
        img_path = _make_temp_image()
        try:
            params = _base_params(
                ic_loras=[{"type": "pose", "weight": 1.0, "guide_video": video_path}],
                image_guides=[{
                    "image": img_path,
                    "anchors": [{"frame": 0, "weight": 1.0}],
                }],
            )
            task = _convert(params)
            vpt = task.parameters["video_prompt_type"]
            assert vpt == "PVG", f"Expected 'PVG', got '{vpt}'"
        finally:
            os.unlink(video_path)
            os.unlink(img_path)

    def test_guide_video_missing_raises(self):
        """Local guide_video path that doesn't exist must raise ValueError."""
        params = _base_params(ic_loras=[
            {"type": "pose", "weight": 1.0, "guide_video": "/nonexistent/video.mp4"}
        ])
        with pytest.raises(ValueError, match="guide_video local path does not exist"):
            _convert(params)


# ===================================================================
# 3) Standard LoRAs
# ===================================================================
class TestStandardLoras:
    """Test standard LoRA parsing."""

    def test_single_lora(self):
        params = _base_params(loras=[
            {"path": "/models/my_lora.safetensors", "weight": 0.7}
        ])
        task = _convert(params)
        assert "/models/my_lora.safetensors" in task.parameters["activated_loras"]
        assert "0.7" in task.parameters["loras_multipliers"]

    def test_multiple_loras(self):
        params = _base_params(loras=[
            {"path": "/models/lora_a.safetensors", "weight": 1.0},
            {"path": "/models/lora_b.safetensors", "weight": 0.5},
        ])
        task = _convert(params)
        activated = task.parameters["activated_loras"]
        assert len(activated) == 2
        assert activated[0] == "/models/lora_a.safetensors"
        assert activated[1] == "/models/lora_b.safetensors"
        mults = task.parameters["loras_multipliers"].split()
        assert mults == ["1.0", "0.5"]

    def test_empty_path_skipped(self):
        params = _base_params(loras=[
            {"path": "", "weight": 1.0},
            {"path": "/models/valid.safetensors", "weight": 0.8},
        ])
        task = _convert(params)
        assert len(task.parameters["activated_loras"]) == 1


# ===================================================================
# 4) Image Guides
# ===================================================================
class TestImageGuides:
    """Test image_guides parsing, anchor expansion, and validation."""

    def test_single_anchor(self):
        img_path = _make_temp_image()
        try:
            params = _base_params(image_guides=[{
                "image": img_path,
                "anchors": [{"frame": 0, "weight": 1.0}],
            }])
            task = _convert(params)
            guides = task.parameters["guide_images"]
            assert len(guides) == 1
            assert guides[0]["image"] == img_path
            assert guides[0]["frame_idx"] == 0
            assert guides[0]["strength"] == 1.0
        finally:
            os.unlink(img_path)

    def test_multi_anchor_expansion(self):
        """1 image with 3 anchors → 3 entries in guide_images."""
        img_path = _make_temp_image()
        try:
            params = _base_params(image_guides=[{
                "image": img_path,
                "anchors": [
                    {"frame": 0, "weight": 1.0},
                    {"frame": 50, "weight": 0.8},
                    {"frame": 100, "weight": 0.5},
                ],
            }])
            task = _convert(params)
            guides = task.parameters["guide_images"]
            assert len(guides) == 3
            assert [g["frame_idx"] for g in guides] == [0, 50, 100]
        finally:
            os.unlink(img_path)

    def test_frame_out_of_range_raises(self):
        img_path = _make_temp_image()
        try:
            params = _base_params(
                video_length=100,
                image_guides=[{
                    "image": img_path,
                    "anchors": [{"frame": 200, "weight": 1.0}],
                }],
            )
            with pytest.raises(ValueError, match="out of range"):
                _convert(params)
        finally:
            os.unlink(img_path)

    def test_missing_image_raises(self):
        params = _base_params(image_guides=[{
            "anchors": [{"frame": 0}],
        }])
        with pytest.raises(ValueError, match="must have an 'image' field"):
            _convert(params)

    def test_missing_anchors_raises(self):
        img_path = _make_temp_image()
        try:
            params = _base_params(image_guides=[{
                "image": img_path,
                "anchors": [],
            }])
            with pytest.raises(ValueError, match="must have at least one anchor"):
                _convert(params)
        finally:
            os.unlink(img_path)

    def test_local_path_missing_raises(self):
        """Local image path that doesn't exist must raise ValueError."""
        params = _base_params(image_guides=[{
            "image": "/nonexistent/image.png",
            "anchors": [{"frame": 0, "weight": 1.0}],
        }])
        with pytest.raises(ValueError, match="image_guides local path does not exist"):
            _convert(params)

    def test_frame_minus_one_accepted(self):
        """frame=-1 means last frame and should be accepted."""
        img_path = _make_temp_image()
        try:
            params = _base_params(image_guides=[{
                "image": img_path,
                "anchors": [{"frame": -1, "weight": 1.0}],
            }])
            task = _convert(params)
            guides = task.parameters["guide_images"]
            assert guides[0]["frame_idx"] == -1
        finally:
            os.unlink(img_path)


# ===================================================================
# 5) Combined scenarios
# ===================================================================
class TestCombined:
    """Test combinations of IC-LoRAs, standard LoRAs, and image guides."""

    def test_all_three_together(self):
        """IC-LoRA + standard LoRA + image_guides all work together."""
        video_path = _make_temp_video()
        img_path = _make_temp_image()
        try:
            params = _base_params(
                ic_loras=[{"type": "pose", "weight": 1.0, "guide_video": video_path}],
                loras=[{"path": "/models/style.safetensors", "weight": 0.6}],
                image_guides=[{
                    "image": img_path,
                    "anchors": [{"frame": 0, "weight": 1.0}],
                }],
            )
            task = _convert(params)

            # IC-LoRA set vpt, image_guides added G
            assert task.parameters["video_prompt_type"] == "PVG"
            # Standard LoRA
            assert "/models/style.safetensors" in task.parameters["activated_loras"]
            # Image guides resolved
            assert len(task.parameters["guide_images"]) == 1
        finally:
            os.unlink(video_path)
            os.unlink(img_path)

    def test_metadata_tracking(self):
        """Applied controls metadata should track all active controls."""
        video_path = _make_temp_video()
        img_path = _make_temp_image()
        try:
            params = _base_params(
                ic_loras=[{"type": "depth", "weight": 0.9, "guide_video": video_path}],
                loras=[{"path": "/models/a.safetensors", "weight": 1.0}],
                image_guides=[{
                    "image": img_path,
                    "anchors": [{"frame": 0, "weight": 1.0}],
                }],
            )
            task = _convert(params)
            meta = task.parameters["_applied_controls_metadata"]
            assert "ic_loras" in meta
            assert "loras" in meta
            assert "image_guides" in meta
        finally:
            os.unlink(video_path)
            os.unlink(img_path)

    def test_G_flag_only_with_guides(self):
        """'G' must NOT appear without image_guides, even with IC-LoRA."""
        video_path = _make_temp_video()
        try:
            # IC-LoRA only, no image_guides
            params_no_guides = _base_params(
                ic_loras=[{"type": "canny", "weight": 1.0, "guide_video": video_path}],
            )
            task_no = _convert(params_no_guides)
            assert "G" not in task_no.parameters["video_prompt_type"]

            # Now with image_guides
            img_path = _make_temp_image()
            try:
                params_with_guides = _base_params(
                    ic_loras=[{"type": "canny", "weight": 1.0, "guide_video": video_path}],
                    image_guides=[{
                        "image": img_path,
                        "anchors": [{"frame": 0, "weight": 1.0}],
                    }],
                )
                task_yes = _convert(params_with_guides)
                assert task_yes.parameters["video_prompt_type"].endswith("G")
            finally:
                os.unlink(img_path)
        finally:
            os.unlink(video_path)


# ===================================================================
# 6) GPU integration test
# ===================================================================
@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")
class TestImageGuidesOnlyGenerationGPU:
    """GPU test: generate video with only image_guides (no IC-LoRA)."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        d = tmp_path / "ctrl_outputs"
        d.mkdir()
        return d

    @pytest.fixture
    def chdir_to_wan2gp(self):
        original = os.getcwd()
        os.chdir(WAN2GP_PATH)
        yield WAN2GP_PATH
        os.chdir(original)

    def test_image_guides_only_generation(self, chdir_to_wan2gp, output_dir, guide_images):
        """Generate a video using only image_guides (no IC-LoRA)."""
        from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
        from source.models.wgp.orchestrator import WanOrchestrator

        params = {
            "prompt": "Abstract organic forms floating on black background",
            "video_length": 57,
            "resolution": "768x512",
            "num_inference_steps": 10,
            "guidance_scale": 3.0,
            "seed": 42,
            "image_guides": [
                {
                    "image": os.path.abspath(guide_images[0]),
                    "anchors": [{"frame": 0, "weight": 1.0}],
                },
                {
                    "image": os.path.abspath(guide_images[1]),
                    "anchors": [{"frame": -1, "weight": 1.0}],
                },
            ],
        }

        task = db_task_to_generation_task(
            db_task_params=params,
            task_id="test-guides-only-gpu",
            task_type="ltx2_controlled",
            wan2gp_path=WAN2GP_PATH,
        )

        orch = WanOrchestrator(chdir_to_wan2gp, main_output_dir=str(output_dir))
        orch.load_model(task.model)

        result = orch.generate(prompt=task.prompt, **task.parameters)

        assert result is not None
        assert os.path.exists(result)
        size = os.path.getsize(result)
        assert size > 1024, f"Output too small ({size} bytes)"
        print(f"\nGenerated video: {result}  ({size / 1024:.1f} KB)")


# ===================================================================
# Quick standalone runner
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LTX2 Controlled Task Type Tests")
    print("=" * 60)
    if not HAS_CUDA:
        print("No CUDA available — running smoke tests only.")
        sys.exit(pytest.main([__file__, "-v", "-k", "not GPU", "--tb=short"]))
    else:
        print("CUDA available — running all tests.")
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
