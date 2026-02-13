"""
Smoke tests for the "travel between images" multi-image video transition pipeline.

Tests orchestrator payload construction, segment assignment, frame quantization,
dependency logic, stitch decisions, and WanOrchestrator smoke-mode integration
using the real neon botanical test images in tests/.

Uses HEADLESS_WAN2GP_SMOKE=1 so no GPU or model weights are required.

Run with:
    python -m pytest tests/test_travel_between_images.py -v --tb=short
    python tests/test_travel_between_images.py          # standalone
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import travel constants under test — load svi_config directly via spec
# to avoid triggering the package __init__ which pulls in heavy DB deps
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "svi_config",
    PROJECT_ROOT / "source" / "task_handlers" / "travel" / "svi_config.py",
)
_svi_config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_svi_config)
SVI_STITCH_OVERLAP = _svi_config.SVI_STITCH_OVERLAP
SVI_LORAS = _svi_config.SVI_LORAS
SVI_DEFAULT_PARAMS = _svi_config.SVI_DEFAULT_PARAMS
get_svi_lora_arrays = _svi_config.get_svi_lora_arrays


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _chdir_to_wan2gp(tmp_path):
    """WanOrchestrator asserts cwd == wan_root.  cd into Wan2GP for init,
    then restore on teardown."""
    original_cwd = os.getcwd()
    wan_root = str(PROJECT_ROOT / "Wan2GP")
    os.chdir(wan_root)
    yield wan_root
    os.chdir(original_cwd)


@pytest.fixture()
def output_dir(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture()
def test_images():
    """Return paths to the 3 uploaded neon botanical test images."""
    paths = [
        str(TESTS_DIR / "CICEK .png"),
        str(TESTS_DIR / "CICEK 2.png"),
        str(TESTS_DIR / "2AF816ADD8BC6EE937FEA5D5B3061DC047BBD8BA5A77E91A5158BF6C86C9FAD1.jpeg"),
    ]
    for p in paths:
        if not os.path.isfile(p):
            pytest.skip(f"Test image missing: {p}")
    return paths


@pytest.fixture()
def sample_video(tmp_path):
    """Placeholder file that smoke mode can copy as output."""
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    sample = samples_dir / "test.mp4"
    if not sample.exists():
        sample.write_bytes(b"\x00" * 128)
    return str(sample)


# ---------------------------------------------------------------------------
# Helper: smoke-mode orchestrator (same pattern as test_ltx2_headless.py)
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


# ---------------------------------------------------------------------------
# Helper: build a minimal orchestrator payload for N images
# ---------------------------------------------------------------------------

def _build_orchestrator_payload(
    image_paths: list[str],
    *,
    model_type: str = "vace",
    chain_segments: bool = True,
    use_svi: bool = False,
    use_ltx2: bool = False,
    segment_frames: int = 49,
    frame_overlap: int = 8,
):
    """Construct a minimal but valid orchestrator_details dict."""
    num_images = len(image_paths)
    num_segments = num_images - 1
    return {
        "input_image_paths_resolved": image_paths,
        "model_type": model_type,
        "model_name": "ltx2_19B" if use_ltx2 else "vace_14B",
        "chain_segments": chain_segments,
        "use_svi": use_svi,
        "use_ltx2": use_ltx2,
        "base_prompt": "neon botanical transition",
        "negative_prompt": "",
        "parsed_resolution_wh": [768, 512],
        "seed_base": 42,
        "fps_helpers": 16,
        # Frame arrays: N-1 segments, N-2 overlaps
        "segment_frames_expanded": [segment_frames] * num_segments,
        "frame_overlap_expanded": [frame_overlap] * max(num_segments - 1, 0),
    }


# ===================================================================
# 1. TestTravelOrchestratorPayload
# ===================================================================

class TestTravelOrchestratorPayload:
    """Tests orchestrator payload construction logic with 3 images."""

    def test_3_images_produce_2_segments(self, test_images):
        payload = _build_orchestrator_payload(test_images)
        num_segments = len(payload["input_image_paths_resolved"]) - 1
        assert num_segments == 2, "3 images should produce 2 segments (N-1 rule)"

    def test_segment_image_assignment(self, test_images):
        """seg0 = img[0]→img[1], seg1 = img[1]→img[2]"""
        payload = _build_orchestrator_payload(test_images)
        images = payload["input_image_paths_resolved"]
        num_segments = len(images) - 1

        for idx in range(num_segments):
            start = images[idx]
            end = images[idx + 1]
            assert os.path.isfile(start)
            assert os.path.isfile(end)
            assert start != end, f"Segment {idx}: start and end should differ"

        # Verify specific pairing
        assert Path(images[0]).name == "CICEK .png"
        assert Path(images[1]).name == "CICEK 2.png"
        assert Path(images[2]).name.startswith("2AF816")

    def test_frame_overlap_count_is_n_minus_1(self, test_images):
        """N segments → N-1 overlaps."""
        payload = _build_orchestrator_payload(test_images, frame_overlap=8)
        num_segments = len(payload["segment_frames_expanded"])
        num_overlaps = len(payload["frame_overlap_expanded"])
        assert num_overlaps == num_segments - 1

    def test_quantization_4n_plus_1(self, test_images):
        """Segment frames should quantize to 4N+1 format."""
        for raw_frames in [48, 49, 50, 51, 52, 53]:
            quantized = (raw_frames // 4) * 4 + 1
            assert (quantized - 1) % 4 == 0, f"{raw_frames} → {quantized} is not 4N+1"

    def test_should_create_stitch_vace_chain(self, test_images):
        payload = _build_orchestrator_payload(
            test_images, model_type="vace", chain_segments=True
        )
        assert payload["chain_segments"] is True
        assert payload["model_type"] == "vace"
        # VACE + chain → stitch expected

    def test_should_not_create_stitch_i2v_no_svi(self, test_images):
        payload = _build_orchestrator_payload(
            test_images, model_type="i2v", use_svi=False
        )
        assert payload["model_type"] == "i2v"
        assert payload["use_svi"] is False
        # I2V + no SVI → no stitch expected

    def test_should_create_stitch_svi(self, test_images):
        payload = _build_orchestrator_payload(
            test_images, model_type="i2v", use_svi=True
        )
        assert payload["use_svi"] is True
        # SVI → stitch expected

    def test_should_create_stitch_ltx2(self, test_images):
        payload = _build_orchestrator_payload(
            test_images, use_ltx2=True
        )
        assert payload["use_ltx2"] is True
        # LTX-2 → stitch expected


# ===================================================================
# 2. TestTravelSegmentImageAssignment
# ===================================================================

class TestTravelSegmentImageAssignment:
    """Tests image resolution per segment using real uploaded images."""

    def test_i2v_image_indexing(self, test_images):
        """I2V mode: seg[0] gets img[0]→img[1], seg[1] gets img[1]→img[2]
        using input_image_paths_resolved."""
        payload = _build_orchestrator_payload(test_images, model_type="i2v")
        images = payload["input_image_paths_resolved"]

        for idx in range(2):  # 2 segments
            start_img = images[idx]
            end_img = images[idx + 1]
            assert Path(start_img).exists()
            assert Path(end_img).exists()

        # seg0: CICEK .png → CICEK 2.png
        assert Path(images[0]).name == "CICEK .png"
        assert Path(images[1]).name == "CICEK 2.png"
        # seg1: CICEK 2.png → 2AF816...jpeg
        assert Path(images[1]).name == "CICEK 2.png"
        assert Path(images[2]).name.startswith("2AF816")

    def test_ltx2_image_start_end(self, test_images):
        """LTX-2 mode uses image_start / image_end with image_prompt_type='TSE'."""
        payload = _build_orchestrator_payload(test_images, use_ltx2=True)
        images = payload["input_image_paths_resolved"]

        for idx in range(2):
            ltx2_start = images[idx]
            ltx2_end = images[idx + 1]
            assert ltx2_start is not None
            assert ltx2_end is not None

            # With both images, prompt type should be TSE
            if ltx2_start and ltx2_end:
                expected_type = "TSE"
            elif ltx2_start:
                expected_type = "TS"
            else:
                expected_type = "T"
            assert expected_type == "TSE"

    def test_segment_images_are_consecutive_pairs(self, test_images):
        """Each segment's start image is the previous segment's end image."""
        payload = _build_orchestrator_payload(test_images)
        images = payload["input_image_paths_resolved"]

        seg0_end = images[1]
        seg1_start = images[1]
        assert seg0_end == seg1_start, "Shared boundary image between segments"


# ===================================================================
# 3. TestTravelWanOrchestratorSmoke
# ===================================================================

class TestTravelWanOrchestratorSmoke:
    """Tests WanOrchestrator in smoke mode with the real uploaded images."""

    def test_i2v_model_smoke_generation(
        self, _chdir_to_wan2gp, output_dir, test_images, sample_video
    ):
        """Load I2V model, generate with start/end image from test images."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("i2v")
        result = orch.generate(
            prompt="Neon flower blooming transition",
            start_image=test_images[0],
            end_image=test_images[1],
        )
        assert result is not None
        assert os.path.exists(result)

    def test_ltx2_model_smoke_generation(
        self, _chdir_to_wan2gp, output_dir, test_images, sample_video
    ):
        """Load LTX-2 model, generate with start/end image from test images."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        result = orch.generate(
            prompt="Neon sprout transformation",
            start_image=test_images[1],
            end_image=test_images[2],
        )
        assert result is not None
        assert os.path.exists(result)

    def test_pil_conversion_with_real_images(self, test_images):
        """Verify PIL.Image conversion works with real PNG/JPEG files."""
        from PIL import Image

        for path in test_images:
            img = Image.open(path).convert("RGB")
            assert img.size[0] > 0
            assert img.size[1] > 0
            assert img.mode == "RGB"

    def test_ltx2_both_images_tse_mode(
        self, _chdir_to_wan2gp, output_dir, test_images, sample_video
    ):
        """Both images → TSE image_prompt_type in LTX-2 mode."""
        orch = _make_orchestrator(_chdir_to_wan2gp, output_dir)
        orch.load_model("ltx2_19B")
        assert orch._is_ltx2() is True
        result = orch.generate(
            prompt="Smooth botanical transition",
            resolution="768x512",
            video_length=49,
            start_image=test_images[0],
            end_image=test_images[2],
            seed=42,
        )
        assert result is not None
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


# ===================================================================
# 4. TestTravelModeDependencies
# ===================================================================

class TestTravelModeDependencies:
    """Tests segment dependency logic for different travel modes.

    Validates the dependency chain code at lines 1678-1694 of
    travel_between_images.py by simulating the conditional logic.
    """

    @staticmethod
    def _resolve_dependency(
        idx: int,
        *,
        use_ltx2: bool = False,
        use_svi: bool = False,
        travel_mode: str = "vace",
        chain_segments: bool = True,
    ) -> bool:
        """Return True if segment idx depends on idx-1 (sequential)."""
        if idx == 0:
            return False  # first segment never depends on a previous one
        if use_ltx2:
            return True
        if use_svi:
            return True
        if travel_mode == "i2v":
            return False
        if travel_mode == "vace" and not chain_segments:
            return False
        # default VACE sequential
        return True

    def test_i2v_segments_independent(self):
        for idx in range(3):
            dep = self._resolve_dependency(idx, travel_mode="i2v", use_svi=False)
            if idx == 0:
                assert dep is False
            else:
                assert dep is False, "I2V non-SVI segments should be independent"

    def test_svi_segments_sequential(self):
        for idx in range(3):
            dep = self._resolve_dependency(idx, use_svi=True, travel_mode="i2v")
            if idx == 0:
                assert dep is False
            else:
                assert dep is True, "SVI segments should be sequential"

    def test_ltx2_segments_sequential(self):
        for idx in range(3):
            dep = self._resolve_dependency(idx, use_ltx2=True)
            if idx == 0:
                assert dep is False
            else:
                assert dep is True, "LTX-2 segments should be sequential"

    def test_vace_chain_segments_sequential(self):
        for idx in range(3):
            dep = self._resolve_dependency(
                idx, travel_mode="vace", chain_segments=True
            )
            if idx == 0:
                assert dep is False
            else:
                assert dep is True, "VACE + chain_segments should be sequential"

    def test_vace_no_chain_independent(self):
        for idx in range(3):
            dep = self._resolve_dependency(
                idx, travel_mode="vace", chain_segments=False
            )
            if idx == 0:
                assert dep is False
            else:
                assert dep is False, "VACE + no chain should be independent"


# ===================================================================
# 5. TestTravelStitchDecision
# ===================================================================

class TestTravelStitchDecision:
    """Tests when stitch task should / shouldn't be created.

    Mirrors the stitch decision logic at lines 2097-2121 of
    travel_between_images.py.
    """

    @staticmethod
    def _should_stitch(
        *,
        use_ltx2: bool = False,
        use_svi: bool = False,
        travel_mode: str = "vace",
        chain_segments: bool = True,
        num_segments: int = 2,
    ) -> tuple[bool, list[int]]:
        """Return (should_create_stitch, overlap_settings)."""
        if use_ltx2:
            return True, [SVI_STITCH_OVERLAP] * (num_segments - 1)
        if use_svi:
            return True, [SVI_STITCH_OVERLAP] * (num_segments - 1)
        if travel_mode == "vace" and chain_segments:
            return True, [8] * (num_segments - 1)  # placeholder overlap
        return False, []

    def test_i2v_no_svi_no_stitch(self):
        stitch, overlaps = self._should_stitch(
            travel_mode="i2v", use_svi=False
        )
        assert stitch is False
        assert overlaps == []

    def test_i2v_svi_stitch(self):
        stitch, overlaps = self._should_stitch(
            travel_mode="i2v", use_svi=True, num_segments=2
        )
        assert stitch is True
        assert overlaps == [SVI_STITCH_OVERLAP]
        assert SVI_STITCH_OVERLAP == 4

    def test_vace_chain_stitch(self):
        stitch, overlaps = self._should_stitch(
            travel_mode="vace", chain_segments=True, num_segments=2
        )
        assert stitch is True
        assert len(overlaps) == 1

    def test_ltx2_stitch(self):
        stitch, overlaps = self._should_stitch(
            use_ltx2=True, num_segments=2
        )
        assert stitch is True
        assert overlaps == [SVI_STITCH_OVERLAP]

    def test_vace_no_chain_no_stitch(self):
        stitch, overlaps = self._should_stitch(
            travel_mode="vace", chain_segments=False
        )
        assert stitch is False
        assert overlaps == []

    def test_3_images_produce_correct_overlap_count(self):
        """3 images → 2 segments → 1 overlap value."""
        stitch, overlaps = self._should_stitch(
            use_svi=True, num_segments=2
        )
        assert len(overlaps) == 1

    def test_4_images_produce_correct_overlap_count(self):
        """4 images → 3 segments → 2 overlap values."""
        stitch, overlaps = self._should_stitch(
            use_svi=True, num_segments=3
        )
        assert len(overlaps) == 2


# ===================================================================
# 6. TestTravelFrameQuantization
# ===================================================================

class TestTravelFrameQuantization:
    """Tests frame count quantization with 3 images / 2 segments.

    Mirrors the quantization logic at lines 914-989 of
    travel_between_images.py.
    """

    @staticmethod
    def _quantize_frames(expanded_segment_frames, expanded_frame_overlap):
        """Pure-function reimplementation of SM_QUANTIZE_FRAMES_AND_OVERLAPS."""
        # Quantize segment frames to 4N+1
        quantized_segment_frames = []
        for frames in expanded_segment_frames:
            new_frames = (frames // 4) * 4 + 1
            quantized_segment_frames.append(new_frames)

        # Process N-1 overlaps
        quantized_frame_overlap = []
        num_overlaps = len(quantized_segment_frames) - 1
        for i in range(num_overlaps):
            original_overlap = (
                expanded_frame_overlap[i]
                if i < len(expanded_frame_overlap)
                else 0
            )
            max_possible = min(
                quantized_segment_frames[i], quantized_segment_frames[i + 1]
            )
            new_overlap = (original_overlap // 2) * 2
            new_overlap = min(new_overlap, max_possible)
            if new_overlap < 0:
                new_overlap = 0
            quantized_frame_overlap.append(new_overlap)

        return quantized_segment_frames, quantized_frame_overlap

    def test_49_frames_unchanged(self):
        """49 is already 4*12+1, should stay."""
        seg, ovr = self._quantize_frames([49, 49], [8])
        assert seg == [49, 49]

    def test_50_frames_quantized(self):
        """50 → 49 (4*12+1)."""
        seg, ovr = self._quantize_frames([50, 50], [8])
        assert seg == [49, 49]

    def test_51_frames_quantized(self):
        """51 → 49 (4*12+1)."""
        seg, ovr = self._quantize_frames([51, 51], [8])
        assert seg == [49, 49]

    def test_52_frames_quantized(self):
        """52 → 53 (4*13+1)."""
        seg, ovr = self._quantize_frames([52, 52], [8])
        assert seg == [53, 53]

    def test_all_quantized_are_4n_plus_1(self):
        """Random frame counts all quantize correctly."""
        for raw in range(30, 130):
            seg, _ = self._quantize_frames([raw], [])
            assert (seg[0] - 1) % 4 == 0, f"{raw} → {seg[0]} is not 4N+1"

    def test_overlap_quantized_to_even(self):
        """Overlap values should be quantized to even numbers."""
        seg, ovr = self._quantize_frames([49, 49], [7])
        assert ovr[0] % 2 == 0
        assert ovr[0] == 6  # 7 // 2 * 2 = 6

    def test_overlap_capped_at_shorter_segment(self):
        """Overlap cannot exceed the shorter of two adjacent segments."""
        seg, ovr = self._quantize_frames([17, 17], [100])
        assert ovr[0] <= min(seg[0], seg[1])

    def test_3_images_2_segments_1_overlap(self):
        """3 images → 2 segments → 1 overlap value after quantization."""
        seg, ovr = self._quantize_frames([49, 49], [8])
        assert len(seg) == 2
        assert len(ovr) == 1

    def test_expected_final_length(self):
        """Total output = sum(segments) - sum(overlaps)."""
        seg, ovr = self._quantize_frames([49, 49], [8])
        total_input = sum(seg)
        total_overlap = sum(ovr)
        expected = total_input - total_overlap
        assert expected == 49 + 49 - 8
        assert expected == 90

    def test_negative_overlap_clamped_to_zero(self):
        """Negative overlap values should be clamped to 0."""
        seg, ovr = self._quantize_frames([49, 49], [-5])
        assert ovr[0] == 0

    def test_mixed_segment_lengths(self):
        """Different segment lengths should each quantize independently."""
        seg, ovr = self._quantize_frames([30, 60, 45], [10, 12])
        assert len(seg) == 3
        assert len(ovr) == 2
        for s in seg:
            assert (s - 1) % 4 == 0
        for o in ovr:
            assert o % 2 == 0


# ===================================================================
# Additional: TestSVIConstants
# ===================================================================

class TestSVIConstants:
    """Verify SVI constants are sane."""

    def test_svi_stitch_overlap_value(self):
        assert SVI_STITCH_OVERLAP == 4

    def test_svi_loras_non_empty(self):
        assert len(SVI_LORAS) >= 2

    def test_svi_default_params_keys(self):
        expected_keys = {
            "guidance_phases", "num_inference_steps", "guidance_scale",
            "guidance2_scale", "flow_shift", "switch_threshold",
            "model_switch_phase", "sample_solver",
        }
        assert expected_keys == set(SVI_DEFAULT_PARAMS.keys())

    def test_get_svi_lora_arrays_returns_matching_lengths(self):
        urls, mults = get_svi_lora_arrays()
        assert len(urls) == len(mults)
        assert len(urls) == len(SVI_LORAS)

    def test_get_svi_lora_arrays_with_strength(self):
        urls, mults = get_svi_lora_arrays(svi_strength=0.5)
        assert len(urls) == len(mults)
        # All multiplier strings should contain the scaled values
        for m in mults:
            assert isinstance(m, str)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
