"""
Test suite for multi-structure video compositing.

Run with:
    python -m pytest tests/test_multi_structure_video.py -v

Or run directly for visual output:
    python tests/test_multi_structure_video.py
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from source.media.structure import (
    create_neutral_frame,
    validate_structure_video_configs,
    create_composite_guidance_video,
    load_structure_video_frames_with_range,
    calculate_segment_stitched_position,
    extract_segment_structure_guidance,
    segment_has_structure_overlap)


# =============================================================================
# Constants
# =============================================================================

BYTES_PER_MB = 1024 * 1024


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_path():
    """
    Get a sample video path for testing.
    Uses a small test video if available, otherwise skips.
    """
    # Check for common test video locations
    possible_paths = [
        Path("tests/fixtures/sample_video.mp4"),
        Path("test_data/sample.mp4"),
        Path("outputs/test_video.mp4"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Skip if no test video available
    pytest.skip("No sample video found for testing. Create tests/fixtures/sample_video.mp4")


def create_test_video(output_path: Path, num_frames: int = 30, resolution: tuple = (256, 256), fps: int = 16) -> Path:
    """Create a simple test video with colored frames."""
    import cv2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    for i in range(num_frames):
        # Create a frame with color gradient based on frame number
        # This makes it easy to visually verify frame extraction
        hue = int(180 * i / num_frames)  # Cycle through hues
        hsv = np.full((resolution[1], resolution[0], 3), [hue, 255, 200], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add frame number text
        cv2.putText(bgr, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(bgr)
    
    writer.release()
    return output_path


# =============================================================================
# Unit Tests: create_neutral_frame
# =============================================================================

class TestCreateNeutralFrame:
    """Tests for neutral frame creation."""
    
    def test_flow_neutral_is_gray(self):
        """Flow neutral frame should be uniform gray (128)."""
        frame = create_neutral_frame("flow", (64, 64))
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8
        assert np.all(frame == 128)
    
    def test_canny_neutral_is_black(self):
        """Canny neutral frame should be black (0)."""
        frame = create_neutral_frame("canny", (64, 64))
        assert frame.shape == (64, 64, 3)
        assert np.all(frame == 0)
    
    def test_depth_neutral_is_gray(self):
        """Depth neutral frame should be mid-gray (128)."""
        frame = create_neutral_frame("depth", (64, 64))
        assert np.all(frame == 128)
    
    def test_raw_neutral_is_black(self):
        """Raw neutral frame should be black (0)."""
        frame = create_neutral_frame("raw", (64, 64))
        assert np.all(frame == 0)
    
    def test_unknown_type_is_black(self):
        """Unknown type should default to black."""
        frame = create_neutral_frame("unknown_type", (64, 64))
        assert np.all(frame == 0)
    
    def test_resolution_respected(self):
        """Frame should match requested resolution."""
        frame = create_neutral_frame("flow", (320, 240))
        # Note: resolution is (width, height), but numpy is (height, width, channels)
        assert frame.shape == (240, 320, 3)


# =============================================================================
# Unit Tests: validate_structure_video_configs
# =============================================================================

class TestValidateStructureVideoConfigs:
    """Tests for config validation."""
    
    def test_empty_configs(self):
        """Empty config list should return empty."""
        result = validate_structure_video_configs([], 100)
        assert result == []
    
    def test_valid_single_config(self):
        """Single valid config should pass."""
        configs = [{"path": "/test.mp4", "start_frame": 0, "end_frame": 50}]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 1
    
    def test_valid_multiple_configs(self):
        """Multiple non-overlapping configs should pass."""
        configs = [
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 30},
            {"path": "/b.mp4", "start_frame": 50, "end_frame": 80},
        ]
        result = validate_structure_video_configs(configs, 100)
        assert len(result) == 2
    
    def test_configs_sorted_by_start(self):
        """Configs should be sorted by start_frame."""
        configs = [
            {"path": "/b.mp4", "start_frame": 50, "end_frame": 80},
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 30},
        ]
        result = validate_structure_video_configs(configs, 100)
        assert result[0]["start_frame"] == 0
        assert result[1]["start_frame"] == 50
    
    def test_missing_path_raises(self):
        """Missing 'path' should raise ValueError."""
        configs = [{"start_frame": 0, "end_frame": 50}]
        with pytest.raises(ValueError, match="missing 'path'"):
            validate_structure_video_configs(configs, 100)
    
    def test_missing_start_frame_raises(self):
        """Missing 'start_frame' should raise ValueError."""
        configs = [{"path": "/test.mp4", "end_frame": 50}]
        with pytest.raises(ValueError, match="missing 'start_frame'"):
            validate_structure_video_configs(configs, 100)
    
    def test_overlapping_configs_raises(self):
        """Overlapping frame ranges should raise ValueError."""
        configs = [
            {"path": "/a.mp4", "start_frame": 0, "end_frame": 50},
            {"path": "/b.mp4", "start_frame": 40, "end_frame": 80},  # Overlaps!
        ]
        with pytest.raises(ValueError, match="overlaps"):
            validate_structure_video_configs(configs, 100)
    
    def test_out_of_bounds_clips(self):
        """Frame range exceeding total_frames should be clipped."""
        configs = [{"path": "/test.mp4", "start_frame": 0, "end_frame": 150}]
        result = validate_structure_video_configs(configs, 100)
        # end_frame should be clipped to total_frames
        assert result[0]["end_frame"] == 100
    
    def test_invalid_range_raises(self):
        """start_frame >= end_frame should raise."""
        configs = [{"path": "/test.mp4", "start_frame": 50, "end_frame": 30}]
        with pytest.raises(ValueError, match="start_frame.*>= end_frame"):
            validate_structure_video_configs(configs, 100)


# =============================================================================
# Integration Tests: Composite Video Creation
# =============================================================================

class TestCompositeGuidanceVideo:
    """Integration tests for composite video creation."""
    
    def test_single_config_full_coverage(self, temp_output_dir):
        """Single config covering entire timeline."""
        # Create test video
        test_video = create_test_video(temp_output_dir / "source.mp4", num_frames=30)
        
        configs = [{
            "path": str(test_video),
            "start_frame": 0,
            "end_frame": 20,
        }]
        
        output_path = temp_output_dir / "composite.mp4"
        
        result = create_composite_guidance_video(
            structure_configs=configs,
            total_frames=20,
            structure_type="raw",  # raw type doesn't need GPU
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result.exists()
        assert result.stat().st_size > 0
    
    def test_multiple_configs_with_gap(self, temp_output_dir):
        """Multiple configs with neutral gap between them."""
        # Create two test videos
        video_a = create_test_video(temp_output_dir / "source_a.mp4", num_frames=20)
        video_b = create_test_video(temp_output_dir / "source_b.mp4", num_frames=20)
        
        configs = [
            {"path": str(video_a), "start_frame": 0, "end_frame": 10},
            {"path": str(video_b), "start_frame": 15, "end_frame": 25},
            # Gap at frames 10-14
        ]
        
        output_path = temp_output_dir / "composite_gap.mp4"
        
        result = create_composite_guidance_video(
            structure_configs=configs,
            total_frames=25,
            structure_type="raw",  # raw type doesn't need GPU
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result.exists()
        
        # Verify frame count
        import cv2
        cap = cv2.VideoCapture(str(result))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 25
    
    def test_source_range_extraction(self, temp_output_dir):
        """Test extracting specific range from source video."""
        # Create test video with 60 frames
        test_video = create_test_video(temp_output_dir / "source.mp4", num_frames=60)
        
        # Extract frames 20-40 from source, place at timeline 0-20
        configs = [{
            "path": str(test_video),
            "start_frame": 0,
            "end_frame": 20,
            "source_start_frame": 20,
            "source_end_frame": 40,
        }]
        
        output_path = temp_output_dir / "composite_range.mp4"
        
        result = create_composite_guidance_video(
            structure_configs=configs,
            total_frames=20,
            structure_type="raw",  # raw type doesn't need GPU
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result.exists()
    
    def test_different_treatments(self, temp_output_dir):
        """Test different treatment modes (adjust vs clip)."""
        video_a = create_test_video(temp_output_dir / "source_a.mp4", num_frames=30)
        video_b = create_test_video(temp_output_dir / "source_b.mp4", num_frames=50)
        
        configs = [
            {"path": str(video_a), "start_frame": 0, "end_frame": 20, "treatment": "adjust"},
            {"path": str(video_b), "start_frame": 25, "end_frame": 40, "treatment": "clip"},
        ]
        
        output_path = temp_output_dir / "composite_treatments.mp4"
        
        result = create_composite_guidance_video(
            structure_configs=configs,
            total_frames=40,
            structure_type="raw",  # raw type doesn't need GPU
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result.exists()


# =============================================================================
# Tests for Segment-Level Guidance Extraction
# =============================================================================

class TestCalculateSegmentStitchedPosition:
    """Tests for segment position calculation in stitched timeline."""
    
    def test_first_segment(self):
        """First segment starts at 0."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        start, count = calculate_segment_stitched_position(0, seg_frames, overlaps)
        
        assert start == 0
        assert count == 81
    
    def test_second_segment(self):
        """Second segment starts at first_frames - overlap."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        start, count = calculate_segment_stitched_position(1, seg_frames, overlaps)
        
        assert start == 71  # 81 - 10
        assert count == 81
    
    def test_third_segment(self):
        """Third segment position."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        start, count = calculate_segment_stitched_position(2, seg_frames, overlaps)
        
        assert start == 142  # 71 + 81 - 10
        assert count == 81
    
    def test_fifth_segment(self):
        """Fifth segment (last in 5-segment config)."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        start, count = calculate_segment_stitched_position(4, seg_frames, overlaps)
        
        assert start == 284  # 0, 71, 142, 213, 284
        assert count == 81
    
    def test_varying_segment_frames(self):
        """Test with varying frame counts per segment."""
        seg_frames = [81, 65, 49, 81]
        overlaps = [10, 10, 10]
        
        # Segment 0: start=0, count=81
        start0, count0 = calculate_segment_stitched_position(0, seg_frames, overlaps)
        assert start0 == 0
        assert count0 == 81
        
        # Segment 1: start=71 (81-10), count=65
        start1, count1 = calculate_segment_stitched_position(1, seg_frames, overlaps)
        assert start1 == 71
        assert count1 == 65
        
        # Segment 2: start=126 (71+65-10), count=49
        start2, count2 = calculate_segment_stitched_position(2, seg_frames, overlaps)
        assert start2 == 126
        assert count2 == 49
    
    def test_no_overlap(self):
        """Test with no overlap."""
        seg_frames = [81, 81, 81]
        overlaps = [0, 0]
        
        start0, _ = calculate_segment_stitched_position(0, seg_frames, overlaps)
        start1, _ = calculate_segment_stitched_position(1, seg_frames, overlaps)
        start2, _ = calculate_segment_stitched_position(2, seg_frames, overlaps)
        
        assert start0 == 0
        assert start1 == 81
        assert start2 == 162


class TestSegmentHasStructureOverlap:
    """Tests for segment overlap checking helper."""
    
    def test_segment_with_overlap(self):
        """Segment that overlaps with config returns True."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        structure_videos = [{"start_frame": 140, "end_frame": 297}]
        
        # Segment 2 covers [142, 223), config covers [140, 297) - overlap!
        assert segment_has_structure_overlap(2, seg_frames, overlaps, structure_videos) is True
    
    def test_segment_without_overlap(self):
        """Segment that doesn't overlap returns False."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        structure_videos = [{"start_frame": 0, "end_frame": 50}]
        
        # Segment 4 covers [284, 365), config covers [0, 50) - no overlap!
        assert segment_has_structure_overlap(4, seg_frames, overlaps, structure_videos) is False
    
    def test_empty_structure_videos(self):
        """Empty structure_videos returns False."""
        seg_frames = [81, 81, 81]
        overlaps = [10, 10]
        
        assert segment_has_structure_overlap(0, seg_frames, overlaps, []) is False
        assert segment_has_structure_overlap(0, seg_frames, overlaps, None) is False
    
    def test_multiple_configs_one_overlaps(self):
        """Returns True if any config overlaps."""
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        structure_videos = [
            {"start_frame": 0, "end_frame": 50},    # No overlap with segment 2
            {"start_frame": 150, "end_frame": 200}, # Overlaps with segment 2 [142, 223)
        ]
        
        assert segment_has_structure_overlap(2, seg_frames, overlaps, structure_videos) is True


class TestExtractSegmentStructureGuidance:
    """Tests for segment-level guidance extraction."""
    
    def test_segment_with_no_overlap(self, temp_output_dir):
        """Segment that doesn't overlap with any config returns None (no guidance needed)."""
        # Create a test video
        test_video = create_test_video(temp_output_dir / "source.mp4", num_frames=30)
        
        # Config covers frames 0-50, but segment 4 starts at 284
        structure_videos = [{
            "path": str(test_video),
            "start_frame": 0,
            "end_frame": 50,
            "structure_type": "raw",
        }]
        
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        output_path = temp_output_dir / "segment_guidance.mp4"
        
        result = extract_segment_structure_guidance(
            structure_videos=structure_videos,
            segment_index=4,  # Starts at 284, no overlap with 0-50
            segment_frames_expanded=seg_frames,
            frame_overlap_expanded=overlaps,
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        # No overlap = returns None (segment proceeds without structure guidance)
        # This is cleaner than creating an all-neutral video
        assert result is None
    
    def test_segment_with_full_overlap(self, temp_output_dir):
        """Segment fully covered by a config."""
        test_video = create_test_video(temp_output_dir / "source.mp4", num_frames=100)
        
        # Config covers frames 0-200, segment 0 is 0-80
        structure_videos = [{
            "path": str(test_video),
            "start_frame": 0,
            "end_frame": 200,
            "structure_type": "raw",
        }]
        
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        output_path = temp_output_dir / "segment_guidance_full.mp4"
        
        result = extract_segment_structure_guidance(
            structure_videos=structure_videos,
            segment_index=0,
            segment_frames_expanded=seg_frames,
            frame_overlap_expanded=overlaps,
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result is not None
        assert result.exists()
    
    def test_segment_with_partial_overlap(self, temp_output_dir):
        """Segment partially covered by a config."""
        test_video = create_test_video(temp_output_dir / "source.mp4", num_frames=50)
        
        # Config covers frames 60-184, segment 2 covers 142-222
        # Overlap: 142-184 (42 frames)
        structure_videos = [{
            "path": str(test_video),
            "start_frame": 60,
            "end_frame": 184,
            "structure_type": "raw",
        }]
        
        seg_frames = [81, 81, 81, 81, 81]
        overlaps = [10, 10, 10, 10]
        
        output_path = temp_output_dir / "segment_guidance_partial.mp4"
        
        result = extract_segment_structure_guidance(
            structure_videos=structure_videos,
            segment_index=2,  # Covers 142-222
            segment_frames_expanded=seg_frames,
            frame_overlap_expanded=overlaps,
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result is not None
        assert result.exists()
        
        # Should have 81 frames
        import cv2
        cap = cv2.VideoCapture(str(result))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 81
    
    def test_empty_structure_videos(self, temp_output_dir):
        """Empty structure_videos returns None."""
        seg_frames = [81, 81, 81]
        overlaps = [10, 10]
        
        output_path = temp_output_dir / "segment_guidance_empty.mp4"
        
        result = extract_segment_structure_guidance(
            structure_videos=[],
            segment_index=0,
            segment_frames_expanded=seg_frames,
            frame_overlap_expanded=overlaps,
            target_resolution=(128, 128),
            target_fps=16,
            output_path=output_path,
            download_dir=temp_output_dir)
        
        assert result is None


# =============================================================================
# Visual Test Runner (for manual inspection)
# =============================================================================

def run_visual_tests():
    """
    Run visual tests that output files for manual inspection.
    
    This creates several composite videos with different configurations
    so you can visually verify the results.
    """
    import cv2
    
    output_dir = Path("outputs/test_multi_structure")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VISUAL TESTS FOR MULTI-STRUCTURE VIDEO COMPOSITING")
    print("=" * 60)
    
    # Create test source videos
    print("\n[1/5] Creating test source videos...")
    video_a = create_test_video(output_dir / "source_video_a.mp4", num_frames=60, resolution=(256, 256))
    video_b = create_test_video(output_dir / "source_video_b.mp4", num_frames=40, resolution=(256, 256))
    print(f"  Created: {video_a}")
    print(f"  Created: {video_b}")
    
    # Test 1: Single source covering partial timeline (raw type - no GPU needed)
    print("\n[2/5] Test: Single source, partial coverage (raw type)...")
    configs_1 = [{
        "path": str(video_a),
        "start_frame": 10,
        "end_frame": 40,
    }]
    result_1 = create_composite_guidance_video(
        structure_configs=configs_1,
        total_frames=50,
        structure_type="raw",  # raw type doesn't need GPU
        target_resolution=(256, 256),
        target_fps=16,
        output_path=output_dir / "test1_single_partial_raw.mp4",
        download_dir=output_dir)
    print(f"  Output: {result_1}")
    
    # Test 2: Two sources with gap (raw type)
    print("\n[3/5] Test: Two sources with neutral gap (raw type)...")
    configs_2 = [
        {"path": str(video_a), "start_frame": 0, "end_frame": 20},
        {"path": str(video_b), "start_frame": 30, "end_frame": 50},
    ]
    result_2 = create_composite_guidance_video(
        structure_configs=configs_2,
        total_frames=50,
        structure_type="raw",
        target_resolution=(256, 256),
        target_fps=16,
        output_path=output_dir / "test2_two_sources_gap_raw.mp4",
        download_dir=output_dir)
    print(f"  Output: {result_2}")
    
    # Test 3: Source range extraction (raw type)
    print("\n[4/5] Test: Source range extraction (frames 20-50 from 60-frame video)...")
    configs_3 = [{
        "path": str(video_a),
        "start_frame": 0,
        "end_frame": 30,
        "source_start_frame": 20,
        "source_end_frame": 50,
    }]
    result_3 = create_composite_guidance_video(
        structure_configs=configs_3,
        total_frames=30,
        structure_type="raw",
        target_resolution=(256, 256),
        target_fps=16,
        output_path=output_dir / "test3_source_range_raw.mp4",
        download_dir=output_dir)
    print(f"  Output: {result_3}")
    
    # Test 4: Different treatment modes
    print("\n[5/5] Test: Different treatments (adjust vs clip)...")
    configs_4 = [
        {"path": str(video_a), "start_frame": 0, "end_frame": 25, "treatment": "adjust"},
        {"path": str(video_b), "start_frame": 30, "end_frame": 45, "treatment": "clip"},
    ]
    result_4 = create_composite_guidance_video(
        structure_configs=configs_4,
        total_frames=45,
        structure_type="raw",
        target_resolution=(256, 256),
        target_fps=16,
        output_path=output_dir / "test4_treatments_raw.mp4",
        download_dir=output_dir)
    print(f"  Output: {result_4}")
    
    print("\n" + "=" * 60)
    print("VISUAL TESTS COMPLETE")
    print(f"Check outputs in: {output_dir.absolute()}")
    print("=" * 60)
    
    # Print summary
    print("\nGenerated files:")
    for f in output_dir.glob("*.mp4"):
        size_mb = f.stat().st_size / BYTES_PER_MB
        cap = cv2.VideoCapture(str(f))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {f.name}: {frames} frames, {size_mb:.2f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-structure video compositing")
    parser.add_argument("--visual", action="store_true", help="Run visual tests (creates output files)")
    parser.add_argument("--pytest", action="store_true", help="Run pytest unit tests")
    
    args = parser.parse_args()
    
    if args.visual:
        run_visual_tests()
    elif args.pytest:
        pytest.main([__file__, "-v"])
    else:
        # Default: run visual tests
        print("Running visual tests (use --pytest for unit tests)")
        run_visual_tests()

