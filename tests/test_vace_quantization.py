"""Tests for source/task_handlers/join/vace_quantization.py."""

import pytest

from source.task_handlers.join.vace_quantization import _calculate_vace_quantization


class TestVaceQuantization:
    """Tests for _calculate_vace_quantization."""

    def test_already_valid_4n_plus_1(self):
        """When total frames already match 4n+1, no quantization shift."""
        # context=8, gap=29, total = 8+29+8 = 45 = 4*11+1
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=29,
            replace_mode=False,
        )
        assert result['total_frames'] == 45
        assert result['quantization_shift'] == 0
        assert result['gap_for_guide'] == 29

    def test_quantization_rounds_down(self):
        """When total is not 4n+1, VACE rounds down."""
        # context=8, gap=30, total = 8+30+8 = 46 → 4*11+1 = 45
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=30,
            replace_mode=False,
        )
        assert result['total_frames'] == 45
        assert result['quantization_shift'] == 1
        assert result['gap_for_guide'] == 29  # 30 - 1

    def test_quantization_shift_of_3(self):
        """Maximum shift before hitting next valid count is 3."""
        # context=8, gap=32, total = 8+32+8 = 48 → 4*11+1 = 45 (shift=3)
        # Wait: 48-1=47, 47//4=11, 11*4+1=45
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=32,
            replace_mode=False,
        )
        assert result['total_frames'] == 45
        assert result['quantization_shift'] == 3
        assert result['gap_for_guide'] == 29  # 32 - 3

    def test_next_valid_boundary(self):
        # context=8, gap=33, total = 8+33+8 = 49 = 4*12+1
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=33,
            replace_mode=False,
        )
        assert result['total_frames'] == 49
        assert result['quantization_shift'] == 0
        assert result['gap_for_guide'] == 33

    def test_common_default_gap53_context8(self):
        """Test the common default configuration."""
        # context=8, gap=53, total = 8+53+8 = 69 = 4*17+1
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=53,
            replace_mode=False,
        )
        assert result['total_frames'] == 69
        assert result['quantization_shift'] == 0
        assert result['gap_for_guide'] == 53

    def test_replace_mode_same_math(self):
        """Replace mode doesn't change the math, just the flag."""
        result_replace = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=29,
            replace_mode=True,
        )
        result_insert = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=29,
            replace_mode=False,
        )
        # Both modes use the same formula now
        assert result_replace['total_frames'] == result_insert['total_frames']
        assert result_replace['gap_for_guide'] == result_insert['gap_for_guide']

    def test_gap_for_guide_never_negative(self):
        """gap_for_guide is clamped to 0 minimum."""
        # Tiny gap with large context => shift could exceed gap
        # context=20, gap=1, total = 20+1+20 = 41 = 4*10+1
        result = _calculate_vace_quantization(
            context_frame_count=20,
            gap_frame_count=1,
            replace_mode=False,
        )
        assert result['gap_for_guide'] >= 0

    def test_very_small_values(self):
        """Minimal values still work."""
        result = _calculate_vace_quantization(
            context_frame_count=1,
            gap_frame_count=1,
            replace_mode=False,
        )
        # total = 1+1+1 = 3, (3-1)//4 = 0, 0*4+1 = 1
        assert result['total_frames'] == 1
        assert result['quantization_shift'] == 2
        assert result['gap_for_guide'] == 0  # max(0, 1-2)

    def test_asymmetric_context(self):
        """context_before and context_after override the symmetric default."""
        result = _calculate_vace_quantization(
            context_frame_count=8,  # default, overridden
            gap_frame_count=29,
            replace_mode=False,
            context_before=4,
            context_after=12,
        )
        # total = 4+29+12 = 45 = 4*11+1
        assert result['total_frames'] == 45
        assert result['quantization_shift'] == 0
        assert result['context_before'] == 4
        assert result['context_after'] == 12

    def test_asymmetric_context_partial_override(self):
        """Only overriding one side uses default for the other."""
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=29,
            replace_mode=False,
            context_before=4,
            context_after=None,  # uses default (8)
        )
        # total = 4+29+8 = 41 = 4*10+1
        assert result['total_frames'] == 41
        assert result['context_before'] == 4
        assert result['context_after'] == 8

    def test_result_keys(self):
        """Verify the returned dict has all expected keys."""
        result = _calculate_vace_quantization(
            context_frame_count=8,
            gap_frame_count=53,
            replace_mode=False,
        )
        assert 'total_frames' in result
        assert 'gap_for_guide' in result
        assert 'quantization_shift' in result
        assert 'context_before' in result
        assert 'context_after' in result

    def test_total_frames_always_4n_plus_1(self):
        """For a range of inputs, verify total_frames always satisfies 4n+1."""
        for gap in range(1, 80):
            for ctx in [4, 8, 12, 16]:
                result = _calculate_vace_quantization(ctx, gap, False)
                total = result['total_frames']
                assert (total - 1) % 4 == 0, f"total_frames={total} is not 4n+1 for gap={gap}, ctx={ctx}"
