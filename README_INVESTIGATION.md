# Task Failure Investigation - Complete Analysis

## Overview

Two travel orchestrator segment generation tasks failed with identical error messages and very similar root causes. This investigation provides comprehensive evidence and fixes.

## Files Generated

1. **INVESTIGATION_REPORT.md** - Complete root cause analysis with timelines
2. **TASK_FAILURE_SUMMARY.txt** - Quick reference with key findings
3. **CODE_INVESTIGATION_LOCATIONS.md** - Exact files and lines to investigate
4. **VISUAL_COMPARISON.txt** - Side-by-side comparison of both failures
5. **README_INVESTIGATION.md** - This file

## Quick Summary

### Error
```
'NoneType' object has no attribute 'get'
```

### Location
Parameter resolution loop during generation:
- File: `/source/models/wgp/param_resolution.py` (Lines 70-91)
- Specifically: Parameter resolution between LOOP [8] and LOOP [26] (timing varies)

### Root Cause
The parameter `guidance_scale` or `guidance_phases` triggers an exception when the code attempts to access it from the `resolved_params` dictionary.

### Evidence
1. Both failures have identical error messages
2. Both fail during parameter iteration
3. Both use the same model: `wan_2_2_i2v_lightning_baseline_2_2_2`
4. Both use the same phase configuration
5. Both fail on guidance-related parameters

### Most Likely Root Cause
The `guidance_scale` or `guidance_phases` parameter value in the model defaults is a complex object (not a simple int/float) that doesn't support `.get()` method when accessed, or something internally tries to call `.get()` on a None object.

## Task Details

### Current Failure
- **Task ID**: 8172a7f6-65d9-4a77-8fa7-54558e356286 (travel_orchestrator)
- **Child Task**: 78cd8678-9fa3-4d06-a081-dd8fc90ff35f (travel_segment/segment_0)
- **Error**: Parameter resolution at LOOP [9/32] on guidance_scale
- **Timestamp**: 2026-02-24T14:31:53-54

### Previous Failure
- **Task ID**: e040ae48-4fa8-4af6-8a57-028277e54ba9 (travel_segment)
- **Error**: Parameter resolution at LOOP [26/32] on guidance_phases
- **Timestamp**: 2026-02-24T14:14:44-46

## Impact

- Orchestrator successfully enqueues 13 segments
- All segments fail during generation phase
- Cascading failure marks orchestrator as failed
- Root issue affects all travel segment generation with this model

## Recommendation

### Immediate (1-2 hours)
1. Add exception handler in param_resolution.py with detailed stack trace
2. Add defensive null checks for guidance_scale and guidance_phases
3. Deploy debug version to isolate exact error location

### Short-term (4-8 hours)
1. Fix root cause based on debug output
2. Add type validation for parameter values
3. Add test case with same model/parameters
4. Verify fix with reproducer

### Long-term
1. Add comprehensive parameter validation
2. Implement type checking for all model defaults
3. Add timeout guards for parameter resolution
4. Improve error messages for parameter processing

## How to Use These Files

1. **Start with**: TASK_FAILURE_SUMMARY.txt (quick overview)
2. **Deep dive**: INVESTIGATION_REPORT.md (full analysis)
3. **Implementation**: CODE_INVESTIGATION_LOCATIONS.md (exact lines to fix)
4. **Debugging**: VISUAL_COMPARISON.txt (see exact crash points)

## Testing the Fix

To verify the fix works:

1. Create test with model: `wan_2_2_i2v_lightning_baseline_2_2_2`
2. Use 2-phase configuration with guidance_scale=1, guidance_phases=2
3. Set parameters:
   - Resolution: 896x496
   - Video Length: 81 frames
   - Steps: 6
   - LoRAs: 3 entries
4. Run generation
5. Verify parameter resolution completes successfully

## Contact for Questions

See the detailed reports in this directory for comprehensive analysis.

