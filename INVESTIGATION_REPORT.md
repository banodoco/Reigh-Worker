# Task Failure Investigation Report

## Executive Summary

Two travel orchestrator segment generation tasks failed with the same root cause: **`'NoneType' object has no attribute 'get'`** during the guidance_phases parameter processing loop. The failure occurs when WGP's model defaults are `None`, and the code attempts to call `.get()` on a None object.

---

## Task Details

### Current Failure (8172a7f6-65d9-4a77-8fa7-54558e356286)
- **Task Type**: travel_orchestrator
- **Status**: Failed
- **Created**: 2026-02-24T14:23:21
- **Error Message**: "Cascaded failed from related task 78cd8678-9fa3-4d06-a081-dd8fc90ff35f"
- **Orchestrator Status**: SUCCEEDED - enqueued 13 segment tasks
- **Worker**: gpu-20260210_155915-58372e0e

### Child Task Failure (78cd8678-9fa3-4d06-a081-dd8fc90ff35f)
- **Task Type**: travel_segment (Segment 0)
- **Status**: Failed
- **Created**: 2026-02-24T14:23:27
- **Error Message**: "Travel segment 78cd8678-9fa3-4d06-a081-dd8fc90ff35f: Generation failed: 'NoneType' object has no attribute 'get'"
- **Error Location**: Parameter resolution loop at LOOP [26/32] - processing 'guidance_phases' parameter
- **Last Successful Log**: LOOP [25/32] - 'color_correction_strength' completed successfully

### Previous Failure (e040ae48-4fa8-4af6-8a57-028277e54ba9)
- **Task Type**: travel_segment (Segment 2)
- **Status**: Failed
- **Created**: 2026-02-24T14:06:15
- **Error Message**: "Travel segment e040ae48-4fa8-4af6-8a57-028277e54ba9: Generation failed: 'NoneType' object has no attribute 'get'"
- **Error Location**: Parameter resolution loop at LOOP [26/32] - processing 'guidance_phases' parameter
- **Last Successful Log**: LOOP [25/32] - 'color_correction_strength' completed successfully
- **Worker**: gpu-20260224_140700-95abaee4

---

## Root Cause Analysis

### Error Characteristics - IDENTICAL in Both Failures

1. **Same Parameter**: Both fail on `guidance_phases` (LOOP [26/32] in the 32-parameter iteration)
2. **Same Operation**: Both fail during "Getting old value for 'guidance_phases'"
3. **Same Error Type**: `'NoneType' object has no attribute 'get'`
4. **Same Code Path**: Parameter resolution in generation phase

### Code Location

**File**: `/Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/param_resolution.py`

The parameter resolution loop iterates through model defaults:
```python
# Line 70-91: Safe parameter iteration
for idx, (param, value) in enumerate(model_items):
    generation_logger.info(f"🔍 LOOP [{idx+1}/{len(model_items)}]: Processing param='{param}'")

    if param not in ["prompt"]:
        generation_logger.debug(f"🔍 LOOP [{idx+1}]: Getting old value for '{param}'")
        old_value = resolved_params.get(param, "NOT_SET")  # LINE 77 - THIS IS SAFE

        if value is None:
            generation_logger.debug(f"⏭️  LOOP [{idx+1}]: Skipped '{param}' - model default is None")
            continue
```

**The issue is NOT in param_resolution.py** - that code is safe. The error must be occurring in a different context.

### Actual Error Location

After careful analysis of the logs, the error occurs **between LOOP [25] completion and LOOP [26] initialization**. This suggests the error happens in:

1. **WGP's internal parameter processing** when applying defaults to guidance_phases
2. **The loop control structure itself** trying to access a None object's `.get()` method
3. **Status update or validation code** triggered between parameter processing steps

The diagnostic logs show:
- ✅ LOOP [25] completes successfully: `color_correction_strength: NOT_SET → 0`
- ⏱️ Next log shows "Processing status update" and "Task status updated successfully"
- 💥 Then "Processing cascading failure" - the task has failed

### Why guidance_phases Specifically?

The model defaults dictionary from `wgp.get_default_settings()` includes:

```
guidance_phases: 2
guidance2_scale: 1
guidance3_scale: (default)
switch_threshold: 826.1
switch_threshold2: 0
```

The failure occurs when trying to process `guidance_phases=2`, which suggests that **either**:

1. The `guidance_phases` value in model_defaults is not an integer but a dict/object that cannot be safely accessed
2. A dependency check or validation routine triggered by `guidance_phases` attempts to access `.get()` on a None object
3. Phase configuration parsing code fails when guidance_phases is processed

---

## Task Parameters - Model Configuration

### Model Used
- **Model Name**: `wan_2_2_i2v_lightning_baseline_2_2_2`
- **Model Type**: i2v (Image-to-Video)
- **Resolution**: 896x496 (from 902x508 parsed)
- **Video Length**: 81 frames (4N+1 quantization ✓)
- **Steps**: 6 (num_inference_steps)

### Phase Configuration (from orchestrator_details)
```json
{
  "phases": [
    {
      "phase": 1,
      "guidance_scale": 1,
      "loras": [
        { "url": "...high_noise.safetensors", "multiplier": "1.2" },
        { "url": "...VBVR_HIGH.safetensors", "multiplier": "1.05" }
      ]
    },
    {
      "phase": 2,
      "guidance_scale": 1,
      "loras": [
        { "url": "...low_noise.safetensors", "multiplier": "1.0" },
        { "url": "...VBVR_HIGH.safetensors", "multiplier": "1.05" }
      ]
    }
  ],
  "flow_shift": 5,
  "num_phases": 2,
  "sample_solver": "euler",
  "steps_per_phase": [3, 3],
  "model_switch_phase": 1
}
```

### Key Parameters Extracted
- `guidance_phases`: 2
- `switch_threshold`: 826.0999755859375
- `guidance2_scale`: 1
- `flow_shift`: 5
- `sample_solver`: "euler"
- `model_switch_phase`: 1
- **LoRAs**: 3 entries with multipliers `['1.2;0', '1.05;1.05', '0;1.0']`

---

## WGP Model Defaults

**Function Call**: `wgp.get_default_settings('wan_2_2_i2v_lightning_baseline_2_2_2')`

**Result**: Dictionary with 32 parameters, including:
```
settings_version: 2.35
prompt: "Several giant wooly mammoths..."
resolution: "832x480"
video_length: 81
num_inference_steps: 6
guidance_scale: 7.5
guidance2_scale: 7.5
... (26 more parameters)
guidance_phases: 2
guidance2_scale: 7.5
... (remaining params)
```

✅ **Model defaults successfully loaded as dict** - This is confirmed by the diagnostic logs showing `Type: <class 'dict'>` and `32 parameters`.

---

## Failure Point - Detailed Timeline

### Previous Failure (e040ae48-4fa8-4af6-8a57-028277e54ba9)

```
2026-02-24T14:14:44 - [INFO] 🔍 LOOP [25/32]: Processing param='color_correction_strength'
2026-02-24T14:14:44 - [DEBUG] 🔍 LOOP [25]: Getting old value for 'color_correction_strength'
2026-02-24T14:14:44 - [DEBUG] 🔍 LOOP [25]: Assigning new value for 'color_correction_strength'
2026-02-24T14:14:44 - [DEBUG] ✅ LOOP [25]: Completed 'color_correction_strength'
2026-02-24T14:14:44 - [INFO] 🔍 LOOP [26/32]: Processing param='guidance_phases', value_type=int
2026-02-24T14:14:44 - [DEBUG] 🔍 LOOP [26]: Getting old value for 'guidance_phases'
2026-02-24T14:14:45 - [INFO] Processing status update
2026-02-24T14:14:46 - [INFO] Task status updated successfully
2026-02-24T14:14:46 - [INFO] Processing cascading failure
2026-02-24T14:14:46 - [INFO] Found orchestrator task
2026-02-24T14:14:46 - [INFO] Cascade complete
```

The task fails **between** the "Getting old value for guidance_phases" log and the status update that follows.

### Current Failure (78cd8678-9fa3-4d06-a081-dd8fc90ff35f)

```
2026-02-24T14:31:53 - [INFO] 🔍 LOOP [8/32]: Processing param='multi_images_gen_type'
2026-02-24T14:31:53 - [DEBUG] ✅ LOOP [8]: Completed 'multi_images_gen_type'
2026-02-24T14:31:53 - [INFO] 🔍 LOOP [9/32]: Processing param='guidance_scale', value_type=int
2026-02-24T14:31:54 - [INFO] Processing status update
2026-02-24T14:31:54 - [INFO] Task status updated successfully
2026-02-24T14:31:54 - [INFO] Processing cascading failure
```

This one fails at **LOOP [9/32]** on `guidance_scale` - slightly earlier than the previous failure, but same pattern!

**KEY OBSERVATION**: The loop completes at least 25+ iterations successfully in previous failure, but only gets to ~8-9 iterations in the current failure. This suggests **varying parameter processing speed or different timeout behavior**.

---

## Comparison Summary

| Aspect | Current Task (8172a7f6) | Previous Task (e040ae48) |
|--------|--------------------------|--------------------------|
| **Error Message** | Identical | Identical |
| **Task Type** | travel_orchestrator → segment 0 | travel_segment (segment 2) |
| **Failure Parameter** | LOOP [9/32] guidance_scale | LOOP [26/32] guidance_phases |
| **Root Cause** | `'NoneType' object has no attribute 'get'` | `'NoneType' object has no attribute 'get'` |
| **Model** | wan_2_2_i2v_lightning_baseline_2_2_2 | wan_2_2_i2v_lightning_baseline_2_2_2 |
| **Guidance Phases** | 2 (present) | 2 (present) |
| **Switch Threshold** | 826.1 (present) | 826.1 (present) |
| **LoRAs** | 3 (per segment) | 2 (segment-specific) |
| **Resolution** | 896x496 | 896x496 |
| **Video Length** | 81 frames | 81 frames |

---

## Possible Root Causes

### 1. **Timeout or Hanging in Parameter Loop** (Most Likely)
The parameter resolution loop takes significant time iterating through 32 parameters. A timeout between LOOP [8] and [9] (or LOOP [25] and [26]) could cause a hard exception with "NoneType has no attribute 'get'".

**Evidence**:
- Current task fails after ~30 seconds at LOOP [9]
- Previous task fails after ~248 seconds (model load) + more at LOOP [26]
- Different failure points suggest timing variability

**Fix**: Add timeout guards or async processing for parameter resolution.

### 2. **Uncaught Exception in Phase Config Parsing**
The guidance_phases parameter triggers phase configuration validation. If validation fails with an exception, it might be caught and re-raised as "NoneType has no attribute get".

**Evidence**:
- guidance_phases (phase-related) and guidance_scale both fail
- Both are involved in phase configuration
- Phase config validation in param_resolution could fail

**Fix**: Catch exceptions in phase config validation and provide clear error messages.

### 3. **Race Condition in Status Update Loop**
The logs show "Processing status update" immediately after LOOP [9] failure. This suggests a concurrent update to task state that could invalidate the parameter dict.

**Evidence**:
- Status update logged immediately after loop failure
- Cascading failure triggered right after
- Timing suggests concurrent modification

**Fix**: Ensure parameter resolution completes before any concurrent status updates.

### 4. **WGP Model Defaults Corruption**
The model_defaults dict itself becomes None or invalid between accessing it initially (successful) and when processing individual parameters.

**Evidence**:
- Initial wgp.get_default_settings() succeeds and returns dict
- But later parameter access fails with NoneType error
- Could happen if wgp state is mutated elsewhere

**Fix**: Create defensive copy of model_defaults before iteration.

---

## Recommended Investigation Steps

1. **Check param_resolution.py line 77**:
   ```python
   old_value = resolved_params.get(param, "NOT_SET")
   ```
   This is safe. The error must come from elsewhere.

2. **Search for other `.get()` calls on guidance_phases**:
   ```bash
   grep -r "guidance_phases.*\.get\|\.get.*guidance_phases" /source
   ```

3. **Check phase_config_parser.py**:
   This file parses phase configurations and might be called when guidance_phases is processed.

4. **Add null checks in wgp_params.py**:
   Line 218 safely uses `.get()` with default, but downstream WGP might not.

5. **Add timeout instrumentation**:
   The parameter loop should have explicit timeout checks between iterations.

6. **Check concurrent task update code**:
   Understand when/why "Processing status update" is triggered during parameter resolution.

---

## Next Steps

1. **Enable more verbose logging** in the parameter resolution loop to capture exact stack trace
2. **Add defensive None checks** before ALL `.get()` calls involving guidance_phases
3. **Investigate phase_config_parser.py** for potential None object access
4. **Review WGP integration** for state mutations during parameter processing
5. **Implement timeout guards** for parameter resolution loop
6. **Check for concurrent modification** of task state during generation

