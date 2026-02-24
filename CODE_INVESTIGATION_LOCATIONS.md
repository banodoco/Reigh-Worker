# Code Investigation Locations

## Primary Investigation Files

### 1. `/source/models/wgp/param_resolution.py` (Lines 70-95)
**Status**: SAFE but verify
**Reason**: Main parameter loop, but safe .get() usage detected

```python
# Line 70-91: Loop through model defaults
for idx, (param, value) in enumerate(model_items):
    generation_logger.info(f"🔍 LOOP [{idx+1}/{len(model_items)}]: Processing param='{param}'")

    if param not in ["prompt"]:
        generation_logger.debug(f"🔍 LOOP [{idx+1}]: Getting old value for '{param}'")
        old_value = resolved_params.get(param, "NOT_SET")  # ← SAFE

        # This continues but error happens between logs
```

**Potential Issue**: The error occurs BETWEEN the debug log and the next statement. Check if:
- safe_log_change() function has issues
- Logging infrastructure hangs
- An exception is raised before .get() completes

**Action**: Add exception handler around the entire loop iteration

---

### 2. `/source/core/params/phase_config_parser.py`
**Status**: SUSPECT - needs review for .get() on None objects

**Why**:
- Phase config is parsed when guidance_phases is encountered
- Both failing tasks involve phase-related parameters
- May call .get() on phase dict that could be None

**Action**: Search for all `.get()` calls and verify they have null checks

```bash
grep -n "\.get(" /Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/phase_config_parser.py
```

---

### 3. `/source/models/wgp/generators/wgp_params.py` (Lines 213-220)
**Status**: Safe but verify downstream

```python
# Line 218: Safe extraction with default
guidance_phases_value = resolved_params.get("guidance_phases", 1)

# But this is used at:
# Line 259: In wgp_params dict
'guidance_phases': guidance_phases_value,
```

**Potential Issue**: Something downstream tries to use guidance_phases_value and fails

**Action**: Add validation that guidance_phases_value is not None

---

### 4. `/source/core/log/headless_logger.py` (and related logging)
**Status**: SUSPECT - check safe_log_change()

**Why**: The error might be in logging infrastructure itself

```python
# Line 89 in param_resolution.py calls:
generation_logger.debug(safe_log_change(param, old_value, value))
```

**Action**: Check if safe_log_change() can return None or cause exception

```bash
grep -n "def safe_log_change" /Users/peteromalley/Documents/Headless-Wan2GP/source -r
```

---

### 5. Task Status Update Code
**Status**: SUSPECT - timing issue

**Why**: Logs show "Processing status update" immediately after loop failure

**Lines to Check**:
- Where task status is updated during generation
- Concurrent modification of task state
- Lock/synchronization around parameter resolution

**Action**: Ensure parameter resolution is atomic and not interrupted by status updates

```bash
grep -n "Processing status update" /Users/peteromalley/Documents/Headless-Wan2GP/source -r
```

---

## Secondary Investigation Areas

### Phase Configuration Parsing
**File**: `/source/core/params/phase_config_parser.py`
**Issue**: Phase validation triggered by guidance_phases

**Search for**:
```bash
grep -n "guidance_phases\|phase.*\.get\|\.get.*phase" \
  /Users/peteromalley/Documents/Headless-Wan2GP/source/core/params/phase_config_parser.py
```

### WGP Integration State
**File**: `/source/models/wgp/orchestrator.py`
**Issue**: WGP state mutation during parameter processing

**Search for**:
```bash
grep -n "guidance_phases\|phase_config" \
  /Users/peteromalley/Documents/Headless-Wan2GP/source/models/wgp/orchestrator.py
```

### Task Execution Queue
**File**: `/source/task_handlers/worker/task_processor.py` or similar
**Issue**: Concurrent task update during generation

**Search for**:
```bash
grep -n "Processing status update\|cascading failure" \
  /Users/peteromalley/Documents/Headless-Wan2GP/source -r
```

---

## Error Reproduction Setup

To reproduce the error:

1. **Use Model**: `wan_2_2_i2v_lightning_baseline_2_2_2`
2. **Use Phase Config**: 2 phases with guidance_scale=1
3. **Set Parameters**:
   - guidance_phases: 2
   - guidance_scale: 1
   - guidance2_scale: 1
   - flow_shift: 5
   - sample_solver: "euler"
   - model_switch_phase: 1
   - switch_threshold: 826.1
4. **Set Video Params**:
   - Resolution: 896x496
   - Video Length: 81 frames
   - Steps: 6
   - LoRAs: 3 with multipliers

5. **Trigger**: Run travel segment generation with these exact parameters

---

## Expected Fix Locations

### Fix #1: Add Exception Handler in Parameter Loop
**File**: `/source/models/wgp/param_resolution.py`
**Lines**: 70-91

```python
try:
    for idx, (param, value) in enumerate(model_items):
        # ... existing code ...
except Exception as e:
    generation_logger.error(f"Parameter resolution failed at param '{param}': {e}")
    generation_logger.error(f"Full traceback: {traceback.format_exc()}")
    raise
```

### Fix #2: Add Null Checks for guidance_phases
**File**: `/source/models/wgp/generators/wgp_params.py`
**Lines**: 218-260

```python
guidance_phases_value = resolved_params.get("guidance_phases", 1)
if guidance_phases_value is None:
    guidance_logger.warning("guidance_phases is None, using default=1")
    guidance_phases_value = 1

# ... then use guidance_phases_value ...
'guidance_phases': guidance_phases_value,
```

### Fix #3: Verify Phase Config Parsing
**File**: `/source/core/params/phase_config_parser.py`

Add null checks around all `.get()` calls:

```python
# Before: phases_config.get(...)
# After: phases_config.get(...) if phases_config else ...

if phases_config:
    guidance_scales = [p.get("guidance_scale", 1.0) for p in phases_config if p]
else:
    guidance_scales = [1.0]
```

### Fix #4: Add Defensive Copy of Model Defaults
**File**: `/source/models/wgp/param_resolution.py`
**Lines**: 49-68

```python
if model_defaults:
    # Create defensive copy to prevent mutation during iteration
    model_items = list(model_defaults.items())  # Already done - line 67

    # Add validation
    if model_items is None:
        generation_logger.error("model_items snapshot failed unexpectedly")
        # Fall back to system defaults only
        model_items = []
```

---

## Verification Checklist

- [ ] Verify resolved_params is never None during loop
- [ ] Verify model_items snapshot is valid
- [ ] Verify guidance_phases value type and content
- [ ] Verify no concurrent task state mutations
- [ ] Verify logging infrastructure doesn't fail
- [ ] Verify phase config parser handles None phases
- [ ] Add timeout guards for parameter resolution
- [ ] Test with reproducer: wan_2_2_i2v_lightning_baseline_2_2_2 + 2-phase config

---

## Debug Logging to Add

Add these debug statements to isolate the exact failure point:

```python
# In param_resolution.py, line 76 (before getting old value)
generation_logger.debug(f"[SAFETY_CHECK] idx={idx}, param={param}, resolved_params type: {type(resolved_params)}")
generation_logger.debug(f"[SAFETY_CHECK] About to call resolved_params.get('{param}', 'NOT_SET')")

# After getting old value
generation_logger.debug(f"[SAFETY_CHECK] Successfully got old_value: {old_value}")

# Before the next log statement
generation_logger.debug(f"[SAFETY_CHECK] param={param} iteration completed successfully")
```

Then run the same test and capture the logs to see exactly where it fails.

---

## Timeline to Fix

1. **Immediate (1 hour)**: Add exception handler and debug logging to isolate exact line
2. **Short-term (4 hours)**: Fix root cause based on debug output
3. **Validation (2 hours)**: Test with reproducer and verify fix
4. **Deployment (1 hour)**: Deploy and monitor for resolution

Total estimated time: 8 hours

