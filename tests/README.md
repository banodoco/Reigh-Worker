# Headless Test Suite for Reigh-Worker (Post-LTX2 Regression)

Comprehensive headless test suite (no GPU required) to verify that all existing services
(WAN T2V/I2V/VACE, Flux, Hunyuan, LTXv, Qwen, Z-Image, orchestrators) still work
correctly and that LTX2 itself is properly integrated.

All tests use `HEADLESS_WAN2GP_SMOKE=1` to skip GPU/model weights.

## Test Results

```
platform linux -- Python 3.12.3, pytest-9.0.2
collected 147 items

tests/test_service_health.py ............                   [ 8%]
tests/test_all_services_headless.py ....................    [53%]
tests/test_wan_headless.py ....................              [67%]
tests/test_ltx_headless.py ..................                [79%]
tests/test_task_conversion_headless.py ......               [100%]

147 passed in 3.95s
```

## Test Files

### 1. `test_service_health.py` - Structural Regression Gate (12 tests)

Pure import-based checks. No orchestrator, no smoke mode. Catches accidental removals.

| Test Class | Tests | What it verifies |
|---|---|---|
| `TestTaskTypeIntegrity` | `test_direct_queue_count` (==22), `test_wgp_task_count` (==20), `test_model_mapping_count` (>=24) | Set sizes haven't changed |
| `TestSetConsistency` | `test_wgp_minus_direct_is_inpaint_frames`, `test_direct_minus_wgp_is_qwen_extras`, `test_all_direct_types_have_model_mapping`, `test_no_direct_type_uses_fallback` | WGP/DIRECT sets are consistent |
| `TestHandlerCoverage` | `test_all_non_direct_types_have_handlers` (14 handler types), `test_no_handler_overlaps_direct_queue` | Dispatch covers all non-direct task types |
| `TestModelConfigHealth` | `test_all_config_files_valid_json` (154 JSON files), `test_ltx2_config_not_corrupted`, `test_ltxv_config_exists` | Config files parseable |

```
tests/test_service_health.py::TestTaskTypeIntegrity::test_direct_queue_count PASSED
tests/test_service_health.py::TestTaskTypeIntegrity::test_wgp_task_count PASSED
tests/test_service_health.py::TestTaskTypeIntegrity::test_model_mapping_count PASSED
tests/test_service_health.py::TestSetConsistency::test_wgp_minus_direct_is_inpaint_frames PASSED
tests/test_service_health.py::TestSetConsistency::test_direct_minus_wgp_is_qwen_extras PASSED
tests/test_service_health.py::TestSetConsistency::test_all_direct_types_have_model_mapping PASSED
tests/test_service_health.py::TestSetConsistency::test_no_direct_type_uses_fallback PASSED
tests/test_service_health.py::TestHandlerCoverage::test_all_non_direct_types_have_handlers PASSED
tests/test_service_health.py::TestHandlerCoverage::test_no_handler_overlaps_direct_queue PASSED
tests/test_service_health.py::TestModelConfigHealth::test_all_config_files_valid_json PASSED
tests/test_service_health.py::TestModelConfigHealth::test_ltx2_config_not_corrupted PASSED
tests/test_service_health.py::TestModelConfigHealth::test_ltxv_config_exists PASSED
```

---

### 2. `test_all_services_headless.py` - Registration & Model Detection (67 tests)

| Test Class | Tests | What it verifies |
|---|---|---|
| `TestTaskTypeRegistration` | `test_all_wgp_types_present`, `test_all_direct_queue_types_present`, `test_ltx2_in_both_sets`, `test_ltxv_in_both_sets` | Exact membership of frozen sets |
| `TestModelMappings` | `@parametrize` over all 24 entries + `test_fallback_for_unknown_type` | Every task type -> correct model name |
| `TestModelDetectionMethods` | 6 model families: t2v, i2v, vace, flux, ltx2, qwen | `_is_*()` methods return correct True/False |
| `TestHelperFunctions` | `@parametrize` `is_wgp_task()` and `is_direct_queue_task()` for all known types + negative cases | Helper functions consistent with sets |

```
tests/test_all_services_headless.py::TestTaskTypeRegistration::test_all_wgp_types_present PASSED
tests/test_all_services_headless.py::TestTaskTypeRegistration::test_all_direct_queue_types_present PASSED
tests/test_all_services_headless.py::TestTaskTypeRegistration::test_ltx2_in_both_sets PASSED
tests/test_all_services_headless.py::TestTaskTypeRegistration::test_ltxv_in_both_sets PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[generate_video-t2v] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[vace-vace_14B_cocktail_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[vace_21-vace_14B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[vace_22-vace_14B_cocktail_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[wan_2_2_t2i-t2v_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[t2v-t2v] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[t2v_22-t2v_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[flux-flux] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[i2v-i2v_14B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[i2v_22-i2v_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[hunyuan-hunyuan] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[ltxv-ltxv_13B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[ltx2-ltx2_19B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[join_clips_segment-wan_2_2_vace_lightning_baseline_2_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[inpaint_frames-wan_2_2_vace_lightning_baseline_2_2_2] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[qwen_image_edit-qwen_image_edit_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[qwen_image_hires-qwen_image_edit_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[qwen_image_style-qwen_image_edit_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[image_inpaint-qwen_image_edit_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[annotated_image_edit-qwen_image_edit_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[qwen_image-qwen_image_edit_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[qwen_image_2512-qwen_image_2512_20B] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[z_image_turbo-z_image] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_model_mapping[z_image_turbo_i2i-z_image_img2img] PASSED
tests/test_all_services_headless.py::TestModelMappings::test_fallback_for_unknown_type PASSED
tests/test_all_services_headless.py::TestModelDetectionMethods::test_t2v_detection PASSED
tests/test_all_services_headless.py::TestModelDetectionMethods::test_i2v_detection PASSED
tests/test_all_services_headless.py::TestModelDetectionMethods::test_vace_detection PASSED
tests/test_all_services_headless.py::TestModelDetectionMethods::test_flux_detection PASSED
tests/test_all_services_headless.py::TestModelDetectionMethods::test_ltx2_detection PASSED
tests/test_all_services_headless.py::TestModelDetectionMethods::test_qwen_detection PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[t2v] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[i2v] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[vace] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[flux] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[hunyuan] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[ltxv] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[ltx2] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[generate_video] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[z_image_turbo] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_positive[inpaint_frames] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_negative[travel_orchestrator] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_negative[travel_segment] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_negative[magic_edit] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_negative[nonexistent] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_wgp_task_negative[join_clips_orchestrator] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[t2v] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[i2v] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[vace] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[flux] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[hunyuan] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[ltxv] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[ltx2] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[qwen_image_edit] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[qwen_image_hires] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[qwen_image] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[z_image_turbo] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_positive[z_image_turbo_i2i] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_negative[travel_orchestrator] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_negative[travel_segment] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_negative[magic_edit] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_negative[nonexistent] PASSED
tests/test_all_services_headless.py::TestHelperFunctions::test_is_direct_queue_task_negative[join_clips_orchestrator] PASSED
```

---

### 3. `test_wan_headless.py` - WAN Core Services Smoke Tests (20 tests)

The most important new file. WAN services previously had zero headless test coverage.

| Test Class | Tests | What it verifies |
|---|---|---|
| `TestT2VSmoke` | `test_t2v_model_loads`, `test_t2v_smoke_generation`, `test_t2v_with_parameters`, `test_t2v_22_model_loads` | T2V still works after LTX2 |
| `TestI2VSmoke` | `test_i2v_model_loads`, `test_i2v_with_start_image`, `test_i2v_with_both_images` | I2V image inputs still work |
| `TestVACESmoke` | `test_vace_model_loads`, `test_vace_smoke_generation`, `test_vace_cocktail_loads` | VACE still works |
| `TestFluxSmoke` | `test_flux_model_loads`, `test_flux_smoke_generation`, `test_flux_is_not_ltx2` | Flux still works |
| `TestHunyuanSmoke` | `test_hunyuan_model_loads`, `test_hunyuan_smoke_generation` | Hunyuan still works |
| `TestModelSwitching` | `test_full_rotation` (t2v->i2v->vace->ltx2->flux->t2v), `test_reload_same_is_noop` | Model switching intact |
| `TestWANParameterPassthrough` | `test_t2v_standard_params`, `test_i2v_image_params`, `test_vace_control_params` | Parameters flow through generate() |

```
tests/test_wan_headless.py::TestT2VSmoke::test_t2v_model_loads PASSED
tests/test_wan_headless.py::TestT2VSmoke::test_t2v_smoke_generation PASSED
tests/test_wan_headless.py::TestT2VSmoke::test_t2v_with_parameters PASSED
tests/test_wan_headless.py::TestT2VSmoke::test_t2v_22_model_loads PASSED
tests/test_wan_headless.py::TestI2VSmoke::test_i2v_model_loads PASSED
tests/test_wan_headless.py::TestI2VSmoke::test_i2v_with_start_image PASSED
tests/test_wan_headless.py::TestI2VSmoke::test_i2v_with_both_images PASSED
tests/test_wan_headless.py::TestVACESmoke::test_vace_model_loads PASSED
tests/test_wan_headless.py::TestVACESmoke::test_vace_smoke_generation PASSED
tests/test_wan_headless.py::TestVACESmoke::test_vace_cocktail_loads PASSED
tests/test_wan_headless.py::TestFluxSmoke::test_flux_model_loads PASSED
tests/test_wan_headless.py::TestFluxSmoke::test_flux_smoke_generation PASSED
tests/test_wan_headless.py::TestFluxSmoke::test_flux_is_not_ltx2 PASSED
tests/test_wan_headless.py::TestHunyuanSmoke::test_hunyuan_model_loads PASSED
tests/test_wan_headless.py::TestHunyuanSmoke::test_hunyuan_smoke_generation PASSED
tests/test_wan_headless.py::TestModelSwitching::test_full_rotation PASSED
tests/test_wan_headless.py::TestModelSwitching::test_reload_same_is_noop PASSED
tests/test_wan_headless.py::TestWANParameterPassthrough::test_t2v_standard_params PASSED
tests/test_wan_headless.py::TestWANParameterPassthrough::test_i2v_image_params PASSED
tests/test_wan_headless.py::TestWANParameterPassthrough::test_vace_control_params PASSED
```

---

### 4. `test_ltx_headless.py` - LTX Family (LTXv + Cross-Model) (18 tests)

Expands LTX coverage beyond the existing `test_ltx2_headless.py` to include LTXv and cross-model switching.

| Test Class | Tests | What it verifies |
|---|---|---|
| `TestLTXvModelDetection` | `test_ltxv_detected_as_t2v`, `test_ltxv_not_ltx2`, `test_ltxv_not_flux`, `test_ltxv_not_vace` | LTXv detection distinct from LTX2 |
| `TestLTXvTaskTypes` | `test_ltxv_in_wgp`, `test_ltxv_in_direct_queue`, `test_ltxv_model_mapping`, `test_ltxv_config_exists` | LTXv registration intact |
| `TestLTXvSmoke` | `test_ltxv_smoke_generation`, `test_ltxv_with_parameters` | LTXv generates in smoke mode |
| `TestCrossModelSwitching` | `test_ltx2_to_ltxv`, `test_ltxv_to_t2v`, `test_ltxv_to_ltx2`, `test_full_ltx_family_rotation` | Switching between LTX variants |
| `TestLTXModelConfigs` | `test_ltx2_config_valid`, `test_ltxv_config_valid`, `test_ltx2_handler_exists`, `test_ltx2_model_dir_exists` | LTX filesystem artifacts intact |

```
tests/test_ltx_headless.py::TestLTXvModelDetection::test_ltxv_detected_as_t2v PASSED
tests/test_ltx_headless.py::TestLTXvModelDetection::test_ltxv_not_ltx2 PASSED
tests/test_ltx_headless.py::TestLTXvModelDetection::test_ltxv_not_flux PASSED
tests/test_ltx_headless.py::TestLTXvModelDetection::test_ltxv_not_vace PASSED
tests/test_ltx_headless.py::TestLTXvTaskTypes::test_ltxv_in_wgp PASSED
tests/test_ltx_headless.py::TestLTXvTaskTypes::test_ltxv_in_direct_queue PASSED
tests/test_ltx_headless.py::TestLTXvTaskTypes::test_ltxv_model_mapping PASSED
tests/test_ltx_headless.py::TestLTXvTaskTypes::test_ltxv_config_exists PASSED
tests/test_ltx_headless.py::TestLTXvSmoke::test_ltxv_smoke_generation PASSED
tests/test_ltx_headless.py::TestLTXvSmoke::test_ltxv_with_parameters PASSED
tests/test_ltx_headless.py::TestCrossModelSwitching::test_ltx2_to_ltxv PASSED
tests/test_ltx_headless.py::TestCrossModelSwitching::test_ltxv_to_t2v PASSED
tests/test_ltx_headless.py::TestCrossModelSwitching::test_ltxv_to_ltx2 PASSED
tests/test_ltx_headless.py::TestCrossModelSwitching::test_full_ltx_family_rotation PASSED
tests/test_ltx_headless.py::TestLTXModelConfigs::test_ltx2_config_valid PASSED
tests/test_ltx_headless.py::TestLTXModelConfigs::test_ltxv_config_valid PASSED
tests/test_ltx_headless.py::TestLTXModelConfigs::test_ltx2_handler_exists PASSED
tests/test_ltx_headless.py::TestLTXModelConfigs::test_ltx2_model_dir_exists PASSED
```

---

### 5. `test_task_conversion_headless.py` - DB->GenerationTask Pipeline (30 tests)

Tests the critical `db_task_to_generation_task()` and `parse_phase_config()` functions.

| Test Class | Tests | What it verifies |
|---|---|---|
| `TestBasicConversion` | One test per core task type: t2v, ltx2, ltxv, i2v, vace, flux, hunyuan, z_image_turbo | Each type produces correct model + params |
| `TestModelOverride` | `test_explicit_model_overrides_default`, `test_missing_model_uses_default` | Model selection priority |
| `TestParameterPassthrough` | `test_whitelisted_params_pass_through`, `test_steps_alias_maps_to_num_inference_steps`, `test_non_whitelisted_params_filtered` | Parameter mapping correct |
| `TestDefaults` | `test_seed_defaults`, `test_negative_prompt_defaults`, `test_empty_prompt_raises_for_non_img2img`, `test_empty_prompt_allowed_for_img2img` | Default values applied |
| `TestPhaseConfigIntegration` | `test_phase_config_sets_steps`, `test_phase_config_sets_guidance_phases`, `test_phase_config_sets_flow_shift`, `test_phase_config_with_loras`, `test_invalid_phase_count_raises` | Phase config parsing into generation params |
| `TestParsePhaseConfig` | `test_basic_2_phase`, `test_basic_3_phase`, `test_unipc_solver`, `test_dpm_solver`, `test_steps_mismatch_raises`, `test_lora_deduplication` | Direct parse_phase_config tests |
| `TestOrchestratorPriority` | `test_normal_task_default_priority`, `test_explicit_priority_passed_through` | Priority handling |

```
tests/test_task_conversion_headless.py::TestBasicConversion::test_t2v_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_ltx2_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_ltxv_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_i2v_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_vace_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_flux_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_hunyuan_conversion PASSED
tests/test_task_conversion_headless.py::TestBasicConversion::test_z_image_turbo_conversion PASSED
tests/test_task_conversion_headless.py::TestModelOverride::test_explicit_model_overrides_default PASSED
tests/test_task_conversion_headless.py::TestModelOverride::test_missing_model_uses_default PASSED
tests/test_task_conversion_headless.py::TestParameterPassthrough::test_whitelisted_params_pass_through PASSED
tests/test_task_conversion_headless.py::TestParameterPassthrough::test_steps_alias_maps_to_num_inference_steps PASSED
tests/test_task_conversion_headless.py::TestParameterPassthrough::test_non_whitelisted_params_filtered PASSED
tests/test_task_conversion_headless.py::TestDefaults::test_seed_defaults PASSED
tests/test_task_conversion_headless.py::TestDefaults::test_negative_prompt_defaults PASSED
tests/test_task_conversion_headless.py::TestDefaults::test_empty_prompt_raises_for_non_img2img PASSED
tests/test_task_conversion_headless.py::TestDefaults::test_empty_prompt_allowed_for_img2img PASSED
tests/test_task_conversion_headless.py::TestPhaseConfigIntegration::test_phase_config_sets_steps PASSED
tests/test_task_conversion_headless.py::TestPhaseConfigIntegration::test_phase_config_sets_guidance_phases PASSED
tests/test_task_conversion_headless.py::TestPhaseConfigIntegration::test_phase_config_sets_flow_shift PASSED
tests/test_task_conversion_headless.py::TestPhaseConfigIntegration::test_phase_config_with_loras PASSED
tests/test_task_conversion_headless.py::TestPhaseConfigIntegration::test_invalid_phase_count_raises PASSED
tests/test_task_conversion_headless.py::TestParsePhaseConfig::test_basic_2_phase PASSED
tests/test_task_conversion_headless.py::TestParsePhaseConfig::test_basic_3_phase PASSED
tests/test_task_conversion_headless.py::TestParsePhaseConfig::test_unipc_solver PASSED
tests/test_task_conversion_headless.py::TestParsePhaseConfig::test_dpm_solver PASSED
tests/test_task_conversion_headless.py::TestParsePhaseConfig::test_steps_mismatch_raises PASSED
tests/test_task_conversion_headless.py::TestParsePhaseConfig::test_lora_deduplication PASSED
tests/test_task_conversion_headless.py::TestOrchestratorPriority::test_normal_task_default_priority PASSED
tests/test_task_conversion_headless.py::TestOrchestratorPriority::test_explicit_priority_passed_through PASSED
```

---

## How to Run

Run all headless tests:
```bash
python -m pytest tests/test_service_health.py tests/test_all_services_headless.py tests/test_wan_headless.py tests/test_ltx_headless.py tests/test_task_conversion_headless.py -v --tb=short
```

Run individually:
```bash
python -m pytest tests/test_service_health.py -v           # Fastest - pure imports
python -m pytest tests/test_wan_headless.py -v              # WAN regression
python -m pytest tests/test_ltx_headless.py -v              # LTX family
python -m pytest tests/test_all_services_headless.py -v     # Full registration check
python -m pytest tests/test_task_conversion_headless.py -v  # Conversion pipeline
```

## Existing Tests (Pre-LTX2)

| File | Purpose |
|---|---|
| `test_ltx2_headless.py` | LTX-2 specific: model detection, image/audio bridge, parameter wiring, travel toggle |
| `test_travel_between_images.py` | Multi-image video transition pipeline |
| `test_lora_flow.py` | LoRA URL resolution and download flow |
| `test_travel_real_gpu.py` | GPU-dependent travel tests (skipped without GPU) |

## Key Source Files Referenced

| File | Purpose |
|---|---|
| `source/task_types.py` | All task type sets, model mappings, helper functions |
| `headless_wgp.py` | WanOrchestrator (smoke mode, model detection, generate) |
| `source/task_registry.py` | TaskRegistry.dispatch(), handler routing dict |
| `source/task_conversion.py` | db_task_to_generation_task(), parse_phase_config() |
| `headless_model_management.py` | GenerationTask, TaskStatus dataclasses |
| `Wan2GP/defaults/*.json` | Model config files (154 total) |
