# PR #19 Custom Preservation Matrix

| Area | Artifact | Required | Ported | Validated | Notes |
|---|---|---|---|---|---|
| Uni3C module | `Wan2GP/models/wan/uni3c/__init__.py` | yes | yes | partial | Restored from custom branch lineage; syntax verified |
| Uni3C module | `Wan2GP/models/wan/uni3c/controlnet.py` | yes | yes | partial | Restored from custom branch lineage; syntax verified |
| Uni3C module | `Wan2GP/models/wan/uni3c/load.py` | yes | yes | partial | Restored from custom branch lineage; syntax verified |
| Uni3C in pipeline | `Wan2GP/models/wan/any2video.py` helper methods + kwargs wiring | yes | yes | partial | `_load_uni3c_guide_video`, `_apply_uni3c_frame_policy`, `_encode_uni3c_guide` restored; compile check passed |
| Uni3C in model forward | `Wan2GP/models/wan/modules/model.py` | yes | yes | partial | ControlNet states + per-block residual injection restored; compile check passed |
| Uni3C API passthrough | `Wan2GP/wgp.py` `generate_video` signature + call-through | yes | yes | partial | Uni3C args accepted and forwarded to `wan_model.generate`; compile check passed |
| Silent drop guard | `source/models/wgp/orchestrator.py::_filter_wgp_params` | yes | yes | yes | Guard added + static regression test (`tests/test_orchestrator_uni3c_filter.py`) |
| Legacy default ID | `Wan2GP/defaults/wan_2_2_i2v_lightning_baseline_2_2_2.json` | yes | yes | yes | Reintroduced + service-health compatibility tests |
| Legacy default ID | `Wan2GP/defaults/wan_2_2_i2v_lightning_baseline_3_3.json` | yes | yes | yes | Reintroduced + service-health compatibility tests |
| Legacy default ID | `Wan2GP/defaults/wan_2_2_i2v_lightning_svi_3_3.json` | yes | yes | yes | Reintroduced + service-health compatibility tests |
| Legacy default ID | `Wan2GP/defaults/wan_2_2_i2v_lightning_svi_endframe.json` | yes | yes | yes | Reintroduced + service-health compatibility tests |
| Legacy default ID | `Wan2GP/defaults/wan_2_2_vace_lightning_baseline_2_2_2.json` | yes | yes | yes | Reintroduced + service-health compatibility tests |
| Legacy default ID | `Wan2GP/defaults/wan_2_2_vace_lightning_baseline_3_3.json` | yes | yes | yes | Reintroduced + service-health compatibility tests |
| Legacy default ID | `Wan2GP/defaults/z_image_img2img.json` | yes | yes | yes | Restores `z_image_turbo_i2i` path + service-health compatibility tests |
| Task default stability | `source/task_handlers/tasks/task_types.py` mappings | yes | unchanged | yes | Asserted in service-health mapping check |
| VACE detection safety | `source/task_handlers/travel/segment_processor.py` | yes | yes | yes | Strict detection restored + service-health static regression check |

## Validation Plan
- Run targeted unit tests for task mappings, defaults compatibility, VACE detection, and Uni3C filter behavior.
- Run lightweight compile checks on edited Python modules.
- Run selected service-health checks for defaults parseability.
