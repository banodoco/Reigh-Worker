# PR #19 Diff Manifest

## Baselines
- `main`: `81ab6fb`
- `pr-19-head`: `21750ae`
- Integration branch base: `codex/pr19-layered-port` at `21750ae`

## Headline Delta (`main...pr-19-head`)
- `485 files changed, 270945 insertions(+), 6101 deletions(-)`

## Focused Delta Inventory (high-impact)
- `Wan2GP/models/wan/any2video.py` modified
- `Wan2GP/models/wan/modules/model.py` modified
- `Wan2GP/wgp.py` modified
- `Wan2GP/models/wan/uni3c/*` deleted in PR #19 baseline
- `Wan2GP/defaults/*` includes major adds/removes (LTX-2, Flux 2 Klein, TTS + legacy baseline removals)
- `source/models/wgp/orchestrator.py` modified
- `source/task_handlers/tasks/task_types.py` modified
- `source/task_handlers/travel/segment_processor.py` modified

## Critical Remove/Add Snapshot
### Removed in PR baseline
- `Wan2GP/models/wan/uni3c/__init__.py`
- `Wan2GP/models/wan/uni3c/controlnet.py`
- `Wan2GP/models/wan/uni3c/load.py`
- `Wan2GP/defaults/wan_2_2_i2v_lightning_baseline_2_2_2.json`
- `Wan2GP/defaults/wan_2_2_i2v_lightning_baseline_3_3.json`
- `Wan2GP/defaults/wan_2_2_i2v_lightning_svi_3_3.json`
- `Wan2GP/defaults/wan_2_2_i2v_lightning_svi_endframe.json`
- `Wan2GP/defaults/wan_2_2_vace_lightning_baseline_2_2_2.json`
- `Wan2GP/defaults/wan_2_2_vace_lightning_baseline_3_3.json`
- `Wan2GP/defaults/z_image_img2img.json`

### Added in PR baseline (examples)
- `Wan2GP/defaults/ltx2_19B.json`
- `Wan2GP/defaults/ltx2_distilled.json`
- `Wan2GP/defaults/flux2_klein_4b.json`
- `Wan2GP/defaults/flux2_klein_9b.json`
- `Wan2GP/defaults/ace_step_v1_5.json`
- `Wan2GP/defaults/heartmula_oss_3b.json`

## Reconciliation Policy
- Keep PR #19 as functional base.
- Re-apply custom deltas as explicit overlays:
  - Uni3C end-to-end support.
  - Legacy compatibility default IDs.
  - Orchestrator strict parameter-drop detection for Uni3C.
  - VACE detection guard to avoid non-VACE misclassification.
