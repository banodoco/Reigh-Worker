# PR #19 Risk Register

## High
- Uni3C runtime mismatch with v10.83 internals.
  - Mitigation: restored Uni3C helpers + forward-path injection; added orchestrator fail-fast for dropped Uni3C params.
- Legacy default IDs referenced by task mappings missing in baseline.
  - Mitigation: reintroduced compatibility default files and added tests.

## Medium
- `z_image_turbo_i2i` indirect mapping regression if `z_image_img2img` missing.
  - Mitigation: restored `z_image_img2img.json`; added mapping/assertion tests.
- Non-VACE models misclassified as VACE due broad keyword matching.
  - Mitigation: VACE detection now requires explicit `"vace"` in model name.

## Medium
- Orchestrator API drift with future WGP updates.
  - Mitigation: centralized defaults application helper (`_apply_model_settings_overrides`) and strict param filter validation.

## Validation Gates
- Compile check: edited Python modules compile cleanly.
- Unit checks:
  - task model mapping coverage (legacy IDs, z-image i2i)
  - defaults compatibility files existence + parse
  - segment processor VACE detection regression test
  - orchestrator Uni3C param-drop fail-fast test

## Remaining Operational Risks
- Full end-to-end GPU Uni3C generation path still requires runtime smoke on target hardware.
- Very large upstream delta (PR #19) may still contain unrelated behavior shifts outside touched areas.
