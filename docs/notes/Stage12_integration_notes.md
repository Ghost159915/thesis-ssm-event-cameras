# Stage 12 — PureSSM Integration + Overfit Smoke: Notes

**Date:** 2026-07-12 · **Branch:** `stage12-puressm-integration` (base `47056a1`) · **Plan:** `docs/superpowers/plans/2026-07-11-stage12-puressm-integration.md`

## What was wired

- **Registration** (`code/event_ssm/integration/register.py`): the monkeypatched `build_recurrent_backbone` now dispatches `backbone_cfg.name == "PureSSM"` → `ResNetMambaBackbone(spatial=BiMambaSpatialStages(...))`, mapping config keys `depths`/`spatial_d_state`/`drop_path_rate`/`checkpoint_blocks` to the spatial module and keeping the temporal path identical to ResNetMamba (`d_state=64`, blocks on `in_stages=[2,3,4]` only). The config-modifier dispatch widened to both names (same multiple-of-32 padding → `in_res_hw=(256,320)`, `num_classes=2` on Gen1). ResNetMamba branch byte-identical (review-verified against the pre-change file).
- **Hydra config pair** (tracked): `code/event_ssm/configs/puressm_yolox/default.yaml` (only the backbone block differs from `resnet_mamba_yolox` — fpn/head/postprocess diff-verified byte-identical) and `configs/experiment/gen1/puressm.yaml` (**zero recipe-value drift** vs `resnet_mamba.yaml`: reviewer-run diff shows the model-group defaults line as the sole difference — the 400k recipe, batch sizes, workers, dataset block are byte-identical, preserving the controlled comparison).
- **Symlinks** into the gitignored RVT tree (`config/model/puressm_yolox/default.yaml`, `config/experiment/gen1/puressm.yaml`), recorded in `docs/patches/README.md` for re-clone recovery.
- **Selection string** (train/eval, identical shape to the baseline's): `model=rnndet +experiment/gen1=puressm`
- **Smoke harness**: `compose_smoke_config(..., experiment="puressm")` kwarg (default `"resnet_mamba"` keeps Stage-5 callers byte-identical).

## Overfit smoke (Stage-12 exit gate)

`code/event_ssm/proofs/smoke_overfit_puressm.py` — the Stage-5 harness with `experiment="puressm"` and `model.backbone.checkpoint_blocks=True` (local 16 GB card; the override was traced end-to-end Hydra→register→module in review).

**First official run (150 ep, 2026-07-11):** 20.334 → 8.113 = **2.5× — gate missed** (3×). Diagnosed cause (not a learning bug — loss halved twice, descending, no NaN): the smoke left **DropPath 0.1 active**; stochastic depth randomly drops residual branches and directly fights single-batch memorization, while the 3× threshold was calibrated on the DropPath-free ResNet smoke. Plus from-scratch init (pre-registered as slower vs pretrained ResNet). **Fix (smoke-only):** `model.backbone.drop_path_rate=0.0` override in the smoke; the real training recipe keeps 0.1. 10-epoch evidence check pre/post fix: 1.3× → **3.5×** (the gate's 3× threshold, cleared in 10 epochs once the regularizer was removed — diagnosis confirmed).

**Second official run (300 ep, user-executed 2026-07-12):** loss 26.543 → 4.391, **6.0× reduction — PASS** (gate 3×; double margin), no NaN. Curve: `results/smoke_test/puressm_overfit_loss_curve.png`. The DropPath diagnosis is fully vindicated: identical script scored 2.5× (150 ep) with DropPath active vs 6.0× (300 ep) and 3.5× (10 ep!) with it off.

## Test suite

92 passed (+1 gpu-deselected) at close: 87 from Stages ≤11 plus 5 new registration/compose tests (`test_register_puressm.py`).

## Notes / deviations

- The plan's experiment-yaml header example had a latent off-by-one in its parity-check line count; the implemented yaml aligns 1:1 with `resnet_mamba.yaml` (7-line header), making the parity property strictly stronger than the plan's literal check. Recorded by the Task-2 reviewer.
- Carried minors for the whole-branch review live in `.superpowers/sdd/progress.md` (Stage-12 section).
