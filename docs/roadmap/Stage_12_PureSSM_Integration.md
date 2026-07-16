# Stage 12 — PureSSMDetector Integration + Smoke Test
**PureSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **✅ STAGE COMPLETE (2026-07-12).** PureSSM selectable via `model=rnndet +experiment/gen1=puressm`
> (register-dispatch + config pair + Hydra symlinks recorded in `docs/patches`). **Overfit smoke 6.0× PASS**
> (DropPath-off smoke fix — stochastic depth fights single-batch memorization).
> **Authoritative:** plan `docs/plans/2026-07-11-stage12-puressm-integration.md`, notes
> `docs/notes/Stage12_integration_notes.md`.

---

## Overview

Wire the Stage-11 BiMamba backbone into the verified S5-RVT/RVT training stack so it is selectable by config, and
prove end-to-end learning with an overfit smoke test — the same gate EventSSM passed at Stage 5.

**System:** Linux PC, RTX 5070 Ti. **Prerequisite:** Stage 11 (backbone built + tested).

---

## Goal

- PureSSM buildable through the same `build_recurrent_backbone` dispatch as EventSSM (no pipeline forks).
- A config pair (`puressm_yolox/default.yaml` + `experiment/gen1/puressm.yaml`) mirroring the EventSSM configs,
  differing only in backbone.
- An **overfit smoke test**: the model drives training loss toward zero on a single real Gen1 batch (proves
  gradients flow through the BiMamba scan end-to-end into the shared head).

---

## What Was Done

- Extended `integration/register.py` so `backbone.name == "PureSSM"` injects `BiMambaSpatialStages` into the shared
  `ResNetMambaBackbone` skeleton (temporal path/config **identical** to EventSSM — the controlled experiment).
- Added the config pair + the four Hydra config symlinks (recorded in `docs/patches` so they survive an
  `external/` re-clone).
- Ran the overfit smoke: initially plateaued — **root cause was DropPath** (stochastic depth actively fights
  single-batch memorization). Disabling DropPath for the smoke → **6.0× loss reduction, PASS**.

## Result

PureSSM is a first-class, config-selectable model in the pipeline with **zero recipe drift** from EventSSM, and it
demonstrably learns. Ready for real training.

---

## Next Stage

→ **Stage 13: Cloud Short Run** — a compressed sanity run on a rented RTX 5090 to confirm PureSSM trains cleanly
before committing to the full 400k run.
