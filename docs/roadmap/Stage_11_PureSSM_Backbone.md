# Stage 11 — PureSSMDetector Backbone Build (BiMamba Spatial)
**PureSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **✅ STAGE COMPLETE (2026-07-11).** BiMamba spatial backbone built & review-clean — **8.38 M spatial
> params, 27 new tests**. Probe: eager **39.5 Hz** / CUDA-graph replay **3.77 ms** (~91 Hz projected) /
> ckpt-train 8.55 GB. Decisions locked: **fully-pure kept** (no conv fallback), training → **rented RTX 5090**
> (Stages 13–14), speed reported as eager + a *labeled* graph-replay column (Stage 16).
> **Authoritative:** plan `docs/plans/2026-07-11-stage11-puressm-backbone.md`, spec
> `docs/specs/2026-07-11-puressm-backbone-design.md`, notes `docs/notes/Stage11_build_notes.md`.
> Code: `code/event_ssm/models/puressm/` (moved from `spatial/` in the 2026-07-16 restructure).

---

## Overview

The **second model** of Thesis B. Replace EventSSM's four ResNet-18 conv spatial stages with a
**bidirectional-Mamba (BiMamba) spatial mixer**, keeping the FPN, Mamba temporal path, YOLOX head, loss, and
Gen1 pipeline **identical**. This makes EventSSM → PureSSM a *backbone-only* swap: any delta is attributable
solely to CNN-vs-SSM spatial processing.

**System:** Linux PC, RTX 5070 Ti (build + probe). **Prerequisite:** EventSSM complete (Stages 0–10).

---

## Goal

- A `BiMambaSpatialStages` module that is a **duck-type drop-in** for `ResNetSpatialStages` (same `stage_dims`,
  same output feature maps at strides 8/16/32) so it plugs into the shared `ResNetMamba` recurrent skeleton.
- No mainline `mamba-ssm` bidirectional 2D primitive exists → **hand-built** on the unified `_scan.py`/`_scan2d`
  kernels: a row+column bidirectional selective scan (`BiMamba1DScan` → `BiMamba2DBlock` → `BiMambaSpatialStages`).
- Verified numerics (kernel-vs-reference parity, flip-equivariance, bf16 no-NaN, gradients both directions).

---

## Why This Stage Exists

Stage 8 found EventSSM's deficit vs the ViT baseline was **almost entirely large-object** (AP_L −5.96), pointing
at ResNet-18's *local, hierarchical* receptive field. The hypothesis: a **global-spatial SSM** recovers the
long-range context the CNN lacks. Stage 11 builds the instrument to test it.

---

## What Was Done

- Built `code/event_ssm/models/puressm/`: `_scan2d.py` (bidirectional 2D selective scan), `bimamba_block.py`
  (`BiMamba2DBlock`, `LayerNorm2d`, `DropPath`), `bimamba_spatial.py` (`BiMambaSpatialStages`).
- 27 new unit tests (kernel parity across widths, flip-equivariance, backward-direction liveness, bf16 autocast).
- Probe (`stage11_probe.py`, `stage11_erf.py`, `stage11_u1/u2_proof.py`): eager 39.5 Hz; CUDA-graph replay 3.77 ms;
  training VRAM 8.55 GB per stream at batch 4.

## Result

**8.38 M spatial params** (vs ResNet-18's 11.23 M — leaner), review-clean, all tests green. Three decisions locked:
(1) keep it **fully pure** (no conv shortcut); (2) **train on a rented RTX 5090 32 GB** (same `sm_120`, zero stack
mismatch) since local VRAM is tight; (3) report deploy speed as **eager + a clearly-labeled graph-replay column**,
never substituting one for the other.

---

## Next Stage

→ **Stage 12: Integration + Smoke** — make PureSSM Hydra-selectable in the RVT pipeline and prove it can overfit a
single real Gen1 batch.
