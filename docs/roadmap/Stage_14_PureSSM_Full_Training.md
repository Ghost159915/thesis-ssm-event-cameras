# Stage 14 — PureSSMDetector Full Training (400k)
**PureSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **✅ STAGE COMPLETE (2026-07-15).** 400k-step run on the rented **vast.ai RTX 5090** (run `aekvsalq`,
> `MAX_STEPS=400000 VAL_EVERY=10000`, ~27 h, W&B `honest-violet-3`). **Best val/AP 0.48 @ step 310k**; the run
> then finished the full 400k cleanly (`max_steps reached`, no further val improvement). Best + last checkpoints
> scp'd home & **sha256-verified byte-identical** → instance destroyed.
> **Authoritative:** cloud notes `docs/notes/Stage13_cloud_notes.md`; runbook `docs/runbooks/Cloud_Runbook_5090.md`.
> Checkpoint: `results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt`.

---

## Overview

The primary PureSSM training run — the exact counterpart of EventSSM's Stage 7. Full Gen1, 400k-step OneCycle,
bf16, batch 4, on the rented 5090 using the Stage-13 self-healing flow.

**System:** rented **vast.ai RTX 5090 32 GB**. **Prerequisite:** Stage 13 (flow verified, sanity run passed).

---

## Goal

- Train PureSSM to convergence on full Gen1 under the **same recipe as EventSSM** (so the eventual comparison is
  fair — only the backbone differs).
- Retrieve the best checkpoint home, integrity-verified, and release the rented GPU.

---

## What Was Done

- Launched `MAX_STEPS=400000 VAL_EVERY=10000`; monitored via W&B (`honest-violet-3`) and the per-stage norm watch.
- **val/AP climbed to 0.48 by step 310k**, then plateaued; the run completed the full schedule cleanly rather than
  early-stopping, confirming the peak.
- Copied the best (`step=310000`) and last (`step=400000`) checkpoints home; **sha256 byte-verified** each against
  the cloud copy before **destroying the instance** (cost control).

## Result

A converged PureSSM checkpoint (val/AP 0.48) safely home and verified. Note val/AP is an *in-training* number on
the validation split — the fair cross-model comparison is the held-out **test** set (Stage 15), where the only
apples-to-apples numbers live.

---

## Next Stage

→ **Stage 15: Evaluation** — one-shot Gen1 test-set evaluation vs EventSSM and the S5-RVT baseline (mirrors Stage 8).
