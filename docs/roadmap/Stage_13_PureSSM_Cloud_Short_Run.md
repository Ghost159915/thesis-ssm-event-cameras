# Stage 13 — PureSSMDetector Cloud Short Run (Sanity)
**PureSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **✅ STAGE COMPLETE (2026-07-13).** 25k-step sanity run on a rented **vast.ai RTX 5090** → **val/AP = 0.351**
> (0.155@5k → 0.286@15k → 0.351@25k; beats the 0.10–0.15 gate; monitor norms stable/finite → PureSSM trains
> cleanly). **NOT comparable to EventSSM** — a compressed sanity schedule, not the real run.
> **Authoritative:** plan `docs/plans/2026-07-12-stage13-cloud-short-run.md`, notes
> `docs/notes/Stage13_cloud_notes.md`, runbook `docs/runbooks/Cloud_Runbook_5090.md`. Cloud scripts:
> `code/event_ssm/scripts/cloud/`.

---

## Overview

Before spending ~27 h on a full 400k run, de-risk it: a short, cheap run on the rented GPU to confirm PureSSM's
loss/AP move in the right direction and its per-stage feature norms stay finite (Mamba-R instability watch).

**System:** rented **vast.ai RTX 5090 32 GB** (`sm_120`, same stack as local — zero mismatch). **Prerequisite:**
Stage 12 (integrated + smoke-passed); dataset uploaded to HF Hub.

---

## Goal

- An idempotent 5090 bootstrap (`setup_env_5090.sh`): pinned `external/` RVT re-clone, cu128 torch stack,
  auth-guarded dataset pull, config symlinks, one-shot checkpoint upload.
- A beginner runbook (`Cloud_Runbook_5090.md`) so the flow is repeatable for Stage 14.
- Training monitors (`PURESSM_MONITOR=1`): per-stage feature-norm + NaN watch.
- A short run clearing a **val/AP 0.10–0.15 gate** with stable norms.

---

## What Was Done

- Built + ran the cloud flow end-to-end; 25k-step run reached **val/AP 0.351** with finite/stable monitor norms.
- **Fixed six env gremlins invisible to local offline runs** so `setup_env_5090.sh` self-heals (committed
  `f788e75`): ROS-contaminated lock → per-line skip; missing `transformers`; `torchvision`/`torchaudio` via cu128
  index; `packaging` file:// path; `RMSNormGated`→`RMSNorm` verify; `log_model=False` auto-patch for the
  online-wandb `_entity` crash.
- **Host lesson:** the first rented host advertised 2460 Mbps but delivered ~11 kB/s real CDN bandwidth →
  **curl-test bandwidth before bootstrapping.**

## Result

PureSSM trains cleanly on the 5090; the flow is reproducible and self-healing. 25k checkpoints copied home
(sanity only, **not for eval**). Green light for the full run.

---

## Next Stage

→ **Stage 14: Full 400k Training** — the real training run (matches EventSSM's Stage-7 recipe).
