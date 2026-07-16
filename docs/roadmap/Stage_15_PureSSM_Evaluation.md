# Stage 15 — PureSSMDetector Evaluation (Gen1 Test Set)
**PureSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **✅ STAGE COMPLETE (2026-07-15).** Gen1 **test/AP = 46.43** (vs EventSSM 46.22 / S5-RVT 47.72).
> **Headline: AP_L 44.70 → 47.65 (+2.95) — ~half the large-object gap closed → the Stage-8 receptive-field
> hypothesis is CONFIRMED.** Overall essentially tied with EventSSM (+0.21, within single-seed noise → report
> AP_L, not overall). Trained-ERF figure confirms the mechanism.
> **Authoritative:** results table `docs/results/Stage15_results_comparison.md`; ERF figure
> `code/event_ssm/proofs/out/u5_erf_trained.png`. Eval script `stage14_puressm_test_eval_local.sh`;
> ERF probe `stage15_erf.py`.

---

## Overview

The primary results stage — a **one-shot** evaluation of the Stage-14 checkpoint on the held-out Gen1 test split,
through the *same* Prophesee/COCO evaluator as EventSSM and the S5-RVT baseline. This is the number the thesis
hinges on. (The eval **script** keeps the `stage14_` prefix, named after the training stage, mirroring EventSSM's
`stage7_test_eval_local.sh` → Stage-8 table.)

**System:** Linux PC, RTX 5070 Ti. **Prerequisite:** Stage 14 (converged checkpoint home).

---

## Goal

- Gen1 test-set mAP (COCO 0.50:0.95), per-class (car/ped), and per-size (AP_S/M/L) for PureSSM.
- A three-model comparison table (S5-RVT / EventSSM / PureSSM) under one evaluator.
- A trained-weights **effective-receptive-field (ERF)** figure to explain the AP_L result mechanistically.

---

## What Was Done

- Ran the backbone-only-swapped eval (`+experiment/gen1=puressm`, identical Stage-8 recipe) → **test/AP 46.43**.
- Built the comparison table and confirmed the evaluator is internally consistent (mean(car, ped) = overall AP).
- Probed the **trained** ERF (`stage15_erf.py`, gradient-based, Luo et al. 2016): BiMamba σ **85.6 / 105.5 px**
  (stages 3/4) vs ResNet **41.8 / 84.4** — and training *widens* the SSM field (+30–34 px) but barely the CNN's
  (+0.2). Visual mechanism for the AP_L win.

## Result

| Metric | S5-RVT | EventSSM | **PureSSM** |
|---|---|---|---|
| test/AP | 47.72 | 46.22 | **46.43** |
| **AP_L** | 50.66 | 44.70 | **47.65 (+2.95)** |
| Car / Ped | 63.88 / 31.56 | 61.67 / 30.78 | 62.83 / 30.02 |

A pure state-space spatial backbone — no attention, no spatial convolution — **matches** the CNN hybrid overall,
**beats** it on large objects, and is the **smallest** model. The AP_L gain concentrates in **cars** (+1.16, the
class holding the large instances); pedestrians (never "large") dip slightly (−0.76) — the receptive-field
signature, not an aggregate fluke. **Caveat: single seed** → the AP_L +2.95 is the reportable win; the overall
+0.21 is within noise.

---

## Next Stage

→ **Stage 16: Pillars & Visuals** — complete PureSSM's evidence base (efficiency, temporal robustness, CUDA-graph
deployment, qualitative videos) to match EventSSM's Stages 9–10.
