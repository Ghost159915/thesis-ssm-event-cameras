# Stage 16 — PureSSMDetector Pillars & Visuals (Efficiency · Robustness · Deploy · Qualitative)
**PureSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **✅ STAGE COMPLETE (2026-07-16).** Completes PureSSM's evidence base to match EventSSM's Stages 9–10.
> **Efficiency:** fewest FLOPs **10.12 G**, **best mAP/GFLOP 4.59**, eager 37 Hz (launch-bound) → **CUDA-graph
> deploy 181 Hz** (5.3×). **Robustness:** true-10× retention **69.7% — most rate-robust of all models** (vs
> EventSSM 63.0 / S5-RVT 62.2 / ConvLSTM 17.7). Plus 3-model regime tables and EventCV GT-vs-pred / large-car videos.
> **Authoritative:** `docs/results/Stage16_results.md`; plan `docs/plans/2026-07-15-stage16-pillars-visuals.md`.
> Harness reused: `code/event_ssm/benchmark/`, `stage9_*` sweeps, `code/event_ssm/viz/`.

---

## Overview

Stages 9 and 10 gave EventSSM its robustness and efficiency pillars; Stage 16 does the same for PureSSM (and adds a
CUDA-graph deployment column + qualitative videos), so the two own-models are compared on identical axes. All three
models share the frozen neck/head/pipeline/evaluator → controlled throughout.

**System:** Linux PC, RTX 5070 Ti. **Prerequisite:** Stage 15 (accuracy + ERF done).

---

## Goal

- **Efficiency** (Stage-10 analog): params, FLOPs, mAP/GFLOP, latency, energy, streaming-state size — 3 models.
- **Temporal robustness** (Stage-9 analog): both regimes (fixed-cadence window sweep + true rate change), retention.
- **CUDA-graph deployment mode:** state-carrying graph-replay latency, reported as a *separate labeled* column.
- **Qualitative:** EventCV GT-vs-pred overlay videos on large-car sequences (model name burned into each frame).

---

## What Was Done

- Extended the benchmark harness with a `puressm` kind + `--graph` column; ran all three models in one session.
- Built state-carrying CUDA-graph capture (`benchmark/graph_capture.py`), GPU-verified that replay matches eager
  and state carries across steps.
- Parameterized the Stage-9 sweeps (env-var driven, EventSSM defaults byte-identical) → ran PureSSM's two regimes.
- Shared EventCV renderer (`viz/event_render.py`) + GT-vs-pred dump on 12 large-car recs.

## Result

| Pillar | Finding |
|---|---|
| Size | PureSSM **smallest** (16.33 M) |
| FLOPs / efficiency | **fewest 10.12 G**, **best mAP/GFLOP 4.59** — algorithmically leanest |
| Wall-clock | slowest eager (37 Hz, launch-overhead-bound) → **graph-deploy 181 Hz** (biggest boost, 5.3×) |
| Robustness | **true-10× retention 69.7% — most rate-robust**; Δt-compensation falsified (consistent with Stage 9) |

**Honest read:** PureSSM is the leanest-compute, most rate-robust, best-AP_L model, but **not** the wall-clock
fastest (EventSSM's cuDNN convs utilise the GPU better in both eager and graph modes). Event sparsity does **not**
buy GPU speed — the efficiency claim is FLOP-efficiency + rate-robust deployability, not "sparse = fast."

---

## Summary — Both Investigations Complete

With Stage 16 done, **EventSSM (Stages 0–10) and PureSSM (Stages 11–16) are both complete** across accuracy,
temporal robustness, and efficiency. The controlled ablation delivered its headline: a *pure state-space* spatial
backbone closes half the large-object gap (AP_L +2.95), is the smallest and most rate-robust model, at a wall-clock
latency cost that CUDA-graph deployment largely repairs.

**Full synthesis:** `docs/results/Thesis_Progress_Writeup.md`.

---

## Next

→ Thesis chapters (integrate results into `thesis/latex/`); then the **1Mpx/Gen4 resolution study** (planned,
deferred) and the **Loihi 2 / spiking-SSM** fork (`docs/research/INRC_Loihi_Access_Notes.md`).
