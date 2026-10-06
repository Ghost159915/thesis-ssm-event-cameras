# Stage 3c — Mamba Temporal Block  ✅ BUILT
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

> **STATUS (2026-06-12):** Implemented & verified (2026-06-06). Source of truth = `code/event_ssm/temporal/`.
> This document was reconciled to the **as-built** design per `PLAN_FIXES_FOR_CLAUDE.md` (ISSUE-01 scan axis,
> ISSUE-02 single scan path). The original draft described a *spatial* scan over `H*W` tokens with three
> Mamba modules placed *after* the FPN and an "Approach C" train-stateless shortcut — **all superseded below.**

---

## Overview

The temporal block is the most novel component of EventSSMDetector: it gives the detector **memory across
consecutive event windows**. It is a causal **Mamba-1** block applied over the **TIME axis**, independently per
spatial location, and **interleaved after each ResNet backbone stage** (not after the FPN).

**Files:** `code/event_ssm/temporal/mamba_temporal.py` (`MambaTemporalBlock`), `code/event_ssm/temporal/_scan.py`
(dual-path scan). **Placement/wiring:** `code/event_ssm/backbone/resnet_mamba.py`.

---

## Goal (as built)

A `MambaTemporalBlock` that:
- Receives a per-stage feature sequence `(L, B, C, H, W)` (L = time/windows), folds it to `(B*H*W, L, C)`.
- Applies `num_layers` causal Mamba-1 block(s) over `L = time`.
- Returns enriched features `(B*H*W, L, C)` + per-layer state; **carries state across clips during eval**.
- Passes shape, state-influence, gradient, bf16-autocast, and **step/parallel equivalence** tests.

---

## Why This Component

### The Temporal Understanding Problem
Event cameras produce a continuous stream of temporal change rather than discrete snapshots. A model that processes
each window independently throws away the most valuable information event cameras provide — the *history* of what
has been moving and where.

**Failure of ConvLSTM** (`Zubic_2024_SSM_EventCameras_CVPR`, CVPR 2024 Spotlight): ConvLSTM temporal models (original
RVT) degrade severely when inference frequency differs from training frequency (>20 mAP drop at 4× vs only 3.76 for
SSMs). ConvLSTM is discrete-time — trained for a fixed step between inputs; changing the step breaks its dynamics.

**Why SSMs solve this:** SSMs model temporal dynamics in continuous time (`Gu_2022_S4_StructuredStateSpaces_ICLR`):
`h'(t) = A h(t) + B u(t)` can be discretised at any Δ without retraining — critical for event cameras where the
effective event rate varies with scene dynamics.

**Why Mamba over S5:** `Gu_2023_Mamba_SelectiveStateSpaces_arXiv` introduced *selective* state spaces — B, C and Δ are
functions of the input `x_t`. The model adaptively decides what to retain (small Δ → long memory) vs forget (large Δ).
For event cameras this selectivity is directly motivated: update state fast during high activity, preserve it when quiet.

**SMamba evidence** (`Yang_2025_SMamba_EventDetection_AAAI`): replacing fixed S4/S5 spatial processing with selective
Mamba improves Gen1 mAP from 47.7 to 50.4 (+2.7) — empirical support for Mamba over S5 in this domain.

---

## Scan Axis = TIME, Per Spatial Location (corrects the original spatial-scan design)

The decisive design point (errata **ISSUE-01**): the recurrence runs over the **time/window axis**, with one
independent sequence per spatial location and per stage — mirroring the baseline (RVT ConvLSTM, S5-RVT S5), which
recur over time *per pixel*. Concretely, for a stage feature map `(L, B, C, H, W)` the backbone reshapes to
`(B*H*W, L, C)` (`fold`) and scans over `L`; `unfold` restores `(L, B, C, H, W)`.

Spatial mixing is **not** Mamba's job here — the ResNet convolutions and the FPN handle it. Therefore the earlier
"flatten `H*W` spatial tokens" framing and the row-major-vs-Hilbert scan-order discussion are **moot and removed**.

**Why this matters (controlled comparison):**
- Temporal memory is **symmetric** across the frame (every pixel sees the same history). A spatial scan would inject
  cross-window memory only at scan-token 0, decaying over ~1100 spatial steps → weak, spatially asymmetric memory.
- The temporal **mechanism matches the baseline**, so any mAP delta vs S5-RVT is attributable solely to S5→Mamba
  selectivity — the thesis's controlled experiment. A spatial scan would make Stage 9 apples-to-oranges.

---

## Mamba Block Internals

Each Mamba-1 block transforms `(N, L, C) → (N, L, C)` with `N = B*H*W`, `L = time`, `C = d_model = stage width`:
1. Input projection `C → expand·C` (expand = 2).
2. Depthwise 1-D causal conv (`d_conv = 4`) — short-range mixing along time.
3. Selective scan (SSM): input-dependent `B(x_t)`, `C(x_t)`, `Δ(x_t)`; hidden state `(N, expand·C, d_state)`,
   `d_state = 16`.
4. Output gating + projection `expand·C → C`.

The SSM hidden state `(N, expand·C, d_state)` is what carries temporal information across windows.

---

## Dual-Path Scan: Training vs Inference (replaces "Approach A/B/C")

**Spike finding** (`mamba-ssm 2.3.2`, `proofs/spike_state.py`): no single stock Mamba call is **both**
autograd-trainable **and** able to carry explicit cross-clip state.
- `mamba(x)` parallel forward → trainable, but starts from **zero** state each clip (no cross-clip carry).
- `mamba.step()` loop → carries explicit `(conv, ssm)` state, but mutates it in-place and uses the inference-only
  `selective_state_update` / `causal_conv1d_update` kernels → **not** trainable.

**As-built decision** (`temporal/_scan.py`, `mamba_scan_time`):
- **TRAINING** (`module.training = True`): trainable **parallel scan** over `L = T`. Gradients flow through **all T
  windows** of the clip; state is truncated at the clip boundary — exactly as the baseline truncates at
  subsequence/recording boundaries.
- **EVAL / INFERENCE**: stateful **step loop** carrying `(conv, ssm)` state across windows; reset at a new recording.

Both paths compute the **same** selective-SSM function; they differ only in the initial state (zero-per-clip in
training vs carried in inference) — the standard SSM train/infer setup. The two earlier "Approaches A/B/C"
(train-stateless / manual warm-up) are **deleted**: they introduced a train/test distribution mismatch the errata
(**ISSUE-02**) flags as misleading for preliminary numbers.

> **OPEN CAVEAT — resolve before Stage 6/7 full training.** Full cross-clip *training* state (TBPTT parity with the
> S5 baseline) needs a custom differentiable selective scan accepting an initial state (**option β**), **or** evaluate
> without carried state to match training. Tracked in the Stage-3 spec §9 and `CLAUDE.md`. Not required to build/verify
> the Stage-3 backbone, but it is a prerequisite to a defensible Stage-6 mAP and the Stage-9 study.

---

## Placement: Interleaved Per Backbone Stage (not 3 modules at FPN scales)

> **Correction (2026-10-06):** as built (and in every version since commit `3a93a61`) the temporal blocks are
> per-scale *taps*: `ResNetMambaBackbone.forward` runs the whole spatial pyramid first and sends each temporal
> output to the FPN only; the next spatial stage consumes the per-frame features. "Interleaved / mirrors
> `RNNDetectorStage`" below overstates the similarity to RVT, whose recurrent output feeds the next stage. Since
> Stage 6, temporal blocks exist on stages 2–4 only. Thesis wording: main.tex §3.1.1.

The original plan placed three Mamba modules **after** the FPN (all `d_model = 256`). The **built** design interleaves
**one `MambaTemporalBlock` after each of the 4 ResNet stages**, at the stage's native width
(64 / 128 / 256 / 512), **before** the FPN (`code/event_ssm/backbone/resnet_mamba.py`). Stages 2–4
(strides 8/16/32) then feed the reused PAFPN. Rationale:
- Temporal enrichment at every spatial resolution, at the stage's native width.
- Mirrors RVT's `RNNDetectorStage` (conv stage → recurrent block) so the **recurrent-backbone contract** holds and the
  reused PAFPN / YOLOX head / Lightning training apply unchanged (see Stage 4).
- With `num_layers_per_stage = 1` (default) this is **4 Mamba blocks total** — lighter than the old "4 layers × 3
  scales" sketch.

---

## Implementation Specification (as built)

**`MambaTemporalBlock(d_model, d_state=16, d_conv=4, expand=2, num_layers=1)`**
- `nn.ModuleList` of `num_layers` `Mamba(d_model, d_state, d_conv, expand)` blocks.
- `forward(x:(N,L,C), state=None) → (y:(N,L,C), new_state)` — iterates layers via `mamba_scan_time`;
  `new_state` is a per-layer list (`None` in train, `(conv_state, ssm_state)` in eval).
- static `fold((L,B,C,H,W)) → ((B*H*W, L, C), dims)`; static `unfold` inverts it.

**State shapes per layer (eval):** `conv_state (N, d_inner, d_conv)`, `ssm_state (N, d_inner, d_state)`, where
`d_inner = expand · d_model` and `N = B*H*W` **differs per stage** (stride-8 stage has the most rows).
`reset_state` ⇔ passing `state=None` (allocates zero `(conv, ssm)` via `allocate_inference_cache`).

---

## Tests (as built — `code/event_ssm/tests/test_mamba_temporal.py`)

- **shape preserved** — `(N,L,C)` in → out; state returned in eval.
- **state influence** — output with carried state ≠ output from fresh state (`> 1e-4`). *Necessary but not sufficient.*
- **gradients** — flow to all params, no NaN (train path).
- **bf16 autocast** — forward under `torch.autocast('cuda', bfloat16)`.
- **step/parallel equivalence (errata ISSUE-01 correctness criterion)** — from zero initial state, the eval step-mode
  output matches the training parallel-scan output to ~1e-3 (fp32). This is the real recurrent-correctness check.

**Visual proof:** `proofs/proof_mamba.py → out/u2_state_influence.png` (state influence persists across 8 clips);
`proofs/spike_state.py → out/spike_state.md` (the dual-path spike finding).

---

## Deliverable / Status

`code/event_ssm/temporal/{mamba_temporal.py,_scan.py}` — **DONE**, all tests pass.

## Next Stage

→ **Stage 3d: Detection Head Interface** (verified) → **Stage 4: Drop-in Integration into RVT.**
