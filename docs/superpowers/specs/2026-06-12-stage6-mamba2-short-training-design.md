# Stage 6 — Mamba-2 Temporal Migration + Short Training Run — Design Spec

**Date:** 2026-06-12 · **Depends on:** Stage 5 (smoke COMPLETE) · **Author decision-mode:**
collaborative (user chose the parity resolution — Mamba-2 — and approved this design; the spec + the
stage report are the review artifacts).

## 1. Goal

Resolve the standing **train/eval temporal-state parity** prerequisite (Stage-3 spec §9 item 1, Stage-5
report §9) and run a **short training run** on the full Gen1 train split to validate training dynamics
(loss/LR/grad curves, no divergence) before the Stage-7 full run.

Parity is the gating decision because the as-built dual-path scan **optimises the model under one function
and evaluates it under another** (training zero-inits temporal state per subsequence; eval carries
cross-clip state), which biases val mAP and **confounds the controlled comparison vs S5-RVT**. No number
that enters the thesis is trustworthy until train and eval compute the same function.

## 2. Decision record — parity → Mamba-2 (full TBPTT)

**Decision:** migrate the temporal block from **Mamba-1 → Mamba-2** and carry state in **both** training
(detached, TBPTT) and eval, replacing the dual-path scan with a single stateful path.

**Why Mamba-2 (option 3) over a custom Mamba-1 β-scan (option 2):** the parity problem is precisely
"the trainable kernel will not accept a carried-in initial state." Mamba-2's official kernel
`mamba_chunk_scan_combined(..., initial_states=…, return_final_states=True)` solves this **natively, on
the fast fused (tensor-core) trainable path** — eliminating the correctness and throughput risk of a
hand-rolled differentiable selective scan. Mamba-2 is also the more modern (2024 SSD) architecture
(CLAUDE.md prefers post-2023 selective SSMs) and ships in the **same `mamba-ssm==2.3.2` wheel already
built for Blackwell `sm_120`**, so no environment change is required.

**Mamba-1 vs Mamba-2 (the relevant difference):** Mamba-1 (S6) uses a **diagonal, per-channel** state
transition `A ∈ ℝ^(d_inner×N)` with small `N≈16`, computed by a selective-scan kernel that does **not**
expose an initial state on the trainable forward. Mamba-2 (SSD) restricts `A` to a **scalar per head**
(`hₜ = aₜ·hₜ₋₁ + Bₜxₜ`), which makes the scan a chunked matmul — faster, large state dim (`N=64–256`),
multi-head, and **initial-state in / final-state out** on the trainable kernel. The scalar-`A`
restriction is theoretically less expressive per step but negligible over our short time axis (T≈5–21)
and offset by the larger state dim.

**Consistency:** Mamba-2 is used in **both** EventSSMDetector (this stage) and the later PureSSMDetector
(Stage 8), so the SSM variant is never a confound in the cross-model comparison.

**Rejected:** option 2 (custom β-scan, keep Mamba-1) — most engineering + correctness risk for no
benefit now that the official kernel supports the needed capability; option "eval-without-state" — cheap
parity but discards the cross-clip streaming memory the baseline uses, weakening both mAP and the
controlled comparison; "defer parity" — leaves the short run internally inconsistent.

## 3. Scope / non-goals

- **In scope (Phase A, Claude executes + smokes):** Mamba-2 temporal block; unified stateful TBPTT scan
  (delete the dual-path); Finding §8 (temporal only on FPN-fed stages); equivalence test; re-run overfit
  smoke + health; param-count + curves.
- **In scope (Phase B, handed to the user):** finalize the Hydra training config; fixed-seed 10%-recording
  subset builder; a local RTX 5070 Ti run command **and** a Katana SLURM script; success gates; a
  monitoring checklist; post-run curve/report analysis from the user's logs.
- **Out of scope:** the full-length training run + final mAP (Stage 7); PureSSMDetector (Stage 8);
  temporal-generalisation study (Stage 9); any change to the reused RVT neck/head/loss/data/eval.

## 4. Train/eval regime — dual-path vs unified (and effect on results)

| | Dual-path (as-built) | Unified stateful (Mamba-2) |
|---|---|---|
| Train initial state | zero, per subsequence | carried (detached) across subsequences |
| Cross-clip memory in training | none | yes (TBPTT) |
| Gradient to make persistent state useful | no | yes |
| `train fn == eval fn` | no | **yes** |
| Matches S5-RVT regime | no | **yes** |

**Why the dual-path is wrong:** at eval, `h_t` accumulates over hundreds of windows, but during training
`h_t` only ranged over values reachable within L windows *from zero* → the eval path feeds the SSM and the
downstream conv/FPN/head **out-of-distribution** initial states, and training gave **no gradient signal to
write a state useful past L windows**. (This is the "train-stateless / eval-stateful" pattern `PLAN_FIXES`
ISSUE-02 marked for deletion.)

**Expected effect on results:**
1. **Validity (certain):** the unified val mAP is the number the model was actually optimised for; the
   dual-path number is biased and not reportable. This holds regardless of whether the number moves.
2. **Direction (expected ≥, usually higher; magnitude uncertain):** training now gets gradient to exploit
   long temporal memory (motion cues across many windows — what helps event-stream detection) and removes
   the OOD-state risk that can depress/destabilise the dual-path eval → equal-or-higher mAP, lower
   run-to-run variance. No fixed gain is promised (a few mAP on Gen1 in the RVT literature, data/run
   dependent).
3. **Controlled comparison (thesis-critical):** "mAP delta vs S5-RVT = the S5→Mamba backbone swap" holds
   only if the train/eval state regime also matches the baseline (which trains *and* evals stateful). The
   unified path makes this true; the dual-path confounds it.
4. **Where it shows up:** **not** in the overfit smoke (one batch = one subsequence, no cross-clip state —
   both paths behave identically there); it manifests in **val mAP over full recordings** (Phase B sanity,
   and ultimately Stage 7). Both paths still truncate *gradient* at subsequence boundaries (TBPTT); the
   unified path carries *forward state* across them like the baseline → "inference memory = whole
   recording, gradient horizon = subsequence."

## 5. Phase A — code changes

All changes are in the tracked package `code/event_ssm/`; **no reused RVT baseline file is modified.**

### A1. Mamba-2 temporal block — `temporal/mamba_temporal.py`
- `MambaTemporalBlock` constructs `Mamba2` layers instead of `Mamba`. Config: `d_state=64` (default; up
  from Mamba-1's 16), `headdim=64`, `ngroups=1`, `expand=2`. With Finding §8 the smallest temporal
  `d_model` is **128** (stages 2/3/4 = 128/256/512) → `d_inner = 256/512/1024` → `4/8/16` heads, all valid
  Mamba-2 head configs. (Stage-1's awkward `d_model=64` case is removed by A3.)
- `fold`/`unfold` (the `(L,B,C,H,W) ↔ (B·H·W, L, C)` layout helpers) are unchanged.

### A2. Unified stateful TBPTT scan — `temporal/_scan.py`
- **Delete the dual-path.** Replace `mamba_scan_time` (training = `mamba(x)`; eval = step loop) with a
  single `mamba2_scan_time(layer, x, state)` that, for **both** train and eval, runs the chunked scan with
  `initial_states = state.ssm_state` and `return_final_states=True`, returning `(y, new_state)`.
- Because Mamba-2's public `forward` does not expose `initial_states`, the scan replicates Mamba-2's
  forward internals: in-proj → causal depthwise conv → `mamba_chunk_scan_combined(initial_states=,
  return_final_states=True)` → norm → out-proj. **Both** state components are carried — the **SSM state**
  (`initial_states`/`return_final_states` of the chunk scan, the long-range memory) **and** the depthwise
  **conv state** (seed the causal conv's left context with the last `d_conv-1` inputs) — so the carried
  scan matches a full scan exactly at subsequence boundaries (required by the equivalence gate A4.1). This
  mirrors the current eval path, which already carries `(conv, ssm)`.
- **Parity mechanism:** RVT's `RNNStates` already performs TBPTT for free — it carries `prev_states`
  across subsequences and calls `recursive_detach` between windows / `recursive_reset` at sequence starts.
  The dual-path existed *only* because the Mamba-1 kernel could not accept a carried state, not because of
  an RVT limit. So the backbone simply threads `prev_states` in and returns `new_states` in **both**
  modes; RVT does the detach/reset.

### A3. Finding §8 — temporal only on FPN-fed stages — `backbone/resnet_mamba.py`
- Thread `fpn.in_stages` (`[2,3,4]`) into `ResNetMambaBackbone` (new `temporal_stages` arg, defaulting
  from the FPN config via the builder in `integration/register.py`).
- `self.temporal` becomes a `ModuleDict` keyed by FPN-consumed stage only. Stage-1 spatial features are
  still computed (ResNet needs them downstream) but receive **no** temporal block. Removes the dead ~9% of
  backbone params and the grad-flow caveat.
- The `forward` loop applies temporal only on `temporal_stages`; other stages pass their spatial features
  through unchanged. The `if self.training: zero-init` branch (`resnet_mamba.py:65-67`) is **removed** —
  both modes thread state via A2.
- State format: Mamba-2 per-layer state is `(conv_state, ssm_state)` with `ssm_state` of shape
  `(N, nheads, headdim, d_state)`. The existing `_state_to_bmajor`/`_state_from_bmajor` reshapers are
  shape-agnostic (`*shape[1:]`) and already generalise; the `dim0=B` storage contract for RVT
  `recursive_reset`/`recursive_detach` is preserved.

### A4. Verification & visual proofs (per-stage convention — visual proof each stage)
1. **Equivalence test (correctness gate, ISSUE-01):** T subsequences run with carried state must equal one
   full parallel scan to **~1e-3 (fp32)**. Proves the TBPTT state-threading is correct. Artifact: a
   max-abs-diff plot / printed table. *(This is the test that justifies trusting the carried state.)*
2. **Re-run overfit smoke** (`proofs/smoke_overfit.py`) with the Mamba-2 backbone → must still overfit a
   real Gen1 batch **≥3×**; save the loss curve. (Wiring check; not where parity shows.)
3. **Health** (`proofs/smoke_health.py`): grad-flow (now clean — no dead stage-1 temporal), VRAM sweep,
   eval single-window step latency; a **param-count delta table** (Mamba-1→Mamba-2, minus dead temporal).
4. Full `pytest tests/` green (unit tests updated for Mamba-2 state shapes; the equivalence test replaces
   the trivial "state diff > 1e-4" test per ISSUE-01).

## 6. Phase B — short training run (handed to the user)

- **Data:** full Gen1 **train** split (**user downloads** first); training samples a **random 10% of
  recordings** (fixed seed, **log the recording list**), contiguous windows within each recording
  (ISSUE-10 — not "first 10%"). A reproducible subset-builder script produces the recording list /
  filtered split.
- **Precision/protocol:** **bf16 autocast, NO GradScaler** (ISSUE-09); match the S5-RVT baseline's
  augmentation and precision exactly (controlled comparison). Batch size **B held constant** across carried
  subsequences (the state cache is `N=B·H·W`-sized — Stage-3 prereq 2). BatchNorm sees flattened `L·B`
  (Stage-3 prereq 3) — monitor; GroupNorm / frozen-BN is the documented fallback if it destabilises.
- **Budget (default, user-overridable):** ~1 epoch over the 10% subset (or ~2–3k steps), purely to
  validate dynamics.
- **Deliverables:** finalized Hydra training config (selected as the baseline does:
  `model=rnndet +experiment/gen1=resnet_mamba`); the subset-builder; a **local `events_signals` run
  command**; a **Katana SLURM batch script** (module loads, VRAM, single-GPU; CLAUDE.md default, reused for
  Stage 7); a **monitoring checklist** of what to paste back (loss sub-terms, LR, grad-norm, VRAM,
  throughput, any NaN). Claude produces the curves/report from the returned logs.

## 7. Success criteria

**Phase A (Claude, before handoff):**
- Equivalence test passes (~1e-3); `pytest tests/` green; overfit smoke ≥3×; health probes pass with **no
  dead temporal block**; param-count table produced.

**Phase B (after the user's run):**
- Training loss (total + cls/obj/iou) trends down; **no NaN/divergence**; LR schedule sane; grad-norms
  stable; VRAM/throughput acceptable on the 5070 Ti.
- A rough **val-mAP sanity** on a few val sequences is **non-degenerate** (confirms learning; **not** the
  final reportable number — that is Stage 7).

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Replicated Mamba-2 forward threads state incorrectly | **Equivalence test (A4.1)** is the gate — must match a full scan to ~1e-3 before anything else proceeds. |
| Mamba-2 head config invalid for a stage dim | Finding §8 removes `d_model=64`; remaining dims (128/256/512) all yield integer head counts at `headdim=64`. Asserted in unit tests. |
| Conv-state seeding wrong at boundaries | Conv state is carried (last `d_conv-1` inputs as left context); the equivalence test (A4.1) catches any boundary mismatch to ~1e-3. |
| BN over `L·B` destabilises training | Monitor in Phase B; GroupNorm / freeze-BN documented fallback. |
| Batch size changes mid-sequence breaks state cache | Keep B constant across carried subsequences / reset on change (Phase B config). |
| Short-run dynamics don't transfer to full run | Short run validates *dynamics only*; Stage 7 owns the final number. |

## 9. Carried-forward prerequisite status (post-Stage-6)

- **Parity (Stage-3 §9.1):** RESOLVED by this stage (unified Mamba-2 TBPTT).
- **State-cache batch constraint (§9.2):** addressed by Phase-B config (constant B / reset on change).
- **BN over `L·B` (§9.3):** monitored in Phase B; fallback documented; revisit at Stage 7 if needed.
- **Finding §8 (dead stage-1 temporal):** RESOLVED by A3.
- **Full Gen1 train split:** user download (Phase B precondition).
