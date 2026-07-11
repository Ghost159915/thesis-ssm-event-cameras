# Stage 9 — Zubić (2024) frequency-generalisation methodology: verdict

**Date:** 2026-06-20 · **Source:** deep-research workflow (`wvxnhx4lv`), ~25 verified primary
sources, cross-checked against the paper, its appendix, and the vendored repo. Settles why our
S5-RVT reproduction degraded ~9.3 mAP at 4× while the paper reports far less.

## Question
Does Zubić et al. 2024 ("State Space Models for Event Cameras", CVPR; repo `uzh-rpg/ssms_event_cameras`)
run its event-rate degradation experiment by (a) coupling accumulation window = frame stride (true
rate change), (b) fixing the stride and varying only the window, or (c) rescaling the SSM's
discretisation step Δt at inference without re-rendering?

## Verdict: it does **(b) AND (c) together** — we did only (b)

**Data side — identical to our sweep.** The repo's `scripts/genx/conf_preprocess/extraction/frequencies/const_duration_*hz.yaml`
change only the accumulation **window** (`method: DURATION`, `value` ms = 1000/Hz → 200/100/25/12/10/5 ms
for 5/10/40/80/100/200 Hz; training base = 50 ms = 20 Hz). The representation **stride** is hardcoded
`ts_step_ev_repr_ms = 50` in `preprocess_dataset.py` and is **decoupled** from the window. So the paper
also uses fixed-50 ms-stride + variable-window (overlapping at low Hz, gapped at high Hz). **Our renders
already match the paper's data methodology — the gaps/overlap are inherent to the method, not a bug, and
no re-render is required.**

**Model side — what we omitted.** The S5 SSM rescales its continuous-time step Δt at inference by the
new/old sampling-rate ratio via **`step_scale`** (`RVT/models/s5/s5_model.py`, `forward`/`forward_rnn`:
`step = step_scale * torch.exp(self.log_step)`, default `1.0`). Paper §4.3.3: *"rate r is halved for
double the inference frequency"*; the RVT/ConvLSTM baseline receives **no** compensation. The robustness
mechanism is the **combination** of re-rendered shorter windows **plus** the SSM-side Δt rescaling (plus
anti-aliasing output masking / H2-norm regularisation).

**Reported numbers (Gen1):** S5 **47.71 → 39.84** over 20 → 200 Hz; RVT (ConvLSTM) **47.16 → 8.35**.

## Root cause of our gap (confirmed in code)
`grep step_scale` across `RVT/` **outside** `s5_model.py` returns nothing → `step_scale` is **never
plumbed** through the detection/eval path; every call uses the default `1.0`. Our S5-RVT eval therefore
ran with no Δt compensation, which is exactly why it dropped to 38.4 at 4× (12.5 ms window). The single
missing ingredient is the inference-time `step_scale`.

## Corrected fix (supersedes the earlier "re-render stride=window" plan)
1. **Plumb `step_scale = freq_ratio`** from an eval/CLI argument → detector backbone → each S5 block →
   `s5_model.forward(step_scale=…)`. Derive the exact convention from §4.3.3 + `s5_model` and validate
   empirically.
2. **Re-eval the S5-RVT baseline** with `step_scale` per rate on the EXISTING window renders.
   *Falsifiable gate:* S5 should recover from 38.4 toward the paper's ~40–44 at the fast extreme.
3. **EventSSM (Mamba) analog:** `mamba-ssm` has no `step_scale`; implement the equivalent by scaling the
   selective-scan `delta` (or `dt_bias`) by the same ratio at inference.
4. **Research question:** compare degradation **with vs without** Δt-rescaling, S5 vs Mamba — does Mamba's
   input-dependent Δt auto-adapt, or does it also need the explicit knob? That contrast is the novel
   temporal-generalisation contribution.

*No re-rendering is needed; the current fixed-stride window sweep is retained as the valid
"no-compensation" baseline curve.* Full workflow output: task `wvxnhx4lv`.

---

## ADDENDUM (2026-07-10) — the fix above was tested and FALSIFIED; corrected two-regime methodology

**Empirical result.** `step_scale` was plumbed into the detection path (env hook in `S5Block`,
`docs/patches/s5_step_scale_hook.patch`) and evaluated on the fixed-stride 4× (12 ms-window) test set
with the frozen `gen1_base.ckpt`:

| step_scale @4× (fixed 50 ms stride) | test/AP |
|---|---|
| 0.24 (= window/50, the convention recommended above) | 0.3015 |
| **1.0 (no compensation)** | **0.3842** |
| 4.1667 (= 50/window, inverted probe) | 0.2940 |

1× anchor with the hook at 1.0 reproduced 0.47689520442617495 **byte-identically** → the hook is inert;
the degradation is the knob's own effect. The response is a symmetric hill peaked at 1.0: **under
fixed-stride rendering there is nothing for Δt-rescaling to compensate** — the model's step cadence
never changes; only per-frame event mass does, and no Δt rescale restores missing events. Small objects
suffer most under a mistuned Δt (pedestrian AP 31.5 → 15.3 at ss=4.17).

**What this falsifies.** The verdict's data-side conclusion ("the paper's frequency experiments use the
same fixed-stride renders as ours; only `step_scale` was missing") cannot be correct for the paper's
*frequency-generalisation* experiments. Corroborating code evidence: the stride is hardcoded —
`preprocess_dataset.py:920`: `ts_step_ev_repr_ms = 50  # Could be an argument of the script.` — so the
repo **as shipped cannot produce a true rate change at all**, and the paper's reported RVT/ConvLSTM
collapse to 8.35 mAP at 200 Hz is only coherent if the model genuinely stepped ~10× faster (our
fixed-stride 4× only cost the S5 baseline ~9 mAP with no compensation). Conclusion: the paper's
frequency runs almost certainly used locally modified preprocessing with **stride = window** (true rate
change), and `step_scale = window/50` is the correct convention **in that regime only**.

**Corrected methodology (regime 2, adopted 2026-07-10).** Render true-rate test sets with the new
`TS_STEP_EV_REPR_MS` env hook (`docs/patches/preprocess_full_stage9.patch`): stride = window, which
tiles the event stream gap-free. Gen1's 250 ms label grid constrains valid strides to {25 ms = 2×,
10 ms = 5×, 5 ms = 10×} (4×/12 ms impossible — another sign the paper's grid handling was modified).
1× true-rate ≡ the existing dt=50 render (window = stride = 50), reused as the shared anchor.
Launcher: `stage9_render_truerate.sh` (→ `data/gen1_stage9/preproc_tr/`); eval grid:
`stage9_truerate_sweep.sh` ({S5, EventSSM} × {no-comp, comp = window/50} at 2× and 10×; paper targets
@200 Hz: S5+comp 39.84, RVT 8.35). The Mamba-side analog (post-softplus delta scaling,
`MAMBA_STEP_SCALE` in `code/event_ssm/temporal/`) passed 4/4 parity tests.

**Standing interpretation.** The fixed-stride sweep is *retained* as the deployment-relevant
fixed-cadence regime (regime 1), where Δt-compensation is now *measured* to be inert-to-harmful and
robustness is governed by input sparsity. The thesis chapter reports both regimes; the novel question
is whether Mamba's input-dependent Δt self-adapts in regime 2 where S5 requires the explicit knob.

---

## ADDENDUM 2 (2026-07-11) — regime-2 results: the mechanism does not reproduce in ANY regime

The full regime-2 grid ran overnight 2026-07-10→11 ({S5-RVT, EventSSM} × {no-comp, comp=window/50}
at true 2× and 10×; all compensated logs carry in-log engagement proofs; anchors shared with regime 1):

| true rate | S5 no-comp | S5 comp | EventSSM no-comp | EventSSM comp |
|---|---|---|---|---|
| 1× (anchor) | 47.69 | =47.69 | 46.20 | =46.20 |
| 2× (25 ms) | 45.37 | 44.51 | 44.13 | 43.67 |
| **10× (5 ms)** | **29.67** | **19.96** | **29.09** | **20.18** |

**Findings.**
1. **`step_scale` compensation failed at every tested point in both regimes** (7 compensated evals:
   4×-fixed both directions; 2× and 10× true-rate paper convention). At 10× it *costs ~10 mAP for
   both architectures* (S5 29.67→19.96; Mamba 29.09→20.18) — opposite sign to the paper's claim, and
   architecture-independent (the two compensated curves land within 0.2 mAP of each other), implicating
   the shared discretisation-rescaling math itself rather than either implementation.
2. **The paper's Gen1 200 Hz headline (S5+comp = 39.84) is not reproducible** with the published
   checkpoint + published code under any tested setting; our best 10× result (no compensation, 29.67)
   sits ~10 mAP below it, and their published preprocessing cannot even generate a true-rate render
   (the line-920 stride hardcode, Addendum 1).
3. **The paper's *conclusion* survives with the mechanism reassigned:** uncompensated SSM recurrence
   is itself strongly rate-robust — both models retain ~63% of their 1× AP at a true 10× shift
   (S5 62.2%, Mamba 63.0%) vs the paper's ConvLSTM at 17.7% (8.35/47.16). Robustness is intrinsic to
   the state-space recurrence, not conferred by inference-time Δt-rescaling.
4. **Mamba self-adaptation: consistent but marginal.** EventSSM retains slightly more than S5 at every
   uncompensated point (63.0% vs 62.2% @10×; −4.5% vs −4.9% @2×) and is less damaged by knob
   mistuning at 4×-fixed and 2×; at 10× the two are statistically indistinguishable. Claimable as a
   consistently-signed marginal advantage of input-dependent discretisation, not a decisive one.

Figure: `results/stage9/stage9_truerate_curve.{png,pdf}` (4 measured curves + the paper's two
reference points); regime-1 figure: `results/stage9/stage9_degradation_curve.{png,pdf}`;
tables: `truerate_table.csv`, `degradation_table.csv`. Eval count for the chapter: 19
(5 rates × 2 models regime-1 no-comp; 3 compensated probes; 8 regime-2 grid; +2 shared anchors
counted once; 1× rebuild-validation).
