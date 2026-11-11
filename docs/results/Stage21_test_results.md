# Stage 21 — SpikingSSM test-set results (Gen1 test, COCO AP x 100) — 2026-10-16

Single seed per arm; all evaluations captured for the bootstrap (`results/stage21_bootstrap/captures/*.pkl`).
**No confidence intervals yet**: every "gap" below is a point estimate until the paired bootstrap has run
(`docs/runbooks/Stage21_bootstrap_ghostmachine.md` step 5). Logs: `results/stage21_test_eval/<arm>/` (spiking arms),
`results/stage14_test_eval/test_eval_20261016_130130.txt` (PureSSM 100k anchor). Collected by
`thesis/figures/test_metrics.py` → `results/report_figs/test_metrics.json`.

## 1. The 100k ladder (all `[2,3,4]`, 100k OneCycle, same recipe)

| Arm | AP | AP_50 | AP_S | AP_M | AP_L | car | ped | step cost |
|---|---|---|---|---|---|---|---|---|
| PureSSM anchor (`igk6myyb`) | **42.52** | — | 34.03 | 49.68 | 42.62 | 60.4 | 24.6 | — |
| analog (`sx14rpo8`) | 42.25 | — | 33.24 | 49.72 | 42.41 | 59.7 | 24.8 | **−0.28** LIF dynamics |
| graded (`rnak3sms`) | 40.72 | — | 31.36 | 48.48 | 38.41 | 59.1 | 22.4 | **−1.52** sparsity |
| spike (`5ykgl80y`) | 40.18 | — | 31.30 | 47.42 | 41.14 | 58.8 | 21.6 | **−0.54** binarisation |

- Step costs are differences of the unrounded APs (42.522 → 42.246 → 40.725 → 40.185), so they can differ by 0.01
  from differences of the rounded table values.
- Total spiking cost at 100k: −2.34. Pre-registered ordering analog ≥ graded > spike **holds on test**.
- Same pattern as validation (−0.4 / −1.4 / −0.7): the largest step is **sparsity**, not binarisation. The
  binarisation cost was 3.5 at 25k (validation, Stage 19) and 0.5–0.7 at 100k ⇒ mostly a convergence-speed cost.
- The cost falls mainly on pedestrians (PureSSM → spike: car −1.6, pedestrian −3.0).
- Val → test: −1.7 (PureSSM), −1.6 (analog), −1.7 (graded), −1.6 (spike): uniform, so the ladder transfers.

## 2. The 400k models (best-validation checkpoint of each)

| Model | AP | AP_S | AP_M | AP_L | car | ped |
|---|---|---|---|---|---|---|
| S5-RVT (published ckpt) | **47.72** | 38.81 | 54.92 | **50.66** | 63.9 | 31.6 |
| EventSSM (`8zotrwjw`, 320k) | 46.22 | 37.42 | 53.68 | 44.70 | 61.7 | 30.8 |
| PureSSM (`aekvsalq`, 310k) | 46.43 | 37.50 | 53.96 | 47.65 | 62.8 | 30.0 |
| **SpikingSSM graded (`a43jctgx`, 350k)** | **46.24** | 37.38 | 53.54 | **49.63** | 61.9 | 30.5 |

- **Graded SpikingSSM 400k = 46.24**, 0.19 below its ANN twin PureSSM (46.43) and level with EventSSM (46.22).
  The graded-vs-PureSSM gap shrinks from −1.80 at 100k to −0.19 at 400k ⇒ most of the 100k spiking cost is slower
  convergence, not a capacity limit (to be confirmed by the paired bootstrap interval).
- **AP_L 49.63** is the highest of our models (+1.98 over PureSSM, +4.93 over EventSSM) and 1.03 below S5-RVT.
  Validation AP_L is noisy (±4 between neighbouring checkpoints), but graded's validation AP_L was above
  EventSSM's at 16 of the 20 shared validation points and at every one from 240k (`fig_training_400k` (b); corrected
  2026-10-16, it is not above at 40k, 100k, 120k and 220k), so the EventSSM comparison is systematic; the
  PureSSM comparison needs the bootstrap (and PureSSM's curve, W&B login pending).
- Val → test: graded 47.28 → 46.24 (−1.0), PureSSM 47.53 → 46.43 (−1.1).

## 3. Training-time observations (`fig_training_400k*`, `fig_spiking_firing`)

- Loss curves of EventSSM and graded 400k overlap within noise throughout; no instability after the resumes.
- Graded 400k took **55.5 h** of training wall-clock vs EventSSM's 27.9 h on the same RTX 5070 Ti (activation
  recomputation + the per-timestep LIF loop): a training-cost limitation to report.
- Threshold-crossing rates diverge by arm: spike rises to 0.36–0.48, graded falls to 0.10–0.28 (100k) and settles at
  0.15–0.28 (400k); analog 0.18–0.51. The graded readout is the sparsest; the spike arm compensates for
  binarisation by firing more. Relevant to the SOP energy accounting (Stage 22).

## 4. Firing pattern on a real test recording (Stage 22 item 5, 2026-10-16)

`code/event_ssm/scripts/stage22_firing_pattern.py`, 100k checkpoints, recording `18-03-29_13-15-02_122500000_182500000`
(1200 frames = 60 s, mean 48.7k events/frame); outputs `results/stage22/firing/{spike,graded}_s234_*.{json,png,svg}`.

| Arm | stage | mean rate | min–max | r with events/frame |
|---|---|---|---|---|
| graded | 2 | 0.281 | 0.244–0.293 | **+0.64** |
| graded | 3 | 0.123 | 0.015–0.158 | +0.13 |
| graded | 4 | 0.100 | 0.091–0.131 | −0.00 |
| spike | 2 | 0.480 | 0.399–0.495 | −0.32 |
| spike | 3 | 0.356 | 0.323–0.364 | +0.22 |
| spike | 4 | 0.391 | 0.349–0.393 | +0.13 |

After the first 32 frames (state warm-up from zero, the SOP benchmark's convention; used by the report figure
`fig_firing_pattern` and the thesis text, 2026-10-16): graded 0.281 / 0.125 / 0.100, range 0.268–0.293 / 0.095–0.158 /
0.097–0.102, r **+0.65** / +0.00 / +0.01; spike 0.480 / 0.357 / 0.391, range 0.467–0.495 / 0.350–0.364 / 0.389–0.393,
r −0.41 / +0.21 / +0.15. The warm-up transient caused graded stage 3's r = 0.13 above; excluding it strengthens the
finding.

- Rates at inference match the training-time monitor (`fig_spiking_firing`), so the monitor is representative.
- **The sparsity is structural, not event-driven.** Only graded stage 2 tracks scene activity (r = 0.64), and even
  there the rate moves within 27–29 %; stages 3–4 and every spike stage are flat. The readout does not fall silent
  when the scene is quiet, because the dense Mamba state upstream keeps driving it. Consequence for the energy
  argument (Stage 22 SOP): savings come from a fixed low rate (graded ≈ 10–28 %), not from input-dependent activity;
  the spike arm at 36–48 % saves little. Discussion point; also a design hint (event-gated input to the readout).
- Report-style figure: `thesis/figures/fig_firing_pattern.py` → `fig_firing_pattern.{pdf,png}` (colour = arm, one panel
  per stage, warm-up shaded).

## 5. Stage-22 benchmark + SOP energy accounting, graded 400k (2026-10-16)

Run: `stage10_run_local.sh --models all+spikingssm --spikingssm-ckpt a43jctgx/…350000… --spikingssm-test-ap 0.4624`
(VS Code closed for the idle guard; sha `516710c`, dirty tree = only the user's `draft_main.tex` edit). All four
models re-measured in one session. Results `results/stage22/graded_400k_with_ann/` (renamed from the default
`graded_s234_with_ann/` so a later graded-100k run cannot overwrite it): `bench_results.json`, `efficiency_table.md`,
`sop_table.md`, figures.

| | S5-RVT | EventSSM | PureSSM | SpikingSSM graded 400k |
|---|---|---|---|---|
| Params | 18.19 M | 19.18 M | 16.33 M | 16.34 M |
| GFLOPs | 11.06 | 12.71 | 10.03 | 10.03 |
| Latency p50 (eager, full) | 16.93 ms | 12.09 ms | 26.46 ms | 27.19 ms |
| Backbone p50 | 10.41 | 5.19 | 19.64 | 20.30 |
| Energy / frame (GPU, measured) | 0.701 J | 0.425 J* | 0.641 J | 0.552 J* |
| State / stream | 4.7 MB | 144.5 MB | 144.5 MB | 145.6 MB |

\* idle-power drift flagged (idle 48.0 → 43.1 W for SpikingSSM, 52.3 → 44.5 W for EventSSM): with the post-run idle
the SpikingSSM figure would be 0.686 J, so the measured GPU energy is 0.55–0.69 J, overlapping PureSSM. **No GPU energy
difference is claimed.** The ANN latencies are 1–4 % below the Stage-16 run (12.60 / 17.14 / 26.86 ms): session-to-
session variation; compare models within one session only.

- The spiking model costs **+0.73 ms** over PureSSM (the per-timestep LIF loop) and **+1.1 MB** of state (membrane).
- **SOP accounting (operation-count estimate, Horowitz 45 nm constants):** spikes feed 52.4 M of 5.00 G MACs per frame
  (**1.05 %**); at the measured nonzero rates (s2 0.263, s3 0.140, s4 0.157 over 16 test clips; graded ⇒ priced as
  sparse MACs) the driven ops are 10.1 M ⇒ estimated **saving 0.84 %** (22.81 vs 23.01 mJ/frame; ceiling 1.04 %).
  Confirms the pre-registered bound: the energy case for choice C is not an op-count saving on dense hardware; it is
  neuromorphic deployment of the temporal readout (future work). Memory traffic is not modelled.

## 6. Overnight 2026-10-17 chain: bootstrap CIs, threshold sweeps, per-arm SOP (read 2026-11-10)

Chain: `code/event_ssm/scripts/overnight_2026-10-17_{gpu,bootstrap}.sh` (sha `279a8d8`). **Three of five parts
finished; two did not** (see 6.4).

### 6.1 Bootstrap CIs on the 100k ladder (`results/stage21_bootstrap/ladder_100k/bootstrap.{md,json}`)

Paired sequence-level bootstrap, B = 1000, seed 0, 470 test recordings, 95 % percentile intervals (test-set sampling
uncertainty only; one training seed per arm). Point estimates reproduce §1 exactly.

| Rung (b − a) | What it costs | ΔAP [95 % CI] | ΔAP_S | ΔAP_M | ΔAP_L |
|---|---|---|---|---|---|
| PureSSM → analog | LIF dynamics (leak + reset) | **−0.28 [−0.63, +0.05]** | −0.78 [−1.08, −0.45] | +0.03 [−0.74, +0.70] | −0.21 [−2.90, +3.01] |
| analog → graded | sparsity | **−1.52 [−1.98, −1.06]** | −1.88 [−2.24, −1.44] | −1.23 [−1.91, −0.53] | −4.00 [−7.93, +0.45] |
| graded → spike | binarisation | **−0.54 [−1.07, −0.07]** | −0.06 [−0.63, +0.32] | −1.06 [−1.87, −0.37] | +2.74 [−0.14, +6.19] |

- The pre-registered ordering **analog ≥ graded > spike holds with both strict steps' intervals excluding zero**.
  The dynamics rung's interval includes zero ⇒ the LIF dynamics cost is not distinguishable from zero on this test set.
- Sparsity is the dominant cost (≈ 3× binarisation). AP_L intervals are wide (±3–4 AP) ⇒ no AP_L claim per rung.
- Per-arm AP with CIs: PureSSM 42.52 [40.40, 44.39], analog 42.25 [40.14, 43.99], graded 40.72 [38.66, 42.33],
  spike 40.18 [38.09, 41.88]. Per-arm intervals overlap; only the *paired* gaps are tight.

### 6.2 Post-hoc threshold sweeps, 100k spike + graded (`results/stage22/threshold_sweep/`)

Same weights re-evaluated at θ ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} (trained at θ = 1.0 only). θ = 1.0 reproduces the
canonical test AP for both arms (40.18 / 40.72) ⇒ preflight identity holds. Figure: `threshold_pareto.{png,svg,json}`.

| θ | spike AP | spike rate | graded AP | graded rate |
|---|---|---|---|---|
| 0.5 | 38.57 | 47.6 % | 39.87 | 26.5 % |
| 0.75 | 39.97 | 44.5 % | 40.68 | 21.1 % |
| **1.0** | **40.18** | **41.5 %** | **40.72** | **17.3 %** |
| 1.5 | 37.84 | 35.8 % | 38.66 | 12.2 % |
| 2.0 | 28.35 | 30.9 % | 34.61 | 8.9 % |
| 3.0 | 23.04 | 23.7 % | 27.79 | 5.3 % |

(rate = spike-weighted nonzero rate, i.e. the fraction of spike-fed operations driven.)
- **The trained threshold is the AP maximum for both arms**; lowering θ adds firing without accuracy, raising θ cuts
  firing at a steep accuracy cost (spike −2.3 AP at θ = 1.5, −11.8 at θ = 2). No post-hoc operating point dominates
  the trained one ⇒ a lower-rate operating point would need retraining at that θ, not a post-hoc change.
- The binary arm fires ≈ 2.4× more than graded at θ = 1 (41.5 vs 17.3 %): binarisation is compensated by rate.

### 6.3 Per-arm SOP benchmarks, 100k (`results/stage22/{analog,graded,spike}_100k/`)

Spiking model only, one session, VS Code closed, no idle-drift flag. Tables `efficiency_table.md` + `sop_table.md`
(written 2026-11-10 by `stage10_report.py --json … --out …`).

| Arm | Op class | Nonzero rate s2 / s3 / s4 | Driven ops (M of 52.4) | Est. mJ/frame (dense twin 23.007) | Saving | p50 full | GPU J/frame |
|---|---|---|---|---|---|---|---|
| analog | MAC | 1.000 / 1.000 / 1.000 | 52.43 | 23.009 | −0.01 % | 27.34 ms | 0.716 |
| graded | sparse MAC | 0.257 / 0.125 / 0.100 | 9.05 | 22.809 | 0.86 % | 27.10 ms | 0.704 |
| spike | AC | 0.484 / 0.357 / 0.391 | 21.74 | 22.787 | 0.96 % | 27.52 ms | 0.714 |

- Ceiling 1.04 % (spike-fed share 1.05 % of 5.00 G MACs) for every arm. The energy side of the ladder is
  analog ≈ 0 → graded 0.86 % → spike 0.96 %: the spike arm saves slightly more than graded despite 2.4× the firing
  because its ops are accumulates (0.9 pJ) rather than MACs (4.6 pJ).
- Latency and measured GPU energy are identical across arms within noise (27.1–27.5 ms, 0.70–0.72 J) ⇒ on a GPU the
  sparsity is invisible, as pre-registered. Do not compare these absolute numbers with §5 (different session).
- ⚠️ The `mAP/GFLOP` column in these per-arm tables is AP-fraction / GFLOP (0.040–0.042) because `--spikingssm-test-ap`
  was given as a fraction; in the report's convention (AP × 100) it is 4.21 / 4.06 / 4.01.

### 6.4 What did not finish

1. **True-rate sweeps (graded 100k, spike 100k, graded 400k, PureSSM 100k): no output.** `data/gen1_stage9/` (the
   raw test `.dat.h5` and the `preproc_tr/` 2×/10× renders) no longer exists on GhostMachine, so
   `stage9_truerate_sweep.sh` found 0/469 rendered sequences and skipped every eval (it skips unrendered rates by
   design, so the chain moved on silently). Redo = `stage9_prep_test.sh` (extract `data/test_{a,b}.7z` + dat→h5) →
   `stage9_render_truerate.sh` (2× ≈ 26 GB, 10× ≈ 35 GB) → the four sweeps (≈ 3 h each at 10×).
2. **`models_400k` bootstrap: no output.** It failed at once with `CaptureMismatch: graded vs s5rvt: different
   recording/element layout` (found 2026-11-10; the machine going down later that morning was unrelated). Cause:
   Part A numbers recordings per (worker, batch slot) in order of appearance, and S5-RVT was evaluated with RVT's
   recipe (batch 8), so the same 30 605 elements / 470 recordings / 66 304 boxes arrive in another interleaving under
   other ids. Our three 400k captures share one layout. **Fix (commit `5af0b20`):** `--align-by-labels` matches each
   recording by its ground-truth sequence (all 470 are unique, verified on the real captures) and reorders into the
   first capture's layout, then re-runs the alignment check; 3 new unit tests. Reordering moves a reordered arm's
   full-sample AP by ≈ 1e-5 (COCO breaks score ties by list order), so PureSSM goes first as the reference: the own
   models keep their exact official values and S5-RVT gives 47.7155 / AP_L 50.66 (same to 2 dp). Redo:
   `code/event_ssm/scripts/overnight_2026-11-10_bootstrap.sh`.

## 7. Final-results chain 2026-11-10 (`code/event_ssm/scripts/overnight_2026-11-10_{gpu,bootstrap}.sh`, logs `results/overnight_2026-11-10/`)

Read 2026-11-10 22:20. The true-rate test set was rebuilt first (raw test events re-extracted and converted, then
true 10× = 5 ms and true 2× = 25 ms windows rendered, 469/469 recordings each; `data/gen1_stage9/preproc_tr/`).
Chain finished 2026-11-11 02:47 (`ALL DONE`); each 100k sweep took ≈ 2 h 20 min (2× ≈ 27 min, 10× ≈ 1 h 51 min).

### 7.1 Bootstrap CIs on the four 400k models (`results/stage21_bootstrap/models_400k/bootstrap.{md,json}`)

Same method as §6.1 (paired, sequence-level, B = 1000, seed 0, 470 recordings, 95 % percentile; `--align-by-labels`,
PureSSM as the layout reference). Point estimates reproduce §2.

| Gap (b − a) | ΔAP [95 % CI] | ΔAP_L [95 % CI] |
|---|---|---|
| S5-RVT → EventSSM | **−1.49 [−2.59, −0.59]** | **−5.95 [−9.33, −1.94]** |
| S5-RVT → PureSSM | **−1.29 [−2.30, −0.23]** | −3.01 [−7.72, +2.38] |
| S5-RVT → graded 400k | **−1.48 [−2.54, −0.53]** | −1.03 [−5.43, +3.64] |
| EventSSM → PureSSM | +0.20 [−0.12, +0.72] | +2.94 [−0.09, +6.35] |
| PureSSM → graded 400k | −0.19 [−0.66, +0.09] | +1.98 [−0.40, +3.59] |

- **S5-RVT is better overall than each of our three models** (≈ 1.3–1.5 AP, every interval excludes zero).
- **EventSSM, PureSSM and graded 400k are tied overall** (both intervals include zero) ⇒ at the full schedule the
  graded spiking readout costs nothing measurable relative to PureSSM.
- **AP_L:** EventSSM's large-object deficit to S5-RVT is significant; PureSSM's and graded's are not (intervals
  include zero). The EventSSM → PureSSM AP_L gain +2.94 has an interval that just touches zero (−0.09) ⇒ the
  receptive-field result is **suggestive, not significant at 95 %** on this test set. ⚠️ Wording "hypothesis CONFIRMED"
  (Stage 15 notes, draft) must be softened to "consistent with / closes the gap to within test-set noise".

### 7.2 True-rate robustness incl. ConvLSTM (uncompensated; retention = AP at rate ÷ the model's own 1× AP)

| Model | 1× | true 2× (25 ms) | ret. 2× | true 10× (5 ms) | ret. 10× |
|---|---|---|---|---|---|
| RVT ConvLSTM (published ckpt) | 47.14 | **46.08** | **97.7 %** | **37.84** | **80.3 %** |
| S5-RVT (Stage 9) | 47.72 | 45.37 | 95.1 % | 29.67 | 62.2 % |
| EventSSM (Stage 9) | 46.22 | 44.13 | 95.5 % | 29.09 | 63.0 % |
| PureSSM 400k (Stage 16) | 46.43 | 44.53 | 95.9 % | 32.38 | 69.7 % |
| graded 400k | 46.24 | 43.88 | 94.9 % | 30.16 | 65.2 % |
| PureSSM 100k | 42.52 | 40.88 | 96.1 % | 29.40 | 69.1 % |
| analog 100k | 42.25 | 40.53 | 95.9 % | 28.54 | 67.6 % |
| graded 100k | 40.72 | 38.81 | 95.3 % | 25.73 | 63.2 % |
| spike 100k | 40.18 | 39.04 | 97.2 % | 28.26 | **70.3 %** |

PureSSM 100k (`igk6myyb` step 100k, verified in the log): `results/stage14_test_eval/test_eval_20261111_{003236,005902}.txt`.
Logs: ConvLSTM `results/stage9_rvt_lstm/rvt_lstm_{gen1,preproc_tr_dt=25,preproc_tr_dt=5}_20261110_*.log`;
spiking arms `results/stage21_test_eval/{graded,spike,analog}_s234_dt{25,5}/` (analog 100k run 2026-11-11 09:09–11:26, ckpt `sx14rpo8` step 100k); sweep summaries `results/stage9/sweep_tr/`. PureSSM 100k 2× is 96.1 % (40.884 / 42.522 = 96.148 %; an earlier version of this table had 96.2 %).

- ⚠️ **The ConvLSTM is the most rate-robust model at both rates**, in retention *and* in absolute AP at 10×
  (37.84 vs 32.38 for PureSSM, the best SSM). Its 1× score reproduces the RVT paper (47.14 vs 47.2), and it ran on
  the same rendered test set as every SSM. **The Zubić et al. ConvLSTM figure (8.35 AP, 17.7 %) is not reproduced** —
  the second of their 10× numbers that does not reproduce under the published artefacts (after S5-RVT 39.84).
- ⇒ The claim "SSM recurrence is more rate-robust than a ConvLSTM" is **false on our measurements** and must be
  removed wherever it relies on the 17.7 % figure (draft abstract/§5.4, report Ch.4–6, INRC proposal already dropped
  it). What survives: (a) Δt compensation is not reproducible (`tab:dt_comp`); (b) among the SSMs, PureSSM is the most
  robust; (c) the spiking readout keeps the SSM-level robustness (graded 400k 65.2 % vs PureSSM 69.7 %, graded 100k
  63.2 %).
- Note ConvLSTM AP_L rises at 2× (49.14 → 49.62) and falls only to 40.14 at 10×.
- **100k ladder at true 10× (single seed, no CIs; complete since 2026-11-11 11:26):** PureSSM 69.1 % → analog
  67.6 % → graded 63.2 % → spike 70.3 %. Ladder steps in absolute AP, 1× vs 10× (from unrounded APs):

  | Step | 1× | true 10× |
  |---|---|---|
  | PureSSM → analog (LIF dynamics) | −0.28 | −0.86 |
  | analog → graded (sparsity) | −1.52 | −2.81 |
  | graded → spike (binarisation) | −0.54 | **+2.53** |

  The dynamics and sparsity costs grow with the rate; the binarisation step changes sign. Retention steps
  −1.6 / −4.4 / +7.2 pp (unrounded). Sparsity (graded) costs the most robustness, but binarisation wins it back:
  spike beats graded at 10× by **+2.5 AP absolute (28.26 vs 25.73)** although it is 0.54 AP lower at 1×, and its
  retention is the highest of all SSM models. Same pattern at 400k is untested (no spike 400k). Hypothesis only: a
  binary spike has a fixed amplitude, so when sparser true-rate frames shrink the membrane input, the spike output
  degrades by count, whereas the graded output (s·m) also shrinks in amplitude. The analog arm fits this reading
  (dense membrane value, shrinks in amplitude only: −0.86 AP at 10×, less than graded's −2.81) but does not test it.
  Schedule length barely changes robustness (PureSSM 69.1 % at 100k vs 69.7 % at 400k; graded 63.2 % vs 65.2 %).
  Figure `fig_robustness_spiking` (redrawn 2026-11-11 as one x position per arm, 2× → 10× drop, so analog yellow is
  never a line next to PureSSM orange).
