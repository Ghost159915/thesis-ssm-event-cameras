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
