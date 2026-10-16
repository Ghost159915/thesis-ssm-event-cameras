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
| analog (`sx14rpo8`) | 42.25 | — | 33.24 | 49.72 | 42.41 | 59.7 | 24.8 | **−0.27** LIF dynamics |
| graded (`rnak3sms`) | 40.72 | — | 31.36 | 48.48 | 38.41 | 59.1 | 22.4 | **−1.53** sparsity |
| spike (`5ykgl80y`) | 40.18 | — | 31.30 | 47.42 | 41.14 | 58.8 | 21.6 | **−0.54** binarisation |

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
  Validation AP_L is noisy (±4 between neighbouring checkpoints), but graded's validation AP_L stayed above
  EventSSM's for the whole 400k run (`fig_training_400k` (b)), so the EventSSM comparison is systematic; the
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

- Rates at inference match the training-time monitor (`fig_spiking_firing`), so the monitor is representative.
- **The sparsity is structural, not event-driven.** Only graded stage 2 tracks scene activity (r = 0.64), and even
  there the rate moves within 27–29 %; stages 3–4 and every spike stage are flat. The readout does not fall silent
  when the scene is quiet, because the dense Mamba state upstream keeps driving it. Consequence for the energy
  argument (Stage 22 SOP): savings come from a fixed low rate (graded ≈ 10–28 %), not from input-dependent activity;
  the spike arm at 36–48 % saves little. Discussion point; also a design hint (event-gated input to the readout).
- The figure from the tool is not yet in the report style (its stage colours reuse model colours; it has a title):
  restyle before it goes into the report.
