# Report figures — plan, decisions and provenance (started 2026-10-16)

The final report (`thesis/latex/main.tex`, ~50 pp body) had two results figures (trained ERF, large-car
qualitative). The rubric gives 20 % to presentation and asks for figures that are "clear, consistent and
economical" with "no redundant figures", so every figure below must show something its table cannot.

Code: `thesis/figures/` (one script per figure + `report_style.py`); outputs `thesis/latex/images/<name>.{pdf,png}`
(PDF force-added, `*.pdf` is gitignored) and the plotted values `results/report_figs/<name>.csv` (table view).
Data collectors: `extract_val_histories.py` (W&B offline files → `val_histories.json`) and `test_metrics.py`
(canonical test-eval logs → `test_metrics.json`). Run everything in `events_signals` (matplotlib 3.10).

## 1. Figure set (agreed with the user 2026-10-16)

| # | Figure | Section | Status |
|---|---|---|---|
| 1 | `fig_accuracy_by_size`: test AP minus S5-RVT, overall / S / M / L | Results, accuracy | **done**; graded 400k added automatically after its Stage-21 eval; bootstrap error bars once an S5-RVT capture exists |
| 2 | trained ERF | Results | in report |
| 3 | large-car qualitative | Results | in report |
| 4 | `fig_robustness`: retention, regime 1 + regime 2 (± Δt comp., published points) | Results, robustness | **done**; panel (c) spiking arms appears when the Stage-22 true-rate logs exist |
| 5 | `fig_efficiency`: latency breakdown + CUDA-graph marker, energy, state | Results, efficiency | **done**; replaces `results/stage10/stage10_pareto.png` |
| 6 | ladder (test AP per arm + paired gaps with CIs) | §5.7 | tool built (`eval/ladder_plot.py`); needs captures + bootstrap |
| 7 | `fig_training_curves`: (a) 100k arms minus PureSSM, (b) 400k curves | §5.7 / Discussion | **done** |
| 8 | threshold Pareto | §5.7 | tool built; needs the sweep |
| 9 | firing pattern vs event density | §5.7 / Discussion | tool built; needs a GPU run |
| 10 | dense ops vs spike-fed ops (≈ 1 %) | Discussion | to build after the Stage-22 benchmark |
| 11 | spike vs graded on the same frames | Discussion | needs SpikingSSM support in `stage16_dump_predictions.py` |
| 12 | LIF membrane panel | Methodology | exists (`thesis/supervisor_report/figs/fig_lif_panel.png`) |
| 13 | β per channel | Appendix | exists (`results/stage22/beta/beta_100k.png`) |
| 14 | `fig_gantt`: the Thesis-A plan, redrawn (task names were under the first bar in the old PNG) | Appendix C | **done**; bars/milestones measured from the old PNG's pixels; bars split and shaded by term (ordinal sequential blue); `main.tex` + `draft_main.tex` now include `images/fig_gantt.pdf` (old `Ghantt Chart.png` kept for `LitReview_ThesisA.tex`) |
| 15 | `fig_gantt_actual`: the timeline as executed (user request 2026-10-16) | Appendix C, after `fig_gantt` | **done**; see §4 |

Not made, on purpose: a plot of our models against the SNN detectors (the pre-registration rules out leaderboard
framing; `tab:snn_landscape` already gives the context).

## 2. Decisions

- **D1 Colour = model, everywhere.** S5-RVT neutral grey `#6f6e69` (the reference recedes), EventSSM blue `#2a78d6`,
  PureSSM orange `#eb6834`, graded aqua `#1baf7a`, analog yellow `#eda100`, spike violet `#6250d6`; each model also
  has its own marker. Validated with the dataviz palette validator on white (`--pairs all`): blue/orange/aqua pass;
  grey vs those three passes separation (normal-vision ΔE ≥ 17.4; its chroma "fail" is the intended neutral);
  analog/graded/spike pass together (worst CVD ΔE 9.1, normal 22.9).
- **D2 No four-hue line charts.** No ordering of four palette hues passes all-pairs (yellow–orange 13.7,
  pink–orange 12.9, violet–blue 10.5 normal-vision ΔE, all below the 15 floor). A one-hue aqua ramp for the three
  arms also fails as lines (ΔE 7.2). So in ladder figures PureSSM is a reference (zero line or its own x position),
  never a fourth line, and spiking arms get their own panel next to the ANN models.
- **D3 No titles inside figures**; panels carry `(a)`, `(b)` + a two-word note; the caption explains. Sans 8–9 pt
  at the report's text width (6.27 in), vector PDF with TrueType text.
- **D4 Every figure writes the CSV of its plotted values** (the "table view"; also the audit trail).

## 3. Provenance findings made while building the figures

- **PureSSM 400k best val/AP is 0.4753, not 0.48.** "0.48" is the checkpoint filename's two-decimal rounding; the
  exact `best_model_score` was read from the checkpoint's ModelCheckpoint state
  (`results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt`). Graded 400k (0.4728) is therefore 0.25
  below PureSSM on val, not 0.2–1.2. Corrected in `Stage20_fullrun_notes.md` §3.6.
- **Two benchmark runs exist; the report uses the second.** `results/stage10/bench_results.json` now holds the
  three-model Stage-16 run (2026-07-16, sha `0f6271a`, dirty tree): EventSSM 12.60 / S5-RVT 17.14 / PureSSM
  26.86 ms p50, 0.438 / 0.723 / 0.652 J. CLAUDE.md's Stage-10 bullet quotes the earlier two-model citable run
  (sha `561554b`: 14.25 / 19.67 ms, 0.40 / 0.73 J). `draft_main.tex` uses the Stage-16 run throughout (checked: no
  Stage-10-run number in the draft), and so do the figures.
- **CUDA-graph 210 / 181 Hz are correct**: graph network replay (4.16 / 4.92 ms) + eager post-processing
  (0.61 / 0.62 ms) = 4.77 / 5.54 ms (`docs/results/Stage16_results.md`, "graph-deploy"); the JSON's 240 / 203 Hz
  are the network replay alone.
- **Training-curve joins verified**: the restore steps derived from the W&B files (analog 50k; graded 100k 40k,
  80k; graded 400k 180k, 340k) match the Stage-20 notes, and the extracted values match the supervisor-report
  numbers and every value recorded in the notes. EventSSM's Stage-7 run validated every 20k (not 10k).
- **PureSSM 400k training curve is not available locally** (cloud run, W&B online `honest-violet-3`; no W&B login
  on GhostMachine), so `fig_training_curves` (b) shows its best checkpoint as a point. A CSV export of `val/AP`
  from the W&B web UI would let it be drawn as a curve.

## 4. How `fig_gantt_actual` is built (evidence, not recollection)

- **Work-stream rows:** a calendar week is filled when at least one commit of that stream was authored in it. Each
  commit subject is mapped by ordered regex rules (`STREAM_RULES`, first match wins; `chore*` and `refactor(repo)`
  are housekeeping and dropped). 285 of 301 commits map to a stream; the remaining 16 are housekeeping. Every
  commit's mapping is in `results/report_figs/fig_gantt_actual_commits.csv`, so the chart can be audited and the
  rules corrected. Two rule pitfalls found and fixed while building it: "100k/400k" also matched the EventSSM and
  PureSSM 400k-run commits, and "bootstrap" matched the Stage-13 cloud *environment* bootstrap.
- **GPU-training row:** exact spans of every run from the W&B offline folders (folder timestamp = start, newest file
  = end), coloured by model (the PureSSM 100k anchor inside the October ladder is orange); the two cloud PureSSM
  runs (13–15 Jul) only to day precision (no local W&B files).
- **Milestones:** dated results from CLAUDE.md and the stage notes; labels sit in free space beside the week block.
- **Not captured:** work outside the repository, and Thesis-A work before the first commit (15 Apr 2026). Only T3
  is shaded: the T1/T2 2026 term dates are not recorded in the repo, and a web search (2026-10-16) found only
  aggregator sites that contradict the known T3 dates, so they were not used. Add them to `fig_gantt_actual.py`
  if the user supplies them.
- Visible pattern: bursts in early June, mid June, mid July, early September and October; nothing committed
  20 Jul – 6 Sep and 14 Sep – 4 Oct.

