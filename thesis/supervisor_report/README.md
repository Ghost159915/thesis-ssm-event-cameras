# Supervisor progress report — August 2026

Two-page progress report on Thesis B (both original detectors complete) plus the proposed Thesis C
direction (spiking selective SSMs on Intel Loihi 2).

## Files

| File | Purpose |
|---|---|
| `progress_report_2026-08.tex` | The report. Self-contained; standard `article` class. |
| `figs/erf_stage3.png` | Trained receptive-field probe, stage-3 pair (ResNet vs BiMamba) — **used in the report**. |
| `figs/erf_trained.png` | Full 4-panel version of the same probe (stages 3 and 4). Kept for the thesis chapters. |
| `figs/truerate_curve.pdf` | Regime-2 true-rate degradation curve. **Legend differs** from `results/stage9/` — see below. |
| `figs/label_noise.png` | GT-vs-prediction crop showing unlabelled objects the model detects. |

## Compiling

Compiles locally on `GhostMachine` — `pdflatex`, `latexmk` and the Latin Modern fonts are installed:

```bash
cd thesis/supervisor_report
latexmk -pdf progress_report_2026-08.tex   # runs pdflatex twice so \ref resolves
evince progress_report_2026-08.pdf &
```

`latexmk -C` cleans all build artefacts. **Overleaf** also works: zip this folder, set the compiler to
pdfLaTeX and `progress_report_2026-08.tex` as the main document.

## The two-page limit

The layout is tuned to land on **exactly two pages** and is verified by compiling — check with
`pdfinfo progress_report_2026-08.pdf | grep Pages` after any edit. It has very little slack, so adding a
sentence will likely push to a third page. Levers, cheapest first:

1. `figs/label_noise.png` width (`0.92\textwidth`) — the largest single block of space.
2. Subfigure widths in §3 (`0.57` / `0.40\textwidth`).
3. `\parskip` (currently `1.5pt`) and `margin` (currently `0.65in`).
4. Caption text — the captions restate numbers already in the body.

## Figure provenance

`figs/erf_stage3.png` and `figs/label_noise.png` were cropped from committed result artefacts with ffmpeg
(no re-running of any model):

```bash
# erf_stage3.png — stage-3 row extracted from the 2x2 probe output
ffmpeg -i figs/erf_trained.png -filter_complex \
  "[0:v]crop=600:485:25:60,pad=612:485:0:0:white[a];[0:v]crop=600:485:25:548[b];[a][b]hstack=inputs=2[out]" \
  -map "[out]" figs/erf_stage3.png

# label_noise.png — panels t=25.7s and t=51.4s from the Stage-16 GT-vs-pred montage
ffmpeg -i results/stage16/17-04-14_14-59-17_732500000_792500000__puressm_gt_vs_pred_montage.png \
  -filter_complex \
  "[0:v]crop=608:320:1824:160,pad=628:320:0:0:white[a];[0:v]crop=608:320:1216:640[b];[a][b]hstack=inputs=2[out]" \
  -map "[out]" figs/label_noise.png
```

`figs/truerate_curve.pdf` was re-rendered from the same Stage-9 eval logs with the legend labels changed
from `EventSSM (ours, Mamba)` / `PureSSM (ours, BiMamba)` to `EventSSM (Mamba)` / `PureSSM (BiMamba)` —
the report avoids first-person. The plotted data is unchanged (retention 62.2 / 63.0 / 69.7 % reproduced
exactly). `results/stage9/stage9_truerate_curve.pdf` was **deliberately left untouched**, so the committed
Stage-9 artefact still carries the original legend. To regenerate:

```bash
cp code/event_ssm/scripts/stage9_truerate_plot.py /tmp/replot.py
sed -i 's|EventSSM (ours, Mamba)|EventSSM (Mamba)|; s|PureSSM (ours, BiMamba)|PureSSM (BiMamba)|; \
        s|^OUT = REPO / "results/stage9"|OUT = Path("/tmp")|' /tmp/replot.py
~/miniforge3/envs/events_signals/bin/python /tmp/replot.py   # CPU-only, reads existing logs
cp /tmp/stage9_truerate_curve.pdf thesis/supervisor_report/figs/truerate_curve.pdf
```

## Number provenance

Every figure in the report traces to a committed result doc — do not edit numbers here without updating
the source:

- Accuracy table → `docs/results/Stage15_results_comparison.md` (and Stage 8 for EventSSM)
- Receptive field → `code/event_ssm/proofs/out/u5_erf_extent.md`
- Rate robustness → `docs/results/Stage16_results.md` §3, `docs/results/Stage9_TwoRegime_Results.md`
- Efficiency → `docs/results/Stage16_results.md` §1–2
- Label quality → `docs/notes/Gen1_resolution_labels_and_eventvis_notes.md` §3
- Full synthesis → `docs/results/Thesis_Progress_Writeup.md`
