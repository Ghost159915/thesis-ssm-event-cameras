# Thesis Presentation Guide

2026-10-11 · Benas Vaiciulis · exported from the Claude doc [Thesis Presentation Guide](https://claude.ai/code/artifact/11fdd46c-062c-4dbd-a5b9-71e9d1b600ce). The finished 10-slide deck is in [`deck/`](deck/) (live version: [SSMs for Event-Camera Detection](https://claude.ai/artifact/NYt2pws5yf3ydGDcFHSuxy)); this guide was written for a 12-slide plan and maps onto it as noted in [`README.md`](README.md).

## Overview

Plan for 12 slides in 10 minutes: about 2 minutes on the problem, 3 on what you built, 3.5 on results and 1.5 on discussion. Everything hangs on one sentence the audience should leave with:

> **We asked what a state-space model actually contributes to event-camera detection, by changing one part at a time inside a reproduced state-of-the-art pipeline. SSMs see further, stay robust when the event rate changes without the trick the literature credits for it, and can be made to spike at a cost we can measure piece by piece.**

| Part | Slides | Time | Goal for the audience |
| --- | --- | --- | --- |
| Problem | 1–3 | 2 min | Why event cameras, why SSMs, what is unclear |
| What we did | 4–6 | 3 min | Trust the comparison: one change at a time |
| Results | 7–10 | 3.5 min | Four findings, one per slide |
| Discussion | 11–12 | 1.5 min | What it means, limits, what comes next |

Rule of thumb: one idea and one picture per slide, about 50 seconds each. If you run long, cut words from slides 2, 5 and 11 first; never cut a results slide.

## Slide-by-slide script

Each slide lists what to **show** and what to **say**. The spoken lines are prompts, not a script to read word for word.

### Part 1 — The problem (2 min)

1. **Title** (10 s)
   - Show: title, your name, supervisor (Dr Will Midgley), UNSW.
   - Say: one line — "I studied what state-space models actually contribute to object detection with event cameras."
2. **Event cameras** (50 s)
   - Show: `fig_event_camera_principle.png`.
   - Say: each pixel reports brightness changes on its own, with microsecond latency and very high dynamic range. The output is sparse and asynchronous, not frames. That suits small drones, where power and weight are tight.
3. **The gap and the question** (60 s)
   - Show: the research question in large text, plus two short bullets for the two gaps.
   - Say: SSMs were recently proposed as the natural model for event streams (S5-RVT, 47.7 mAP on Gen1). But published comparisons change several things at once, so it is unclear what the SSM itself contributes. Separately, spiking detectors have caught up on accuracy, but nobody has measured where the cost of spiking comes from.

### Part 2 — What I did (3 min)

4. **The method: one change at a time** (60 s)
   - Show: a simple diagram — a fixed pipeline (data → backbone → neck → head → evaluator) with only the backbone box highlighted as swappable.
   - Say: I first reproduced S5-RVT exactly (47.7, matching the paper). Then I kept everything else frozen and swapped only the backbone, so any difference comes from that swap alone.
5. **Four models** (60 s)
   - Show: four boxes in a row — S5-RVT → EventSSM → PureSSM → SpikingSSM — with the one part that changes between neighbours labelled on each arrow.
   - Say: EventSSM uses a CNN to look at space and Mamba for time. PureSSM replaces the CNN with a bidirectional Mamba, so it is SSM all the way. SpikingSSM adds spiking neurons after PureSSM's temporal blocks. Data: Prophesee Gen1, cars and pedestrians, scored with COCO mAP.
6. **The spiking model** (60 s)
   - Show: `fig_spiking_architecture.png`.
   - Say: a leaky integrate-and-fire neuron reads out each Mamba block. Three versions share the same neuron: analog, graded and binary. Stepping PureSSM → analog → graded → binary separates the cost of the neuron dynamics, of sparsity and of binarisation. I fixed the predicted order before running it.

### Part 3 — Results (3.5 min)

7. **Accuracy: the overall scores tie, large objects do not** (50 s)
   - Show: the three-model table (mAP, params, GFLOP) with AP for large objects added; optionally `fig_qualitative_large_car.png`.
   - Say: EventSSM 46.2, PureSSM 46.4, baseline 47.7. The overall difference between my two models is within noise. The real result is large objects: PureSSM gains about 3 AP there, closing half the gap to the baseline, while being the smallest model (16.3 M).
8. **Why: the SSM sees further** (50 s)
   - Show: `fig_erf_trained.png`.
   - Say: a receptive-field probe shows the SSM backbone sees up to twice as far as the CNN, and its field widens during training while the CNN's barely changes. That explains the large-object gain.
9. **Event-rate robustness, and a claim that did not reproduce** (60 s)
   - Show: the true-rate degradation curve (accuracy vs event-rate multiple, one line per model).
   - Say: at ten times the event rate PureSSM keeps 69.7 % of its accuracy, the most of any model. The paper credits SSM robustness to a rescaling trick at inference. Under its published checkpoint and code, that trick did not recover accuracy at any point I tested, and its headline high-rate result was not reproducible. The robustness is real, but it is there without the trick.
10. **What spiking costs** (50 s) — *fill in after the runs*
    - Show: the ladder as a bar chart — PureSSM, analog, graded, binary at 100k steps, with error bars.
    - Say: whether the pre-registered order held, the size of each step (dynamics, sparsity, binarisation), and the firing rate. Current numbers: binary 41.8 validation AP at 100k; graded ahead of binary at matched steps (41.6 vs 40.4 at 80k).

### Part 4 — Discussion and conclusion (1.5 min)

11. **Trade-offs and limits** (50 s)
    - Show: three short bullets.
    - Say: PureSSM does the least computation but is slow in plain PyTorch (26.9 ms) because of many small operations; with CUDA graphs it runs at 181 Hz. The Mamba models carry a large streaming state (144 MB vs 4.7 MB). Every result is a single training run, so small differences are reported as ties. The original drone obstacle-avoidance aim was descoped to make this controlled study possible.
12. **Conclusion and next steps** (40 s)
    - Show: three takeaways and one future-work line.
    - Say: SSMs help through spatial reach, their rate robustness is intrinsic, and the cost of spiking can be split into parts. Next: run the spiking model on Intel Loihi 2 via the NIR format (proposal submitted to Intel's research community), and make the spatial backbone spike too. End on the one-sentence story from the overview.

## Key numbers

Every number the talk uses, from the repo's results files. Test-set numbers unless marked validation.

| Metric | S5-RVT | EventSSM | PureSSM | Source |
| --- | --- | --- | --- | --- |
| Gen1 test mAP (COCO 0.50:0.95) | 47.7 | 46.2 | 46.4 | Stage15_results_comparison.md |
| AP, large objects | 50.7 | 44.7 | 47.7 | Stage16_results.md |
| Parameters | 18.2 M | 19.2 M | 16.3 M | Stage16_results.md |
| GFLOP per frame | 11.1 | 12.7 | 10.0 | Stage16 (FLOPs corrected 2026-10-07) |
| mAP per GFLOP | 4.32 | 3.64 | 4.63 | Stage16_results.md |
| Latency, eager PyTorch | 17.1 ms | 12.6 ms | 26.9 ms | Stage16_results.md |
| Throughput with CUDA graphs | — | — | 181 Hz | Stage16_results.md |
| Energy per frame | 0.72 J | 0.44 J | 0.65 J | Stage16_results.md |
| Streaming state per stream | 4.7 MB | 144 MB | 144 MB | Stage16_results.md |
| Accuracy kept at true 10× event rate | 62.2 % | 63.0 % | 69.7 % | Stage16_results.md |

Other numbers:

- **Reproduction:** 47.7 test mAP, an exact match to the paper; AP50 = 75.3.
- **Receptive field (trained):** SSM vs CNN spread at stage 3 is 85.6 vs 41.8 px; at stage 4, 105.5 vs 84.4 px.
- **Δt-rescaling:** at a true 10× rate it costs 9–13 mAP; the paper's 200 Hz result (39.84) was not reproducible, best achievable 29.67.
- **Spiking, 100k steps (validation AP):** binary 41.8 (final); graded 41.6 vs binary 40.4 at 80k. 25k pilot ladder: PureSSM 35.1, analog 35.1, graded 34.0, binary 30.5.
- **Spiking energy ceiling:** spikes feed about 1 % of the detector's operations.

## Figures to reuse

Five slides have a figure already made; up to three need a new diagram or chart.

| Slide | Figure | Where it is | Status |
| --- | --- | --- | --- |
| 2 | Event camera principle | `thesis/latex/images/fig_event_camera_principle.png` | Ready |
| 4 | Frozen pipeline, backbone swappable | `thesis/diagrams/architecture_concept.png` (check it fits) or a new box diagram | Check / draw |
| 5 | Four models side by side | none yet | Draw (four boxes, one arrow label each) |
| 6 | Spiking architecture and membrane trace | `thesis/latex/images/fig_spiking_architecture.png` | Ready |
| 7 | Large-car qualitative example | `thesis/latex/images/fig_qualitative_large_car.png` | Ready |
| 8 | Trained receptive fields | `thesis/latex/images/fig_erf_trained.png` | Ready |
| 9 | True-rate degradation curve | `results/stage9/stage9_truerate_curve.*` on the workstation (not on the Mac); regenerate with `code/event_ssm/scripts/stage9_truerate_plot.py` | Copy over |
| 10 | Spiking ladder bar chart with error bars | none yet (`python -m event_ssm.eval.ladder_plot`) | Make after the runs |

Thesis-format figures are dense; on slides, crop to the one panel you talk about and enlarge the labels.

## Likely questions

Prepare short answers to these. The first three are the most likely.

1. **"Your models score below the baseline. Why is that a success?"**
   - The aim was to isolate what the SSM contributes, not to top a leaderboard. Within 1.5 mAP of the baseline with the smallest model, and a clear large-object gain explained by the receptive field.
2. **"Is it just noise? Each model was trained once."**
   - Yes for the overall tie (+0.21), and I say so. The large-object gain (about 3 AP) is bigger and has a measured mechanism behind it. For the spiking ladder, give the bootstrap error bars.
3. **"You say a published result doesn't reproduce. Are you sure?"**
   - Tested with the authors' own published checkpoint and code, at many operating points, under two separate protocols. The claim is narrow: not reproducible under the published artefacts. The robustness itself is reproduced.
4. **"What happened to the drone part?"**
   - Descoped deliberately: a controlled four-model comparison was the stronger contribution in the time available. Power and weight limits remain the motivation; deployment on neuromorphic hardware is the next step.
5. **"Does spiking actually save energy here?"**
   - Not much in this design: only the temporal readout spikes, and spikes feed about 1 % of the operations. The contribution is measuring where the accuracy cost of spiking comes from; the energy case needs the spatial backbone to spike too.
6. **"Why is PureSSM slow if it does the least work?"**
   - It runs many small operations, so on a GPU it waits on kernel launches. With CUDA graphs it reaches 181 Hz.
7. **"Why Mamba and not a transformer?"**
   - Linear cost in sequence length and a state carried across time, which suits continuous event streams; attention cost grows with the square of the sequence.
8. **"If you bin events into 50-ms frames, isn't the event camera's speed advantage lost?"**
   - Yes, at this setting: a detection waits about 25 ms on average for its window, plus 13–27 ms of compute, so it is no faster than a frame camera. I kept 50 ms to stay comparable with the published baseline. What remains is dynamic range, no motion blur and sparse output. The window is a trade-off, and that is where SSMs help: with 5-ms windows and no retraining they keep 62–70 % of their accuracy, PureSSM the most, and EventSSM and PureSSM run at 210 and 181 Hz with CUDA graphs. A full window sweep with retraining is future work.

## Delivery and what to fill in

Make each slide title the finding, not the topic: "PureSSM gains 3 AP on large objects", not "Accuracy results".

- One message per slide; at most three bullets, at most one table.
- Say a number once and round it (46.4, not 46.43); the full numbers live in the paper.
- Rehearse aloud with a timer at least twice; the first run is usually 30–50 % over.
- Keep a backup slide each for the Δt evidence table, the efficiency table and the limitations, ready for questions.

To fill in once training and evaluation finish:

- [ ] Slide 10: spiking ladder test AP at 100k (PureSSM anchor, analog, graded, binary) with bootstrap error bars
- [ ] Slide 10: firing rate on the final checkpoints and the threshold trade-off curve, if made
- [ ] Slide 10: whether the pre-registered order analog ≥ graded > binary held
- [ ] Slide 9: copy the true-rate curve from the workstation; add the spiking models if the rate test is run on them
- [ ] Slide 12: one-line spiking takeaway
- [ ] Draw the four-model diagram (slide 5) and check the pipeline diagram (slide 4)
