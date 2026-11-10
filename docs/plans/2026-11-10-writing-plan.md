# Writing plan: full thesis by Sunday 15 November 2026

The report is `thesis/latex/Thesis_Report_final.tex`. **Reference copy: `thesis/latex/Thesis_Report_full.tex`** (created 10 Nov): every paragraph still to write has, in green, the points it must contain and a word target, followed by an example paragraph; write your own version in the final file and compare. Its per-paragraph word targets supersede the per-section estimates below (they total about 9,000 words). Every section to write has a `% TODO-<section name>` comment
under its heading (Ctrl+F `TODO-`); delete the comment when the section is done. Sources for each section: the
prose and every final number are in `thesis/latex/draft_main.tex`; the bullet guides, tables and equations are in
`thesis/latex/main.tex`. Write in your own words, shorter than the draft.

## The budget

The body (Chapter 1 to the end of Chapter 6) may be at most 50 pages. With Chapters 1-2 written and every figure in
place it is already 32 pages, and Chapter 2 is 15 of them. Plan: Chapter 2 is cut to about 10 pages on Sunday,
which leaves about 23 pages for Chapters 3-6 and the new parts of Chapter 1, of which about 4 are tables. That is about **9,400 words of new text**
(about 500 words per page).

| Chapter | Pages | New words |
|---|---|---|
| 1 Introduction (written; + research question, aims, hypotheses) | 4.5 | ~500 |
| 2 Literature Review (written; cut 15 -> 10 pages) | 10 | 0 |
| 3 Methodology | 7 | ~2,800 |
| 4 Results (figures already in) | 18 | ~3,500 |
| 5 Discussion | 5 | ~2,250 |
| 6 Conclusion | 1 | ~400 |
| **Total** | **~46 + 4 spare** | **~9,500** |

## Day by day

About 1,900 new words per full day. Each day ends with "check <sections>" (Claude reads the day's text and returns a
numbered list of fixes) and a commit.

### Tuesday 10 (afternoon/evening) — research question, aims, hypotheses + start of Chapter 3 (~1,100 words)

| Section | Words | Source |
|---|---|---|
| 1.1 Research Question and Aims: add the numbered aims after the question | ~200 | `main.tex` Ch. 3 bullets (now Chapter 1 of the report) (aims 1-6) |
| 1.2 Hypotheses (H1-H3, H3 pre-registered) | ~150 | `main.tex` Ch. 3 bullets (now Chapter 1 of the report) |
| 3.1 Event Representation | ~250 | outline in the chat of 10 Nov; input is **1.5 % non-zero** |
| 3.2 The Frozen Detection Pipeline (+ 3.2.1 Neck, Head and Loss, 3.2.2 The Two Mixing Axes) | ~550 | draft "The Frozen Pipeline", "The Two Mixing Axes" |

Overnight: the remaining runs finish by themselves (nothing to do).

### Wednesday 11 — rest of Chapter 3 (~2,100 words) -> Methodology done

Morning: say "update"; Claude puts the overnight results into `draft_main.tex` (400k intervals, spiking true-rate,
ConvLSTM) so every number you need later is final.

| Section | Words | Source |
|---|---|---|
| 3.3 Model Architectures (S5-RVT ~100, EventSSM ~250, PureSSM ~250, SpikingSSM ~200, Readout Modes ~100) + table `tab:four_models` | ~900 | draft "The Four Models" and subsections; table in `main.tex` |
| 3.4 Dataset and Metrics | ~350 | draft "Dataset", "Metrics" |
| 3.5 Training Protocol | ~300 | draft "Training Protocol" (batch 4 vs 8, bf16, 400k OneCycle) |
| 3.6 Evaluation Protocol (Accuracy, Temporal Robustness, Efficiency, Spiking Activity and Energy, Statistical Treatment) | ~400 | draft "Evaluation Protocol", "Metrics", the bootstrap paragraph of the spiking results |
| 3.7 Implementation and Hardware | ~150 | draft "Hardware and Software" |

### Thursday 12 — Results 4.1-4.6 (~1,600 words + 5 tables)

| Section | Words | Source |
|---|---|---|
| 4.1 Baseline Reproduction | ~80 | draft "Baseline Reproduction" (47.72 exact) |
| 4.2 Accuracy + table `tab:accuracy` | ~300 | draft "Accuracy" |
| 4.3 Effective Receptive Field | ~250 | draft "Mechanism: Effective Receptive Field" |
| 4.4 Temporal Robustness (Fixed Cadence ~150, True Event-Rate Change ~150, Δt-Rescaling ~200) + tables `tab:regime1`, `tab:regime2`, `tab:dt_comp` | ~500 | draft "Temporal Robustness"; soft wording only: "not reproducible under the published artefacts" |
| 4.5 Efficiency + table `tab:efficiency` | ~300 | draft "Efficiency" |
| 4.6 Qualitative Results | ~150 | draft "Qualitative Results" |

### Friday 13 — Results 4.7-4.8 (~1,900 words + 3 tables) -> Results done

| Section | Words | Source |
|---|---|---|
| 4.7 Spiking SSM Detector (intro) | ~100 | draft "Spiking SSM Detector" |
| 4.7.1 Accuracy of the Readouts + table `tab:spiking_ablation` | ~150 | draft |
| 4.7.2 The Readout Ladder | ~300 | draft (intervals are final) |
| 4.7.3 The Graded Readout on the Full Schedule + table `tab:spiking_400k` | ~250 | draft (400k interval arrives Wednesday) |
| 4.7.4 Firing Activity and the Threshold Trade-off | ~350 | draft (threshold paragraph is final) |
| 4.7.5 Rate Robustness of the Spiking Readouts | ~150 | draft, filled Wednesday from the overnight run |
| 4.7.6 Synaptic Operations and Estimated Energy + tables `tab:sop`, `tab:spiking_bench` | ~300 | draft |
| 4.7.7 Training Dynamics | ~150 | draft |
| 4.8 Summary of Results | ~150 | new: one paragraph pulling the chapter together |

### Saturday 14 — Chapter 5 Discussion (~2,250 words) -> Discussion done

| Section | Words | Source |
|---|---|---|
| 5.1 Answering the Research Question (each aim met / partly / not; each hypothesis held or not) | ~300 | `main.tex` bullets; aims table `tab:aims` |
| 5.2 What the State-Space Model Contributes | ~300 | draft "What the Four Models Show" |
| 5.3 Temporal Robustness and the Δt-Rescaling Claim | ~250 | draft "What the Four Models Show" (robustness part) |
| 5.4 Efficiency Trade-offs | ~200 | draft (efficiency part) |
| 5.5 Latency and the Choice of Window | ~150 | draft "Limitations" (window) |
| 5.6 The Neuromorphic Case | ~250 | draft "The Neuromorphic Case" |
| 5.7 Scope Change: the Obstacle-Avoidance Aim | ~250 | draft "Scope Change" |
| 5.8 Limitations | ~300 | draft "Limitations" |
| 5.9 Future Research Opportunities | ~250 | draft "Future Work" (pick the best four) |

### Sunday 15 — finish and polish (~650 new words + editing) -> thesis complete

| Task | Time |
|---|---|
| 6 Conclusion (answer the question, headline numbers, main limitation, next step) | ~400 words, 1 h |
| Chapter 1: rewrite Thesis Organisation for the new chapter order (fixes the "Chapter ??"), check Contributions against the final numbers | ~150 words, 45 min |
| Abstract: add the Δt sentence, the three spiking costs and a closing sentence; re-check "the most of the models tested" | ~80 words, 20 min |
| Cut Chapter 2 from 15 to about 10 pages (Claude marks candidate cuts on request; you decide) | 2 h |
| Appendices: copy the equations (A), the reproducibility text (B), the timeline text (C) and the risk table (D) from `main.tex` / draft | 1 h |
| Acknowledgements | 10 min |
| Final checks with Claude: builds with no errors, no `??`, no `\res{`, no `TODO-` left, body <= 50 pages, every number against the results files, spelling | 1-2 h |

## After Sunday (buffer to the submission, Fri 20 Nov, date unconfirmed)

Send the PDF to Will for feedback; prepare the ~10-minute presentation; apply feedback.
