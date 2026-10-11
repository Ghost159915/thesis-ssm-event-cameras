# Post-training plan: strengthening the spiking results + condensing to a conference paper

**Agreed 2026-10-11.** Start AFTER all Stage-20 training and the Stage-21 test evals are done (user target: within
one week of 2026-10-11). None of the items below needs a new training run — all use the existing checkpoints.

## Part A — Spiking analysis (priority order)

| # | Item | What | Compute | Why it matters |
|---|---|---|---|---|
| 1 | **Error bars without more seeds** | Bootstrap CIs over test sequences on each arm's test AP **and on each ladder gap** (PureSSM→analog, analog→graded, graded→spike); 90k vs 100k ckpt spread as a nearby-checkpoint variance proxy | CPU (needs saved per-sequence predictions) | Single-seed is the main attack surface; a gap whose CI includes 0 is reported as such |
| 2 | **Post-hoc threshold sweep** | Re-eval trained spike + graded ckpts at several thresholds (`stage21_spikingssm_test_eval_local.sh`, outputs go to `<tag>_posthoc/`); plot firing rate / SOPs vs mAP | GPU evals, idle GPU only | Energy–accuracy Pareto curve; answers the high firing rate (0.36–0.48 in training) |
| 3 | **Rate robustness of spiking arms** | Stage-9/16 true-rate protocol (true 2× and 10×) on spike + graded; compare retention with PureSSM 69.7 % | GPU evals | Ties the spiking chapter to the thesis's temporal-generalisation thread; not reported by any SNN detector we surveyed |
| 4 | **SNN-convention energy** | `benchmark/sop.py` on FINAL ckpts (trained rates ≈ 2× the 25k rates); convert with Horowitz 2014 per-op energies (≈ 0.9 pJ AC vs 4.6 pJ MAC, 45 nm); report temporal-path and whole-detector separately; state the ≈1 % spike-fed ceiling up front | CPU + one benchmark run | Comparable to SpikeYOLO/HsVT reporting; pre-empts the "1 %" objection |
| 5 | **Firing-pattern figure** | Per-stage firing rate on a real Gen1 sequence vs input event density (`SPIKING_MONITOR` hooks) | GPU, minutes | Visual evidence of event-driven behaviour |
| 6 | **β (membrane leak) finding** | Per-channel β on final ckpts (spike arm moved s3 0.900→0.919 at 100k — recheck before writing "fixed at init") | CPU | Report as a finding (Mamba already integrates upstream), not a footnote |

Minimum set if time is short: **1, 2, 3.**

**Item 1 status (2026-10-11):** Part B (resampling + intervals, `code/event_ssm/eval/bootstrap.py`, 16 CPU tests) done;
Part A (recording the evaluator inputs) + identity check to do on GhostMachine — `docs/runbooks/Stage21_bootstrap_ghostmachine.md`.

Writing rules for §5.7 / Ch.6: phrase the ladder so it survives gaps within noise ("small and ordered as
pre-registered", with CIs); Ch.6 must address the energy ceiling (≈1 % spike-fed ops, high firing rates) head-on;
the graded 100k arm's data-order deviation (two resumes) goes in Limitations.

## Part B — Deliverables and workflow (user decision, 2026-10-11)

**What is submitted:** ONLY a ~20-page conference-style paper, plus a ~10-minute presentation (problem → what was
done → results → discussion). The 60-page `thesis/latex/main.tex` is NOT submitted.

**Workflow:**
1. Finish `main.tex` completely as the full thesis (the long-form source of truth): spiking results (§5.7),
   abstract, contribution 6, limitations, Ch.6 energy discussion.
2. The user then condenses it into a **separate** conference-format version **themselves** — Claude does not
   build that version unless asked.
3. Presentation (~10 min) prepared from the paper.

**Implications:**
- `main.tex` is source material, not the graded artefact: finish content and correctness; don't spend time
  polishing or restructuring sections (e.g. the Thesis-A UAV-era lit review) that will be cut in the paper anyway.
- Leave enough calendar time after `main.tex` for the condensation and the slides.
- Guidance for the condensation (for the user): cut the lit review hard (drop contrast maximisation, UAV
  navigation, micro-UAV compute, broad dataset survey; keep event cameras briefly, SSMs for events, SNN landscape,
  gaps); UAV descope as one justified paragraph; one spine — "what does an SSM actually contribute to event-based
  detection" — with the controlled ablation, ERF mechanism, Δt reproduction and spiking decomposition under it.
- Open: the conference template and exact page limit.

## Part C — Window length and latency (decided 2026-10-11)

**Decision.** All models keep the 50-ms window (the RVT / S5-RVT protocol, for comparability with the baseline).
This bounds latency by the window, not the sensor: on average 25 ms (up to 50 ms) of waiting plus 13–27 ms of
inference, so the sensor's microsecond latency is not exploited at this setting. What remains: dynamic range, no
motion blur within the 5-ms bins, sparse sensor output. The window is a latency–accuracy trade-off: SSMs tolerate
shorter windows (true 10× = 5-ms windows at 200 Hz, zero-shot: 62–70 % retention; PureSSM best).

**Written into `thesis/latex/main.tex` (2026-10-11):** one sentence in §3.1 (input bullet), a "Window-bounded
latency" bullet in Limitations, and Future Work §"The latency–accuracy trade-off of the window"
(`subsec:fw_window`). Carry the same point into the conference paper and expect it as an examiner question.

**Does a smaller window need retraining?** No to *use* one, yes to get the *best* accuracy from it:
- zero-shot (no retraining): already measured — 2× (25 ms) keeps 95–96 %, 10× (5 ms) keeps 62–70 %;
- fine-tune from the 50-ms checkpoint: the cheap middle option (a fraction of a full run per model);
- retrain at the target window, or on a mixture of windows (one model for every rate): full cost per model.
Retraining also needs (a) the training/validation sets re-preprocessed at the new window (the Stage-9 tooling only
rebuilt the TEST set), and (b) longer sequences: 21 frames × 5 ms = 105 ms of context, so keeping the recipe's
1.05 s needs ~210-frame sequences (more memory). Out of scope for Thesis C; Future Work.
