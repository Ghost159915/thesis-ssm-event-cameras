# Stage 19 — SpikingSSM Smoke and Short Run: Notes and Decision Record

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · 2026-10-06 · **Branch:** `stage19-smoke` (base `62b65b2`, on `stage18-spikingssm`)
**Predecessor:** `docs/notes/Stage18_integration_notes.md` (§8 checklist, §9 next steps) · **Schedule:** `docs/plans/Thesis_C_Project_Timeline.md` (Week 4, kill-switch Sun 11 Oct)

Stage 19 asks the first empirical question of the spiking model: does it train? It has two parts, an overfit smoke on one real
Gen1 batch (minutes) and a 25k-step short run on full Gen1 (hours) whose outcome decides the kill-switch. Every problem, decision
and measured number is recorded here (§4) so the thesis chapters can be drafted from this file.

---

## 1. What was built

| Component | Location | Role |
|---|---|---|
| Smoke verdict, recorder, figure | `code/event_ssm/integration/spiking_smoke.py` | `smoke_verdict` (gates), `median_step_ms`, `verify_arm` (built-model arm check, raises), `json_safe`, `output_stem` (reruns numbered, never overwritten), `SpikingSmokeRecorder` (Lightning callback: loss, step time, per-stage firing rate and beta), `plot_smoke` (3 panels: loss, firing rate with the band, beta with its cap). CPU-tested. |
| Smoke script | `code/event_ssm/proofs/smoke_overfit_spikingssm.py` | Clone of the Stage-12 PureSSM smoke for `+experiment/gen1=spikingssm`; `--mode`, `--stages` (ladder rungs only), `--epochs`, `--monitor-every`. Writes `results/smoke_test/spikingssm_<arm>_overfit[_e<N>][_run<n>].{png,json}`; exit 1 on FAIL. |
| 25k short-run wrapper | `code/event_ssm/scripts/stage19_short_local.sh` | Thin wrapper over `stage7_midrun_local.sh`. Budget and recipe knobs pinned to Stage 13; W&B group and run dir derived from the arm; extra args allow-listed. |
| Stage-4-isolated gradient test | `code/event_ssm/tests/models/spikingssm/test_backbone_gpu.py` (last test) | GPU: loss on stage 4 only, gradient must reach the stage-4 temporal Mamba through the binary spikes (D4). |
| Tests | `tests/models/spikingssm/test_spiking_smoke.py`, `test_stage19_launcher.py` | 38 + 22 CPU tests. Suite: **249 CPU pass** (was 189), 28 GPU deselected. |

## 2. Gates (fixed before any run)

**Smoke** (per arm, 150 steps, constant LR 1e-3, `checkpoint_blocks=True`, `drop_path_rate=0.0`, bf16-mixed, batch 2; the
Stage-12 smoke-only overrides):
1. every loss finite; 2. loss reduction (mean of first 3 / mean of last 3 steps) ≥ 3× (the Stage-5/12 gate, unchanged);
3. spike/graded only: final firing rate (mean of the last 3 steps) of every spiking stage in [0.01, 0.90] (the band of
`attach_spiking_monitor`; NaN fails). Analog: reported, not gated (it emits the pre-reset membrane, not the spike).

**25k short run / kill-switch** (written into the wrapper header before any run): finite loss throughout; for spike/graded no
SILENT/SATURATED `[spk-monitor]` line after warm-up; val/AP rising across the 5 checkpoints (5k…25k); final val/AP ≥ 0.15 (the
Stage-13 gate). The gap to PureSSM's 25k val/AP **0.351** (same compressed recipe) is reported, not gated.

## 3. Results

### 3.1 Overfit smoke, ladder rung `[4]` (git `0efe360`)

| arm | loss (first 3 → last 3) | reduction | gate | stage-4 final rate | β mean / max | peak VRAM | median step |
|---|---|---|---|---|---|---|---|
| analog | 20.47 → 5.12 | **4.0×** | PASS | 0.250 (reported) | 0.8999 / 0.9019 | 4.37 GB | 200.5 ms |
| graded | 24.68 → 6.50 | **3.8×** | PASS | 0.187 OK | 0.8997 / 0.9030 | 4.37 GB | 204.3 ms |
| spike | 21.03 → 7.92 | **2.65×** | **FAIL** (< 3×) | 0.238 OK | 0.9000 / 0.9033 | 4.37 GB | 202.1 ms |

Figures: `results/smoke_test/spikingssm_{analog,graded,spike}_s4_overfit.png`. Peak VRAM uses 2^30 bytes (the Stage-11 probe's
unit; PureSSM's checkpointed-training peak there was 8.55 GB at B = 4; this smoke is B = 2). For reference the Stage-12 PureSSM
smoke reached 6.0× (single run, unseeded).

### 3.2 Gradient into the stage-4 temporal Mamba (diagnostic, D4)

Same initialisation (seed 0), same input, `spiking_stages=(4,)`, loss = random-weighted mean of the stage-4 output only, small
spatial depths (1,1,1,1), 64×96 frame:

| arm | Σ\|grad\| into stage-4 Mamba | vs analog | mean \|stage-4 output\| |
|---|---|---|---|
| analog | 332.8 | 1.00× | 0.634 |
| graded | 168.5 | 0.51× | 0.107 |
| spike | 103.1 | **0.31×** | 0.081 |

Same ordering as the smoke reductions (4.0 > 3.8 > 2.65) and as the pre-registered ladder ordering `analog ≥ graded > spike`.

### 3.3 Spike-arm spread and budget (H1 / H2 of D4)

| run | epochs | loss (first 3 → last 3) | reduction | gate | stage-4 final rate | β max | git |
|---|---|---|---|---|---|---|---|
| run 1 (§3.1) | 150 | 21.03 → 7.92 | 2.65× | FAIL | 0.238 | 0.9033 | `0efe360` |
| run 2 | 150 | 17.90 → 7.97 | 2.25× | FAIL | 0.215 | 0.9026 | `d74d351` |
| run 3 | 150 | 22.44 → 9.50 | 2.36× | FAIL | 0.267 | 0.9035 | `d74d351` |
| e300 | 300 | 22.10 → 4.47 | **4.95×** | PASS | 0.241 | 0.9044 | `d74d351` |

- **H1 (noise) rejected.** All three 150-epoch runs miss the gate (2.25–2.65×, mean 2.42×). The FAIL is systematic, not a
  draw. (`d74d351` differs from `0efe360` only in review fixes to bookkeeping/tests; model code is identical.)
- **H2 (slow, not stuck) confirmed.** With twice the steps the spike arm reaches 4.95×, above analog's 150-step 4.0×; the
  curve declines steadily with no plateau at the end, firing stays in band (0.21–0.27) and β stays far from its cap.
- **Caveat.** analog (4.0×) and graded (3.8×) are single runs; their spread is unmeasured. If it is similar to spike's
  (±0.2×) both remain clear of 3×.
- Peak VRAM 4.37 GB and ~202 ms/step in every run (B = 2).

### 3.4 25k short runs

*Complete* (spike 21:53→01:51, run `ax1lj36q`; graded 01:51→05:40, run `js9sthxu`; both stopped on
`max_steps=25000 reached`). Launched 2026-10-06 21:53 in tmux `stage19`: `spike` then `graded`, rung `[4]`
(`ARM=spike STAGES=4 bash code/event_ssm/scripts/stage19_short_local.sh ; ARM=graded …`). Verified at launch: composed
config = SpikingSSM / spike / `[4]` / 25k / val 5k / batch 4 / bf16 / LR 2e-4 OneCycle / `checkpoint_blocks` true /
group `stage19_short_spike_s4`. Spike run id `ax1lj36q` (checkpoints `external/ssms_event_cameras/RVT/RVT/ax1lj36q/`,
console log `results/stage19/spike_s4/`). Throughput 2.27 it/s; one full validation ≈ 12 min; GPU 12.4/16 GB; host RAM
≈ 5 GB available (close heavy apps). Stage-4 firing rate 0.21–0.24 throughout, β ≈ 0.900, no SILENT/SATURATED, no NaN.

| arm | 5k | 10k | 15k | 20k | 25k | verdict |
|---|---|---|---|---|---|---|
| spike | 0.1338 | 0.2401 | 0.2791 | 0.3176 | **0.3448** | PASS |
| graded | 0.0918 | 0.2271 | 0.2925 | 0.3171 | **0.3430** | PASS |
| PureSSM (Stage 13, cloud, same compressed schedule) | 0.155 | — | 0.286 | — | 0.351 | anchor |

Values are val/AP from the ModelCheckpoint log lines (`'val/AP' reached …`). Monitors over the whole run (≈240 lines
per arm, every 200 backbone forwards incl. validation): **zero SILENT / SATURATED / non-finite lines** in either arm.
Stage-4 firing rate: spike 0.18–0.25, stable (0.230 → 0.219); graded 0.09–0.24, **falling** (0.221 → 0.120): the graded
readout learns to fire about half as often, plausibly because each spike carries its magnitude. β mean stayed at 0.900
(to three decimals) in both arms: the learnable leak barely moves in 25k steps at LR 2e-4 (a scalar logit behind a
sigmoid with derivative ≈ 0.09 at β = 0.9), so in practice β behaves as fixed at its initialisation over this budget.

### 3.5 Kill-switch verdict (criterion pre-registered in the wrapper header, 2026-10-06)

| criterion | spike | graded |
|---|---|---|
| finite loss throughout (completed, no non-finite monitor line, val/AP finite and rising) | ✅ | ✅ |
| no SILENT / SATURATED `[spk-monitor]` line after warm-up | ✅ | ✅ |
| val/AP rising across the 5 checkpoints | ✅ (strictly) | ✅ (strictly) |
| final val/AP ≥ 0.15 | ✅ 0.345 | ✅ 0.343 |

**Verdict: PASS — the spiking model trains stably on ladder rung `[4]`.** Both arms land within 0.01 of the
PureSSM anchor (0.351) on the same compressed schedule, inside single-run noise.

Interpretation, with its limits:
- The smoke's slower spike optimisation (D4) does **not** carry over to full-data training at this budget: spike
  leads graded at 5k and 10k and ties it at 25k.
- The pre-registered ordering `analog ≥ graded > spike` is **not** observed at 25k on `[4]` (spike − graded = +0.002,
  noise). Rung `[4]` spikes only the coarsest stage (8×10), so a small cost is expected; the ordering is to be judged on
  the full runs and higher rungs, not here.
- Caveats: one seed per arm; compressed 25k OneCycle; rung `[4]` only; analog not run at 25k; data order differs
  from the anchor (2/1 vs 6/2 workers) and the anchor ran on a different GPU without block checkpointing.

## 4. Decision record

### D1. Smoke design: clone, do not edit
- **Decision (user, bounded design approved 2026-10-06).** A new script cloning the Stage-12 smoke, with the gate logic in a
  CPU-tested module so a GPU run can fail only on the model, never on the bookkeeping. Nothing existing was edited except one
  appended GPU test (D4).

### D2. Monitor cadence in the smoke
- **Problem.** `SPIKING_MONITOR_EVERY` / `PURESSM_MONITOR_EVERY` default to 200 forwards; a 150-step smoke would never print a
  monitor line. **Decision (controller).** The smoke sets both to `--monitor-every` (default 10).

### D3. 25k comparability anchor and arms
- **Evidence.** The Stage-13 PureSSM short run used 25k steps, val every 5k, batch 4, OneCycle over 25k → val/AP 0.351.
- **Decision (user, option B).** Run `spike` then `graded`, both on rung `[4]`, back to back; `spike` is the hardest case (if it
  trains, the kill-switch passes for every arm), `graded` is the timeline's 400k headline arm. Budget pinned to Stage 13.
- **Known deviation.** Workers: Stage 13 ran 6/2 on the cloud; the local host (15 GB RAM) runs 2/1. Under RVT mixed sampling the
  batch mixture is set by batch size only (2 stream + 2 random at B = 4), but the worker count sets how stream sequences are
  partitioned across workers (`RVT/modules/data/genx.py:158-166`), so the **data order** differs from the anchor. Also
  `checkpoint_blocks=True` locally (activation recomputation, same maths) vs False on the 32 GB cloud card.

### D4. The spike smoke missed the 3× gate: bug or binarisation cost?
- **Symptom.** spike 2.65× (FAIL) while analog 4.0× and graded 3.8× PASS. Firing rate healthy (0.24, flat); β unmoved.
  The spike curve falls 21 → ~10 by step 40, plateaus, and falls again from step ~120 — still decreasing at step 150.
- **Confounder.** Only stage 4 spikes; the ANN stages 2–3 feed the PAFPN too, so a falling loss does not prove gradient flows
  *through* the spikes. The existing integrated GPU test summed the loss over stages 2+3+4 and checked the stem, which the ANN
  stages also reach.
- **Hypotheses.** H3 bug (stage-4 spike path dead in the integrated backbone); H1 run-to-run noise (unseeded, one run each);
  H2 expected cost of binarisation (spike output carries gradient only through the arctan surrogate; graded `spk·mem` also has a
  direct path through the membrane; analog is the membrane). H2 is litreview §8 risk 1: the backbone *replaces* stage features
  with the temporal output, so the PAFPN sees binary stage-4 maps.
- **Test of H3.** New GPU test `test_spike_gradient_reaches_stage4_mamba_through_the_spikes_alone`: loss on stage 4 only, output
  verified binary, gradient must be non-zero and finite on every stage-4 Mamba parameter and on spatial stage 4, and `None` on the
  stage-2/3 temporal blocks (proves the isolation). **Passes.** Mutation check: replacing the surrogate with a gradient-blocking
  step makes it fail (backward has no graph). **H3 rejected.**
- **Quantifying H2.** §3.2: spike delivers 0.31× analog's gradient into the stage-4 Mamba at initialisation, graded 0.51×; the
  ordering matches the smoke. **H2 supported**: the spike arm learns, more slowly, through a 3× weaker gradient.
- **Decision (controller, for the user to confirm).** Report the pre-registered smoke result as a FAIL, unedited, with this
  diagnosis. The gate is *not* moved (no re-running at more epochs until it passes). The smoke gate exists to catch no-learning
  regressions; D4 shows the spike path learns. The kill-switch is decided on the 25k run (§2), which is the experiment that
  measures "trains stably". H1 is measured with numbered reruns (§6) as supporting evidence only.
- **Outcome of the reruns (§3.3).** H1 rejected (2.25–2.65× over three runs: systematic), H2 confirmed (4.95× at 300 steps).
  Reported result: *the spike arm fails the 150-step overfit gate reproducibly and passes it at twice the budget; the cause is
  a ~3× weaker gradient through the binary readout, not a defect.*
- **Thesis use.** Ch.5 §5.7 / Ch.6: first measured instance of the binarisation cost, before any full run; the gradient table is
  a mechanism-level explanation of the ladder gap.

### D5. Final review (code review, "with fixes") and the fixes taken
- **Important — passthrough args could relabel a run.** Hydra lets the last value of a key win, so
  `ARM=spike bash stage19_short_local.sh model.backbone.spiking.residual=True` (the escape hatch, which must be reported) would
  have run another arm under `stage19_short_spike_s4`. **Fix:** extra args allow-listed (`--cfg job`, `hydra.verbose=…`), exit 2
  otherwise; 10-case test.
- **Minor, all taken:** pin `PRECISION`/`VAL_FRAC`/`MAX_EPOCHS`/`DATASET` (stale-export guard); tests tightened against
  window/median/count mutations with hand-derived values; arm check moved from `assert` (stripped by `python -O`) to
  `verify_arm` (raises, CPU-tested); verdict fails if recorded stages ≠ requested; summary is strict JSON (NaN → null) and records
  the full LIF config; `--stages` limited to ladder rungs; header wording (kill-switch firing criterion scoped to spike/graded;
  monitor counts forwards incl. validation; worker/data-order deviation, D3); step-time docstring completed; minor log-tick labels
  styled.
- **Declined:** a final-β finiteness gate (a NaN created by exactly the last smoke step is negligible; the 25k monitor prints β).
- **Process note.** `python -m pytest` run from inside `code/event_ssm` puts that directory on `sys.path`, so `event_ssm/models`
  shadows RVT's `models` (31 spurious `ModuleNotFoundError: models.detection`). The documented invocation —
  `pytest code/event_ssm/tests/` from the repo root — is correct.

## 5. Verification

- CPU suite from the repo root: **249 passed**, 28 GPU deselected; new tests also pass with `-W error`.
- Smoke script dry-run with CUDA hidden (analog, spike): config compose → model build → built-model arm check pass; stops at the
  Trainer's GPU request as expected.
- Wrapper `--cfg job` through the real launcher, both the tee path and the interactive `script -c` path (`[3,4]` list override
  through `printf %q`): composed config shows the arm, `checkpoint_blocks: true`, `max_steps: 25000`, `val_check_interval: 5000`,
  batch 4, `group_name: stage19_short_<arm>_s<stages>`, recipe LR 2e-4 with OneCycle, mixed sampling, `drop_path_rate: 0.1`,
  full Gen1 path, bf16-mixed. Dry-run logs deleted afterwards so the real run directories start clean.

## 6. Next

1. ~~Spike-smoke spread (H1)~~ done (§3.3): FAIL systematic at 150 steps, PASS at 300 → slow, not stuck.
2. ~~25k short runs~~ done (§3.4): **kill-switch PASS** (§3.5).
3. **Decided (user, 2026-10-07): all checks first.** 25k on the full rung `[2,3,4]` for all three arms, in the order
   spike → graded → analog (≈ 4 h each, run back to back in tmux), judged by the same pre-registered criterion as §3.5
   (firing band not gated for analog). Only then the Stage-20 full runs. Rationale: `[4]` spikes only the 8×10 map, a weak
   spiking claim; `[2,3,4]` is the configuration default and the headline candidate, and analog had not yet trained on
   full data.
4. Superseded by 3 — original wording: the Stage-20 plan — which rung (`[4]` passed at ~zero cost; `[2,3,4]` is the config default and
   the more meaningful spiking claim but untested) and which arms at which budget (timeline: graded at 400k, spike and
   analog at 100k), local (~2.3 days per 400k incl. validation) or rented 5090 (~27 h).
