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

### 3.3 Run-to-run spread and the 25k runs

*Pending (user-run; §6).*

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

1. Spike-smoke spread (H1): two more 150-epoch reruns (auto-numbered `_run2`, `_run3`) and one 300-epoch run (`_e300`, slow vs
   stuck). Supporting evidence only; does not change the D4 verdict.
2. 25k short runs, `spike` then `graded`, rung `[4]`, in tmux. Record val/AP at 5k…25k, `[spk-monitor]` lines, peak VRAM and
   it/s here; then the kill-switch call against §2 (due Sun 11 Oct).
