# Stage 20 — SpikingSSM Full Runs: Plan and Decision Record

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · started 2026-10-07 · **Branch:** `stage19-smoke`
**Predecessor:** `docs/notes/Stage19_smoke_notes.md` (kill-switch PASS, §3.5) · **Schedule:** `docs/plans/Thesis_C_Project_Timeline.md`

## 1. Plan (user decision, 2026-10-07)

| Order | Run | Rung | Budget | Purpose | Local time |
|---|---|---|---|---|---|
| 1 | spike | `[2,3,4]` (D5) | 100k | equal-budget ladder | ≈ 14 h |
| 2 | graded | same | 100k | equal-budget ladder | ≈ 14 h |
| 3 | analog | same | 100k | equal-budget ladder | ≈ 14 h |
| 4 | PureSSM (anchor, D6) | — | 100k | first ladder rung, PureSSM → analog | ≈ 14 h |
| 5 | graded | `[2,3,4]` | 400k | pre-registered headline (as long as PureSSM) | ≈ 56 h |

Total ≈ 112 h sequential on the RTX 5070 Ti (2.3 it/s, ≈ 12 min per full validation; row 4 added 2026-10-09, D6). The ladder runs first: it gives
the cost-of-spiking decomposition about two days earlier and exposes any longer-schedule instability before the 56-h
run. The rung is `[2,3,4]` (D5, decided 2026-10-08 after the Stage-19 checks).

## 2. Decision record

### D1. A 100k checkpoint from a 400k run is not a 100k arm
- **Problem.** The timeline (§4, "readout ablation, done cheaply") planned graded at 400k and spike/analog at 100k,
  stating that "the 400k run passes through 100k, so the graded@100k arm comes free".
- **Why it is wrong.** The OneCycle schedule is stretched over the run's total length (`total_steps: ${..max_steps}`):
  at step 100k of a 400k run the learning rate is still near its peak, whereas a 100k run has annealed to its floor.
  The two checkpoints are trained under different schedules and are not comparable.
- **Decision (user, option "100k ladder + 400k graded").** All three arms at 100k on one schedule length, plus graded
  at 400k. Alternatives rejected: 400k graded + the 25k ladder only (cheapest, weakest decomposition evidence); all
  three at 400k (≈ 7 days local or ≈ 3 rented runs).

### D2. Launcher
`code/event_ssm/scripts/stage20_full_local.sh`, a sibling of the Stage-19 wrapper (which stays unchanged).
`ARM`, `STAGES` and `BUDGET ∈ {100k, 400k}` are required. The recipe is pinned (validation every 10k as in the PureSSM
400k run, batch 4, bf16, full Gen1, `spikingssm`), and labels are derived (`stage20_<arm>_s<stages>_<budget>`).
Arguments are allow-listed. Compute-only knobs pass through for a possible rented-GPU run (workers,
`CHECKPOINT_BLOCKS`, `REPO`, `CONDA_SH`, `WANDB_MODE`), and `STAGE7_RESUME` passes through for multi-day runs
(guarded, D4). A `--cfg job` dry run composed max_steps 400000, val 10000, batch 4, graded, `[2,3,4]`,
checkpoint_blocks true.

### D3. Do not run the full test suite while a GPU training run is active
- **Problem.** Running `pytest code/event_ssm/tests/` during the Stage-19 `[2,3,4]` spike run gave 7 CUDA
  out-of-memory failures in `tests/test_resnet_mamba.py`. Those tests use the Mamba GPU kernels but are not marked
  `gpu`, so the default (`-m "not gpu"`) run still puts them on the GPU.
- **Impact.** The training run survived (the log continued, no OOM), but test allocations could equally have starved
  the run. Not caused by the Stage-20 change: the same tests pass on an idle GPU.
- **Decision.** During GPU runs, run only the relevant CPU tests. Follow-up (deferred, a test-hygiene edit to an
  existing file): mark the CUDA tests in `test_resnet_mamba.py` as `gpu`.

### D4. Code review ("merge after one fix") and the resume guard
- **Important finding.** `STAGE7_RESUME` was not checked against the budget. Lightning restores the scheduler with
  `load_state_dict`, which for OneCycleLR replaces `total_steps` and the schedule phases, while `Trainer.max_steps`
  comes from the config. A 400k checkpoint resumed under `BUDGET=100k` would train at ≈ 0.75× peak LR and stop cleanly
  at 100k under a "100k" label, which is the D1 confound itself, and silent. The reverse direction crashes only once
  the scheduler runs past its end (up to ≈ 14 h wasted, with mislabelled checkpoints). Arm and rung were already
  guarded (Stage-18 D14 arm contract; strict state-dict load).
- **Fix.** `event_ssm/integration/resume_guard.py` (`check_resume_schedule`). The wrapper exports
  `RESUME_EXPECT_TOTAL_STEPS=$MAX_STEPS`; `scripts/stage6_train.py`'s local-file resume path loads the checkpoint on CPU
  and refuses a mismatched `total_steps`, or a checkpoint already at the end of its schedule, before training starts.
  When the variable is unset (Stage 7/13/14/19 launchers), behaviour is unchanged.
- **Minor fixes taken.** `SPIKING_ALLOW_ARM_OVERRIDE` unset in the wrapper; tests for empty `STAGES`, rungs `1` and
  `[2,3,4]`, the allow-list's positive cases, the OneCycle-follows-max_steps premise (pure config compose), and
  return-code assertions; the comment on workers corrected (they change data order). Not applied to the Stage-19
  wrapper, which was mid-use by the running checks.
- **Tests.** 8 guard tests (incl. three through the real patched resume path) + 28 launcher tests; with Stage 19's 22,
  58 pass. The full suite waits for an idle GPU (D3).

### D5. Rung `[2,3,4]` for every Stage-20 run
- **Evidence (Stage-19 notes §3.6, 25k).** On `[2,3,4]` the ladder separates: PureSSM 0.351 → analog 0.351 → graded
  0.340 → spike 0.305, with the pre-registered ordering holding at every evaluation from 5k onwards. On `[4]` the arms
  tie (spike 0.345, graded 0.343), so a `[4]` ladder would have no decomposition to report.
- **Secondary reason.** `[2,3,4]` carries the larger spike-fed operation share (SOP ceiling ≈ 1.04 % vs 0.21 % on `[4]`),
  which the Stage-22 energy accounting needs.
- **Decision (user, 2026-10-08).** `[2,3,4]` for all three 100k arms and the 400k graded run. The `[4]` and `[3,4]` rungs
  stay as 25k evidence only (de-risking ladder, Stage 19).
- **Pre-launch state.** Machine rebooted 2026-10-08 into kernel `7.0.0-38` (matching prebuilt NVIDIA module; the
  2026-10-08 GPU loss on `-34` is fixed). `nvidia-smi` and torch CUDA verified; the `--cfg job` dry run composed
  spike, `[2,3,4]`, max_steps 100000, val 10000, batch 4, bf16-mixed, checkpoint_blocks true.
- **Full test suite on the idle GPU (2026-10-08, closes the D3 wait).** Default run: 398 passed, 1 skipped, 29
  deselected (97.9 s); the 7 unmarked CUDA tests in `test_resnet_mamba.py` pass on an idle GPU. `-m gpu`: 29 passed
  (44.8 s). The one skip is `test_rvt_lstm.py::test_rvt_checkpoint_loads_strictly` (the RVT-B checkpoint
  `checkpoints/rvt-b-gen1.ckpt` is not on this machine); it concerns the ConvLSTM baseline only.
- **First launch aborted (2026-10-08, user needed the GPU).** The chain was launched 17:14 in tmux `stage20` and
  stopped with Ctrl-C at step ≈ 1.2k (2.26 it/s; W&B offline id `55m2unj4`), before the first checkpoint at 10k.
  Nothing is resumed: the ladder is relaunched from scratch overnight with the same command. Ctrl-C is safe for
  the `&&` chain because Lightning 2.6.5 exits 1 on KeyboardInterrupt and `script -e` + `pipefail` pass that on,
  so the next arm does not start. The aborted console log stays in `results/stage20/spike_s234_100k/`.
- **Relaunched 2026-10-08 22:41** (tmux `stage20`, chain spike → graded → analog; spike W&B offline id `5ykgl80y`).

### D6. A PureSSM 100k anchor for the first ladder rung
- **Problem (found 2026-10-09, while the spike arm was at ≈ 91k).** The ladder's first gap, PureSSM → analog (cost of
  the LIF dynamics: leak + reset), needs PureSSM on the same 100k schedule. The only PureSSM run is the Stage-14 400k
  one, which is not a 100k model (D1: OneCycle follows the run length). Without an anchor that gap would exist only at
  25k (Stage-19 §3.6: 0.351 vs 0.351), although the pre-registration commits to reporting all three gaps.
- **Decision (user, 2026-10-09).** Train PureSSM for 100k locally, after the analog arm and before the 400k graded run
  (≈ 14 h; all runs finish ≈ 14 h later). Not run on a rented GPU: same card ⇒ same kernels and timing conditions.
- **Launcher.** `code/event_ssm/scripts/stage20_puressm_local.sh`, a new sibling. `stage20_full_local.sh` is not edited:
  the running `&&` chain re-reads it when each later arm starts, and it knows spiking arms only. The new wrapper
  pins the same recipe (100k, val 10k, batch 4, bf16-mixed, full val/Gen1, default workers 2/1 ⇒ same data order),
  `+experiment/gen1=puressm`, labels `stage20_puressm_100k`, `RESUME_EXPECT_TOTAL_STEPS=100000`, the same
  allow-list and compute-only knobs, and unsets the Δt hooks and the spiking-only knobs. Using the real `puressm`
  experiment (not `spikingssm` with `spiking_stages=[]`, numerically equivalent by the Stage-18 null test) keeps the
  anchor the same model as the Stage-14 PureSSM.
- **Tests (TDD).** `tests/models/spikingssm/test_stage20_puressm_launcher.py`, 18 CPU tests: written first, 17 failed
  with the script missing (the one pass is a premise check on the existing PureSSM config). Includes a parity test
  that runs both wrappers against a recorder dumping the WHOLE exec'd environment and argument list, and requires
  both to match except `EXPERIMENT`/`GROUP_NAME`/`RUNDIR`/`SPIKING_MONITOR` and the `model.backbone.spiking.*`
  arguments (mutation-checked: an extra knob in one wrapper, or a drifted `VAL_EVERY`, fails it). With
  `test_stage20_launcher.py` and `test_resume_guard.py`: 54 pass (CPU only, GPU hidden; the full suite waits for an
  idle GPU, D3).
- **Dry run.** `--cfg job` composed PureSSM, max_steps 100000, val 10000, batch 4/4, workers 2/1, bf16-mixed,
  checkpoint_blocks true. Diff against the live spike arm's `.hydra/config.yaml`: only `wandb.group_name`, the
  backbone name and the `spiking:` block differ.
- **Code review: ready to commit, no Critical/Important.** Minor items taken: the parity test above was hardened (it
  compared a fixed list of 21 variables, so a knob outside the list would have slipped through, and it discarded
  the arguments); the header claim was matched to it; one test renamed (labels are pinned, not derived). Not taken:
  sharing the test scaffolding with `test_stage20_launcher.py` (would edit that file; low value); a GPU-busy guard
  (the sibling has the same exposure; the launch below starts only after the chain ends). Reviewer-verified: a
  spiking 100k checkpoint passes the schedule guard but fails Lightning's strict state-dict load (LIF parameters), so
  a wrong-model resume stops before training. The dry run left a config-only `console_20261009_112810.log` in
  `results/stage20/puressm_100k/`; harmless, the real run's log is newer.
- **Launch (after the ladder chain ends, same tmux session):**
  `S=/home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts; bash $S/stage20_puressm_local.sh && ARM=graded STAGES=2,3,4 BUDGET=400k bash $S/stage20_full_local.sh`

## 3. Results

### 3.1 Spike arm, `[2,3,4]`, 100k (W&B offline `5ykgl80y`, finished 2026-10-09 12:31, exit 0)
| step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| val/AP | 0.205 | 0.303 | 0.318 | 0.351 | 0.370 | 0.379 | 0.393 | 0.404 | 0.408 | **0.418** |

- Every validation was a new best; best = last = step 100k. Checkpoint:
  `external/ssms_event_cameras/RVT/RVT/5ykgl80y/checkpoints/epoch=002-step=100000-val_AP=0.42.ckpt` (test eval:
  Stage 21, after the GPU frees up).
- 2.3 it/s, ≈ 13.8 h wall-clock including ten full validations. Zero SILENT / SATURATED / NON-FINITE lines.
- **Firing rates roughly doubled over training** (training-time monitor, first → last): s2 0.25 → 0.48, s3 0.19 →
  0.36, s4 0.22 → ≈ 0.39. The 25k pilot's ≈ 0.22 is therefore not the trained rate; Stage-22 SOP accounting
  measures rates on the final checkpoint and must be the number quoted.
- **β moved at stage 3** (0.900 → 0.919 channel mean; s2 0.898, s4 ≈ 0.905). Small, but outside the 25k range
  (0.897–0.906): recheck per channel on the final checkpoints before writing "β effectively fixed at its init"
  (Stage-19 notes D6).
- Not comparable to the 400k runs (D1); it is compared only with the other 100k arms and the PureSSM 100k anchor (D6).

### 3.2 Graded arm, `[2,3,4]`, 100k (W&B offline `5d1y0skm` → `uqhexctz` after the resume, started 2026-10-09 12:31) — interrupted, resumed 2026-10-10 14:30
- val/AP 0.192 / 0.278 / 0.309 / 0.352 at 10k / 20k / 30k / 40k; zero SILENT / SATURATED / NON-FINITE lines.
- **Interrupted 2026-10-09 19:02:55 (Ctrl-C, user needed the GPU)** at step ≈ 48.1k (epoch 0 = 47,364 steps + 715).
  Exit code 1, so the `&&` chain did not start analog (as designed). Last checkpoint: `last_epoch=000-step=40000.ckpt`
  (18:04) in `external/ssms_event_cameras/RVT/RVT/5d1y0skm/checkpoints/` ⇒ ≈ 8.1k steps are redone.
- **Resume deviation (to report):** `STAGE7_RESUME` restores weights, optimizer, OneCycle state and global step (the
  resume guard checks the 100k schedule), but not the data loaders' stream positions, so from 40k on the graded arm
  sees a different data order than an uninterrupted run would. Affects the graded 100k arm only.
- **Resumed 2026-10-10 14:30:08 (+11:00)**, ≈ 19.5 h after the interruption, in a new tmux `stage20` session with the
  runbook chain (`docs/runbooks/Stage20_resume_after_interrupt.md`: graded resume → analog → PureSSM anchor → graded
  400k). Log `results/stage20/graded_s234_100k/console_20261010_143008.log`: the local-file resume loaded the 40k
  checkpoint, `resume guard: checkpoint schedule matches 100000 steps`, and training continued at step 40,001.
  W&B offline cannot resume a run, so the continuation logs under a **new offline id `uqhexctz`**. Its checkpoints
  from 50k onward go to `external/ssms_event_cameras/RVT/RVT/uqhexctz/checkpoints/`, not `5d1y0skm/`. The 10k–40k
  curve stays in `5d1y0skm`, and the two runs together make up the arm.

*Analog and PureSSM-anchor 100k arms: pending.*
