# Stage 20 — SpikingSSM Full Runs: Plan and Decision Record

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · started 2026-10-07 · **Branch:** `stage19-smoke`
**Predecessor:** `docs/notes/Stage19_smoke_notes.md` (kill-switch PASS, §3.5) · **Schedule:** `docs/plans/Thesis_C_Project_Timeline.md`

## 1. Plan (user decision, 2026-10-07)

| Order | Run | Rung | Budget | Purpose | Local time |
|---|---|---|---|---|---|
| 1 | spike | from the Stage-19 `[2,3,4]` checks | 100k | equal-budget ladder | ≈ 14 h |
| 2 | graded | same | 100k | equal-budget ladder | ≈ 14 h |
| 3 | analog | same | 100k | equal-budget ladder | ≈ 14 h |
| 4 | graded | same | 400k | pre-registered headline (as long as PureSSM) | ≈ 56 h |

Total ≈ 98 h sequential on the RTX 5070 Ti (2.3 it/s, ≈ 12 min per full validation). The ladder runs first: it gives
the cost-of-spiking decomposition about two days earlier and exposes any longer-schedule instability before the 56-h
run. The rung is set after the three Stage-19 `[2,3,4]` checks finish (2026-10-07 ≈ 22:00).

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

## 3. Results

*Pending.*
