# Stages 21–22 — SpikingSSM Evaluation and Benchmark Tooling: Decision Record

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · 2026-10-07 · **Branch:** `stage19-smoke`
Built while the Stage-19 `[2,3,4]` checks trained, so that Stages 21–22 can run as soon as the Stage-20 checkpoints
exist. **Commits:** `8e78ac7`, `2fff812`, `e9e0ced` and the benchmark-fix commit after it.

## 1. What was built

| Component | Location | Role |
|---|---|---|
| Arm reader | `code/event_ssm/integration/spiking_ckpt.py` | Reads a SpikingSSM checkpoint's ablation arm from its `_extra_state` (`temporal.<s>` → residual; `temporal.<s>.lif` → output mode, reset, alpha, detach_reset, learn flags, and a non-learned beta/threshold). Refuses non-spiking, pre-D14, malformed or per-stage-inconsistent checkpoints. Emits Hydra overrides, the run tag, the full `stage7_eval.py` argv, and a verdict on extra args. CLI: `tag \| overrides \| argv \| check-extra`. |
| Test-set eval | `code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh` | Sibling of the PureSSM test eval (unchanged), using the same recipe. The arm comes from the checkpoint, so a checkpoint cannot be evaluated as another arm. Log: `results/stage21_test_eval/<tag>[_posthoc]/test_eval_<run>_step<N>_<ts>.txt`. |
| Benchmark entry | `benchmark/bench_models.py`, `scripts/stage10_benchmark.py`, `scripts/stage10_report.py`, `scripts/stage10_run_local.sh` | `--models spikingssm \| all+spikingssm --spikingssm-ckpt PATH [--spikingssm-test-ap AP]`. The arm comes from the checkpoint and is recorded per model. Output defaults to `results/stage22/<tag>[_with_ann]/`. |

Usage, once a Stage-20 checkpoint exists and the GPU is idle (the user runs these):
```bash
bash code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh /abs/.../epoch=...-val_AP=....ckpt
bash code/event_ssm/scripts/stage10_run_local.sh --models all+spikingssm --spikingssm-ckpt /abs/...ckpt --spikingssm-test-ap 0.4xx
```

## 2. Decisions

- **D1. The arm is read from the checkpoint, never typed.** The checkpoint arm contract (Stage-18 D14) would reject
  a mismatch at load anyway. Deriving the overrides from the checkpoint removes the possibility of the mistake rather
  than only detecting it. The overrides were verified on the three real Stage-19 checkpoints
  (`graded_s4`, `spike_s234`, `spike_s4`); the PureSSM checkpoint is refused.
- **D2. Fixed beta/threshold compare within tolerance.** A fixed beta is stored through its logit, so 0.89999998
  comes back as 0.90000004. The arm contract's 1e-5 relative tolerance (`lif.py` `_FIXED_REL_TOL`) absorbs this.
  The round-trip test (overrides → compose → LIF arm record) checks those two values within that tolerance and
  everything else exactly.
- **D3. Code review ("with fixes") — items taken.**
  - **Important 1:** a spikingssm benchmark would have overwritten the citable `results/stage10/bench_results.json`
    (gitignored, unrecoverable). Spiking runs now default to `results/stage22/`, and any results JSON holding a
    different model set is never overwritten.
  - **Important 2:** the eval launcher accepted any extra argument. In particular, an override of a *learned*
    threshold or beta is silently ignored by the model, so a "sweep" would have measured one operating point under
    the canonical label. Extra args are now allow-listed; learned knobs and recipe keys are refused; post-hoc sweeps
    of non-learned knobs need `SPIKING_ALLOW_ARM_OVERRIDE=1` and are filed under `<tag>_posthoc/`.
  - **Minor items taken:**
    - the arm and its tag are recorded per model and shown in report labels (a residual arm must be visible);
    - the test-AP units are checked;
    - the checkpoint path is stored absolute;
    - stray spiking flags are errors;
    - a flops note says SOPs are not accounted;
    - the benchmark refuses to run with training monitors on, and the run script unsets them;
    - one-line errors replace tracebacks, with a clear message for pre-D14 checkpoints;
    - the checkpoint step appears in the log name;
    - tests were added for fixed beta, residual disagreement, the round trip, the allow-list, and the Stage-9 hook.
  - **Not taken:** the pre-existing `load_weights` dropping of unexpected keys. For spiking checkpoints every arm
    difference still surfaces as a missing key or an extra-state mismatch.
- **D4. Graph replay is not measured for SpikingSSM.** Capturing the LIF time loop as a CUDA graph has not been
  validated (Stage-18 deferred item). The JSON records an honest skip with this reason.
- **D5. SOP / energy accounting is separate (Stage 22, 2c).** A first-principles bound computed from the neck's
  layer shapes: the spikes of stages 2–4 feed ≈ 52 M multiply–adds (≈ 0.10 GFLOPs) of PureSSM's 10.12 GFLOPs per
  frame, so at most ≈ 1 % of the network's arithmetic can become spike-driven accumulates. The scope of the
  accounting tool is a user decision (options in the chat of 2026-10-07; spec before code).

## 3. Tests

187 CPU tests pass across the files touched on 2026-10-07: `test_spiking_ckpt.py`, `test_stage21_eval_script.py`,
`test_bench_{models,schema,report,metrics,clip}.py`, `test_resume_guard.py`, and the Stage-19/20 launcher tests.
GPU-dependent paths (real evaluation, benchmark timing) run only when the GPU is idle.
