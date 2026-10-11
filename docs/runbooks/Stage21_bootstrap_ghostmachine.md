# Stage 21 — Bootstrap CIs: what to do on GhostMachine

**Written 2026-10-11.** Item 1 of `docs/plans/2026-10-11-post-training-spiking-analysis.md`.
**Part B (resampling + intervals) is DONE** and pushed: `code/event_ssm/eval/bootstrap.py`, 16 CPU tests in
`code/event_ssm/tests/eval/` (pass on the Mac). **Part A (recording the evaluator's inputs) is left** — it needs
the RVT code in `external/`, which exists only on this machine.

Steps 1–3 are CPU / code work and are safe while training runs. Steps 4–5 need an **idle GPU** (test evals).

---

## 1. Pull and run the Part B tests (CPU, safe during training)

```bash
cd ~/Desktop/thesis-ssm-event-cameras && git pull
conda activate events_signals
cd code && pytest event_ssm/tests/eval -q          # expect: 16 passed
```

## 2. Find the hook points in RVT (read only, no edits to `external/`)

Confirm, in `external/ssms_event_cameras/RVT/`:

- **The evaluator:** `utils/evaluation/prophesee/evaluator.py` → `PropheseeEvaluator`. Note its
  constructor arguments (expected: `dataset`, `downsample_by_2`), `add_labels(list)`, `add_predictions(list)`
  and `evaluate_buffer(img_height, img_width)`. `RvtScorer` in `bootstrap.py` assumes exactly this; if the names
  differ, adjust `RvtScorer` (and say so in the commit).
- **What one list element is:** check whether each element passed to `add_labels`/`add_predictions` is one
  sample at one time step (a structured array with a `t` field), and that the scoring treats elements
  independently. The identity check (step 4) proves it either way.
- **Where a recording starts:** the detection module's test step (`modules/detection.py`,
  `_val_test_step_impl` or similar) gets a batch with a per-slot "first sample of a new sequence" flag
  (used to reset the recurrent state; `DataType.IS_FIRST_SAMPLE` or similar). Counting those flags per batch
  slot gives a recording id for every element. Check how labels/predictions map to batch slots in that step.

## 3. Write Part A (new files + two one-line hooks)

- **New module `code/event_ssm/eval/bootstrap_capture.py`**, patching at run time like
  `scripts/perclass_patch.py` (never edit `external/`):
  - wrap the test step to track the current recording id per batch slot (increment on the first-sample flag);
  - wrap `PropheseeEvaluator.__init__` / `add_labels` / `add_predictions` / `evaluate_buffer` to record the
    constructor kwargs, every element with its recording id, and the `evaluate_buffer` kwargs;
  - at `evaluate_buffer` for the **test** mode, write the file with
    `bootstrap.save_capture(path, seq_ids, labels, predictions, meta)`, where
    `meta = {"evaluator_kwargs": …, "evaluate_kwargs": …, "ckpt": …, "argv": sys.argv}`.
  - Switch: `BOOTSTRAP_CAPTURE=/abs/path/capture.pkl` (unset = do nothing, byte-identical eval).
  - Must not change any metric: the 6 aggregate numbers and the per-class print stay identical.
- **Hook line** (user-approved, opt-in) in BOTH eval entry points, right after `perclass_patch.apply()`:
  - `code/event_ssm/scripts/stage7_eval.py` (EventSSM, PureSSM, SpikingSSM — all our launchers call it)
  - `code/event_ssm/scripts/stage8_baseline_eval.py` (S5-RVT baseline — needed only for baseline CIs)

  ```python
  from event_ssm.eval import bootstrap_capture; bootstrap_capture.apply()   # no-op unless BOOTSTRAP_CAPTURE is set
  ```
- **Tests (CPU):** `code/event_ssm/tests/eval/test_bootstrap_capture.py` — with the env var unset, `apply()`
  patches nothing; with a fake evaluator class, recorded elements and recording ids come out as fed.
- Run `pytest event_ssm/tests/eval -q` again, then commit.

## 4. The identity check — the test that matters (idle GPU, ~14 min)

Re-run the PureSSM 400k test eval with recording on, then re-score the capture with no resampling:

```bash
cd ~/Desktop/thesis-ssm-event-cameras
BOOTSTRAP_CAPTURE=$PWD/results/stage21_bootstrap/captures/puressm_400k.pkl \
  bash code/event_ssm/scripts/stage14_puressm_test_eval_local.sh
cd code && python -m event_ssm.eval.bootstrap \
  --capture puressm=../results/stage21_bootstrap/captures/puressm_400k.pkl \
  --identity --expect puressm=0.4643
```

- **Pass:** exit code 0, `MATCH`, and the eval log still shows test/AP 0.4643 (the patch changed nothing).
- **Fail:** stop. The capture or `RvtScorer` is wrong; no interval is usable until this matches.
- Then time a full scoring pass (`time python -m event_ssm.eval.bootstrap … --n-boot 5 --workers 1`) to pick
  `--n-boot` and `--workers` (target: B = 1000; fall back to 200–500 if one pass is slow).

## 5. After Stage 20/21: capture every arm, then run the bootstrap

Capture each 100k arm during its normal Stage-21 test eval (one eval each, idle GPU):

```bash
CAP=$PWD/results/stage21_bootstrap/captures
BOOTSTRAP_CAPTURE=$CAP/spike_100k.pkl  bash code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh <spike 100k ckpt>
BOOTSTRAP_CAPTURE=$CAP/graded_100k.pkl bash code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh <graded 100k ckpt>
BOOTSTRAP_CAPTURE=$CAP/analog_100k.pkl bash code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh <analog 100k ckpt>
BOOTSTRAP_CAPTURE=$CAP/puressm_100k.pkl bash code/event_ssm/scripts/stage14_puressm_test_eval_local.sh <PureSSM 100k anchor ckpt>
```

Check each `BOOTSTRAP_CAPTURE` value reaches the python process (the launchers `exec`/call python directly;
if one strips the environment, export it first). Then, on CPU:

```bash
cd code && python -m event_ssm.eval.bootstrap \
  --capture puressm=$CAP/puressm_100k.pkl --capture analog=$CAP/analog_100k.pkl \
  --capture graded=$CAP/graded_100k.pkl --capture binary=$CAP/spike_100k.pkl \
  --pair puressm:analog --pair analog:graded --pair graded:binary \
  --n-boot 1000 --workers 8 --out ../results/stage21_bootstrap/ladder_100k
```

Output: `results/stage21_bootstrap/ladder_100k/bootstrap.{json,md}` → §5.7 table and slide 8.

Then the ladder figure (two panels: test AP per arm, and the paired cost of each step about zero):

```bash
python -m event_ssm.eval.ladder_plot --bootstrap ../results/stage21_bootstrap/ladder_100k/bootstrap.json \
  --label puressm=PureSSM --label analog=Analog --label graded=Graded --label binary=Binary \
  --gap-label puressm:analog="Neuron dynamics" --gap-label analog:graded=Sparsity \
  --gap-label graded:binary=Binarisation --out ../results/stage21_bootstrap/ladder_100k/ladder_AP
```

→ `ladder_AP.png` (300 dpi) + `ladder_AP.svg` for §5.7 and slide 8. Repeat with `--metric AP_L` (or any metric the
evaluator reports) for a large-object version.

**Optional, worth it:** the same for the earlier models — capture EventSSM 400k (`stage7_test_eval_local.sh`)
and pair it with `puressm_400k.pkl` (`--pair eventssm:puressm`) to turn the 46.2 vs 46.4 "tie" and the
+3.0 AP_L into measured intervals. S5-RVT via `stage8_baseline_eval_local.sh` only if baseline CIs are wanted.

## Reporting rule

Write: "95 % paired bootstrap CIs over the 470 test recordings (B = …); one training seed per arm." The
intervals cover test-set sampling only, not training randomness. A gap whose interval includes 0 is
reported as not distinguishable on Gen1.
