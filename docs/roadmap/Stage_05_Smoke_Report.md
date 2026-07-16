# Stage 5 — Smoke Testing of the Assembled Drop-in System — Stage Report

**Date:** 2026-06-12 · **Branch:** `main` · **Status:** ✅ COMPLETE (code-reviewed, 21/21 tests green)

> One-line summary: the **real** assembled system — drop-in `ResNetMamba` backbone + RVT YOLO-PAFPN
> neck + YOLOX head + SimOTA loss + the real Gen1 data pipeline + the real PyTorch-Lightning `Module`
> — **overfits a fixed real Gen1 batch (≈4–5× loss reduction, monotonic, no NaN)** and is numerically
> healthy (grad-flow OK, VRAM 2.5 GB @ bs4, eval step ≈8 ms). No baseline files edited; the system is
> wired correctly end-to-end and ready for a real short training run (Stage 6).

---

## 1. What this stage set out to do

Prove — before spending GPU-hours on real training — that the assembled drop-in detector can actually
learn and is numerically sound. The overfit test is the gold standard: a model that cannot memorise a
single real batch is fundamentally broken (gradient, loss, or wiring fault). Stage 5 drives the **real**
Lightning `Module` and the **real** Hydra-composed `train` config (the same path Stage 6 will use), not
a hand-rolled loop, so a pass here de-risks the whole training stack.

Spec: `docs/specs/2026-06-12-stage5-smoke-design.md`.
Plan: `docs/plans/2026-06-12-stage5-smoke.md` (6 tasks).

---

## 2. What was ADDED

| File | Purpose |
|---|---|
| `code/event_ssm/integration/make_smoke_dataset.py` | Builds a tiny `data/gen1_smoke/{train,val,test}` tree by symlinking 2 real Gen1 `test/` sequences (idempotent; data dir gitignored). Lets the real data module run on a few sequences. |
| `code/event_ssm/integration/smoke_harness.py` | `setup_paths()` (adds `code/`+RVT to `sys.path`, imports `hdf5plugin` for blosc-compressed H5), `register()`, and `compose_smoke_config()` — Hydra-composes the **real** RVT `train` config with smoke overrides via the baseline mechanism `model=rnndet +experiment/gen1=resnet_mamba`. |
| `code/event_ssm/configs/experiment/gen1/resnet_mamba.yaml` | Tracked Hydra **experiment** config: pulls in `/model/resnet_mamba_yolox` + the training/dataset blocks (mirrors `experiment/gen1/default.yaml`, drops MaxViT-only `partition_split_32`). Symlinked into the gitignored RVT config tree. |
| `code/event_ssm/proofs/smoke_overfit.py` | Overfit proof: real `Module` + `pl.Trainer(overfit_batches=1)` overfits a fixed real batch; asserts ≥3× loss reduction + no NaN; writes the loss curve. |
| `code/event_ssm/proofs/smoke_health.py` | Health probes: grad-flow (deterministic path vs SimOTA positive-only branches), VRAM sweep {1,2,4}, eval single-window step latency. |
| `results/smoke_test/overfit_loss_curve.png`, `results/smoke_test/smoke_results.md` | Visual proof artifacts (force-added; `results/` is otherwise gitignored). |
| 1 new test | `test_config_modifier_injects_hw_and_num_classes` — composes the real config and asserts the modifier patch injects `in_res_hw=(256,320)` and `head.num_classes=2`. |

## 3. What was CHANGED

| File | Change |
|---|---|
| `code/event_ssm/integration/register.py` | Refactored into `register_backbone_builder()` + `register_config_modifier()` + `register_resnet_mamba()`. The new config-modifier patch makes the stock (MaxViTRNN-only) `dynamically_modify_train_config` handle our backbone: sets `backbone.in_res_hw` (multiple-of-32 padding) and injects `head.num_classes`, dispatching on `name=="rnndet" and backbone=="ResNetMamba"`, falling through to `orig` otherwise. Also replicates the stock **SLURM_JOB_ID** bookkeeping (added in review) so Stage-6 SLURM runs keep `config.slurm_job_id`. |
| `docs/roadmap/Stage_05_Smoke_Testing.md` | Reconciled the stale standalone-`EventSSMDetector` plan to the as-built drop-in design: STATUS banner, a stale-vs-as-built fact table (20-ch input, `RNNStates`/`LstmStates` contract, SimOTA + IoU `1−iou²`, real `YoloXDetector`/`Module`), and a filled results table. Inline pedagogical code kept; proof scripts are authoritative. |

## 4. What was REMOVED

Nothing deleted. The plan's pre-execution `smoke_overfit.py` approach (`limit_train_batches=1` + random
sampling + stock OneCycle scheduler) was **superseded** during execution by `overfit_batches=1` +
scheduler-off + fixed lr (see §7) — a behaviour change, not a file removal. No baseline RVT files edited.

---

## 5. Proof of working (visual)

**Overfit** (`results/smoke_test/overfit_loss_curve.png`, RTX 5070 Ti, `events_signals`, bf16, 150 ep):
monotonic decrease on a fixed real Gen1 batch through the full stack, e.g. **23.4 → 4.7 (5.0×)** this
run; **19.0 → 4.9 (3.9×)** an earlier run. Always ≥3× (the gate); no NaN.

**Health** (`results/smoke_test/smoke_results.md`):

| test | result | notes |
|---|---|---|
| gradient flow (deterministic path) | **PASS** | real-missing=0, nan=0 (excl. documented dead `temporal[0]`) |
| SimOTA pos-only head branches w/o grad | 0–10 / 30 | data-dependent per-level matching; **≥1 alive enforced** |
| VRAM @ bs1/bs2/bs4 | **0.92 / 1.47 / 2.50 GB** | target < 10 GB @ bs4 ✓ |
| eval single-window step latency | **≈8 ms** | vs ~12 ms S5-RVT/window ✓ |
| max throughput | **≈125 Hz** | window dt=50 ms ⇒ need < 50 ms ✓ |

Full test suite: **21 passed** (`pytest tests/ -q`).

---

## 6. What went well

- **The drop-in really is end-to-end functional.** The real Lightning `Module`, the real Hydra config
  (selected exactly like the S5-RVT baseline), the real Gen1 data pipeline, and the SimOTA loss all
  cooperate with the new backbone and *learn* — strong evidence the controlled-experiment wiring is
  correct and Stage 6 will run.
- **Efficiency signal is favourable.** Eval single-window step ≈8 ms < S5-RVT's ~12 ms, and VRAM is
  tiny (2.5 GB @ bs4 on a 16 GB card) — head-room for larger batch / sequence length in real training.
- **Faithful Stage-6 dry-run.** Because the smoke composes the *real* `train` config, the registration
  monkeypatch, the config modifier, and the data path were all exercised together, surfacing the
  `hdf5plugin` and modifier issues now rather than mid-training.

## 7. What went wrong (and how it was handled)

- **Weak/noisy overfit at first (≈1.2×).** Two compounding causes: (1) `limit_train_batches=1` with
  random sampling drew a *different* batch each epoch (no fixed target to memorise); (2) the stock
  OneCycle scheduler (`total_steps=400 000`) sat in warmup (~1e-5) over a ~150-step run. **Fix:**
  `overfit_batches=1` (Lightning re-fetches index 0 each epoch — a fixed target with *fresh* label
  objects), plus `lr_scheduler.use=False` + fixed `lr=1e-3`. Result jumped to ≈4–5×.
- **Captured-batch replay corrupts labels.** An attempt to capture one batch and re-feed it hit an
  assertion: `training_step → to_prophesee() → numpy_()` mutates the label objects to numpy **in place,
  irreversibly**, breaking the next epoch's device transfer. `overfit_batches=1` (fresh objects each
  epoch) is the correct mechanism and avoids this entirely.
- **Wrong config-selection mechanism.** The plan's `model=resnet_mamba_yolox/default` gave the modifier
  no `model.name` to dispatch on. **Fix:** use the baseline mechanism `model=rnndet
  +experiment/gen1=resnet_mamba` (added the experiment config), which also makes the smoke a more
  faithful copy of how Stage 6 selects the model.
- **Stock modifier raised `NotImplementedError`** for any non-MaxViTRNN backbone. **Fix:**
  `register_config_modifier()` patches it to handle `ResNetMamba`.
- **`hdf5plugin` missing** → h5py `can't open directory (/usr/local/lib/plugin)` on the blosc-compressed
  Gen1 H5. **Fix:** import `hdf5plugin` in `setup_paths()` (mirrors `train.py`).
- **Grad-flow "FAIL" false alarm.** The first health run flagged 10 missing grads — exactly the SimOTA
  **positive-only** level-0 head branches (`cls_convs.0`, `cls_preds.0`, `reg_preds.0`). With a single
  synthetic box, SimOTA assigned no positive anchor to the finest level, so those branches legitimately
  got no gradient (the objectness BCE covers *all* anchors → obj/reg-stem always fire; cls/box reg only
  fire on positives). **This is an assignment artifact, not a wiring fault.** Handled honestly (not
  masked, mirroring the Stage-4 `temporal[0]` finding): multi-scale synthetic boxes + a check that
  splits the **deterministic path** (backbone − `temporal[0]`, FPN, obj/reg-stem — *must* get grad)
  from the positive-only branches (may be grad-less on an unmatched level), enforcing **≥1 positive
  branch alive** so a *total* cls/box disconnection still fails.

## 8. Finding carried forward — overfit reduction is stochastic, not a fixed number

The overfit reduction varies run-to-run (observed 3.9× and 5.0×) because the run is unseeded (random
init + CUDA/mamba nondeterminism). The reported headline is therefore a **range (≈4–5×)** that always
clears the ≥3× gate, not a single reproducible figure. The grad-flow positive-only count is likewise
data/RNG-dependent (0–10 of 30 across runs) — which is why the health check enforces "≥1 alive" rather
than an exact count. The deterministic-path grad check, VRAM, and latency are stable.

## 9. Deferred / open items

- **Finding §8 (Stage 4) — dead stage-1 temporal Mamba.** The FPN's `in_stages=[2,3,4]` leaves the
  stage-1 temporal block (`backbone.temporal.0.`, ~9% of backbone params) unused / grad-less. Recommended
  fix (build temporal only for FPN-consumed stages) is **user-approved**, deferred to after the smoke.
- **β cross-clip training-state parity.** Training still zero-inits temporal state per clip; the custom
  differentiable scan (full TBPTT) or eval-without-state decision is a **Stage-6 prerequisite**
  (Stage-3 spec §9).
- **Full Gen1 train split.** The smoke uses 2 `test/` sequences; the real short training (Stage 6) needs
  the full Gen1 **train** split — user to download.

## 10. Code review outcome

Reviewed via the requesting-code-review subagent over `3ab962c..f62abbe`. **No Critical issues**;
verdict "ready with fixes". All three Important issues fixed before sign-off: (1) tightened the grad-flow
assertion so it cannot RNG-drift (deterministic-vs-positive-only split + ≥1-alive enforcement);
(2) added a CPU-fast config-modifier test (`test_config_modifier_injects_hw_and_num_classes`);
(3) restored the stock SLURM_JOB_ID bookkeeping in the patched modifier. Plus two correctness Minors
(default 150 epochs; restored the `assert reduction>=3.0` gate). The reviewer confirmed the three plan
deviations (§4, §7) are sound improvements and the grad-flow split is honest characterization, not
bug-masking (verified against the YOLOX loss source). Suite: 21 passed.

## 11. Commits (this stage)

| SHA | Subject |
|---|---|
| `30a5df7` | docs(stage5): smoke-testing design spec |
| `3ab962c` | docs(stage5): smoke-testing implementation plan (6 tasks) |
| `adec0a8` | feat(stage5): smoke dataset builder (symlink real test seqs) |
| `da727b0` | feat(stage5): experiment config + register patches config-modifier; harness composes real train config |
| `dc5b473` | feat(stage5): overfit smoke on real Gen1 batch (loss reduction + curve) |
| `6a0d550` | feat(stage5): health probes (grad-flow + VRAM sweep + eval-step latency) |
| `f62abbe` | docs(stage5): reconcile smoke-testing doc to the drop-in design |
| `ca67c91` | fix(stage5): address code-review (config-modifier test, tighter grad gate, SLURM bookkeeping) |

## 12. Next: Stage 6 — Short Training Run

A short real training run on the full Gen1 train split to validate training dynamics (loss/LR curves,
no divergence) before the full run. Per the terminal policy, **the training command will be handed to
the user to run** (Claude runs only smokes/tests). Prerequisites: resolve **β cross-clip state parity**
and download the **full Gen1 train split**. The Finding §8 architecture change should also land before
training to avoid carrying dead weight.
