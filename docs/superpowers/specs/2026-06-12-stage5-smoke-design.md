# Stage 5 — Smoke Testing (de-risk the assembled pipeline) — Design Spec

**Date:** 2026-06-12 · **Depends on:** Stage 4 (drop-in integration COMPLETE) · **Author decision-mode:**
autonomous (user delegated all Stage-5/6 decisions; this spec + the stage report are the review artifacts).

## 1. Goal

Prove that the **assembled real system** — drop-in `ResNetMamba` backbone + reused RVT YOLO-PAFPN/YOLOX
head + SimOTA loss + **real Gen1 data pipeline** + **real PyTorch-Lightning `Module`** — can actually
**learn** and is **numerically healthy**, before any real training (Stage 6). A model that cannot memorise a
handful of real samples is broken; this stage catches that for the cost of minutes, not GPU-days.

This stage also **reconciles the stale `stages/Stage_05_Smoke_Testing.md`** (which still describes the
superseded standalone `EventSSMDetector`: 10-channel input, `model.reset_state(...)`, Focal/GIoU loss) to the
as-built drop-in design (20-channel stacked histogram, `LstmStates` contract, **SimOTA** loss, real `Module`).

## 2. Scope / non-goals

- **In scope:** overfit a tiny **real-data** batch; gradient-flow health; VRAM @ batch sizes; eval-step
  latency; a results table + loss curve; doc reconciliation.
- **Out of scope:** real training (Stage 6), full Gen1 download (user-run), mAP evaluation (Stage 8), the
  β cross-clip training-state fix (Stage 6), the Finding-§8 architecture change (deferred — see §7).

## 3. Data reality & the smoke dataset

Locally only the Gen1 **`test/`** split is present (`data/gen1_raw/gen1/test/`, 470 sequences) — already in
the **correct preprocessed format** (`event_representations_v2/stacked_histogram_dt=50_nbins=10/
event_representations.h5` + `labels_v2/labels.npz`). The RVT data module expects a root with `train/val/test`
subdirs (`data/genx_utils/dataset_rnd.py`: `dataset_path / {train,val,test}`).

**Decision:** build a tiny **`data/gen1_smoke/`** tree whose `train/`, `val/`, `test/` each **symlink the same
K=2 real sequences** from `data/gen1_raw/gen1/test/`. Overfitting on test data is valid here — the objective is
to prove the *machinery learns*, not to measure generalisation. The builder is a reproducible script; the data
dir is gitignored (not committed). The full Gen1 **train** split (~40 GB) is **downloaded by the user** for
Stage 6.

## 4. Architecture of the solution (units)

Each unit is an independent, runnable script/module under `code/event_ssm/`.

### Unit 1 — Smoke dataset builder
`code/event_ssm/integration/make_smoke_dataset.py`
- Input: source split dir (default `data/gen1_raw/gen1/test`), K (default 2), dest (`data/gen1_smoke`).
- Creates `dest/{train,val,test}/<seq>` as **symlinks** to K real sequence dirs. Idempotent (clears/rebuilds).
- Prints the resulting tree + asserts each leaf has the `event_representations.h5` and `labels.npz`.
- **Interface:** `build_smoke_dataset(src, dest, k) -> Path`. **Depends on:** filesystem only.

### Unit 2 — Register + Hydra-compose harness
`code/event_ssm/integration/smoke_harness.py`
- `setup_paths()` (code + RVT on `sys.path`), `register_resnet_mamba()` **before** building the module.
- `compose_smoke_config(dataset_path, **overrides)`: Hydra `compose(config_name="train", overrides=[...])`
  from the RVT `config/` dir, with overrides: `dataset=gen1`, `model=resnet_mamba_yolox/default`,
  `dataset.path=<smoke>`, `dataset.train.sampling=random`, `batch_size.train=2`, `batch_size.eval=2`,
  `hardware.gpus=0`, `hardware.num_workers.train=0`, `hardware.num_workers.eval=0`,
  `wandb.project_name=null`/offline, `training.max_epochs=...`. Then run
  `dynamically_modify_train_config(cfg)` (injects `num_classes=2`, in_res_hw, etc.).
- **Interface:** returns a resolved `DictConfig`. **Depends on:** RVT `config/`, `register`.
- **Why it's its own unit:** config composition + register order is the single most failure-prone seam; a
  load-one-real-batch check (below) validates it before the heavier overfit.

### Unit 3 — Overfit smoke
`code/event_ssm/proofs/smoke_overfit.py`
- First a **load-one-batch check**: build `fetch_data_module(cfg)`, pull one train batch, assert the event
  tensor is 20-channel and labels are present (de-risks the data path).
- Build `fetch_model_module(cfg)` (real `Module` → builds `YoloXDetector` with our patched backbone).
- `pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs≈50, limit_train_batches=1,
  limit_val_batches=0, num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
  enable_progress_bar=False, gradient_clip_val=1.0)`. A small `Callback` captures per-epoch `train/loss`.
- **Pass:** final loss ≤ ⅓ initial (reduction ≥ 3×; target ≥ 10×) **and** no NaN at any epoch.
- **Fallback if reduction marginal due to batch variation:** shrink to K=1 recording / raise epochs; documented
  in the report (a smoke proves *learning trend + finite*, not a publication metric).
- Writes `results/smoke_test/overfit_loss_curve.png` + initial/final/reduction numbers.

### Unit 4 — Health probes
`code/event_ssm/proofs/smoke_health.py`
- **Grad-flow:** one real forward+backward; every **detection-path** trainable param has a finite, non-zero
  grad. **Document-excluded:** `backbone.temporal.0.*` (the stage-1 temporal Mamba is unused by the FPN —
  Finding §8 / Stage-4 report). Report any other missing/NaN grads.
- **VRAM:** peak `torch.cuda.max_memory_allocated` for forward+backward on a real-shaped clip at batch sizes
  {1, 2, 4}. Target < 10 GB @ bs 4 (5070 Ti has 16 GB).
- **Latency:** eval-mode **single-window step** latency (stateful step path), mean ± std over 200 iters after
  20 warmup, `torch.cuda.synchronize()`. Context: S5-RVT ≈ 12 ms/window.
- Writes rows into `results/smoke_test/smoke_results.md`.

### Unit 5 — Doc reconcile + report
- Reconcile `stages/Stage_05_Smoke_Testing.md` to the drop-in design (20-ch, `LstmStates`, SimOTA, real
  `Module`); keep the 4-test structure; fix the success criteria to match the as-built model.
- Code review of the Stage-5 changes; `reports/Stage_05_Smoke_Report.md`.

## 5. Data flow

`make_smoke_dataset` → `data/gen1_smoke/{train,val,test}` → `smoke_harness.compose_smoke_config` →
`fetch_data_module` (genx, random-access) ⟶ batch → real `Module.training_step` (inside `Trainer.fit`) →
`forward_backbone` (our backbone) → `forward_detect` (PAFPN+head+SimOTA) → loss → backward → optimiser step.
Eval/latency uses `forward_backbone(train_step=False)` step path + `forward_detect` (no targets).

## 6. Error handling / risks

| Risk | Mitigation |
|---|---|
| Hydra compose can't find configs / wrong override keys | Unit 2 is validated by the load-one-batch check before the overfit; overrides mirror baseline keys verified against `config/train.yaml`. |
| genx `random` sampler shuffles → batch varies epoch-to-epoch | K=2 (tiny pool) keeps variation low; fallback K=1; smoke asserts trend+finite, not a hard metric. |
| Streaming/mixed sampling incompatible with `limit_train_batches` | Force `dataset.train.sampling=random` (map-style). |
| Val path (Prophesee eval) errors on tiny data | `limit_val_batches=0`, `num_sanity_val_steps=0` — val never runs. |
| OOM at bs 4 | 5070 Ti 16 GB; bf16; if needed reduce `sequence_length`. Reported, not fatal. |
| wandb tries to log/login | offline/null logger (`logger=False` in Trainer; wandb disabled in config). |

## 7. Decision on Finding §8 (dead stage-1 temporal Mamba)

Stage 4 found `temporal[0]` receives no gradient (FPN `in_stages=[2,3,4]` drops `feats[1]`). **Decision: defer
the architecture change.** Injecting a fresh, unverified backbone change into the training path immediately
before the user runs real training is riskier than proving-it-learns-as-is. Stage 5 therefore **documents** the
exclusion in the grad-flow probe and **recommends** Finding-§8 option (a) (only instantiate temporal blocks for
FPN-consumed stages — removes ~9% dead params) as a small, isolated, user-approved change *after* the smoke
passes. Re-evaluated in the Stage 5 report.

## 8. Testing strategy

The proof scripts **are** the tests (they assert pass/fail and write visual artifacts — consistent with the
project's per-stage visual-proof convention). The pre-existing `pytest tests/` suite must stay green (20/20).
No new `pytest` unit tests are required for Stage 5 (it is an integration/smoke stage), but the harness'
`compose_smoke_config` gets a fast CPU-only assert (keys present) if practical.

## 9. Deliverables / success criteria

- `data/gen1_smoke/` builder; smoke harness; overfit + health proof scripts.
- `results/smoke_test/`: `smoke_results.md` (overfit reduction, grad-flow PASS, VRAM@bs4, latency) +
  `overfit_loss_curve.png`.
- Reconciled `stages/Stage_05_Smoke_Testing.md`; `reports/Stage_05_Smoke_Report.md`.
- **Success:** overfit loss reduction ≥ 3× with no NaN; all detection-path params get gradients; VRAM < 10 GB
  @ bs 4; eval-step latency reported vs ~12 ms baseline; `pytest` stays 20/20.
