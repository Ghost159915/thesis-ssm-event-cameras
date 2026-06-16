# Stage 6 — Short Training Run + Monitoring Checklist

Short Gen1 training of the **Mamba-2 `ResNetMamba`** detector on a fixed-seed 10% train subset.
RVT's `train.py` is reused **unmodified**; only the backbone is ours (`stage6_train.py` registers it
first — see that file's docstring). Goal: confirm training works on real data and produce prelim
loss / LR / grad-norm curves + a rough val-mAP. **Per the terminal policy, the user runs this.**

## Prerequisites

1. **Full Gen1 extracted** (done): `data/gen1_raw/gen1/{train,val,test}` = 1458 / 429 / 470 recordings.
2. **Build the 10% train subset** (fast — just symlinks):
   ```bash
   PY=/home/ghost/miniforge3/envs/events_signals/bin/python
   cd /home/ghost/Desktop/thesis-ssm-event-cameras/code
   $PY -m event_ssm.integration.make_train_subset       # -> data/gen1_subset10 (train=10%, val/test=full)
   ```
   Logs the chosen recordings to `data/gen1_subset10/train_subset_recordings.txt` (reproducible, seed=1234).

## Launch

**Local (RTX 5070 Ti):**
```bash
bash code/event_ssm/scripts/stage6_run_local.sh
```

**Katana (SLURM):** fill the two `<--` placeholders (CUDA module, conda path), then from the repo root:
```bash
sbatch code/event_ssm/scripts/stage6_short_train.slurm
```

The **progress bar is ON** — you'll see Lightning's per-step tqdm bar (loss, it/s, ETA) live.

## Knobs (top of `stage6_run_local.sh`)

| knob | default | note |
|---|---|---|
| `MAX_STEPS` | 2000 | short prelim; raise for longer. `val_check_interval` is tied to it → one val pass at the end. |
| `BATCH` | 8 | drop to 4/2 if VRAM-bound (`d_state=64`; health showed 6.45 GB @ bs4 bf16). |
| `PRECISION` | `bf16-mixed` | bf16 autocast, no GradScaler (ISSUE-09). fp32 fallback: `PRECISION=32`. |
| `validation.limit_val_batches=<N>` | (unset = full) | optional add-on: cap val for a faster *rough* mAP (full val = 429 recordings, slow). |

> Note (ISSUE-09): for the **final** comparison runs, verify and match the S5-RVT baseline precision
> exactly — bf16 here is a deliberate short-run choice (speed/VRAM), flagged as a minor confound.

## Launcher wiring — issues found via dry-run & how they were fixed (2026-06-16)

Verified the launcher with Hydra `--cfg job` (composes + exits) and a bogus-path run (reaches the
modifier, then fails fast on the missing dataset). Three non-obvious wiring bugs were found and fixed
— recorded here so we don't re-hit them:

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `MissingConfigException: Primary config module 'config' not found` | `import train; train.main()` makes Hydra use **module/package** config search; `RVT/config` is a YAML dir, not a Python package. | Launcher runs train.py **as `__main__`** via `runpy.run_path(..., run_name="__main__")` (after registering) → Hydra uses **file-based** search → `RVT/config`. train.py stays unmodified. |
| 2 | `ConfigCompositionException: You must specify 'dataset'` | `+experiment/gen1=resnet_mamba` does not select the `dataset` group. | Add `dataset=gen1` to the command (all run scripts updated). |
| 3 | `ConfigAttributeError: Key 'mode' is not in struct` (`Could not override 'wandb.mode'`) | RVT's wandb config is struct-locked with no `mode` key. | Use `export WANDB_MODE=offline` (env var), **not** a `wandb.mode=` Hydra override. |

Proof the patched path is live end-to-end: the bogus-path run printed
`[resnet_mamba] set in_res_hw=(256, 320), num_classes=2` (our modifier — the stock one
`NotImplementedError`s on ResNetMamba) before failing on the missing dataset.

## Monitoring checklist — paste back after the run

- [ ] **Total loss** trends down; no NaN/Inf, no divergence.
- [ ] **Sub-losses** (cls / obj / iou) each trending down.
- [ ] **LR schedule** (OneCycle) — warmup then decay, as logged by `LearningRateMonitor`.
- [ ] **Grad-norm** (RVT's `GradFlowLogCallback`) — finite, not exploding/vanishing.
- [ ] **VRAM** peak (compare to the health probe's 6.45 GB @ bs4 bf16).
- [ ] **Throughput** (it/s from the progress bar).
- [ ] **Rough val-mAP** from the end-of-run validation (COCO mAP, IoU 0.50:0.95 — the Prophesee
      evaluator; label it "rough / short-run", not a final number).
- [ ] Any warnings/errors worth noting.

Paste these and I'll produce the loss/LR/grad curves and fill the Phase-B section of the Stage-6 report.
