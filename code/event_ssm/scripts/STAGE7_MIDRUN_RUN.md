# Stage 7a — Local Mid-Run + Monitoring Checklist

Overnight full-Gen1 training of the Mamba-2 `ResNetMamba` detector for a trustworthy
"is this architecture competitive?" signal. RVT's `train.py` is reused **unmodified**; only the
backbone is ours (`stage6_train.py` registers it first). **Per the terminal policy, the user runs this.**

This is a **rough signal**, not a clean S5-RVT comparison: bf16 is a deliberate speed/VRAM choice
(the ISSUE-09 precision confound + the full final protocol belong to the Katana run). Validation is
**full** (the whole 429-recording val split) — see the launch fix below for why it can't be a fraction.

## Why these settings (the one thing that matters)

RVT uses a **OneCycle** LR schedule over `total_steps = max_steps`. A run that **completes** its
schedule (LR fully annealed) gives a higher, more representative mAP than a longer run cut off
mid-anneal. So `MAX_STEPS=100000` is sized to *finish overnight even at the pessimistic ~2.8 it/s*
(~10.5 h incl. 5 full validations; ~6.8 h at the observed 4.5 it/s). 100k steps = **25% of the 400k
baseline budget**, on full data.

## Prerequisites

Full Gen1 extracted at `data/gen1_raw/gen1/{train,val,test}` = 1458 / 429 / 470 recordings (done).
No subset build needed — the mid-run points straight at the full splits.

## Launch

```bash
bash code/event_ssm/scripts/stage7_midrun_local.sh
```

Lightning's per-step tqdm progress bar is ON (loss / it·s⁻¹ / ETA). Artifacts land under
`results/stage7_midrun/` (pinned via `hydra.run.dir`).

## Resume (stop anytime)

Validation runs every `VAL_EVERY` steps and triggers the checkpoint callback, which writes a
`last_epoch=…-step=….ckpt` (save_last) plus the best-`val/AP` checkpoint. To resume full training
state (optimizer + scheduler + global step → the OneCycle schedule continues):

```bash
STAGE7_RESUME="$(ls -t results/stage7_midrun/**/last_epoch=*-step=*.ckpt | head -1)" \
  bash code/event_ssm/scripts/stage7_midrun_local.sh
```
(Confirm the exact checkpoint path from Lightning's `saving checkpoint to …` log line on the first
validation at step 10000.)

## Knobs (top of `stage7_midrun_local.sh`)

| knob | default | note |
|---|---|---|
| `MAX_STEPS` | 100000 | sized to complete overnight; pushing past ~120k risks overrun at 2.8 it/s |
| `VAL_EVERY` | 20000 | FULL val + checkpoint cadence (5 points); lower it for finer resume / more points (more val overhead) |
| `VAL_FRAC` | 1.0 | full val (clean mAP). Must be `1.0` or an **int** (num batches) — Gen1 val is an `IterableDataset`, so a fraction is rejected |
| `BATCH` | 4 | bs8 OOMs at seq_len=21 on 16 GB; drop to 2 if tight |
| `PRECISION` | bf16-mixed | fp32 fallback: `PRECISION=32` |

## Launch fixes — bugs found running the launcher (2026-06-17)

The Hydra `--cfg job` dry-run composes the config and **exits before `trainer.fit`**, so it cannot
catch trainer-/dataloader-time errors. The first real launch surfaced one; recorded so we don't re-hit it.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `MisconfigurationException: When using an IterableDataset, Trainer(limit_val_batches) must be 1.0 or an int` — crash at the pre-train **val sanity check** (before step 0). | The Gen1 val loader is a streaming **`IterableDataset`**; Lightning forbids a *fractional* `limit_val_batches`. The mid-run shipped with `VAL_FRAC=0.25` (a fraction). | Use **full val** `VAL_FRAC=1.0` (the Stage-6-proven setting — it validated cleanly at `val/AP=0.125`) and cut overhead via frequency instead: `VAL_EVERY=20000` (5 full-val points). For a faster *rough* val, set `VAL_FRAC` to an **int** = number of val batches — never a fraction. |

## Monitoring checklist — paste back after (or during) the run

- [ ] **Total loss** trends down; no NaN/Inf, no divergence.
- [ ] **Sub-losses** (cls / obj / iou) each trending down.
- [ ] **LR schedule** (OneCycle) — warmup then anneal toward ~0 by step 100k.
- [ ] **Grad-norm** (`GradFlowLogCallback`) — finite, not exploding/vanishing.
- [ ] **VRAM** peak (compare to ~6.45 GB @ bs4 bf16 from the health probe).
- [ ] **Throughput** (it·s⁻¹ from the progress bar) — sanity-check vs the 2.8–4.5 it/s envelope.
- [ ] **Val-mAP curve** — the ~5 full-val points (COCO mAP, IoU 0.50:0.95; Prophesee evaluator).
      Label "mid-run / short schedule", not a final S5-RVT-comparable number.
- [ ] Any warnings/errors worth noting.

Paste these and I'll produce the loss / LR / grad / mAP-trajectory curves and write the mid-run
results paragraph for the Thesis B report.
