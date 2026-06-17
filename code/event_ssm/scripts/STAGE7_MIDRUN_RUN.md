# Stage 7a — Local Mid-Run + Monitoring Checklist

Overnight full-Gen1 training of the Mamba-2 `ResNetMamba` detector for a trustworthy
"is this architecture competitive?" signal. RVT's `train.py` is reused **unmodified**; only the
backbone is ours (`stage6_train.py` registers it first). **Per the terminal policy, the user runs this.**

This is a **rough signal**, not a clean S5-RVT comparison: bf16 + 25%-capped validation are
deliberate speed choices (the ISSUE-09 precision confound + full-val belong to the final Katana run).

## Why these settings (the one thing that matters)

RVT uses a **OneCycle** LR schedule over `total_steps = max_steps`. A run that **completes** its
schedule (LR fully annealed) gives a higher, more representative mAP than a longer run cut off
mid-anneal. So `MAX_STEPS=100000` is sized to *finish overnight even at the pessimistic ~2.8 it/s*
(~10.3 h incl. validation; ~6.4 h at the observed 4.5 it/s). 100k steps = **25% of the 400k baseline
budget**, on full data.

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
| `VAL_EVERY` | 10000 | val + checkpoint cadence; halve to 5000 for finer resume granularity (2× val overhead) |
| `VAL_FRAC` | 0.25 | fraction of the 429-recording val set per pass (rough mAP) |
| `BATCH` | 4 | bs8 OOMs at seq_len=21 on 16 GB; drop to 2 if tight |
| `PRECISION` | bf16-mixed | fp32 fallback: `PRECISION=32` |

## Monitoring checklist — paste back after (or during) the run

- [ ] **Total loss** trends down; no NaN/Inf, no divergence.
- [ ] **Sub-losses** (cls / obj / iou) each trending down.
- [ ] **LR schedule** (OneCycle) — warmup then anneal toward ~0 by step 100k.
- [ ] **Grad-norm** (`GradFlowLogCallback`) — finite, not exploding/vanishing.
- [ ] **VRAM** peak (compare to ~6.45 GB @ bs4 bf16 from the health probe).
- [ ] **Throughput** (it·s⁻¹ from the progress bar) — sanity-check vs the 2.8–4.5 it/s envelope.
- [ ] **Rough val-mAP curve** — the ~10 points (COCO mAP, IoU 0.50:0.95; Prophesee evaluator).
      Label "rough / mid-run / 25% val", not a final number.
- [ ] Any warnings/errors worth noting.

Paste these and I'll produce the loss / LR / grad / mAP-trajectory curves and write the mid-run
results paragraph for the Thesis B report.
