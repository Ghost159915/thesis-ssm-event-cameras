# Stage 6 — Short Training Run (Validation Run)
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Train EventSSMDetector on 10% of the Gen1 dataset for 20 epochs. This is not about achieving good performance — it is about confirming training dynamics are stable and generating preliminary results for the Thesis B report.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** 1–3 hours (compute) + 1 day (analysis)  
**Prerequisite:** Stage 5 complete (all smoke tests passing)

---

## Goal

- Stable loss curve over 20 epochs (decreasing, not oscillating)
- Validation mAP improving (even if slowly)
- No crashes, OOM, or NaN losses
- Preliminary results for Thesis B report (mAP curve, training dynamics)
- Time-per-epoch measurement to estimate full training duration

---

## Why This Stage Exists

### Contribution to Thesis B

The Thesis B report specifically requires you to report *"achievements done so far, expanding on the preliminary results that you already reported in the Thesis A document."* Stage 6 produces those preliminary results. Even with only 20 epochs on 10% of the data, you will have:
- A learning curve (proof that the architecture trains)
- A preliminary mAP number (probably 15–30 mAP@0.5)
- Evidence that training dynamics are stable

This is the minimum viable experimental result for Thesis B. If you reach Stage 6 before the Thesis B deadline, you can write the report. If you have also completed Stage 7 (full training), even better.

### Why 10% Data, 20 Epochs (Not Full Training)

Full training on Gen1 takes approximately 12–24 hours. The short run takes 1–3 hours. Running the short run first:
1. Confirms nothing will crash mid-way through the 24-hour full run
2. Gives you time to observe and adjust learning rate if needed
3. Provides preliminary results quickly for the thesis report
4. Gives you a per-epoch timing estimate to plan compute resources

---

## Training Protocol

### Must Match S5-RVT Exactly

**This cannot be emphasised enough:** every hyperparameter must match `Zubic_2024_SSM_EventCameras_CVPR` and `Gehrig_2023_RVT_EventDetection_CVPR`. Any deviation creates a confound — a difference in mAP that could be due to training settings rather than architecture.

| Hyperparameter | Value | Source |
|---|---|---|
| Optimiser | AdamW | `Gehrig_2023_RVT` |
| Base LR | 2×10⁻⁴ | `Gehrig_2023_RVT` |
| Backbone LR | 2×10⁻⁵ (0.1× base) | Differential LR for pretrained backbone |
| β1, β2 | 0.9, 0.999 | Standard AdamW |
| Weight decay | 0.05 | `Gehrig_2023_RVT` |
| LR schedule | Cosine annealing + 5-epoch linear warmup | `Gehrig_2023_RVT` |
| Batch size | 4 sequences | `Zubic_2024` |
| Sequence length | T=5 windows | `Zubic_2024` |
| Gradient clipping | max_norm=1.0 | `Gehrig_2023_RVT` |
| Precision | bfloat16 | RTX 5070 Ti supports bf16 |
| Augmentation | Horizontal flip, random crop | Same as baseline |
| Dataset | Gen1, 10% of train split | Short run |

### The 10% Subset

Subsample 10% of training recordings — take the first 10% by filename order (not random sampling). This preserves temporal sequence integrity within each recording. Do NOT subsample individual windows — take complete recordings.

```python
# In your dataset config or data loading code:
train_recordings = sorted(glob.glob('data/gen1/train/*.hdf5'))
short_run_recordings = train_recordings[:int(0.1 * len(train_recordings))]
```

### Short Run Config

Create `config/gen1_short.yaml` as a copy of `config/gen1.yaml` with these changes:
```yaml
training:
  epochs: 20
  data_fraction: 0.1   # or list specific recording files
  checkpoint_dir: results/short_run/checkpoints/
  log_dir: results/short_run/logs/

validation:
  frequency: 1         # evaluate every epoch (important for short run)
```

---

## Logging Requirements

You need these metrics logged per epoch:
1. Training loss (total, and per loss component: Focal + GIoU)
2. Validation mAP@0.5 (cars, pedestrians, overall)
3. Learning rate (to verify schedule is working)
4. Gradient norm (to check for explosion)
5. Training time per epoch (for full-run estimate)

Your existing training code should already log these for S5-RVT. If not, add logging before this stage.

```python
# Minimum logging at each epoch:
print(f"Epoch {epoch}/{total_epochs} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Val mAP: {val_map:.2f} | "
      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
      f"Time: {epoch_time:.1f}s")
```

---

## Expected Results

### Expected mAP Range

After 20 epochs on 10% of Gen1:
- **Car AP@0.5:** Roughly 20–35
- **Pedestrian AP@0.5:** Roughly 10–20
- **Overall mAP@0.5:** Roughly 15–30

These numbers are not the point. The point is that mAP improves over 20 epochs.

### Expected Loss Curve Shape

```
Epoch  1-5:  Loss drops quickly (warmup phase, gradients large)
Epoch  5-15: Loss decreases steadily but slower
Epoch 15-20: Loss stabilises or continues slow decrease
```

Validation mAP should correlate with training loss decrease — both should improve together.

### Time Estimate

After measuring epoch time in the short run:
- Short run epoch time: T_short seconds
- Full training estimate: T_short × (100 / 20) × (10 / 1) seconds
  - (multiply by 100/20 for epochs, by 10 for full dataset)
- Convert to hours: full_training_seconds / 3600

---

## Red Flags That Require Investigation Before Full Training

| Symptom | Likely Cause | Fix |
|---|---|---|
| Loss oscillates wildly | LR too high | Reduce base LR to 5×10⁻⁵ |
| Loss decreases but mAP stays at 0 | Evaluation pipeline mismatch | Verify evaluation code uses same protocol as S5-RVT |
| Loss decreases normally but car AP >> pedestrian AP by extreme margin | Normal (cars are easier) but check | If pedestrian AP is 0 after 20 epochs, check class weighting in Focal Loss |
| OOM after a few epochs | Memory leak (state not detached) | Add explicit `del` and `torch.cuda.empty_cache()` |
| GPU utilisation < 60% | Data loading bottleneck | Increase DataLoader num_workers |
| NaN loss at epoch 5+ | Gradient explosion (not caught early) | Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Backbone not training (backbone LR works) | `no_grad()` context wrapping backbone | Check `get_param_groups()` setup |

---

## Comparing to S5-RVT Short Run

If you also have S5-RVT results from a short run (or from the literature), compare:

| Model | 20 epochs, 10% data mAP | Comment |
|---|---|---|
| S5-RVT (from literature — full 100 epochs) | 47.7 | Reference, not directly comparable |
| EventSSMDetector (your result) | ? | Preliminary — thesis report content |

You cannot directly compare 20-epoch/10%-data results to 100-epoch/100%-data published results. Acknowledge this in your Thesis B report: *"Preliminary results after 20 epochs on 10% of the training data show mAP@0.5 of X, with loss decreasing steadily, indicating stable training dynamics. Full results after 100 epochs of training are reported in Section X."*

---

## Results to Save

```bash
results/short_run/
├── train_loss.csv          # epoch, total_loss, focal_loss, giou_loss
├── val_map.csv             # epoch, car_AP, ped_AP, mAP
├── learning_rate.csv       # epoch, lr
├── gradient_norm.csv       # epoch, grad_norm
├── loss_curve.png          # plot of training loss
├── map_curve.png           # plot of validation mAP
├── checkpoints/
│   ├── epoch_10.pth
│   └── epoch_20.pth
└── README.md               # notes on run settings, observations
```

---

## Deliverable

- Loss curve plot and mAP curve plot (20 epochs, 10% data)
- Time-per-epoch measurement and full-training estimate
- All results saved to `results/short_run/`
- A written paragraph describing the results for the Thesis B report

## Success Criteria

Loss is lower at epoch 20 than epoch 1. Validation mAP is improving. No crashes. Time-per-epoch gives a plausible full-training estimate.

---

## Next Stage

→ **Stage 7: Full Training on Gen1** — 100 epochs on the full dataset to produce your primary thesis result.
