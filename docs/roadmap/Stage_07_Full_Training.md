# Stage 7 — Full Training on Gen1
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

> **As-built note (2026-06-17):** the standalone `train.py --config gen1_full.yaml` / epoch-based
> design described below is **legacy**. The as-built path reuses RVT's `train.py` UNMODIFIED via the
> `code/event_ssm/scripts/stage7_midrun_local.sh` launcher (step-based OneCycle, Hydra config) — see
> `docs/superpowers/specs/2026-06-17-local-midrun-design.md` and
> `code/event_ssm/scripts/STAGE7_MIDRUN_RUN.md`. The local card is for the *mid-run* signal; the full
> 400k-step run is destined for Katana. This doc is retained for the protocol rationale
> (hyperparameters, expected-mAP ranges, checkpoint strategy) only.

---

## Overview

Train EventSSMDetector on the full Gen1 dataset for 100 epochs. This is the primary experimental run — the result that anchors your thesis and appears in your comparison table.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** 12–24 hours (compute) + 1–2 days (monitoring and analysis)  
**Prerequisite:** Stage 6 complete (short run passed, training dynamics confirmed stable, no red flags)

---

## Goal

- Best validation mAP checkpoint saved
- Training and validation curves (100 epochs)
- Final mAP numbers ready for Stage 8 evaluation on the test set

---

## Why This Stage Exists

### Contribution to Thesis B

This is the core experiment. The mAP number from Stage 8 evaluation of this trained model is what gets reported in:
- Your abstract
- Your results table
- Your comparison against S5-RVT (47.7) and other baselines
- Your conclusions about whether CNN-SSM hybrid can match Transformer-SSM

Every design decision from Stages 0–6 was made in service of this stage producing a valid, reproducible, scientifically comparable result.

---

## Training Protocol

### Why 100 Epochs

`Zubic_2024_SSM_EventCameras_CVPR` trained S5-RVT for 100 epochs on Gen1. `Gehrig_2023_RVT_EventDetection_CVPR` also used 100 epochs. To make your mAP comparable to the published 47.7, you must use the same epoch count. Using fewer epochs would unfairly disadvantage your model; using more would not be a meaningful difference but could be questioned.

### Complete Training Configuration

```yaml
# config/gen1_full.yaml

model:
  name: EventSSMDetector
  num_classes: 2
  pretrained_backbone: true

data:
  train_path: data/gen1/train/
  val_path: data/gen1/val/
  test_path: data/gen1/test/
  batch_size: 4
  sequence_length: 5        # T=5 windows per sequence
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2

training:
  epochs: 100
  warmup_epochs: 5
  
  optimiser:
    type: AdamW
    base_lr: 2.0e-4
    backbone_lr_scale: 0.1  # backbone LR = 2e-5
    betas: [0.9, 0.999]
    weight_decay: 0.05
  
  scheduler:
    type: cosine_annealing
    min_lr: 2.0e-6           # 1% of base LR
  
  gradient_clip:
    max_norm: 1.0
  
  precision: bfloat16        # for RTX 5070 Ti Tensor Cores
  
  checkpoint:
    dir: results/full_training/checkpoints/
    frequency: 10            # save every 10 epochs
    keep_best: true          # always save best validation mAP

logging:
  dir: results/full_training/logs/
  frequency: 1               # log every epoch
  metrics: [train_loss, val_map, lr, grad_norm, epoch_time]
```

---

## Differential Learning Rate (Critical)

Your ResNet-18 backbone is pretrained on ImageNet. The remaining components (FPN lateral convs, Mamba temporal modules, YOLOX head) are randomly initialised and need to learn from scratch.

If you use the same learning rate for all parameters:
- The randomly initialised components will have large gradients (they know nothing — large error signal)
- These large gradients will propagate backwards through the backbone and destroy the carefully learned ImageNet features

The fix is a differential (or discriminative) learning rate:

```python
# In your optimiser setup:
param_groups = [
    # Pretrained backbone: 10× lower LR
    {'params': model.backbone.parameters(), 'lr': 2e-5},
    # Random-init components: full LR
    {'params': model.fpn.parameters(), 'lr': 2e-4},
    {'params': model.temporal_P3.parameters(), 'lr': 2e-4},
    {'params': model.temporal_P4.parameters(), 'lr': 2e-4},
    {'params': model.temporal_P5.parameters(), 'lr': 2e-4},
    {'params': model.detection_head.parameters(), 'lr': 2e-4},
]
optimiser = torch.optim.AdamW(param_groups, weight_decay=0.05)
```

The cosine annealing scheduler should be applied to all groups simultaneously (the ratio between groups stays constant throughout training).

---

## Mixed Precision Training (bfloat16)

Your RTX 5070 Ti supports bfloat16 (bf16) mixed precision. Unlike fp16, bf16 has the same dynamic range as fp32 (8 exponent bits) but lower precision (7 mantissa bits vs 23). This means:
- No gradient scaling needed (fp16 requires GradScaler to prevent underflow)
- 1.5–2× training speedup from Tensor Core utilisation
- ~50% memory reduction for activations

```python
# Training loop with bf16:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    preds, hidden_state = model(window, hidden_state)
    loss = criterion(preds, targets)

# No GradScaler needed for bf16
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimiser.step()
optimiser.zero_grad()
```

**Note on mamba-ssm and bf16:** Verify that your mamba-ssm installation handles bf16 correctly. Some versions require explicit dtype specification in Mamba's internal operations. If you get dtype errors, add `.float()` casts around the Mamba forward call.

---

## Launching and Monitoring

### Starting the Training Job

```bash
# Use nohup so training continues if terminal disconnects
nohup python train.py --config config/gen1_full.yaml > results/full_training/train_log.txt 2>&1 &
echo "Training started, PID: $!"

# Or with explicit GPU selection:
CUDA_VISIBLE_DEVICES=0 nohup python train.py --config config/gen1_full.yaml \
    > results/full_training/train_log.txt 2>&1 &
```

### Monitoring

```bash
# Watch the log file for latest epoch results
tail -f results/full_training/train_log.txt

# Check GPU utilisation (want > 80%)
watch -n 5 nvidia-smi

# Check checkpoint files are being saved
ls -la results/full_training/checkpoints/ | tail -10

# Estimate remaining time
# If epoch 10 just completed and took 12 minutes total:
# Remaining: 90 epochs × (12/10 min/epoch) = ~108 minutes ≈ 1.8 hours
```

### Halfway Check (Epoch 50)

If validation mAP at epoch 50 is below 20 mAP, something may be wrong. Compare to your short run results — if training dynamics look similar (just slower because of full data volume), trust the process and continue. If significantly different, investigate.

A normal mAP progression for 100-epoch Gen1 training:
- Epochs 1–20: rapid improvement (10→30 mAP range)
- Epochs 20–60: steady improvement (30→40 mAP range)
- Epochs 60–100: slow improvement and plateau (40–47 mAP range)

---

## Checkpoint Strategy

Save every 10 epochs. Additionally, always save the best validation mAP checkpoint regardless of epoch.

**Use the BEST checkpoint (not last epoch) for test evaluation.** The model typically peaks at around epoch 70–90. Using the final checkpoint (epoch 100) may give slightly lower mAP if overfitting occurred in the last epochs.

```bash
# After training, identify best checkpoint:
grep "Best val mAP" results/full_training/train_log.txt

# Or check your checkpoint directory for 'best_model.pth'
ls -la results/full_training/checkpoints/
```

---

## Data Augmentation Details

You are using the same augmentations as `Gehrig_2023_RVT` and `Zubic_2024`. If your codebase inherited these augmentations correctly (Stage 2 audit: REUSE), nothing changes. For reference:

- **Horizontal flip (p=0.5):** Mirror the event stream horizontally. Since events and ground truth boxes are both flipped, this doubles effective dataset size.
- **Random spatial crop:** Crop to a fixed size for consistent batch dimensions. Matches the baseline's protocol from `Perot_2020_Gen1`.
- **No temporal augmentation** for the standard training run. Temporal augmentation (varying time window) is explored in Stage 9 (temporal generalisation).
- **No colour jitter:** Event cameras do not produce colour information.

---

## If Training Takes Too Long (> 48 Hours)

If the RTX 5070 Ti takes longer than expected:

**Option 1: UNSW Katana HPC Cluster**
```bash
# Transfer code and data to Katana
scp -r /path/to/code z1234567@katana.restech.unsw.edu.au:/home/z1234567/

# Submit SLURM job
sbatch train_katana.sh  # write a SLURM script for GPU job submission
```

**Option 2: Reduce sequence length**
Reduce T from 5 to 3 windows per sequence. Faster training, slightly less temporal context. Note this deviation from baseline.

**Option 3: Reduce batch size**
Reduce from 4 to 2. Use gradient accumulation to maintain effective batch size:
```python
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimiser.step()
    optimiser.zero_grad()
```

---

## What to Do While Training Runs

Training will run for 12–24 hours. Use this time to:
1. Start writing the Thesis B report methodology section (architecture description)
2. Prepare the Stage 9 (temporal generalisation) evaluation script
3. Prepare the Stage 10 (efficiency benchmarking) script
4. Begin literature review on recent papers that may have appeared since Thesis A

---

## Deliverable

- `results/full_training/checkpoints/best_model.pth`
- Loss and mAP curves (100 epochs) as CSV and plots
- Complete training log

## Success Criteria

Training completed without crashes. Best validation mAP checkpoint identified. Loss and mAP curves saved.

**Do NOT evaluate on the test set yet** — that is Stage 8. Test set evaluation must happen exactly once, with the final model, following the official protocol.

---

## Next Stage

→ **Stage 8: Evaluation and Analysis** — evaluate the best checkpoint on the official Gen1 test set.
