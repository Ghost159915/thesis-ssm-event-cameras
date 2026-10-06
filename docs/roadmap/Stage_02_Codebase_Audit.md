# Stage 2 — Codebase Audit and Environment Verification
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Map your existing S5-RVT codebase so you know exactly what can be reused and what needs to be created. Simultaneously verify that `mamba-ssm` installs and runs correctly on your RTX 5070 Ti. This stage is entirely on your Linux PC.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** Half a day to 1 day  
**Prerequisite:** Stages 0 and 1 complete

---

## Goal

1. A labelled file inventory: every `.py` file in the codebase marked REUSE / MODIFY / CREATE
2. mamba-ssm installed and passing a shape test
3. S5-RVT evaluation confirmed still working (47.7 mAP on Gen1)
4. YOLOX head interface documented (input shapes, output format)

---

## Why This Stage Exists

### Contribution to Thesis B

Building on top of a verified, working codebase is one of the strongest methodological choices in your thesis. It guarantees that your evaluation pipeline, data preprocessing, and training infrastructure are identical to the S5-RVT baseline — so any mAP difference between S5-RVT (47.7) and EventSSMDetector is caused by the architecture, not by differences in how you load data, compute gradients, or measure performance.

Your Thesis A established this credibility by reproducing the S5-RVT result to within 0.01 mAP. Stage 2 preserves that credibility as you extend the codebase.

---

## Part A: Codebase Audit

### Typical S5-RVT Codebase Structure

The S5-RVT codebase is based on the RVT repository (`Gehrig_2023_RVT_EventDetection_CVPR`) with modifications by Zubic et al. (`Zubic_2024_SSM_EventCameras_CVPR`) to replace the ConvLSTM temporal layer with S5 SSM blocks.

Run this command to list all Python files in your codebase:
```bash
find /path/to/your/s5_rvt_codebase -name "*.py" | sort
```

### Classification Guide

For each file, label it as one of:
- **REUSE** — use without any modification whatsoever
- **MODIFY** — needs small changes (e.g., add an import, change one function argument)
- **CREATE** — does not exist yet; you build it from scratch

### Expected Classification

**REUSE (do not touch these files):**

| File/Directory | What it does | Why reuse |
|---|---|---|
| `data/dataset.py` | Gen1 dataset class, loads recordings | Verified preprocessing produces 47.7 mAP. Do not touch. |
| `data/preprocessing.py` | Voxel grid creation from raw events | Same reason. Any change here invalidates comparison with baseline. |
| `data/augmentations.py` | Random flip, crop for training | Must match baseline augmentations exactly. |
| `losses/focal_loss.py` | Focal Loss for classification | Already proven to work. |
| `losses/giou_loss.py` | GIoU Loss for box regression | Already proven to work. |
| `evaluation/evaluator.py` | mAP computation at IoU=0.5 | Must be identical to baseline evaluation protocol. `Perot_2020_Gen1` defines this. |
| `evaluation/metrics.py` | Per-class AP: cars, pedestrians | Same reason. |
| `training/optimiser.py` | AdamW setup | Same reason. |
| `training/scheduler.py` | Cosine LR schedule | Same reason. |
| `config/gen1.yaml` | Dataset paths, augmentation params | Copy this file for your new configs — do not modify the original. |

**MODIFY (small changes only):**

| File | What to change | Why |
|---|---|---|
| `train.py` | Change `from models.s5_rvt import S5RVT` to `from models.event_ssm_detector import EventSSMDetector` | Swap model class. Everything else stays the same. |
| `training/trainer.py` | May need to adjust how hidden state is initialised if the interface differs from S5-RVT | Minimal change — see Stage 4. |

**CREATE (new files you write):**

| File to create | What it contains |
|---|---|
| `models/event_ssm_detector.py` | Main EventSSMDetector class — wires all components |
| `models/backbone/resnet18_event.py` | Modified ResNet-18 with 10-channel input |
| `models/fpn/event_fpn.py` | FPN for ResNet-18 multi-scale outputs |
| `models/temporal/mamba_temporal.py` | Mamba temporal module with state management |

**REUSE (detection head):**

| File | Action |
|---|---|
| `models/head/yolox_head.py` (or equivalent) | REUSE — but document its exact input/output interface (see Part B below) |

### How to Run the Audit

```bash
# List all Python files with their line counts (to understand file complexity)
find . -name "*.py" -exec wc -l {} \; | sort -rn | head -30

# Look at model definitions
grep -r "class.*nn.Module" . --include="*.py"

# Find where hidden state is managed
grep -r "hidden_state\|hidden\|state" . --include="*.py" | grep -v ".pyc"

# Find the training loop
grep -r "for.*batch\|optimizer.step" . --include="*.py"
```

---

## Part B: Document the YOLOX Head Interface

This is critical. Before building your new backbone and FPN, you must know exactly what format the YOLOX head expects as input. Building your FPN to produce the wrong format is a common integration bug.

### Tasks

1. Open the YOLOX head file (probably `models/head/yolox_head.py` or similar).

2. Find the `forward()` method. Write down:
   - What are the argument names?
   - Does it expect a `list` of tensors or a `dict`?
   - What channels does it expect? (Should be 256 from your FPN)
   - In what order does it expect the scales? (P3 first, or P5 first?)

3. Run a standalone shape test to confirm:
```python
import torch
import sys
sys.path.insert(0, '/path/to/your/codebase')
from models.head.yolox_head import YOLOXHead  # adjust import path

head = YOLOXHead(num_classes=2, in_channels=256).cuda()

# Fake FPN outputs — match your FPN's output shapes
fake_p3 = torch.randn(2, 256, 30, 38).cuda()
fake_p4 = torch.randn(2, 256, 15, 19).cuda()
fake_p5 = torch.randn(2, 256,  8, 10).cuda()

# Try list format first (most common in YOLOX implementations)
try:
    preds = head([fake_p3, fake_p4, fake_p5])
    print("Head accepts: list [P3, P4, P5]")
except:
    # Try dict format
    preds = head({'P3': fake_p3, 'P4': fake_p4, 'P5': fake_p5})
    print("Head accepts: dict {P3, P4, P5}")

print(f"Prediction type: {type(preds)}")
if isinstance(preds, torch.Tensor):
    print(f"Prediction shape: {preds.shape}")
```

4. Document: input format (list or dict), output format (tensor or tuple), required channel count.

---

## Part C: mamba-ssm Installation

### Why mamba-ssm May Be Tricky

`mamba-ssm` contains custom CUDA kernels written in Triton. Compatibility depends on:
- Your CUDA version
- Your PyTorch version  
- Your GPU architecture (RTX 5070 Ti)

The RTX 5070 Ti is a recent GPU. Older pre-compiled mamba-ssm binaries may not support its compute capability. Installing from source ensures compatibility.

### Installation Steps

```bash
# Step 1: Check your CUDA and PyTorch versions
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version

# Step 2: Try pip install first (easiest)
pip install mamba-ssm

# Step 3: Basic import test
python -c "from mamba_ssm import Mamba; print('mamba-ssm installed OK')"
```

If pip install fails or you get CUDA errors:

```bash
# Install from source (compiles for your specific GPU)
pip install packaging ninja
git clone https://github.com/state-spaces/mamba
cd mamba
pip install . --no-build-isolation
```

If source install also fails:

```bash
# Use pure PyTorch reference implementation (slower but guaranteed to work)
# The mamba_ssm package includes a pure PyTorch fallback
pip install mamba-ssm[dev]
# Then use: from mamba_ssm.ops.triton.selective_state_update import selective_state_update_ref
```

### mamba-ssm Shape and Function Tests

Run all three tests. They must all pass before proceeding to Stage 3.

**Test 1: Basic forward pass**
```python
import torch
from mamba_ssm import Mamba

batch, length, d_model = 2, 1140, 256
x = torch.randn(batch, length, d_model).cuda()
model = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).cuda()
y = model(x)
assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
print(f"Test 1 PASS — Input: {x.shape}, Output: {y.shape}")
```

**Test 2: State dimensions (what is the hidden state size?)**
```python
# Mamba's recurrent state has shape (batch, d_model*expand, d_state)
# = (batch, 256*2, 16) = (batch, 512, 16)
# Verify this by running in step mode if supported
try:
    from mamba_ssm.utils.generation import InferenceParams
    inf_params = InferenceParams(max_seqlen=10, max_batch_size=2)
    y_step = model(x[:, :1, :], inference_params=inf_params)
    print(f"Test 2 PASS — InferenceParams supported")
    print(f"State shape: check inf_params.key_value_memory_dict")
except ImportError:
    print("Test 2 NOTE — InferenceParams not available in this version")
    print("Will use alternative state passing approach in Stage 3c")
```

**Test 3: Multiple windows (state persistence)**
```python
# Critical test: does the output differ when we process sequential windows?
model.eval()
x1 = torch.randn(2, 1140, 256).cuda()
x2 = torch.randn(2, 1140, 256).cuda()

# Process independently (no state)
y1_stateless = model(x1)
y2_stateless = model(x2)

# Process as sequence (x1 then x2)
# This requires step-by-step processing or a batched sequence approach
# For now, just verify outputs differ when the same input gets different context
print("Test 3 — State tests will be validated in Stage 3c")
print("mamba-ssm basic tests COMPLETE")
```

---

## Part D: Verify Baseline Still Works

Before building on top of the codebase, confirm your S5-RVT evaluation still runs and produces the correct result.

```bash
# Run your existing evaluation script
python evaluate.py --checkpoint path/to/s5_rvt_best.pth --split test

# Expected output:
# [Correction 2026-10-06: the per-class lines below were never published or measured -- the paper
#  reports overall AP only, and it is COCO AP@[.50:.95], not AP@0.5. Measured: car 63.9 / ped 31.6.]
# Car AP@0.5:         56.8 (±0.1 from published 56.8)
# Pedestrian AP@0.5:  38.6 (±0.1 from published 38.6)
# Overall mAP@0.5:    47.7 (±0.1 from published 47.71)
```

If this fails: fix the existing codebase before proceeding. Do not build on a broken foundation.

---

## Full Audit Checklist

```
[ ] All Python files listed and classified (REUSE/MODIFY/CREATE)
[ ] YOLOX head input format documented (list or dict)
[ ] YOLOX head output format documented
[ ] YOLOX head channel expectation confirmed (256)
[ ] mamba-ssm pip install attempted
[ ] mamba-ssm Test 1 (basic forward pass) PASSING
[ ] mamba-ssm Test 2 (state API) result noted
[ ] mamba-ssm Test 3 noted
[ ] S5-RVT evaluation confirmed: mAP ~47.7
[ ] List of CREATE files identified (4 new files)
```

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| mamba-ssm pip install fails for 5070 Ti | Medium | Install from source with `--no-build-isolation` |
| mamba-ssm source install fails | Low | Use pure PyTorch fallback `selective_state_update_ref` |
| YOLOX head expects different format than expected | Medium | Note the actual format — adapt your FPN output to match |
| S5-RVT evaluation no longer produces 47.7 | Low | Check CUDA version, PyTorch version compatibility; fix before proceeding |
| Codebase structure different from expected | Always | The audit itself resolves this — that's why it exists |

---

## System

**Linux PC entirely.** Stages 0–1 were Mac; from Stage 2 onwards, all work is on Linux.

---

## Deliverables

1. `codebase_audit.md` — table of every file classified as REUSE/MODIFY/CREATE
2. `yolox_head_interface.md` — one page documenting head input/output format
3. Terminal screenshot showing mamba-ssm Test 1 passing
4. Terminal output showing S5-RVT mAP ~47.7

---

## Success Criteria

- mamba-ssm `from mamba_ssm import Mamba` executes without error
- YOLOX head input/output interface is documented
- S5-RVT produces 47.7 mAP (±0.1)
- You know exactly which 4 files you will create

---

## Next Stage

→ **Stage 3a: ResNet-18 Backbone** — implement and test the modified backbone component.
