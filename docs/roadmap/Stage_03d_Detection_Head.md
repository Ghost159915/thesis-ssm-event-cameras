# Stage 3d — Detection Head Verification
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Verify that the existing YOLOX detection head from your S5-RVT codebase accepts the output format produced by your new FPN and Mamba modules. No new implementation — just verification and any necessary format adaptation.

**System:** Linux PC, RTX 5070 Ti  
**No new files to create** (detection head reused from existing codebase)  
**Estimated time:** Half a day  
**Prerequisite:** Stages 3a, 3b, 3c complete

---

## Goal

- YOLOX head input format fully documented
- Shape test passing: head accepts your FPN outputs without error
- Loss test passing: loss computation produces a scalar with valid gradients
- Any format mismatch resolved

---

## Why This Stage Exists

The detection head is the component where your new model's spatial/temporal features get converted into bounding box predictions. If the head expects features in a different format than your FPN produces, the entire pipeline fails at the last step. Finding this before Stage 4 integration saves significant debugging time.

More importantly: **you are reusing this head without modification** because it was part of the verified S5-RVT pipeline that produced 47.7 mAP. Reusing it guarantees that any mAP difference between S5-RVT and EventSSMDetector is caused by the backbone and temporal module — not the detection head. This is the cornerstone of your controlled experiment.

**Paper justification:** `Perot_2020_Gen1_1Mpx_Detection_NeurIPS` established the Gen1 detection benchmark. `Gehrig_2023_RVT_EventDetection_CVPR` applied YOLOX-style anchor-free detection to event data and produced the RVT baseline. `Zubic_2024_SSM_EventCameras_CVPR` used the same head for S5-RVT. Continuity of the detection head across all these baselines is why your comparison is valid.

---

## YOLOX Background

### What YOLOX Is

YOLOX (`Ge et al. 2021`) is an anchor-free single-stage object detector. "Anchor-free" means it does not use pre-defined anchor boxes at different scales — instead, it predicts bounding boxes directly as offsets from grid cell locations. This eliminates the tedious anchor tuning required by earlier YOLO variants.

### The Decoupled Head

YOLOX's key innovation is the **decoupled head**: rather than using one branch for all predictions (class + box + objectness), it uses three separate branches:
- **Classification branch:** "What class is this object?" → sigmoid output for each class
- **Regression branch:** "Where is the bounding box?" → 4 values (x, y, w, h)
- **Objectness branch:** "Is there an object here at all?" → binary sigmoid

In earlier detectors, these tasks conflict — optimising the classification loss can hurt regression accuracy. Decoupling allows each branch to specialise.

### Why YOLOX vs FCOS for Your Thesis

Both YOLOX and FCOS are anchor-free and achieve similar mAP on standard benchmarks. The reason to use YOLOX:

1. It is already implemented and verified in your codebase (47.7 mAP proven)
2. Using the same head as your baselines (RVT, S5-RVT) ensures your comparison is apples-to-apples
3. Implementing FCOS from scratch introduces a new component that needs its own debugging — for zero scientific benefit

Using FCOS would mean: any mAP difference between your model and S5-RVT could be due to the head, not the backbone. That breaks your controlled experiment. YOLOX keeps the experiment clean.

---

## Step 1: Find and Document the Head Interface

Open your YOLOX head file (probably `models/head/yolox_head.py` or similar in your S5-RVT codebase). Find the `forward()` method.

Document:

```
Head file location: [fill in]

forward() signature:
  def forward(self, ___):

Input format:
  [ ] List of tensors: [P3, P4, P5]
  [ ] Dict of tensors: {'P3': ..., 'P4': ..., 'P5': ...}
  [ ] Named arguments: forward(P3, P4, P5)
  [ ] Other: ___

Expected channel count per scale: ___
Expected spatial sizes: ___

Output format:
  [ ] Single tensor (all predictions concatenated): shape ___
  [ ] Tuple of tensors: ___
  [ ] Dict: ___

Loss function signature:
  How is the head output fed to the loss functions?
  Focal Loss input shape: ___
  GIoU Loss input shape: ___
```

---

## Step 2: Shape Test

Run this test, adapting the head import to your actual codebase path:

```python
import torch
import sys
sys.path.insert(0, '/path/to/your/s5_rvt_codebase')

# Adjust import to match your actual file structure
from models.head.yolox_head import YOLOXHead  # or whatever the class name is

# Instantiate the head
head = YOLOXHead(num_classes=2, in_channels=256).cuda()  # adjust args as needed

# Create fake FPN outputs matching your FPN's output shapes
fake_features = {
    'P3': torch.randn(2, 256, 30, 38).cuda(),
    'P4': torch.randn(2, 256, 15, 19).cuda(),
    'P5': torch.randn(2, 256,  8, 10).cuda(),
}

# Try calling the head — adapt the calling convention to what you found in Step 1
# Option A: list
try:
    preds = head([fake_features['P3'], fake_features['P4'], fake_features['P5']])
    print(f"Head accepts list. Output type: {type(preds)}, shape if tensor: {preds.shape if hasattr(preds, 'shape') else 'tuple/list'}")
except Exception as e:
    print(f"List calling failed: {e}")

# Option B: dict  
try:
    preds = head(fake_features)
    print(f"Head accepts dict. Output type: {type(preds)}")
except Exception as e:
    print(f"Dict calling failed: {e}")
```

---

## Step 3: Loss Computation Test

After confirming the head runs, verify the loss functions work:

```python
# Create fake ground truth labels (adjust format to match your loss functions)
# This mimics what the training loop does
fake_gt = ...  # from your existing training code — find an example batch

# Compute loss
loss_cls = focal_loss(preds, fake_gt)
loss_box = giou_loss(preds, fake_gt)
total_loss = loss_cls + loss_box

print(f"Total loss: {total_loss.item():.4f}")
assert not torch.isnan(total_loss), "Loss is NaN!"
assert total_loss.item() > 0, "Loss is zero or negative — something wrong"

# Verify gradients
total_loss.backward()
print("Loss backward: PASS")
```

---

## Step 4: Resolve Any Format Mismatches

If the head expects a list but your FPN returns a dict, or expects different channel counts, you have two options:

**Option A: Adapt your FPN output (recommended)**
In `event_fpn.py`, change the return format to match what the head expects. E.g., if head expects a list:
```python
# In EventFPN.forward():
return [fpn_out['P3'], fpn_out['P4'], fpn_out['P5']]  # list instead of dict
```

**Option B: Adapt the head**
If the change to the FPN is complex, add a thin wrapper around the head that converts your dict to the expected format. Do NOT modify the head internals — only a wrapper that translates formats.

---

## What "Anchor-Free" Means for Your Evaluation

Because YOLOX is anchor-free, predictions are generated at every spatial location in P3, P4, P5. For Gen1 (2 classes):
- P3 (30×38): 1140 candidate boxes for small objects
- P4 (15×19): 285 candidate boxes for medium objects
- P5 (8×10): 80 candidate boxes for large objects
- **Total:** 1505 candidates per window

After applying objectness thresholding and NMS (non-maximum suppression), typically 5–50 boxes remain.

The `mAP@0.5` metric evaluates predictions against ground truth with IoU threshold 0.5. Your evaluation pipeline (reused from S5-RVT, following `Perot_2020_Gen1`) already computes this correctly.

---

## Checklist

```
[ ] YOLOX head file located
[ ] forward() signature documented
[ ] Input format confirmed (list or dict)
[ ] Channel expectation confirmed (should be 256)
[ ] Output format documented
[ ] Shape test: head runs without error on fake FPN outputs
[ ] Loss test: loss computes a scalar, backward() succeeds
[ ] Any format mismatch resolved (FPN output adapted if needed)
```

---

## Deliverable

- `yolox_head_interface.md` — documented head interface (created in Stage 2, now updated with complete info)
- All tests passing

## Success Criteria

Head accepts your FPN format without error. Loss produces a non-NaN scalar. Backward pass succeeds.

---

## Next Stage

→ **Stage 4: Full Model Integration** — wire backbone + FPN + Mamba + head into a single `EventSSMDetector` class.
