# Stage 3b — Feature Pyramid Network (FPN)
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Build and test the Feature Pyramid Network that fuses ResNet-18's multi-scale feature maps into a pyramid of uniform-channel feature maps for the detection head.

**System:** Linux PC, RTX 5070 Ti  
**File to create:** `models/fpn/event_fpn.py`  
**Estimated time:** 1 day  
**Prerequisite:** Stage 3a complete (ResNet-18 backbone passing all tests)

---

## Goal

An `EventFPN` class that:
- Takes multi-scale backbone features: `{C2, C3, C4}`
- Returns pyramid features: `{P3: (B,256,30,38), P4: (B,256,15,19), P5: (B,256,8,10)}`
- Passes shape test and gradient test

---

## Why This Component

### The Multi-Scale Detection Problem

Objects in the Gen1 dataset (`Perot_2020_Gen1_1Mpx_Detection_NeurIPS`) appear at dramatically different scales:
- A pedestrian 50 metres away: ~5×10 pixels at Gen1's resolution
- A car 2 metres away: ~100×50 pixels

A single-scale detector cannot handle this range well. Deep layers of ResNet-18 (C4, stride 32) produce semantically rich features but at very coarse resolution (8×10 pixels for Gen1) — small pedestrians become one or two pixels and are effectively invisible. Shallow layers (C2, stride 8) have high resolution (30×38 pixels) but low semantic content (they only understand edges and textures, not object-level concepts like "this is a car").

The Feature Pyramid Network (`Lin et al. 2017` — standard detection architecture) solves this by combining the spatial precision of shallow layers with the semantic richness of deep layers through a top-down pathway.

### Why FPN Specifically

**`Gehrig_2023_RVT_EventDetection_CVPR`** (RVT, your primary baseline) uses a multi-scale FPN architecture and demonstrates that multi-scale detection is essential for event-based automotive detection. Their ablation (Table 2) shows multi-scale detection consistently outperforms single-scale across all tested architectures.

**`Yang_2025_SMamba_EventDetection_AAAI`** (SMamba, current SOTA) also uses multi-scale FPN, confirming it is the consensus architecture for this task.

**`Perot_2020_Gen1_1Mpx_Detection_NeurIPS`** (Gen1 dataset paper) first demonstrated competitive event-based detection using learned multi-scale event representations with anchor-free detection heads.

---

## How FPN Works (Technical Detail)

### Bottom-Up Pathway (Free — Already Done)

This is just your ResNet-18 backbone. As the input passes through successive stages, spatial resolution decreases and semantic content increases. You get C2, C3, C4 at strides 8, 16, 32 respectively.

### Lateral Connections

Each backbone scale is passed through a 1×1 convolution to project to a uniform 256 channels:

```
lateral_C2: Conv2d(128, 256, kernel_size=1)  — C2 has 128 channels
lateral_C3: Conv2d(256, 256, kernel_size=1)  — C3 has 256 channels (no-op in terms of info)
lateral_C4: Conv2d(512, 256, kernel_size=1)  — C4 has 512 channels
```

The 1×1 convolution does not change spatial size, only channel count.

### Top-Down Pathway

Start from the deepest scale (C4, most semantic) and propagate information upward:

```
P5 = lateral(C4)                    → (B, 256, 8, 10)
P4 = lateral(C3) + upsample(P5)     → (B, 256, 15, 19)
P3 = lateral(C2) + upsample(P4)     → (B, 256, 30, 38)
```

Upsampling brings P5's semantic information to P4's spatial locations. Adding the lateral connection mixes semantic understanding from P5 with fine-grained spatial information from C3.

### Output Convolutions (Anti-Aliasing)

After each top-down addition, a 3×3 convolution smooths the feature map:
```
P3 = conv3x3(P3_merged)
P4 = conv3x3(P4_merged)
P5 = conv3x3(P5_merged)
```

This is standard in FPN implementations and reduces aliasing introduced by bilinear upsampling.

---

## The Critical Upsampling Issue for Gen1

### The Odd-Dimension Problem

For Gen1 at 240×304 input:
- C4 at stride 32: (B, 512, **8**, **10**)
- C3 at stride 16: (B, 256, **15**, **19**)

If you naively upsample P5 (8×10) by ×2, you get (16×20). But C3 is (15×19). They don't match — element-wise addition is impossible.

**Wrong approach:**
```python
upsampled = F.interpolate(P5, scale_factor=2)  # → (16, 20) ≠ (15, 19) CRASH
```

**Correct approach:**
```python
upsampled = F.interpolate(P5, size=(C3.shape[-2], C3.shape[-1]), mode='bilinear', align_corners=False)
# → matches C3's exact spatial dimensions (15, 19)
```

This is the single most common FPN implementation bug. Always use `size=` not `scale_factor=` when your spatial dimensions may be odd.

**`torchvision.ops.FeaturePyramidNetwork` handles this automatically** — it uses size-matching interpolation internally. Using it is the safest approach.

---

## Implementation Options

### Option A: Use torchvision.ops.FeaturePyramidNetwork (Recommended)

```python
from torchvision.ops import FeaturePyramidNetwork

fpn = FeaturePyramidNetwork(
    in_channels_list=[128, 256, 512],  # C2, C3, C4 channels
    out_channels=256
)
# Input: OrderedDict with keys '0', '1', '2'
# Output: OrderedDict with same keys, all 256 channels
```

**Pros:** Handles the odd-dimension issue automatically. Well-tested. Minimal code.  
**Cons:** Requires OrderedDict input format. Minor wrapping needed to convert `{C2, C3, C4}` to OrderedDict and back.

### Option B: Manual Implementation

Implement the lateral convs, top-down pathway, and output convs yourself. More flexible but requires careful handling of the odd-dimension issue.

**Decision: Use torchvision.ops.FeaturePyramidNetwork.** This is a well-tested, standard component and allows you to focus your effort on the novel parts (Mamba temporal module). Using PyTorch's built-in FPN is not a shortcut — it is good engineering practice.

---

## Implementation Specification

### Class: `EventFPN`

**File:** `models/fpn/event_fpn.py`

**`__init__(self, in_channels_list=[128, 256, 512], out_channels=256)`**
1. Instantiate `torchvision.ops.FeaturePyramidNetwork` with in_channels_list and out_channels
2. Store as `self.fpn`

**`forward(self, features: dict) → dict`**
1. Convert input dict `{'C2': tensor, 'C3': tensor, 'C4': tensor}` to `OrderedDict` with keys `'0'`, `'1'`, `'2'`
2. Pass through `self.fpn`
3. Convert output OrderedDict to `{'P3': tensor, 'P4': tensor, 'P5': tensor}`
4. Return

**Note on key naming:** torchvision's FPN expects an `OrderedDict` where keys are strings. The FPN assigns output keys to match input keys. Map: `'C2'→'0'`, `'C3'→'1'`, `'C4'→'2'`, then remap outputs `'0'→'P3'`, `'1'→'P4'`, `'2'→'P5'`.

---

## Shape Test (Must Pass)

```python
import torch
from models.backbone.resnet18_event import ResNet18EventBackbone
from models.fpn.event_fpn import EventFPN

# Create components
backbone = ResNet18EventBackbone(pretrained=False, in_channels=10).cuda()
fpn = EventFPN(in_channels_list=[128, 256, 512], out_channels=256).cuda()

# Test input
x = torch.randn(2, 10, 240, 304).cuda()

# Forward pass through backbone then FPN
backbone_features = backbone(x)
fpn_features = fpn(backbone_features)

# Shape assertions
assert fpn_features['P3'].shape == (2, 256, 30, 38), f"P3 wrong: {fpn_features['P3'].shape}"
assert fpn_features['P4'].shape == (2, 256, 15, 19), f"P4 wrong: {fpn_features['P4'].shape}"
assert fpn_features['P5'].shape == (2, 256,  8, 10), f"P5 wrong: {fpn_features['P5'].shape}"

print("FPN shape test: PASS")
```

## Gradient Test (Must Pass)

```python
backbone.train()
fpn.train()
x = torch.randn(2, 10, 240, 304).cuda()

backbone_features = backbone(x)
fpn_features = fpn(backbone_features)

loss = sum(f.sum() for f in fpn_features.values())
loss.backward()

# FPN lateral convs should receive gradients
for name, param in fpn.named_parameters():
    assert param.grad is not None, f"No gradient in FPN parameter: {name}"

print("FPN gradient test: PASS")
```

## Semantic Sanity Check (Optional but Useful)

Verify that deeper FPN levels (P5) contain different information than shallow levels (P3):

```python
backbone.eval()
fpn.eval()
with torch.no_grad():
    x = torch.randn(2, 10, 240, 304).cuda()
    feats = fpn(backbone(x))
    p3_norm = feats['P3'].norm(dim=1).mean()
    p5_norm = feats['P5'].norm(dim=1).mean()
    print(f"P3 activation norm: {p3_norm:.2f}")
    print(f"P5 activation norm: {p5_norm:.2f}")
# Both should be non-zero. Very similar values are expected with random weights.
# After training, P5 will have higher magnitude (more semantic content is concentrated there).
```

---

## FPN Parameter Count

```python
fpn_params = sum(p.numel() for p in fpn.parameters()) / 1e6
print(f"FPN parameters: {fpn_params:.2f}M")
# Expected: ~0.4-0.6M
# Lateral convs: 3 × (in_ch × 256) ≈ (128×256 + 256×256 + 512×256) = ~229K weights (plus biases)
# Output convs: 3 × (256 × 256 × 3 × 3) ≈ 1.77M weights
# Total: ~2M approximately (torchvision's FPN includes output convs)
```

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Odd-dimension FPN mismatch crash | Use torchvision.ops.FeaturePyramidNetwork (handles automatically) |
| Wrong key format for torchvision FPN | Wrap input dict with OrderedDict, remap keys to '0','1','2' |
| P5 feature map too small (8×10=80 tokens) for meaningful detection | This is inherent to the dataset resolution. Include P5 for large objects. If it hurts mAP, ablate by removing it. |
| FPN not receiving gradients from detection head | Verify loss propagates through head → FPN → backbone in Stage 4 |

---

## Deliverable

`models/fpn/event_fpn.py` — passing shape and gradient tests.

## Success Criteria

Shape test and gradient test both print "PASS." FPN parameter count ~0.4–2M.

---

## Next Stage

→ **Stage 3c: Mamba Temporal Module** — the most novel and technically complex component.
