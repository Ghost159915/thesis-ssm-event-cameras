# Stage 3a — ResNet-18 Event Backbone
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Build and test the modified ResNet-18 backbone as a standalone, independently testable module. This component handles spatial feature extraction from event camera voxel grids.

**System:** Linux PC, RTX 5070 Ti  
**File to create:** `models/backbone/resnet18_event.py`  
**Estimated time:** 1–2 days  
**Prerequisite:** Stage 2 complete (codebase audited, mamba-ssm installed)

---

## Goal

A `ResNet18EventBackbone` class that:
- Takes voxel grid input: `(B, 10, 240, 304)`
- Returns multi-scale features: `{C2: (B,128,30,38), C3: (B,256,15,19), C4: (B,512,8,10)}`
- Loads ImageNet pretrained weights with correct adaptation for 10-channel input
- Passes shape test, gradient test, and pretrained weight verification

---

## Why This Component

### The Spatial Understanding Problem

Event cameras produce asynchronous streams of brightness change events. Each pixel fires independently when its log-luminance changes. This raw stream contains rich temporal information but no explicit spatial structure — nearby pixels do not fire together unless they are part of the same moving edge.

Your voxel grid representation (`Zhu_2019_VoxelGrids_UnsupOpticalFlow_CVPR`) discretises this stream into B=10 temporal bins, creating a pseudo-image where each "channel" is a time slice. Within each time slice, events cluster around moving edges, texture boundaries, and moving objects.

ResNet-18 is designed to extract spatial features from image data by applying learned convolutional filters — edge detectors, corner detectors, texture analysers — hierarchically. This spatial inductive bias (local patterns are meaningful) is well-suited to event data because objects, edges, and shapes are spatially contiguous in the voxel grid representation.

### Why ResNet-18 Specifically

**`Floreano_2015_FutureSmallDrones_Nature`** establishes that sub-250g UAVs face severe SWaP constraints. ResNet-18 (~11.7M parameters, ~1.8 GFLOPs) is the lightest standard ResNet variant that still produces competitive detection features. Lighter alternatives (ResNet-8 used by `Niculescu_2022_NanoDroneDNN_Deployment_JETCAS`) are too light for competitive mAP on Gen1.

**`Zheng_2023_DeepLearningEventVision_Survey_arXiv`** reviews transfer learning for event cameras and confirms that ImageNet pretraining helps for event-based tasks. Low-level features (edge, gradient, corner detectors) learned from RGB images transfer to event data because the underlying visual structure is similar — both represent edges and contrast changes.

**The scientific argument:** By replacing the Vision Transformer (ViT-Base, ~86M params, O(L²) complexity) with ResNet-18 (11.7M params, O(L) complexity) while keeping the temporal module and detection head identical, any mAP difference is attributable entirely to the choice of spatial feature extractor. This is the controlled experiment at the heart of your thesis.

---

## Architecture Deep Dive

### ResNet-18 Structure

ResNet-18 consists of a stem (Conv1 + MaxPool) followed by 4 residual stages (Layers 1–4). Each residual block uses skip connections to allow gradients to flow directly through the network, enabling training of deep networks (`He et al. 2016` — fundamental paper, not in your collection but well known).

```
Layer    Blocks    Stride    Output Channels    Output Shape (for 240×304 input)
─────────────────────────────────────────────────────────────────────────────
Stem     -         2×2       64                 (B, 64, 60, 76)
Layer1   2 blocks  1         64                 (B, 64, 60, 76)   ← C1 (not used in FPN)
Layer2   2 blocks  2         128                (B, 128, 30, 38)  ← C2
Layer3   2 blocks  2         256                (B, 256, 15, 19)  ← C3
Layer4   2 blocks  2         512                (B, 512,  8, 10)  ← C4
```

### Why Not Extract C1 (Layer1)?

C1 at (B, 64, 60, 76) would be P2 in FPN terminology. This is a very high-resolution feature map — 60×76 = 4560 spatial locations. Adding this to the FPN significantly increases memory and computation cost. More importantly, Layer1 features are very low-level (simple edges, raw texture) with limited semantic content. Standard detection benchmarks on automotive data (Gen1, `Perot_2020_Gen1`) show that P2 features rarely improve mAP significantly for the object sizes in the dataset. Start without C1/P2 and add it only if mAP is disappointing.

---

## The 10-Channel First Conv Problem

### The Problem

Standard ResNet-18's first conv: `Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)`

ImageNet pretrained weights for this layer have shape: `(64, 3, 7, 7)` — 64 filters, each operating on 3 RGB channels.

Your input has 10 channels. You need: `Conv2d(in_channels=10, out_channels=64, kernel_size=7, stride=2, padding=3)` with weights of shape `(64, 10, 7, 7)`.

### Option 1: Random Initialisation (Baseline Approach)
Don't use pretrained weights for the first conv at all. Simple, but slower convergence — the network must learn low-level feature detectors from scratch.

**Why rejected:** Training takes longer and may not fully converge within 100 epochs. The event camera community has shown that pretraining helps.

### Option 2: Average Projection (Recommended)
Average the 3 pretrained channels to get a single template filter, then tile it across 10 channels with magnitude rescaling:

```python
pretrained_weight: (64, 3, 7, 7)
step 1 → average over channel dim → (64, 1, 7, 7)
step 2 → repeat 10 times → (64, 10, 7, 7)
step 3 → scale by 3/10 → preserves activation magnitude
```

**Why this works:** The 3 RGB channels in the pretrained first conv all detect similar things (edges, gradients) — they are not wildly different from each other. The average is a reasonable representation of "what this filter responds to." Scaling by 3/10 ensures the activation magnitude at the output is similar to the original 3-channel case (if all 10 input channels had the same statistics as 1 RGB channel, the unscaled version would produce 10/3× the original activations).

**Paper supporting transfer:** `Zheng_2023_DeepLearningEventVision_Survey_arXiv` and general practice in the event camera community.

### Option 3: Replicate RGB Three Times Then Pad
Copy the pretrained weights directly for 3 channels, then zero-initialise the remaining 7 channels.

**Issue:** The zero-initialised channels contribute no gradient initially. The model must slowly learn to use them. This is less efficient than Option 2.

**Decision: Use Option 2 (Average Projection).**

---

## Implementation Specification

### Class: `ResNet18EventBackbone`

**File:** `models/backbone/resnet18_event.py`

**`__init__(self, pretrained=True, in_channels=10)`**
1. Load standard ResNet-18 from torchvision with `weights='IMAGENET1K_V1'`
2. Save the pretrained first conv weights before modifying
3. Create new first conv: `Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)`
4. Apply average projection initialisation (see above)
5. Assign new conv to `self.backbone.conv1`
6. Remove the fully-connected classification head (`self.backbone.fc` — not needed)
7. Store references to `self.backbone.layer2`, `layer3`, `layer4` for feature extraction

**`forward(self, x) → dict`**
1. Pass x through stem (conv1, bn1, relu, maxpool) → after_stem
2. Pass through layer1 → layer1_out (not returned, just passed through)
3. Pass through layer2 → C2 (returned)
4. Pass through layer3 → C3 (returned)
5. Pass through layer4 → C4 (returned)
6. Return: `{'C2': C2, 'C3': C3, 'C4': C4}`

**`set_backbone_lr_scale(self, scale)` (optional helper)**
Returns param groups with scaled learning rate for the backbone, allowing differential learning rate in the optimiser.

---

## Shape Test (Must Pass Before Moving to Stage 3b)

```python
import torch
from models.backbone.resnet18_event import ResNet18EventBackbone

# Create model
backbone = ResNet18EventBackbone(pretrained=True, in_channels=10)
backbone = backbone.cuda()
backbone.eval()

# Test input: batch=2, 10 event bins, Gen1 resolution
x = torch.randn(2, 10, 240, 304).cuda()

with torch.no_grad():
    features = backbone(x)

# Shape assertions
assert features['C2'].shape == (2, 128, 30, 38), f"C2 shape wrong: {features['C2'].shape}"
assert features['C3'].shape == (2, 256, 15, 19), f"C3 shape wrong: {features['C3'].shape}"
assert features['C4'].shape == (2, 512,  8, 10), f"C4 shape wrong: {features['C4'].shape}"

print("Shape test: PASS")
print(f"C2: {features['C2'].shape}")
print(f"C3: {features['C3'].shape}")
print(f"C4: {features['C4'].shape}")
```

## Gradient Test (Must Pass)

```python
backbone.train()
x = torch.randn(2, 10, 240, 304, requires_grad=True).cuda()
features = backbone(x)
loss = sum(f.sum() for f in features.values())
loss.backward()

# Check gradients flow to all parts
assert backbone.layer2[0].conv1.weight.grad is not None, "No gradient in Layer2"
assert backbone.layer3[0].conv1.weight.grad is not None, "No gradient in Layer3"
assert backbone.layer4[0].conv1.weight.grad is not None, "No gradient in Layer4"
# Check first conv (our modified layer)
assert backbone.backbone.conv1.weight.grad is not None, "No gradient in first conv"

print("Gradient test: PASS")
```

## Pretrained Weight Verification

```python
# Verify that the pretrained weights were loaded correctly
# The first conv weights should NOT be zero or random-looking
weight = backbone.backbone.conv1.weight.data  # (64, 10, 7, 7)
print(f"First conv weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}")
# Expected: std ~ 0.01-0.1 (consistent with initialised convolutional filters)
# If std is 1.0+ : weights are random (initialisation failed)
# If std is very close to 0 : averaging produced near-zero weights (check the code)

# Deeper layers should have ImageNet pretrained values
layer2_weight = backbone.layer2[0].conv1.weight.data
print(f"Layer2 weight stats: mean={layer2_weight.mean():.4f}, std={layer2_weight.std():.4f}")
# This std should be in the range 0.01-0.05 (ImageNet pretrained, well-scaled)
```

---

## Parameter Count

```python
backbone_params = sum(p.numel() for p in backbone.parameters()) / 1e6
print(f"ResNet-18 backbone parameters: {backbone_params:.1f}M")
# Expected: ~11.2M (slightly less than full ResNet-18's 11.7M because we removed the FC head)
```

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Average projection produces worse results than random | Can compare in Stage 6 (short run) with both initialisations. Average projection is the default. |
| Layer4 output shape wrong due to odd input dimensions | Verify: 15 / 2 = 7.5 → floor = 7, but with standard ResNet-18 padding, Layer4 may produce (8, 10) for input (15, 19). Verify with the shape test. |
| ImageNet weights not loading (network error) | Pre-download weights: `torch.hub.load('pytorch/vision', 'resnet18', weights='IMAGENET1K_V1')` on a machine with internet before running on Katana/offline |
| Gradient not flowing through first conv | Check that `requires_grad=True` is set on the new first conv's weight parameter |

---

## Common Mistakes to Avoid

1. **Forgetting to remove the FC layer:** `model.fc` is the classification head. It expects a 512-d vector, not your spatial feature maps. Remove it or never call it.

2. **Shape mismatch at Layer4:** For input (B, 10, 240, 304), Layer4 produces (B, 512, 8, 10) approximately. The exact values depend on padding — run the shape test to confirm.

3. **Wrong scale factor in average projection:** The scale factor must be `3 / new_channels` not `1 / new_channels`. The 3 comes from the original number of channels in the pretrained weights.

4. **Forgetting `.to(device)` after weight assignment:** If you reassign `model.backbone.conv1`, call `.to('cuda')` again to ensure the new layer is on GPU.

---

## Deliverable

`models/backbone/resnet18_event.py` — passing all three tests (shape, gradient, pretrained weight verification).

## Success Criteria

All three tests print "PASS." Parameter count is ~11.2–11.7M. Weight std in first conv is in a reasonable range (0.01–0.1).

---

## Next Stage

→ **Stage 3b: Feature Pyramid Network** — build the FPN that fuses ResNet-18's multi-scale outputs.
