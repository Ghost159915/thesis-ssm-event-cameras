# Stage 1 — Architecture Blueprint
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Translate Stage 0's decisions into a precise, annotated data flow diagram with exact tensor shapes at every step. This is the engineering blueprint before construction begins. Every dimension mismatch that would take hours to debug in code is caught in 10 minutes on paper.

**System:** Mac (documentation only — draw diagram digitally or by hand and scan)  
**Estimated time:** 1 day  
**Prerequisite:** Stage 0 complete (all design decisions locked)

---

## Goal

A complete annotated diagram showing every component and the exact tensor shape flowing between them, calculated for the Gen1 dataset resolution (240×304 pixels, 10 event bins).

---

## Why This Stage Exists

### Contribution to Thesis B

Every thesis report includes an architecture figure. This stage produces that figure. Building it before coding ensures it is accurate (matches your actual implementation) rather than drawn retrospectively and potentially inconsistent with the code.

More practically: students who skip this stage spend days debugging dimension mismatches that would have been caught in 10 minutes on paper. The FPN top-down pathway is a particularly common source of off-by-one errors when spatial dimensions are odd numbers.

---

## The Gen1 Dataset Resolution

The Gen1 dataset (`Perot_2020_Gen1_1Mpx_Detection_NeurIPS`) uses a 240×304 pixel event camera (Prophesee Gen1 sensor). Your input voxel grid is:

```
(Batch, B_bins, H, W) = (B, 10, 240, 304)
```

All tensor shapes in this document use this resolution. Verify your actual data loader produces this exact shape in Stage 2.

---

## Component 1: Modified ResNet-18 Backbone

### Standard ResNet-18 Architecture

ResNet-18 has 4 residual stages (Layer1–Layer4), each progressively downsampling spatial resolution while increasing channel count.

### Exact Shape Calculation for Gen1 (240×304 input)

```
Input:         (B, 10, 240, 304)
               ↓
Conv1:         7×7 conv, 64 filters, stride=2, padding=3
               → (B, 64, 120, 152)
               ↓
MaxPool:       3×3, stride=2, padding=1
               → (B, 64, 60, 76)    ← C1 (stride-4 from input)
               ↓
Layer1:        2× residual blocks, stride=1 (no downsampling)
               → (B, 64, 60, 76)    ← same as C1
               ↓
Layer2:        2× residual blocks, stride=2
               → (B, 128, 30, 38)   ← C2 (stride-8 from input)
               ↓
Layer3:        2× residual blocks, stride=2
               → (B, 256, 15, 19)   ← C3 (stride-16 from input)
               ↓
Layer4:        2× residual blocks, stride=2
               → (B, 512, 8, 10)    ← C4 (stride-32 from input)
```

**Important note on "stride-8, 256 channels":** This phrasing in your Thesis A document was shorthand. At stride-8, ResNet-18 outputs 128 channels (Layer2). The 256 channels come from the FPN's lateral 1×1 convolutions, which project all scales to a uniform width. This is standard and correct — the FPN converts 128→256, 256→256, 512→256.

### First Conv Layer Modification

Standard ResNet-18: `Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)`  
Modified: `Conv2d(in_channels=10, out_channels=64, kernel_size=7, stride=2, padding=3)`

**Weight initialisation strategy (Average Projection):**
```
ImageNet pretrained first conv: shape (64, 3, 7, 7)
Step 1: Average over channel dimension → (64, 1, 7, 7)
Step 2: Repeat 10 times → (64, 10, 7, 7)
Step 3: Scale by 3/10 to preserve activation magnitude
```

This preserves the low-level edge detection features learned from ImageNet.

### Features Extracted

| Name | Shape | Stride from Input | Used by FPN? |
|---|---|---|---|
| C1 / Layer1 out | (B, 64, 60, 76) | 4 | No (too large) |
| C2 / Layer2 out | (B, 128, 30, 38) | 8 | Yes → P3 |
| C3 / Layer3 out | (B, 256, 15, 19) | 16 | Yes → P4 |
| C4 / Layer4 out | (B, 512, 8, 10) | 32 | Yes → P5 |

---

## Component 2: Feature Pyramid Network

### How FPN Works

FPN (`Lin et al. 2017`) creates a multi-scale feature pyramid using two pathways:

1. **Bottom-up pathway:** The backbone (ResNet-18) naturally creates multi-scale features as you go deeper. Lower layers have high spatial resolution but low semantic content; deeper layers have low spatial resolution but high semantic content.

2. **Lateral connections:** 1×1 convolutions project each backbone scale to a uniform 256 channels.

3. **Top-down pathway:** Start from the deepest (most semantic) scale. Upsample 2×. Add to the lateral connection from the shallower scale. Repeat. This combines semantic information from deep layers with spatial precision from shallow layers.

### Exact FPN Shape Calculation

```
Backbone inputs to FPN:
  C2: (B, 128, 30, 38)
  C3: (B, 256, 15, 19)
  C4: (B, 512,  8, 10)

Step 1 — Lateral 1×1 convolutions:
  C2 → lateral_C2 → (B, 256, 30, 38)
  C3 → lateral_C3 → (B, 256, 15, 19)
  C4 → lateral_C4 → (B, 256,  8, 10)

Step 2 — Top-down pathway:
  P5 = lateral_C4                           → (B, 256,  8, 10)
  P4 = lateral_C3 + upsample(P5, scale=2)  → (B, 256, 15, 20) ← ISSUE!
```

### Critical Dimension Issue: Odd Spatial Sizes

Notice: upsampling P5 (8×10) by 2× gives (16×20), but C3 is (15×19). They don't match.

**Solution:** Use `align_corners=False` bilinear upsample and match the *output size* to the lateral connection's spatial dimensions, not scale by 2:
```python
torch.nn.functional.interpolate(
    x, size=(lateral.shape[-2], lateral.shape[-1]),
    mode='bilinear', align_corners=False
)
```

Or use `torchvision.ops.FeaturePyramidNetwork` — it handles this automatically.

### Corrected FPN Output Shapes

```
P5: (B, 256,  8, 10)   ← stride 32, largest objects
P4: (B, 256, 15, 19)   ← stride 16, medium objects
P3: (B, 256, 30, 38)   ← stride  8, small objects (pedestrians at distance)
```

### Final 3×3 Convolution

After top-down addition, each P-level passes through a 3×3 conv (same channels, same spatial size) to reduce aliasing from upsampling. This is standard FPN and your codebase likely already includes it.

---

## Component 3: Mamba Temporal Module

### Input and Output Contract

**Per FPN scale, independently:**
- Input: `(B, 256, H_scale, W_scale)` + optional `hidden_state`
- Output: `(B, 256, H_scale, W_scale)` + `new_hidden_state`

### Spatial Linearisation

Mamba operates on 1D sequences. You must flatten the 2D spatial feature map to a sequence of tokens:

```
For P3 (B, 256, 30, 38):
  Reshape: (B, 30×38, 256) = (B, 1140, 256)
  ← 1140 spatial tokens, each with 256-dimensional embedding

For P4 (B, 256, 15, 19):
  Reshape: (B, 15×19, 256) = (B, 285, 256)

For P5 (B, 256, 8, 10):
  Reshape: (B, 8×10, 256) = (B, 80, 256)
```

After Mamba processing: reshape back to (B, 256, H, W).

### Mamba Block Internal Shape

Each Mamba block (`Gu_2023_Mamba_SelectiveStateSpaces_arXiv`):
```
Input: (B, L, d_model)   where L = H*W, d_model = 256
Inner expansion: (B, L, d_model * expand) = (B, L, 512)
Output: (B, L, d_model) = (B, L, 256)   ← same as input
```

The hidden state per layer:
```
(B, d_model * expand, d_state) = (B, 512, 16)
```

With 4 layers per scale and 3 scales, total hidden state: list of 12 tensors.

### State Management Across Windows

This is the most important interface in the entire model:

```
Window t:   (x_t, h_{t-1}) → Mamba → (y_t, h_t)
Window t+1: (x_{t+1}, h_t.detach()) → Mamba → (y_{t+1}, h_{t+1})
```

`.detach()` is mandatory between windows. It preserves the state values (temporal memory is maintained) but disconnects the gradient computation graph (prevents backpropagating through the entire sequence history, which would be memory-prohibitive). See Stage 4 for implementation details.

### Mamba Applied Separately per FPN Scale

Three separate Mamba module instances: one for P3, one for P4, one for P5. This allows different temporal dynamics at different spatial scales — large nearby objects (P5) have different motion patterns than small distant objects (P3).

---

## Component 4: YOLOX Detection Head

### Input

Three feature maps: P3, P4, P5 (after Mamba temporal processing)

```
P3 temporal: (B, 256, 30, 38)
P4 temporal: (B, 256, 15, 19)
P5 temporal: (B, 256,  8, 10)
```

### YOLOX Head Structure

Each scale has a separate head instance with decoupled branches:

```
Per scale:
  Input: (B, 256, H_k, W_k)
  │
  ├── cls branch: 2× Conv → (B, num_classes, H_k, W_k)
  ├── reg branch: 2× Conv → (B, 4, H_k, W_k)  [x, y, w, h offsets]
  └── obj branch: 2× Conv → (B, 1, H_k, W_k)  [objectness score]
```

For Gen1 (2 classes: car, pedestrian):

```
From P3 (30, 38): cls (B,2,30,38)  reg (B,4,30,38)  obj (B,1,30,38)
From P4 (15, 19): cls (B,2,15,19)  reg (B,4,15,19)  obj (B,1,15,19)
From P5 ( 8, 10): cls (B,2, 8,10)  reg (B,4, 8,10)  obj (B,1, 8,10)
```

Total predictions per window: (30×38 + 15×19 + 8×10) = 1140 + 285 + 80 = **1505 candidate boxes**.

After NMS (non-maximum suppression), typically 5–50 final detections per window.

---

## Full Data Flow Summary (Annotated)

```
INPUT WINDOW
(B, 10, 240, 304)
     │
     ▼ [MODIFIED RESNET-18 — first conv: 10→64 channels, ImageNet pretrained]
     │
     ├── C2: (B, 128, 30, 38)  ← stride 8
     ├── C3: (B, 256, 15, 19)  ← stride 16
     └── C4: (B, 512,  8, 10)  ← stride 32
     │
     ▼ [FPN — lateral 1×1 convs, top-down pathway, 3×3 output convs]
     │
     ├── P3: (B, 256, 30, 38)
     ├── P4: (B, 256, 15, 19)
     └── P5: (B, 256,  8, 10)
     │
     ▼ [MAMBA TEMPORAL — applied per scale, carries hidden state across windows]
     │
     ├── P3_t: (B, 256, 30, 38)   + state_P3 → next window
     ├── P4_t: (B, 256, 15, 19)   + state_P4 → next window
     └── P5_t: (B, 256,  8, 10)   + state_P5 → next window
     │
     ▼ [YOLOX HEAD — decoupled cls/reg/obj branches per scale]
     │
     ├── P3 head: cls(B,2,30,38) + reg(B,4,30,38) + obj(B,1,30,38)
     ├── P4 head: cls(B,2,15,19) + reg(B,4,15,19) + obj(B,1,15,19)
     └── P5 head: cls(B,2, 8,10) + reg(B,4, 8,10) + obj(B,1, 8,10)
     │
     ▼
OUTPUT: Bounding boxes + class scores per detection window
```

---

## Parameter Count Estimation

| Component | Approx Parameters |
|---|---|
| ResNet-18 backbone (modified) | ~11.7M |
| FPN lateral + output convs | ~0.5M |
| Mamba module (4 blocks × 3 scales, d_model=256, d_state=16) | ~3–4M |
| YOLOX head (3 scales, 2 classes) | ~2–3M |
| **Total estimate** | **~18–21M** |

Compare: S5-RVT ViT-Base variant has ~18–22M parameters. You should land in the same ballpark — this makes the comparison fair.

---

## Step-by-Step Tasks

1. **Draw the architecture diagram** using the data flow summary above. Label every tensor with its exact shape. Hand-drawn and scanned is fine.

2. **Annotate the state boundary**: Draw a dashed vertical line between window t and window t+1. Show where `.detach()` is called on the hidden state.

3. **Annotate pretrained vs. random initialisation:**
   - Mark in green: ResNet-18 layers that load ImageNet weights
   - Mark in red: First conv (re-initialised), FPN lateral convs (random), Mamba module (random), YOLOX head (random)

4. **Write the interface contracts** (one line each):
   - `Backbone.forward(x:(B,10,240,304)) → {C2:(B,128,30,38), C3:(B,256,15,19), C4:(B,512,8,10)}`
   - `FPN.forward({C2,C3,C4}) → {P3:(B,256,30,38), P4:(B,256,15,19), P5:(B,256,8,10)}`
   - `MambaModule.forward(P3:(B,256,30,38), state) → (P3_out:(B,256,30,38), new_state)`
   - `YOLOXHead.forward({P3,P4,P5}) → predictions`

5. **Verify the odd-dimension FPN issue**: Calculate 15/2 = 7.5 (not integer). Confirm your FPN uses size-matching interpolation, not ×2 scaling.

6. **Count total candidate detections**: 30×38 + 15×19 + 8×10 = 1505. Note this.

7. **Save the diagram** as `architecture_blueprint.png` or `.pdf` in your Thesis_Plan folder.

---

## Deliverable

- Annotated architecture diagram (all shapes labelled)
- Interface contract table
- Saved to `Thesis_Plan/architecture_blueprint.pdf`

## Success Criteria

You can trace any tensor from input to output and state its exact shape at every step, including after FPN upsampling.

---

## Next Stage

→ **Stage 2: Codebase Audit** — map your existing S5-RVT code and verify mamba-ssm installation on your Linux PC.
