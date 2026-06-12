# Stage 0 — Design Lock-In
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

> **RECONCILED (2026-06-12)** to the as-built design + `PLAN_FIXES_FOR_CLAUDE.md`. Key corrections below:
> input = **20-channel** stacked histogram (not 10-ch voxel grid, ISSUE-06); baseline backbone = **MaxViT-style
> attention, ≈18M total** (not "ViT-Base 86M", ISSUE-03); losses = **BCE + IoU `1−iou²` + SimOTA** (not Focal/GIoU,
> Stage-2 audit); temporal Mamba = **interleaved per backbone stage, over time** (not per-FPN-scale over space,
> ISSUE-01); metric = **COCO mAP (IoU 0.50:0.95)** (ISSUE-04). The as-built temporal design lives in Stage 3c / Stage 4.

## Overview

Before writing a single line of code, every architectural decision must be committed to in writing. This stage has **no code output**. Its sole deliverable is a one-page written specification that resolves every open question in the architecture. The cost of changing an architectural decision mid-implementation is enormous — potentially a full rewrite. The cost of deciding now is zero.

**System:** Mac (planning and documentation only)  
**Estimated time:** 1–2 days  
**Prerequisite:** None — this is the starting point

---

## Goal

Produce a signed-off design specification confirming: input representation, backbone, temporal module, detection head, loss functions, and training protocol. Every subsequent stage references this document as ground truth.

---

## Why This Stage Exists

### Contribution to Thesis B

The Thesis B report requires you to describe and justify every design choice in your methodology. Stage 0 is essentially writing the Methods section of your thesis *before* you implement it. When you sit down to write the Thesis B report, you will not need to reconstruct reasoning from memory — it is already written here.

It also aligns with your supervisor's expectation that you understand *why* you are building what you are building, not just *that* you are building it.

---

## Decision 1: Input Representation — Stacked 2-Polarity Histogram (10 bins → 20 channels)

> Reconciled: the locked representation matches the **baseline pipeline exactly** — a stacked histogram with
> 10 temporal bins × 2 polarities = **20 channels** (`stacked_histogram_dt=50_nbins=10`), *not* a 10-channel voxel
> grid. An identical representation to S5-RVT is required for the controlled comparison (errata ISSUE-06). Like a
> voxel grid it preserves temporal ordering across the 10 bins, but it is the baseline's exact tensor.

### Decision
Stacked 2-polarity histogram, B=10 temporal bins → **20 channels**. Raw sensor shape per window **(20, 240, 304)**
for Gen1; the pipeline zero-pads to **(20, 256, 320)** before the backbone.

### Justification from Your Papers

**Zhu et al. 2019** (`Zhu_2019_VoxelGrids_UnsupOpticalFlow_CVPR`) introduced voxel grids and proved they preserve temporal ordering better than event frames, which collapse the entire temporal dimension into a single snapshot. Events are distributed across B temporal bins by normalising their timestamps within the accumulation window.

**Gehrig et al. 2019** (`Gehrig_2019_EndToEnd_Representations_ICCV`) showed through comparative study that representations retaining temporal localisation — like voxel grids — consistently outperform simpler accumulation methods on downstream detection tasks. Their learned Event Spike Tensors (ESTs) are even better, but at higher computational cost.

**Gallego et al. 2022** (`Gallego_2022_EventVision_Survey_TPAMI`) is the canonical comprehensive survey. Confirms voxel grids are the dominant representation for detection tasks in the literature, appearing in every SOTA detection paper including RVT and S5-RVT.

### Why Not Alternatives

| Representation | Why Rejected |
|---|---|
| Event frame (histogram) | Collapses all temporal information. Loses the asynchronous advantage of event cameras entirely. Equivalent to treating an event camera like a bad frame camera. |
| Time surface (`Lagorce_2017_HOTS_TimeSurfaces_TPAMI`) | Encodes recency per pixel. Good for optical flow, but collapses the multi-event temporal structure needed for detection. Better for per-pixel regression tasks. |
| Event Spike Tensors (ESTs, `Gehrig_2019`) | Higher temporal fidelity via learned kernels but introduce additional parameters and training complexity. Unjustifiable for thesis timescale. |
| Raw event stream (graph, `Schaefer_2022_AEGNN_AsyncGraphNN_CVPR`) | Preserves asynchronous nature but requires graph neural network processing. Incompatible with ResNet-18 backbone. |
| Hybrid multi-channel (`Perot_2020`) | Promising but not validated on Gen1 with the architecture you are using. Introduces risk. |

### Why 10 Bins Specifically

B=10 is established in the S5-RVT baseline (`Zubic_2024_SSM_EventCameras_CVPR`) and the original RVT (`Gehrig_2023_RVT_EventDetection_CVPR`). Consistent with prior work, which allows direct comparison without introducing a confound.

---

## Decision 2: Spatial Backbone — ResNet-18

### Decision
Modified ResNet-18 with **20-channel input** (first conv adapted from 3→20 channels via average-projection init),
**ImageNet pretrained** weights.

### Justification

**Parameter efficiency:** ResNet-18 backbone ≈ 11.7M params. The S5-RVT / RVT-B baseline uses a **MaxViT-style
attention backbone**, and the *whole* baseline detector is ≈ **18M** params (errata ISSUE-03 — the "ViT-Base 86M"
figure is an ImageNet ViT-B/16 number that does **not** apply here). EventSSMDetector at ~20–25M is therefore
**comparable in parameters, not lighter**. The SWaP argument rests on removing quadratic attention and on the
FLOPs/latency measured in Stage 10 — not on a raw parameter count. SWaP context: `Floreano_2015_FutureSmallDrones_Nature`,
`Niculescu_2022_NanoDroneDNN_Deployment_JETCAS`.

**Computational cost (hypothesis to measure):** convolutions are O(L) in tokens whereas self-attention is O(L²) — the
efficiency advantage of the conv backbone over the baseline's attention backbone is a **hypothesis measured on
identical hardware in Stage 10** (FLOPs + latency), not an assumed constant. Quadratic-attention motivation:
`Vaswani_2017_AttentionIsAllYouNeed_NeurIPS`.

**Thesis scientific question:** By replacing the transformer backbone with CNN and keeping everything else equal (same temporal module type, same head, same dataset), you test whether spatial locality bias (convolutions) helps or hurts relative to global attention (transformers) for event-based detection. This is the controlled experiment.

**Transfer learning:** `Zheng_2023_DeepLearningEventVision_Survey_arXiv` confirms that ImageNet pretrained weights help for event camera tasks because low-level feature detectors (edges, corners, gradients) are modality-independent.

### Why Not Alternative Backbones

| Backbone | Parameters | FLOPs | Reason Rejected |
|---|---|---|---|
| ResNet-34 | ~21.8M | ~3.7G | Heavier with no justified accuracy gain for this task. Undermines SWaP argument. |
| ResNet-50 | ~25.6M | ~4.1G | Too large. Negates the efficiency motivation entirely. |
| EfficientNet-B0 | ~5.3M | ~0.4G | Depthwise separable convolutions are harder to optimise on micro-UAV hardware (irregular memory access). ResNet-18 is more hardware-friendly. |
| MobileNetV2 | ~3.4M | ~0.3G | Better SWaP but lower feature quality. Risk of mAP too low to be scientifically interesting. |
| Attention backbone (MaxViT-style, S5-RVT/RVT-B) | ~18M (whole detector) | measure (Stage 10) | What you are *replacing* — quadratic O(L²) attention. (Not ImageNet ViT-B/16's 86M/17.6G.) |
| VMamba (`Liu_2024_VMamba`) | ~31M | ~8.4G | Pure SSM spatial — this is your *second model* (PureSSMDetector). Do not conflate. |

---

## Decision 3: Temporal Module — Causal Mamba

### Decision (reconciled — see Stage 3c for the as-built design)
One causal (unidirectional) **Mamba-1 block interleaved after each ResNet backbone stage** (at the stage's native
width 64/128/256/512), scanning over the **time/window axis** per spatial location. Hidden state carried across
windows. *(The earlier "4 blocks per FPN scale at d_model=256" placement was superseded; the recurrence runs over
time, not over flattened spatial tokens — errata ISSUE-01.)*

### Justification

**Why Mamba over S5 (your baseline):**
S5-RVT uses the S5 (Simplified Structured State Space) model (`Smith_2023_S5_SimplifiedSSM_ICLR`). S5 is a fixed-parameter SSM — its transition matrices B, C, and Δ do not depend on the input. Mamba (`Gu_2023_Mamba_SelectiveStateSpaces_arXiv`) introduces *input-dependent selectivity*: B, C, and Δ are all computed as functions of the current input x_t. This allows Mamba to selectively retain or discard information based on content — irrelevant events are forgotten quickly, important events are retained.

**SMamba evidence (`Yang_2025_SMamba_EventDetection_AAAI`):** Replacing fixed S4/S5 spatial processing with selective Mamba (S6) blocks improves Gen1 mAP from 47.7 (S5-RVT) to 50.4 — a 2.7 mAP gain purely from selectivity. This is your strongest paper argument for Mamba over S5.

**Why Mamba over ConvLSTM (original RVT baseline):**
`Zubic_2024_SSM_EventCameras_CVPR` (your key baseline paper, CVPR 2024 Spotlight) demonstrated that ConvLSTM-based models exhibit >20 mAP performance degradation when inference frequency differs from training frequency. This is catastrophic for event cameras, where the event rate varies naturally with scene dynamics and UAV speed. SSMs do not suffer this degradation because they can be discretised at arbitrary step sizes — the continuous-time formulation is central to this property.

**Why causal (unidirectional) Mamba:**
For streaming event camera data, the model must produce predictions using only past context. Bidirectional Mamba (as used in Vision Mamba, `Zhu_2024_VisionMamba_ICML`, for image classification) requires the full sequence to be available before processing. In the event camera setting, this would require buffering future windows before predicting on the current one — unacceptable latency for obstacle avoidance. `Falanga_2019_PerceptionLatency_SenseAvoid_RAL` established that reaction latency < 3.5ms is needed for obstacle avoidance at 10 m/s.

### Why Not Alternatives

| Temporal Model | Why Rejected |
|---|---|
| ConvLSTM | >20 mAP degradation at variable event rates (`Zubic_2024`). Sequential processing limits parallelism. |
| S5 SSM | Fixed-parameter — less expressive than Mamba. Still works for temporal generalisation but lower accuracy potential. This is what the baseline already uses; no improvement. |
| S4 (`Gu_2022_S4`) | Predecessor to S5. Uses FFT computation. More complex, superseded by Mamba. |
| Transformer self-attention | Quadratic O(L²) complexity. For sequences of H×W spatial tokens, computationally intractable. |
| GRU/RNN | Sequential processing bottleneck. Vanishing gradient for long sequences. Same variable-rate degradation problem as ConvLSTM. |
| Bidirectional Mamba | Requires future context. Introduces lookahead latency incompatible with real-time UAV operation. |

### Mamba Hyperparameters (Starting Values)

| Parameter | Value | Reasoning |
|---|---|---|
| d_model | per stage: 64/128/256/512 | Mamba runs at each backbone stage's native width |
| d_state | 16 | Mamba default. Higher = more capacity but slower. |
| d_conv | 4 | Mamba default local conv width |
| expand | 2 | Inner dimension = 2 × d_model |
| blocks per stage | 1 (default) | One Mamba block interleaved per ResNet stage (4 stages) |
| Application | Interleaved per backbone stage, over time | Mirrors RVT's RNNDetectorStage; temporal axis, per pixel |

---

## Decision 4: Detection Head — YOLOX

### Decision
Reuse the **YOLOX detection head** from the existing S5-RVT codebase. No modifications.

### FCOS vs YOLOX Explanation

Both are anchor-free detection heads that predict from a Feature Pyramid Network. Neither requires pre-defined anchor boxes.

**FCOS (Fully Convolutional One-Stage):** Each positive spatial location predicts its distance to the four edges of the ground truth box. Uses a centerness score to downweight low-quality predictions (those far from object centres). Tends to produce clean, confident predictions near centres but can generate false positives at boundaries.

**YOLOX:** Uses a *decoupled head* — separate lightweight sub-networks for classification, bounding box regression, and objectness. Empirically, decoupled cls/reg converges faster and achieves equivalent accuracy. Standard in the detection literature since 2021.

**For your thesis, both achieve similar mAP.** The decision is entirely pragmatic: YOLOX is already implemented and producing 47.7 mAP in your baseline. Using it eliminates a new implementation that could introduce bugs and consume 1-2 weeks for zero scientific benefit.

Using the same detection head as S5-RVT also makes your comparison cleaner — any mAP difference is attributable to the backbone and temporal module, not the head.

---

## Decision 5: Loss Functions

> Reconciled: corrected after the Stage-2 code audit — the baseline does **not** use Focal/GIoU.

### Decision
Reuse the baseline YOLOX losses **unmodified**: **BCE** for classification *and* objectness, **IoU loss `1 − iou²`
(weight ×5)** for box regression, with **SimOTA** dynamic label assignment.

**BCE (cls + obj):** YOLOX applies binary cross-entropy on the decoupled classification and objectness branches.
Class imbalance is handled primarily by **SimOTA**, which selects a small, high-quality positive set per ground-truth
box — so the foreground/background ratio the loss sees is balanced by *assignment*, not by a focal term.

**IoU loss `1 − iou²` (×5):** directly optimises box overlap (works for non-overlapping boxes) — the standard YOLOX
regression objective. Evaluation uses **COCO mAP (IoU 0.50:0.95)**, not IoU@0.5 only (errata ISSUE-04).

---

## Decision 6: Training Protocol

### Decision
Match S5-RVT exactly. No deviations.

| Hyperparameter | Value | Source |
|---|---|---|
| Optimiser | AdamW | `Gehrig_2023_RVT`, `Zubic_2024` |
| β1, β2 | 0.9, 0.999 | Standard AdamW |
| Weight decay | 0.05 | `Gehrig_2023_RVT` |
| Base learning rate | 2×10⁻⁴ | `Gehrig_2023_RVT` |
| Backbone LR | 0.1× base (2×10⁻⁵) | Differential LR for pretrained backbone |
| LR schedule | Cosine annealing | `Gehrig_2023_RVT` |
| Warmup epochs | 5 | `Gehrig_2023_RVT` |
| Total epochs | 100 | Match S5-RVT protocol |
| Batch size | 4 sequences | Match S5-RVT |
| Sequence length | T=5 windows | Match S5-RVT |
| Precision | bfloat16 mixed | RTX 5070 Ti supports bf16 |
| Dataset | Gen1 (official splits) | `Perot_2020_Gen1` |
| Augmentations | Horizontal flip, random crop | Match S5-RVT |

---

## Complete Design Specification Table

| Component | Decision | Key Paper |
|---|---|---|
| Input | Stacked 2-pol histogram, 10 bins → 20ch, (20,240,304)→pad (20,256,320) | baseline `stacked_histogram_dt=50_nbins=10` |
| Backbone | ResNet-18, 20-channel modified, ImageNet pretrained | `Floreano_2015`, `Niculescu_2022` |
| FPN | 3-scale (P3, P4, P5), 256 channels | Standard; `Gehrig_2023_RVT` |
| Temporal module | Causal Mamba interleaved per backbone stage, over time | `Gu_2023_Mamba`, `Yang_2025_SMamba` |
| Detection head | YOLOX (reused from baseline) | `Gehrig_2023_RVT` |
| Loss functions | BCE (cls+obj) + IoU `1−iou²` ×5 + SimOTA (reused) | Existing codebase |
| Epochs | 100 | Match `Zubic_2024` |
| Optimiser | AdamW, LR=2e-4, cosine decay | Match `Zubic_2024` |
| Hardware | Linux PC, RTX 5070 Ti | Local |

---

## Step-by-Step Tasks

1. Print or copy the table above into a document called `design_specification.md`
2. Fill in every row — do not leave any cell blank
3. For any row where you are uncertain: write "TBD — discuss with Will Midgley" and bring it to your next supervision meeting
4. Write one paragraph describing how EventSSMDetector differs from S5-RVT and why each difference is justified — this becomes the core of your Thesis B methodology section
5. Get verbal sign-off from supervisor on the design before proceeding to Stage 1

---

## Deliverable

`design_specification.md` — one page, all decisions locked in.

## Success Criteria

You can explain every design choice to your supervisor without referring to notes. If you cannot justify a choice, Stage 0 is not complete.

---

## Next Stage

→ **Stage 1: Architecture Blueprint** — draw the full pipeline with exact tensor shapes before writing any code.
