# Stage 8 — Evaluation and Analysis
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Evaluate your best checkpoint on the official Gen1 test set, produce the final mAP numbers, populate the comparison table, and analyse what the model gets right and wrong.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** 2–3 days  
**Prerequisite:** Stage 7 complete (best training checkpoint identified)

---

## Goal

- Final mAP@0.5 results: car AP, pedestrian AP, overall mAP
- Completed comparison table against all baselines
- Qualitative analysis of detection success and failure cases
- Interpretation of results in context of your thesis hypothesis

---

## Why This Stage Exists

### Contribution to Thesis B

This is the primary results section of your thesis. The comparison table and its interpretation is the scientific contribution. The Thesis B report's "Progress and Discussion" section (3 pages) is anchored by this result.

---

## Evaluation Protocol (Strict Adherence Required)

### The Protocol

The evaluation protocol for Gen1 is established in `Perot_2020_Gen1_1Mpx_Detection_NeurIPS` and followed by every subsequent paper including `Gehrig_2023_RVT` and `Zubic_2024_SSM`. You must follow it exactly:

- **Dataset split:** Official test split only (not validation, not train)
- **IoU threshold:** 0.5 (mAP@0.5)
- **Classes:** Car and pedestrian separately, then mean
- **Metric:** Average Precision (AP) per class, mean AP (mAP) = mean(AP_car, AP_ped)
- **Evaluation tool:** Your existing evaluation pipeline from S5-RVT (reused, not modified)

**Never modify the evaluation code.** If you suspect a bug, compare its output on S5-RVT to the published 47.7 — if it still produces 47.7, it is correct.

### One-Shot Rule

Test set evaluation happens **exactly once** with your final best checkpoint. Do not:
- Run evaluation to see if you should train more
- Evaluate multiple checkpoints on the test set
- Use test set mAP to guide any decisions

Using the test set multiple times inflates reported performance (you are implicitly selecting the best run). This is a form of data leakage and would invalidate your results for publication.

### Running Evaluation

```bash
# Evaluate best checkpoint on official test set
python evaluate.py \
    --checkpoint results/full_training/checkpoints/best_model.pth \
    --split test \
    --output results/evaluation/

# Expected output format:
# Car AP@0.5:         XX.X
# Pedestrian AP@0.5:  XX.X
# Overall mAP@0.5:    XX.X
```

---

## The Comparison Table

After running evaluation, complete this table:

| Model | Backbone | Temporal Module | Car AP@0.5 | Ped AP@0.5 | mAP@0.5 | Venue |
|---|---|---|---|---|---|---|
| RVT | CNN+ViT | ConvLSTM | 57.0 | 37.5 | 47.2 | CVPR 2023 |
| S5-RVT (baseline) | CNN+ViT | S5 SSM | 56.8 | 38.6 | 47.7 | CVPR 2024 |
| **EventSSMDetector** | **ResNet-18** | **Mamba** | **?** | **?** | **?** | Your thesis |
| SMamba | CNN+Sparse Mamba | ConvLSTM | — | — | 50.4 | AAAI 2025 |

Note: RVT and S5-RVT per-class AP numbers taken from their respective papers. Reproduce exactly as published.

---

## Interpreting Your Result

### Scenario 1: mAP > 47.7 (Better than S5-RVT)

Your CNN-SSM hybrid outperforms the Transformer-SSM hybrid on Gen1.

**What this means:**
- The strong spatial inductive bias of convolutional networks (local processing of spatially contiguous features) provides richer spatial representations than global transformer attention for event-based detection
- ResNet-18's efficient hierarchical feature extraction captures the spatial structure of event camera data at least as well as ViT-Base
- Combined with Mamba's selective temporal processing (better than S5's fixed-parameter approach), this achieves better overall detection

**Thesis argument:** CNN spatial features combined with Mamba temporal processing offer a superior efficiency-accuracy trade-off compared to Transformer-SSM hybrids for event-based UAV perception.

**Paper citations for this argument:**
- `Zheng_2023_DeepLearningEventVision_Survey_arXiv`: CNN features transfer well to event data
- `Gu_2023_Mamba`: Selective SSMs improve over fixed-parameter S5
- `Niculescu_2022_NanoDroneDNN_Deployment_JETCAS`: CNN-based architectures are more deployable on embedded UAV hardware

### Scenario 2: mAP 44–47.7 (Competitive, Within 4 Points)

Your model achieves comparable performance to S5-RVT with significantly less computational cost.

**What this means:**
- Replacing the ViT backbone with ResNet-18 has a moderate accuracy cost (if any)
- The efficiency gain from ResNet-18 (7× fewer parameters, significantly lower FLOPs) may outweigh this accuracy cost for deployment-constrained applications
- The argument shifts to the efficiency axis

**Thesis argument:** EventSSMDetector achieves near-state-of-the-art detection accuracy while offering significantly better computational efficiency, making it more suitable for micro-UAV deployment where SWaP constraints dominate.

**Paper citations:**
- `Floreano_2015_FutureSmallDrones_Nature`: SWaP constraints on micro-UAVs justify efficiency over marginal accuracy gains
- `Li_2024_TamingEventCameras_DroneAvoidance_MobiCom`: Real-world drone avoidance — efficiency and latency matter as much as accuracy
- `Santos_2026_EventVisionUAV_SystematicReview_Sensors`: Systematic review confirming efficiency requirements for UAV event-vision systems

### Scenario 3: mAP < 44 (Below S5-RVT)

Your model underperforms the baseline by more than 3 mAP points.

**What this means — and why it is still a valid thesis result:**

This is a controlled experiment. A negative result with rigorous methodology is scientifically valid. The finding is: *"Direct replacement of the Vision Transformer backbone with ResNet-18 in an SSM-based event detection architecture degrades mAP by X points."*

**Possible explanations (analyse which applies):**
1. **Receptive field issue:** ResNet-18's limited receptive field cannot capture the long-range spatial context that ViT's global attention provides. For event cameras, where the same moving object may generate events at spatially distant locations simultaneously, global context matters.
2. **Feature capacity:** ResNet-18 may not have sufficient representational capacity for the task despite being comparable in parameter count to S5-RVT's ViT.
3. **Temporal resolution:** ResNet-18's stride structure may lose fine-grained temporal information that the ViT's patch processing preserves better.

**Thesis argument:** *"We find that global spatial attention (ViT) provides information critical for event-based detection that local convolutional processing (CNN) cannot replicate, suggesting that the spatial processing method — not just the temporal model — is a key determinant of event-based detection accuracy. This motivates the investigation of pure SSM spatial processing (PureSSMDetector) as an alternative that combines global spatial coverage with SSM temporal efficiency."*

**This result directly motivates PureSSMDetector** — your second model. It strengthens the thesis narrative by making the CNN vs. SSM spatial comparison empirically meaningful.

---

## Qualitative Analysis

### What to Visualise

Generate visualisations for at least 10 diverse test sequences (mix of urban, highway, day, night scenarios in Gen1). Use your existing visualisation code from S5-RVT.

For each sequence, generate:
1. Event frame visualisation (accumulated events as grayscale)
2. Ground truth bounding boxes (green)
3. Your model's predicted bounding boxes (red, with confidence scores)
4. S5-RVT's predicted bounding boxes (blue) — for direct comparison

### Failure Mode Analysis

Categorise failures into:

**False Negatives (missed detections):**
- Small objects (distant pedestrians): check P3 scale predictions
- Fast-moving objects: temporal context may be insufficient
- Partially occluded objects: inherent dataset challenge
- Low event density regions: event camera produce no output in static scenes

**False Positives (phantom detections):**
- Background noise predicted as objects
- Flickering lights or reflections activating the detector
- Non-maximum suppression threshold too low

**Box Quality Issues:**
- Correct class but inaccurate box localisation (lower IoU)
- Check if this is concentrated in P3 (small objects, harder to localise)

### Documenting Failure Cases

Write 2–3 sentences per failure mode:
*"The model consistently misses pedestrians at distances beyond approximately 20 metres, corresponding to objects smaller than 8×4 pixels in the Gen1 resolution. This suggests P3-scale (stride-8) feature extraction is insufficient for very small objects, consistent with the known limitation of stride-8 detection for sub-10-pixel objects."*

---

## Results Table to Complete

```
=== EventSSMDetector Evaluation Results ===

Dataset:     Gen1 (official test split)
Checkpoint:  best_model.pth (epoch ?)
Protocol:    mAP@0.5, following Perot et al. 2020

Per-Class Results:
  Car AP@0.5:          ___
  Pedestrian AP@0.5:   ___
  Overall mAP@0.5:     ___

Comparison:
  vs. RVT (47.2):         +/- ___ mAP points
  vs. S5-RVT (47.7):      +/- ___ mAP points
  vs. SMamba (50.4):       +/- ___ mAP points

Interpretation:
  [Write 2-3 sentences]
```

---

## Deliverable

- Completed comparison table with your results filled in
- `results/evaluation/` containing: mAP numbers, per-class AP, qualitative visualisations for 10 sequences
- Written analysis paragraph for Thesis B report

## Success Criteria

Test set evaluated once. Results recorded. Comparison table completed. At least 5 qualitative visualisations generated. Written interpretation produced.

---

## Next Stage

→ **Stage 9: Temporal Generalisation Experiments** — test your model at different event rates without retraining.
