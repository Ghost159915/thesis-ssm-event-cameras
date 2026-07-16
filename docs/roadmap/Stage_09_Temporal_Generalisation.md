# Stage 9 — Temporal Generalisation Experiments
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Evaluate your trained model at event accumulation frequencies different from the training frequency — without any retraining. This tests the core theoretical advantage of SSMs over ConvLSTM: temporal generalisation across variable event rates.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** 1–2 days (mostly automated — inference only)  
**Prerequisite:** Stage 8 complete (best checkpoint evaluated on test set)

---

## Goal

- mAP@0.5 at 5 different frequency multipliers: {0.25×, 0.5×, 1.0×, 2.0×, 4.0×}
- Degradation curve compared against published S5-RVT and ConvLSTM results from Zubic et al.
- Quantified relative degradation (mAP drop from 1× to 4×)

---

## Why This Stage Exists

### This Is Your Strongest Differentiating Experiment

Your primary evaluation (Stage 8) compares mAP numbers. Many architectures achieve similar mAP on Gen1, and it can be hard to argue one is definitively better based on a 1–2 mAP difference. But temporal generalisation is different — it tests a fundamental architectural property, not just accuracy.

**The core argument from `Zubic_2024_SSM_EventCameras_CVPR` (your key paper):**

Event cameras naturally operate at variable event rates. When a scene is static, almost no events are produced. When a car passes quickly or the camera moves fast, thousands of events per millisecond are produced. A model trained at one event accumulation rate (50ms windows = "standard" rate) that fails catastrophically at a different rate is practically useless for real UAV deployment.

**Published degradation numbers (from Zubic 2024, Table 2):**

| Model | Temporal Module | mAP@1× | mAP@4× | Degradation |
|---|---|---|---|---|
| RVT | ConvLSTM | 47.2 | ~26 | ~21 mAP drop |
| S5-RVT | S5 SSM | 47.7 | ~44 | ~3.76 mAP drop |

Your model uses Mamba. **Mamba should degrade less than S5** because Mamba's input-dependent Δ_t allows the model to adaptively scale its temporal dynamics based on input content. When you give it events from a shorter time window (2× frequency), the input statistics change, and Mamba can adapt its state update to compensate.

**What you expect to find:**
- Mamba degradation < S5 degradation < ConvLSTM degradation
- If Mamba degrades *more* than S5 at variable rates, this is a surprising and noteworthy finding worth reporting

### Connecting to UAV Application

`Santos_2026_EventVisionUAV_SystematicReview_Sensors` reviews event-based UAV vision systems and explicitly notes that variable event rates are a key challenge. During UAV flight, the effective event rate changes continuously — hovering produces very few events, aggressive manoeuvring produces very many. A detector that degrades at non-training event rates is unsuitable for deployment on a real UAV.

`Gallego_2022_EventVision_Survey_TPAMI` documents that event rates can vary by orders of magnitude across typical dynamic vision scenarios. This is inherent to the event camera sensing modality.

`Falanga_2019_PerceptionLatency_SenseAvoid_RAL` and `Li_2024_TamingEventCameras_DroneAvoidance_MobiCom` implicitly require robust performance across varying conditions — which includes varying event rates during flight.

---

## What "Changing Frequency" Means Technically

### Standard Evaluation (1× baseline)

- Events are accumulated over **50ms** windows
- Each window produces 10 temporal bins (B=10)
- The window is then passed through the model
- This is how the model was trained

### At 2× Frequency

- Events are accumulated over **25ms** windows (half the time)
- Still 10 temporal bins, but each bin covers 2.5ms instead of 5ms
- Fewer events per window (half as many, assuming constant event rate)
- The model receives the same input format (10, H, W) but the temporal content is from a shorter period

### At 0.5× Frequency

- Events are accumulated over **100ms** windows (double the time)
- Same 10 bins, each covering 10ms
- More events per window (double, assuming constant event rate)

### The SSM Advantage (From Zubic 2024)

Standard RNNs and transformers are trained at a fixed temporal resolution. Changing the accumulation window violates the distribution they were trained on — performance degrades significantly.

SSMs, viewed as discretised continuous-time systems, can adjust their discretisation step Δ_t:
```
Standard training: Δ_t = Δ_trained (learned parameter)
At 2× frequency:   Δ_t = Δ_trained × 0.5 (shorter time window → smaller step)
At 0.5× frequency: Δ_t = Δ_trained × 2.0 (longer time window → larger step)
```

For Mamba, Δ_t is computed as a function of the input (`Gu_2023_Mamba_SelectiveStateSpaces_arXiv`):
```
Δ_t = Softplus(Parameter_Δ + Linear_Δ(x_t))
```

The `Parameter_Δ` is a global learned timescale. At inference time, when the input statistics change (different event rate), the model's Δ_t will naturally adjust based on the new input content. This is the selectivity advantage.

### Practical Implementation

You need to modify the event accumulation in your evaluation data loader to use different window sizes. Look in your Gen1 data loading code for where `time_window_ms=50` is set — this is the parameter to change.

```python
# Evaluate at each frequency multiplier
frequency_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
time_windows = [50 / f for f in frequency_multipliers]  # [200, 100, 50, 25, 12.5] ms

for f, t_window in zip(frequency_multipliers, time_windows):
    # Load dataset with modified time window
    dataset = Gen1Dataset(split='test', time_window_ms=t_window, ...)
    
    # Evaluate
    map_result = evaluate(model, dataset)
    print(f"Frequency {f}×, window {t_window}ms: mAP = {map_result:.2f}")
```

**Important:** The model is not retrained between frequency evaluations. Load the same best checkpoint from Stage 8 for every evaluation.

---

## Evaluation Script

```python
# evaluate_temporal_generalisation.py
import torch
import json
from pathlib import Path

from models.event_ssm_detector import EventSSMDetector
from data.dataset import Gen1Dataset
from evaluation.evaluator import compute_map

# Load best checkpoint
model = EventSSMDetector(num_classes=2, pretrained_backbone=True)
checkpoint = torch.load('results/full_training/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# Frequency multipliers to test
frequency_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
results = {}

for f in frequency_multipliers:
    time_window_ms = 50.0 / f  # adjust window duration
    
    print(f"\nEvaluating at {f}× frequency (window = {time_window_ms:.1f}ms)...")
    
    dataset = Gen1Dataset(
        split='test',
        time_window_ms=time_window_ms,
        # keep all other settings identical
    )
    
    car_ap, ped_ap, mAP = compute_map(model, dataset, device='cuda')
    results[f] = {'car_AP': car_ap, 'ped_AP': ped_ap, 'mAP': mAP}
    
    print(f"  Car AP: {car_ap:.2f} | Ped AP: {ped_ap:.2f} | mAP: {mAP:.2f}")

# Save results
with open('results/temporal_generalisation/results.json', 'w') as f_out:
    json.dump(results, f_out, indent=2)

print("\nFull results table:")
print(f"{'Frequency':>12} | {'Time Window':>12} | {'Car AP':>8} | {'Ped AP':>8} | {'mAP':>8}")
print("-" * 60)
for f in frequency_multipliers:
    r = results[f]
    print(f"{f:>11.2f}× | {50/f:>10.1f}ms | {r['car_AP']:>8.2f} | {r['ped_AP']:>8.2f} | {r['mAP']:>8.2f}")
```

---

## Expected Results

### Baseline from Zubic et al. 2024

| Model | 0.5× | 1× | 2× | 4× | Max degradation |
|---|---|---|---|---|---|
| RVT (ConvLSTM) | ~44 | 47.2 | ~38 | ~26 | ~21 mAP |
| S5-RVT (S5 SSM) | ~46 | 47.7 | ~46 | ~44 | ~3.76 mAP |

### Your Expected Results

**Optimistic (Mamba matches/beats S5):**

| Model | 0.5× | 1× | 2× | 4× | Degradation |
|---|---|---|---|---|---|
| EventSSMDetector | ~X-0.5 | X (your mAP) | ~X-1 | ~X-3 | < S5's 3.76 |

Mamba's input-dependent selectivity should allow it to adapt to different event rates better than S5's fixed parameters.

**Pessimistic (Mamba degrades more than S5):**

If your Mamba implementation does not properly handle variable-rate discretisation, it may degrade similarly to ConvLSTM. This would be an important finding — suggesting that naive Mamba application does not automatically inherit SSM temporal generalisation properties.

---

## Plotting the Results

```python
import matplotlib.pyplot as plt
import json

# Load your results
with open('results/temporal_generalisation/results.json') as f:
    your_results = json.load(f)

# Published S5-RVT results (from Zubic 2024, Figure 4 or Table)
# Extract or approximate from the paper
s5_rvt_results = {0.25: 46.0, 0.5: 46.5, 1.0: 47.7, 2.0: 46.0, 4.0: 43.94}
rvt_results = {0.25: 41.0, 0.5: 44.0, 1.0: 47.2, 2.0: 38.0, 4.0: 26.0}  # approximate

frequencies = [0.25, 0.5, 1.0, 2.0, 4.0]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(frequencies, [your_results[str(f)]['mAP'] for f in frequencies],
        'b-o', label='EventSSMDetector (yours)', linewidth=2, markersize=8)
ax.plot(frequencies, [s5_rvt_results[f] for f in frequencies],
        'g--s', label='S5-RVT (Zubic 2024)', linewidth=2, markersize=8)
ax.plot(frequencies, [rvt_results[f] for f in frequencies],
        'r:^', label='RVT (Gehrig 2023)', linewidth=2, markersize=8)

ax.set_xlabel('Frequency Multiplier (× training frequency)', fontsize=12)
ax.set_ylabel('mAP@0.5', fontsize=12)
ax.set_title('Temporal Generalisation: mAP vs. Event Rate', fontsize=13)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Training frequency')
ax.legend(fontsize=10)
ax.set_xscale('log', base=2)
ax.set_xticks(frequencies)
ax.set_xticklabels(['0.25×', '0.5×', '1.0×\n(training)', '2.0×', '4.0×'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/temporal_generalisation/degradation_curve.png', dpi=150)
plt.show()
```

---

## Results Table to Complete

| Model | 0.25× | 0.5× | 1.0× (training) | 2.0× | 4.0× | Max degradation |
|---|---|---|---|---|---|---|
| RVT (ConvLSTM) | — | — | 47.2 | — | — | ~21 mAP |
| S5-RVT (S5 SSM) | — | — | 47.7 | — | 43.94 | 3.76 mAP |
| **EventSSMDetector** | **?** | **?** | **?** | **?** | **?** | **?** |

---

## Deliverable

- Temporal generalisation results JSON file
- Degradation curve plot (saved as PDF for thesis)
- Completed comparison table
- Written analysis paragraph for Thesis B report

## Success Criteria

All 5 frequency evaluations completed. Degradation curve plot generated. Results compared against published S5-RVT curve.

---

## Next Stage

→ **Stage 10: Efficiency Benchmarking** — measure computational cost vs. S5-RVT to complete the efficiency argument for micro-UAV deployment.
