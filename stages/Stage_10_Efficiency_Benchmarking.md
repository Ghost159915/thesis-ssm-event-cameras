# Stage 10 — Efficiency Benchmarking
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Measure and compare the computational cost of EventSSMDetector vs. S5-RVT on identical hardware. Translate architectural choices into concrete hardware-relevant numbers that support the micro-UAV deployment argument.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** 1–2 days  
**Prerequisite:** Stages 8–9 complete (model evaluated, temporal generalisation measured)

---

## Goal

- Parameter count (both models)
- FLOPs per forward pass (both models)
- Inference latency in ms (both models, same hardware)
- Peak VRAM during inference (both models)
- Completed efficiency comparison table

---

## Why This Stage Exists

### Addressing Research Gap 3 Directly

Your Thesis A (Section 2.11) identified four research gaps. Gap 3: *"Computational efficiency benchmarking for onboard deployment. While the theoretical computational advantages of SSMs over transformers and RNNs are well-established, systematic benchmarking of SSM inference latency, memory consumption, and energy usage on the constrained hardware platforms typical of micro UAVs has not been conducted."*

This stage directly addresses Gap 3. Without these numbers, the SWaP argument is theoretical. With them, it becomes empirical. Your RTX 5070 Ti is not a micro-UAV processor, but it provides hardware-independent relative comparisons: if EventSSMDetector is 3× faster than S5-RVT on the 5070 Ti, it will also be approximately 3× faster on a Jetson Nano or similar embedded GPU.

### Paper Connections

**`Floreano_2015_FutureSmallDrones_Nature`:** Establishes that sub-250g UAVs are the most versatile small platform but face the most severe SWaP constraints. Any perception system must justify its computational footprint.

**`Niculescu_2022_NanoDroneDNN_Deployment_JETCAS`:** Demonstrates DNN deployment on a 27g nano-drone (Crazyflie). Their deployed model runs at ~10–100 FPS on an Arm Cortex-M4 microcontroller. Your model is much larger than their deployed models, but the comparison context is valuable.

**`Li_2024_TamingEventCameras_DroneAvoidance_MobiCom`:** Event camera for drone collision avoidance on commercial hardware. Shows that practical drone avoidance requires inference at 20–100 Hz.

**`Falanga_2019_PerceptionLatency_SenseAvoid_RAL`:** Reaction latency < 3.5ms for avoidance at 10 m/s. At 50ms event windows, your model runs at 20 Hz — sufficient for reactive avoidance at speeds accessible to micro-UAVs (< 2 m/s indoors).

**`Davies_2021_Loihi_NeuromorphicComputing_ProcIEEE`:** Intel's Loihi neuromorphic processor. Context: if your SSM model could be mapped to neuromorphic hardware, power consumption could drop to milliwatts. This is speculative for your thesis but motivates the research direction.

**`Santos_2026_EventVisionUAV_SystematicReview_Sensors`:** Reviews efficiency requirements across UAV event-vision tasks — your efficiency numbers slot directly into this review's framework.

---

## Metric 1: Parameter Count

Parameter count is a hardware-independent proxy for model complexity. More parameters → more memory → harder to deploy on embedded systems.

```python
import torch
from models.event_ssm_detector import EventSSMDetector

# Load your model
model = EventSSMDetector(num_classes=2, pretrained_backbone=False)

# Total parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Per-component breakdown
backbone = sum(p.numel() for p in model.backbone.parameters())
fpn = sum(p.numel() for p in model.fpn.parameters())
temporal = sum(p.numel() for p in model.temporal_P3.parameters()) * 3  # 3 scales
head = sum(p.numel() for p in model.detection_head.parameters())

print("=== Parameter Count ===")
print(f"Backbone (ResNet-18):    {backbone/1e6:.2f}M")
print(f"FPN:                     {fpn/1e6:.2f}M")
print(f"Temporal (3× Mamba):     {temporal/1e6:.2f}M")
print(f"Detection Head:          {head/1e6:.2f}M")
print(f"Total:                   {total/1e6:.2f}M")
print(f"S5-RVT reference:        ~18-22M")

# Also run for S5-RVT (load your existing model)
# s5_rvt = load_s5_rvt()
# s5_total = sum(p.numel() for p in s5_rvt.parameters())
# print(f"\nS5-RVT total:            {s5_total/1e6:.2f}M")
```

---

## Metric 2: FLOPs per Forward Pass

FLOPs (Floating Point Operations) measure computational work per inference call. Lower FLOPs = faster inference on equivalent hardware.

**Note:** For Mamba, FLOPs are O(L) where L is sequence length (H×W spatial tokens). For the ViT in S5-RVT, self-attention is O(L²). For Gen1's 30×38 = 1140 tokens at P3: SSM uses 1140 ops, ViT attention uses 1140² = 1,299,600 ops. This 1000× difference in attention FLOPs is the core efficiency argument.

```bash
pip install fvcore
```

```python
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch

model = EventSSMDetector(num_classes=2, pretrained_backbone=False).cuda()
model.eval()

# Single window input
x = torch.randn(1, 10, 240, 304).cuda()
hidden = model.reset_state(batch_size=1, device='cuda')

# FLOPs analysis
# Note: fvcore may not fully support Mamba's custom CUDA kernels
# It will analyse what it can and skip custom ops
flop_analyzer = FlopCountAnalysis(model, (x, hidden))
flop_analyzer.unsupported_ops_warnings(False)

total_flops = flop_analyzer.total()
print(f"Total FLOPs: {total_flops/1e9:.2f} GFLOPs")
print(f"(Note: Mamba custom ops may be partially counted)")

# Per-module breakdown
print(flop_analyzer.by_module())
```

**Alternative FLOPs tool if fvcore fails:**
```bash
pip install torchprofile
```

```python
from torchprofile import profile_macs
macs = profile_macs(model, (x, hidden))
print(f"MACs: {macs/1e9:.2f} GMACs")  # 1 MAC ≈ 2 FLOPs
```

---

## Metric 3: Inference Latency

This is the most practically relevant metric — how many milliseconds does one forward pass take?

### Setup for Accurate Timing

GPU timing requires special care:
1. **Warmup:** First several runs are slower due to kernel compilation
2. **Synchronization:** GPU operations are asynchronous — you must `torch.cuda.synchronize()` before stopping the timer
3. **Steady state:** Take mean over 200 runs (after warmup) and report mean ± std

```python
import time
import torch

# MUST use evaluation mode and no_grad for inference benchmarking
model.eval()
hidden = model.reset_state(batch_size=1, device='cuda')
x = torch.randn(1, 10, 240, 304).cuda()

# Warmup (GPU kernel compilation, cache warming)
print("Warming up...")
for _ in range(50):
    with torch.no_grad():
        preds, hidden = model(x, hidden)
torch.cuda.synchronize()

# Benchmark
print("Benchmarking...")
times_ms = []
hidden = model.reset_state(batch_size=1, device='cuda')

for i in range(300):
    torch.cuda.synchronize()           # ensure GPU is idle before timing
    start = time.perf_counter()
    with torch.no_grad():
        preds, hidden = model(x, hidden)
    torch.cuda.synchronize()           # wait for GPU to finish
    elapsed_ms = (time.perf_counter() - start) * 1000
    times_ms.append(elapsed_ms)

mean_ms = sum(times_ms) / len(times_ms)
std_ms = (sum((t - mean_ms)**2 for t in times_ms) / len(times_ms)) ** 0.5
min_ms = min(times_ms)
max_hz = 1000 / mean_ms

print(f"\n=== Inference Latency (batch_size=1) ===")
print(f"Mean: {mean_ms:.2f} ± {std_ms:.2f} ms")
print(f"Min:  {min_ms:.2f} ms")
print(f"Max throughput: {max_hz:.0f} Hz")
print(f"S5-RVT reference: ~12 ms (~83 Hz)")
print(f"Real-time threshold (50ms window): {'PASS' if mean_ms < 50 else 'FAIL'}")
```

### Run at Multiple Batch Sizes

```python
for bs in [1, 4, 8]:
    hidden = model.reset_state(batch_size=bs, device='cuda')
    x = torch.randn(bs, 10, 240, 304).cuda()
    # ... [same timing code]
    print(f"Batch size {bs}: {mean_ms:.2f} ms ({1000*bs/mean_ms:.0f} windows/sec)")
```

---

## Metric 4: Peak VRAM Usage

```python
# Inference memory (no gradients)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model.eval()
hidden = model.reset_state(batch_size=1, device='cuda')
x = torch.randn(1, 10, 240, 304).cuda()

with torch.no_grad():
    for _ in range(10):
        preds, hidden = model(x, hidden)

peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak VRAM (inference, batch=1): {peak_mb:.0f} MB ({peak_mb/1024:.2f} GB)")

# Training memory (for comparison)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model.train()
hidden = model.reset_state(batch_size=4, device='cuda')
x = torch.randn(4, 10, 240, 304).cuda()
preds, hidden = model(x, hidden)
loss = sum(p.sum() for p in preds if isinstance(p, torch.Tensor))
loss.backward()
training_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak VRAM (training, batch=4):  {training_mb:.0f} MB ({training_mb/1024:.2f} GB)")
```

---

## Complete Efficiency Comparison Table

After running all benchmarks on both models:

| Metric | EventSSMDetector | S5-RVT | Ratio |
|---|---|---|---|
| Parameters (M) | ? | ~18–22 | ?× |
| FLOPs (GFLOPs) | ? | ? | ?× |
| Inference latency (ms) | ? | ~12 | ?× |
| Max throughput (Hz) | ? | ~83 | ?× |
| VRAM inference (MB) | ? | ? | ?× |
| mAP@0.5 | ? | 47.7 | — |
| **mAP/GFlop** | **?** | **?** | **—** |

The `mAP/GFlop` ratio is the efficiency metric: detection accuracy per unit of computation. Even if EventSSMDetector has slightly lower mAP, a better mAP/GFlop ratio means it achieves more "detection value" per computational unit — the critical metric for micro-UAV deployment.

---

## Contextualising for Micro-UAV Deployment

Your RTX 5070 Ti is a 200W desktop GPU. Micro-UAVs use processors like:
- NVIDIA Jetson Nano: ~10 TOPS, ~5–15W
- NVIDIA Jetson Orin Nano: ~40 TOPS, ~7–15W
- Qualcomm Snapdragon Flight: ~4 TOPS, ~3–5W

You cannot directly report "inference latency on micro-UAV" without hardware access. But you can contextualise:

*"On an RTX 5070 Ti (approximately 20× the TFLOPS of a Jetson Nano), EventSSMDetector achieves X ms inference latency. Scaling to a Jetson Nano, estimated inference latency would be approximately X × 20 = Y ms, or roughly Z Hz. `Li_2024_TamingEventCameras_DroneAvoidance_MobiCom` demonstrates event-based drone avoidance at similar frequencies. Full deployment benchmarking on embedded hardware remains future work."*

If your inference is < 50ms on the 5070 Ti, scaled to Jetson Nano would be < 1000ms — which at 1 Hz is technically real-time for Gen1's 50ms event windows (one detection per 50ms). This is the lower bound; actual deployment would require optimisation (TensorRT, quantisation, pruning).

---

## Deliverable

- Completed efficiency table with all four metrics for both models
- Benchmarking scripts saved to `scripts/benchmark/`
- Written efficiency analysis section for Thesis B report

## Success Criteria

All four metrics measured for EventSSMDetector. S5-RVT measured under identical conditions for comparison. Efficiency table completed. Written analysis contextualising results for micro-UAV deployment.

---

## Summary: What You Now Have After Stage 10

At this point, you have completed the full EventSSMDetector investigation:

| Component | Status |
|---|---|
| Architecture design | Documented (Stage 0–1) |
| Implementation | Complete (Stages 3a–4) |
| Verification | Complete (Stage 5) |
| Preliminary results | Available (Stage 6) |
| Full training | Complete (Stage 7) |
| Primary evaluation | Complete (Stage 8) |
| Temporal generalisation | Measured (Stage 9) |
| Efficiency characterisation | Measured (Stage 10) |

**Next: PureSSMDetector.** The second half of your Thesis B work is the PureSSMDetector (pure SSM spatial processing). With EventSSMDetector complete, you have:
- A working pipeline (reuse everything except the backbone)
- A controlled comparison baseline (EventSSMDetector = your new baseline)
- The hypothesis to test: *"Does replacing CNN spatial extraction with SSM spatial extraction improve over the CNN-SSM hybrid?"*

The PureSSMDetector plan is the next document to write.

---

## Final Result Overview for Thesis B Report

By the end of Stage 10, you have answered three of your four research aims:

**Aim 1 (Architecture Development):** EventSSMDetector built and trained ✓  
**Aim 2 (Comparative Benchmarking):** mAP vs. S5-RVT, RVT, SMamba ✓  
**Aim 3 (Temporal Generalisation):** Degradation curve at 5 frequencies ✓  
**Aim 4 (System Integration):** Obstacle avoidance — future work (Thesis C) ✗

This gives you a complete, substantive Thesis B report.
