# Stage 5 — Smoke Testing
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Verify that EventSSMDetector can actually learn before committing GPU time to a real training run. The overfit test is the gold standard: a model that cannot memorise 5 training examples is fundamentally broken.

**System:** Linux PC, RTX 5070 Ti  
**Estimated time:** 2–3 days  
**Prerequisite:** Stage 4 complete (integration test passing)

---

## Goal

- Training loss drops to < 30% of initial value when trained on 5 samples for 50 epochs
- All parameters receive non-zero gradients
- Memory usage quantified (< 10GB for batch size 4)
- Inference speed measured (target < 20ms per window)
- No NaN values at any point

---

## Why This Stage Exists

### Contribution to Thesis B

A model that does not learn is not a research finding — it is an infrastructure failure. Smoke testing ensures that when you report "EventSSMDetector achieved X mAP after 100 epochs of training," the marker can trust that the training machinery was fundamentally correct.

The overfit test result (loss curves on 5 samples) also provides an early proof of learning for your Thesis B preliminary results section: *"We verified that the architecture can memorise a small training set, confirming correct gradient flow and loss computation."*

The inference speed measurement directly contributes to your efficiency argument — the comparison with S5-RVT's ~12ms inference is a key thesis claim.

**Relevant papers:**
- `Falanga_2019_PerceptionLatency_SenseAvoid_RAL`: Reaction latency < 3.5ms needed for obstacle avoidance at 10 m/s. Your inference speed contributes to this discussion.
- `Santos_2026_EventVisionUAV_SystematicReview_Sensors`: Systematic review of event-based UAV systems including latency requirements.
- `Niculescu_2022_NanoDroneDNN_Deployment_JETCAS`: Nano-drone DNN deployment — confirms practical latency requirements for onboard inference.

---

## Test 1: The Overfit Test

### What It Is and Why It Works

If you train a model on only 5 samples for enough iterations with sufficient capacity, it should memorise those 5 samples — the training loss should approach near-zero. If it doesn't, something is wrong: either the gradients are not flowing correctly, the loss function has a bug, or the model architecture has a logical error that prevents learning.

This test was popularised as a debugging technique in the deep learning community. The logic: if you cannot prove your model can learn the *easiest possible task* (memorise 5 examples), you have no basis for claiming it can learn the *real task* (detect objects in 1M+ examples).

### Setup

```python
# smoke_test.py
import torch
import torch.nn as nn
from torch.utils.data import Subset
from models.event_ssm_detector import EventSSMDetector

# 1. Load your Gen1 dataset (same as training)
dataset = Gen1Dataset(split='train', ...)  # from your existing data loader

# 2. Take EXACTLY 5 samples
smoke_dataset = Subset(dataset, indices=[0, 1, 2, 3, 4])
dataloader = DataLoader(smoke_dataset, batch_size=2, shuffle=False)

# 3. Create model — no pretrained weights (test pure learning)
model = EventSSMDetector(num_classes=2, pretrained_backbone=False).cuda()
model.train()

# 4. Optimiser — large LR for fast memorisation
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 5. Training loop — 50 epochs on 5 samples
initial_loss = None
for epoch in range(50):
    epoch_loss = 0
    hidden_state = model.reset_state(batch_size=..., device='cuda')
    
    for batch in dataloader:
        windows, targets = batch  # adjust to your data format
        for t in range(windows.shape[1]):
            preds, hidden_state = model(windows[:, t], hidden_state)
            loss = compute_loss(preds, targets[:, t])  # your existing loss function
            loss.backward()
            hidden_state = {k: v.detach() for k, v in hidden_state.items()}
        
        optimiser.step()
        optimiser.zero_grad()
        epoch_loss += loss.item()
    
    if epoch == 0:
        initial_loss = epoch_loss
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: loss = {epoch_loss:.4f}")

final_loss = epoch_loss
reduction = initial_loss / final_loss
print(f"\nInitial loss: {initial_loss:.4f}")
print(f"Final loss:   {final_loss:.4f}")
print(f"Reduction:    {reduction:.1f}×")

if reduction > 3.0:
    print("OVERFIT TEST: PASS ✓")
else:
    print("OVERFIT TEST: FAIL ✗ — investigate before proceeding")
```

### Interpreting Results

| Outcome | What It Means |
|---|---|
| Loss → near zero in < 20 epochs | Excellent. Model has high capacity and correct learning dynamics. |
| Loss drops 5–10× but doesn't reach zero | Good. Focal Loss prevents perfect loss=0 due to confidence calibration. Acceptable. |
| Loss drops 2–3× | Marginal. Model is learning but slowly. Try higher LR or check gradient magnitudes. |
| Loss does not decrease | Critical failure. See debugging section below. |
| Loss decreases then oscillates | LR too high. Reduce by 10×. |
| Loss is NaN from epoch 1 | Gradient explosion. Reduce LR to 1e-5. Add gradient clipping. |

---

## Test 2: Gradient Flow Check

Run after the overfit test. Every trainable parameter must receive a non-zero gradient.

```python
model.train()
hidden_state = model.reset_state(batch_size=2, device='cuda')

# One forward pass on a single batch
window = torch.randn(2, 10, 240, 304).cuda()
preds, _ = model(window, hidden_state)
loss = compute_loss(preds, fake_targets)
loss.backward()

# Check every parameter
issues = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.grad is None:
        issues.append(f"NO GRADIENT: {name}")
    elif param.grad.abs().max() < 1e-9:
        issues.append(f"VANISHING GRADIENT (< 1e-9): {name}")
    elif torch.isnan(param.grad).any():
        issues.append(f"NaN GRADIENT: {name}")

if issues:
    print("GRADIENT ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("GRADIENT FLOW: PASS ✓ — all parameters receiving gradients")
```

### Common Gradient Issues

**`NO GRADIENT` for backbone layers:**
- Symptom: `backbone.layer2.0.conv1.weight: NO GRADIENT`
- Cause: A `.detach()` in the wrong place in the forward pass, or backbone wrapped in `torch.no_grad()`
- Fix: Check your backbone forward() call — remove any `with torch.no_grad():` blocks during training

**`NO GRADIENT` for first conv:**
- Cause: New first conv layer was created but not registered with `nn.Module` correctly
- Fix: Make sure you assign it as `self.backbone.conv1 = new_conv` (not a local variable)

**`VANISHING GRADIENT` in Mamba layers:**
- Cause: Common in deep SSMs. Mamba's architecture is designed to avoid this (skip connections, gating), so vanishing gradients indicate an implementation error.
- Fix: Check that Mamba blocks include proper residual connections

---

## Test 3: Memory Profiling

```python
# Run for each batch size
for batch_size in [1, 2, 4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    hidden = model.reset_state(batch_size, 'cuda')
    x = torch.randn(batch_size, 10, 240, 304).cuda()
    
    # Forward + backward
    preds, hidden = model(x, hidden)
    loss = sum(p.sum() for p in preds if isinstance(p, torch.Tensor))
    loss.backward()
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Batch size {batch_size}: peak VRAM = {peak_mem:.1f} GB")

# Expected (rough):
# batch_size=1: ~2-3 GB
# batch_size=2: ~4-5 GB
# batch_size=4: ~7-9 GB  ← target for full training
# batch_size=8: ~13-15 GB (may fit on 16GB 5070 Ti but tight)
```

**If batch size 4 exceeds 10GB:**
- Enable gradient checkpointing in ResNet-18 (recomputes activations during backward to save memory)
- Reduce sequence length T from 5 to 3
- Reduce Mamba d_model from 256 to 128 (significant architecture change — discuss with supervisor first)

---

## Test 4: Inference Speed Benchmark

```python
import time

model.eval()
hidden = model.reset_state(batch_size=1, device='cuda')
x = torch.randn(1, 10, 240, 304).cuda()

# Warmup (GPU needs to warm up for accurate timing)
for _ in range(20):
    with torch.no_grad():
        preds, hidden = model(x, hidden)
torch.cuda.synchronize()

# Benchmark
times = []
hidden = model.reset_state(batch_size=1, device='cuda')
for i in range(200):
    start = time.perf_counter()
    with torch.no_grad():
        preds, hidden = model(x, hidden)
    torch.cuda.synchronize()  # CRITICAL: wait for GPU to finish
    times.append(time.perf_counter() - start)

mean_ms = 1000 * sum(times) / len(times)
std_ms = 1000 * (sum((t - mean_ms/1000)**2 for t in times) / len(times))**0.5
max_hz = 1000 / mean_ms

print(f"Inference latency: {mean_ms:.1f} ± {std_ms:.1f} ms")
print(f"Max throughput: {max_hz:.0f} Hz")
print(f"Comparison — S5-RVT: ~12 ms (~83 Hz)")

# Context for UAV deployment:
# Gen1 events accumulate over 50ms windows → need < 50ms inference for real-time
# Falanga 2019 requires < 3.5ms for collision avoidance — not achievable with this model
# Li 2024 uses 10-50ms perception — your model should fit this range
```

---

## Debugging Guide

### NaN Loss

```python
# Add this check inside training loop to find WHERE NaN first appears
for name, module in model.named_modules():
    def hook(m, inp, out, name=name):
        if isinstance(out, torch.Tensor) and torch.isnan(out).any():
            print(f"NaN detected in {name}")
    module.register_forward_hook(hook)
```

Common causes:
1. Learning rate too high → gradient explosion → NaN weights
2. Log(0) in Focal Loss (probabilities too close to 0 or 1)
3. Division by zero in GIoU Loss (degenerate predicted boxes with zero area)

Fixes:
1. Reduce learning rate 10×
2. Add epsilon: `log(p + 1e-8)` in loss
3. Clamp predicted box dimensions: `w = torch.clamp(w, min=1e-4)`

### Loss Not Decreasing

Check in order:
1. Are targets in the expected format for the loss function?
2. Is the loss being called with the right arguments? (swap preds/targets accidentally?)
3. Is gradient clipping too aggressive? (Check `max_norm` in gradient clipper)
4. Is the optimiser actually being stepped? (Common bug: `optimizer.step()` before `loss.backward()`)

### Training Very Slow (< 50% GPU utilisation)

```bash
nvidia-smi  # check GPU utilisation
```

If GPU < 50%: data loading is the bottleneck. Increase DataLoader `num_workers`:
```python
DataLoader(..., num_workers=8, pin_memory=True, prefetch_factor=2)
```

---

## Smoke Test Results Table (Fill In)

| Test | Result | Notes |
|---|---|---|
| Overfit test (loss reduction) | ___× | Initial loss: ___, Final loss: ___ |
| Gradient flow | PASS / FAIL | Any problematic layers: ___ |
| Memory @ batch_size=4 | ___ GB | Target: < 10 GB |
| Inference latency | ___ ms | Target: < 20 ms |
| Max throughput | ___ Hz | Target: > 50 Hz |

Save this table in `results/smoke_test/smoke_test_results.md`.

---

## Deliverable

- Completed smoke test results table
- All 4 tests passing
- No NaN values observed
- `results/smoke_test/` directory with loss curves and results

## Success Criteria

Overfit test: loss reduction ≥ 3× (ideally ≥ 10×). All parameters have gradients. VRAM < 10GB at batch size 4. Inference < 25ms.

---

## Next Stage

→ **Stage 6: Short Training Run** — train on 10% of Gen1 for 20 epochs to validate training dynamics before the full run.
