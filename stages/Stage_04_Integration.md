# Stage 4 — Full Model Integration
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Wire all four tested components into a single `EventSSMDetector` class with correct state management, and integrate it into the existing training loop with minimal code changes.

**System:** Linux PC, RTX 5070 Ti  
**File to create:** `models/event_ssm_detector.py`  
**Files to modify:** `train.py` (swap model class only)  
**Estimated time:** 3–5 days  
**Prerequisite:** Stages 3a–3d all passing their individual tests

---

## Goal

- A single `EventSSMDetector` class with correct `forward()`, `reset_state()`, and `load_pretrained()` methods
- The existing training loop runs at least one batch without error
- Parameter count confirmed ~18–22M
- Full integration test passes (3-window sequence without NaN)

---

## Why This Stage Exists

### Contribution to Thesis B

This stage produces the actual research contribution — the novel architecture. Everything before Stage 4 was preparation; Stage 4 is construction. The model class you create here is what will be trained, evaluated, and compared against the baseline. The quality of the state management implementation here directly determines whether your temporal generalisation results (Stage 9) are scientifically valid.

---

## State Management: The Critical Design

### Why State Management is the Hardest Part

All four components individually are straightforward. The complexity in Stage 4 is managing the Mamba hidden state correctly across the training loop's sequence of windows.

Get this wrong and one of two things happens:
1. **Memory explosion:** GPU OOM after a few batches because gradients are accumulating across the entire sequence history
2. **No temporal memory:** The model effectively treats each window independently because the state is reset too aggressively, losing the core advantage of using an SSM

### The Correct Pattern: Truncated Backpropagation Through Time (TBPTT)

Truncated BPTT is the standard approach for training recurrent models on sequences. The key insight: you maintain the **values** of the hidden state across windows (temporal memory is preserved) but **detach** the hidden state from the computational graph at each window boundary (gradients are not backpropagated across windows).

```python
# Correct training loop (pseudocode)
hidden_state = model.reset_state(batch_size=B, device='cuda')

for t in range(T):  # T = 5 windows per sequence
    window = sequence[:, t, :, :, :]         # (B, 10, H, W)
    predictions, hidden_state = model(window, hidden_state)
    
    loss = criterion(predictions, targets[:, t])
    loss.backward()                    # Backprop only through current window
    
    # CRITICAL: detach state before next window
    # This preserves state VALUES but disconnects the gradient graph
    hidden_state = {k: v.detach() for k, v in hidden_state.items()}

optimiser.step()
optimiser.zero_grad()
```

### Why `.detach()` is Not the Same as Setting to None

```python
# .detach() = keep values, disconnect from grad graph
hidden_state = hidden_state.detach()      # ✓ temporal memory preserved, no grad explosion

# .clone() = keeps values AND gradient connection
hidden_state = hidden_state.clone()       # ✗ gradients still flow backwards — memory explosion

# = None = reset to zero
hidden_state = None                       # ✗ temporal memory lost — model is stateless

# Correct analogy: .detach() is like "save the brain state but start a fresh gradient tape"
```

**Reference:** `Zubic_2024_SSM_EventCameras_CVPR` uses this exact approach in their S5-RVT implementation. Look at how their training loop manages the S5 hidden state — your code should be structurally identical.

### Hidden State Structure

With 3 FPN scales and 4 Mamba layers per scale:

```python
hidden_state = {
    'P3': [h_layer0, h_layer1, h_layer2, h_layer3],  # 4 tensors, each (B, 512, 16)
    'P4': [h_layer0, h_layer1, h_layer2, h_layer3],
    'P5': [h_layer0, h_layer1, h_layer2, h_layer3],
}
```

Or simplified (if your MambaTemporalModule returns a single combined state):
```python
hidden_state = {
    'P3': tensor_or_list,
    'P4': tensor_or_list,
    'P5': tensor_or_list,
}
```

The exact structure depends on your Stage 3c implementation. Use whatever your `MambaTemporalModule` returns — just make sure `reset_state()` creates the same structure with zeros.

### Sequence Resets

A new recording in Gen1 is an independent event sequence. The hidden state must be reset at sequence boundaries:

```python
# In your dataset/dataloader:
# Each batch item is either 'same_sequence_continue' or 'new_sequence_start'
# Pass this information to the training loop

if is_new_sequence:
    hidden_state = model.reset_state(batch_size=B, device='cuda')
```

How does your existing S5-RVT training loop handle this? It already does this for the S5 state — copy the same logic.

---

## Model Interface (Must Match S5-RVT's Interface)

The most important design goal for Stage 4: EventSSMDetector should have the **identical calling interface** as S5-RVT. This means the training loop code barely changes (just swapping the model class).

Look at your S5-RVT model and note its exact calling convention:
```python
# How is S5-RVT called in your training loop?
predictions, hidden_state = s5_rvt_model(window, hidden_state)
# OR
output = s5_rvt_model(window, hidden_state=state)
# OR something else?
```

Make EventSSMDetector match this exactly.

---

## Implementation Specification

### Class: `EventSSMDetector`

**File:** `models/event_ssm_detector.py`

**`__init__(self, num_classes=2, pretrained_backbone=True)`**
1. Instantiate `ResNet18EventBackbone(pretrained=pretrained_backbone, in_channels=10)`
2. Instantiate `EventFPN(in_channels_list=[128, 256, 512], out_channels=256)`
3. Instantiate 3 separate `MambaTemporalModule` objects (one per FPN scale)
   - Store as `self.temporal_P3`, `self.temporal_P4`, `self.temporal_P5`
4. Instantiate detection head (reused YOLOX head from existing codebase)
   - Adapt instantiation arguments to match what Stage 3d documented

**`forward(self, x, hidden_states=None) → (predictions, new_hidden_states)`**
```
1. x: (B, 10, H, W) — current event window
2. backbone_features = self.backbone(x)         → {C2, C3, C4}
3. fpn_features = self.fpn(backbone_features)   → {P3, P4, P5}
4. For each scale:
   - h_prev = hidden_states[scale] if hidden_states else None
   - feat_out, h_new = self.temporal_Pscale(fpn_features[scale], h_prev)
   - temporal_features[scale] = feat_out
   - new_hidden_states[scale] = h_new
5. predictions = self.detection_head(temporal_features)
6. return predictions, new_hidden_states
```

**`reset_state(self, batch_size, device) → dict`**
```
Returns:
{
    'P3': self.temporal_P3.reset_state(batch_size, device),
    'P4': self.temporal_P4.reset_state(batch_size, device),
    'P5': self.temporal_P5.reset_state(batch_size, device),
}
```

**`get_param_groups(self, base_lr) → list`**
```
Returns parameter groups with differential learning rates:
- Backbone (pretrained): lr = base_lr * 0.1
- FPN, Mamba, Head (randomly init): lr = base_lr * 1.0
```

---

## Integration Test

This test simulates a 3-window sequence — the minimal proof that integration works end-to-end:

```python
import torch
from models.event_ssm_detector import EventSSMDetector

model = EventSSMDetector(num_classes=2, pretrained_backbone=False).cuda()
model.train()

B = 2    # batch size
T = 3    # number of windows in sequence

# Initial state
hidden_state = model.reset_state(batch_size=B, device='cuda')

total_loss = 0
for t in range(T):
    # Fake event window
    window = torch.randn(B, 10, 240, 304).cuda()
    
    # Forward pass
    predictions, hidden_state = model(window, hidden_state)
    
    # Fake loss (in real training, this uses actual ground truth)
    if isinstance(predictions, torch.Tensor):
        loss = predictions.sum()
    else:
        loss = sum(p.sum() for p in predictions if isinstance(p, torch.Tensor))
    
    loss.backward()
    total_loss += loss.item()
    
    # Detach state (CRITICAL)
    hidden_state = {k: (v.detach() if isinstance(v, torch.Tensor)
                        else [vi.detach() for vi in v])
                    for k, v in hidden_state.items()}

print(f"Integration test: PASS — total loss over {T} windows: {total_loss:.4f}")
assert not torch.isnan(torch.tensor(total_loss)), "NaN loss!"
```

---

## Parameter Count Verification

```python
total_params = sum(p.numel() for p in model.parameters()) / 1e6
backbone_params = sum(p.numel() for p in model.backbone.parameters()) / 1e6
fpn_params = sum(p.numel() for p in model.fpn.parameters()) / 1e6
temporal_params = sum(p.numel() for p in model.temporal_P3.parameters()) / 1e6 * 3
head_params = sum(p.numel() for p in model.detection_head.parameters()) / 1e6

print(f"Backbone:    {backbone_params:.1f}M")
print(f"FPN:         {fpn_params:.1f}M")
print(f"Temporal:    {temporal_params:.1f}M (3 modules)")
print(f"Head:        {head_params:.1f}M")
print(f"Total:       {total_params:.1f}M")
print(f"S5-RVT ref:  ~18-22M")

# Acceptable range: 15-25M
assert 10 < total_params < 35, f"Parameter count out of expected range: {total_params:.1f}M"
print("Parameter count: OK")
```

---

## Modifying `train.py`

This is the minimal change to make your new model work with the existing training loop:

```python
# OLD (S5-RVT):
# from models.s5_rvt import S5RVT
# model = S5RVT(num_classes=2)

# NEW (EventSSMDetector):
from models.event_ssm_detector import EventSSMDetector
model = EventSSMDetector(num_classes=2, pretrained_backbone=True)

# Differential learning rate (add to optimiser setup):
param_groups = model.get_param_groups(base_lr=2e-4)
optimiser = torch.optim.AdamW(param_groups, weight_decay=0.05)
```

If the training loop already manages hidden state (for S5-RVT), the state management logic should work without changes as long as EventSSMDetector's interface matches S5-RVT's.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Training loop hidden state logic assumes a specific format | Look at S5-RVT's state format and make EventSSMDetector return the same format structure |
| NaN loss from the first training step | Check gradient norms — if too large, reduce initial LR or add gradient clipping (should already be in codebase from S5-RVT) |
| Memory explosion (OOM) after first few batches | Check that detach() is called on hidden state in the training loop. Add `torch.cuda.empty_cache()` temporarily to diagnose. |
| Head integration error (format mismatch) | Fix in the FPN output format adapter found in Stage 3d |

---

## Deliverable

`models/event_ssm_detector.py` passing integration test. `train.py` modified to use the new model. Parameter count printed and confirmed.

## Success Criteria

Integration test runs 3 windows without error, without NaN loss, without OOM. Parameter count 15–25M.

---

## Next Stage

→ **Stage 5: Smoke Testing** — verify the model can actually learn before committing GPU time to full training.
