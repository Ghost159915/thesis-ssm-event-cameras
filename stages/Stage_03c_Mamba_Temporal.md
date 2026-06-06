# Stage 3c — Mamba Temporal Module
**EventSSMDetector | Thesis B | MMAN4952 | UNSW Sydney**

---

## Overview

Build and test the Mamba temporal module — the most novel and technically important component of EventSSMDetector. This module maintains temporal memory across consecutive event windows, enabling the model to understand motion and context over time.

**System:** Linux PC, RTX 5070 Ti  
**File to create:** `models/temporal/mamba_temporal.py`  
**Estimated time:** 2–4 days (most complex component)  
**Prerequisite:** Stages 3a and 3b complete. mamba-ssm installed and tested.

---

## Goal

A `MambaTemporalModule` class that:
- Takes spatial features `(B, 256, H, W)` + optional hidden state from previous window
- Applies N causal Mamba blocks
- Returns enriched features `(B, 256, H, W)` + new hidden state
- Correctly propagates hidden state across consecutive windows
- Passes shape test, state persistence test, and gradient test

---

## Why This Component

### The Temporal Understanding Problem

Event cameras are fundamentally different from frame-based cameras: they produce a continuous stream of temporal change events rather than discrete snapshots. A model that processes each window independently (no temporal memory) throws away the most valuable information event cameras provide — the *history* of what has been moving and where.

**The failure of ConvLSTM:** `Zubic_2024_SSM_EventCameras_CVPR` (your primary SSM paper, CVPR 2024 Spotlight) demonstrated experimentally that ConvLSTM-based temporal models (used in the original RVT) suffer severe performance degradation when inference frequency differs from training frequency. Table 2 in their paper shows >20 mAP degradation for ConvLSTM at 4× training frequency vs. only 3.76 mAP degradation for SSMs. This happens because ConvLSTM is a discrete-time model — it was trained assuming a fixed time step between inputs, and changing that time step breaks its learned dynamics.

**Why SSMs solve this:** SSMs model temporal dynamics in continuous time (`Gu_2022_S4_StructuredStateSpaces_ICLR`). The continuous-time formulation (h'(t) = Ah(t) + Bu(t)) can be discretised at any time step Δ without retraining. For event cameras, where the effective event rate varies naturally with scene dynamics (fast motion → many events → smaller effective Δ; slow motion → few events → larger effective Δ), this is critical.

**Why Mamba over S5:** `Gu_2023_Mamba_SelectiveStateSpaces_arXiv` introduced *selective state spaces* — the B, C, and Δ parameters are computed as functions of the input x_t, not learned as fixed parameters. This allows the model to adaptively decide what information to retain (large Δ = forget fast) and what to preserve (small Δ = long memory). For event cameras, this selectivity is directly motivated: during periods of high event activity (object moving fast), the model should update its state rapidly; during quiet periods, it should preserve its state.

**SMamba evidence** (`Yang_2025_SMamba_EventDetection_AAAI`): Replacing fixed-parameter S4/S5 spatial processing with selective Mamba (S6) improves Gen1 mAP from 47.7 to 50.4. The performance gain from selectivity alone is 2.7 mAP — strong empirical evidence for Mamba over S5 in this domain.

---

## Mamba Architecture (Internal Detail)

A Mamba block transforms an input sequence:
```
Input:  (B, L, d_model) where L = H×W spatial tokens, d_model = 256
Output: (B, L, d_model) — same shape
```

Internally, each Mamba block applies a selective scan mechanism:
1. Input projection: `(B, L, d_model)` → `(B, L, d_model * expand)` via linear layer
2. Depthwise conv 1D: short-range local context (d_conv=4 width)
3. SSM (the selective scan): processes the sequence and maintains state
   - B(x_t), C(x_t): input-dependent projection matrices
   - Δ(x_t): input-dependent discretisation step
   - Hidden state: `(B, d_model * expand, d_state)` = `(B, 512, 16)` per block
4. Output gating: element-wise multiplication with a gate
5. Output projection: `(B, L, d_model * expand)` → `(B, L, d_model)`

The hidden state `(B, 512, 16)` is what carries temporal information across windows.

---

## The Core Challenge: State Management

### The Problem in Detail

During training, you process sequences of T windows. The standard Mamba `forward()` call processes the entire sequence at once using a parallel scan — it is not aware of "window boundaries." You need to:

1. Process window 1 → get hidden state h_1
2. Carry h_1 to window 2 → process window 2 with h_1 as initial state → get h_2
3. Carry h_2 to window 3 → and so on

But the standard parallel scan in Mamba always starts from zero state. To maintain state across windows, you need to either:
- Use the **step-by-step (recurrent) inference mode** which explicitly handles state
- Use the **initial state injection** approach where you prepend the previous state to the sequence

### Approach: Step-by-Step Processing Per Window

For training (where you process a sequence of T=5 windows per batch):

```
For window t in sequence [1, 2, 3, 4, 5]:
    1. Flatten spatial: (B, 256, H, W) → (B, H*W, 256)  [treat spatial as sequence]
    2. Process with Mamba in recurrent mode, providing h_{t-1} as initial state
    3. Get output (B, H*W, 256) and new state h_t
    4. Reshape output: (B, H*W, 256) → (B, 256, H, W)
    5. h_t.detach() before passing to next window
```

### How to Implement with mamba-ssm

The `mamba_ssm.Mamba` class supports state passing via the `inference_params` mechanism. However, the API varies between versions. Check which approach works with your installed version:

**Approach A (mamba-ssm >= 1.2):**
```python
from mamba_ssm.utils.generation import InferenceParams
# InferenceParams stores the key-value (state) cache across steps
```

**Approach B (any version — manual state):**
Use Mamba in full-sequence mode but manually track what the "state" would be by running a warm-up pass at training time. This is less accurate but simpler.

**Approach C (simplest — sequence mode with detach):**
Process each window as an independent sequence (resetting state to zero each time) *during training* — this means the model does not actually use inter-window context during training, only during inference. This is a valid simplification for an initial implementation. The model learns to encode temporal context within a window from the 10 temporal bins, and the inter-window state is used at inference time.

**Decision for initial implementation: Use Approach C (simplest) for the smoke test and short training run.** Implement proper state passing (Approach A) before full training. The reason: getting the model training at all is the priority; state management can be refined.

### State Passing During Inference (Always Required)

During inference (evaluation), you process one window at a time and must always pass state:

```python
hidden_state = model.reset_state(batch_size=1)
for window in event_stream:
    predictions, hidden_state = model(window, hidden_state)
    hidden_state = hidden_state.detach()  # Not strictly needed at inference (no backprop)
    yield predictions
```

---

## The Spatial Linearisation Decision

### Which Scan Order?

When you flatten `(B, 256, H, W)` → `(B, H*W, 256)`, the spatial ordering of tokens affects what Mamba "sees" as neighbours in the sequence.

**Row-major (standard):**
```
token_0 = position (0,0), token_1 = (0,1), ..., token_W-1 = (0,W-1),
token_W = (1,0), ...
```
Token W-1 (last pixel of row 0) is adjacent to token W (first pixel of row 1) in the Mamba sequence, but they are not spatially adjacent. This breaks spatial locality.

**Why this is acceptable for your architecture:**
Your model has ResNet-18 doing all the spatial feature extraction with proper convolutions. By the time features reach the Mamba module, they already encode spatial context through convolutional receptive fields. Mamba's job is *temporal processing* — tracking changes across windows — not spatial reasoning. The scan order matters much more for pure SSM spatial models (like your PureSSMDetector) where there is no CNN to establish spatial context.

**Decision: Row-major flatten. Simple, effective, justified by the hybrid architecture.**

For context: `Liu_2024_VMamba_VisualStateSpace_arXiv` and `Zhu_2024_VisionMamba_ICML` explore elaborate scan patterns (horizontal + vertical, cross-scan) but these are for pure SSM spatial processing without a CNN backbone.

---

## Separate Mamba Modules per FPN Scale

You create three independent `MambaTemporalModule` instances — one each for P3, P4, P5.

**Why three separate modules:**
- P3 handles small objects (distant pedestrians). Their motion pattern in feature space is different from large nearby cars (P5).
- Separate modules allow specialised temporal dynamics per scale.
- Parameter cost: 3× the Mamba parameters, but Mamba is lightweight compared to ResNet-18.

**Why not shared weights:**
Shared weights would force the same temporal dynamics at all scales. This is likely suboptimal given the different object sizes and motion patterns at each scale.

---

## Implementation Specification

### Class: `MambaTemporalModule`

**File:** `models/temporal/mamba_temporal.py`

**`__init__(self, d_model=256, d_state=16, d_conv=4, expand=2, num_layers=4)`**
1. Create `nn.ModuleList` of `num_layers` Mamba blocks
2. Each: `Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)`
3. Store `self.num_layers`, `self.d_model`, `self.d_state`, `self.expand`

**`forward(self, x, hidden_state=None) → (output, new_hidden_state)`**
1. `B, C, H, W = x.shape`
2. Flatten: `x = x.view(B, H*W, C)` → `(B, L, d_model)` where L=H*W
3. If `hidden_state is None`: process normally (zero initial state)
4. For each Mamba layer: apply block, accumulate output
5. Reshape: `output = output.view(B, C, H, W)`
6. Extract final hidden state (see implementation notes)
7. Return `(output, new_hidden_state)`

**`reset_state(self, batch_size, device) → hidden_state`**
- Returns a dict or list of zero tensors representing the initial hidden state
- Each layer's state shape: `(batch_size, d_model * expand, d_state)` = `(batch_size, 512, 16)`
- Return: list of `num_layers` tensors

---

## Shape Test (Must Pass)

```python
import torch
from models.temporal.mamba_temporal import MambaTemporalModule

module = MambaTemporalModule(d_model=256, d_state=16, num_layers=4).cuda()

# Test at P3 scale
x_p3 = torch.randn(2, 256, 30, 38).cuda()
out, state = module(x_p3, hidden_state=None)

assert out.shape == (2, 256, 30, 38), f"Output shape wrong: {out.shape}"
print(f"P3 shape test: PASS — output {out.shape}")

# Test at P4 scale
x_p4 = torch.randn(2, 256, 15, 19).cuda()
out_p4, state_p4 = module(x_p4, hidden_state=None)
assert out_p4.shape == (2, 256, 15, 19)
print(f"P4 shape test: PASS")

# Test at P5 scale
x_p5 = torch.randn(2, 256, 8, 10).cuda()
out_p5, state_p5 = module(x_p5, hidden_state=None)
assert out_p5.shape == (2, 256, 8, 10)
print(f"P5 shape test: PASS")
```

## State Persistence Test (Critical — Must Pass)

```python
module.eval()
x = torch.randn(2, 256, 30, 38).cuda()

# Window 1: no prior state
out1, state1 = module(x, hidden_state=None)

# Window 2: same input but with state from window 1
out2_with_state, state2 = module(x, hidden_state=state1)

# Window 2: same input but no state (reset)
out2_no_state, _ = module(x, hidden_state=None)

# Output with state SHOULD differ from output without state
# (The temporal context from window 1 affects window 2's output)
diff = (out2_with_state - out2_no_state).abs().mean()
assert diff > 1e-4, f"State has no effect on output! diff={diff:.6f}"
print(f"State persistence test: PASS — state influence: {diff:.4f}")
```

## Gradient Test (Must Pass)

```python
module.train()
x = torch.randn(2, 256, 30, 38, requires_grad=True).cuda()

out, state = module(x, hidden_state=None)
loss = out.sum()
loss.backward()

# Gradients should flow to all Mamba parameters
for name, param in module.named_parameters():
    if param.requires_grad:
        assert param.grad is not None, f"No gradient: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient: {name}"

print("Gradient test: PASS")
```

---

## Mamba Parameters

```python
mamba_params = sum(p.numel() for p in module.parameters()) / 1e6
print(f"Mamba module parameters (4 layers): {mamba_params:.2f}M")
# Expected per layer: roughly d_model*(d_model*expand*2 + d_conv*d_model*expand + ...)
# For d_model=256, d_state=16, expand=2: approximately 0.5-0.8M per layer
# 4 layers ≈ 2-3M
# 3 scales ≈ 6-9M total Mamba parameters
```

---

## Common Implementation Errors

1. **Forgetting to reshape back:** After Mamba processing, you have `(B, H*W, 256)`. You MUST reshape to `(B, 256, H, W)` before passing to the YOLOX head, which expects 2D spatial feature maps.

2. **H and W are swapped:** When you `view(B, H*W, 256)`, make sure you save H and W first. A common error: after processing, you try to `view(B, 256, W, H)` instead of `view(B, 256, H, W)`, silently transposing the feature map.

3. **Not detaching state between windows:** Without `.detach()`, gradients accumulate across the entire sequence history, causing memory explosion. Symptom: GPU OOM after a few batches when it worked fine for the first batch.

4. **Passing the wrong scale's state:** With 3 separate modules (P3, P4, P5), make sure you pass `state_P3` to the P3 module and `state_P4` to the P4 module — not the same state to all three.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| mamba-ssm state passing API doesn't work as expected | Use Approach C (no state passing during training initially); fix before full training |
| NaN losses after a few iterations | State explosion from missing detach — add assert `not torch.isnan(state).any()` after each state update |
| mamba-ssm Triton kernel crash on 5070 Ti | Fall back to pure PyTorch reference scan; flag in Stage 2 |
| State persistence test fails (diff ≈ 0) | State is being zeroed or not passed — check forward() implementation |

---

## Deliverable

`models/temporal/mamba_temporal.py` — passing shape test, state persistence test, and gradient test.

## Success Criteria

All three tests print "PASS." Parameter count is ~0.5–0.8M per layer (2–3M for 4 layers). State influence diff > 1e-4.

---

## Next Stage

→ **Stage 3d: Detection Head Verification** — confirm the existing YOLOX head works with your new FPN outputs.
