# Spike: trainable cross-clip state for Mamba-1 (mamba-ssm 2.3.2)

Tensor: N=256 (B*H*W), L=5 (time), C=64. Device: NVIDIA GeForce RTX 5070 Ti.

| option | trainable | carries state across clips | notes |
|---|---|---|---|
| alpha_step_loop | False | True | carry_diff=0.010909 |
| gamma_parallel_reset | True | False | carry_diff=- |

## Decision: dual-path (see temporal/_scan.py)
No single stock Mamba call is both trainable and cross-clip-stateful.
- **Training** → trainable parallel scan `mamba(x)` (per-clip zero init).
- **Eval / inference** → stateful `step()` loop (continuous streaming memory; carry_diff=0.011 ✓).

Both compute the same selective-SSM function (parallel == step unrolled from zero state);
they differ only in initial state. This is the standard SSM train/infer pattern.

**Implication for design (updates D5):** cross-clip state is verified at **inference**;
full cross-clip state **during training** (TBPTT parity with S5 baseline) needs a custom
differentiable scan accepting an initial state ("beta") — deferred as an optional
enhancement before Stage 7 full training, not required to build/verify the Stage-3 backbone.
