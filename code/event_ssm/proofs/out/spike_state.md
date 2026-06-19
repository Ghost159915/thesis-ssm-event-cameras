# Spike: trainable cross-clip state for Mamba-1 (mamba-ssm 2.3.2)

Tensor: N=256 (B*H*W), L=5 (time), C=64. Device: NVIDIA GeForce RTX 5070 Ti.

| option | trainable | carries state across clips | notes |
|---|---|---|---|
| alpha_step_loop | False | True | carry_diff=0.010781 |
| gamma_parallel_reset | True | False | carry_diff=- |
