# Stage 6 -- scan equivalence (TBPTT carried-state == full scan)

Carry the detached (conv, ssm) state across two sub-sequences; output must match a single
full scan to kernel precision at every timestep (tolerance 2e-3).

| d_model | max\|diff\| | pass (<2e-3) |
|---|---|---|
| 128 | 2.04e-05 | yes |
| 256 | 1.73e-05 | yes |
| 512 | 2.61e-05 | yes |
