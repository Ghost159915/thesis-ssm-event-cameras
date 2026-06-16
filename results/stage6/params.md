# Stage 6 - parameter counts (Mamba-2 backbone, d_state=64)

Temporal blocks on FPN stages 2/3/4 only (Finding S8). Compare to Stage-5 (Mamba-1,
d_state=16, *with* the now-removed dead stage-1 temporal block): ~19.26M total.

| component | params | % of total |
|---|---|---|
| backbone.spatial | 11.230M | 58.5% |
| backbone.temporal | 2.203M | 11.5% |
| fpn | 3.861M | 20.1% |
| yolox_head | 1.891M | 9.9% |
| **total** | **19.184M** | 100% |
