| Model | test/AP | Params (M) | GFLOPs total | (counted) | (analytic) | Full p50 (ms) | Full p95 (ms) | Net p50 (ms) | Hz | fps@B4 | fps@B8 | VRAM inf (MB) | VRAM train (MB) | State (KB/stream) | J/frame | Load (W) | mAP/GFLOP | vs Baseline (lat/Hz/J) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eventssm | 0.462 | 19.184 | 12.714 | 12.485 | 0.229 | 12.094 | 12.143 | 11.478 | 82.685 | 297.500 | 341.602 | 1314.972 | 7261.000 | 147980.002 | 0.425 | 87.340 | 0.036 | 1.40× lat / 1.40× Hz / 1.65× J/frame |
| baseline | 0.477 | 18.192 | 11.056 | 10.026 | 1.030 | 16.934 | 17.085 | 16.334 | 59.052 | 213.750 | 229.251 | 901.633 | 9550.696 | 4800.000 | 0.701 | 92.332 | 0.043 | — |
| puressm | 0.464 | 16.334 | 10.026 | 9.060 | 0.966 | 26.464 | 26.703 | 25.879 | 37.787 | 144.039 | 212.025 | 1308.080 | 10075.557 | 147980.004 | 0.641 | 68.687 | 0.046 | 0.64× lat / 0.64× Hz / 1.10× J/frame |
| spikingssm | 0.462 | 16.335 | 10.027 | 9.061 | 0.966 | 27.187 | 27.490 | 26.652 | 36.783 | 143.616 | 211.259 | 1311.322 | 10079.590 | 149100.004 | 0.552 | 68.325 | 0.046 | 0.62× lat / 0.62× Hz / 1.27× J/frame |

*Latency = bf16 streaming (B=1, L=1, state carried), full pipeline incl. postprocess/NMS (headline) and network-only. fps@B4/B8 are network-only throughput (postprocess excluded); headline Hz is full-pipeline. Energy = differential J/frame vs idle, desktop-GPU proxy. FLOPs = counted (torch.profiler) + an analytic add-on for the work inside custom SSM kernels that the profiler cannot see (projections are profiled, not re-added; split shown).*
