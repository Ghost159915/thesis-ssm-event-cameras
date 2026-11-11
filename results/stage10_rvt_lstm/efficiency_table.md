| Model | test/AP | Params (M) | GFLOPs total | (counted) | (analytic) | Full p50 (ms) | Full p95 (ms) | Net p50 (ms) | Hz | fps@B4 | fps@B8 | VRAM inf (MB) | VRAM train (MB) | State (KB/stream) | J/frame | Load (W) | mAP/GFLOP | vs Baseline (lat/Hz/J) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eventssm | 0.462 | 19.184 | 12.714 | 12.485 | 0.229 | 12.201 | 12.259 | 11.585 | 81.959 | 297.269 | 341.590 | 1331.222 | 7277.250 | 147980.002 | 0.429 | 87.005 | 0.036 | 1.40× lat / 1.40× Hz / 1.68× J/frame |
| baseline | 0.477 | 18.192 | 11.056 | 10.026 | 1.030 | 17.139 | 17.381 | 16.574 | 58.346 | 213.379 | 229.255 | 2278.847 | 10930.160 | 4800.000 | 0.722 | 92.418 | 0.043 | — |
| puressm | 0.464 | 16.334 | 10.026 | 9.060 | 0.966 | 26.960 | 27.182 | 26.386 | 37.092 | 142.011 | 209.936 | 1332.939 | 10098.801 | 147980.004 | 0.663 | 68.780 | 0.046 | 0.64× lat / 0.64× Hz / 1.09× J/frame |
| rvt_lstm | 0.471 | 18.536 | 10.528 | 10.528 | 0.000 | 13.173 | 13.286 | 12.569 | 75.915 | 323.754 | 651.187 | 619.091 | 3263.634 | 4800.000 | 0.169 | 61.363 | 0.045 | 1.30× lat / 1.30× Hz / 4.28× J/frame |

*Latency = bf16 streaming (B=1, L=1, state carried), full pipeline incl. postprocess/NMS (headline) and network-only. fps@B4/B8 are network-only throughput (postprocess excluded); headline Hz is full-pipeline. Energy = differential J/frame vs idle, desktop-GPU proxy. FLOPs = counted (torch.profiler) + an analytic add-on for the work inside custom SSM kernels that the profiler cannot see (projections are profiled, not re-added; split shown).*
