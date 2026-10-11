| Model | test/AP | Params (M) | GFLOPs total | (counted) | (analytic) | Full p50 (ms) | Full p95 (ms) | Net p50 (ms) | Hz | fps@B4 | fps@B8 | VRAM inf (MB) | VRAM train (MB) | State (KB/stream) | J/frame | Load (W) | mAP/GFLOP | vs Baseline (lat/Hz/J) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eventssm | 0.462 | 19.184 | 12.714 | 12.485 | 0.229 | 12.600 | 13.399 | 11.657 | 79.362 | 297.354 | 341.502 | 1628.376 | 7574.374 | 147980.002 | 0.438 | 86.072 | 0.036 | 1.36× lat / 1.36× Hz / 1.65× J/frame |
| baseline | 0.477 | 18.192 | 11.056 | 10.026 | 1.030 | 17.145 | 17.295 | 16.536 | 58.327 | 213.490 | 229.553 | 918.633 | 9571.821 | 4800.000 | 0.723 | 90.567 | 0.043 | — |
| puressm | 0.464 | 16.334 | 10.026 | 9.060 | 0.966 | 26.862 | 27.150 | 26.223 | 37.227 | 142.855 | 210.624 | 1630.128 | 10396.291 | 147980.004 | 0.652 | 67.513 | 0.046 | 0.64× lat / 0.64× Hz / 1.11× J/frame |

*Latency = bf16 streaming (B=1, L=1, state carried), full pipeline incl. postprocess/NMS (headline) and network-only. fps@B4/B8 are network-only throughput (postprocess excluded); headline Hz is full-pipeline. Energy = differential J/frame vs idle, desktop-GPU proxy. FLOPs = counted (torch.profiler) + an analytic add-on for the work inside custom SSM kernels that the profiler cannot see (projections are profiled, not re-added; split shown).*
