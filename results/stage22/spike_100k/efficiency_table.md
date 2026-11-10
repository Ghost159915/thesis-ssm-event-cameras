| Model | test/AP | Params (M) | GFLOPs total | (counted) | (analytic) | Full p50 (ms) | Full p95 (ms) | Net p50 (ms) | Hz | fps@B4 | fps@B8 | VRAM inf (MB) | VRAM train (MB) | State (KB/stream) | J/frame | Load (W) | mAP/GFLOP | vs Baseline (lat/Hz/J) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spikingssm | 0.402 | 16.335 | 10.027 | 9.061 | 0.966 | 27.519 | 27.786 | 26.865 | 36.339 | 142.734 | 210.633 | 1301.104 | 10075.488 | 149100.004 | 0.714 | 65.216 | 0.040 | — |

*Latency = bf16 streaming (B=1, L=1, state carried), full pipeline incl. postprocess/NMS (headline) and network-only. fps@B4/B8 are network-only throughput (postprocess excluded); headline Hz is full-pipeline. Energy = differential J/frame vs idle, desktop-GPU proxy. FLOPs = counted (torch.profiler) + an analytic add-on for the work inside custom SSM kernels that the profiler cannot see (projections are profiled, not re-added; split shown).*
