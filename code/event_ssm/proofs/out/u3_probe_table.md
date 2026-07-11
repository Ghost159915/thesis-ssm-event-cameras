| metric | value | gate | pass |
|---|---|---|---|
| backbone streaming p50 (eager) | 18.09 ms | <= 12.4 ms | ❌ |
| projected pipeline (eager) | 39.5 Hz | >= 51 Hz | ❌ |
| train step peak VRAM (checkpointed) | 8.55 GB | < 16 GB (local_fallback) | ✅ |
| spatial module alone | 15.25 ms | (EventSSM ResNet ref 5.83) | — |
| compiled streaming p50 (secondary) | 3.77 ms (manual_cudagraph) | never a gate | — |
