# Stage 6 - smoke health probes (Mamba-2 backbone, d_state=64)

Synthetic real-shaped clips through the assembled `YoloXDetector` (drop-in `ResNetMamba` +
PAFPN + YOLOX head). Temporal blocks are built ONLY on FPN stages 2/3/4 (Finding S8 removed the
dead stage-1 block), so the deterministic grad-flow path spans the WHOLE backbone -- no exclusion.

| test | result | notes |
|---|---|---|
| gradient flow (deterministic path) | PASS | real-missing=0 nan=0 (whole backbone, incl. temporal 2/3/4) |
| SimOTA pos-only head branches w/o grad | 10/30 | expected (data-dependent per-level matching); >=1 alive (PASS) |
| VRAM @ bs1/bs2/bs4 | 1.77/3.33/6.45 GB | target <10GB @ bs4 (d_state=64 > S5) |
| eval step latency | 9.27 +/- 0.66 ms | S5-RVT ~12 ms/window |
| max throughput | 108 Hz | window dt=50ms -> need <50ms |

SimOTA pos-only head w/o grad (10): yolox_head.cls_convs.0.0.conv.weight, yolox_head.cls_convs.0.0.bn.weight, yolox_head.cls_convs.0.0.bn.bias, yolox_head.cls_convs.0.1.conv.weight, yolox_head.cls_convs.0.1.bn.weight, yolox_head.cls_convs.0.1.bn.bias, yolox_head.cls_preds.0.weight, yolox_head.cls_preds.0.bias, yolox_head.reg_preds.0.weight, yolox_head.reg_preds.0.bias
