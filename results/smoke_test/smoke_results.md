# Stage 5 - smoke health probes

Synthetic real-shaped clips through the assembled `YoloXDetector` (drop-in `ResNetMamba` +
PAFPN + YOLOX head). Grad-flow excludes the stage-1 temporal Mamba (`backbone.temporal.0.`),
which the FPN's `in_stages=[2,3,4]` legitimately leaves unused (Finding S8 - documented).

| test | result | notes |
|---|---|---|
| gradient flow (deterministic path) | PASS | real-missing=0 nan=0 (excl. stage-1 temporal[0], unused by FPN) |
| SimOTA pos-only head branches w/o grad | 10 | expected (data-dependent per-level matching), not a fault |
| VRAM @ bs1/bs2/bs4 | 0.92/1.47/2.50 GB | target <10GB @ bs4 |
| eval step latency | 7.89 +/- 0.40 ms | S5-RVT ~12 ms/window |
| max throughput | 127 Hz | window dt=50ms -> need <50ms |

SimOTA pos-only head w/o grad (10): yolox_head.cls_convs.0.0.conv.weight, yolox_head.cls_convs.0.0.bn.weight, yolox_head.cls_convs.0.0.bn.bias, yolox_head.cls_convs.0.1.conv.weight, yolox_head.cls_convs.0.1.bn.weight, yolox_head.cls_convs.0.1.bn.bias, yolox_head.cls_preds.0.weight, yolox_head.cls_preds.0.bias, yolox_head.reg_preds.0.weight, yolox_head.reg_preds.0.bias
