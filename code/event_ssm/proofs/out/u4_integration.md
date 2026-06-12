# Stage 4 - full-model integration proof

Unmodified RVT `YoloXDetector` (PAFPN + YOLOX head + SimOTA losses) assembled with the
drop-in `ResNetMamba` backbone; one synthetic train step (+backward) and one eval step
(with RVT `RNNStates` detach/reset) on a `(L=5, B=2, 20, 256, 320)` clip.

| metric | value |
|---|---|
| params total | 19.26 M |
| params backbone / fpn / head | 13.51 / 3.86 / 1.89 M |
| train feats[2] (L,B,c,h,w) | (5, 2, 128, 32, 40) |
| train loss (finite) | 27.7334 |
| backbone grad coverage | 91% (all finite; stage-1 temporal unused by FPN) |
| eval output (B, anchors, 5+ncls) | (2, 1680, 7) |
| train step | PASS |
| eval step + state detach/reset | PASS |
