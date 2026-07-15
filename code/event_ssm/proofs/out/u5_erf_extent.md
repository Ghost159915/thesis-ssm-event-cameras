# Stage-15 ERF spread σ (RMS spatial spread of gradient mass, input px)

| Backbone | Stage | σ untrained | σ trained | Δ (trained−untrained) |
|---|---|---|---|---|
| ResNet-18 (EventSSM) | 3 | 41.5 | 41.8 | +0.2 |
| ResNet-18 (EventSSM) | 4 | 78.2 | 84.4 | +6.2 |
| BiMamba (PureSSM) | 3 | 51.1 | 85.6 | +34.4 |
| BiMamba (PureSSM) | 4 | 75.4 | 105.5 | +30.1 |

Wider σ = more global spatial context. Compare BiMamba vs ResNet-18 at each stage: the pure-SSM backbone's larger trained σ is the mechanism behind the Stage-15 AP_L gain (44.70 → 47.65).