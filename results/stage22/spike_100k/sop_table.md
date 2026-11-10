| Model | Op class | Nonzero rate per stage (min–max over clips) | Spike-fed MACs (M) | Driven ops (M) | Dense MACs (G) | Est. energy (mJ/frame) | Dense twin (mJ/frame) | Saving (%) | Ceiling (%) |
|---|---|---|---|---|---|---|---|---|---|
| SpikingSSM (ours, BiMamba + LIF) [spike_s234] | AC | s2 0.484 (0.467–0.513), s3 0.357 (0.351–0.372), s4 0.391 (0.389–0.394) | 52.43 | 21.74 | 5.002 | 22.787 | 23.007 | 0.96 | 1.04 |

*Operation-count estimate. Driven ops = nonzero rate × spike-fed MACs, priced as accumulates (AC: binary spikes, no residual) or as MACs (graded, analog, residual). Each LIF neuron costs one MAC per frame. Dense twin = the same network priced dense (PureSSM-equivalent). Ceiling = the saving at zero activity. Memory traffic is not modelled; sparse-MAC savings assume zero-skipping hardware. Constants: Horowitz, ISSCC 2014, 45 nm, 32-bit float: MAC 4.6 pJ (3.7 mult + 0.9 add), AC 0.9 pJ; operation-count estimate, memory traffic not modelled.*
