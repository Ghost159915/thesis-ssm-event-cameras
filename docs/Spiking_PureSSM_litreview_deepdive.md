# Deep-Dive Literature Review: A Spiking PureSSMDetector
### From state-space to *spiking* state-space models for event-based object detection

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis** · **Date:** 2026-06-18

> **Purpose.** A rigorous review + design pathway for turning **PureSSMDetector** (pure-SSM spatial + Mamba temporal) into a **spiking** detector ("Spiking-PureSSMDetector" / **SSSMDetector**) deployable on neuromorphic hardware (Intel Loihi). Covers (i) the formal SSM↔SNN mathematical matching, (ii) the structured-spiking-SSM literature, (iii) the event-SNN-detection literature, (iv) the novelty gap, (v) a concrete implementation roadmap from where we are now. Companion to `docs/SSSMDetector_Loihi_opportunity.md`.

---

## 1. Starting point — what PureSSMDetector is

PureSSMDetector (planned model #2) is the **pure-SSM** member of the thesis's controlled-comparison family:

- **Spatial:** `BiMambaBackbone` — bidirectional Mamba over patch tokens replaces the ResNet conv stages (4 stages, dims 64/128/256/512).
- **Temporal:** a **causal Mamba** block interleaved per stage (sequence axis = time, per spatial location), state carried across clips.
- **Neck/Head/Loss:** YOLO-PAFPN → YOLOX head → BCE + IoU loss + SimOTA — reused unmodified.

Every learnable block is an **SSM**. That matters: the route to a spiking model is to make those SSM blocks spiking, and the SSM↔LIF equivalence (below) means this is a *principled reparametrization*, not a redesign.

---

## 2. The formal bridge — SSM ↔ LIF (the math that makes this work)

### 2.1 The discretized structured SSM (what Mamba/S4 compute)

A continuous-time linear state-space model
```
h'(t) = A h(t) + B x(t),    y(t) = C h(t)
```
is discretized with step size Δ (zero-order hold) into the recurrence actually run:
```
h_t = Ā h_{t-1} + B̄ x_t ,   y_t = C h_t
Ā = exp(ΔA),   B̄ = (ΔA)^{-1}(exp(ΔA) − I)·ΔB   ( ≈ ΔB for small Δ / S4D )
```
- **S4D / Mamba** make `A = diag(a_1,…,a_N)` (diagonal) → the recurrence is N independent 1-D scalar recurrences, parallelizable via associative scan.
- **Mamba** additionally makes `Δ, B, C` *input-dependent* (selective): `Δ_t = softplus(Δ_param + Linear(x_t))` — this is the input-dependent timescale that gives the event-rate robustness studied in Stage 9.

### 2.2 The LIF spiking neuron (what an SNN computes)

A current-based Leaky Integrate-and-Fire neuron:
```
u_t = β u_{t-1} + I_t            (leaky integration; β = exp(−Δ/τ_m) ∈ (0,1) )
s_t = H(u_t − θ)                 (spike when membrane crosses threshold θ; H = Heaviside)
u_t ← u_t − θ s_t   (soft reset)  OR  u_t ← u_t (1 − s_t)   (hard reset)
```

### 2.3 The matching (the key insight)

Line up §2.1 and §2.2:

| SSM (Mamba/S4D) | LIF neuron | Correspondence |
|---|---|---|
| state `h_t` (N-dim) | membrane potential `u_t` (1-D) | **the recurrent state = the membrane** |
| `Ā = exp(ΔA)` (diagonal decay) | `β = exp(−Δ/τ_m)` (leak) | **state decay = membrane leak** — *identical form* |
| `B̄ x_t` (input projection) | `I_t` (input current) | input integration |
| `y_t = C h_t` (**linear** readout) | `s_t = H(u_t − θ)` + reset (**nonlinear, stateful**) | **the ONLY real difference** |
| `Δ` (Mamba: input-dependent) | `Δ`/`τ_m` (usually fixed) | timescale |

**Conclusion:** a LIF neuron is a **1-state diagonal SSM with a real leak, whose linear readout is replaced by a threshold-spike-and-reset nonlinearity.** Equivalently, a structured SSM is a **multi-state, trainable-dynamics, optionally-complex LIF without the spike.** Therefore "make the SSM spiking" = **keep the (well-behaved, parallelizable) linear state recurrence and attach a spiking readout**, trained with surrogate gradients. The recurrence — the hard part of SNN temporal modelling — is exactly what your Mamba blocks already do well.

### 2.4 The rigorous version — SiLIF [8] (2025)

SiLIF makes §2.3 exact for the **adaptive** LIF (2-state, with adaptation variable `w`):
```
u_t = α u_{t-1} + (1−α)(I_t − w_{t-1}) − θ s_{t-1}
w_t = β w_{t-1} + a u_{t-1} + b s_{t-1}
s_t = (u_t ≥ θ)
```
which is an SSM with state `x_t = [u_t, w_t]ᵀ` and
```
Ā = [ α    α−1 ]      B̄ = [ 1−α ]     C̄ = [1  0]
    [ a     β  ]           [  0  ]
```
SiLIF then borrows three things directly from S4/S4D to fix SNN training pathologies:
1. **Continuous-domain parameters** `λ^α = 1/τ_u`, `λ^β = 1/τ_w` trained instead of raw decays.
2. **Per-neuron learnable timestep Δ** → a wider range of dynamical regimes per neuron.
3. **Logarithmic reparametrization** `log(λ)` (S4-style) → numerical stability, no vanishing/exploding gradients.
4. **C-SiLIF** uses a *complex* eigenvalue (S4D-Lin init) → **oscillatory** membrane regimes impossible for vanilla LIF.

Result: SiLIF reaches 82.0% on SSC (spiking-SOTA) and **SiLIF+Delays beats S4D with half the synaptic operations** — direct evidence that SSM parametrization *improves* spiking neurons, not just unifies them notationally.

---

## 3. Literature — structured spiking state-space models

The 2024–2025 literature has converged on the §2 insight. Two families:

### 3.1 "SSM block → LIF readout" (the practical recipe) — **SpikingSSMs** [7]
- **Architecture:** an **S4D** block runs in parallel to produce `y_t`, which is the **input current** to a LIF neuron:
  ```
  y_t = C h_t            (S4D, parallel)
  u'_t = τ u_{t-1} + y_t  (LIF membrane)
  s_t = H(u'_t − v_th)    (spike)
  u_t = u'_t (1 − s_t)    (hard reset)
  ```
  framed biologically as a "dendritic neuron" (SSM states = dendrites, LIF = soma).
- **The parallel-training problem & fix:** the reset makes `u_t` depend on past spikes → sequential, kills SSM parallelism. **Surrogate Dynamic Network (SDN):** a tiny 3-layer 1-D conv (<200 params) *predicts the spike train in parallel* (`s_{1:T} = f(I_{1:T})`); membrane potentials are then computed from predicted spikes. **Discarded at inference** (revert to exact iterative spiking). **Training speedup 7.5×→101×** for sequence lengths 1K→8K.
- **Results:** sMNIST 99.6%; Long-Range-Arena 84.30% avg at **~90% sparsity**; WikiText-103 ppl 33.94, beating SpikeGPT at **1/3 the parameters**. Learnable thresholds via input scaling.

### 3.2 "Reparametrize the neuron as an SSM" (the neuron-level family)
- **SiLIF / C-SiLIF** [8] — §2.4; LIF *is* an S4/S4D block.
- **SPikE-SSM** [9] — decomposes the membrane potential into parallel-computable components ("boundary compression") for fast long-sequence training; sparse, precise, efficient.
- **P-SpikeSSM** [10] — "rethinking SNNs as state-space models"; a **probabilistic** spiking SSM with a scalable learning framework for long-range dependencies.
- **Spiking-S4 for speech** [11] — extends an S4 block's output into an LIF layer; ~2 orders-of-magnitude energy reduction vs S4 at comparable quality.

### 3.3 Spiking *selective* SSMs (Mamba-level)
- **SpikeMba** [12] (2024) — "Multi-Modal Spiking Saliency Mamba" for temporal video grounding. **A spiking Mamba exists** — but for video grounding, **not** event-based vision or detection.

**Comparison of structured spiking SSMs**

| Method | SSM core | Where spiking enters | Parallel training | Task evaluated |
|---|---|---|---|---|
| SpikingSSMs [7] | S4D | LIF readout after SSM | **SDN** (predict spikes in parallel) | seq-MNIST, LRA, LM |
| SiLIF [8] | S4/S4D | neuron *is* the SSM (2-state) | surrogate-grad BPTT | speech (SSC/SHD) |
| SPikE-SSM [9] | S4-like | membrane decomposition | boundary compression | long-sequence |
| P-SpikeSSM [10] | SSM (probabilistic) | stochastic spiking state | parallel scan | long-range deps |
| SpikeMba [12] | **Mamba (selective)** | spiking saliency tokens | — | video grounding |

---

## 4. Literature — SNN object detection on event cameras (the task side)

| Model | Type | Gen1 mAP@0.5:0.95 | Timesteps | Notes |
|---|---|---|---|---|
| S5-RVT [1] (your baseline) | ANN-SSM | **47.7** | — | dense, GPU |
| EMS-YOLO [13] (ICCV'23) | SNN | 0.267–0.310 | 4 | first directly-trained SNN detector; EMS-ResNet full-spike residual |
| SpikeYOLO [14] (ECCV'24) | SNN | **0.385** (mAP50 0.672) | 5 | integer-valued training + spike-driven inference (I-LIF) |
| EAS-SNN [16] (ECCV'24) | recurrent SNN | — | adaptive | end-to-end adaptive event sampling |
| **SpikSSD [15] (2025)** | SNN | **0.408 (SNN SOTA)** | — | full-spike MDS-ResNet + bi-directional fusion; lowest firing rate |
| Hybrid Spiking ViT [17] (ICML'25) | hybrid SNN-ViT | — | — | spiking + transformer for event detection |

**Takeaways:** (1) SNN detection on Gen1 is established but trails ANN/SSM detectors by **~7 mAP** (best SNN = SpikSSD 40.8 vs S5-RVT 47.7) — *the gap is the opportunity*. (2) The accuracy-recovery tricks that work are **richer information per spike** (SpikeYOLO integer training), **full-spike residuals** (EMS-YOLO), and **recurrence** (EAS-SNN) — all of which a spiking *SSM* supplies natively.

---

## 5. The novelty gap (positioning)

Cross the two literatures:

- Structured spiking SSMs exist — but on **sequence/speech/LM** tasks ([7]–[11]), **not** event vision.
- A spiking **Mamba** exists (SpikeMba [12]) — but for **video grounding**, not event detection.
- SNN **detectors** on Gen1 exist (EMS-YOLO, SpikeYOLO, SpikSSD) — but their backbones are **spiking CNNs/ViTs, not spiking SSMs**.

> **To the best of this review's knowledge, a spiking *structured/selective* SSM detector for event-based object detection — built as a *controlled spiking-vs-non-spiking swap* from an established SSM detector — is unoccupied.** That intersection is the thesis's contribution, and the controlled-comparison design (EventSSM → PureSSM → **Spiking-PureSSM**, identical neck/head/data/eval) is itself a methodological contribution: it isolates *the exact cost of spiking* an SSM detector, which no existing paper does.

---

## 6. Design — how to make PureSSMDetector spiking

### 6.1 Three architectural choices (increasing ambition)

- **(A) Spiking readout (recommended first):** keep each BiMamba/Mamba block's *linear recurrence* exact (run via `mamba-ssm`), and wrap its **output** in a LIF spiking layer — the SpikingSSMs recipe (§3.1). Inter-block signals become binary spikes → sparse, neuromorphic-friendly. Lowest risk; directly reuses [7]'s SDN for parallel training.
- **(B) Spiking neuron-state (deeper):** replace the SSM blocks with **SiLIF-style** neurons whose state dynamics *are* the structured SSM (§2.4). More faithful to "one model," more novel, more finicky.
- **(C) Hybrid:** spike only the **temporal** Mamba block (the time axis = the natural spike-train axis), keep the **spatial** BiMamba in ANN form first. A clean ablation and a de-risked milestone.

**Recommended path: C → A → (B optional).** Spike the temporal core first (smallest, highest-synergy with the event-rate story), then the spatial backbone, then optionally go full neuron-level.

### 6.2 Key design decisions
- **Time axis / timesteps `T`:** the temporal Mamba already unrolls over clip time; reuse that as the spiking time axis (no extra `T` tax). For the spatial backbone, pick small `T` (4–5, per EMS-YOLO/SpikeYOLO) to preserve efficiency.
- **Spike encoding of input:** Option 1 — keep the 20-ch stacked-histogram, spike-encode it (least disruptive, keeps the pipeline). Option 2 (more neuromorphic) — feed events as spikes directly. Start with Option 1.
- **Surrogate gradient:** arctan or fast-sigmoid (standard, robust). Train with BPTT over `T`.
- **Stability tricks (from the lit):** S4D-Lin / HiPPO init for the SSM state matrices; log-reparametrized continuous decays (SiLIF); threshold-dependent BatchNorm; learnable thresholds via input scaling (SpikingSSMs).
- **Parallel training:** adopt the **SDN** trick [7] to keep training throughput viable on the long event sequences.
- **Information bottleneck mitigation:** integer-valued training / graded spikes (SpikeYOLO [14]; Loihi-2 graded spikes) to narrow the §4 accuracy gap.

### 6.3 Frameworks
- **snnTorch** or **SpikingJelly** for LIF neurons + surrogate gradients (PyTorch-native, drop into the existing `nn.Module` backbone).
- **`mamba-ssm`** (already built for `sm_120`) for the SSM recurrence — wrap its output with the spiking layer.
- **Lava-DL** + **NIR** [19] for the eventual Loihi export (PoC artefact for the INRC application).

---

## 7. Implementation roadmap (where we are → there)

Mirrors the existing stage system; **everything downstream of the backbone is reused unmodified**, exactly as for EventSSM/PureSSM.

| Step | Deliverable | Reuses | Risk |
|---|---|---|---|
| **S-0 Neuron validation** | A `SpikingSSMBlock` (SSM recurrence + LIF readout + surrogate grad); unit-test forward/backward, firing-rate sanity, gradient flow | mamba-ssm + snnTorch | low |
| **S-1 Spiking temporal core** (choice C) | swap the temporal Mamba block for `SpikingSSMBlock`; verify it trains on a clip | Stage-3/4 backbone, smoke harness | low-med |
| **S-2 Spiking spatial backbone** (choice A) | spike the BiMamba stages; integrate → PAFPN → YOLOX | full pipeline | med |
| **S-3 Smoke** | overfit one real Gen1 batch (as Stage 5) | Stage-5 harness | low |
| **S-4 Train** | short → full Gen1 run; record val/AP + **firing rate / spike sparsity** | stage6/7 launchers | med |
| **S-5 Eval** | Gen1 **test** mAP vs EventSSM/PureSSM/SpikeYOLO/S5-RVT | stage8 eval | low |
| **S-6 Efficiency** | params, **SOPs** (synaptic ops, not MACs), estimated energy, sparsity — the SNN's whole point | Stage-10 methodology | med |
| **S-7 Loihi PoC** | export via NIR → Lava-DL; software-sim demo (the INRC artefact) | — | med-high |

**Discipline:** lock EventSSM (Stage 8) + PureSSM first; build S-0…S-3 *while the current 400k run trains* (CPU/dev work, minimal GPU). The SNN is the high-upside extension — never a load-bearing dependency.

---

## 8. Risks & open problems (honest)
1. **Binary information bottleneck** (§ accuracy gap) — *fundamental*; mitigate with integer/graded spikes, not eliminate.
2. **Surrogate-gradient mismatch over depth** — pick robust surrogates; SiLIF-style reparametrization helps.
3. **Spatial bidirectional spiking** — BiMamba's non-causal spatial scan is awkward to spike (spikes are causal-in-time, but the spatial scan isn't a time axis); choice (C) sidesteps this initially.
4. **Parallel training of reset dynamics** — solved in principle by the SDN [7], but adds an auxiliary network to implement.
5. **Energy-claim methodology** — SNN energy advantages must be measured as **SOPs/sparsity** (addition-dominated, event-driven), with a clearly stated ANN-MAC vs SNN-SOP model; sloppy energy claims are a common reviewer target.
6. **Timestep–accuracy–latency trilemma** — keep `T` small; report the trade-off curve rather than a single point.

---

## 9. References
- [1] N. Zubić *et al.*, "State Space Models for Event Cameras," *CVPR*, 2024. arXiv:2402.15584.
- [2] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023. arXiv:2312.00752.
- [3] A. Gu *et al.*, "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), *ICLR*, 2022. arXiv:2111.00396.
- [4] A. Gu *et al.*, "On the Parameterization and Initialization of Diagonal State Space Models" (S4D), *NeurIPS*, 2022. arXiv:2206.11893.
- [5] J. T. H. Smith *et al.*, "Simplified State Space Layers for Sequence Modeling" (S5), *ICLR*, 2023. arXiv:2208.04933.
- [6] E. O. Neftci, H. Mostafa, F. Zenke, "Surrogate Gradient Learning in Spiking Neural Networks," *IEEE Signal Processing Magazine*, 2019. arXiv:1901.09948.
- [7] "SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models," 2024. arXiv:2408.14909.
- [8] "SiLIF: Structured State Space Model Dynamics and Parametrization for Spiking Neural Networks," 2025. arXiv:2506.06374.
- [9] Y. Zhong *et al.*, "SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning," 2024. arXiv:2410.17268.
- [10] M. Bal and A. Sengupta, "P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks," 2024. arXiv:2406.02923.
- [11] "Spiking Structured State Space Model for Monaural Speech Enhancement," 2023. arXiv:2309.03641.
- [12] "SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding," 2024. arXiv:2404.01174 *(verify)*.
- [13] Q. Su *et al.*, "Deep Directly-Trained Spiking Neural Networks for Object Detection" (EMS-YOLO), *ICCV*, 2023. arXiv:2307.11411.
- [14] X. Luo *et al.*, "Integer-Valued Training and Spike-Driven Inference Spiking Neural Network..." (SpikeYOLO), *ECCV*, 2024. arXiv:2407.20708.
- [15] Y. Fan *et al.*, "SpikSSD: Better Extraction and Fusion for Spiking Object Detection," 2025 *(verify arXiv id)*.
- [16] "EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks," *ECCV*, 2024. arXiv:2403.12574.
- [17] "Hybrid Spiking Vision Transformer for Object Detection with Event Cameras," *ICML*, 2025. arXiv:2505.07715.
- [18] Intel Corporation, "Intel Advances Neuromorphic with Loihi 2, New Lava Software Framework," 2021.
- [19] J. Pedersen *et al.*, "Neuromorphic Intermediate Representation," 2023. arXiv:2311.14641.

> **BibTeX note (thesis standards):** verify author lists + arXiv IDs for [7],[12],[14],[15] before entry; protect acronyms — `{SiLIF}`, `{SNN}`, `{SSM}`, `{LIF}`, `{S4D}`, `{Loihi}`, `{Mamba}`.

---

## Appendix A — Temporal-block spiking design (the concrete first build target)

The temporal Mamba block is the **highest-synergy, lowest-risk** place to introduce spiking (choice C, §6.1), for one elegant reason developed below: **the event clip already has a real time axis, so spiking over it is natural — no artificial timesteps.**

### A.1 Current (non-spiking) temporal block

At backbone stage *s*, for each spatial location `(h,w)`, the block processes a clip-time sequence `z_1…z_T`, `z_t ∈ ℝ^d` (d = stage dim 64/128/256/512), with a **causal selective SSM (Mamba)**, carrying state across clips via the `LstmStates` contract:
```
y_t, h_t = Mamba(z_t , h_{t-1})        # continuous SSM output y_t, hidden state h_t
```

### A.2 Spiking temporal block — `SpikingMambaTemporal` (SSM core + LIF readout)

Keep the **exact** Mamba recurrence (runs on your existing `mamba-ssm` kernel, stays real-valued/differentiable) and attach a **LIF spiking readout over the same clip-time axis**:
```
y_t , h_t = Mamba(z_t , h_{t-1})          # 1) SSM core — unchanged, continuous
u_t = λ ⊙ u_{t-1} + y_t                    # 2) membrane integrates the SSM output (λ = exp(−Δ/τ), learnable/channel, log-param)
s_t = Θ(u_t − θ)                           # 3) spike (Θ = surrogate Heaviside; arctan/fast-sigmoid in backward)
u_t ← u_t − θ ⊙ s_t                        # 4) soft reset (subtractive — preserves residual potential → better accuracy than hard reset)
```
- **Output:** binary spikes `s_t` flow to the next spiking block (sparse, neuromorphic-friendly).
- **State contract:** carry `(h_t, u_t)` across clips — one extra tensor (the membrane `u`) added to the existing `LstmStates` tuple; otherwise mirrors `RNNDetectorStage` exactly.
- **Trainable dynamics (from SiLIF [8]):** per-channel `λ` (membrane decay) and `θ` (threshold) as log-reparametrized learnable parameters; optionally a complex `λ` for oscillatory regimes.

### A.3 The interface to the (non-spiking) YOLOX head — analog readout

The PAFPN neck + YOLOX head are reused **unmodified** and expect *continuous* features. So the backbone is **spiking internally** but must **decode back to analog at its output** (the StereoSpike "analog readout from spikes" paradigm [1, §2.2]):
```
feat_out = Σ_t w_t · s_t            # learned weighted spike-rate decode  (or: final membrane potential u_T)
```
This keeps the controlled-comparison clean: *only the backbone internals change*; neck/head/loss/data/eval are identical to PureSSM.

### A.4 Why the temporal axis is the *natural* spike axis (the key idea)

A standard image-SNN invents `T` artificial timesteps and feeds the **same static input `T` times** (rate coding over fake time) — wasteful, latency-adding. Here the clip's **real** time axis `t=1…T` *is* the membrane integration axis. Each step is a distinct real moment of event data, so:
- the membrane integrates **genuine temporal information** (no repeated-input waste),
- a spike is **semantically meaningful** ("this location changed at this real time"),
- it composes directly with the **Stage-9 event-rate study** (Mamba's input-dependent `Δ_t` already adapts the membrane time-constant to event rate).

### A.5 Training & first milestone
- **Surrogate gradient** on `Θ`; **BPTT over the clip** (your existing TBPTT already does this — the Mamba part is already differentiable, only the spike readout needs the surrogate).
- **Spike-vanishing guard:** membrane shortcut / full-spike residual (EMS-ResNet style [13]) to keep firing rates healthy across depth.
- **Parallel-training option:** adopt the **SDN** [7] if BPTT-over-`T` is too slow on long clips.
- **First build target (Stage S-1):** `SpikingMambaTemporal = Mamba + LIFReadout`; unit-test forward/backward, check firing rate lands ~10–30% and gradients flow; swap it into **one** temporal slot of PureSSM and overfit a single real Gen1 clip. Smallest possible concrete step.

---

## Appendix B — Cross-hardware comparison: is an NPU model even comparable to a GPU model?

**Yes — if you compare on hardware-*independent* axes and treat real-chip numbers as a separate deployment demonstration.** This is exactly how the SNN-detection literature (EMS-YOLO [13], SpikeYOLO [14]) does it.

| Axis | Metric | Hardware-independent? | Measured where |
|---|---|---|---|
| **Accuracy** | Gen1 **test mAP** | **Yes** | all 3 models simulated on **your GPU**, identical protocol |
| **Size** | # parameters | **Yes** | static count |
| **Compute** | ANN: **MACs/FLOPs** (dense) · SNN: **SOPs** = Σ(spikes × fan-out) (sparse) | **Yes (theoretical)** | analytic count × measured firing rate |
| **Energy (est.)** | `E_ANN = MACs·E_MAC` vs `E_SNN = SOPs·E_AC` | **Yes** | 45 nm estimates (Horowitz): `E_MAC ≈ 4.6 pJ`, `E_AC ≈ 0.9 pJ` |
| **Activity** | firing rate / sparsity | **Yes** | measured on test set |
| **Deployment (separate)** | Loihi **power (mW)**, **latency**, throughput | **No** — chip-specific | measured on Loihi *once granted* (or Lava-sim estimate) |

**The principles that make it fair:**
1. **mAP is a property of the model, not the chip.** The spiking model's Gen1 mAP is the *same number* whether simulated on your GPU or run on Loihi (modulo fixed-point precision). So the three-way accuracy comparison needs **no hardware**.
2. **Never compare GPU-milliseconds to Loihi-milliseconds** — different process nodes, maturity, clocking; meaningless. For the *efficiency* comparison use **theoretical energy** (MACs vs SOPs × per-op energy), which is the standard, hardware-independent currency. The SNN's advantage falls out of **sparsity** (only spiking neurons compute) × **cheaper op** (addition `E_AC` vs multiply-accumulate `E_MAC`).
3. **The Loihi run is a bonus, not the benchmark.** Report on-chip power/latency as evidence of *deployability*, in its own row — not as the head-to-head metric against the GPU models.

**Resulting thesis table** (all GPU-measurable): EventSSM / PureSSM / **Spiking-PureSSM** × {mAP, params, MACs-or-SOPs, est. energy, sparsity}; plus a separate "on Loihi" row for the spiking model if/when hardware is granted.

**Rigor caveat:** the theoretical-energy model is an *estimate* (assumes a process node, often omits memory-movement energy). State the assumptions explicitly; treat real on-chip measurement as the gold standard — which is precisely what the Loihi PoC buys you.

---

## Appendix C — Building the PoC *without* the hardware (the chicken-and-egg)

The apparent paradox — "need Loihi to demo Loihi, need a demo to get Loihi" — dissolves because **the PoC is software**. No chip is required to build a convincing INRC application.

1. **Train on GPU / Katana.** Surrogate-gradient SNN training is ordinary CUDA training (snnTorch / SpikingJelly are PyTorch; your `mamba-ssm` kernels already target `sm_120`). Output = a trained spiking SSM detector.
2. **Measure everything hardware-independent on the GPU** (Appendix B): Gen1 mAP, SOPs, estimated energy, firing-rate/sparsity.
3. **Implement in Lava and run the Lava *software simulator*.** Lava's simulation backend emulates Loihi's fixed-point neuron dynamics on CPU/GPU — this **demonstrates Loihi-compatibility without the chip**.
4. **Optionally export via NIR [19]** to show the model maps onto neuromorphic primitives.
5. **PoC deliverable** = *"a trained spiking SSM event-detector achieving X mAP on Gen1 at Y% sparsity / Z estimated energy, implemented in Lava and validated in Loihi software simulation, ready to deploy."* → a credible INRC request.
6. **Granted hardware → final on-chip power/latency** (the thesis deployment chapter / future work).

**Training feasibility (honest):** BPTT over `T` timesteps multiplies activation memory ≈ `T×`. On the 16 GB RTX 5070 Ti this is fine for small `T` (4–5) — likely with a smaller batch than the ANN run. **Katana** is better for the heavier runs but needs the `mamba-ssm`/`causal-conv1d` wheels rebuilt for the cluster GPU arch + snnTorch installed (same deferred Katana env work as the ANN models). Plan: prototype + train on the 5070 Ti; scale on Katana once the env is reproduced.
