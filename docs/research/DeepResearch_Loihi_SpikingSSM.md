# Deep Research: Intel Loihi 2 + Spiking State-Space Models for Event-Based Vision

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis**
**Compiled:** 2026-06-20 · **Scope:** A citation-backed, critically-analysed survey of (1) Intel Loihi 2 + the Lava SDK as a deployment target, (2) the spiking state-space model (SSSM) research landscape, (3) concrete implementation paths, and (4) a de-risked PoC recommendation for extending the existing `EventSSMDetector` (Gen1 test mAP 46.2, COCO 0.50:0.95) toward neuromorphic hardware.

**Source grading used throughout:** **[PR]** peer-reviewed (named venue) · **[WS]** workshop · **[PRE]** preprint (arXiv, no confirmed venue) · **[V]** vendor / marketing · **[W]** community/secondary reference. Every accuracy/energy figure is annotated *measured* vs *theoretical (operation-count estimate)*.

**Companion docs:** `docs/research/SSSMDetector_Loihi_opportunity.md` (strategic options), `docs/research/Spiking_PureSSM_litreview_deepdive.md` (math deep-dive), `docs/research/spiking_ssm_references.md` (master `.bib` staging). This report **supersedes** the Gen1 spiking-detection numbers in those docs where they conflict — see the metric-discipline warning in §5.1.

---

## Executive Summary

The strategic question is whether a **spiking state-space model (SSSM)** is a defensible bridge from the thesis's existing Mamba-based event detector to Intel Loihi 2 / the INRC. The answer is a **qualified yes**, with the qualifications being load-bearing:

1. **The SSM↔spiking-neuron bridge is now formal and citable, not hand-waved.** A diagonal SSM (S4D/S5) is mathematically a bank of independent first-order linear recurrences; each **real** mode equals a leaky-integrate-and-fire (LIF) leak, each **complex** mode equals a resonate-and-fire (RF) oscillatory neuron. This is stated explicitly in SiLIF [4], the MIMO-spiking-neuron paper [5], Huber et al. [13], and operationalised on silicon. The LIF neuron is the **1-D, scalar, real-eigenvalue special case** of a structured SSM.

2. **A non-selective diagonal SSM has actually run on Loihi 2 silicon — S5 and Mamba have not.** Intel's Meyer et al. [25] deployed **S4D** on Loihi 2, reporting ~1000× energy / ~75× latency advantage over a *recurrent* (streaming) S4D baseline on a Jetson Orin Nano — but only in the token-by-token regime; the GPU wins when batched. **No S5, and no selective/Mamba SSM, has ever run on Loihi.** This makes "S5-on-Loihi" and "a spiking SSM for event *detection* (vs classification)" two genuine, in-reach firsts.

3. **Mamba's selectivity is the hard wall, and it is well-documented.** Input-dependent Δt (the S6 contribution) is a continuous, per-step, content-computed multiplicative gate. It defeats event-driven sparsity, requires a dense per-step projection, makes the recurrence time-varying (so the convolution / clean train-parallel-deploy-sequential equivalence is lost), and quantises catastrophically (PTQ fails; QAT required) [37]. **Every published "spiking Mamba" either keeps the selective core in floating point on GPU or drops selectivity entirely.** The de-risked target is therefore a **spiking diagonal S5 with fixed (learned) per-channel Δt** — which deliberately drops selectivity.

4. **Object detection on Loihi is now demonstrated but the accuracy collapses, and almost all "neuromorphic detection" energy numbers are theoretical.** Gamage et al. [27] (*Neurocomputing* 2026) is the first multi-class bounding-box SNN detector on Loihi 2 silicon; on Gen1 it reaches ~0.42 mAP@0.5 (SNN) vs ~0.45 (ANN) — but this is mAP@**0.5**, and at COCO-style mAP@0.5:0.95 it is roughly **0.19–0.22**, less than half the thesis's S5-RVT (47.7). Every GPU-only spiking detector (SFOD, EAS-SNN, SpikeYOLO) reports *operation-count-estimated* energy, never chip-measured.

5. **Two practicality landmines.** (a) The canonical `lava-nc` repos (lava, lava-dl) are **archived / read-only** as of mid-2026, frozen at the Aug-2024 releases (lava 0.10.0, lava-dl 0.6.0); Intel signals an undisclosed next-gen SDK. (b) INRC has **no individual-student access path** — membership is institutional and the realistic route is a faculty PI; the public application survey was reported intermittently broken (Jan 2026); export-control posture for non-US institutions is unpublished.

**Recommended PoC path (detail in §6):** (a) build an *algorithmic* spiking diagonal-SSM block on GPU first (quantise/graded-spike the current backbone's temporal core, fixed Δt, surrogate-gradient + QAT, report accuracy + an op-count energy proxy + firing rate, validated in the open Lava CPU simulator); then (b) port a **single** spiking-SSM block to Lava-DL as the Loihi-readiness artefact for the INRC application. Keep the SNN strictly **high-upside, never load-bearing** — lock `EventSSMDetector` and `PureSSMDetector` first.

**The single most defensible critical claim of the whole subfield:** essentially every spiking-SSM "energy win" in the literature is a theoretical AC-vs-MAC operation count at assumed CMOS pJ, *not* silicon-measured. The only measured-hardware SSM results are Meyer et al. (Loihi 2, S4D) [25] and a compute-in-memory RRAM SSM (Nature Communications 2026) [26] — and the latter is **non-spiking**. That measured-vs-theoretical gap is precisely what a Loihi PoC would begin to close, which is the cleanest motivation for the supervisor's INRC suggestion.

---

## 1. Intel Loihi 2 + the Lava SDK (deployment reality)

### 1.1 Loihi 2 hardware relevant to a recurrent SSM-like model

Loihi 1 had a single hard-wired generalised-LIF neuron. **Loihi 2's headline change for an SSM is that the neuron model is fully programmable via a per-core microcode engine, plus native graded (integer) spikes** — together these are what make a real-valued linear recurrence expressible. [V, Intel Loihi 2 technology brief, 2021] [PR, Davies et al., *Proc. IEEE* 2021] [W, Open Neuromorphic "A Look at Loihi 2"].

**Neuron models.** Native/demonstrated: LIF, adaptive-LIF (with after-hyperpolarisation), **resonate-and-fire (RF)** (complex two-state oscillatory neuron `z[t]=λ·e^{iω}·z[t−1]+a`, λ≤1), sigma-delta (Δ-Σ) encapsulated neurons, and multi-compartment / dendritic neurons; Izhikevich via third-party microcode. **The RF neuron is the most SSM-relevant native primitive: a complex diagonal linear recurrence structurally identical to one S4D mode** [PR, Shrestha et al., ICASSP 2024, arXiv:2310.03251] [PRE, Orchard et al., arXiv:2111.03746].

**Programmable microcode engine (the instruction set, from the Intel brief, [V]):** read-modify-write / read-clear state (`RMW`, `RDC`); move / conditional move (`MOV`, `SEL`); bitwise (`AND`, `OR`, `SHL`); integer arithmetic (`ADD`, `NEG`, `MIN`); **fixed-point multiply-shift (`MUL_SHR`) — there is NO floating-point unit**; comparisons (`LT`, `GE`, `EQ`); conditional skip/jump (`SKP_C`, `JMP_C`, i.e. branching); `SPIKE`, `PROBE`. So a neuron model is a short integer/fixed-point program with branching — and nothing more. The microcode program-size ceiling is not published; it is bounded by the shared per-core SRAM.

**Graded / integer spikes.** Loihi 1 = binary only. Loihi 2 adds graded spikes. The bit-widths conflict in secondary sources; the resolution is: the spike *packet payload field* is 32-bit [V], but the **usable signed integer magnitude is 24-bit** per the peer-reviewed silicon characterisation [PR, arXiv:2310.03251] and the S4D-on-Loihi port (which uses **24-bit spike messages and 24-bit neuron state**) [WS, arXiv:2409.15022]. **Cite 24-bit as the deployable magnitude.** Graded spikes are the enabler for SSMs: the `B̄u` / `C̄x` real-valued increments ride on integer-valued events instead of being rate-coded across many binary spikes. Sigma-delta coding delta-encodes only supra-threshold changes (measured: 12× fewer synaptic ops, 11× fewer activations on PilotNet) [PR, arXiv:2310.03251].

**On-chip learning.** Loihi 1: trace-based local rules + one global reward third factor. Loihi 2 adds **localised three-factor learning** (per-neuron "third factors" mapping to synaptic rules, intended for on-chip backprop *approximations*) — still trace-based and local; **no native global backpropagation** [PR, Davies et al. 2021]. Not turnkey for an SSM; for a PoC, train off-chip and deploy inference.

**Hard numbers (Loihi 2 | Loihi 1):**

| Metric | Loihi 2 | Loihi 1 |
|---|---|---|
| Process | Intel 4 (pre-production) | 14 nm |
| Die area | ~31 mm² | ~60 mm² |
| Neuro-cores / chip | 128 | 128 |
| Embedded x86 cores | 6 | 3 |
| Max neurons / chip | ~1 million | ~130,000 |
| Neurons / core | up to 8192 | ~1024 |
| Max synapses / chip | ~120 million | ~130 million |
| SRAM / core | ~192 KB (soft-partitioned) | 208 KB (fixed) |
| Neuron-state allocation | 0–4096 B/neuron (variable) | fixed 24 B |
| Weight precision | ~1–8 bit typical | 1–9 bit |
| Graded spike | 24-bit magnitude (32-bit packet) | 1-bit binary |
| Scaling | up to 16,384 chips (Hala Point = 1152) | Pohoiki Springs = 768 |

**Caveats on the numbers (do not over-quote):** (a) "1 M neurons" and "120 M synapses" are *non-coexisting maxima* — both draw from the same 192 KB/core SRAM. (b) The "up to 10× faster than Loihi 1" headline is a per-operation decomposition (≈2× state update / 5× synaptic / 10× spike-gen), partly from pre-silicon simulation — not a workload-level 10×. (c) Hala Point (Sandia, 2024): 1152 Loihi 2 chips, 1.15 B neurons, ≤2600 W — but is not a self-service resource.

### 1.2 What is and isn't deployable for an SSM

**Deployable (grounded in Intel's own S4D port [25]):** a **diagonal, LTI (non-selective)** SSM. Because the state matrix is diagonal, every state dimension is an *independent* self-recurrent programmable neuron — no cross-neuron recurrent synapses are needed. Real-valued increments are carried by graded spikes. The discretised `Ā, B̄, C̄` are stored as fixed quantised parameters.

**Not deployable / breaks:**
1. **No native floating point** — everything must be quantised. Intel used 8-bit weights, 24-bit state/messages, a 16-bit descaling factor, with quantisation-aware fine-tuning (QAFT). PTQ alone dropped sequential-CIFAR 86.5→71.7; QAFT recovered to 84.1 (still ~2.4 pts under FP32). The recurrent `A` matrix degrades hardest below 8-bit (corroborated by Q-S5, arXiv:2406.09477; QS4D, arXiv:2507.06079 — the latter targets analog CIM, not Loihi).
2. **No large dense matmul** — cores are connectivity engines, not GEMM. A dense (non-diagonal) state transition has no efficient mapping.
3. **No input-dependent continuous gating → Mamba selectivity is the blocker** (see §3.2).
4. **Even the successful S4D port was simplified** — GLU/GeLU replaced by ReLU "to maintain sparsity," and normalisation + residual connections removed. LayerNorm/RMSNorm/softplus/exp/complex gating are not free.
5. **Footprint** — S4D fit on one chip (31 cores small / 111 cores large; 67k–265k params). A detection-scale backbone spans multiple chips, where inter-chip links bottleneck.

**Where Loihi wins (the honest balance):** token-by-token *streaming* (batch-1, online) — exactly the event-camera/UAV regime. In batched/offline mode the GPU wins. The Loihi advantage is strictly the low-latency, low-power, online inference regime.

### 1.3 The Lava software framework + lava-dl

- **Lava core (magma):** `Process` (backend-agnostic state/ports/API — the *what*) vs `ProcessModel` (per-backend behaviour — the *how*); a `RunConfig` selects which model compiles. CPU sim via `Loihi2SimCfg` (fully open, NumPy, no hardware); silicon via `Loihi2HwCfg` (INRC-gated NxCore/NxKernel). **Switching sim↔silicon is a RunConfig change, no retraining.**
- **lava-dl:** **SLAYER** (direct backprop-through-time with surrogate gradients, PyTorch); **Bootstrap** (accelerated ANN→SNN); **Netx** (imports a trained net from platform-independent **HDF5** as Lava Processes). Workflow: define blocks → train (SLAYER/Bootstrap) → `export_hdf5()` → `netx.hdf5.Network(...)` → run on chosen backend.
- **How an SSM recurrence maps:** the S4D linear recurrence `x_t = Ā x_{t−1} + B̄ u_t` is implemented as **programmable stateful neurons (not sigma-delta)** [25]; because `Ā` is diagonal, each state dimension is one self-recurrent programmable neuron. **Note:** the S4D port used an *alpha* NxKernel API (NxCore 2.5.8) for the custom neurons, so a faithful reproduction likely needs some INRC-gated custom microcode, not pure `netx`.
- **License & maturity:** lava core BSD-3-Clause (low-level HW-mapping magma reported LGPL-2.1); lava-dl / dnf / optimization BSD-3; the Loihi compiler/runtime is proprietary, INRC-gated. **Both lava and lava-dl are archived read-only (mid-2026), frozen at 0.10.0 / 0.6.0 (Aug 2024).** Expect torch/CUDA-compat friction (akin to the Blackwell `sm_120` porting already done in this project). Do not plan around the announced-but-unreleased "next-gen SDK."

### 1.4 INRC access

- **Tiers:** **Research Member (PI)** — technical, gets hardware, needs a proposal and to be "a permanent employee of an established research organization"; **Affiliate** — community/non-technical, explicitly **no hardware**.
- **No individual-student pathway.** A final-year student is not a "permanent employee"; the participation agreement is **institutional (signed by the organisation, with mutual NDA), not personal**. Realistic route: be added by a faculty PI/supervisor who holds the membership. Universities qualify; academic members get a **free Loihi 2 loan up to 1 year**.
- **Access mechanism:** primarily the **Neuromorphic Research Cloud ("vLab")** via SSH (systems: Oheo Gulch = 1× Loihi 2 + FPGA; Kapoho Point = 8× Loihi 2). Membership is free; cloud cost is unpublished (presumed free).
- **On-silicon prerequisites (priority order):** (1) project plan, (2) an already-built SNN, (3) implemented in Lava, (4) prior cloud/sim evaluation. **i.e. demonstrate in simulation first — which is exactly what a software PoC delivers.**
- **Friction (primary, Jan 2026):** an Intel Community thread documents the "Join the INRC" survey showing "not currently active" and an emailed inquiry going unanswered. Go via a PI/existing contact and escalate by email (`inrc_interest@intel.com`). Timeline is not published (multiple serial gates incl. the university contracts office). Export-control posture for non-US (AU/UK/EU) students is **unpublished — confirm with Intel before relying on access.**

### 1.5 Prior event-vision on Loihi (SWaP / latency / energy)

| Work | Task / dataset | Hardware | Accuracy | Energy / latency | GPU/CPU comparison | Grade |
|---|---|---|---|---|---|---|
| Massa et al. 2020 [arXiv:2006.09985] | 11-class gesture, DvsGesture | Loihi 1 | 89.6% | 11.4 ms/frame | **no energy reported** | [PR] IJCNN |
| Amir et al. 2017 (ref) | DvsGesture | TrueNorth | **96.5%** | 105 ms, <200 mW | — | [PR] CVPR |
| CarSNN [arXiv:2107.00401] | 2-class car/bg, N-CARS | Loihi 1 | 86→83% | 319.7 µJ, ≤0.72 ms | "orders" (unquantified) | [PR] IJCNN |
| Robot dodging [arXiv:2410.10601] | real-robot dodging | Loihi 1 (Kapoho Bay) | 97–98% | 1.39 mJ, 122 inf/s | **4.30% of Jetson Orin NX (32.3 mJ) ≈ 23×** | [PR] 2025 |
| Blouw et al. 2019 (audio ref) [arXiv:1812.01739] | keyword spotting | Loihi 1 | = ANN | **0.27 mJ**, 0.081 W | ~110× GPU / ~23× CPU (batch-1, 2014-era GPU) | [WS] NICE |
| **S4D on Loihi 2** [arXiv:2409.15022] | sMNIST/psMNIST/sCIFAR | **Loihi 2** | within 1–3 pts FP32 | streaming | **~1000× energy / ~75× latency vs Jetson recurrent; GPU wins batched** | [WS] Intel |
| **Gamage et al. 2026** [arXiv:2605.00146] | **multi-class detection, Gen1** + others | **Loihi 2** | **~0.42 mAP@0.5** (vs ~0.45 ANN); ≈0.19–0.22 COCO mAP | 0.96–1.9 mJ, 6–16 ms, 2.05–2.57 W | **10–35× vs Jetson Nano B01; 1.6–3.7× vs Orin Nano; Orin 1.3–2.6× faster** | [PR] *Neurocomputing* |

**Object-detection verdict.** Classification (CarSNN, gestures) and single-object localisation existed for years; **full multi-class bounding-box detection on Loihi appeared only in 2026** (Gamage et al., on Gen1). But every Loihi detector is forced to **drop temporal recurrence (no Conv-LSTM), drop multi-scale FPN, and stay tiny (<1M neurons)** due to Loihi 2's no-branching / fixed-neuron-count / memory restrictions (Gamage et al. list these explicitly: no branching → single coupled detection head; fixed neuron count → recompile per resolution; off-chip comms degrade energy; bias + max-pool removed; BN reduced to mean-only; 7 inference timesteps + 1 reset). **No recurrent/SSM-class, RVT-scale detector has run on Loihi.** Davies et al. 2021 confirms Loihi favours recurrent/temporal/sparse workloads (>3 orders EDP on LASSO) and admits plain feedforward vision is *not* where it wins — which both explains the gap and motivates an SSM. **The open research gap your thesis sits in: a recurrent SSM-based event detector on Loihi 2.**

---

## 2. Spiking State-Space Models (the core)

### 2.1 The theoretical bridge: SSM as a generalisation of the LIF neuron

A LIF neuron's subthreshold dynamics are a **first-order linear recurrence** (a leaky integrator) followed by a threshold-and-reset nonlinearity. Discretised by forward-Euler, `U[t] = αU[t−1] + (1−α)RI[t]` with `α = e^{−Δt/τ}`. A structured SSM is the *same* linear-recurrence family, generalised along three axes:

1. **Scalar → N-dimensional state** (the hidden state becomes a vector "membrane potential").
2. **Real → complex eigenvalues** of `A` — real λ = pure leak (LIF), complex λ = decay-and-rotate (resonate-and-fire, oscillatory).
3. **Fixed leak → learnable, HiPPO-initialised, log-reparametrised `A` with a learnable timestep Δ.**

The cleanest citable anchors:
- **SiLIF** [4] [PRE] gives the explicit identity: a 2-state adaptive-LIF neuron *is* an SSM block with `x_t=[u_t,w_t]ᵀ`, `Ā=[[α, α−1],[a, β]]`, `B̄=[[1−α],[0]]`, `C̄=[[1],[0]]ᵀ`, `α=e^{−Δt/τ_u}`, `β=e^{−Δt/τ_w}`; the *only* differences from a generic SSM are state-expansion and the spike-triggered reset feedback. C-SiLIF uses a complex transition = a resonate-and-fire neuron with S4D-Lin initialisation, importing two S4 tricks (learnable per-neuron Δt, log-reparametrisation). New SOTA *among spiking neuron models* on SHD/SSC/GSC, Pareto-optimal at ~⅓ of S4D's synaptic-op count. **The strongest theory anchor; tasks are small keyword-spotting, so generalisation to Gen1 detection is unproven.**
- **MIMO spiking neurons** [5] [PR, NICE 2025]: neuron = discrete SSM `v[t+1]=Av[t]+Bi[t]`; **LIF is the SISO special case** (`n=1, A=α, B=1−α, C=[1]`); generalises to SIMO/MISO/MIMO. The most explicit "LIF = SISO SSM → MIMO generalisation" statement in the literature.
- **PSN** [7] [PR, NeurIPS 2023]: *dropping the reset* turns LIF into a pure parallelisable linear recurrence — the conceptual ancestor, and the proof that reset is the one thing distinguishing a spiking neuron from an SSM.
- **PRF / Balanced-RF** [8] [PRE] / [PR, ICML 2024]: complex-domain RF neuron `z[t]=e^{δ(b+iω)}z[t−1]+δI[t]` — identical in form to S4D's diagonal-complex recurrence; Balanced-RF derives the explicit decay-vs-timestep **stability bound** (the spiking analogue of SSM eigenvalue stability under discretisation).
- **LRU** [9] [PR, ICML 2023] (non-spiking anchor): isolates "linear recurrence is what matters" and is the source of the log/exp reparametrisation SiLIF imports.

### 2.2 The core algorithmic SSSMs

**SpikingSSMs** [1] **[PR, AAAI 2025].** S4D diagonal backbone + hard-reset LIF; the key trick is a lightweight **Surrogate Dynamic Network (SDN)** that *predicts the after-reset membrane potential in parallel*, dissolving the sequential-reset bottleneck (the SDN is a training-time artefact, switched to inference-mode for the task and discarded at deployment). LRA avg **84.33%** vs S4D-Lin 85.52 (−1.2); solves Path-X (94.82); WikiText-103 ppl **33.94 @ 75M** (beats SpikeGPT 39.75 @ 213M, far from non-spiking S4 ~20.9). ~**90% sparsity** (theoretical energy). **Loihi-mappability: low** — the SDN is GPU-only; the real-valued complex-diagonal S4D state is not neuromorphic-native.

**P-SpikeSSM** [2] **[PR, ICLR 2025 poster].** *Its arXiv v1 was literally titled "Rethinking Spiking Neural Networks as State Space Models"* — renamed for camera-ready (this resolves the brief's "if it exists" item: it is *this* paper, not a separate one). Treats the N-D SSM hidden state as membrane potential; **stochastic Bernoulli spiking** via `SpikeSampler` (spike if `z < p_s[t]`, backward uses `E[S_t]=p_s[t]`) → parallel training, no BPTT; `SpikeMixer` + `FuseClamp`. psMNIST 98.4% (S4 98.7); LRA trails S4 by 3–10; Speech Commands 95.6%. ~90% near-zero spike probability; **>70× theoretical energy** (45nm: MAC 4.6 pJ vs ACC 0.9 pJ). **Loihi-mappability: low–moderate** — stochastic spiking needs on-chip RNG; the BatchNorm in FuseClamp is a non-local op hostile to neuromorphic HW (authors acknowledge).

**SPikE-SSM** [3] **[PRE]** (ICLR'25 withdrawn). S4D + trainable-threshold refractory-LIF; core contribution **PMBC (Parallel Max-based Boundary Compression)** brackets the after-reset potential to parallelise training (M=3 iterations resolve ~99% of spikes; 25.6× speedup at L=1K, 81.7× at L=8K). LRA avg **84.18%** vs S4D-Lin 85.52; **beats the ANN on Path-X**. ~8% firing; ~95% energy cut (theoretical). **Loihi-mappability: low** — PMBC is a non-event-driven training trick.

**Stan & Rhodes** [10] **[PR, *Sci. Reports* 2024], "Learning long sequences in spiking neural networks."** Systematic S4/S5/S4D + binary spiking activations (the de-facto "Binary-S4D" baseline, which has no standalone paper — see §4). Shows SSM-based SNNs can beat the Transformer on the full LRA and beat prior SNNs with fewer params on sequential image classification. The foundational, peer-reviewed reference that SSMs are the right substrate for long-sequence SNNs.

### 2.3 Spiking-Mamba and selective-SSM variants (taxonomy caution)

**No verified paper genuinely runs Mamba's input-dependent Δ/A/B as a spiking, neuromorphic-mappable recurrence.** They fall into four buckets:

- **(a) Spiking front-end + ANN Mamba core:** Mamba-Spike [14] (CGI 2024 — graphics venue, lower confidence); SpikeMba [15] [PRE] (the "spiking" part is only a binary saliency gate; the Mamba is conventional — tangential to neuromorphic claims).
- **(b) Spiking the projections, SSM core full-precision:** SpikingMamba (LLM) [12] [PR, TMLR 2026] — signed-integer LIF (SI-LIF) spikes the linear projections (>90% of params; MAC→AC) while the selective SSM core stays full-precision; KD from Mamba-2. 1.3B: zero-shot avg 59.40→54.62 (−4.78); 4.76× theoretical energy. Honest, but the efficiency comes from the part that *isn't* the SSM.
- **(c) LIF interleaved with selective Mamba (event vision — most thesis-relevant):** SpikMamba [11] [PR, ACM MM Asia 2024] — Spike-Form Mamba on event-based human action recognition (PAF 96.28%, HARDVS 97.32%, DVS-Gesture 99.01%; 0.18M params, 0.12 GFLOPs). Closest spiking analogue to the EventSSM thesis, but still GPU surrogate-gradient training; input-dependent A/B/C/Δ clashes with fixed neuromorphic fabric. Spiking Point Mamba [PR, ICCV 2025] for 3D point clouds is similar.
- **(d) ANN→SNN conversion with power-of-two arithmetic (the only Loihi-targeted one):** SpikySpace [PRE 2026] — Average-IF neurons, spike-driven selective scan, power-of-two state transition `2^⌊Δ·A⌉` + bit-shift activations → multiplication-free; INT8 weights + binary spikes. **Best-designed candidate for hardware, but validated only in Lava simulation, and the power-of-two trick effectively removes true continuous selectivity.**

### 2.4 Important non-spiking neighbours (do NOT mis-cite as spiking)

- **S7** [21] [PRE] — **NOT spiking.** A non-spiking selective SSM (S5 + input-dependent transitions, avoiding Mamba's hardware-specific kernels), merely *evaluated on* neuromorphic/event datasets. **Same lab (UZH RPG / Scaramuzza) as the thesis's S5-RVT baseline.** DVS-Gesture 99.2%, but LRA avg **71.82%** (vs S5 87.46) — a cautionary datapoint that selectivity can *hurt* on long-range vision. An excellent non-spiking comparison anchor; verify whether it is spiking before citing (it is not).
- **SMamba** [22] [PR, AAAI 2025] — **NOT spiking.** "Sparse" = adaptive event-token dropping. A name-collision trap; relevant only as a *non-spiking sparse-Mamba event-detection baseline* (Gen1/1Mpx/eTram).
- **Event-SSM** [23] [PRE] — **non-spiking** deep SSM processing raw events event-by-event; a high-accuracy/high-compute baseline.
- **Compute-in-memory SSM** [26] [PR, *Nature Communications* 2026] — **non-spiking**, but the **only measured-silicon SSM-on-hardware work besides Loihi-S4D**: a diagonalised real-valued SSM with shared decay on RRAM/memristor crossbars; SHD 95.7%, DVS128-Gesture 97.3%, ~62× FLOP reduction. Proves SSMs map to in-memory neuromorphic-style silicon — as a non-spiking diagonalised SSM.

### 2.5 Comparison table — spiking SSMs

| Paper | Year / Venue | Spiking mechanism | Task / dataset | Acc vs non-spiking SSM | Energy / sparsity | HW-mappable to Loihi? |
|---|---|---|---|---|---|---|
| SpikingSSMs [1] | 2025 / **AAAI** | S4D + hard-reset LIF; **SDN** predicts after-reset potential | LRA, WikiText-103 | LRA 84.33 vs S4D-Lin 85.52 (−1.2); ppl 33.94@75M | ~90% sparsity; *theoretical* | Low (SDN GPU-only) |
| P-SpikeSSM [2] | 2025 / **ICLR (poster)** | N-D state = membrane; **stochastic Bernoulli** spikes | psMNIST, LRA, SC10 | psMNIST 98.4 vs S4 98.7; LRA −3..−10 | ~90% sparse; **>70×** *theoretical* | Low–moderate (RNG, BN) |
| SPikE-SSM [3] | 2024 / preprint (ICLR'25 withdrawn) | S4D + refractory-LIF; **PMBC** parallel reset | LRA, WikiText-103 | LRA 84.18; **beats ANN on Path-X** | ~8% firing; ~95% cut *theoretical* | Low (PMBC GPU-only) |
| Stan & Rhodes [10] | 2024 / **Sci. Reports** | S4/S5/S4D + binary spike activations (≈Binary-S4D) | LRA, sequential images | beats Transformer on LRA; SOTA SNN | sparsity; *theoretical* | Moderate (diagonal core) |
| SiLIF / C-SiLIF [4] | 2025 / preprint | AdLIF *written as* 2-state SSM; learnable Δt, log-reparam, S4D init | SHD, SSC, GSC | SSC 82.0; SOTA *among neurons*; ⅓ S4D SOPs | SOP proxy; *theoretical* | **High (in principle)** — reset spikes, RF primitive |
| MIMO neuron [5] | 2025 / **NICE** | neuron = SSM; LIF = SISO special case | SHD | SIMO 89.5 vs cont. 95.5–96.3 | binary/ternary; no measured | Conceptual |
| PSN [7] | 2023 / **NeurIPS** | reset *removed* → parallel linear-recurrence neuron | sMNIST, CIFAR10-DVS, ImageNet | n/a (vs SNNs) | speedup/sparsity | Causal variants mappable |
| PRF [8] | 2024 / preprint | resonate-and-fire (=S4D diagonal); differentiable reset; parallel scan | LRA | "comparable to S4" | ~2 orders lower *theoretical* | Medium (RF primitive; FFT) |
| SpikMamba [11] | 2024 / **ACM MM Asia** | LIF spikes through ZOH-discretised **selective** Mamba | Event HAR (PAF/HARDVS/DVS-Gesture) | +1.5..+7.2 vs ANN/SNN | 0.18M params, 0.12 GFLOPs (no power) | Doubtful (input-dep A/B/C/Δ) |
| SpikingMamba-LLM [12] | 2026 / **TMLR** | SI-LIF spikes **projections only**; SSM core FP | Zero-shot LM | 59.40→54.62 (−4.78) | **4.76×** *theoretical* | Projections only |
| SpikySpace | 2026 / preprint | AIF + ANN→SNN; spike-driven scan; power-of-2 / bit-shift | Time-series | R² 0.994 vs iTransformer 0.983 | **96–99%** *theoretical* | **Best-designed**; Lava-sim only |
| SpikeMba [15] | 2024 / preprint | binary saliency gate; Mamba FP | Temporal video grounding | "beats SOTA"; no ablation | none quantified | No (GPU saliency mask) |
| LRU [9] (anchor) | 2023 / **ICML** | **non-spiking** diagonal complex linear recurrence | LRA | matches S4/S5 | n/a | n/a |
| S7 [21] ⚠️ | 2024 / preprint | **NOT spiking** — selective SSM (S5-RVT lab) | DVS-Gesture 99.2; LRA 71.82 | n/a | n/a | n/a |
| SMamba [22] ⚠️ | 2025 / **AAAI** | **NOT spiking** — event-token dropping | Event detection Gen1/1Mpx/eTram | n/a | token sparsity | n/a |
| CIM-SSM [26] | 2026 / **Nat. Commun.** | **NOT spiking** — diagonalised SSM on RRAM | Event vision + audio | SHD 95.7; DVS-Gesture 97.3 | **measured** in-memory | **Yes (measured)** — non-spiking |

---

## 3. Implementation paths (concrete)

### 3.1 Open-source implementations

| Repo | Paper (venue, year) | Framework | Maturity | License | Exports to Loihi? |
|---|---|---|---|---|---|
| NeuroCompLab-psu/PSpikeSSMs | P-SpikeSSM (ICLR 2025) | PyTorch (Lightning+Hydra) | ~8★, runnable, **no weights** | MIT | No |
| shenshuaijie/SDN | SpikingSSMs (AAAI 2025) | PyTorch | ~23★ (**most active**), runnable | MIT | No |
| typistchen/SpikMamba | SpikMamba (ACM MM Asia 2024) | PyTorch + `mamba_ssm 1.0.1` | ~37★, "to be cleaned", no weights | Apache-2.0 | No |
| HuuYuLong/SpikingMamba | SpikingMamba (TMLR 2026) | PyTorch | ~5★, runnable | Apache-2.0 | No |
| Maxtimer97/SSM-inspired-LIF | SiLIF (preprint 2025) | PyTorch | ~8★, runnable | **None (all rights reserved)** | No (closest SSM↔LIF bridge) |
| (none) | SPikE-SSM / SpikeMba | — | **no public repo** | n/a | No |
| lava-nc/lava-dl | SLAYER/Bootstrap/Netx | PyTorch→Lava | ~181★, **ARCHIVED** | BSD-3 | **Yes — direct** (HDF5→Loihi 2) |
| lava-nc/lava | Lava core / Loihi runtime | Lava/Python | ~733★, **ARCHIVED** | BSD-3 / LGPL-2.1 | **It is the runtime** (INRC-gated) |
| jeshraghian/snntorch | (training lib) | PyTorch | ~2.0k★ | MIT | Indirect (NIR→Lava) |
| fangwei123456/spikingjelly | (training lib) | PyTorch+CUDA | ~2.0k★ | custom/NOASSERTION | Weak/none |
| norse/norse | (training lib) | PyTorch | ~0.8k★ | **LGPL-3.0** | Indirect (NIR-supported) |
| state-spaces/mamba | Mamba | PyTorch+CUDA | ~18.5k★ | Apache-2.0 | n/a (non-spiking base) |
| lindermanlab/S5 | S5 | **JAX/Flax** | ~321★ | MIT | n/a (non-spiking base) |
| uzh-rpg/ssms_event_cameras | S5-RVT (CVPR 2024) | PyTorch-Lightning | ~133★ | **no declared license** | n/a (thesis baseline) |

**Key facts:** (1) **No spiking-SSM repo exports to Loihi** — the bridge must be built, realistically via **NIR (Neuromorphic Intermediate Representation, *Nature Communications* 2024, arXiv:2311.14641) → Lava-DL → Loihi 2**. (2) SSM ops (parallel scan, complex-state dynamics, learnable Δt) are **not** standard NIR primitives, so the lowering is itself a contribution. (3) There is **no published SSM example in the public lava-nc repos** — but RF and RF-Izhikevich neurons *are* in lava-dl (`lava.lib.dl.slayer.neuron.rf`), which is the natural primitive for complex SSM modes. (4) **SiLIF has no LICENSE (all rights reserved)** — contact the authors before reuse; the licenses to watch are AGPL (Rockpool) and LGPL (Norse).

### 3.2 Realistic S5/Mamba → spiking mapping

**The friendly parts:** ZOH discretisation (`Ā=exp(AΔ)`, `B̄=(Ā−I)A⁻¹B`) — the per-mode decay `exp(λΔ)` *is* the LIF leak (real λ) or RF decay-and-rotate (complex λ); fixed-point quantisation of weights+state (8-bit weights / 24-bit graded-spike state, PTQ→QAFT, demonstrated for S4D [25]); replacing softplus/SiLU/GLU with thresholds/ReLU; surrogate-gradient training; reset rule (non-trivial for oscillatory RF modes — Balanced-RF shows hard/soft reset corrupts phase).

**What breaks (state honestly in the thesis):**
1. **Input-dependent Δt / Mamba selectivity (the hard wall).** The S6 selection makes Δ, B, C functions of the input — a continuous, per-step, per-channel, content-computed multiplicative gate. Running it on-chip requires recomputing each neuron's leak every step from a learned dense projection (MLP + softplus) — not a native neuron op, and it defeats event-driven sparsity. It also quantises catastrophically: PTQ fails on Mamba's activation outliers; QAT is required [37, LightMamba/Mamba-quant literature]. **No faithful selective SSM has been deployed on Loihi.**
2. **The parallel/associative scan is a training construct, not a neuromorphic primitive.** S5's parallel scan and Mamba's selective scan are GPU prefix-sums that parallelise the recurrence over time *at training*. Hardware does the opposite — sequential, one timestep per tick (the inference mode Loihi-S4D exploits as its *advantage*). **For an LTI (non-selective) SSM, train-parallel and deploy-sequential are mathematically identical, so the GPU-train / Loihi-deploy split is clean for S4D/S5 — and breaks for Mamba** (time-varying parameters make it non-LTI, the very reason Mamba cannot use a convolution). *This validates the thesis's existing dual-path scan design: train = parallel, eval = stateful step loop.*
3. **Real-valued multiplicative gating, large d_state, dense projections.** Mamba's `y = SSM(x) ⊙ SiLU(z)` element-wise product of two learned signals is not a native spiking op; large d_state = many compartments/channel; dense B/C/in/out projections map to crossbars (feasible, but this is where the *dense, non-sparse* parameter energy goes, partially undermining the event-driven pitch). Non-selective S4D/S5 (linear `Ch` readout, modest N≈64) avoid all three.

**The Loihi-friendly subset — anchor the chapter here:** a **diagonal SSM decomposes exactly into independent 1-D first-order linear recurrences (one per mode), each mapping to one independent neuron state** — real λ = LIF, complex λ = RF. S5 is the cleanest non-trivial target: one MIMO diagonal SSM with **fixed (learned) timescales, not input-dependent**. Demonstrated on both substrates: Loihi 2 (S4D, Meyer et al. [25]) and RRAM CIM (non-spiking, [26]). Cite the stability caveat (Balanced-RF [PR, ICML 2024]): vanilla RF neurons diverge for bad (ω, b, δ) — the explicit decay-vs-timestep bound is the spiking analogue of SSM discretisation stability.

### 3.3 Event-based object detection on neuromorphic HW

**Is bounding-box detection feasible on Loihi today?** As of 2026, **yes, but barely.** Until 2026 all spiking detectors (EMS-YOLO, SFOD, EAS-SNN, SpikeYOLO) were GPU-only with *operation-count-estimated* energy; Spiking CenterNet explicitly states "suitable SNN hardware is not yet available." **Gamage et al. [27]** is the first credible event detector on Loihi 2 silicon — but at ~0.19–0.22 COCO mAP (≈0.42 mAP@0.5), roughly half the GPU-SNN SOTA and well under half the thesis's 47.7.

**SOTA spiking detection on Gen1 (metric discipline is critical — see §5.1):**

| Method | Venue / Year | Gen1 COCO mAP (0.5:0.95) | AP50 (context) | Hardware | Grade |
|---|---|---|---|---|---|
| **S5-RVT (thesis baseline, non-spiking)** | CVPR 2024 | **47.7** | 75.3 | GPU | [PR] |
| **EventSSMDetector (thesis, non-spiking)** | this thesis | **46.2 (test)** | — | GPU | — |
| EMS-YOLO | ICCV 2023 | 26.7–31.0 | ~59 | GPU (theoretical energy) | [PR] |
| Spiking CenterNet | IJCNN 2024 | 22.3 | — | GPU | [PR] |
| SFOD | CVPR 2024 | **32.1** | — | GPU | [PR] |
| SpikeYOLO | ECCV 2024 (oral) | ~40.4 | **67.2** | GPU (theoretical energy) | [PR] |
| **EAS-SNN** | ECCV 2024 | **43.7** (best peer-reviewed) | ~69.9 | GPU | [PR] |
| SpikeDet/SpikSSD ⚠️ | preprint 2025 | ~47.6 (unverified) | ~70.1 | GPU | [PRE] |
| **Loihi 2 SNN (Gamage)** | *Neurocomputing* 2026 | **~0.19–0.22** | **~0.42** | **Loihi 2 (silicon)** | [PR] |

**The gap.** Best peer-reviewed spiking = **EAS-SNN 43.7** → S5-RVT leads by **+4.0** COCO mAP (the gap closed from ~16 in 2023 to ~4 in 2024). The headline-matching SpikeDet 47.6 is an **unverified preprint — do not cite as established SOTA.** **Why detection is harder for SNNs than classification (stated explicitly by SpikeYOLO):** spike-binarisation error hurts continuous bbox *regression* far more than argmax classification; deep-SNN trainability (surrogate-gradient vanishing) blocks the deep multi-scale backbones detection needs; multi-scale fusion is ill-behaved in spiking form. Classification gap ≈ 0; detection gap is real. **Gen4/1Mpx spiking detection is nearly untouched — a defensible open angle.**

---

## 4. Recommendation for THIS thesis

### 4.1 The de-risked PoC path

**Target architecture:** a **spiking diagonal-S5 temporal block** — fixed (learned) per-channel Δt, complex modes as resonate-and-fire neurons + graded-spike state, surrogate-gradient + quantisation-aware training on GPU using the parallel scan, deployed step-by-step. This deliberately **drops Mamba selectivity** (the documented hard wall), extends Meyer et al.'s S4D-on-Loihi (classification) toward event *detection*, and exploits the now-formal diagonal-SSM↔RF-neuron identity.

**Milestone sequence (each gate is a stop/continue decision):**

- **M0 — Lock the non-spiking contributions first (non-negotiable).** `EventSSMDetector` (done, test 46.2) and `PureSSMDetector` complete and benchmarked before any SNN time is spent. The SNN is **high-upside, never load-bearing**.
- **M1 — Algorithmic spiking-SSM block on GPU (lowest risk, highest information).** Replace only the Mamba temporal block in the existing backbone with a spiking diagonal-S5 block (fixed Δt, graded-spike state, surrogate gradient). Train on Gen1 (reusing the unmodified YOLO-PAFPN neck + YOLOX head + pipeline + Prophesee eval). **Deliverables:** Gen1 COCO mAP, firing rate, and an op-count (AC-vs-MAC) energy *proxy* with the explicit caveat that it is theoretical. This is a clean ablation against `EventSSMDetector` (the controlled spiking-vs-non-spiking swap). **Highest-risk sub-step:** surrogate-gradient detection training (regression-head instability — mitigate with integer/graded-spike training à la SpikeYOLO, and a refractory-LIF / Balanced-RF stability bound).
- **M2 — Quantisation-aware fine-tune + Lava CPU-sim validation.** Apply PTQ→QAFT (8-bit weights, 24-bit state), confirm accuracy retention, and validate the block runs in the **open Lava simulator** (`Loihi2SimCfg`, no hardware). This is the artefact the INRC application asks for ("implemented in Lava, evaluated in sim").
- **M3 — Lava-DL port of a SINGLE spiking-SSM block as the Loihi-readiness PoC.** Lower one diagonal-SSM block (RF/LIF neurons via `lava.lib.dl.slayer.neuron.rf`, SLAYER training → HDF5 → `netx`) to a Loihi-deployable form. **Do not attempt the full detector on-chip** — Gamage et al. shows even a tiny detector requires dropping recurrence/FPN. **Highest-risk sub-step:** SSM ops (scan, complex state, learnable Δt) are not standard NIR/Lava primitives — budget this as a research task, possibly needing INRC-gated custom microcode (the S4D port used an alpha NxKernel API).
- **M4 (future work / handover) — on-silicon measurement.** Once INRC access lands (via a PI), measure on-chip power/latency/throughput and close the sim-to-silicon gap. Stage as documented future work — it is *not* required for thesis completion.

**Highest-risk steps overall, ranked:** (1) INRC hardware access (institutional, no student path, broken survey, export-control unknown) → mitigate by making M1–M3 hardware-*independent*; (2) SSM-op lowering to Lava/NIR (custom-primitive research); (3) spiking *detection*-head training stability; (4) lava-nc being archived/frozen (pin 0.6.0, expect torch-compat friction).

### 4.2 Defensible vs over-reach claims (SWaP / micro-UAV narrative)

**Defensible:**
- "A LIF neuron is the scalar, real-eigenvalue special case of a structured SSM; moving from the Mamba detector to a spiking SSM is a principled reparametrisation, not a new field" (anchored by SiLIF [4], MIMO-neuron [5], PSN [7]).
- "A non-selective diagonal SSM (S4D) has been deployed on Loihi 2 with large energy/latency gains in the online streaming regime; S5 and a spiking SSM for event *detection* are unprecedented" (Meyer et al. [25]; the gap is real).
- "Spiking SSMs trade a measurable accuracy cost for sparsity and neuromorphic deployability; the contribution is closing the *deployable-on-neuromorphic* gap, not beating GPU mAP" (EAS-SNN 43.7 vs S5-RVT 47.7; Gamage on-silicon ~0.2 COCO mAP).
- "For SWaP-constrained micro-UAV perception, the relevant regime is batch-1 online inference — exactly where neuromorphic hardware wins and the GPU does not."

**Over-reach (avoid):**
- Quoting any spiking-SSM energy figure as a *measured* win — almost all are theoretical op-counts. State this every time.
- "1000× more efficient than a GPU" without the streaming caveat (true only batch-1 vs a recurrent GPU baseline; the GPU wins batched, and the Jetson baseline was not TensorRT-optimised).
- "Mamba on Loihi" — no faithful selective SSM has run on neuromorphic silicon; do not imply the selective core is deployable.
- "10× faster Loihi 2" unqualified (per-operation, partly pre-silicon).
- Treating SpikeDet/SpikSSD 47.6 as established SNN SOTA (unverified preprint).
- Conflating mAP@50 (~65–70) with COCO mAP@0.5:0.95 (~40–44) for spiking detectors (the core metric pitfall — see §5.1).

---

## 5. Open questions / could-not-verify

### 5.1 Metric-discipline warning (the most important correction)
Spiking-detection papers routinely headline **AP50** (~65–70 on Gen1), which is **not** comparable to the thesis's COCO mAP@0.5:0.95 = 47.7. **Independently verified:** SpikeYOLO's Gen1 "67.2%" is mAP@**50** (its 48.9 COCO figure is on the COCO *image* dataset, not Gen1); SFOD's 32.1% is the COCO-style Gen1 SNN figure; Gamage et al.'s "~0.42 mAP" is mAP@**0.5** (≈0.19–0.22 at 0.5:0.95). **The existing thesis docs list "SpikeYOLO 0.385 Gen1 COCO mAP" and "SpikSSD 40.8 SNN SOTA" — these should be re-checked against the original tables for which IoU range they report; this report uses EAS-SNN 43.7 as the best *verified peer-reviewed* COCO-mAP spiking detector.** Always confirm the IoU range before any cross-model comparison goes in the thesis.

### 5.2 Items to confirm from PDFs before citing in the `.bib`
1. **"Rethinking SNNs as State Space Models" is NOT a separate paper** — it is the arXiv v1 title of P-SpikeSSM [2] (ICLR 2025). Cite once.
2. **"Binary-S4D" / "Binarized S4" has no standalone paper** — it appears only as a *baseline method* (naïve binary activation on S4D) inside SSSM papers; the foundational peer-reviewed work is Stan & Rhodes [10] (*Sci. Reports* 2024). Treat Binary-S4D as a method, not a citable primary work.
3. **"SpikingLRU" does not exist** as a standalone paper; the LRU's spiking realisation in practice = SiLIF / C-SiLIF and RF-SSM variants.
4. **Meyer et al. [25] specifics** — exact neuron mechanism (custom microcode SSM neuron vs RF), and the energy/latency figure (arXiv "~75× latency" vs a "29× faster" virtual-page figure); confirm from the PDF. Whether Intel released the S4D code publicly (appears INRC-gated).
5. **Fixed-point bit-widths + per-dataset accuracy drop** for Loihi-S4D and QS4D.
6. **Gamage et al. [27] exact Gen1-on-Loihi mAP and IoU range** — table values ~0.19–0.22 (0.5:0.95) vs ~0.42 (mAP@0.5); confirm which model (event vs frame) was profiled on-silicon for each metric.
7. **SpikeDet/SpikSSD 47.6 Gen1** — preprint, no confirmed venue, highest-impact number with the weakest provenance.
8. **EAS-SNN exact Gen1/Gen4 numbers** — appears as both 33.8 and 37.5(S) and 43.7(M) across sources; verify the model size and IoU.
9. **Lava-nc archival date** (reported mid-2026 / "13 May 2026" in one source) and Intel's current recommended Loihi 2 SDK.
10. **INRC export-control posture for AU/UK/EU students; timeline; cloud cost** — all unpublished/inferred.
11. **CIM-SSM [26] author list** — article body paywalled; verify before citing.
12. **Several preprint result tables** (SiLIF per-task, SpikySpace full tables, SpikMamba surrogate-gradient choice, Delays-SSM math) read from abstracts/HTML, not full PDFs.

### 5.3 Suspected hype / vendor-marketing to discount
- All spiking-SSM "energy reduction" claims are **theoretical operation counts** unless explicitly on Loihi/RRAM silicon.
- "Loihi 2 = 1M neurons / 120M synapses" (non-coexisting maxima) and "32-bit graded spikes / weights" (usable magnitude is 24-bit; weights ~8-bit).
- Generic SNN-on-Loihi energy ratios usually exclude idle/host power and assume batch-1 vs old GPUs — none are free wins.
- The announced Lava "next-gen SDK" — no public release/version/license/date; do not plan around it.

---

## 6. IEEE-style reference list

[1] S. Shen, C. Wang, R. Huang, Y. Zhong, Q. Guo, Z. Lu, J. Zhang, and L. Leng, "{SpikingSSMs}: Learning Long Sequences with Sparse and Parallel Spiking State Space Models," *Proc. AAAI Conf. Artif. Intell. (AAAI)*, vol. 39, no. 19, pp. 20380–20388, 2025. doi:10.1609/aaai.v39i19.34245. arXiv:2408.14909. [PR]

[2] M. Bal and A. Sengupta, "{P-SpikeSSM}: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks," *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025. arXiv:2406.02923 (v1 titled "Rethinking Spiking Neural Networks as State Space Models"). https://openreview.net/forum?id=Sf4ep9Udjf [PR]

[3] Y. Zhong, R. Zhao, C. Wang, Q. Guo, J. Zhang, Z. Lu, and L. Leng, "{SPikE-SSM}: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning," arXiv:2410.17268, 2024. [PRE]

[4] M. Fabre, L. Dudchenko, Y. Bouhadjar, and E. Neftci, "{SiLIF}: Structured State Space Model Dynamics and Parametrization for Spiking Neural Networks," arXiv:2506.06374, 2025. https://arxiv.org/abs/2506.06374 [PRE]

[5] S. Karilanova, S. Dey, and A. Özçelikkale, "State-Space Model Inspired Multiple-Input Multiple-Output Spiking Neurons," *Proc. Neuro-Inspired Computational Elements (NICE)*, 2025. arXiv:2504.02591. [PR]

[6] S. Karilanova, S. Dey, and A. Özçelikkale, "Delays in Spiking Neural Networks: A State Space Model Approach," arXiv:2512.01906, 2025. [PRE]

[7] W. Fang, Z. Yu, Z. Zhou, D. Chen, Y. Huang, T. Masquelier, and Y. Tian, "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-Term Dependencies," *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023. arXiv:2304.12760. [PR]

[8] Y. Huang et al., "{PRF}: Parallel Resonate and Fire Neuron for Long Sequence Learning in Spiking Neural Networks," arXiv:2410.03530, 2024. [PRE]

[9] A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De, "Resurrecting Recurrent Neural Networks for Long Sequences," *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 202, 2023. arXiv:2303.06349. [PR]

[10] M. I. Stan and O. Rhodes, "Learning long sequences in spiking neural networks," *Scientific Reports*, vol. 14, art. 21957, 2024. doi:10.1038/s41598-024-71678-8. [PR]

[11] J. Chen, Y. Yang, S. Deng, D. Teng, and L. Pan, "{SpikMamba}: When {SNN} Meets {Mamba} in Event-Based Human Action Recognition," *Proc. ACM Int. Conf. Multimedia in Asia (MM Asia)*, 2024. arXiv:2410.16746. [PR]

[12] Y. Huang, J. Tang, C. Wang, Z. Wang, J. Zhang, Z. Lu, B. Cheng, and L. Leng, "{SpikingMamba}: Towards Energy-Efficient Large Language Models via Knowledge Distillation from {Mamba}," *Trans. Mach. Learn. Res. (TMLR)*, 2026. arXiv:2510.04595. [PR]

[13] L. Huber et al., "Scaling Up Resonate-and-Fire Networks for Fast Deep Learning," arXiv:2504.00719, 2025 (RF neuron understood as an SSM). [PRE]

[14] J. Qin and F. Liu, "{Mamba-Spike}: Enhancing the {Mamba} Architecture with a Spiking Front-End," *Proc. Computer Graphics International (CGI)*, 2024. arXiv:2408.11823. [PR-adjacent/graphics]

[15] W. Li, X. Hong, R. Xiong, and X. Fan, "{SpikeMba}: Multi-Modal Spiking Saliency {Mamba} for Temporal Video Grounding," arXiv:2404.01174, 2024. [PRE]

[16] R.-J. Zhu, Q. Zhao, G. Li, and J. K. Eshraghian, "{SpikeGPT}: Generative Pre-trained Language Model with Spiking Neural Networks," *Trans. Mach. Learn. Res. (TMLR)*, 2024. arXiv:2302.13939. [PR]

[17] M. Yao, J. Hu, Z. Zhou, L. Yuan, Y. Tian, B. Xu, and G. Li, "Spike-driven Transformer," *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023. arXiv:2307.01694. [PR]

[18] T. Higuchi et al., "Balanced Resonate-and-Fire Neurons," *Proc. Int. Conf. Mach. Learn. (ICML)*, 2024. arXiv:2402.14603. [PR]

[19] A. Gu, K. Goel, and C. Ré, "Efficiently Modeling Long Sequences with Structured State Spaces ({S4})," *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022. arXiv:2111.00396. [PR]

[20] J. T. H. Smith, A. Warrington, and S. W. Linderman, "Simplified State Space Layers for Sequence Modeling ({S5})," *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023. arXiv:2208.04933. [PR]

[21] T. Soydan, N. Zubić, N. Messikommer, S. Mishra, and D. Scaramuzza, "{S7}: Selective and Simplified State Space Layers for Sequence Modeling," arXiv:2410.03464, 2024. *(non-spiking; same group as the S5-RVT baseline)*. [PRE]

[22] N. Yang, Y. Wang, Z. Liu, M. Li, Y. An, and X. Zhao, "{SMamba}: Sparse {Mamba} for Event-Based Object Detection," *Proc. AAAI Conf. Artif. Intell. (AAAI)*, 2025. arXiv:2501.11971. *(non-spiking)*. [PR]

[23] M. Schöne, N. M. Sushma, J. Zhuge, C. Mayr, A. Subramoney, and D. Kappel, "Scalable Event-by-event Processing of Neuromorphic Sensory Signals with Deep State-Space Models," arXiv:2404.18508, 2024. *(non-spiking Event-SSM)*. [PRE]

[24] A. Gu, A. Gupta, K. Goel, and C. Ré, "On the Parameterization and Initialization of Diagonal State Space Models ({S4D})," *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022. arXiv:2206.11893. [PR]

[25] S. M. Meyer, P. Weidel, P. Plank, L. Campos-Macias, S. B. Shrestha, P. Stratmann, and M. Richter, "A Diagonal Structured State Space Model on {Loihi 2} for Efficient Streaming Sequence Processing," *NeurIPS 2024 Workshop MLNCP*, 2024. arXiv:2409.15022. [WS, Intel]

[26] (CIM-SSM) "Compute-in-Memory Implementation of State Space Models for Event Sequence Processing," *Nature Communications*, 2026. arXiv:2511.13912 *(non-spiking, measured RRAM silicon — verify author list before citing)*. [PR]

[27] U. G. W. K. N. Gamage, Y. Zeng, C. Cadena, M. Fumagalli, and S. Tolu, "Real-Time Frame- and Event-based Object Detection with Spiking Neural Networks on Edge Neuromorphic Hardware: Design, Deployment and Benchmark," *Neurocomputing*, 2026. doi:10.1016/j.neucom.2026.133820. arXiv:2605.00146. [PR]

[28] S. B. Shrestha, J. Timcheck, P. Frady, L. Campos-Macias, and M. Davies, "Efficient Video and Audio Processing with {Loihi 2}," *Proc. IEEE ICASSP*, 2024, pp. 13481–13485. arXiv:2310.03251. [PR]

[29] M. Davies, A. Wild, G. Orchard, Y. Sandamirskaya, G. A. Fonseca Guerra, P. Joshi, P. Plank, and S. R. Risbud, "Advancing Neuromorphic Computing With {Loihi}: A Survey of Results and Outlook," *Proc. IEEE*, vol. 109, no. 5, pp. 911–934, 2021. doi:10.1109/JPROC.2021.3067593. [PR]

[30] M. Davies et al., "{Loihi}: A Neuromorphic Manycore Processor with On-Chip Learning," *IEEE Micro*, vol. 38, no. 1, pp. 82–99, 2018. doi:10.1109/MM.2018.112130359. [PR]

[31] Intel Labs, "Taking Neuromorphic Computing to the Next Level with {Loihi 2}" (Technology Brief), 2021. https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf [V]

[32] J. Pedersen et al., "Neuromorphic Intermediate Representation ({NIR}): A Unified Instruction Set for Interoperable Brain-Inspired Computing," *Nature Communications*, 2024. arXiv:2311.14641. [PR]

[33] R. Massa, A. Marchisio, M. Martina, and M. Shafique, "An Efficient Spiking Neural Network for Recognizing Gestures with a {DVS} Camera on the {Loihi} Neuromorphic Processor," *Proc. IJCNN*, 2020. arXiv:2006.09985. [PR]

[34] A. Viale, A. Marchisio, M. Martina, G. Masera, and M. Shafique, "{CarSNN}: An Efficient Spiking Neural Network for Event-Based Autonomous Cars on the {Loihi} Neuromorphic Research Processor," *Proc. IJCNN*, 2021. arXiv:2107.00401. [PR]

[35] P. Blouw, X. Choo, E. Hunsberger, and C. Eliasmith, "Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware," *Proc. NICE*, 2019. arXiv:1812.01739. [WS]

[36] Q. Su, Y. Chou, Y. Hu, J. Li, S. Mei, Z. Zhang, and G. Li, "Deep Directly-Trained Spiking Neural Networks for Object Detection ({EMS-YOLO})," *Proc. IEEE/CVF ICCV*, 2023. arXiv:2307.11411. [PR]

[37] X. Luo, M. Yao, Y. Chou, B. Xu, and G. Li, "Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection ({SpikeYOLO})," *Proc. ECCV*, 2024. arXiv:2407.20708. [PR]

[38] Y. Fan, W. Zhang, C. Liu, M. Li, and W. Lu, "{SFOD}: Spiking Fusion Object Detector," *Proc. IEEE/CVF CVPR*, 2024. arXiv:2403.15192. [PR]

[39] (EAS-SNN) "End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks," *Proc. ECCV*, 2024. arXiv:2403.12574. [PR]

[40] H. Zhang, Y. Li, L. Leng, K. Che, Q. Liu, Q. Guo, J. Liao, and R. Cheng, "Automotive Object Detection via Learning Sparse Events by Spiking Neurons ({SpikeFPN})," *IEEE Trans. Cognitive and Developmental Systems*, 2024. arXiv:2307.12900. [PR]

[41] A. Gu and T. Dao, "{Mamba}: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, 2023. [PRE]

[42] N. Zubić, M. Gehrig, and D. Scaramuzza, "State Space Models for Event Cameras ({S5-RVT})," *Proc. IEEE/CVF CVPR*, 2024. arXiv:2402.15584. [PR]

[43] M. Gehrig and D. Scaramuzza, "Recurrent Vision Transformers for Object Detection with Event Cameras ({RVT})," *Proc. IEEE/CVF CVPR*, 2023. arXiv:2212.05598. [PR]

[44] E. Abreu et al. (Intel Labs), "Neuromorphic Principles for Efficient Large Language Models on {Loihi 2}," *ICLR 2025 Workshop SCOPE*, 2025. arXiv:2503.18002. [WS, Intel]

[45] G. Orchard, M. Davies et al., "Efficient Neuromorphic Signal Processing with {Loihi 2}," *Proc. IEEE SiPS*, 2021. arXiv:2111.03746. [PR]

[46] M. Yao et al., "Spike-driven Transformer {V2}: Meta Spiking Neural Network Architecture," *Proc. ICLR*, 2024. [PR]

[47] Z. Zhou, Y. Zhu, C. He, Y. Wang, S. Yan, Y. Tian, and L. Yuan, "{Spikformer}: When Spiking Neural Network Meets Transformer," *Proc. ICLR*, 2023. [PR]

[48] L. Cordone, B. Miramond, and P. Thierion, "Object Detection with Spiking Neural Networks on Automotive Event Data," *Proc. IJCNN*, 2022. arXiv:2205.04339. [PR]

[49] N. Hagenaars et al., "Spiking CenterNet: A Distillation-boosted Spiking Neural Network for Object Detection," *Proc. IJCNN*, 2024. arXiv:2402.01287. [PR]

> **BibTeX note (thesis standards):** verify and complete all author lists (no `and others`); protect acronyms in titles (`{SSM}`, `{SNN}`, `{LIF}`, `{S4D}`, `{S5}`, `{Mamba}`, `{Loihi}`, `{SiLIF}`); confirm the venue line for every [PRE]/[WS] item and the IoU range for every Gen1 mAP before entry. Refs [13], [39] need full author lists from the arXiv records.
