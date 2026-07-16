# Spiking SSMs & Intel Loihi — Opportunity, Literature Review, and Thesis-Incorporation Options

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis**
**Date:** 2026-06-18 (Thesis B, week 3 of 13) · **Status:** strategic note — discussed, nothing built yet

---

## 1. The opportunity

From a supervisor conversation (2026-06-18):

- **Intel** produces the **Loihi / Loihi 2** neuromorphic research chip (NB: supervisor said "IBM" — it is **Intel**; IBM's neuromorphic chips are TrueNorth / NorthPole).
- Hardware access is granted via the **Intel Neuromorphic Research Community (INRC)** application program.
- The way to **maximise the chance of being granted hardware** is to arrive with a **working proof-of-concept (PoC)**.
- Loihi is architected for **Spiking Neural Networks (SNNs)** — event-driven, binary-spike, low-power computation — which is the paradigm it accelerates natively.
- Supervisor suggested implementing/improving an existing event-vision SNN as the PoC, naming **StereoSpike**, **EV-FlowNet / Spike-FlowNet**, and **Spiking-UNet**, noting it could be both a strong PoC for the Intel application **and** publishable.

**Why this is aligned, not a distraction.** The thesis already targets the neuromorphic *sensor* (event camera) and motivates **SWaP-constrained micro-UAV** deployment. Loihi is the neuromorphic *compute* endpoint of exactly that stack: **event sensor → spiking/stateful perception → neuromorphic accelerator → micro-UAV**. This opportunity *completes* the existing narrative rather than replacing it.

---

## 2. Literature review

### 2.1 The SSM ↔ SNN connection — why *your* work is the natural bridge

A **Leaky Integrate-and-Fire (LIF)** spiking neuron is, mathematically, a **first-order linear state-space recurrence** (a leaky integrator of input current into a membrane potential) followed by a **spike-and-reset nonlinearity**. Structured State-Space Models (S4/S5/Mamba) are the *same* family — a linear stateful recurrence — without the spiking nonlinearity. This is now an explicit, active research direction:

- **P-SpikeSSM** [8] reformulates SNNs *as* state-space models, arguing the SSM view gives a "more accurate and efficient representation of [SNN] dynamics," and builds a scalable probabilistic spiking learning framework for long-range dependencies.
- **SpikingSSMs** [6] fuse sparse spiking computation with parallel SSM training, reaching ~90% activation sparsity while matching dense-model accuracy, with a surrogate dynamic network that accelerates training and is discarded at inference.
- **SPikE-SSM** [7] decomposes the membrane potential into parallel-computable components for fast training on long sequences.

**Implication for the thesis:** moving from your Mamba-based detector to a *spiking* SSM detector is a **theoretically-motivated single step**, not an arbitrary new model. Your SSM expertise is directly transferable, which is a strong position few students start from.

### 2.2 Event-vision SNNs (the supervisor's suggestions)

- **StereoSpike** [1] — event-based stereo **depth** estimation with a U-Net-like SNN, trained on MVSEC via surrogate-gradient descent. Notably reports that the spiking model **generalises *better* than its non-spiking counterpart**. Open code (`urancon/StereoSpike`).
- **Spike-FlowNet** [2] — event-based **optical flow** via a hybrid SNN–ANN, self-supervised on MVSEC; outperforms its ANN counterpart with large energy savings. (NB: the *original* EV-FlowNet is a CNN/ANN; the spiking version is **Spike-FlowNet**, ECCV 2020.) Open code (`chan8972/Spike-FlowNet`).
- **Spiking-UNet** — spiking segmentation backbone (dense prediction).

**Caveat:** all three are **dense-prediction** tasks (depth / flow / segmentation), **not object detection**. Adopting one wholesale means changing task; adapting their SNN *techniques* to detection keeps your task.

### 2.3 SNN object detection on Gen1 — the key positioning data

This is the most important finding: **directly-trained SNN object detectors already exist and have been benchmarked on Prophesee Gen1** — your exact dataset and task.

| Model | Type | Gen1 mAP@0.5:0.95 (COCO) | Params | Notes |
|---|---|---|---|---|
| RVT [baseline] | ANN (CNN+ViT, ConvLSTM) | **47.2** | ~19 M | Gehrig 2023 |
| S5-RVT [your baseline] | ANN (CNN+ViT, S5 SSM) | **47.7** | ~18 M | Zubic 2024 |
| **EventSSMDetector [yours]** | ANN (ResNet+Mamba) | **~0.44 val → 400k run pending** | 19.2 M | this thesis |
| EMS-YOLO [3] | **SNN** (EMS-ResNet, T=4) | **0.267** (R10) – **0.310** (R34) | 6.2 / 14.4 M | ICCV 2023; first directly-trained SNN detector |
| SpikeYOLO [4] | **SNN** (I-LIF, T=5) | **0.385** (mAP@50 = 0.672) | 23.1 M | ECCV 2024; current SNN SOTA on Gen1 |
| EAS-SNN [5] | **SNN** (recurrent, adaptive sampling) | — | — | ECCV 2024 |

**The gap *is* the opportunity.** The best SNN detector on Gen1 (**SpikeYOLO, 38.5**) trails the best ANN/SSM detectors (**~47–48**) by **~9 mAP points**, and EMS-YOLO trails by ~17–21. SNNs buy energy-efficiency and neuromorphic deployability at an accuracy cost. **The open question your thesis is uniquely placed to ask:** *can a spiking **SSM** detector — bringing principled SSM temporal modelling into the spiking domain — narrow this accuracy gap while remaining Loihi-deployable?* That is a genuine, publishable research question, and it sits one step from work you have already built.

### 2.4 Intel Loihi 2 & the Lava framework — the deployment target

- **Loihi 2** (Intel, Sept 2021): a mesh of up to 128 asynchronous neuromorphic cores, each hosting thousands of **programmable** spiking-neuron instances and up to ~1M synapses; supports **glueless event-sensor integration** (Ethernet / FPGA bridge) for direct event streaming. Programmable neuron models matter here — they make richer (SSM-like) neuron dynamics feasible, not just vanilla LIF.
- **Lava** / **Lava-DL**: Intel's open-source neuromorphic software framework. *Prototyping in Lava signals Loihi-readiness on the INRC application.*
- **NIR (Neuromorphic Intermediate Representation)** [10]: a cross-platform IR letting models trained in PyTorch SNN frameworks export toward Loihi and other neuromorphic backends — a practical bridge from prototype to hardware.
- Recent 2024–2025 work shows real event-vision deployments on neuromorphic hardware (low-power face detection, eye/pupil tracking), confirming the pipeline is mature enough for a student PoC.

**Practical training path:** prototype in **snnTorch** or **SpikingJelly** (PyTorch, surrogate-gradient training) → export via **NIR** / port to **Lava-DL** for Loihi. No chip is required to build the PoC — a software-simulated SNN on event data is sufficient for the INRC application.

---

## 3. The research gap (one sentence)

> Bring **state-space temporal modelling into a spiking, neuromorphic-deployable detector** and measure where it lands on the **accuracy ↔ energy ↔ deployability** trade-off, against both ANN-SSM detectors (your EventSSM/PureSSM, S5-RVT) and existing SNN detectors (EMS-YOLO, SpikeYOLO) on Gen1.

This turns the supervisor's "build an SNN PoC" into a question that **extends, rather than competes with, the thesis**.

---

## 4. Thesis-incorporation options

Three models already planned: **EventSSMDetector** (CNN spatial + Mamba temporal), **PureSSMDetector** (SSM spatial + Mamba temporal). A spiking detector — **"SSSMDetector"** (Spiking State-Space Model Detector) — would complete a clean progression from accuracy-optimised → fully neuromorphic.

### Option 1 — Add **SSSMDetector** as a 4th model *(highest contribution, highest scope)*
- **What:** a spiking SSM detector reusing the same Gen1 pipeline, YOLO-PAFPN neck, and YOLOX head; the new part is a **spiking SSM backbone/temporal core** (surrogate-gradient trained, T timesteps), referencing EMS-YOLO/SpikeYOLO architectures.
- **Pros:** maximal novelty; completes the trade-off study; *is* the Loihi PoC; reuses all existing infra; clearly publishable.
- **Cons / risks:** SNN training is finicky (surrogate gradients, BPTT-through-spikes, timestep tuning, spike-vanishing); detection SNNs are immature; biggest time risk.
- **Three-model arc becomes four:** EventSSM (ANN accuracy) → PureSSM (pure SSM) → **SSSM (spiking, Loihi-deployable)**.

### Option 2 — Lighter-touch variants *(preserve all work, lower risk)*
- **2a — Spiking *temporal core* only:** keep the spatial backbone in ANN form; replace only the Mamba temporal block with a spiking SSM block. Smaller, faster, still a valid "spiking SSM" PoC and a clean ablation against EventSSM.
- **2b — PoC + future-work chapter:** a software spiking-SSM demo on a Gen1 *subset* (e.g. single-scale, or a classification proxy) — enough for the INRC application and a thesis "deployment" chapter, with full benchmarking framed as future work.
- **2c — Quantisation bridge:** show the existing SSM detector quantises toward spike-compatible (binary/low-bit activations) as a stepping stone, without a full SNN. Lowest risk, weakest novelty.

### Option 3 — Reframe the spine *(narrative pivot, minimal waste)*
- Reframe the thesis as **"Neuromorphic perception for event vision: from state-space to spiking state-space models, toward Loihi deployment."** EventSSM/PureSSM become the **ANN-SSM accuracy reference**; SSSMDetector becomes the **efficiency/deployability contribution**. Nothing already built is wasted — it is recast as the upper-accuracy anchor of the trade-off study.

---

## 5. Timeline reality (week 3 of 13)

~10 weeks remain. Rough budget if the current pace holds:

- EventSSMDetector: 400k training (running) + Stages 8–10 → ~3–4 weeks
- PureSSMDetector (reuses infra): ~2–3 weeks
- **SSSMDetector PoC** (snnTorch/SpikingJelly, reusing pipeline + head): ~3–4 weeks if scoped tightly

Doing **all four models fully benchmarked** in 10 weeks is ambitious but not impossible at the current pace. The **realistic, de-risked plan:** EventSSM (full) + PureSSM (full) as the guaranteed contributions, then **SSSMDetector at PoC level** (software SNN, possibly subset) as the high-upside extension + Intel application artefact, with Loihi *hardware* results as future work.

**Discipline that protects the thesis:** lock EventSSM (through Stage 8) and PureSSM **first** as completed contributions, *then* invest in SSSMDetector. The SNN is the high-upside extension, **never a load-bearing dependency** — if it proves hard, the thesis is still complete.

---

## 6. Recommendation & next steps

1. **Recommended path:** Option 1 (add SSSMDetector), executed with Option-2b discipline — i.e. commit to the 4th model but scope its first milestone as a PoC, and only deepen it once EventSSM + PureSSM are locked.
2. **Frame to the supervisor as:** *"a spiking extension of my SSM event-detector (SSSMDetector), prototyped in snnTorch with a Lava/Loihi path,"* which serves the thesis trade-off study **and** the INRC application in one artefact.
3. **Immediate, no-GPU-cost steps** (while the 400k run trains): deeper read of EMS-YOLO [3] and SpikeYOLO [4] (closest references — open code, Gen1-benchmarked); install/scope snnTorch or SpikingJelly; sketch how the spiking SSM block slots into the existing backbone interface.
4. **Decision needed (with supervisor):** which option (1 / 2 / 3), and whether SSSMDetector is "full model" or "PoC + future work." This is a scope/risk call, not a technical blocker.

---

## 7. References

- [1] U. Rançon *et al.*, "StereoSpike: Depth Learning with a Spiking Neural Network," *IEEE Access*, 2022. arXiv:2109.13751.
- [2] C. Lee *et al.*, "Spike-FlowNet: Event-Based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks," *ECCV*, 2020. arXiv:2003.06696.
- [3] Q. Su *et al.*, "Deep Directly-Trained Spiking Neural Networks for Object Detection" (EMS-YOLO), *ICCV*, 2023. arXiv:2307.11411.
- [4] X. Luo *et al.*, "Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-Performance and Energy-Efficient Object Detection" (SpikeYOLO), *ECCV*, 2024. arXiv:2407.20708.
- [5] "EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks," *ECCV*, 2024. arXiv:2403.12574.
- [6] "SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models," 2024. arXiv:2408.14909.
- [7] Y. Zhong *et al.*, "SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning," 2024. arXiv:2410.17268.
- [8] M. Bal and A. Sengupta, "P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks," 2024. arXiv:2406.02923.
- [9] Intel Corporation, "Intel Advances Neuromorphic with Loihi 2, New Lava Software Framework and New Partners," Sept. 2021.
- [10] J. Pedersen *et al.*, "Neuromorphic Intermediate Representation: A Unified Instruction Set for Interoperable Brain-Inspired Computing," 2023. arXiv:2311.14641.

> **BibTeX note (per thesis standards):** author lists for [4], [5], [6] should be verified and completed from the arXiv records before entry into the thesis `.bib` (avoid `and others`; protect acronyms in titles, e.g. `{StereoSpike}`, `{SNN}`, `{Loihi}`).
>
> **Full reference library:** `docs/research/spiking_ssm_references.md` (all session papers, full titles/authors/arXiv IDs/code). **Technical deep-dive + math:** `docs/research/Spiking_PureSSM_litreview_deepdive.md`.

---

## 8. Staging SSSMDetector as **future work** — the PoC + a handover roadmap

A deliberate, defensible scoping decision: **this thesis delivers SSSMDetector as a software proof-of-concept**, and stages the full neuromorphic deployment as **documented future work** — a turnkey handover so a successor (a next-year thesis student, or a PhD continuation) can pick up *from a working base*, ideally once Intel Loihi hardware is granted via INRC. This is good thesis practice (bounded, deliverable now) **and** strengthens the INRC application (it presents a roadmap, not a one-off demo).

### 8.1 What this thesis leaves the successor (the PoC baseline)
- A **trained spiking SSM event-detector** (Spiking-PureSSM) with Gen1 **mAP**, **SOPs/estimated-energy/firing-rate** numbers, and **Lava software-simulation** validation (Loihi-compatible, no chip needed) — see Appendices B–C of the deep-dive.
- The **controlled comparison** EventSSM vs PureSSM vs Spiking-PureSSM (identical neck/head/data/eval) — the scientific anchor.
- The **full reusable pipeline** (Gen1 data, YOLO-PAFPN, YOLOX head, evaluation, training/eval launchers) + the design docs + this **reference library**. A successor inherits a runnable system, not a blank page.

### 8.2 Handover roadmap (what a successor could do, in phases)
- **Phase I — Hardware deployment** (needs Loihi access): port the PoC from Lava-sim → real **Loihi 2**; measure on-chip **power / latency / throughput**; close the sim-to-hardware gap (fixed-point quantization, neuron-model fidelity, async event streaming). *This is the natural "first thing with the chip."*
- **Phase II — Architecture depth:** spike the **spatial BiMamba** (choice A), then the **full SiLIF neuron-state** (choice B), incl. **complex/oscillatory C-SiLIF**; ablate reset types, surrogate functions, timesteps; apply **integer/graded-spike** training (SpikeYOLO / Loihi-2) to close the ~7-mAP accuracy gap to ANN-SSM detectors.
- **Phase III — Task & input expansion:** **temporal-generalisation under spiking** (event-rate robustness, on-chip); **direct event streaming** to Loihi (drop the histogram, fully neuromorphic front-to-back); DSEC/stereo; **closed-loop micro-UAV** perception.
- **Phase IV — Efficiency science:** rigorous **measured** energy (on-chip SOPs/power), latency under asynchronous streaming, head-to-head vs GPU/edge accelerators — the complete SWaP case for micro-UAV deployment.

### 8.3 Why this is the right shape for *this* thesis
SSSMDetector could be a standalone thesis (a strength — it signals independent merit and seeds a publication/PhD). For Thesis B it is therefore the **third pillar / extension**: lock EventSSM (Stage 8) + PureSSM first as guaranteed contributions; deliver SSSMDetector at PoC depth; hand off Phases I–IV. The SNN is **high-upside, never load-bearing**.

---

## 9. Draft message to supervisor (direction statement)

> **Subject: Direction — incorporating spiking SSMs / Loihi into the thesis**
>
> Hi [supervisor],
>
> Thanks for the Loihi/INRC pointer — I've done a deep literature review and I'd like to take it on. My plan keeps the work I've already built and extends it rather than pivoting:
>
> - I keep my two SSM detectors (**EventSSMDetector**, **PureSSMDetector**) as the accuracy-focused, GPU-side contributions — the EventSSM full Gen1 run is training now.
> - I add a **third model, a *spiking* SSM detector (SSSMDetector)**, built on the PureSSM backbone with the SSM blocks made spiking. This gives a clean controlled comparison (non-spiking vs spiking SSM) and is the **proof-of-concept for the INRC application**.
> - Key point that de-risks it: a LIF spiking neuron is mathematically a state-space recurrence, so my existing SSM/Mamba work transfers directly — this is a principled reparametrization, not a new field from scratch. There's a 2024–2025 literature (SpikingSSMs/AAAI'25, SiLIF, SpikeMba) confirming the bridge, and SNN detection already exists on Gen1 (SpikeYOLO, SpikSSD), ~7 mAP behind the ANN baselines — narrowing that gap with a spiking *SSM* looks novel and publishable.
> - The PoC is **software-only** (trained on my GPU/Katana, validated in Intel's Lava simulator) — no hardware needed to apply. I'd stage the **on-chip Loihi deployment and deeper variants as future work / a handover** for a successor once hardware is granted.
>
> Two things I'd value your steer on: (1) scope — full benchmarked 4th model vs PoC-plus-future-work; (2) timing of the INRC application. I've written up the full review, math, and roadmap and can share the docs.
>
> [Benas]

*(Adapt freely — see `docs/research/Spiking_PureSSM_litreview_deepdive.md` and `docs/research/spiking_ssm_references.md` for the backing detail to attach or cite.)*
