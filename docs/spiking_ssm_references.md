# Reference Library — Spiking SSMs, Event-SNN Detection & Neuromorphic Hardware

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis** · compiled 2026-06-18

Master citation list for the spiking-SSM / Loihi thesis direction (companion to `SSSMDetector_Loihi_opportunity.md` and `Spiking_PureSSM_litreview_deepdive.md`). Full titles, authors (where verified), venues, arXiv IDs, and code. **Items marked ⚠️verify** still need author-list / ID confirmation before entry into the thesis `.bib`. Per thesis BibTeX standards: protect acronyms in titles (`{SNN}`, `{SSM}`, `{LIF}`, `{S4D}`, `{Mamba}`, `{Loihi}`, `{SiLIF}`), use full author lists (no `and others`).

---

## A. SSM foundations & the thesis baseline

| Key | Title | Authors | Venue / arXiv | Code |
|---|---|---|---|---|
| Zubić2024 | State Space Models for Event Cameras (S5-RVT) | N. Zubić, M. Gehrig, D. Scaramuzza | CVPR 2024 · arXiv:2402.15584 | uzh-rpg |
| Gehrig2023 | Recurrent Vision Transformers for Object Detection with Event Cameras (RVT) | M. Gehrig, D. Scaramuzza | CVPR 2023 · arXiv:2212.05598 | uzh-rpg/RVT |
| Mamba2023 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | A. Gu, T. Dao | 2023 · arXiv:2312.00752 | state-spaces/mamba |
| S4-2022 | Efficiently Modeling Long Sequences with Structured State Spaces (S4) | A. Gu, K. Goel, C. Ré | ICLR 2022 · arXiv:2111.00396 | state-spaces/s4 |
| S4D-2022 | On the Parameterization and Initialization of Diagonal State Space Models (S4D) | A. Gu, A. Gupta, K. Goel, C. Ré | NeurIPS 2022 · arXiv:2206.11893 | — |
| S5-2023 | Simplified State Space Layers for Sequence Modeling (S5) | J. T. H. Smith, A. Warrington, S. W. Linderman | ICLR 2023 · arXiv:2208.04933 | lindermanlab/S5 |

## B. Spiking foundations (training)

| Key | Title | Authors | Venue / arXiv |
|---|---|---|---|
| Neftci2019 | Surrogate Gradient Learning in Spiking Neural Networks | E. O. Neftci, H. Mostafa, F. Zenke | IEEE Sig. Proc. Mag. 2019 · arXiv:1901.09948 |

## C. Structured / selective spiking SSMs — **the bridge (core of the new direction)**

| Key | Title | Authors | Venue / arXiv | Code |
|---|---|---|---|---|
| **SpikingSSMs** | SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models | S. Shen, C. Wang, R. Huang, Y. Zhong, Q. Guo, Z. Lu, J. Zhang, L. Leng | **AAAI 2025** · arXiv:2408.14909 | shenshuaijie/SDN |
| **SiLIF** | SiLIF: Structured State Space Model Dynamics and Parametrization for Spiking Neural Networks | ⚠️verify | 2025 · arXiv:2506.06374 | — |
| **SPikE-SSM** | SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning | Y. Zhong *et al.* ⚠️verify | 2024 · arXiv:2410.17268 | — |
| **P-SpikeSSM** | P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks ("Rethinking SNNs as State Space Models") | M. Bal, A. Sengupta | 2024 · arXiv:2406.02923 | — |
| Spiking-S4 | Spiking Structured State Space Model for Monaural Speech Enhancement | ⚠️verify | 2023 · arXiv:2309.03641 | — |
| **SpikeMba** | SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding | W. Li, X. Hong, R. Xiong, X. Fan | 2024 · arXiv:2404.01174 | — |
| SpikingMamba | SpikingMamba: Towards Energy-Efficient Large Language Models via Knowledge Distillation from Mamba | ⚠️verify | 2025 · arXiv:2510.04595 | — |
| SmolMamba | Vision SmolMamba: Spike-Guided Token Pruning for Energy-Efficient Spiking State-Space Vision Models | ⚠️verify | 2026 · arXiv:2604.25570 | — |

## D. SNN object detection (event cameras / Gen1) — **the task side**

| Key | Title | Authors | Venue / arXiv | Gen1 mAP@0.5:0.95 | Code |
|---|---|---|---|---|---|
| **EMS-YOLO** | Deep Directly-Trained Spiking Neural Networks for Object Detection | Q. Su *et al.* ⚠️verify | ICCV 2023 · arXiv:2307.11411 | 0.267–0.310 | — |
| **SpikeYOLO** | Integer-Valued Training and Spike-Driven Inference SNN for High-performance and Energy-efficient Object Detection | X. Luo *et al.* (5 auth.) ⚠️verify | ECCV 2024 (Oral/Best-Paper Cand.) · arXiv:2407.20708 | 0.385 | BICLab/SpikeYOLO |
| **SpikSSD** | SpikSSD: Better Extraction and Fusion for Object Detection with Spiking Neuron Networks | Y. Fan *et al.* ⚠️verify (note: arXiv title may read "SpikeDet") | 2025 · arXiv:2501.15151 | **0.408 (current SNN SOTA)** | yimeng-fan/SpikSSD |
| EAS-SNN | EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent SNNs | ⚠️verify | ECCV 2024 · arXiv:2403.12574 | — | — |
| HybridSpikeViT | Hybrid Spiking Vision Transformer for Object Detection with Event Cameras | ⚠️verify | ICML 2025 · arXiv:2505.07715 | — | — |

## E. Event-vision SNNs — dense prediction (supervisor's suggestions)

| Key | Title | Authors | Venue / arXiv | Code |
|---|---|---|---|---|
| **StereoSpike** | StereoSpike: Depth Learning with a Spiking Neural Network | U. Rançon *et al.* ⚠️verify | IEEE Access 2022 · arXiv:2109.13751 | urancon/StereoSpike |
| **Spike-FlowNet** | Spike-FlowNet: Event-Based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks | C. Lee, A. K. Kosta, A. Z. Zhu, K. Chaney, K. Daniilidis, K. Roy | ECCV 2020 · arXiv:2003.06696 | chan8972/Spike-FlowNet |
| EV-FlowNet | EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras (original ANN) | A. Z. Zhu, L. Yuan, K. Chaney, K. Daniilidis | RSS 2018 · arXiv:1802.06898 | daniilidis-group/EV-FlowNet |
| Spiking-UNet | Spiking-UNet: SNN-based image segmentation | ⚠️verify | ⚠️verify | SNNresearch/Spiking-UNet |

## F. Neuromorphic hardware & deployment (Loihi / Lava)

| Key | Title | Authors | Venue / arXiv |
|---|---|---|---|
| Loihi2018 | Loihi: A Neuromorphic Manycore Processor with On-Chip Learning | M. Davies *et al.* | IEEE Micro 2018 |
| Loihi2-2021 | Efficient Neuromorphic Signal Processing with Loihi 2 | G. Orchard *et al.* ⚠️verify | IEEE SiPS 2021 · arXiv:2111.03746 |
| LavaIntel2021 | Intel Advances Neuromorphic with Loihi 2, New Lava Software Framework | Intel Corp. (press) | 2021 |
| DaviesSurvey2021 | Advancing Neuromorphic Computing with Loihi: A Survey of Results and Outlook | M. Davies *et al.* | Proc. IEEE 2021 |
| NIR2023 | Neuromorphic Intermediate Representation: A Unified Instruction Set for Interoperable Brain-Inspired Computing | J. Pedersen *et al.* ⚠️verify | Nature Comms 2024 · arXiv:2311.14641 |
| Horowitz2014 | Computing's Energy Problem (and What We Can Do About It) — source of MAC/AC pJ energy estimates | M. Horowitz | ISSCC 2014 |

## G. Software tooling (not papers, but for methods/reproducibility)
- **snnTorch** — PyTorch SNN training (surrogate gradients). `jeshraghian/snntorch`.
- **SpikingJelly** — PyTorch SNN framework. `fangwei123456/spikingjelly`.
- **Lava / Lava-DL** — Intel's open-source neuromorphic framework (Loihi). `lava-nc/lava`.
- **mamba-ssm / causal-conv1d** — the selective-SSM kernels already built for `sm_120` in this project.

---

### Key positioning facts captured this session
- **Gen1 detection landscape (COCO mAP@0.5:0.95):** S5-RVT **47.7** (ANN-SSM, your baseline) ▸ SpikSSD **40.8** ▸ SpikeYOLO **38.5** ▸ EMS-YOLO **26.7–31.0** (SNNs). **SNN↔ANN gap ≈ 7 mAP** (best SNN = SpikSSD).
- **The bridge is real & current:** LIF neuron ≡ 1-state diagonal SSM + spike/reset readout; formalized by SiLIF (LIF↔S4) and operationalized by SpikingSSMs (S4D + LIF + Surrogate Dynamic Network for parallel training, AAAI 2025).
- **Novelty niche:** spiking SSMs exist on sequence/speech/LM; spiking Mamba exists (SpikeMba, video grounding); SNN detectors exist on Gen1 (spiking CNN/ViT backbones). A **spiking structured/selective-SSM *detector* for event vision, as a controlled swap from a non-spiking SSM detector,** appears unoccupied.
