# Roadmap — Stage Files Index
**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis**

The stage-by-stage narrative of the whole investigation: **EventSSMDetector = Stages 0–10**, **PureSSMDetector =
Stages 11–16**. Both investigations are **complete**. Each file is a self-contained record of one stage (Stages
0–10 were written as forward plans; Stages 11–16 are written as completed-stage records with the real results and
pointers to the authoritative plan/spec/results/notes docs).

---

## How to Use These Files

When working with Claude on a specific stage, share that stage's file as context. For the cross-cutting synthesis,
see `docs/results/Thesis_Progress_Writeup.md`.

**System legend:**
- 🍎 = Mac (planning, documentation)
- 🐧 = Linux PC with RTX 5070 Ti (implementation, evaluation)
- ☁️ = rented RTX 5090 (cloud training, Stages 13–14)

---

## Stage Files

| File | Stage | System | Est. Time | Status |
|---|---|---|---|---|
| `Stage_00_Design_Lock_In.md` | Design decisions (backbone, head, temporal module) | 🍎 | 1–2 days | ✅ |
| `Stage_01_Architecture_Blueprint.md` | Tensor shape diagram for all components | 🍎 | 1 day | ✅ |
| `Stage_02_Codebase_Audit.md` | File inventory, mamba-ssm install, baseline verify | 🐧 | 0.5–1 day | ✅ |
| `Stage_03a_ResNet18_Backbone.md` | Modified ResNet-18 with 10-channel input | 🐧 | 1–2 days | ✅ |
| `Stage_03b_FPN.md` | Feature Pyramid Network | 🐧 | 1 day | ✅ |
| `Stage_03c_Mamba_Temporal.md` | Mamba temporal module (most complex) | 🐧 | 2–4 days | ✅ |
| `Stage_03d_Detection_Head.md` | YOLOX head verification | 🐧 | 0.5 day | ✅ |
| `Stage_04_Integration.md` | Wire all components, state management | 🐧 | 3–5 days | ✅ |
| `Stage_05_Smoke_Testing.md` | Overfit test, gradients, memory, speed | 🐧 | 2–3 days | ✅ |
| `Stage_06_Short_Training.md` | 20 epochs on 10% Gen1, preliminary results | 🐧 | 1–3 hrs compute | ✅ |
| `Stage_07_Full_Training.md` | 100 epochs full Gen1, primary result | 🐧 | 12–24 hrs compute | ✅ |
| `Stage_08_Evaluation.md` | Test set evaluation, comparison table | 🐧 | 2–3 days | ✅ |
| `Stage_09_Temporal_Generalisation.md` | Variable rate evaluation (0.25×–4×) | 🐧 | 1–2 days | ✅ |
| `Stage_10_Efficiency_Benchmarking.md` | Params, FLOPs, latency, VRAM | 🐧 | 1–2 days | ✅ |
| — *EventSSM complete (test/AP 46.2)* — | — | — | — | — |
| `Stage_11_PureSSM_Backbone.md` | BiMamba spatial backbone build (8.38 M, 27 tests) | 🐧 | — | ✅ |
| `Stage_12_PureSSM_Integration.md` | Hydra-selectable + overfit smoke (6.0× PASS) | 🐧 | — | ✅ |
| `Stage_13_PureSSM_Cloud_Short_Run.md` | 25k sanity run (val/AP 0.351, flow de-risked) | ☁️ | — | ✅ |
| `Stage_14_PureSSM_Full_Training.md` | 400k full run (best val/AP 0.48 @ 310k, ~27 h) | ☁️ | — | ✅ |
| `Stage_15_PureSSM_Evaluation.md` | Gen1 test eval (test/AP 46.43, **AP_L +2.95**) | 🐧 | — | ✅ |
| `Stage_16_PureSSM_Pillars_and_Visuals.md` | Efficiency + robustness + CUDA-graph + videos | 🐧 | — | ✅ |

**Total estimated time:** 6–8 weeks (both models now complete)

---

## Key Files to Create (New Code)

| File | Stage |
|---|---|
| `models/backbone/resnet18_event.py` | 3a |
| `models/fpn/event_fpn.py` | 3b |
| `models/temporal/mamba_temporal.py` | 3c |
| `models/event_ssm_detector.py` | 4 |

**Everything else is reused from the S5-RVT baseline codebase.**

---

## Key Papers Referenced

| Paper | Filename | Used In Stage |
|---|---|---|
| Zhu et al. 2019 (Voxel Grids) | `Zhu_2019_VoxelGrids_UnsupOpticalFlow_CVPR.pdf` | 0 |
| Gallego et al. 2022 (Event Survey) | `Gallego_2022_EventVision_Survey_TPAMI.pdf` | 0, 9 |
| Gu et al. 2023 (Mamba) | `Gu_2023_Mamba_SelectiveStateSpaces_arXiv.pdf` | 0, 3c, 9 |
| Smith et al. 2023 (S5) | `Smith_2023_S5_SimplifiedSSM_ICLR.pdf` | 0, 3c |
| Zubic et al. 2024 (S5-RVT) | `Zubic_2024_SSM_EventCameras_CVPR.pdf` | 0, 3c, 4, 9 |
| Yang et al. 2025 (SMamba) | `Yang_2025_SMamba_EventDetection_AAAI.pdf` | 0, 8 |
| Gehrig et al. 2023 (RVT) | `Gehrig_2023_RVT_EventDetection_CVPR.pdf` | 0, 3b, 6, 7 |
| Perot et al. 2020 (Gen1) | `Perot_2020_Gen1_1Mpx_Detection_NeurIPS.pdf` | 1, 3b, 8 |
| Zheng et al. 2023 (DL Survey) | `Zheng_2023_DeepLearningEventVision_Survey_arXiv.pdf` | 3a |
| Floreano et al. 2015 (Drones) | `Floreano_2015_FutureSmallDrones_Nature.pdf` | 0, 10 |
| Niculescu et al. 2022 (Nano-drone) | `Niculescu_2022_NanoDroneDNN_Deployment_JETCAS.pdf` | 0, 5, 10 |
| Falanga et al. 2019 (Avoidance) | `Falanga_2019_PerceptionLatency_SenseAvoid_RAL.pdf` | 3c, 5, 10 |
| Li et al. 2024 (Drone avoidance) | `Li_2024_TamingEventCameras_DroneAvoidance_MobiCom.pdf` | 8, 10 |
| Santos et al. 2026 (UAV Review) | `Santos_2026_EventVisionUAV_SystematicReview_Sensors.pdf` | 5, 9, 10 |
| Davies et al. 2021 (Loihi) | `Davies_2021_Loihi_NeuromorphicComputing_ProcIEEE.pdf` | 10 |

---

## After EventSSMDetector Is Complete

Build **PureSSMDetector** — the pure SSM spatial model.

Changes from EventSSMDetector:
- Replace `ResNet18EventBackbone` with `BiMambaBackbone` (Bidirectional Mamba over 16×16 patch tokens)
- Keep FPN, Mamba temporal module, detection head **identical**

This gives you the controlled ablation:
- EventSSMDetector: CNN spatial + Mamba temporal
- PureSSMDetector: SSM spatial + Mamba temporal  
- S5-RVT: Transformer spatial + S5 temporal

Any mAP difference between EventSSMDetector and PureSSMDetector is due solely to the spatial extractor choice.
