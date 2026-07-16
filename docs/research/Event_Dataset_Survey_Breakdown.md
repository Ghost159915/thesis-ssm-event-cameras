# Event-Vision Dataset Survey — Full Breakdown (37 surveyed, 22 fact-checked)

Companion to `Augmentation_Dataset_and_Resolution_Research.md`. This lists **what each dataset is, why it was made,
its camera + resolution, and whether it is actually built for 2D object detection** (the thesis's hard requirement).
Facts (V) = adversarially verified in the deep-research run; (?) = unverified. Fuller citations in the companion doc.

**Legend — "Detection?":** ✅ = native 2D bounding-box detection · ⚠️ = boxes exist but caveated (retrofit / derived /
single-class / inherited) · ❌ = not a detection dataset (segmentation / flow / depth / classification / no event sensor).

---

## A. Core automotive event-**detection** benchmarks (the relevant family)

| Dataset | Camera + Resolution | Why it was made (purpose) | Detection? |
|---|---|---|---|
| **Gen1** (Prophesee, 2020) — *your current dataset* | ATIS Gen1, **304×240**, event-only | First large in-car automotive event-detection benchmark; car + pedestrian detection. ~39 h, ~255k boxes @1–4 Hz. | ✅ (2 classes) |
| **1Mpx / Gen4** (Prophesee, 2020) — *★ recommended augmentation* | Gen4, **1280×720**, event-only | Push event detection to **HD**; 7 classes (3 for mAP). 14.65 h, ~25M boxes @~60 Hz. Canonical HD companion to Gen1. | ✅ |
| **eTraM** (ASU, CVPR 2024) | EVK4 HD, **1280×720** (V), event-only | Event-based **traffic monitoring from a STATIC roadside camera** (ITS/infrastructure). ~10 h, ~2M boxes (V), 8 classes. | ✅ (static domain) |
| **DSEC-Detection** (UZH, 2024; base 2021) | **Gen3.1, 640×480** (V) stereo + RGB + LiDAR | Base **DSEC** built for stereo depth/flow driving; **detection labels added later** (QDTrack + manual). 390k boxes (V), 8 classes. | ✅ (retrofit) |
| **PKU-DAVIS-SOD** (PKU, 2023) | DAVIS346, **346×260**, event+RGB | **Event+RGB fusion detection** research (SODFormer). 220 seq, ~1.08M boxes @25 Hz, 3 classes. | ✅ (fusion-framed) |
| **PKU-DDD17-CAR** (2019) | DAVIS346, **346×260** | **Car-only** detection labels retrofitted onto DDD17 (a steering dataset). 3,155 seq (V). Superseded by PKU-DAVIS-SOD. | ⚠️ (retrofit, 1 class) |

## B. Recent HD detection (2024–2026)

| Dataset | Camera + Resolution | Why it was made | Detection? |
|---|---|---|---|
| **PEOD** (AAAI 2026) | **1280×720** pixel-aligned event+RGB (V) | First large HD **pixel-aligned E-RGB detection** benchmark under **low-light / overexposure / high-speed**; robustness focus. ~340k boxes (V), 6 classes; 14 detectors benchmarked. | ✅ (fusion-framed; has event-only config; **no SSM baseline**) |
| **EvDET200K / OpenEvDET** (CVPR 2025) | EVK4-HD, **1280×720** (V), event-only | Broaden event detection **beyond driving** — 10 everyday-object classes. 10,054 clips, ~202k boxes (V). Baseline: MvHeat-DET (niche). | ✅ (general domain) |
| **SEVD** (ASU, CVPRW 2024) — *critic-flagged gap* | CARLA synthetic, **1280×720** | **Synthetic** multi-view event detection (ego + fixed), **2D and 3D** boxes; scalable training data. Same group/tooling as eTraM; RED/RVT baselines. | ✅ (synthetic) |
| **SPECTRA** (2025) | Stereo EVK4, **1280×720**, event+RGB+LiDAR | HD stereo multimodal AD; abstract **lists** detection GT (details behind access wall). | ⚠️ (stated, **unverified**) |
| **EventDrive** (CVPR 2026) | mixed Gen3.1/Gen4/DAVIS346 | Large **VLM driving QA** benchmark; **inherits** 2D boxes from DSEC-Det / PKU. Go to the source sets for clean detection. | ⚠️ (inherited) |

## C. UAV / anti-drone (ties to the micro-UAV framing)

| Dataset | Camera + Resolution | Why it was made | Detection? |
|---|---|---|---|
| **NU-AIR** (2023) | DVS, **640×480** (V) | **Aerial urban** event detection from a drone (pedestrians/vehicles). 70.75 min, 93k boxes (V) @30 Hz, 2 classes; SNN+DNN baselines. | ✅ (best usable UAV set; no SSM baseline) |
| **NeRDD** (2024) | EVK4 HD, **1280×720** (V), event+RGB | **Anti-drone** (drone detection), synced E+RGB. 3.5 h, 115 videos (V). | ⚠️ (1 class; no SSM/RVT baseline) |
| **FRED** (2025) | EVK4 HD, **1280×720** (V), event+RGB | **Anti-drone** fast-moving drone detection. >7 h, 231 seq (V). | ⚠️ (1 class; no SSM baseline) |
| **EV-UAV / EVSOD** (2025) | DAVIS346, **346×260** (V) | Tiny-UAV benchmark — but ground truth is **per-event SEGMENTATION masks**, not bboxes (corrected verdict). | ❌ (segmentation) |

## D. Other / robotics / roadside (mostly weak fit)

| Dataset | Camera + Resolution | Why it was made | Detection? |
|---|---|---|---|
| **MEVDT** (2024) | DAVIS240c, **240×180** (V), event+grayscale | Campus-traffic vehicle **detection + tracking** (multimodal). ~10k boxes (V). | ⚠️ (**below Gen1 res**) |
| **PEDRo** (2023) | DAVIS346, **346×260** (V) | **Person** detection for robotics. 119 rec, 43k boxes (V). | ⚠️ (1 class, ~Gen1 res) |
| **TUMTraf Event** (2024) | Imago EB, **640×480** (V), event+RGB | Roadside ITS event+RGB detection. 4,111 pairs, 50k boxes (V), 7 classes. | ✅ (roadside, fusion-framed, no SSM baseline) |
| **TUMTraf EMOT** (2025) | DAVIS240, **240×180** (V) | Roadside multi-object **tracking**. Release/counts unclear. | ⚠️ (low res, unclear) |
| **MTevent** (CVPRW 2025) | Stereo DVXplorer, **640×480** (V) | **Indoor robotics** 6D-pose / manipulation; 2D boxes **projected** from 6D pose for 16 rigid objects. | ⚠️ (derived boxes, industrial) |

## E. NOT object-detection datasets — surveyed and **excluded** (the "avoid" list)

| Dataset | Camera + Resolution | What it actually is (purpose) | Detection? |
|---|---|---|---|
| **DDD17 / DDD20** (2017/2020) | DAVIS346, 346×260 | End-to-end **steering-angle prediction** for driving. 12 h / 51 h. No boxes (only via PKU-DDD17-CAR relabel). | ❌ (steering) |
| **M3ED** (2023) / **3EED** (2025) | Stereo Gen4, 1280×720 + LiDAR | Multi-robot **SLAM / optical-flow / segmentation**. TB-scale. 3EED derivative = **3D** boxes only. | ❌ (no 2D bbox) |
| **OctoSense** (2026) | Stereo SilkyEV, **640×480** (V) + multimodal | Automotive **self-supervised**; GT = depth / flow / segmentation / ego-motion (YOLO used only internally). | ❌ (**no detection GT** — confirms earlier call) |
| **CoSEC** (2024) | DAVIS346 coaxial event+RGB | All-day driving **depth / optical flow** fusion. | ❌ (no boxes) |
| **Seeing Through Fog** (2020) | RGB / LiDAR / radar / gated-NIR / FIR | Adverse-weather AD detection — but **no event/DVS sensor at all**. | ❌ (not an event dataset) |
| **N-Caltech101** (2015) | ATIS, ≤240×180 (V) | Neuromorphic **object recognition** (single centered saccaded object). 8,246 samples, 1 box each. | ❌ (single-object classification) |
| **MVSEC** (2018) | DAVIS346 stereo | Stereo automotive **optical flow / depth / VIO**. | ❌ (no boxes) |

---

## How to read this for the thesis
- **Hard filter = "Detection? ✅".** That immediately removes DDD17/20, M3ED/3EED, OctoSense, CoSEC, STF, N-Caltech101,
  MVSEC (Section E) and EV-UAV (segmentation).
- **Then filter for resolution > Gen1** (drops MEVDT/EMOT at 240×180; PEDRo/PKU/DDD17-CAR ≈ Gen1).
- **Then filter for literature comparability** (published RVT/SSM baselines) → **1Mpx** stands alone; eTraM/SEVD next.
- **Purpose matters for the *story*:** 1Mpx = same-domain resolution scale-up (cleanest); eTraM = static-camera
  domain-shift; DSEC-Det = matches your physical Gen3.1 stereo sensor; NU-AIR = your UAV framing.

**Bottom line unchanged:** **1Mpx** is the one dataset that is HD + event-only + same-domain + format-compatible +
richly-benchmarked. Everything else trades away at least one of those.
