# Augmentation Research: Second Dataset Selection + the Resolution Question

**For:** deciding how to extend the Gen1 SSM-detection thesis (PureSSM/EventSSM) beyond Gen1 — *augment, don't pivot*.
**Method:** ultracode deep-research workflow (37 datasets surveyed, 22 adversarially fact-checked) + close reading of
Gehrig & Scaramuzza 2022, *"Are High-Resolution Event Cameras Really Needed?"*. Date: 2026-07-16.

> **Read the verification caveats (§6) before acting** — two agents were safety-classifier-blocked (incl. the 1Mpx
> verify), and the completeness critic flagged two load-bearing eTraM claims as unverified.

---

## 1. Recommendation

**Primary: Prophesee 1 Megapixel Automotive Detection Dataset (1Mpx / Gen4).** Best on every axis:
- **1280×720** (~5.6× Gen1's 304×240) — the regime where a *large-object / receptive-field* claim becomes decisive.
- **Event-only, same automotive ego-motion domain, same Prophesee `.dat`/`_bbox.npy` format** → the existing
  stacked-histogram pipeline applies unchanged (minimal integration; reuse the cloud-5090 training workflow).
- **Richest SSM/RVT-family literature** of any option — RED, RVT, **S5-RVT / Zubić et al. 2024**, SAST, GET, HMNet,
  ASTMNet all report here → direct published baselines (incl. Zubić's own S5 numbers) at HD.
- **Denser labels (~60 Hz** vs Gen1's sparse rate) → strengthens the Stage-9 **rate-robustness** study too.
- Cost: ~25M boxes / 14.65 h HD streams — heavier to preprocess/store than Gen1's ~78 GB.

**Secondary (optional third leg): eTraM (CVPR 2024)** — HD 1280×720, event-only, **static/roadside (no ego-motion)**
→ a genuine *domain-shift* generalization test. **Caveat (see §6):** its headline selling point — that it carries
S5-ViT and SMamba (~32.6 mAP) baselines for direct SSM-vs-Mamba comparison — is **unverified and possibly wrong**;
confirm against the actual paper/leaderboard before relying on it.

**One-line verdict:** augment with **1Mpx** (it simultaneously stresses the large-object claim *and* enriches the
rate-robustness study, with published baselines to compare against); add **eTraM** only if you want a static-camera
domain-shift leg *and* the SSM-baseline claim checks out.

---

## 2. Comparison Table

(V) = adversarially verified in this run; (?) = unverified/inconsistent.

| Dataset | Year | Domain | Camera + Resolution | 2D-det labels? | SSM/RVT baselines | Thesis-fit |
|---|---|---|---|---|---|---|
| **1Mpx / Gen4** | 2020 | Automotive, ego-motion | Gen4, **1280×720**, event-only | **Yes** | RED, RVT, **S5-RVT/Zubić**, SAST, GET | **Best** |
| **eTraM** | 2024 | Traffic, **static** | EVK4 HD, **1280×720** (V) | **Yes** (V) | RVT, RED; **S5-ViT, SMamba (?)** | High (domain-shift; baseline claim unverified) |
| DSEC-Detection | 2024 | Automotive stereo | **Gen3.1, 640×480** (V) +RGB/LiDAR | **Yes** (V) | DAGr; few pure-SSM | Medium — matches your Gen3.1 sensor; only ~2.7× Gen1 |
| PKU-DAVIS-SOD | 2023 | Driving, E+RGB | DAVIS346, 346×260 | **Yes** | SODFormer (fusion) | Low — barely above Gen1 res |
| PEOD | 2025/26 | Challenging E-RGB | **1280×720** E+RGB (V) | **Yes** (V) | none (fusion-framed) | Medium — HD robustness set, no SSM baseline |
| EvDET200K | 2024 | **General** (10 cls) | EVK4-HD, **1280×720** (V) | **Yes** (V) | MvHeat-DET (niche) | Low — non-automotive |
| NU-AIR | 2023 | **UAV** aerial | **640×480** DVS (V) | **Yes** (V) | SNN+DNN (no SSM) | Low-Med — UAV-relevant |
| NeRDD / FRED | 2024/25 | UAV/anti-drone | EVK4 HD **1280×720** (V) | **Yes** (V) | none | Low — **single-class (drone)** |
| **Avoid →** | | | | | | |
| EV-UAV | 2025 | UAV tiny-obj | DAVIS346 | **No — segmentation** (V, corrected) | — | not bbox detection |
| DDD17/DDD20 | 2017/20 | steering | DAVIS346 | **No** (V) | — | no boxes |
| M3ED / 3EED | 2023/25 | robotics | Gen4 1280×720 | **No** 2D / 3EED=3D only | — | no 2D bbox |
| **OctoSense** | 2026 | automotive SSL | 640×480 | **No** (V, corrected) | — | **no detection GT** (confirms earlier call) |
| CoSEC | 2024 | automotive | DAVIS346 | **No** (depth/flow) | — | no boxes |
| Seeing Through Fog | 2020 | AD adverse-wx | **no event sensor** (V) | frame-based | — | not an event dataset |
| N-Caltech101 | 2015 | recognition | ATIS ≤240×180 | single-object | — | not multi-object scenes |
| MEVDT / TUMTraf-EMOT | 2024/25 | traffic | **240×180** (V) | Yes | — | **below Gen1 res** |

---

## 3. Top-candidate profiles (condensed)

**1Mpx / Gen4** — 7 classes (3 used for mAP: pedestrian, two-wheeler, car); 14.65 h (11.19/2.21/2.25 tr/val/te);
~25M boxes @~60 Hz. Directly tests the large-object hypothesis (5.6× pixels → trucks/buses/near-cars dominate the
frame, AP_L becomes decisive) with zero domain drift and the richest baseline set. *The obvious pick.*

**eTraM** — 8 classes; ~10 h; ~2M boxes (V); static EVK4 HD. Feature: domain-shift generalization ("holds beyond
ego-motion"). Requires class remap (8 vs 2) and fine-tune-vs-zero-shot must be stated. **Verify the SSM-baseline claim.**

**DSEC-Detection** — Gen3.1 640×480 stereo (**matches your physical Gen3.1 sensor**) + RGB/LiDAR; 390k boxes (V);
best if the real stereo camera is ever integrated + a fusion chapter is wanted. Only modest resolution gain, sparse
SSM comparability.

**UAV leg (ties to your micro-UAV framing):** **NU-AIR** (640×480, 2 classes, verified 2D boxes, SNN+DNN baselines)
is the most usable; NeRDD/FRED are HD but single-class drone with no SSM baselines.

---

## 4. The resolution question — Gehrig & Scaramuzza 2022 ("Are High-Resolution Event Cameras Really Needed?")

**Claim:** higher-res event cameras are *not* universally better. In **low-light + high-speed** conditions,
*lower*-res cameras **outperform** high-res ones (verified on reconstruction, optical flow, pose; sim + real Gen4).
**Mechanism (sensor physics, task-agnostic):** smaller pixels → brightness-derivative power law → **higher per-pixel
event rate** (Δt ∝ pixel_size^γ) → tiny timestamp errors (slow pixels in dark, fast motion) have bigger effect →
**more temporal noise / ghosting**. On real data, 640×360 beat full 1280×720 once speed > ~0.4 m/s.

**The nuance that matters for us:** the penalty is **trainable-away**. E-RAFT (learning-based, trained on real
noisy day+night data) **always benefits from high res**; E2VID (trained on clean synthetic) degrades. Their words:
*"we may overcome the limitations of noise at high resolutions by adopting a learning-based approach and training on
noisy night-time data … noisy training-data is the key."* **We train deep detectors on the target data → the high-res
penalty largely evaporates when we train on 1Mpx** (reinforces: train, don't zero-shot).

**The thesis connection (novel framing):** high-res's weakness is *temporal noise* (a rate problem); PureSSM's
proven superpower is *rate/temporal robustness* (Stage-9: 69.7% @ true-10×). → **The exact failure mode of high-res
cameras is what our SSM absorbs best.** Proposed framing:

> *"High-resolution event cameras suffer greater temporal noise (Gehrig & Scaramuzza, 2022). We show SSM detectors
> are the most temporally robust; we therefore hypothesise and test that SSM backbones are uniquely positioned to
> exploit high-resolution sensors — i.e. the high-res penalty shrinks for rate-robust architectures."*

**Concrete experiment (on 1Mpx):** does PureSSM's **AP_L advantage grow or shrink** vs Gen1? Competing forces: (↑)
more pixels per large object favour the wide-receptive-field BiMamba; (↓) more temporal noise. Net is empirical, and
our rate-robustness predicts PureSSM eats the noise cost better than EventSSM/S5-RVT → advantage should hold/grow.
**Caveat:** the paper studies low-level vision, not detection — cite it as *mechanism/motivation*, not a detection result.

---

## 5. Recent / emerging to watch
- **SEVD (CVPRW 2024)** — *(critic-flagged gap)* synthetic (CARLA), **1280×720**, 2D **and** 3D boxes, ego + static,
  RED/RVT baselines, same ASU group as eTraM → a synthetic-augmentation option. **Verify.**
- **SPECTRA (2025)** — HD stereo EVK4 multimodal, *lists* detection GT (behind access wall, unverified).
- **PEOD (AAAI 2026)** — HD pixel-aligned E-RGB, low-light/high-speed → future robustness benchmark.
- **EventVOT / VisEvent / FE108** — HD but single-object *tracking* (SOT), not multi-object detection (named-excluded).

## 6. Verification caveats (be honest in the write-up)
- **Safety-classifier-blocked verifiers:** the **1Mpx** and **PKU-DAVIS-SOD** fact-check agents were blocked
  (transient). 1Mpx facts come from the search agents + are well-documented (RED/RVT papers) → reliable, but not
  independently re-verified this run. **Gen1 was verified:** 304×240, ~255k boxes, labels 1–4 Hz.
- **Two unverified, load-bearing eTraM claims** (completeness critic): that eTraM carries an **S5-ViT (Zubić)**
  baseline and an **SMamba ~32.6 mAP** baseline. If false, eTraM's "SSM-vs-Mamba comparability" rationale collapses →
  **verify before committing to eTraM.**
- **Label-rate contrast** "1Mpx ~60 Hz vs Gen1 1–4 Hz" may overstate — Gen1 is sometimes cited up to ~20 Hz; verify.
- **Gap:** SEVD omitted from the main survey (add if a synthetic leg is wanted).

## 7. Citations
- 1Mpx: https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/ · RED (Perot 2020) https://arxiv.org/abs/2009.13436 · toolbox https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
- Gen1: https://arxiv.org/abs/2001.08499
- eTraM: https://eventbasedvision.github.io/eTraM/ · https://arxiv.org/abs/2403.19976 · https://github.com/eventbasedvision/eTraM
- DSEC-Detection: https://dsec.ifi.uzh.ch/dsec-detection/ · https://github.com/uzh-rpg/dsec-det
- PKU-DAVIS-SOD: https://www.pkuml.org/research/pku-davis-sod-dataset.html · PEOD https://echosliu.github.io/PEOD-dataset/ · EvDET200K https://github.com/Event-AHU/OpenEvDET
- UAV: NU-AIR https://arxiv.org/abs/2302.09429 · NeRDD https://arxiv.org/abs/2409.16099 · FRED https://arxiv.org/abs/2506.05163 · EV-UAV(seg) https://arxiv.org/abs/2506.23575
- SEVD https://arxiv.org/abs/2404.10540 · SPECTRA https://openreview.net/forum?id=mqCUxEsfsI
- Avoid: DDD17 https://arxiv.org/abs/1711.01458 · M3ED https://m3ed.io/ · OctoSense https://arxiv.org/abs/2606.27317 · Seeing-Through-Fog https://arxiv.org/abs/1902.08913
- **Gehrig & Scaramuzza 2022, "Are High-Resolution Event Cameras Really Needed?"** — project: https://uzh-rpg.github.io/eres/ (local PDF: `~/Desktop/arxiv22_Gehrig.pdf`)
