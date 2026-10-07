# State-Space Models for Event-Camera Object Detection — Thesis Progress Write-Up

**Thesis B · MMAN4952 · UNSW Sydney · Benas Vaiciulis** · compiled 2026-07-16

> A single-document synthesis of the work to date: what was built, what was measured, what it means, and
> where it goes next. It stitches together the per-stage result docs (Stages 8–16) into one thesis-shaped
> narrative so a reader (supervisor / examiner / future-you) can see the whole arc without reading ten files.
> Numbers here are the measured, reproducible values from the stage docs cited under each section.

---

## 0. One-paragraph summary

Modern event-camera object detectors are dominated by two backbone families: recurrent-CNN hybrids (RVT,
ConvLSTM) and, since 2024, state-space models (S5-RVT, Zubić et al.). This thesis asks a **controlled**
question the literature has not: *if you hold the entire detection pipeline fixed and swap only the spatial
mixer of the backbone, what does a state-space model actually buy you?* Starting from an exact reproduction
of the S5-RVT baseline (**test/AP 47.7**, matching the paper), I built two of my own backbones —
**EventSSM** (ResNet-18 conv spatial + Mamba temporal) and **PureSSM** (BiMamba spatial + Mamba temporal) —
that drop into the verified baseline's neck, head, loss, data pipeline and evaluator **unmodified**. The
result is a clean four-model ablation (RVT → S5-RVT → EventSSM → PureSSM) in which any measured difference is
attributable to a single, isolated architectural change. Across three evidence pillars — **accuracy**,
**temporal robustness**, and **efficiency** — the pure state-space detector emerges as the **smallest**
model (16.33 M params), the **most rate-robust** (69.7 % accuracy retained at a true 10× event-rate change),
the most **compute-efficient** (best mAP-per-GFLOP), and the **best on large objects** (+2.95 AP_L over the
conv hybrid), at the cost of overall accuracy essentially tied with the conv hybrid and ~1.3 mAP behind the
ViT baseline, plus a wall-clock latency penalty that CUDA-graph deployment largely repairs.

---

## 1. Motivation and research question

Event cameras (DVS) report per-pixel brightness changes asynchronously, with microsecond latency, very high
dynamic range (120–140 dB), and no motion blur — properties that make them attractive for **micro-UAV and
automotive perception under tight SWaP (size-weight-and-power) constraints**. But their asynchronous,
variable-rate output is a poor fit for the frame-synchronous CNNs and Transformers that dominate vision.

Two sequence-model families have been proposed to bridge this gap:
- **Recurrent CNNs** (RVT's ConvLSTM) — proven but bounded by the recurrence's fixed time-constant.
- **State-space models** (S4/S5/Mamba) — linear-time sequence modelling with a continuous-time
  parameterisation, which in principle generalises across variable event rates *by construction*.

Zubić et al. (2024) showed an SSM backbone (S5-RVT) is competitive on Gen1 and claimed a headline
**temporal-generalisation** advantage (robustness when the inference event rate differs from training). That
claim, and the broader "SSMs are the right tool for event vision" thesis, are the starting point.

**The gap this thesis fills:** existing SSM-detection results confound *many* changes at once (backbone,
tokeniser, training recipe). The core research question here is narrower and cleaner:

> **Holding the entire detection pipeline fixed, what is the isolated effect of the backbone's spatial mixer
> — convolution vs. state-space — on accuracy, temporal robustness, and efficiency for event-camera object
> detection?**

---

## 2. Method — a controlled four-model ablation

### 2.1 The shared, frozen pipeline

Every model in the study reuses the **verified S5-RVT/RVT** infrastructure *unmodified* and Hydra-selectable:
- **Input representation:** 20-channel stacked histogram `(20, 240, 304)` (2 polarities × 10 temporal bins,
  `dt=50 ms`), zero-padded by the pipeline to `(20, 256, 320)`. Feature maps at strides 8/16/32.
- **Neck:** YOLO-PAFPN. **Head:** YOLOX decoupled head. **Loss:** BCE (cls+obj) + IoU loss `1−iou²` (×5) +
  SimOTA assignment. **Data:** Prophesee Gen1 (cars + pedestrians). **Eval:** Prophesee/COCO evaluator.
- **Training:** PyTorch-Lightning, 400k-step OneCycle, bf16, full Gen1.

Because only the **backbone** changes, and between EventSSM and PureSSM only the **spatial mixer** changes,
every reported delta is causally attributable to that single swap. This is the methodological core of the
thesis — a genuinely controlled experiment, which the prior literature does not provide.

### 2.2 The four models

| Model | Spatial mixer | Temporal mixer | Role |
|---|---|---|---|
| **RVT** | MaxViT (local+grid attention) | ConvLSTM | Literature baseline (recurrent CNN) |
| **S5-RVT** (Zubić 2024) | MaxViT | S5 (diagonal SSM) | Reproduced reference; the number to match |
| **EventSSM** (mine) | ResNet-18 conv (3→20-ch adapted, ImageNet-init) | Mamba (causal, per-pixel time axis) | CNN–SSM hybrid |
| **PureSSM** (mine) | **BiMamba** (bidirectional SSM scan) | Mamba (identical to EventSSM) | **Pure** state-space detector |

The Mamba temporal block is **interleaved per backbone stage** (one after each of the 4 stages), the
sequence axis is **time** (per spatial location, causal), and streaming state is carried across clips via the
baseline's `LstmStates` contract — mirroring the original `RNNDetectorStage`. Kernels: official
`mamba-ssm==2.3.2.post1` + `causal-conv1d==1.6.2.post1`, built for Blackwell `sm_120`. EventSSM and PureSSM
share the **identical** Mamba temporal path — so any difference between *them* isolates the **spatial mixer**
(ResNet conv vs BiMamba scan) with surgical precision.

### 2.3 Evidence pillars

Each own-model is evaluated on three pillars, mirrored across EventSSM and PureSSM so they are directly
comparable:
1. **Accuracy** — Gen1 test-set COCO mAP, per-class, and per-size (AP_S/M/L).
2. **Temporal robustness** — two regimes (fixed-cadence window sweep + true rate change), the thesis's
   distinctive contribution.
3. **Efficiency** — params, FLOPs, mAP/GFLOP, streaming latency (eager + CUDA-graph), energy, state size.

Plus **mechanism** evidence (effective-receptive-field probe) and **qualitative** GT-vs-prediction visuals.

---

## 3. Results

All numbers are Gen1 **test split**, COCO protocol (mAP @ IoU 0.50:0.95), same evaluator for all models.

### 3.1 Accuracy (Stages 8 & 15)

| Metric | S5-RVT baseline | EventSSM | **PureSSM** |
|---|---|---|---|
| **mAP @[.50:.95]** | **47.72** | 46.22 | **46.43** |
| AP_50 | 75.28 | 74.54 | 74.09 |
| AP_75 | 49.76 | 48.08 | 48.25 |
| AP_S (small) | 38.81 | 37.42 | 37.50 |
| AP_M (medium) | 54.92 | 53.68 | 53.96 |
| **AP_L (large)** | **50.66** | 44.70 | **47.65** |
| Car / Ped | 63.88 / 31.56 | 61.67 / 30.78 | 62.83 / 30.02 |

**The central positive result.** EventSSM's dominant deficit vs the ViT baseline was a **−5.96 AP_L**
large-object gap; AP_S/AP_M were near parity. That pattern pointed to ResNet-18's **local, hierarchical
receptive field** as the culprit and motivated a **global-spatial** SSM. The controlled backbone swap
delivered on exactly that axis: **AP_L 44.70 → 47.65 (+2.95)**, closing **~half** the large-object gap to the
ViT (−5.96 → −3.01). The gain concentrates in **cars** (+1.16, the class containing large/close instances)
and not pedestrians (−0.76, almost never "large") — a class-level signature that corroborates the
receptive-field mechanism rather than an aggregate coincidence.

Overall mAP is **essentially tied** with EventSSM (+0.21, within single-seed noise; the AP_L +2.95 is the
reportable win) and **−1.29 behind the ViT baseline**. So: *a pure state-space spatial backbone — no
attention, no spatial convolution — matches the CNN hybrid overall and beats it on large objects, while
being the smallest model.* val/AP peaked at 0.48 (step 310k of 400k); test 46.43 — the ~1.5-pt drop is the
expected val≠test + checkpoint-selection-optimism gap, not a regression.

### 3.2 Mechanism — effective receptive field (Stage 15 ERF)

A gradient-based ERF probe (Luo et al. 2016) on the **trained** backbones measures how much of the input each
spatial mixer actually "sees" at the frame centre (`σ` = RMS spatial spread, in input pixels):

| Backbone | Stage 3 σ trained | Stage 4 σ trained | Δ from training |
|---|---|---|---|
| ResNet-18 (EventSSM) | 41.8 px | 84.4 px | +0.2 / +6.2 |
| **BiMamba (PureSSM)** | **85.6 px** (2.05×) | **105.5 px** (1.25×) | **+34.4 / +30.1** |

Two findings, both supporting the receptive-field story: (1) **BiMamba sees wider at every depth** — a large
object spanning the frame falls *inside* its field but *outside* ResNet's, the direct visual cause of the
AP_L gain; (2) the **SSM receptive field is *learned*** (training widened σ by +30–34 px) while the **CNN's
is architecturally fixed** (+0.2 px). The selective scan can be *taught* to exploit long-range context; the
conv stack cannot. This turns the AP_L *number* into an explained *mechanism*.

### 3.3 Temporal robustness (Stages 9 & 16) — the distinctive contribution

Two regimes, same COCO-mAP ruler:

**Regime 1 — fixed cadence, variable accumulation window.** PureSSM ≈ EventSSM across the sweep (4× retention
76.8 % vs 76.6 %), both marginally below S5-RVT. **Sub-finding (single-seed, AP_L-driven):** at the dense
0.25× window PureSSM *overtakes* the ViT baseline (+0.71 overall, driven by **+3.58 AP_L**) because the ViT's
large-object accuracy **collapses** on smeared dense frames (AP_L −14.3) while PureSSM's holds (−7.7). Honest
framing: PureSSM more robust at the dense/slow end, S5-RVT at the sparse/fast end.

**Regime 2 — true rate change (the headline number):** retention = mAP@rate ÷ mAP@1×.

| | S5-RVT | EventSSM | ConvLSTM (paper) | **PureSSM** |
|---|---|---|---|---|
| true-10× mAP | 29.67 | 29.09 | 8.35 | **32.38** |
| **true-10× retention** | 62.2 % | 63.0 % | 17.7 % | **69.7 %** |
| true-2× retention | 95.1 % | 95.5 % | — | **95.9 %** |

**PureSSM is the most rate-robust model** — retaining **69.7 %** of accuracy at a genuine 10× rate change,
beating both SSM cousins and crushing the paper's ConvLSTM (17.7 %). Since EventSSM and PureSSM share the
*identical* Mamba temporal path, the extra robustness comes from the **BiMamba spatial backbone** — the same
global-receptive-field advantage behind the AP_L result, now paying off under event-sparse fast frames.

**A second, harder finding from Stage 9 (methodological):** the paper's proposed inference-time
**Δt-rescaling compensation was falsified at every tested point** for all three SSM models (at true 10× it
*costs* ~13 mAP). The robustness of SSMs is real, but it comes from the **uncompensated recurrence being
intrinsically rate-tolerant**, *not* from the explicit Δt-rescaling mechanism the paper credits. The paper's
Gen1 200 Hz headline (39.84) was **not reproducible** under the published checkpoint+code (best achievable
29.67; their repo cannot even render true-rate input — stride hardcoded). *The robustness conclusion survives
with the mechanism reassigned* — a genuine, defensible correction to the literature.

### 3.4 Efficiency (Stages 10 & 16)

Full-pipeline streaming inference, RTX 5070 Ti, idle-GPU-guarded harness:

| Metric | S5-RVT | EventSSM | **PureSSM** |
|---|---|---|---|
| Total params | 18.19 M | 19.18 M | **16.33 M** (smallest) |
| FLOPs (G) | 11.06 | 12.71 | **10.03** (fewest) |
| **mAP / GFLOP** | 4.32 | 3.64 | **4.63** (best) |
| Latency p50 eager (ms) | 17.14 | **12.60** | 26.86 |
| Throughput eager (Hz) | 58 | **79** | 37 |
| **CUDA-graph deploy (Hz)** | — | **210** | 181 |
| Energy (J/frame, eager) | 0.723 | **0.438** | 0.652 |
| Streaming state / stream | 4.7 MB | 144 MB | 144 MB |

*(FLOPs corrected 2026-10-07 — the v1 add-on double-counted projections/S5 products and missed the PureSSM spatial scans; v1 values 11.89 / 13.55 / 10.12 and 4.01 / 3.41 / 4.59. See `docs/notes/Stage21_22_tooling_notes.md` D6.)*

**The honest efficiency story (and it is nuanced):**
- PureSSM does the **least compute** (fewest FLOPs *and* params) with the **best accuracy-per-FLOP** (4.63) —
  *algorithmically* the leanest model.
- But it is the **slowest in wall-clock eager (37 Hz)**: its BiMamba backbone is *many small SSM-scan ops* →
  **launch-overhead-bound**, and SSM-scan GPU utilisation is lower than cuDNN convolutions. EventSSM does
  *more* FLOPs *faster*.
- **CUDA-graph replay** (state-carrying capture, verified correct) repairs most of this: PureSSM 37 → **181
  Hz (5.3× — the biggest boost, precisely because it was the most launch-bound)**, comfortably real-time. But
  the fair same-mode comparison still leaves **EventSSM ahead in both modes** (79/210 vs 37/181).
- **Do not over-claim sparsity.** Event sparsity does **not** translate to GPU speed (confirmed here and in
  the wider literature). The efficiency claim is FLOP-efficiency + rate-robust deployability, **not**
  "sparse = fast."
- **Trade-off to state plainly:** the Mamba state-expansion makes streaming state **31× fatter** than the
  ViT baseline (144 vs 4.7 MB) — "fast because fat." For micro-UAV SWaP this is the number to watch.

### 3.5 Qualitative (Stage 16)

GT-vs-prediction overlay videos on 12 large-car sequences (EventCV renderer; GT solid boxes, predictions
dashed with confidence, model name burned into each frame). These make the AP_L result *visible* — the
sequences were selected for large-car density (top rec: 180 large-car instances) precisely to show where the
receptive-field advantage lives. A recurring qualitative observation is **Gen1 label noise** (missing/late
boxes), which caps achievable AP and frames a dataset limitation for the discussion.

---

## 4. Discussion — what the four models actually tell us

1. **A pure SSM is a viable detection backbone.** With no attention and no spatial convolution, PureSSM
   matches the conv hybrid overall and is the smallest, most compute-efficient, most rate-robust of the
   three. That is a non-trivial architectural finding.
2. **The receptive field is the lever.** The single controlled change (conv → SSM spatial) moves large-object
   accuracy by +2.95 AP_L, explained by a 2× wider *learned* receptive field. This is the thesis's cleanest
   causal result.
3. **SSM temporal robustness is real but the credited mechanism is wrong.** The robustness comes from
   intrinsic recurrence tolerance, not the paper's Δt-rescaling — a correction backed by 19 evaluations.
4. **Efficiency is a genuine trade-space, not a free win.** Leanest FLOPs and best mAP/GFLOP, but launch-bound
   in wall-clock (fixed by graph replay) and 31× fatter state. Reported honestly, this *strengthens* the
   thesis's credibility.
5. **PureSSM's honest position vs EventSSM:** the community's proven recipe is *SSM-temporal +
   lightweight-spatial* (= EventSSM). PureSSM pushes to *pure-SSM spatial* and buys smaller/robuster/leaner-
   FLOP/better-AP_L, paying wall-clock latency. Both are legitimate designs; the thesis reports the frontier,
   not a single winner.

---

## 5. Limitations (stated up front)

- **Single seed, no error bars.** The overall-mAP deltas between own-models (±0.2–0.7) are within run-to-run
  noise; only effects well outside it (AP_L +2.95, true-10× ~3 mAP) are reported as findings.
- **Gen1 only, so far.** All results are on one low-resolution (304×240), daytime, 2-class, ego-motion
  dataset. Generalisation across resolution, domain, and conditions is untested — the primary motivation for
  the next phase.
- **Gen1 label noise** caps achievable AP and adds a qualitative confound.
- **Fixed representation.** All models inherit the stacked-histogram tokeniser; the representation itself is
  not part of the ablation.
- **Precision mix** (bf16 own-models vs fp16 baseline) — minor; all models reproduce their reference numbers.

---

## 6. Next phase — augment, don't pivot

The strategic decision (backed by a 37-dataset survey + the Gehrig & Scaramuzza 2022 resolution analysis) is
to **write up the Gen1 core, then extend it**, not to chase a new dataset.

**Primary next step — train on Prophesee 1Mpx / Gen4 (1280×720).** *This is the user's stated inclination and
the single highest-value extension.* Same automotive domain, same vendor/`.dat` format (pipeline unchanged,
reuse the cloud-5090 flow), richest SSM baselines. It is framed not as "HD = better AP" (Gehrig shows that's
naïve) but as a **falsifiable hypothesis test**: *does PureSSM's AP_L / receptive-field advantage grow at
5.6× pixels?* Two competing forces — more pixels-per-object favour the wide BiMamba field (↑) vs more
temporal noise at HD (↓). The **novel framing** fuses two of our results into one story:

> *High-resolution event cameras suffer greater temporal noise (Gehrig & Scaramuzza 2022 — smaller pixels →
> higher per-pixel event rate → timestamp sensitivity at speed/low-light). We show SSM detectors are the most
> temporally robust. We therefore hypothesise and test that SSM backbones are uniquely positioned to exploit
> high-resolution sensors — i.e. the high-resolution penalty shrinks for rate-robust architectures.*

Gehrig's own finding that the penalty is **learnable-away by training on noisy target data** dictates the
protocol: **train on 1Mpx, don't zero-shot from Gen1** (zero-shot would confound resolution with the
temporal-noise penalty).

**Optional further legs** (by priority): **PEOD** (challenging conditions — the *only* way to demonstrate the
event camera's HDR/low-light advantage, and the regime where the Gehrig penalty and our robustness both bite
hardest); **eTraM** (static-camera domain shift); **NU-AIR** (UAV framing). Neither Gen1 nor 1Mpx tests
night/overexposure, so PEOD fills a real gap.

**Future / stretch directions:**
- **Representation/tokenisation** — all models inherit the fixed stacked histogram; activity-driven
  tokenisation + rate-robust SSM is a ready-made future contribution.
- **Neuromorphic deployment (Loihi)** — the diagonal-SSM-on-Loihi-2 line of work (Meyer et al.) validates
  mapping a diagonal state-space model to spiking hardware; our rate-robust SSM is the natural candidate for
  a spiking fork.

---

## 7. Status ledger

| Phase | Deliverable | Status |
|---|---|---|
| Thesis A (MVP) | S5-RVT reproduction, test/AP 47.7 (exact match) | ✅ |
| EventSSM (Stages 0–10) | build → train → **test/AP 46.2**, robustness, efficiency | ✅ |
| PureSSM (Stages 11–16) | build → cloud-train → **test/AP 46.43, AP_L +2.95, 69.7 % @10×** | ✅ |
| Write-up | this document + per-stage docs | ✅ (living) |
| **Next** | **1Mpx augmentation (resolution × robustness study)** | ▶ planned |
| Later | PEOD conditions leg · eTraM domain-shift · Loihi spiking fork | ○ optional |

**Source docs:** Stage 8 `docs/results/Stage8_results_comparison.md` · Stage 9 `docs/results/Stage9_TwoRegime_Results.md`
(+ `Stage9_Zubic_methodology_verdict.md`) · Stage 10 (efficiency harness) · Stage 15
`docs/results/Stage15_results_comparison.md` · Stage 16 `docs/results/Stage16_results.md` · dataset/resolution research
`docs/research/`.
