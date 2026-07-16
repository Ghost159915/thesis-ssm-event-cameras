# Thesis Progress Update — 15 Jul 2026

**Benas Vaiciulis · MMAN4952 (Thesis B) · UNSW Sydney**
Topic: State-Space Models (Mamba) for event-camera object detection on Prophesee Gen1.

---

## TL;DR (the 30-second version)

- **Baseline reproduced exactly**, and I've now built and evaluated **two of my own architectures** on top of it.
- **Latest milestone (this week):** my second model, **PureSSM**, *confirms the central hypothesis* of the thesis — that a state-space (Mamba) spatial backbone recovers the long-range context a CNN lacks, and that this is what large objects need. It closes **~half the large-object accuracy gap** (AP_L +2.95), and I have a **receptive-field figure that shows the mechanism visually**, not just the number.
- **Being honest:** on *overall* accuracy PureSSM is **essentially tied** with my first model (not a SOTA jump), it's still **behind the published baseline overall**, and results are **single-seed**. One experimental block (efficiency + robustness measurements for PureSSM) is **still to do**.

---

## Where the project stands — the four models

The whole design is a **controlled ablation**: every model is a *spatial* block + a *temporal* block, swapped one at a time. All evaluated through the **same** Gen1 test-set evaluator (apples-to-apples).

| Model | Spatial | Temporal | test/AP (COCO) | AP_L (large obj) | Status |
|---|---|---|---|---|---|
| RVT (lit.) | ViT attention | ConvLSTM | ~47.2 (paper) | — | reference |
| **S5-RVT** (baseline I reproduced) | ViT attention | S5 (SSM) | **47.72** | 50.66 | ✅ exact repro |
| **EventSSM** (my model #1) | ResNet-18 conv | Mamba | 46.22 | 44.70 | ✅ complete |
| **PureSSM** (my model #2) | **BiMamba (SSM)** | Mamba | **46.43** | **47.65** | ✅ accuracy done · efficiency/robustness pending |

**Reading it:** swapping the *spatial* block (ResNet → BiMamba, everything else identical) moved AP_L from 44.70 → 47.65. Because only one block changed, that gain is cleanly attributable to the spatial swap — a textbook controlled experiment.

---

## The headline finding — and its honest size

**What's genuinely good:**
- The **large-object hypothesis is confirmed.** EventSSM's main weakness was large objects (AP_L −5.96 vs the ViT baseline). PureSSM recovers **+2.95** of that (car class +1.16; pedestrians, which are never "large", don't benefit — exactly the predicted pattern).
- **Mechanism, not just correlation.** A gradient-based *effective receptive field* figure shows the trained BiMamba backbone "sees" a much wider region than ResNet at every depth (spread σ: 85.6 vs 41.8 px at stage 3; 105.5 vs 84.4 at stage 4). Bonus: training *widens* the SSM's field (+30–34 px) but barely the CNN's (+0.2) — the SSM's reach is learnable, the CNN's is architecturally fixed.
- A **pure SSM (no attention, no spatial convolution)** matches a CNN-hybrid on overall accuracy while beating it on large objects — architecturally notable in its own right.

**What I'm not overclaiming:**
- **Overall mAP is a tie** with my own EventSSM (46.43 vs 46.22; the +0.21 is within single-seed noise). The contribution is a **targeted, mechanism-confirming** result, not a new state of the art.
- Still **−1.29 behind the S5-RVT baseline** overall; the large-object gap is **halved, not closed**.
- The large-object gain trades against a small pedestrian/recall dip — a classic receptive-field trade-off.

---

## The three result "pillars" — status per model

| Pillar | EventSSM (model #1) | PureSSM (model #2) |
|---|---|---|
| **Accuracy** (Gen1 test) | ✅ 46.2 / AP_L 44.7 | ✅ 46.4 / AP_L 47.65 |
| **Efficiency** (latency/energy/FLOPs) | ✅ **1.4–1.5× faster** than baseline (70 vs 51 Hz), **~1.76× more energy-efficient** (0.40 vs 0.73 J/frame); trade-off = 31× larger streaming state | ⏳ **not yet measured** (Stage 16) — probe suggests competitive |
| **Temporal robustness** (event-rate change) | ✅ SSM recurrence keeps **~63%** AP at true 10× rate vs ConvLSTM's 17.7% | ⏳ not yet measured (Stage 16) |

**A secondary finding worth discussing (Stage 9, EventSSM):** the prior paper's inference-time Δt-rescaling trick did **not** reproduce in my hands, and their headline 200 Hz number isn't reproducible under the published checkpoint/code. The robustness *conclusion* still holds, but the *mechanism* is different from what the paper claims. This is a genuine (if awkward) negative result — I'd like your steer on how hard to state it in the write-up.

---

## Honest limitations / risks

- **Single seed, no error bars.** The overall-mAP comparisons are within noise; I should probably run ≥2 more seeds before any publication claim (compute cost is the question).
- **PureSSM story is not complete** until Stage 16 (efficiency + robustness) is measured — right now I can only claim the accuracy/mechanism half.
- **Gen1 label noise** (esp. pedestrians) caps absolute numbers for everyone; worth flagging as a dataset limitation.
- **Hardware/logistics:** local RTX 5070 Ti is the main machine; I trained the 400k PureSSM run on a **rented cloud RTX 5090** (~27 h, ~$10) since it shares the same GPU architecture — worked cleanly. **Katana HPC** migration is scoped but not started (no account yet). **Gen3.1 sensor** integration remains unresolved / possibly out of scope.

---

## What's next

1. **Stage 16 — finish the PureSSM pillars** (the immediate priority): efficiency benchmark (latency/energy/FLOPs vs EventSSM) + event-rate robustness re-run + qualitative GT-vs-prediction videos for the large-object cases. This completes the model's story.
2. **Thesis chapters:** fold the results (accuracy table, receptive-field figure, efficiency table) into Methodology / Results / Discussion. Much of the raw material is already written up per-stage.
3. **Optional forks (need your input on scope):** spiking-SSM direction for Intel Loihi (deep-research done); Katana migration; augmenting Gen1 with a UAV/anti-UAV event dataset.

---

## Questions for you (supervisor)

1. **Scope:** stop at PureSSM and write up (two models + baseline is already a solid controlled study), or pursue one fork (spiking/Loihi, UAV dataset, or Katana)?
2. **Rigour vs compute:** is it worth the extra cloud cost to run multiple seeds for error bars, given the overall-mAP differences are within noise?
3. **Framing:** how firmly should I state the non-reproducibility of the prior paper's rate-robustness headline — soft ("not reproducible under the published artefacts") or a harder claim?
4. **Sensor:** is real Gen3.1 camera integration still expected, or do we keep this a Gen1-benchmark thesis?

---

*Supporting detail if useful: `docs/Stage15_results_comparison.md` (accuracy + receptive-field figure), `docs/Stage8_results_comparison.md` (EventSSM), Stage-9 robustness and Stage-10 efficiency docs. All results reproducible via the `code/event_ssm/scripts/stage*_*.sh` runners.*
