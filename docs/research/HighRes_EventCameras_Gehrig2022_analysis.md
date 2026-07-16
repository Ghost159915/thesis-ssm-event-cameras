# Paper Analysis — Gehrig & Scaramuzza (2022), *"Are High-Resolution Event Cameras Really Needed?"*

**Local PDF:** `papers_correct/01_event_cameras/Gehrig_2022_HighResolutionEventCameras_arXiv.pdf` · project page:
https://uzh-rpg.github.io/eres/ · **Not previously in the library** (added 2026-07-16).
**Why it matters here:** it is the definitive study of the event-camera *resolution* trade-off — directly relevant to
the thesis's 1Mpx-augmentation decision, and it plugs into the PureSSM **rate-robustness** contribution.

---

## 1. What the paper claims (the counter-intuitive result)

Event cameras are trending to higher resolution (Gen4 / 1Mpx = 1280×720), but **higher resolution is not universally
better.** In **low-illumination and high-speed** conditions, **lower-resolution cameras outperform higher-resolution
ones**, while also using far less bandwidth. Validated on three tasks — **image reconstruction, optical flow, camera
pose tracking** — in simulation (resolutions 128²→1280²) *and* on real data (Prophesee Gen4 1280×720 on a slider).

## 2. The mechanism (sensor physics — task-agnostic, so it transfers)

- An event fires when accumulated log-brightness change exceeds a threshold: `p · L_t · Δt > C` (Eq. 1–2).
- Smaller pixels (higher resolution) resolve finer detail → the brightness derivative follows a **power law**
  `L_t ∝ 1/pixel_size^γ` (Eq. 6, γ related to edge fractal dimension).
- Therefore the inter-event time **`Δt ∝ pixel_size^γ`** (Eq. 7): as resolution ↑, `Δt` ↓ → **higher per-pixel event
  rate**.
- A higher event rate makes each event's **timestamp more sensitive** to non-idealities — slow pixel response (low
  cutoff frequency in the **dark**) and **high-speed** motion — producing **more temporal noise / ghosting** (Fig. 5d).
- Net: high-res sensors are *more* sensitive to temporal effects at exactly the hard operating points. On real data,
  **640×360 beat the full 1280×720** across all three tasks once speed exceeded ~0.4 m/s.

## 3. The crucial nuance — the penalty is *trainable-away*

The high-res penalty hits **model-based** and **clean-trained** methods hardest, and **learning-based methods trained
on realistic (noisy, night) data overcome it:**
- **E-RAFT** (learning-based, trained on real day+night data) **always benefits from high resolution** and does not
  degrade harshly in noise.
- **E2VID** (learning-based but trained on *clean synthetic* data) shows the **opposite** (degrades at high res).
- Authors, verbatim: *"we may overcome the limitations of noise at high resolutions by adopting a learning-based
  approach, and training on noisy night-time data, robustifying the network against these error sources. Noisy
  training-data is the key."*
- (Also: contrast-maximisation objectives are more robust at speed/low-light than photometric ones.)

**Implication for us:** our detectors are deep, learning-based, and trained on the target data → the high-res penalty
largely evaporates **if we train on 1Mpx**. This is the direct evidence for **"train on 1Mpx, don't zero-shot from
Gen1"** (zero-shot compounds resolution mismatch *and* the temporal-noise penalty).

## 4. How it applies to *this* thesis (the payoff)

**(a) It reframes the 1Mpx decision.** Don't augment because "higher res = better AP" (the paper shows that's naïve).
Augment to *test a hypothesis*.

**(b) The headline connection — high-res's weakness == PureSSM's strength.**
- Paper: high-res → higher event rate → **more temporal / timestamp noise** (fundamentally a *rate* problem).
- Us (Stage 9): **PureSSM is the most rate-robust model** (69.7% retention at true-10×; SSM recurrence is
  intrinsically rate-robust).
- → The exact failure mode of high-resolution cameras is what a state-space detector absorbs best. **Proposed framing
  (Motivation / Discussion):**
  > *"High-resolution event cameras are known to suffer greater temporal noise (Gehrig & Scaramuzza, 2022). We show
  > that state-space detectors are the most temporally robust; we therefore hypothesise — and test — that SSM
  > backbones are uniquely positioned to exploit high-resolution sensors, i.e. the high-resolution penalty shrinks for
  > rate-robust architectures."*
  This fuses the Stage-9 rate-robustness result and a 1Mpx study into **one coherent narrative.**

**(c) Efficiency gains importance.** The paper stresses the **bandwidth/event-rate burden** of high-res. Our
efficiency pillar (PureSSM leanest FLOPs; graph-replay speed) becomes *more* relevant at 1Mpx, where there are far
more events to process.

**(d) A concrete, falsifiable experiment (on 1Mpx).** Measure whether PureSSM's **AP_L (large-object) advantage grows
or shrinks** vs Gen1. Two competing forces: **↑** more pixels per large object favour the wide-receptive-field
BiMamba; **↓** more temporal noise (this paper) erodes gains at speed. Net is empirical — and our rate-robustness
predicts PureSSM pays the noise cost better than EventSSM/S5-RVT, so its advantage should **hold or grow**.

## 5. Honest caveat when citing it

The paper studies **low-level vision** (reconstruction / flow / pose) with mostly model-based methods — **not deep
object detection.** The transfer is by **analogy**: the event-rate → temporal-noise mechanism is *sensor physics*
(task-agnostic), but cite the paper as **mechanism / motivation**, not as a detection result.

## 6. One-line takeaway
Higher-resolution event cameras trade spatial detail for temporal noise and bandwidth; that penalty is
learnable-away and is *smallest for temporally-robust models* — which is precisely what makes 1Mpx the right
augmentation target for a rate-robust SSM detector, and turns the resolution question into a research contribution.
