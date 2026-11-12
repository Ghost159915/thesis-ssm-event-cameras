# Thesis presentation — script (10 slides, about 9½ minutes)

Speaking pace: about 115–120 words per minute. Total spoken text ≈ 1,100 words.
Numbers are rounded for speaking; the exact values are in the report.

| # | Slide | Time | Ends at |
|---|---|---|---|
| 1 | Title and motivation | 0:40 | 0:40 |
| 2 | The problem | 1:00 | 1:40 |
| 3 | Research question and hypotheses | 0:45 | 2:25 |
| 4 | Method: a controlled experiment | 1:15 | 3:40 |
| 5 | The spiking variant | 0:40 | 4:20 |
| 6 | Results: accuracy and receptive field | 1:00 | 5:20 |
| 7 | Results: faster event rates | 1:00 | 6:20 |
| 8 | Results: efficiency | 0:50 | 7:10 |
| 9 | Results: the cost of spiking | 0:55 | 8:05 |
| 10 | Discussion and conclusion | 1:20 | 9:25 |

About 35 seconds are left as a buffer. If you are running late, shorten slide 5 and slide 8 first.

---

## Slide 1 — Title and motivation (0:40)

**On the slide**
- Title: *State Space Models for Event-Based Object Detection*
- Your name, zID, supervisor (Will Midgley), UNSW
- Image: `thesis/latex/images/fig_event_camera_principle.png` (how an event camera works)

**Say**
> Good morning. My thesis is about object detection for small drones using an event camera, and about what a
> newer kind of neural network, the state space model, really adds to it.
> Small drones have very little room, weight and power for computing, so whatever they use to see must be
> fast and cheap. Event cameras suit this well. Instead of taking whole pictures, each pixel reports a change
> in brightness the moment it happens, with microsecond timing, a very wide brightness range and no motion
> blur.

---

## Slide 2 — The problem (1:00)

**On the slide**
- Events are grouped into short time windows and passed to a detector with a memory
- State space models (S5, Mamba): linear cost, a memory over time, reported to cope with changing event rates
- But: published comparisons change the backbone, the input and the training recipe at once
- → the effect of the state space model itself is unknown

**Say**
> To detect objects, the events are grouped into short time windows of 50 milliseconds and passed through a
> neural network with a memory, so that it can follow objects over time.
> State space models, such as S5 and Mamba, have recently been proposed for this memory. Their cost grows only
> linearly with the length of the sequence, and one published study reported that they stay accurate when the
> event rate changes, much better than older recurrent networks.
> The problem is that every published comparison changes several parts of the detector at once: the
> backbone, the input format and the training settings. So it is not possible to tell what the state space
> model itself contributes. That is the gap this thesis addresses.

---

## Slide 3 — Research question and hypotheses (0:45)

**On the slide**
- **Research question:** with the whole detection pipeline held fixed, what does changing the spatial mixer
  from convolution to a state space model do to accuracy, robustness and efficiency?
- Second question: what does it cost to make the detector spike?
- H1 large objects · H2 more robust than a ConvLSTM · H3 the published correction helps · H4 predicted order
  of the spiking costs

**Say**
> My research question is: if everything else in the detector is held fixed, what does changing the
> backbone's spatial mixer, from convolution to a state space model, do to accuracy, robustness and
> efficiency? A second question is what it costs to make the detector spiking, which is the step towards
> neuromorphic chips.
> I tested four hypotheses: that the state space mixer detects large objects better, that state space models
> are more robust to faster frames than a ConvLSTM, that the published time-step correction helps, and that the
> spiking versions lose accuracy in a predicted order.

---

## Slide 4 — Method: a controlled experiment (1:15)

**On the slide**
- Figure: `thesis/latex/images/fig_spiking_architecture_compact.pdf`, panel (a) (pipeline and the two mixers)
- Small table: S5-RVT (attention + S5) · EventSSM (ResNet + Mamba) · PureSSM (BiMamba + Mamba) ·
  RVT ConvLSTM (published) · SpikingSSM (PureSSM + spiking readout)
- Gen1 driving dataset (cars, pedestrians) · 400,000 training steps · same code for every model

**Say**
> First, I reproduced the published S5-RVT detector exactly: 47.7 mAP on the Gen1 driving dataset. mAP is the
> standard detection score, and higher is better.
> Then I kept everything in that pipeline fixed: the input, the neck, the detection head, the loss, the
> training settings and the evaluation code. Only the backbone changes.
> EventSSM uses a ResNet convolutional network to look across the image, and Mamba to remember over time.
> PureSSM replaces the convolution with a bidirectional Mamba scan, so state space models handle both space and
> time. Because the two differ in just this one part, a difference between them can be traced to it.
> I also tested the published ConvLSTM detector, RVT, on the same data. Every model was measured for accuracy,
> robustness to faster frames, speed, energy and memory.

---

## Slide 5 — The spiking variant (0:40)

**On the slide**
- Figure: panel (b) of `fig_spiking_architecture_compact.pdf` (the leaky integrate-and-fire readout)
- Three readouts: analog (dense) → graded spike → binary spike
- The ladder: leak and reset → sparsity → single bit

**Say**
> For the neuromorphic part, I built a spiking version of PureSSM. Its memory output passes through a leaky
> integrate-and-fire neuron, which only sends a signal when its potential crosses a threshold.
> The neuron can send three kinds of output: a dense value, a graded spike that carries a value, or a binary
> spike. Stepping through these versions splits the cost of spiking into three parts: the neuron's leak and
> reset, sparsity, and the single bit.

---

## Slide 6 — Results: accuracy and receptive field (1:00)

**On the slide**
- Table: overall mAP — S5-RVT 47.7 · EventSSM 46.2 · PureSSM 46.4
- Large objects — S5-RVT 50.7 · EventSSM 44.7 · PureSSM 47.7
- Figure: `thesis/latex/images/fig_erf_trained.png`

**Say**
> First, accuracy. The baseline stays ahead at 47.7. My two models are level with each other, at 46.2 and
> 46.4, about one and a half points behind.
> The clearer difference is on large objects. EventSSM falls about six points behind the baseline there.
> PureSSM closes about half of that gap, a gain of 2.9 points. Its 95 % interval just includes zero, so the
> gain is likely rather than certain.
> This receptive-field probe fits the result: the state space mixer learns to use an area of the image up to
> twice as wide as the convolutional one. So a state space model can be the spatial backbone, and PureSSM does
> it with the fewest parameters and the least computation.

---

## Slide 7 — Results: faster event rates (1:00)

**On the slide**
- Figure: `thesis/latex/images/fig_robustness_truerate.pdf`
- At 10× faster: state space models keep 62–70 % · ConvLSTM keeps 80 % (paper: 18 %)
- Published correction: −9 to −13 points; published 39.8 mAP not reproduced

**Say**
> Next, robustness. I made the frames arrive ten times faster than in training, one every 5 milliseconds,
> without retraining.
> The state space models kept 62 to 70 % of their accuracy, PureSSM the most. But the ConvLSTM, tested on the
> same data, kept 80 %, while the paper reported only 18 %.
> The published correction, which adjusts the model's time step to the new rate, lowered accuracy by 9 to 13
> points instead of helping, and the published result could not be reproduced with the released code and
> model. So the robustness is there without the correction, and it is not special to state space models.

---

## Slide 8 — Results: efficiency (0:50)

**On the slide**
- Best model per limit: speed → EventSSM (12.6 ms) · size and computation → PureSSM (16.3 M, 10.0 GFLOPs) ·
  energy → ConvLSTM (0.17 J) · memory → ConvLSTM and S5-RVT (4.7 MB vs 144.5 MB for Mamba)
- Optional figure: `thesis/latex/images/fig_efficiency.pdf`

**Say**
> On efficiency there is no single winner. EventSSM is the fastest, at 12.6 milliseconds per frame. PureSSM is
> the smallest and needs the least computation, but it is the slowest in normal mode, because its scan is made
> of many small operations.
> The ConvLSTM uses the least energy, less than half of EventSSM's. And both Mamba models keep 144 megabytes of
> memory per video stream, 31 times more than the published models, which matters on a small drone.

---

## Slide 9 — Results: the cost of spiking (0:55)

**On the slide**
- Figure: `thesis/latex/images/fig_ladder.pdf`
- Short schedule: −0.3 (leak and reset) · −1.5 (sparsity) · −0.5 (single bit)
- Full schedule: graded 46.2 vs PureSSM 46.4 · estimated energy saving ≤ 1 %

**Say**
> For the spiking model, on a short training schedule, spiking costs 2.3 points in total. The leak and reset
> cost 0.3, too small to measure; sparsity costs 1.5; and the single bit another 0.5. That order was predicted
> before training, and it held.
> With full training, the graded version reaches 46.2, level with PureSSM. But the estimated energy saving is
> only about 1 %, because in this design the spikes only reach the first layers of the neck, about 1 % of the
> operations.

---

## Slide 10 — Discussion and conclusion (1:20)

**On the slide**
- Hypotheses: H1 ✓ (with a reservation) · H2 ✗ · H3 ✗ · H4 ✓
- Best model depends on the platform: accuracy S5-RVT · energy and robustness ConvLSTM · speed EventSSM ·
  size PureSSM
- Neuromorphic path: fixed step size → smaller memory → export → Loihi 2
- Limitations: single training runs, one driving dataset

**Say**
> To sum up the hypotheses: H1 is supported with a reservation, H2 and H3 are not, and H4 is supported.
> The answer to the research question is that a state space model can replace convolution in the spatial
> backbone, with similar accuracy and less computation, at a cost in speed and memory. Which model to use
> depends on the platform: S5-RVT for accuracy, the ConvLSTM for energy and robustness, EventSSM for speed, and
> PureSSM for size.
> For neuromorphic chips, the spiking model is a starting point. The next steps are a fixed step size, a smaller
> memory and an export to Intel's Loihi 2 chip.
> The main limitations are single training runs and one driving dataset, so repeating the runs and moving to
> drone data come next. Thank you, I am happy to take questions.

---

## Likely questions — short answers

**Why only one training run per model?**
Each full run takes about 30 hours on my graphics card. I used bootstrap intervals for the test-set
uncertainty, and I only treat differences above about one point as real.

**Why is the ConvLSTM more robust than in the paper?**
I tested the published model on the same faster-frame data as the state space models. The paper's 17.7 %
could not be reproduced. Why the ConvLSTM holds up better was not tested.

**Why does the time-step correction make things worse?**
One likely reason, not tested: shrinking the step also weakens how strongly each frame enters the memory,
while each 5-ms frame already holds about a tenth of the events. All three corrected models land at about
20 mAP, which fits this.

**Why not test on a drone?**
The detector is trained on driving data. In a drone simulation, a collision rate would mostly measure the
mismatch, not the backbone. The obstacle-avoidance aim was removed from the scope for that reason.

**Why is the energy saving so small?**
In this hybrid only the temporal readout spikes, and it feeds about 1 % of the operations. Larger savings need
the spatial part to run on a neuromorphic chip as well.

**Why is PureSSM slow if it needs the least computation?**
Its scan is made of many small operations, and each one has a fixed start-up cost on a graphics card. In the
faster replay mode it reaches 181 frames per second.

---

## Delivery tips
- Rehearse with a timer twice; mark on this page where you are at 5:00 (end of slide 6).
- Say rounded numbers (46.4, not 46.43); the exact values are on the slides and in the report.
- Point at the figure when you say "this probe" or "this figure".
- Change any wording that does not sound like you; it will come across more naturally.
