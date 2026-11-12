# Thesis presentation — script (10 slides, about 9 minutes)

Deck: https://claude.ai/artifact/4fzY5fxMLMnfVPuddCW4A1 (the same text is in each slide's speaker notes).
Speaking pace: about 115 words per minute. Total spoken text ≈ 1,045 words ≈ 9:05.
Numbers are rounded for speaking; the exact values are in the report.

| # | Slide | Time | Ends at |
|---|---|---|---|
| 1 | Title and motivation | 0:45 | 0:45 |
| 2 | The problem | 0:50 | 1:35 |
| 3 | Research question and hypotheses | 0:50 | 2:25 |
| 4 | Method: a controlled experiment | 1:00 | 3:25 |
| 5 | Accuracy and receptive field | 1:00 | 4:25 |
| 6 | Faster event rates | 0:55 | 5:20 |
| 7 | Efficiency | 0:40 | 6:00 |
| 8 | The spiking variant and its cost | 0:55 | 6:55 |
| 9 | The path to neuromorphic hardware | 1:15 | 8:10 |
| 10 | Discussion and conclusion | 0:45 | 8:55 |

About a minute is left as a buffer. If you are running late, shorten slide 7 first.

---

## Slide 1 — Title and motivation (0:45)

**On the slide:** title, your name and zID, supervisor, UNSW; the event-camera figure.

> Good morning. My thesis is about object detection for small drones using an event camera, and about what a
> newer kind of neural network, the state space model, really adds to it.
> Small drones have very little room, weight and power for computing, so whatever they use to see must be
> fast and cheap. Event cameras suit this well. Instead of taking whole pictures, each pixel reports a change
> in brightness the moment it happens, with microsecond timing, a very wide brightness range and no motion
> blur.

## Slide 2 — The problem (0:50)

**On the slide:** three cards (how detection works, what has been reported, what is missing) and the gap.

> To detect objects, the events are grouped into 50-millisecond windows and passed through a neural network
> with a memory, so it can follow objects over time.
> State space models, such as S5 and Mamba, have been proposed for this memory. Their cost grows only linearly
> with sequence length, and one study reported that they stay accurate when the event rate changes.
> But every published comparison changes several parts of the detector at once: the backbone, the input and
> the training settings. So it is not possible to tell what the state space model itself contributes. That is
> the gap this thesis addresses.

## Slide 3 — Research question and hypotheses (0:50)

**On the slide:** the research question, the second question, cards H1–H4.

> My research question is: if everything else in the detector is held fixed, what does changing the
> backbone's spatial mixer, from convolution to a state space model, do to accuracy, robustness and
> efficiency? A second question is what it costs to make the detector spiking, which is the step towards
> neuromorphic chips.
> I tested four hypotheses: that the state space mixer detects large objects better, that state space models
> are more robust to faster frames than a ConvLSTM, that the published time-step correction helps, and that the
> spiking versions lose accuracy in a predicted order.

## Slide 4 — Method: a controlled experiment (1:00)

**On the slide:** pipeline figure; "Reproduced first" and "One change" notes; five model cards.

> First, I reproduced the published S5-RVT detector exactly: 47.7 mAP on the Gen1 driving dataset. mAP is the
> standard detection score, and higher is better.
> Then I kept everything in that pipeline fixed, from the input to the evaluation code, and changed only the
> backbone.
> EventSSM uses a ResNet convolutional network to look across the image and Mamba to remember over time.
> PureSSM replaces the convolution with a bidirectional Mamba scan, so state space models handle both space and
> time. The two differ in just this one part.
> I also tested the published ConvLSTM detector, RVT, on the same data. Every model was measured for accuracy,
> robustness to faster frames, speed, energy and memory.

## Slide 5 — Accuracy and receptive field (1:00)

**On the slide:** accuracy table (overall and large objects); receptive-field figure.

> First, accuracy. The baseline stays ahead at 47.7. My two models are level with each other, at 46.2 and
> 46.4, about one and a half points behind.
> The clearer difference is on large objects. EventSSM falls about six points behind the baseline there.
> PureSSM closes about half of that gap, a gain of 2.9 points. Its 95 % interval just includes zero, so the
> gain is likely rather than certain.
> This receptive-field probe fits the result: the state space mixer learns to use an area of the image up to
> twice as wide as the convolutional one. So a state space model can be the spatial backbone, and PureSSM does
> it with the fewest parameters and the least computation.

## Slide 6 — Faster event rates (0:55)

**On the slide:** robustness figure; 62–70 %, 80 %, −9 to −13.

> Next, robustness. I made the frames arrive ten times faster than in training, one every 5 milliseconds,
> without retraining.
> The state space models kept 62 to 70 % of their accuracy, PureSSM the most. But the ConvLSTM, tested on the
> same data, kept 80 %, while the paper reported only 18 %.
> The published correction, which adjusts the model's time step to the new rate, lowered accuracy by 9 to 13
> points instead of helping, and the published result could not be reproduced with the released code and
> model. So the robustness is there without the correction, and it is not special to state space models.

## Slide 7 — Efficiency (0:40)

**On the slide:** best model per limit table; efficiency figure; 144.5 MB memory note.

> On efficiency there is no single winner. EventSSM is the fastest, at 12.6 milliseconds per frame. PureSSM is
> the smallest and needs the least computation, but it is the slowest in normal mode, because its scan is made
> of many small operations.
> The ConvLSTM uses the least energy, less than half of EventSSM's. And both Mamba models keep 144 megabytes of
> memory per video stream, 31 times more than the published models, which matters on a small drone.

## Slide 8 — The spiking variant and its cost (0:55)

**On the slide:** ladder figure; the three step costs; full-schedule result.

> For the neuromorphic part, I built a spiking version of PureSSM. A leaky integrate-and-fire neuron after the
> memory only sends a signal when its potential crosses a threshold. It can send a dense value, a graded spike
> that carries a value, or a binary spike, and stepping through these splits the cost of spiking into three
> parts.
> On a short schedule, the leak and reset cost 0.3 points, too small to measure; sparsity costs 1.5; and the
> single bit another 0.5. That order was predicted before training, and it held. With full training, the
> graded version reaches 46.2, level with PureSSM.

## Slide 9 — The path to neuromorphic hardware (1:15)

**On the slide:** "Done in this thesis" and "What the results mean for a chip" cards; the five-step route
(fixed step size → smaller memory → export through NIR → Intel virtual lab, then Loihi 2 → spiking spatial
mixer).

> Now the path to neuromorphic hardware. Chips such as Intel's Loihi 2 only do work when a spike arrives, so
> their energy follows activity. A similar state space memory on Loihi 2 used about a thousand times less
> energy than on a small GPU when inputs arrived one at a time.
> I built the spiking model in simulation, so it does not depend on hardware access. Its graded readout matches
> Loihi 2's graded spikes, and it is active only 10 to 28 % of the time.
> The results show what is easy and what is hard. Sparsity costs little after full training, and a fixed leak
> is enough. The main obstacle is Mamba's changing step size, and the memory must shrink. Because the spikes
> reach only about 1 % of the operations, real savings need the spatial mixer on the chip too. That gives this
> five-step route.

## Slide 10 — Discussion and conclusion (0:45)

**On the slide:** hypothesis outcomes; the right model per platform; next steps; thank you.

> To sum up: H1 is supported with a reservation, H2 and H3 are not, and H4 is supported.
> A state space model can replace convolution in the spatial backbone, with similar accuracy and less
> computation, at a cost in speed and memory. Which model to use depends on the platform: S5-RVT for accuracy,
> the ConvLSTM for energy and robustness, EventSSM for speed, and PureSSM for size.
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

**Why has the model not run on Loihi 2 yet?**
Hardware access needs an agreement between UNSW and Intel, and Intel stopped its Lava software in 2026. I built
the model in simulation so the thesis did not depend on it; the proposal to Intel's research community is
prepared, and the model will be exported through NIR, a format other chips can also read.

**Why is the energy saving so small?**
In this hybrid only the temporal readout spikes, and it feeds about 1 % of the operations. Larger savings need
the spatial part to run on a neuromorphic chip as well.

**Why not test on a drone?**
The detector is trained on driving data. In a drone simulation, a collision rate would mostly measure the
mismatch, not the backbone. The obstacle-avoidance aim was removed from the scope for that reason.

**Why is PureSSM slow if it needs the least computation?**
Its scan is made of many small operations, and each one has a fixed start-up cost on a graphics card. In the
faster replay mode it reaches 181 frames per second.

---

## Delivery tips
- Rehearse with a timer twice; you should be at about 4:25 when you leave slide 5.
- Say rounded numbers (46.4, not 46.43); the exact values are on the slides and in the report.
- Point at the figure when you say "this probe" or "this route".
- Change any wording that does not sound like you; it will come across more naturally.
