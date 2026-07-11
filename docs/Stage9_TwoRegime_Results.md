# Stage 9 — Temporal Generalisation: Two-Regime Results (chapter draft)

**Status:** results final (2026-07-11) · 19 evaluations · figures: `results/stage9/stage9_degradation_curve.{png,pdf}`, `results/stage9/stage9_truerate_curve.{png,pdf}` · tables: `degradation_table.csv`, `truerate_table.csv` · method provenance: `docs/Stage9_Zubic_methodology_verdict.md` (incl. Addenda 1–2)

---

## 1. Research question and experimental logic

Event cameras produce data at a rate governed by scene dynamics, so a deployed detector must tolerate
inference conditions that differ from its training configuration. Zubić et al. [Zubic2024] attribute
such robustness in SSM-based detectors to inference-time rescaling of the state-space discretisation
step (Δt, exposed as `step_scale`), reporting that their S5 detector retains 39.84 mAP at a 10×
frequency shift on Gen1 where a ConvLSTM baseline collapses to 8.35. This stage tests that mechanism
directly — on the published S5-RVT checkpoint and on the EventSSMDetector developed in this thesis —
under two deployment-relevant regimes:

- **Regime 1 — fixed cadence, variable window.** Frames are generated every 50 ms (the training
  cadence); only the event-accumulation window varies (200/100/50/25/12 ms). The model's stepping
  rate never changes; per-frame event mass does. This regime reproduces what the published
  preprocessing code actually generates: its representation stride is hard-coded to 50 ms
  (`preprocess_dataset.py:920`, "Could be an argument of the script"), so windows shorter than 50 ms
  leave inter-frame gaps (≈76 % of events discarded at 12 ms) and longer windows overlap.
- **Regime 2 — true rate change.** Stride = window (25 ms → 2×; 5 ms → 10×), tiling the event stream
  gap-free so the model genuinely steps faster. Gen1's 250 ms label grid constrains valid strides to
  divisors of 250, which excludes a true 4× (12.5 ms) and — notably — means the published code cannot
  produce this regime without modification (a stride hook was added for this work;
  `docs/patches/preprocess_full_stage9.patch`). The 1× point (window = stride = 50 ms) is shared by
  both regimes and reproduces each model's Stage-8 test result exactly.

All evaluations use the frozen Stage-7/8 checkpoints (no retraining or fine-tuning), the identical
Prophesee/COCO evaluator, and a rebuilt raw→representation pipeline validated to 4 decimal places
against the canonical data (test/AP 0.4620 at 1×). Compensated runs set `step_scale` (S5) or the
equivalent post-softplus Δ-rescaling implemented for Mamba-2 (`MAMBA_STEP_SCALE`;
`code/event_ssm/temporal/_scan.py`, 4/4 parity tests) and print in-log engagement proofs; at
`step_scale = 1` both hooks are bit-exact to the unmodified models (byte-identical 1× anchor,
0.47689520442617495 in repeated runs).

## 2. Results

### 2.1 Regime 1 — fixed cadence (window-only sweep)

COCO mAP (×100), Gen1 test split, both models through the identical rebuilt pipeline:

| rate | window (ms) | EventSSM | S5-RVT | EventSSM dev. vs own 1× | S5-RVT dev. |
|---|---|---|---|---|---|
| 0.25× | 200 | 40.9 | 41.0 | −11.5 % | −14.1 % |
| 0.5× | 100 | 45.5 | 46.5 | −1.4 % | −2.5 % |
| 1× | 50 | 46.2 | 47.7 | 0 | 0 |
| 2× | 25 | 43.5 | 45.0 | −5.8 % | −5.6 % |
| 4× | 12 | 35.4 | 38.4 | −23.4 % | −19.5 % |

Δt-compensation was probed at the 4× extreme in both directions on the S5 baseline:
`step_scale = 0.24` → **30.1**; `= 1.0` (none) → **38.4**; `= 4.17` → **29.4**. The response is a
symmetric maximum at the trained value: under fixed cadence there is no discretisation mismatch to
correct, and any rescaling only perturbs a correctly calibrated system. Degradation in this regime is
attributable to per-frame input statistics (event mass scales linearly with the window; saturation
analysis excluded count-clipping as a cause), which no temporal-parameter adjustment can restore.

### 2.2 Regime 2 — true rate change

| true rate | S5-RVT (none) | S5-RVT (comp.) | EventSSM (none) | EventSSM (comp.) |
|---|---|---|---|---|
| 1× | 47.69 | = | 46.20 | = |
| 2× | 45.37 | 44.51 | 44.13 | 43.67 |
| **10×** | **29.67** | 19.96 | **29.09** | 20.18 |

Retention at 10× relative to each model's own 1×: S5-RVT **62.2 %**, EventSSMDetector **63.0 %**
uncompensated; **41.9 %** and **43.7 %** compensated. For reference, [Zubic2024] report 83.5 %
retention for compensated S5 (39.84) and 17.7 % for ConvLSTM (8.35) at the same nominal shift.

### 2.3 Findings

**F1 — Δt-rescaling failed at every tested operating point.** Across seven compensated evaluations
(two regimes, three rates, both architectures, both sign conventions at the fixed-cadence extreme),
no compensated result exceeded its uncompensated counterpart. At a true 10× shift the prescribed
convention (`step_scale = 0.1`) *costs* ≈10 mAP for both models. The damage is
architecture-independent (compensated S5 and Mamba land within 0.25 mAP of each other at 10×),
implicating the shared discretisation-rescaling operation rather than either implementation.

**F2 — the published headline is not reproducible under the published artefacts.** Using the authors'
checkpoint, their evaluation stack, and their stated inference-time mechanism, no configuration
approached the reported 39.84 at 10×; the best achievable result (no compensation, 29.67) trails it
by ≈10 mAP. Two independent code observations corroborate a methodological gap rather than a defect
in this reproduction: the released preprocessing cannot generate true-rate representations (hard-coded
stride), and `step_scale` is never plumbed into the released detection path (it is dead code outside
`s5_model.py`). The frequency experiments therefore ran on preprocessing and wiring that differ from
the release in unrecorded ways. *(Framing note: stated here as "not reproducible under the published
code and checkpoint" — the demonstrated claim — rather than as an assertion about the paper's own
runs.)*

**F3 — SSM recurrence is intrinsically rate-robust; the mechanism reassigns.** The headline
*conclusion* of [Zubic2024] — SSM detectors degrade gracefully under rate shift where ConvLSTM
collapses — is **confirmed and strengthened**: it holds *without any compensation at all* (63 %
retention at 10× vs 17.7 % for their ConvLSTM reference). The robustness is a property of the
state-space recurrence itself, not of inference-time Δt adjustment.

**F4 — input-dependent discretisation confers a marginal, consistently-signed advantage.**
EventSSMDetector retains more than S5-RVT at every uncompensated off-training point (e.g. 63.0 % vs
62.2 % at 10×; −4.5 % vs −4.9 % at 2×) and is less damaged by imposed Δt mistuning wherever probed.
The effect is small (≲1 mAP) and is reported as a directional observation, not a decisive advantage:
Mamba's per-token Δ = softplus(dt(x) + b) can partially absorb rate shifts and external rescaling,
whereas S5's Δt is a fixed parameter that any rescaling dictates completely.

## 3. Discussion

**Why compensation fails.** The Δt-rescaling argument treats rate shift as a pure time-axis
transformation. In practice a rate shift changes *two* things: the stepping interval **and** the
per-frame input statistics (a 5 ms frame carries ≈10 % of the training frame's event mass, before the
spatial feature extractor — whose normalisation and activation statistics are calibrated at training
mass — ever reaches the SSM). Rescaling Δt addresses only the former; empirically the latter
dominates, and the rescaling itself disturbs the transient response of a system whose remaining
components (spatial CNN/ViT, per-frame normalisations, the detection head's confidence calibration —
and, in EventSSMDetector, the causal time-convolution, which has no Δt notion at all) are not
time-parameterised. A fixed-cadence pipeline (regime 1) makes this exact: there the time axis never
changes, and the measured response to any rescaling is strictly negative and symmetric.

**Deployment reading.** For a micro-UAV stack the practical prescriptions are: (i) a fixed-cadence
pipeline with variable scene activity needs no temporal adjustment — robustness is bounded by input
sparsity; (ii) under genuine frame-rate changes, an SSM detector may simply be run faster,
uncompensated, with predictable graceful degradation (−5 % at 2×, −37 % at 10×); (iii) the selective
(input-dependent) discretisation of Mamba provides a small additional margin at zero engineering cost.

**Limitations.** Single seed per point (evaluation-only protocol; the 1× anchors are deterministic);
Gen1 label noise and its 250 ms grid bound both the rate lattice and the achievable ceiling; the 10×
comparison to published numbers is nominal (5 ms vs the paper's 200 Hz label); bf16 evaluation
(ISSUE-09) is a minor protocol difference shared by all points; findings concern inference-time
compensation only — training-time rate augmentation is untested and remains plausible future work.

## 4. Chapter artefacts

- Figure (regime 1): degradation + falsified probes — `stage9_degradation_curve.pdf`
- Figure (regime 2): four measured curves against both published reference points — `stage9_truerate_curve.pdf`
- Tables: `degradation_table.csv`, `truerate_table.csv`
- Method/patch provenance: `docs/patches/README.md`; full audit trail in `docs/Stage9_Zubic_methodology_verdict.md`
