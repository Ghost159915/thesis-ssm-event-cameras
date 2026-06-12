# UAV Event-Camera Dataset Augmentation — Exploration & Open Questions

**Status:** OPEN / parked for later discussion (raised 2026-06-12). **Not a pivot.** Gen1 stays the
primary benchmark; this explores *adding* a UAV-relevant event dataset as a secondary
generalisation / domain-relevance contribution.

---

## 1. The question

The thesis is framed around **micro-UAV perception**, but the primary benchmark — **Prophesee GEN1** —
is an **automotive** dataset (forward-facing car camera; classes = car, pedestrian). This is a real
**domain gap** for a UAV-framed thesis. Can we add a **UAV-specific event-camera dataset** to:

1. close the domain gap (UAV motion profiles, viewpoints, object scales), and
2. strengthen the **generalisation** story (the model is not just an automotive detector)?

User intent (2026-06-12): **augment, do not pivot.** Keep Gen1 for the controlled comparison; add a
UAV dataset on top.

---

## 2. Why Gen1 must stay (the constraint)

- The entire Thesis-B method is a **controlled experiment**: drop-in `ResNetMamba`/`PureSSM` backbone
  vs the reproduced **S5-RVT baseline (47.7 mAP on Gen1)**, every other component held fixed.
- That comparison is **only valid on Gen1**, because that is where the baseline number exists. Dropping
  Gen1 deletes the scientific control and the headline result.
- ⇒ A UAV dataset is an **addition** (a generalisation / transfer chapter), never a replacement.

---

## 3. The framing fork (must be decided first)

"Event camera + UAV" splits into two *different tasks*. The dataset choice follows from this, not the
other way around.

| | (1) Onboard micro-UAV perception | (2) Anti-UAV / drone-as-target |
|---|---|---|
| Camera | **on** the drone, looking out | **observes** the sky, drone is the object |
| Task | detect ground objects / obstacles for sense-and-avoid | detect/track the (often tiny, fast) drone |
| Matches current framing? | **Yes** ("micro-UAV perception, toward deployment") | Partial — a narrative pivot toward counter-UAV |
| Labeled *detection* datasets | **scarce, small** (mostly obstacle-avoidance, e.g. Falanga sense-and-avoid) | **several recent, bbox-labeled** (see §4) |
| Difficulty flavour | viewpoint/scale variety, ego-motion | tiny-object detection, high dynamic range, fast motion |

**Tension:** the *richest, newest* labeled datasets are all framing (2) (anti-UAV), but the *current
thesis framing* is closer to (1) (onboard). Picking (2) is a (mild) narrative shift the user must
consciously accept; picking (1) means thinner data and possibly simulation.

---

## 4. Candidate datasets (anti-UAV framing — 2024-25)

> Found via a quick search 2026-06-12; **specifics to be verified in a proper `/lit-review`** before
> committing. Mirrored in memory `uav-event-datasets`.

| Dataset | Venue / year | Notes |
|---|---|---|
| **EV-UAV** | ICCV 2025 | Largest: 50+ h raw stream, 120k+ instances (~65% drone), 12 real environments (urban/park/industrial). Real-world small-object detection + baseline. |
| **EVDET200K** ("Event-based Tiny Object Detection: A Benchmark Dataset and Baseline") | 2025 (arXiv 2506.23575) | UAVs in diverse environments, collected 2023-05 → 2024-06; benchmark + baseline. |
| **FRED** (Florence RGB-Event Drone Dataset) | ACM MM 2025 | Multimodal event + RGB. |
| **NeRDD** (Neuromorphic Drone Detection: Event-RGB Multimodal) | ECCV 2024 workshop (arXiv 2409.16099) | Multimodal event + RGB. |
| ODD-SEC / per-pixel-frequency drone detectors | 2025-26 | Niche (spinning camera / frequency analysis). |

For framing (1) (onboard), the relevant assets are sparser: **Falanga et al.** dynamic-obstacle
avoidance data, **EVIMO/EVIMO2** (moving-object masks, indoor), **UZH-FPV** (aggressive flight but
**VIO/ego-motion**, *no* object bboxes), and **simulation** (v2e / ESIM + a UAV sim) for controllable
event rates.

---

## 5. Integration cost (the real gate)

The existing pipeline (RVT) consumes a **specific preprocessed format**:
`event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5` (2 polarities ×
10 bins) + `labels_v2/labels.npz` + timestamp index files, in `train/val/test` split dirs.

⇒ Any new dataset must be **converted** to this format (raw events → stacked-histogram H5; native
annotations → RVT label schema; class-map reconciliation). This is **non-trivial data engineering** and
should be scoped before committing. Different sensor resolutions (vs Gen1 240×304) also touch the
backbone's input padding and stride math.

**Silver lining:** the Stage 5 pipeline smoke validates the *machinery* (data module → model → loss →
Lightning loop), which is **dataset-agnostic** — so it is not wasted regardless of which dataset (if any)
is adopted.

---

## 6. The cheap, strong generalisation axis we already have

Independent of any new dataset: the planned **temporal-generalisation study (train at one event rate,
test across rates)** is *intrinsically* UAV-relevant — UAVs experience extreme variable event rates due
to aggressive ego-motion. It directly showcases the **SSM linear-time / rate-robustness** advantage and
needs **no new labels**. This may already satisfy much of the "UAV generalisation" goal at near-zero data
cost; a UAV dataset would then be the cherry on top, not the foundation.

---

## 7. Open questions (to resolve later)

1. **Framing:** commit to onboard-perception (1) or accept the anti-UAV (2) narrative shift? (Dataset
   choice depends entirely on this.)
2. If (2): **EV-UAV vs EVDET200K** as the single added dataset — which has the cleanest license, the most
   usable raw-event export, and published baselines we can compare against?
3. **Conversion effort:** how many engineer-days to convert the chosen set to the RVT stacked-histogram
   format? Is a subset enough for a transfer/generalisation chapter (vs full retraining)?
4. **Evaluation mode:** zero-shot transfer (Gen1-trained → UAV test), fine-tune, or train-from-scratch on
   the UAV set? (Zero-shot is the strongest generalisation evidence and cheapest.)
5. **Resolution mismatch:** does the chosen sensor resolution break the 256×320 padding / stride
   assumptions, and how much backbone/config change does that imply?
6. **Sim option:** is v2e/ESIM-generated UAV event data (controllable rate) a defensible complement or
   substitute if real labeled UAV detection data proves too costly to integrate?

---

## 8. Proposed next actions (when we return to this)

- [ ] User picks the **framing** (§3) — this unblocks everything else.
- [ ] Run a proper **`/lit-review`** of the §4 candidates (verify size, license, format, baselines).
- [ ] Scope the **format-conversion** effort for the top candidate (§5) → engineer-day estimate.
- [ ] Decide **evaluation mode** (§7.4): default recommendation = **zero-shot transfer** for the cleanest,
      cheapest generalisation result, plus optional fine-tune.
- [ ] Slot as a **Stage 9 (temporal/domain generalisation)** sub-chapter, after Gen1 results land.

> **Working recommendation:** augment Gen1 with **one** anti-UAV event dataset (likely **EV-UAV**) used in
> **zero-shot transfer**, and lead the generalisation story with the **variable-event-rate** study. Revisit
> after Gen1 training results (Stage 6-8) exist.
