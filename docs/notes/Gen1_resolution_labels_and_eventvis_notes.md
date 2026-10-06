# Gen1 Resolution, Label Quality & Event Visualisation — Discussion Notes

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis** · 2026-06-19
*Q&A clarifications from a Stage-9 working session. Sections 2–3 are Discussion/Limitations
material for the dissertation; Sections 4–5 document the event-visualisation tooling.*

---

## 1. Resolution — the training was **not** altered (clarification)

A concern was raised that the input resolution had been changed from the original. It was not.

- **Sensor / data resolution is the native Prophesee Gen1 = 304×240** (≈73k pixels). Every event
  carries an integer `(x, y)` addressing one of these photosites. This is fixed hardware.
- The RVT pipeline **zero-pads** `(20, 240, 304)` → `(20, 256, 320)` so the spatial dims divide
  evenly by the backbone stride (32). **Padding ≠ resampling**: it adds a zero border; no pixel is
  moved, scaled, or discarded. The S5-RVT baseline uses the *identical* padding, so the comparison
  is unaffected.
- The detector predicts on **downsampled feature maps** (strides 8/16/32 → 32×40, 16×20, 8×10).
  This is standard CNN-detector behaviour (the backbone shrinks the map internally); it is **not**
  a change to the input resolution.
- The **×2 upscaling** mentioned applies **only to the visualisation videos** (to make a 304×240
  clip watchable on a monitor). It has no connection to training or evaluation.

**Conclusion:** all reported results use the original Gen1 resolution; nothing in the experiment
was resampled.

## 2. Gen1 vs. Gen4 / 1-Mpx — decision: **stay on Gen1** (Future Work: Gen4)

The baseline repo supports both `gen1` (304×240) and `1mpx`/Gen4 (1280×720). Gen4 has higher
resolution and generally higher mAP, and "looks" closer to a conventional image. Switching now is
nonetheless the wrong move:

| Factor | Implication of switching to Gen4 at week ~4/13 |
|---|---|
| **Retraining** | EventSSMDetector (46.2 mAP) came from a ~30 h Gen1 run. Gen4 = ~6× pixels → far more GPU memory/compute on a single RTX 5070 Ti; days per run, with PureSSM (and possibly the spiking model) still to train. |
| **Sunk work** | Discards the *exactly* reproduced baseline (47.7) and all Stage-8/9 results; everything redone. |
| **Data volume** | Gen4 raw is hundreds of GB to download/preprocess/store. |
| **Label quality** | Gen4 uses the same auto-generated labels with the same noise → does **not** fix §3. |

**On "higher resolution = more trustworthy results":** academically, trust comes from (a) using a
**standard benchmark** and (b) **reproducing the baseline**. Gen1 is *the* canonical event-detection
benchmark — the S5 paper, RVT, etc. all report Gen1 — and the baseline was reproduced to the
decimal (47.7). Gen1 numbers are therefore directly comparable to the published literature; that is
worth more than a higher-resolution visual. The **low resolution is also a strength** for the
micro-UAV thesis: a small sensor = less data = less power, which is precisely the SWaP constraint
the work targets. → **Keep Gen1 as the spine; list Gen4 validation as Future Work.**

## 3. Gen1 label quality — a real limitation (for the Discussion chapter)

Gen1's bounding-box labels are **manual annotations at 1–4 Hz** (de Tournemire et al. 2020, arXiv:2001.08499 —
*corrected 2026-10-06*: an earlier version of this note said "semi-automatically generated", which describes the
later 1Mpx dataset, not Gen1). They are nonetheless **sparse in time and incomplete**, especially for pedestrians
(small, distant, occluded). Measured example on test recording
`17-10-12_16-51-41_1647500000_1707500000` (59 s, busy urban drive):

- **128 GT boxes total → 117 car, only 11 pedestrian.** Visibly more pedestrians appear in the raw
  event stream than are labelled → **pedestrians are genuinely under-annotated**.
- This is **not** the size filter: the train/eval filter (`diag<30 px` or `side<10 px`) removes only
  **3/128** boxes here. The gaps are annotation incompleteness, not filtering.
- Labels exist at only **53 distinct timestamps (≈0.90 Hz)**, median **1 s** gap (max 5 s).

**Why this does not invalidate the results (important nuance):**
1. **Symmetric flaw.** EventSSMDetector, the S5-RVT baseline, and every published Gen1 number all
   train and score against the *same* imperfect labels → the 46.2 vs 47.7 comparison stays fair and
   literature-comparable.
2. **Self-consistent metric.** mAP is computed against the same labels the models learned from — one
   standardised (if imperfect) answer key for everyone.
3. **Robustness to label noise.** Networks learn the pattern from the many correct labels; missing
   ones add noise but do not prevent learning. The missing-pedestrian problem partly **explains** why
   pedestrian AP is low (~31) for *all* models — it is the hard class because the labels are sparse
   and the objects tiny.
4. **One genuine unfairness (worth a sentence):** if the model correctly detects an *unlabelled*
   pedestrian, mAP counts it as a **false positive**. So noisy labels make all scores a slight
   *under*-estimate of true ability — again, equally across models.

**Thesis framing:** acknowledge label noise as a limitation that **caps the absolute ceiling** but
**does not undermine the relative comparison or the contribution** (the same benchmark the whole
field uses). Good critical-analysis credit.

## 4. Why GT boxes "flicker" in the video — labels, not the model

Because GT exists at only ~53 moments (≈0.9 Hz) with ~1 s gaps, and the overlay holds each box for
250 ms, a box is on screen only ~22% of the time → on/off/on/off. This is an artefact of the
annotation rate + the visualisation hold, **not** the detector. Training/eval only score at labelled
frames, so the sparsity is handled correctly.

## 5. Event-visualisation tooling (built this session)

- **`code/event_ssm/scripts/stage9_dat_to_h5.py`** — converts raw Prophesee `.dat` → RVT `.dat.h5`
  (group `events/{x,y,p,t}`, blosc-lz4). Verified bit-exact (822,910 events match raw `.dat`; real
  `H5Reader` accepts it). Used `dat_events_tools.load_td_data` (not `PSEELoader`, which overflows
  under NumPy 2.x / NEP-50). Proof: `code/event_ssm/proofs/out/stage9_dat_to_h5_probe.png`.
- **`code/event_ssm/scripts/stage9_render_event_video.py`** — renders any recording (`.dat` or
  `.dat.h5`) as a red/blue event video (ON=red, OFF=blue), optional GT-box overlay (`--boxes`,
  car=green, ped=orange) and contact-sheet (`--montage`). Example:
  ```bash
  python code/event_ssm/scripts/stage9_render_event_video.py REC.dat.h5 \
      --duration-s 0 --frame-dt-ms 33 --fps 30 --boxes --montage
  ```
  Two reference clips rendered: a sparse static-camera scene (one pedestrian) and a dense urban
  drive (`17-10-12...`, 108.9 M events, 117 car + 11 ped). MP4s are large (~133 MB/60 s) → keep out
  of git; commit only the montage PNGs.

---

## 6. TODO — two visualisation tasks deferred (do later)

1. **Smoother GT-box overlay.** Hold boxes longer and/or interpolate between annotations using each
   object's `track_id` so boxes track cars/pedestrians continuously instead of flickering. Cosmetic;
   good for a clean thesis figure.
2. **Ground-truth vs. model-prediction video.** Run the trained EventSSMDetector inference over a
   recording and overlay **predicted boxes (e.g. magenta) next to GT (green)** → visualises hits,
   misses, and false positives. The "does it actually work" figure. Heavier (needs an inference pass:
   load checkpoint, forward, decode boxes per frame).
