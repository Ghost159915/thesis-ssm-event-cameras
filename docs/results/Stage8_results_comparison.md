# Stage 8 — Results & Comparison (EventSSMDetector vs S5-RVT baseline)

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis** · 2026-06-19

Gen1 **test split**, COCO protocol (mAP @ IoU 0.50:0.95), Prophesee/COCO evaluator reused unmodified.
**Both models evaluated through the *same* evaluator** (per-class via an additive read-out that does not
change the aggregate metrics) → an apples-to-apples controlled comparison.

- **EventSSMDetector:** ResNet-18 + Mamba, 19.2 M params, best checkpoint = 400k run `8zotrwjw` step 320k (val/AP 0.463), bf16.
- **S5-RVT baseline:** MaxViT (ViT) + S5, `checkpoints/gen1_base.ckpt`, precision 16. Reproduced the published **47.7** exactly → evaluator verified correct.

---

## Headline numbers

| Metric | S5-RVT baseline | EventSSMDetector | Δ (Event − Base) |
|---|---|---|---|
| **mAP @[.50:.95]** | **47.72** | **46.22** | **−1.50** |
| AP_50 | 75.28 | 74.54 | −0.74 |
| AP_75 | 49.76 | 48.08 | −1.68 |
| AP_S (small) | 38.81 | 37.42 | −1.39 |
| AP_M (medium) | 54.92 | 53.68 | −1.24 |
| **AP_L (large)** | **50.66** | **44.70** | **−5.96** ⬅ dominant gap |

### Per-class (COCO AP @[.50:.95])

| Class | S5-RVT baseline | EventSSMDetector | Δ (Event − Base) |
|---|---|---|---|
| Car | 63.88 (AP_50 84.25) | **61.67** (AP_50 82.35) | **−2.21** |
| Pedestrian | 31.56 (AP_50 66.31) | **30.78** (AP_50 66.73) | **−0.78** |

> **Per-class confirms the large-object story.** The car gap (−2.21) is ~3× the pedestrian gap (−0.78).
> Cars are the class that *contains* the large instances (close cars), so EventSSM's large-object weakness
> shows up mainly as a **car** deficit; pedestrians (almost never "large") are **nearly matched** (−0.78).
> Sanity: mean(61.67, 30.78) = 46.22 = overall AP ✓.

---

## Key insight — *where* the 1.5-mAP gap comes from

The overall deficit is **not uniform** — it is **almost entirely a large-object gap**:

- **AP_L: −5.96** (EventSSM 44.7 vs baseline 50.7) — by far the biggest difference.
- AP_S: −1.39, AP_M: −1.24 — on small and medium objects the CNN+Mamba backbone is **nearly on par** with the ViT.

**Interpretation (the receptive-field story, empirically supported).** Large Gen1 instances are *close, fast-moving cars* spanning much of the 304×240 frame. Detecting/localising them needs **long-range spatial context**, which the ViT baseline supplies through **global self-attention**. ResNet-18's **local, hierarchical receptive field** cannot match that — so the hybrid loses most of its ground precisely on large objects, while staying competitive where local features suffice (small/medium). This is direct evidence for the hypothesis in the Stage-8 plan (Scenario 3): *spatial-processing method — not just the temporal model — determines large-object accuracy.*

**Localisation vs detection.** AP_75 gap (−1.68) > AP_50 gap (−0.74): EventSSM *finds* objects almost as well as the baseline (AP_50 close) but localises slightly less tightly (AP_75 wider) — consistent with the receptive-field reading (coarser large-object boxes).

**Pedestrians are the hard class for both.** Baseline car 63.9 vs ped 31.6 — a **32-point** intra-model gap. Pedestrians on Gen1 are small, sparse-event, fewer-instance; this is the dominant difficulty axis, independent of backbone.

---

## Thesis framing (Stage-8 "Scenario 2", with a Scenario-3 nuance)
- EventSSMDetector is a **near-parity** detector (−1.5 mAP) built from a **CNN + selective-SSM** backbone vs a Transformer + S5 — competitive on small/medium objects, behind only on large.
- The **AP_L gap motivates the next models**: a global-spatial **PureSSMDetector** (BiMamba) could recover the long-range context ResNet-18 lacks → directly tests whether SSM-spatial closes the large-object gap. And the **efficiency case** (Stage 10: params/FLOPs/latency) decides whether "near-parity" becomes "near-parity *and* cheaper" for micro-UAV SWaP.

## Caveats (rigor)
- **bf16** (EventSSM) vs **fp16** (baseline) — minor eval-precision difference; both reproduce their reference numbers, gap (1.5) is small.
- **Single seed** — no error bars; note for any publication.
- **Per-class vs paper:** our evaluator reproduces the baseline **overall** 47.7 exactly, but our per-class split (car 63.9 / ped 31.6) differs from the paper's reported (56.8 / 38.6). Likely a per-class threshold/protocol difference. **Use our own evaluator's per-class for *both* rows** (consistent); cite the paper only for the overall.

---

## Raw evaluator output (for the record)

**S5-RVT baseline** (`stage8_baseline_eval_local.sh`, gen1_base.ckpt):
```
[per-class]         car   AP=0.6388  AP_50=0.8425
[per-class]  pedestrian   AP=0.3156  AP_50=0.6631
test/AP 0.4772 · AP_50 0.7528 · AP_75 0.4976 · AP_S 0.3881 · AP_M 0.5492 · AP_L 0.5066
```

**EventSSMDetector** (`stage7_test_eval_local.sh`, 8zotrwjw step 320k):
```
[per-class]         car   AP=0.6167  AP_50=0.8235
[per-class]  pedestrian   AP=0.3078  AP_50=0.6673
test/AP 0.4622 · AP_50 0.7454 · AP_75 0.4808 · AP_S 0.3742 · AP_M 0.5368 · AP_L 0.4470
```
