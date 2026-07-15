# Stage 16 — PureSSM Pillars & Visuals (Results)

**Thesis B | MMAN4952 | UNSW Sydney | Benas Vaiciulis** · started 2026-07-15

Completes the PureSSM investigation's evidence base so it matches EventSSM's: **efficiency** (Stage-10
analog), **temporal robustness** (Stage-9 analog), a **CUDA-graph deployment-speed** column, and
**qualitative GT-vs-pred visuals**. Accuracy (Stage 15) and receptive field (Stage 15 ERF) are already
done. `⏳` = awaiting a GPU run; the command that fills each is given beneath its table.

**Legend:** all three models share a **frozen** YOLO-PAFPN neck + YOLOX head + Gen1 pipeline + evaluator —
only the spatial backbone (and, EventSSM→PureSSM, *only* the spatial mixer) differs ⇒ controlled comparison.

---

## 0. Accuracy + size (measured — for reference)

| | S5-RVT baseline | EventSSM | **PureSSM** |
|---|---|---|---|
| test/AP (COCO) | 47.72 | 46.22 | **46.43** |
| **AP_L** (large obj) | 50.66 | 44.70 | **47.65** (+2.95 vs EventSSM) |
| Total params | 18.19 M | 19.18 M | **16.33 M** (smallest) |
| — spatial backbone | 12.44 M (MaxViT) | 11.23 M (ResNet-18) | 8.38 M (BiMamba) |
| — temporal | *(fused)* | 2.20 M (Mamba) | 2.20 M (Mamba) |
| — neck + head (shared) | 5.75 M | 5.75 M | 5.75 M |

*Headline so far: PureSSM is the **smallest** model, **matches** EventSSM overall, and **beats** it on large
objects — a compression win at equal accuracy. Full detail: `docs/Stage15_results_comparison.md`.*

---

## 1. Efficiency pillar (Slice A — Stage-10 analog)

Full-pipeline streaming inference, RTX 5070 Ti, idle-GPU-guarded harness (`code/event_ssm/benchmark/`).

| Metric | S5-RVT baseline | EventSSM | **PureSSM (eager)** |
|---|---|---|---|
| Latency p50 (ms) | 19.67 | 14.25 | ⏳ |
| Throughput (Hz) | 51 | 70 | ⏳ |
| Energy (J/frame) | 0.73 | 0.40 | ⏳ |
| FLOPs (G) | 11.89 | 13.55 | ⏳ |
| Streaming state / stream | 4.7 MB | 144 MB | ⏳ |

**Fill with:** `bash code/event_ssm/scripts/stage10_run_local.sh --models all && python code/event_ssm/scripts/stage10_report.py` *(needs an idle GPU)*

*Provisional (Stage-11 probe, not the harness): PureSSM eager ≈ 39.5 Hz — below EventSSM; see the CUDA-graph
column below for the deployment-mode figure.*

---

## 2. CUDA-graph deployment mode (Slice C)

State-carrying graph-replay latency — reported as a **separate labeled column, never substituted** for the
eager number (Stage-11 decision).

| | eager | **graph-replay (state-carrying)** |
|---|---|---|
| PureSSM backbone p50 (ms) | 18.09 (probe) | ⏳ (target ≈ 3.8) |
| PureSSM pipeline (Hz) | ≈ 39.5 (probe) | ⏳ (projected ≈ 87–91) |

**Fill with:** `bash code/event_ssm/scripts/stage10_run_local.sh --models all --graph` *(after Slice C lands the `graph_capture` helper + the `_scan.py` `.clone()` parity fix; needs idle GPU)*

*Status: not yet built. This is the first **real** state-carrying replay measurement — the Stage-11 probe's
3.77 ms was fixed-state (speed only, not streaming-correct).*

---

## 3. Temporal robustness pillar (Slice B — Stage-9 analog)

Same two regimes as Stage 9, same COCO-mAP metric. PureSSM sweeps **running 2026-07-15**.

### Regime 1 — fixed cadence, variable accumulation window (COCO mAP ×100)

| mult | dt (ms) | S5-RVT | EventSSM | **PureSSM** | PureSSM retention (÷own 1×) |
|---|---|---|---|---|---|
| 0.25× | 200 | 41.0 | 40.9 | **41.65** | 89.7 % |
| 0.5× | 100 | 46.5 | 45.5 | **45.67** | 98.3 % |
| 1× (train) | 50 | 47.7 | 46.2 | **46.45** | 100 % |
| 2× | 25 | 45.0 | 43.5 | **44.09** | 94.9 % |
| 4× | 12 | 38.4 | 35.4 | **35.68** | 76.8 % |

✅ **Regime-1 no-comp VALIDATED** (2026-07-15): PureSSM 1× = **46.45** (AP_L 47.42) matches its canonical
Stage-15 result (46.43, AP_L 47.65) → the eval chain is correct. **Reading:** PureSSM ≈ EventSSM in this
regime (tied within noise; 4× retention 76.8 % vs EventSSM 76.6 %), both marginally below S5-RVT (80.6 % @4×).
*Pending: Regime-1 **compensated** (Δt-scaled) sweep not yet run.*

> **Sub-finding — dense-window AP_L robustness (noted, single-seed).** At **0.25×** (dt 200 ms — a 4×
> accumulation window) PureSSM (41.65) *overtakes* the S5-RVT baseline (40.95, **+0.71**) — the only rate where it
> does (S5-RVT leads at every other rate, widening to −2.73 @4×). The driver is **large objects**: at 0.25×,
> PureSSM AP_L = **39.96** vs S5-RVT **36.38** (+3.58). The ViT baseline's large-object accuracy **collapses** at
> the dense/saturated window (AP_L 50.66 → 36.38, **−14.3**) while PureSSM's holds (47.65 → 39.96, **−7.7**) — the
> SSM-scan receptive field is more *robust* to smeared dense frames, extending the Stage-15 receptive-field
> advantage into robustness. **Do not over-claim:** single-seed ⇒ the overall +0.71 is within noise; the **AP_L
> gap (+3.58) is the trustworthy part.** It is one rate — at the sparse/fast end (4×) S5-RVT is clearly more robust.
> Honest framing: *PureSSM more robust at the dense/slow end, S5-RVT at the sparse/fast end — AP_L-driven, not a
> baseline-beating claim.*

### Regime 2 — true rate change (retention = mAP@rate ÷ mAP@1×) — THE headline robustness number

| | S5-RVT | EventSSM | ConvLSTM (paper) | **PureSSM** |
|---|---|---|---|---|
| true-10× retention | 62.2 % | 63.0 % | 17.7 % | ⏳ (sweep launched 2026-07-15) |

*⚠️ Note: the Regime-2 **true-rate** sweep is the dramatic SSM-vs-ConvLSTM regime and is the one that matters
most — PureSSM's number here is still pending (the ~6 h sweep). `stage9_truerate_plot.py` now prints a 3-model
table + a `retention @10x` line, so PureSSM's value slots in directly against EventSSM 63 % / ConvLSTM 17.7 %.*

**Fill with** (needs GPU; see `docs/Stage16_results.md` §runbook or the sweep commands — set `OURS_WRAP`/`OURS_CKPT`/`OURS_KEY`, then):
`RUN_BASELINE=0 bash code/event_ssm/scripts/stage9_eval_sweep.sh` · `bash code/event_ssm/scripts/stage9_mamba_scale_sweep.sh` · `RUN_BASELINE=0 bash code/event_ssm/scripts/stage9_truerate_sweep.sh` → then `python code/event_ssm/scripts/stage9_degradation_plot.py && python code/event_ssm/scripts/stage9_truerate_plot.py`

*Headline to extract: PureSSM true-10× retention vs EventSSM 63 % / ConvLSTM 17.7 % — does a pure-SSM keep the
intrinsic rate-robustness?*

---

## 4. Qualitative visuals (Slice D — EventCV GT-vs-pred)

Large-car sequences selected (`results/stage16/large_car_recs.txt`, 12 recs; top =
`18-03-29_13-15-02_500000_60500000`, 180 large-car instances). Pending: per-model prediction dump (GPU) →
GT-vs-pred overlay videos + large-car contact sheet.

- [ ] `results/stage16/*_gt_vs_pred.mp4` (EventSSM vs PureSSM boxes on the 12 sequences) ⏳
- [ ] `results/stage16/large_car_contact_sheet.png` ⏳

---

## Cross-references
- Accuracy + ERF: `docs/Stage15_results_comparison.md`
- Plan: `docs/superpowers/plans/2026-07-15-stage16-pillars-visuals.md`
- EventSSM references: Stage 8 (`docs/Stage8_results_comparison.md`), Stage 9 (`docs/Stage9_TwoRegime_Results.md`), Stage 10 (`docs/superpowers/plans/2026-07-10-stage10-efficiency-benchmark.md`)
- Deferred: PureSSM shrink study (memory `puressm-shrink-option`)
