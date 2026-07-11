# PureSSMDetector — BiMamba Spatial Backbone Design (Stages 11–16)

**Date:** 2026-07-11
**Status:** Approved (brainstorm 2026-07-11; all decisions user-locked)
**Dependencies:** EventSSMDetector complete (Stages 0–10, in-house baseline 46.2 AP / 70 Hz / 0.40 J-frame); RVT integration layer (`code/event_ssm/integration/register.py`); unified Mamba-2 temporal scan (`code/event_ssm/temporal/_scan.py`); Stage-10 benchmark harness (`code/event_ssm/benchmark/`); Stage-9 two-regime eval tooling.

---

## 1. Goal & Hypothesis

Build **PureSSMDetector**: the EventSSMDetector skeleton with exactly one organ swapped — the ResNet-18 convolutional spatial stages are replaced by **BiMamba spatial stages** (bidirectional Mamba-2 scans over each frame's token grid, scan axis alternating per block). Temporal Mamba blocks, streaming-state contract, YOLO-PAFPN, YOLOX head, losses, Gen1 pipeline, evaluator, and the 400k-step training recipe are reused **byte-identical**.

**Pre-registered hypothesis (from Stage 8):** EventSSM's deficit vs S5-RVT is concentrated in large objects (AP_L 44.70 vs 50.66, −5.96; AP_S −1.39 / AP_M −1.24 near parity; per-class it is a car gap, −2.21). Cause hypothesised: ResNet-18's local receptive field cannot supply the long-range spatial context that the baseline's global self-attention provides for close, frame-spanning cars. PureSSM applies the treatment — a global-receptive-field spatial mixer at every stage — and tests whether AP_L closes while AP_S/AP_M hold.

**Interpretation bands (pre-registered, from Stage-8 scenario bands):**

| Gen1 test/AP | Reading |
|---|---|
| > 47.7 | Beats the published S5-RVT baseline — headline result |
| 46.2 – 47.7 | Beats the in-house EventSSM baseline — receptive-field hypothesis supported if AP_L closes |
| 44 – 46.2 | Ambiguous: cannot separate spatial-mixer effect from lost ImageNet init (see §5 caveat) |
| < 44 | Negative but valid controlled result; discuss via MambaVision's finding (attention, not SSM, may best capture global context at low res) |

AP_L is reported explicitly in all cases; the hypothesis verdict rides on AP_L (target: 44.70 → toward 50.66 with AP_S ≈ 37.4 / AP_M ≈ 53.7 held), not only overall AP.

---

## 2. Locked Decisions (brainstorm 2026-07-11)

| Decision | Choice | Rationale |
|---|---|---|
| Priority | **Accuracy first, ≥ 51 Hz full-pipeline floor** | Thesis tells a Pareto story: EventSSM = efficiency champion (70 Hz / 0.40 J), PureSSM = accuracy/global-context experiment. Floor = S5-RVT baseline speed. |
| Purity | **Fully pure: SSM mixer at all 4 stages** | Strongest hypothesis test; completes the ablation matrix (attention/conv/SSM × space, SSM × time); preserves "every learnable mixer is an SSM" for the SSSMDetector/Loihi fork. Conv appears only as stem/downsampling + in-block DWConv3×3 (VMamba-standard; the *mixer* defines the family). |
| Pretraining | **From scratch; SSL held in reserve** | S5-RVT and SMamba both train from scratch (SMamba: 50.4 Gen1 mAP from scratch — feasibility proof). The EventSSM asymmetry (ImageNet init) runs *against* PureSSM, so a win is untainted. If the result lands in the ambiguous band and calendar allows, an SSL-pretrain follow-up already has its control arm done. |
| Scan design | **Approach 1: alternating-axis bidirectional scan** (Vim × Mamba-ND) | 2 scans/block along one axis, axis alternates per block; full-frame context every 2 blocks. Mamba-ND: alternating 1D scans beat per-layer multi-direction designs. Lowest risk: builds on the repo-verified `_scan.py` kernel pattern. |
| Upgrade lever (pre-registered) | Cross-scan (SS2D-style 4-direction) **on stages 3–4 only**, *if* the latency probe shows headroom | Stage-3/4 grids are tiny (320 / 80 tokens) — near-free accuracy where large-object semantics live. Decided by measurement, not up front. |
| Rejected: SS2D everywhere | VMamba's most-accurate ordering, but ~2× scan work + transpose overhead; VMamba needed a custom fused Triton kernel to be fast. Latency risk to the 51 Hz floor. |
| Rejected: Hydra quasiseparable | Principled Mamba-2-native bidirectionality (NeurIPS 2024) — cite in the write-up, do not implement: no detection precedent, most derivation risk ("too much fail risk" — user). |
| Rejected: hybrid conv stage 1 | MambaVision precedent, but partial-swap ablation, weaker "Pure" claim, spiking fork inherits a conv. Retained only as the *fallback* if probe gates fail (explicit revisited decision, never silent). |
| Rejected: SSL-pretrain-first / distillation | 3–5 weeks + two 400k runs before the hypothesis is answered / contaminates the controlled ablation. |
| Scope | **All three result pillars** + visual units | Stage-10 bench re-run, Stage-9 two-regime re-eval, EventCV GT-vs-pred unit, ERF figure. |
| SSM variant | **Mamba-2 chunk-scan everywhere** (spatial and temporal) | Stage-6 lock: SSM flavour is never a confound. Only the Mamba-2 path is fwd+bwd verified on sm_120. Mamba3 (present in env) explicitly out of bounds. |

---

## 3. Contract (must not break; all verified in the 2026-07-11 code map)

- **Spatial seam:** the spatial module runs on the time-folded batch — `spat = self.spatial(x.reshape(L*B, C, H, W))` (`code/event_ssm/backbone/resnet_mamba.py:63`). A spatial replacement sees `(N, 20, 256, 320)` and never touches the time axis or state ⇒ **stateless by construction**; bidirectionality is over space within one already-buffered frame. Stage-00's rejection of bidirectional Mamba applies to the *time* axis only (lookahead latency); this design is fully causal in time.
- **Duck type of `ResNetSpatialStages`** (`code/event_ssm/backbone/resnet_spatial.py:15-37`): attributes `stage_dims=(64,128,256,512)`, `strides=(4,8,16,32)`; `forward(x:(N,20,H,W)) → dict {1: (N,64,H/4,W/4), 2: (N,128,H/8,W/8), 3: (N,256,H/16,W/16), 4: (N,512,H/32,W/32)}`. `ResNetMambaBackbone.get_stage_dims/get_strides` index these 1-based (`resnet_mamba.py:52-56`); YoloXDetector consumes them (`external/.../detector.py:25-33`).
- **FPN:** PAFPN asserts exactly 3 maps, `in_stages=[2,3,4]` → channels 128/256/512 at strides 8/16/32 → 32×40 / 16×20 / 8×10 on the padded 256×320 input (`configs/resnet_mamba_yolox/default.yaml:21,29`; `yolo_pafpn.py:35,104-113`).
- **Features layout:** `Dict[int, (L,B,c,h,w)]` — RVT slices `v[tidx]` per timestep (`external/.../modules/detection.py:188-194`).
- **States:** list, one entry per stage; every leaf a detached tensor with batch as dim 0; **no Nones** (placeholder `seq.new_zeros(B,1)` on non-temporal stages, `resnet_mamba.py:75-78`); `RNNStates.recursive_detach/reset` traverse and zero rows by batch index (`external/.../modules/utils/detection.py:96-141`). **PureSSM adds no new state** — the temporal path is untouched, so the state structure is unchanged (streaming state stays ≈ 144 MB/stream).
- **Temporal blocks on FPN-fed stages [2,3,4] only** (Finding §8); temporal Mamba-2 config unchanged: `d_state=64, d_conv=4, expand=2, headdim=64` (`temporal/mamba_temporal.py:14-21`).
- **Padding:** `in_res_hw=(256,320)` (multiple of 32) set by the patched config modifier (`integration/register.py:67-71`).
- **Registration:** extend `register.py`'s patched `build_recurrent_backbone` with a `PureSSM` name branch (generalise the `"ResNetMamba"` hardcode at `register.py:59` to a name set). Config pair `code/event_ssm/configs/puressm_yolox/default.yaml` + `experiment/gen1/puressm.yaml` (fpn/head/postprocess copied verbatim), symlinked into the gitignored RVT config tree; selection `model=rnndet +experiment/gen1=puressm`. No edits to `external/`.
- **Environment:** torch 2.11.0+cu128 (sm_120), mamba-ssm 2.3.2.post1 + causal-conv1d 1.6.2.post1 — installed `--no-deps --no-build-isolation` only. **No new torch-dependent packages.** timm is absent: vendor DropPath (~10 lines) if needed; einops 0.8.2 available. The installed mamba-ssm has **no `bimamba` argument** (Vim-fork only) and causal-conv1d has no bidirectional mode — the bidirectional block must be composed in-repo (this is load-bearing for the "own model" contribution).

---

## 4. Architecture: `BiMambaSpatialStages`

New package `code/event_ssm/spatial/`:

```
code/event_ssm/spatial/
├── __init__.py
├── _scan2d.py          # bidirectional Mamba-2 scan core (extends temporal/_scan.py pattern)
├── bimamba_block.py    # BiMamba2D block: DWConv3x3 local mix + bidirectional scan + gate
└── bimamba_spatial.py  # BiMambaSpatialStages: stem, 4 stages, downsamplers (duck-type)
```

### 4.1 Stem & downsamplers (the only spatial convs)

- **Stem** (stride 4): two 3×3 stride-2 convs (20→32→64) each followed by norm + activation (VMamba-v2-style). Output `(N, 64, 64, 80)`.
- **Downsamplers** between stages: single 3×3 stride-2 conv + norm (64→128, 128→256, 256→512).
- **No positional embeddings**: conv stem + strided downsampling + in-block DWConv provide position (VMamba evidence). Norms are LayerNorm-family (channels-last or ln2d) — no BatchNorm anywhere ⇒ the Stage-3 BN-over-L*B caveat disappears.

### 4.2 BiMamba2D block (per block, pre-norm residual)

1. **Local mix:** depthwise 3×3 conv on the 2D map (+ pointwise gate-free residual) — VMamba lost 0.6 % top-1 removing it; also feeds relative position.
2. **Global mix — bidirectional scan along ONE axis:** flatten the grid row-major (even block index) or column-major via transpose (odd block index). Shared `in_proj` → split z / xBC / dt. **Forward direction:** `causal_conv1d_fn` + `mamba_chunk_scan_combined` on the sequence. **Backward direction:** same kernels on the flipped sequence with **per-direction conv weights, A_log, dt_bias** (Vim/VMamba both learn separate direction params; sharing decay ties the two directions' lengthscales), output flipped back. `y = y_fwd + y_bwd` → single shared `RMSNormGated(y, z)` → shared `out_proj`. `initial_states=None`, `return_final_states=False` — a frame's spatial sequence is complete; nothing is carried.
3. Residual add; optional DropPath (vendored) at rates ≤ 0.1, linearly scaled with depth.

Spatial SSM hyperparameters (deliberately ≠ temporal): `d_state=16` (VMamba: spatial scans don't need large state; temporal keeps 64), `expand=2`, `headdim=64` (d_inner 128/256/512/1024 → 2/4/8/16 heads — all divisible), `ngroups=1`, `chunk_size=256`, `d_ssm=d_inner` (respects the `_scan.py` asserts).

### 4.3 Stage layout & budget

| Stage | Grid (padded 256×320) | Tokens | d_model | Depth | Scan axes (alternating) |
|---|---|---|---|---|---|
| 1 | 64×80 | 5120 | 64 | 2 | rows, cols |
| 2 | 32×40 | 1280 | 128 | 2 | rows, cols |
| 3 | 16×20 | 320 | 256 | 8 | rows, cols, … |
| 4 | 8×10 | 80 | 512 | 2 | rows, cols |

Depths [2,2,8,2] (VMamba-style: depth where stage 3 does semantic heavy lifting). **Param gate: spatial module 8–16 M** (ResNet-18 reference: 11.2 M; estimate ≈ 8–10 M ⇒ total model ≈ 16–18 M vs EventSSM 19.2 M). Exact per-block param count is asserted by a unit test, not hand-waved.

### 4.4 Numerics & known failure modes

- bf16 autocast end-to-end as per the verified temporal path; **NaN-guard test** on the block under bf16 (VMamba documents fp16 scan instability; bf16 is the repo-verified regime — verify explicitly anyway).
- **Mamba-R artifact watch:** Vim-style bidirectional features develop high-norm background artifact tokens even at tiny scale. Mitigation: per-stage feature-norm monitor logged during training from step 0; register tokens documented as the fallback fix (not built pre-emptively — YAGNI).

---

## 5. Training Protocol (byte-identical to Stage 7, for comparability)

400k steps OneCycle lr 2e-4 (pct_start 0.005, div_factor 20), `sequence_length=21`, batch 4 (constant across carried subsequences — state cache is N=B·H·W), bf16-mixed **no GradScaler** (ISSUE-09), workers 2/1 (OOM history 2026-06-17), `stacked_histogram_dt=50_nbins=10`, mixed sampling, selection `model=rnndet +experiment/gen1=puressm`. Run handed to the user (foreground/tmux, live bar). Expected wall-time 40–50 h (accepted under accuracy-first; EventSSM took ~30 h).

**Pre-registered caveat (pretraining):** EventSSM's ResNet stages were ImageNet-initialised (avg-projection); no BiMamba equivalent exists. Interpretation rule: a PureSSM result **below** EventSSM is ambiguous (mixer vs initialisation); a result **at or above** EventSSM cannot be explained by initialisation. PureSSM vs S5-RVT is fully fair (both from scratch).

---

## 6. Units (each: Init / Forward / Tests / Visual proof)

**U1 — `_scan2d.py` bidirectional scan core.**
Init: per-direction conv1d weights + A_log + dt_bias; shared in/out proj + RMSNormGated (extend `temporal/_scan.py` structure; same asserts).
Forward: `(N, S, d_model) → (N, S, d_model)` for a given flatten order.
Tests: equivalence of the forward direction vs `ssd_chunk_scan_combined_ref` (≤1e-3 fp32); flip-symmetry (feeding the flipped sequence to the swapped-direction params reproduces the mirrored output); gradient flows to both directions' params; bf16 NaN guard.
Visual proof: `proofs/out/u1_scan_equivalence.png` (fwd-vs-ref error histogram + fwd/bwd contribution norms).

**U2 — `bimamba_block.py` + `bimamba_spatial.py`.**
Init: stem, downsamplers, [2,2,8,2] stages with alternating axes; `stage_dims`/`strides` attributes.
Forward: `(N,20,256,320) → dict{1..4}` with exact shapes above; statelessness (two calls, same output — no hidden buffers mutate).
Tests: duck-type contract (attribute + shape parity with `ResNetSpatialStages`); axis alternation is exercised (odd/even blocks differ under a transposed input probe); param-count gate 8–16 M; block-level bidirectionality (masking future-in-scan-order tokens changes earlier outputs — proves the backward path is live).
Visual proof: `proofs/out/u2_scan_order.png` (row/col scan-order schematic rendered from the actual index tensors) + per-stage output-shape table.

**U3 — Latency/VRAM probe (GATE — before any training).**
Forward the assembled `ResNetMambaBackbone(spatial=BiMamba…)` at streaming shape `(1,1,20,256,320)`, p50 over ≥300 iters on idle GPU; train-shape `(21,4,20,256,320)` forward+backward peak VRAM.
**Gates: projected full pipeline ≥ 51 Hz** (backbone p50 ≤ ~12.5 ms, since neck+head ≈ 7.2 ms) **and train step < 16 GB.**
Fail → fallback ladder, in order, each an explicit user decision: (a) stage-1 depth 2→1; (b) stage-1 d_state 16→8 / thinner expand; (c) conv stage 1 (hybrid — revisits the purity decision openly).
Visual proof: `proofs/out/u3_probe_table.md` + latency-per-stage bar chart.

**U4 — Registration + configs.**
`register.py`: name-set dispatch (`{"ResNetMamba", "PureSSM"}`); `puressm_yolox/default.yaml` + `experiment/gen1/puressm.yaml` symlinked.
Tests: Hydra compose selects PureSSM; built model passes the RVT-contract integration test (forward with `LstmStates` carry + reset).
Visual proof: config-diff table vs `resnet_mamba` (only backbone block differs).

**U5 — ERF mechanism figure.**
Gradient-based effective-receptive-field maps at matched stages: ResNet stage vs BiMamba stage (untrained + trained-checkpoint variants).
Visual proof: `proofs/out/u5_erf_resnet_vs_bimamba.png` — the thesis's mechanism figure for the AP_L hypothesis.

**U6 — EventCV visual unit (Stage 16).**
`pip install eventcv` into `events_signals` (safe: numpy-only dependency tree, abi3 wheel — verified 2026-07-11 from PyPI metadata; it is NOT a torch-dependent package so the `--no-deps` rule does not bind). Render GT-vs-pred overlays (EventSSM vs PureSSM) on ≥10 large-car test sequences + the two deferred video TODOs (smooth event overlay, GT-vs-pred video). EventCV is **downstream-only**: it never regenerates training/eval tensors (fair-comparison invariant).
Visual proof: `results/stage16/` videos + a contact-sheet figure of large-car cases.

---

## 7. Stage Roadmap (Stages 11–16, mirroring Stages 3–10)

| Stage | Mirrors | Content | Exit gate |
|---|---|---|---|
| **11 — Backbone build** | 3 | U1 + U2 + U3 (probe) + U5 (untrained ERF) | All unit tests green; probe gates passed (or fallback decision taken); param gate |
| **12 — Integration + smoke** | 4+5 | U4; overfit one real Gen1 batch; full test suite alongside existing 60 | Overfit converges; integration tests green |
| **13 — Short training** | 6 | ~25k-step run, monitors live (feature norms, NaN guard) | val/AP in the EventSSM-short-run band (≈0.10–0.15); no instability |
| **14 — Full run** | 7 | 400k steps, Stage-7 recipe byte-identical (user-run, tmux) | Run completes; best-ckpt val/AP recorded |
| **15 — Evaluation** | 8 | Gen1 test AP + size-stratified + per-class; interpretation via §1 bands; trained ERF | Results table + verdict vs pre-registered bands |
| **16 — Pillars + visuals** | 9+10 | Stage-9 two-regime re-eval row; Stage-10 bench re-run row (MAMBA_STEP_SCALE unset; ~5 min); U6 EventCV unit; upgrade-lever decision (cross-scan stages 3–4) if headroom and accuracy warrants | Three pillar artifacts + videos/figures shipped |

Each stage gets its own plan in `docs/superpowers/plans/` (subagent-driven development), written when the stage starts; Stage 11's plan is written now.

---

## 8. Testing (TDD — written before implementation)

Unit: U1 equivalence/flip/gradient/NaN; U2 contract/shapes/statelessness/param-gate/bidirectionality-liveness; axis-alternation. Integration: RVT forward with state carry/reset, Hydra select, `pytest code/event_ssm/tests/` stays green (60 existing + new; gpu-marked where CUDA-only). All CUDA tests on the idle 5070 Ti.

## 9. Success Criteria

1. All tests green (existing 60 + new suite).
2. Probe gates: ≥ 51 Hz projected pipeline, < 16 GB train step, spatial params 8–16 M.
3. 400k run completes on the locked recipe; test AP + AP_L reported against the §1 bands.
4. Three pillar artifacts: Stage-15 results table, Stage-16 bench row, two-regime row.
5. Visual proofs shipped per unit (ERF figure, scan-order figure, probe table, EventCV videos).

## 10. Out of Scope

SSL pretraining (reserve follow-up); Hydra mixer; SS2D-everywhere; register tokens (documented fallback only); Mamba3; any change to temporal blocks, neck, head, losses, data, evaluator, or training recipe; Katana migration; SSSMDetector (but: no non-SSM mixer may be introduced — spiking-fork property preserved); leaderboard chasing vs SMamba (positioned as controlled ablation, not SOTA entry).

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Stage-1 scan latency erodes speed story | U3 probe gates **before** training; explicit fallback ladder; upgrade/downgrade decided by measurement |
| Train VRAM > 16 GB (activations of 5120-token bidirectional scans under TBPTT) | U3 train-shape probe; batch already 4; activation checkpointing on spatial blocks as reserve |
| Training instability from scratch (no ImageNet init) | bf16 (repo-verified), OneCycle warmup already in recipe, NaN guard, feature-norm monitor, short-run gate before 400k commit |
| High-norm artifact tokens (Mamba-R failure mode) | Per-stage norm monitor from step 0; register-token fallback documented |
| Result in ambiguous band (44–46.2) | Pre-registered interpretation rules (§5); SSL follow-up already designed with control arm done |
| Contract regression breaks RVT streaming | Duck-type + integration tests; state structure untouched by construction |
| Kernel surprises (chunk scan on very short seqs, e.g. 80 tokens) | U1 equivalence test covers all four stage lengths (5120/1280/320/80) |

## 12. References (for the thesis chapter)

Vim (arXiv 2401.09417, ICML 2024); VMamba (arXiv 2401.10166, NeurIPS 2024); Mamba-ND (arXiv 2402.05892, ECCV 2024); Hydra (arXiv 2407.09941, NeurIPS 2024); MambaVision (arXiv 2407.08083, CVPR 2025); MambaOut (arXiv 2405.07992, CVPR 2025); Mamba-R (arXiv 2405.14858, CVPR 2025); **SMamba (arXiv 2501.11971, AAAI 2025)** — closest competitor: spatial sparse-SS2D + ConvLSTM temporal, Gen1 50.4 mAP from scratch; differentiate on the temporal axis (Mamba state-carry) and the controlled-ablation framing. Novelty claim (checked 2026-07-11): no published combination of bidirectional spatial Mamba + temporal Mamba recurrent state-carry on Gen1.
