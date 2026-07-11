# Stage 11 — PureSSM BiMamba Spatial Backbone: Build Notes

**Date:** 2026-07-11 · **Branch:** `stage11-puressm-backbone` (base `00e987c`) · **Plan:** `docs/superpowers/plans/2026-07-11-stage11-puressm-backbone.md` · **Spec:** `docs/superpowers/specs/2026-07-11-puressm-backbone-design.md`

## What was built

`code/event_ssm/spatial/` — a fully-pure, stateless, 4-stage bidirectional-Mamba spatial backbone, drop-in duck-type replacement for `ResNetSpatialStages`:

- **`_scan2d.py` — `BiMamba1DScan`:** shared in_proj/RMSNormGated/out_proj, per-direction conv1d/A_log/dt_bias/D, built on the sm_120-verified kernels (`mamba_chunk_scan_combined` + `causal_conv1d_fn`). After the Task-6a optimization, both directions run in **one conv call + one 2-group chunk-scan call** (head→group mapping verified against kernel source + exact-match probe; equivalence vs the two-call reference ≤4e-6, gate 2e-3; state-dict compatible).
- **`bimamba_block.py` — `BiMamba2DBlock`:** zero-init DWConv3×3 local mix + pre-norm bidirectional scan along one axis (row/col alternating per block) + DropPath (vendored). `LayerNorm2d` helper; no BatchNorm anywhere.
- **`bimamba_spatial.py` — `BiMambaSpatialStages`:** conv stem (stride 4) + 3 conv downsamplers, depths **[2,2,8,2]**, dims 64/128/256/512, strides 4/8/16/32, **8.38 M params** (gate 8–16 M). `checkpoint_blocks` flag = local-16 GB training fallback (engagement spy-tested: 14 calls on, 0 off).
- **`backbone/resnet_mamba.py`:** additive `spatial=` injection kwarg (default path byte-identical; forward/state helpers untouched).
- Scripts: `stage11_probe.py` (definitive U3 gate probe incl. CUDA-graph datapoint + `--smoke`), `stage11_erf.py` (ERF mechanism figure), `stage11_u1_proof.py`, `stage11_u2_proof.py`.
- Tests: **~29 new** (scan equivalence/flip/liveness/gradients/bf16, block axis semantics, pyramid contract/param-gate/statelessness, injection + state contract, checkpoint engagement); full suite green throughout (87 + 1 gpu-marked at close).

## Official probe results (user-run, RTX 5070 Ti idle, commit `3e8cf30`)

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| Backbone streaming p50 (eager) | 18.094 ms → **39.5 Hz** projected pipeline | ≥ 51 Hz | ❌ |
| Spatial pyramid alone (eager) | 15.25 ms (pre-6a: 22.7) | — | — |
| Train step, checkpointed (21,4) | **8.55 GB** | < 16 GB (local fallback) | ✅ |
| **CUDA-graph replay (fixed-state)** | **3.773 ms backbone** (4.8×) → ~91 Hz projected (~87 Hz with state copies) | datapoint | — |

**Diagnosis (measured, twice-confirmed):** eager cost is per-kernel-launch overhead, not compute — every block ≈1.6 ms flat regardless of sequence length (5120 tokens ≈ 80 tokens); graph replay collapses the backbone to <4 ms. `torch.compile(reduce-overhead)` is blocked by the view-return at `temporal/_scan.py:62` (fix = one-line `.clone()`, shared temporal path — deferred to Stage 16 with parity test).

## Decisions taken (all user-approved 2026-07-11)

1. **Gate fallback = engineering pass** (not depth cut / not conv stage 1): batched bidirectional scan (Task 6a, 1.45×) + checkpointing flag (Task 6b). The plan's original fallback ladder attacked sequence-length compute — the measured bottleneck was launch overhead; conv-stage-1 could only reach ~45 Hz eager (insufficient), closing that rung.
2. **Training moves to cloud:** rented RTX 5090 32 GB (vast.ai verified / RunPod), Stages 13+14; same sm_120 ⇒ zero stack mismatch; no checkpointing needed at 32 GB; eval + Stage-10 benchmarking stay local. Upload need: `data/gen1_raw/gen1` **train (58 GB) + val (15 GB)**; test (20 GB) stays local.
3. **Speed-floor framing = graph-capture deployment mode:** PureSSM's efficiency chapter reports eager 39.5 Hz AND a labeled graph-replay protocol column (~87–91 Hz projected, beats the 51 Hz floor and EventSSM's 70 Hz); Stage 16 implements state-carrying capture and measures it through the Stage-10 harness. Never silently substituted for the eager protocol.

## Findings the review loop caught (worth remembering)

- **Triton TF32 ≠ torch TF32:** `tl.dot` defaults to TF32 via `triton.knobs.language.fp32_default` (env `TRITON_F32_DEFAULT`), independent of `torch.backends.cuda.matmul.allow_tf32`. A kernel-vs-reference "0.13 error" was precision-mode mismatch; IEEE-forced error is 2.9e-4. Kernel caches must be cleared when toggling (warm-cache pollution).
- **LayerNorm null space:** a uniform all-channel perturbation is invisible to pre-norm blocks — tests probing propagation must perturb single channels.
- **Dead-ReLU ERF artifact:** probing effective receptive fields at a zeros input silences ReLU/bias-free networks entirely (ResNet gradient exactly 0.0) while leaking signal through conv biases elsewhere — ERF must average over random inputs (fixed; figure `proofs/out/u5_erf_resnet_vs_bimamba.png` now honest: ResNet = bounded radial blob, BiMamba = frame-spanning cross).
- `ssd_chunk_scan_combined_ref` doesn't natively support ragged seqlen — reference-side zero-padding to the chunk boundary is causally sound (dt pads to exactly 0, no softplus re-application) while the kernel runs raw.

## Visual proofs shipped

`proofs/out/`: `u1_scan_equivalence.png` (IEEE-forced, max 1.1e-3 @ s=1280), `u2_scan_order.png` + `u2_stage_table.md` (8.38 M), `u3_probe_table.md` + `u3_probe_latency.png` + `u3_probe.json`, `u5_erf_resnet_vs_bimamba.png` (mechanism figure, untrained; Stage 15 re-renders trained).

## Carried minors (for the final whole-branch review)

Suite-wide: 27 warnings characterized as 3 torch library DeprecationWarnings (torch.jit.script; torch.profiler event clearing) and UserWarnings (torch.meshgrid indexing) — none from our code; suite-close 87 passed + 1 gpu-marked. U1 proof figure 1.1e-3 vs nominal 1e-3 needs point-of-use note; `TRITON_F32_DEFAULT` env pollution persists for subprocesses post-test. `_scan2d.py`: `.contiguous()` channel-last micro-opt noted; setattr/getattr param idiom. `bimamba_block.py`: unused import/unpack (plan-verbatim); zero-init dwconv test asserts weights-only. `stage11_u2_proof.py` hardcodes `.cuda()`; duck-type test compares hardcoded literals. `spatial: nn.Module = None` type hint. Probe: `cudagraph_state_semantics` field set on both mechanisms (cosmetic). Process notes: one red-test commit (`f2bb24b`, corrected next commit); task-3 report contained cross-task paste.
