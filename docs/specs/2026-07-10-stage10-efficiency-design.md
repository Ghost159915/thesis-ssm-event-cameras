# Stage 10 — Efficiency Benchmarking Harness: Design Specification

**Date:** 2026-07-10 · **Status:** approved (brainstorm 2026-07-10, user-selected options locked)
**Thesis context:** answers Thesis-A **Research Gap 3** (systematic SSM inference latency / memory /
energy benchmarking for onboard micro-UAV deployment). Completes the third pillar of the
EventSSMDetector investigation: accuracy (Stage 8, test/AP 46.2 vs S5-RVT 47.7) · robustness
(Stage 9, two-regime temporal generalisation) · **efficiency (this stage)**.

## 1. Locked decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Latency headline | **Full pipeline** (backbone → neck → head → postprocess/NMS); network-only and per-component reported underneath |
| Realism | **Real checkpoints + real Gen1 frames** (NMS load realistic → headline defensible) |
| Metric scope | **Strict five**: params, FLOPs, latency, peak VRAM, energy/frame. Nothing else; extras = future work |
| Architecture | **Measure/report split** (approach B): one-shot GPU orchestrator → JSON → CPU reporter |
| Measured models | EventSSMDetector (ours) + S5-RVT baseline, identical protocol. RVT-LSTM = cited numbers only |
| Hardware | RTX 5070 Ti (`GhostMachine`); ratios transfer, Jetson-class contextualisation in the writeup |

## 2. Package layout & data flow

```
code/event_ssm/benchmark/
  __init__.py
  bench_models.py       # construction + uniform BenchModel wrapper (real ckpts)
  bench_metrics.py      # the five measurement primitives
code/event_ssm/scripts/
  stage10_benchmark.py  # GPU orchestrator (run ONCE, idle GPU) -> results/stage10/bench_results.json
  stage10_report.py     # CPU reporter: JSON -> efficiency_table.{md,csv} + 2 figures
  stage10_run_local.sh  # launcher: conda env, PYTHONPATH, idle-GPU guard, tee log
code/event_ssm/tests/
  test_bench_metrics.py # unit tests (analytic FLOPs vs hand cases, timing util, JSON schema)
  test_bench_report.py  # reporter against a fixture JSON (CPU-only)
```

Flow: `stage10_run_local.sh` → orchestrator builds both models with real weights → loads the cached
real clip → runs the five metrics per (model × precision) → writes **one**
`results/stage10/bench_results.json` (single source of truth, plus env metadata) → `stage10_report.py`
regenerates tables/figures from JSON at zero GPU cost. Presentation iterations never re-measure.

## 3. bench_models — construction & the uniform interface

**Config assembly (no hardcoded dims).** Use Hydra's compose API against `external/ssms_event_cameras/
RVT/config` exactly as the eval scripts do:
- Ours: `register_resnet_mamba()` first, then overrides `dataset=gen1 model=rnndet
  +experiment/gen1=resnet_mamba` → `YoloXDetector(cfg.model)`.
- Baseline: overrides `dataset=gen1 model=rnndet +experiment/gen1=base.yaml` → `YoloXDetector(cfg.model)`.
This guarantees dimensions match the checkpoints (the construction path is the one
`proof_integration.py` verified; the config path is the one `stage7_eval.py`/`stage8_baseline_eval.py`
use).

**Checkpoint loading.** `torch.load(ckpt)["state_dict"]`, strip the Lightning `mdl.` prefix, load into
the `YoloXDetector` with `strict=True` (fail loudly on any mismatch). Checkpoints:
- ours: `external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt`
- baseline: `checkpoints/gen1_base.ckpt`
Note: compute metrics are weight-independent; real weights matter only for realistic NMS load. EMA vs
raw weights therefore does not affect any metric; load the plain `mdl.*` weights and record this in the
JSON meta.

**Uniform wrapper.** `BenchModel` exposes:
- `full_step(frame_1x20x256x320, state) -> (detections, state)` — backbone (`forward_backbone`, L=1,
  `train_step=False`, states carried) → features {2,3,4} → `forward_detect` → RVT `postprocess`
  (conf/nms thresholds read from the composed config — the same values as the Stage-8/9 AP evals;
  conservative worst-case NMS load, consistent with the accuracy numbers the table sits next to).
- `network_step(...)` — same minus postprocess (for the network-only latency row and FLOPs tracing).
- `components` — named callables for the per-component pass: `backbone`, `neck+head`, `postprocess`
  (neck and head are timed as one unit: RVT's `forward_detect` fuses them; splitting would require
  modifying `external/`, which this stage does not do).
- Precision: bf16 = `torch.autocast("cuda", dtype=torch.bfloat16)` around network compute (mirrors
  training/eval); fp32 = plain. Both measured for latency/VRAM; FLOPs/params are precision-independent;
  energy measured at bf16 only (the deployment configuration).

## 4. Real data clip

64 consecutive frames from one canonical dt=50 test sequence (`data/gen1_raw/gen1/test`, the validated
Stage-8 data), read **directly from `event_representations.h5`** (no Hydra dataset stack), zero-padded
240×304 → 256×320 exactly as the pipeline does. Tensor dtype/scaling must match what the RVT eval
dataloader feeds the model — the implementer verifies this against the RVT sequence loader (stacked
histograms are stored as uint8 counts; confirm the float conversion point) and records the choice in
the JSON meta. Cached to `results/stage10/bench_clip.pt` with its source sequence name in the meta.
Streaming metrics cycle these 64 frames (B=1, L=1, state carried). Throughput at B=4/8 replicates the
frame across the batch dimension.

## 5. bench_metrics — the five protocols

**5.1 Params.** Module walk: total + {backbone-spatial, backbone-temporal, neck, head} for ours;
{backbone, neck, head} for the baseline (the ViT+S5 stages are not split further). Reported in M.

**5.2 FLOPs (per frame, network-only, B=1, L=1).** Two additive parts, both recorded:
- *Counted:* `fvcore.FlopCountAnalysis` on `network_step`; unsupported-op warnings captured into the
  JSON (never crash — wrap and continue). Known blind spots: `mamba_chunk_scan_combined`,
  `causal_conv1d` (Triton), possibly the S5 `torch.vmap` scan body.
- *Analytic:* closed-form MACs for the blind spots, implemented as pure functions with unit tests
  against hand-computed small cases:
  - Mamba-2 layer, per token (d_model c, d_inner = 2c, heads h = d_inner/headdim, state S, conv k=4,
    ngroups=1): in_proj `c·(2·d_inner + 2S + h)` + depthwise conv `k·(d_inner + 2S)` + selective scan
    `≈ 3·d_inner·S` (state decay, input injection, output read-out) + gating/norm `≈ 2·d_inner` +
    out_proj `d_inner·c`; × N tokens (N = H·W of the stage) × layers × temporal stages.
  - S5 block, per token (dim c = state P): discretisation (bilinear, complex) `≈ 4P` + state update
    `≈ 4P` (complex mult) + input/output maps `≈ 2·P·c` (complex B̄u, Cx real parts) + GLU/FF per
    S5Block config; × N tokens × 4 stages. Exact constants derived in the implementation plan and
    unit-tested; the spec fixes the *approach*: complex MAC = 4 real MACs, report MACs and FLOPs = 2·MACs.
- Table reports counted, analytic, and total, with the split visible (academic transparency). GFLOPs
  per frame; plus **mAP/GFLOP** = Stage-8 test/AP (0.462 / 0.477) ÷ **total** network GFLOPs
  (counted + analytic) per frame.

**5.3 Latency.** Steady-state streaming: 50 warmup steps then 300 timed steps of `full_step` (B=1,
L=1, state carried, real frames cycled), `torch.cuda.synchronize()` bracketing each timing,
`time.perf_counter`. Report mean ± std, p50, p95, and Hz (1000/p50). Second series for
`network_step` (network-only row). **Third pass** times components separately (backbone / neck+head /
postprocess) so instrumentation never pollutes the headline pass. Throughput: B=4 and B=8 batched
steps → frames/s. All × {bf16, fp32}. GPU clocks recorded before/after via
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu`.

**5.4 Peak VRAM.** `torch.cuda.empty_cache() + reset_peak_memory_stats()` then: (a) 20 streaming
inference steps → peak MB; (b) one training-mode forward+backward at B=4, L=5 with synthetic targets
(the proof_integration pattern) → peak MB. Plus **analytic recurrent-state footprint per stream** in
KB: ours = Σ over temporal stages (conv_state + ssm_state sizes × dtype bytes); baseline = Σ S5 state
sizes. This is the embedded-streaming memory number.

**5.5 Energy.** NVML (`pynvml`) sampling thread at 10 Hz: 10 s idle baseline (GPU quiescent) → ~60 s
sustained bf16 streaming loop → 10 s cooldown. J/frame = (mean_load_W − mean_idle_W) × elapsed_s /
frames_executed; also report raw W. Fallback if pynvml unavailable: `nvidia-smi
--query-gpu=power.draw --format=csv -lms 250` subprocess parsing. Caveat recorded in JSON meta and the
writeup: desktop-GPU proxy, relative comparison only.

## 6. Orchestrator, launcher, reporter

**stage10_benchmark.py** (CLI): `--models {both,eventssm,baseline}`, `--smoke` (5-step tiny pass to
validate end-to-end wiring in <2 min, run once between Stage-9 sweeps), `--out results/stage10/`.
Sequencing per model: params → FLOPs → latency (headline, network, components, throughput) → VRAM →
energy; `del model; torch.cuda.empty_cache()` between models. JSON schema versioned (`"schema": 1`);
meta block: torch/CUDA/driver versions, GPU name, clocks, date, ckpt paths, clip source, git SHA.

**stage10_run_local.sh**: conda `events_signals`, PYTHONPATH (code + RVT), **idle-GPU guard**
(abort unless `utilization.gpu < 10%` and `memory.used < 1500 MiB` and no other python GPU process),
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, tee to `results/stage10/console_<ts>.log`.

**stage10_report.py**: reads the JSON → (a) `efficiency_table.md` + `.csv` — the Stage-10 doc's table
fully populated (params, GFLOPs, latency p50 full-pipeline + network-only, Hz, VRAM inference/train,
state KB/stream, J/frame, test/AP, mAP/GFLOP, ratios column); (b) **Pareto figure** — x = full-pipeline
p50 latency (bf16, B=1), y = test/AP, bubble size = params, both models + annotations; (c)
**stacked-bars figure** — per-component latency, model × precision. Both figures follow the validated
dataviz palette (ours `#2a78d6`, baseline `#1baf7a`; chrome tokens as in the Stage-9 plots; direct
value labels satisfy the aqua relief rule). Tolerant of partially-filled JSON (smoke runs).

## 7. Testing & acceptance

Unit tests (run before the real benchmark; CPU except where noted):
1. Analytic Mamba-2/S5 FLOP functions vs hand-computed small cases (exact match).
2. Timing utility: monotonic, warmup-excluded, percentile math (tiny `nn.Linear`, GPU, seconds).
3. State-footprint calculator vs shapes from a constructed (CPU) backbone.
4. Reporter: renders table + both figures from a checked-in fixture JSON; missing-field tolerance.
5. `--smoke` GPU pass: both models build, load ckpts strict, 5 full_steps run, JSON validates.

**Acceptance:** all tests green; `--smoke` clean; then ONE full run on an idle GPU (~20–30 min);
`efficiency_table.md` fully populated for both models; figures render; numbers sanity-checked
(params ≈ 19.2M / ~18M as known; latency plausible vs the ~3.3 it/s observed in evals).

## 8. Risks & fallbacks

- **fvcore chokes on vmap/einops paths** (S5 baseline): fall back to analytic-only for those modules,
  flag `"counted_incomplete": true` for that model — the table's FLOPs split makes this visible.
- **Postprocess latency variance** (CPU, detection-density dependent): mitigated by real weights+data,
  300-step percentiles, and reporting p50/p95 rather than mean-only.
- **Energy sampling noise**: 60 s window + idle subtraction; if idle baseline drifts > 10%, rerun flag.
- **Checkpoint prefix mismatch**: `strict=True` fails loudly; the smoke test catches this before the
  real run.

## 9. Out of scope (future work lines in the chapter)

Sequence-length scaling curve; torch.compile/TensorRT/quantised variants; Jetson/embedded measurement;
RVT-LSTM re-measurement (cited from papers); any modification to `external/`.

## 10. Execution schedule

Code + tests can complete **before Stage 9 finishes** (no idle GPU needed; `--smoke` slots between
Stage-9 sweeps). The one full benchmark run happens on a quiet machine after the overnight 10× sweep.
Implementation via subagent-driven development + code review per the project workflow; implementation
plan follows this spec (superpowers writing-plans).
