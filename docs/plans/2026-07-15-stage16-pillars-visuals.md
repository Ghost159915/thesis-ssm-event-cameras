# Stage 16 — PureSSM Pillars + Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the PureSSM investigation's two remaining pillars — **efficiency** (latency/energy/FLOPs/state) and **temporal robustness** (event-rate change) — plus the **CUDA-graph deployment-mode** speed column and the **EventCV qualitative visuals**, so PureSSM has the same three-pillar evidence base (accuracy ✅ done in Stage 15) that EventSSM has.

**Architecture:** Four independent slices. **A (Efficiency)** extends the existing Stage-10 harness to a third model kind — trivial, because `register_resnet_mamba()` already builds the PureSSM backbone and `BenchModel` is model-agnostic. **B (Robustness)** clones the Stage-9 two-regime sweep scripts to PureSSM variants (the sweeps hardcode the model via wrapper choice) and reuses the already-rendered true-rate data. **C (CUDA-graph)** applies the one-line `_scan.py:62` `.clone()` fix (with a byte-parity test) and builds a *real* state-carrying graph-capture measured through the Stage-10 harness as a labeled column. **D (Visuals)** installs `eventcv`, adds a prediction-dump pass, and extends the existing event-video renderer to overlay GT-vs-pred boxes + a large-car contact sheet.

**Tech Stack:** PyTorch 2.11.0+cu128 (sm_120), Hydra/OmegaConf, `mamba-ssm` 2.3.2, `torch.cuda.CUDAGraph`, matplotlib, OpenCV (`cv2`), `eventcv` (numpy-only), pytest.

## Global Constraints

- **Environment:** torch 2.11.0+cu128 (`sm_120`); mamba-ssm 2.3.2.post1 + causal-conv1d 1.6.2.post1. **No new torch-dependent packages; `--no-deps --no-build-isolation` if ever needed.** `eventcv` (numpy-only, abi3 wheel) is the **sole** approved new install — Stage 16 only, downstream-of-science only.
- **Fair-comparison invariant:** EventCV is **downstream-only** — it must NEVER regenerate training/eval tensors (the frozen `stacked_histogram_dt=50_nbins=10` histogram). Visualization/figures only.
- **Parity invariant (Slice C):** the `_scan.py` `.clone()` change is on the **shared** temporal path (train TBPTT + eval streaming, EventSSM *and* PureSSM). Default forward (no graph capture) must stay **byte-identical** — proven by a parity test before merge.
- **Benchmark integrity:** GPU efficiency runs go through `stage10_run_local.sh` only (fail-closed idle-GPU guard: aborts if util ≥10% or mem ≥1500 MiB). The launcher `unset`s `MAMBA_STEP_SCALE`/`S5_STEP_SCALE` (Stage-9 contamination guard) — do not remove.
- **Graph column labeling:** the CUDA-graph speed is reported as a **separate labeled column** (eager 39.5 Hz AND graph-replay), **never silently substituted** for the eager protocol (Stage-11 decision 3, user-approved 2026-07-11).
- **Terminal policy:** ALL GPU runs (benchmarks, sweeps, eval, prediction dumps) are handed to the USER as paste-ready commands — the plan's GPU steps say "USER runs".
- **Visual proof per stage:** every slice ships a figure/table to `results/stage16/` or `code/event_ssm/proofs/out/`.
- **Docs:** non-obvious changes/bugfixes recorded in `docs/Stage16_*.md`, not just commits.
- **Commits:** conventional style, **no assistant names** in messages or trailers.

## Reference numbers (targets/anchors)

| Model | test/AP | AP_L | Efficiency (Stage 10) | True-10× retention |
|---|---|---|---|---|
| S5-RVT baseline | 47.72 | 50.66 | 19.67 ms / 51 Hz · 0.73 J · 4.7 MB state | 62.2% |
| EventSSM | 46.22 | 44.70 | 14.25 ms / 70 Hz · 0.40 J · 144 MB state | 63.0% |
| **PureSSM** | 46.43 | 47.65 | **← Slice A/C measures** | **← Slice B measures** |

PureSSM probe estimates (Stage 11, un-benchmarked): eager 39.5 Hz; fixed-state graph replay 3.773 ms (~91 Hz projected); ~87 Hz with state copies (estimate only — **no real state-carrying datapoint exists yet; Slice C produces the first**).

---

## File Structure

**Slice A — Efficiency**
- Modify: `code/event_ssm/benchmark/bench_models.py` (add `puressm` to `CKPTS`, `EXPERIMENT`; register call)
- Modify: `code/event_ssm/scripts/stage10_benchmark.py` (add `TEST_AP["puressm"]`, `--models` choices, `kinds`)
- Modify: `code/event_ssm/scripts/stage10_report.py` (generalize to 3 models if it hardcodes 2)
- Test: `code/event_ssm/tests/test_bench_models.py` (add `test_build_puressm_cpu_construct_and_load`)
- Test: `code/event_ssm/tests/test_bench_schema.py` (assert `puressm` in `TEST_AP`)

**Slice B — Robustness**
- Create: `code/event_ssm/scripts/stage16_eval_sweep_puressm.sh` (clone of `stage9_eval_sweep.sh`, Regime-1 no-comp)
- Create: `code/event_ssm/scripts/stage16_mamba_scale_sweep_puressm.sh` (clone of `stage9_mamba_scale_sweep.sh`, Regime-1 compensated)
- Create: `code/event_ssm/scripts/stage16_truerate_sweep_puressm.sh` (clone of `stage9_truerate_sweep.sh`, Regime-2 true-rate)
- Modify: `code/event_ssm/scripts/stage9_degradation_plot.py` (+`puressm` series)
- Modify: `code/event_ssm/scripts/stage9_truerate_plot.py` (+`puressm` series)

**Slice C — CUDA-graph deployment mode**
- Modify: `code/event_ssm/temporal/_scan.py:62` (add `.clone()`)
- Create: `code/event_ssm/benchmark/graph_capture.py` (state-carrying CUDAGraph helper)
- Modify: `code/event_ssm/scripts/stage10_benchmark.py` (add graph-replay latency measurement)
- Test: `code/event_ssm/tests/test_scan_clone_parity.py` (byte-parity of default forward)
- Test: `code/event_ssm/tests/test_graph_capture.py` (@gpu: replay output == eager, state carries)

**Slice D — EventCV visuals**
- Create: `code/event_ssm/integration/pred_dump_callback.py` (saves `PRED_PROPH` per frame)
- Create: `code/event_ssm/scripts/stage16_dump_predictions.py` (inference pass → per-rec prediction `.npy`)
- Create: `code/event_ssm/scripts/stage16_select_large_car_recs.py` (rank test recs by large-car area)
- Modify: `code/event_ssm/scripts/stage9_render_event_video.py` (overlay predicted boxes + `track_id` smoothing)
- Create: `code/event_ssm/scripts/stage16_render_gt_vs_pred.sh` (driver → videos + contact sheet)
- Create: `docs/results/Stage16_results.md` (efficiency + robustness tables, graph column, visual links)

---

# SLICE A — Efficiency benchmark (PureSSM)

### Task 1: Add `puressm` model kind to the benchmark

**Files:**
- Modify: `code/event_ssm/benchmark/bench_models.py:32-36` (CKPTS, EXPERIMENT) and `:168-172` (build_model)
- Test: `code/event_ssm/tests/test_bench_models.py`

**Interfaces:**
- Consumes: `register_resnet_mamba()` (already dispatches `PureSSM` — `register.py:36`), `+experiment/gen1=puressm` config, ckpt `results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt`.
- Produces: `build_model("puressm", device, load_ckpt) -> BenchModel` with the same interface as `eventssm` (spatial+temporal param split, 3 mamba2 temporal blocks).

- [ ] **Step 1: Write the failing test** — append to `code/event_ssm/tests/test_bench_models.py`:

```python
def test_build_puressm_cpu_construct_and_load():
    bm = build_model("puressm", device=torch.device("cpu"), load_ckpt=True)
    pb = bm.param_breakdown()
    assert 8 < pb["total"] < 30                       # pure-SSM backbone; spatial ~8.4 M
    assert {"backbone_spatial", "backbone_temporal", "neck", "head", "total"} <= set(pb)
    th = bm.temporal_hparams()
    assert len(th) == 3 and all(t["kind"] == "mamba2" for t in th)   # temporal identical to EventSSM
    assert bm.num_classes == 2 and 0 < bm.conf < 1
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd code/event_ssm && python -m pytest tests/test_bench_models.py::test_build_puressm_cpu_construct_and_load -v`
Expected: FAIL — `AssertionError` inside `build_model` (`assert kind in CKPTS`) or a KeyError on `EXPERIMENT["puressm"]`.

- [ ] **Step 3: Add the registry + experiment entries** — `bench_models.py`, edit the `CKPTS` and `EXPERIMENT` dicts:

```python
CKPTS = {
    "eventssm": RVT / "RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt",
    "baseline": REPO / "checkpoints/gen1_base.ckpt",
    "puressm": REPO / "results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt",
}
EXPERIMENT = {"eventssm": "+experiment/gen1=resnet_mamba", "baseline": "+experiment/gen1=base.yaml",
              "puressm": "+experiment/gen1=puressm"}
```

- [ ] **Step 4: Register the PureSSM backbone in `build_model`** — change the eventssm-only guard (currently `if kind == "eventssm":`) to cover puressm (same `register_resnet_mamba()` handles both):

```python
def build_model(kind: str, device: torch.device, load_ckpt: bool = True) -> BenchModel:
    assert kind in CKPTS, kind
    if kind in ("eventssm", "puressm"):
        from event_ssm.integration.register import register_resnet_mamba
        register_resnet_mamba()
    cfg = compose_cfg(kind)
    ...
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `cd code/event_ssm && python -m pytest tests/test_bench_models.py -v`
Expected: PASS (all bench_models tests, incl. the new one; the `@gpu` one skips on CPU).

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/benchmark/bench_models.py code/event_ssm/tests/test_bench_models.py
git commit -m "feat(stage16): add puressm kind to the Stage-10 benchmark model builder"
```

### Task 2: Wire `puressm` into the orchestrator + report, then run

**Files:**
- Modify: `code/event_ssm/scripts/stage10_benchmark.py:21` (TEST_AP), `:258` (choices), `:275` (kinds)
- Modify: `code/event_ssm/scripts/stage10_report.py` (only if it hardcodes the 2-model set)
- Test: `code/event_ssm/tests/test_bench_schema.py`

- [ ] **Step 1: Write the failing test** — append to `code/event_ssm/tests/test_bench_schema.py`:

```python
def test_test_ap_has_puressm():
    from event_ssm.scripts.stage10_benchmark import TEST_AP
    assert TEST_AP.get("puressm") == 0.4643
```

(If `scripts` is not importable as a package in the test env, instead assert against the JSON: load `code/event_ssm/tests/fixtures/stage10_fixture.json` is unaffected; use `importlib.util` to load the module by path. Prefer the direct import; fall back only if collection errors.)

- [ ] **Step 2: Run it, verify it fails** — Run: `cd code/event_ssm && python -m pytest tests/test_bench_schema.py::test_test_ap_has_puressm -v` → FAIL (`None != 0.4643`).

- [ ] **Step 3: Add puressm to TEST_AP** — `stage10_benchmark.py:21`:

```python
TEST_AP = {"eventssm": 0.462, "baseline": 0.477, "puressm": 0.4643}   # Stage-8/15 one-shot test results
```

- [ ] **Step 4: Add `puressm` + `all` to the CLI** — `stage10_benchmark.py`, the arg and the `kinds` line:

```python
ap.add_argument("--models", default="both",
                choices=["both", "all", "eventssm", "baseline", "puressm"])
...
if args.models == "both":
    kinds = ["eventssm", "baseline"]
elif args.models == "all":
    kinds = ["eventssm", "baseline", "puressm"]
else:
    kinds = [args.models]
```

- [ ] **Step 5: Generalize the report if needed** — inspect the reporter:

Run: `grep -nE "eventssm|baseline|puressm|models\[|for .* in .*models" code/event_ssm/scripts/stage10_report.py`
If it iterates `results["models"]` generically (no hardcoded 2-name list), no change. If it hardcodes `["eventssm","baseline"]` for the comparison table, add `"puressm"` to that list and give it a row. Re-run `python -m pytest code/event_ssm/tests/test_bench_report.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/scripts/stage10_benchmark.py code/event_ssm/scripts/stage10_report.py code/event_ssm/tests/test_bench_schema.py
git commit -m "feat(stage16): benchmark --models all (eventssm+baseline+puressm) + report row"
```

- [ ] **Step 7: USER runs the smoke wiring check** (needs idle GPU, <2 min):

```bash
bash code/event_ssm/scripts/stage10_run_local.sh --smoke --models all
```
Expected: `bench_results_smoke.json` with a `models.puressm` block containing `params_m/flops/latency/vram/energy`; `[stage10] results validate clean`.

- [ ] **Step 8: USER runs the full benchmark** (idle GPU, ~10-15 min):

```bash
bash code/event_ssm/scripts/stage10_run_local.sh --models all
python code/event_ssm/scripts/stage10_report.py
```
Expected: `results/stage10/bench_results.json` with all three models; report prints a 3-model latency/energy/FLOPs/state table. Record PureSSM's p50 ms / Hz / J-per-frame / state-KB for `docs/results/Stage16_results.md`.

---

# SLICE B — Two-regime robustness (PureSSM)

The Stage-9 sweeps hardcode the model via the wrapper they call. A PureSSM wrapper (`stage14_puressm_test_eval_local.sh`) and the compensation hook (`MAMBA_STEP_SCALE`, reused unchanged) already exist; the true-rate data is model-agnostic and already rendered (`data/gen1_stage9/preproc_tr`). So each slice-B task **clones** a Stage-9 sweep, swaps the wrapper + `OURS_CKPT` + output labels — no new eval plumbing.

### Task 3: PureSSM Regime-1 sweeps (no-comp + compensated)

**Files:**
- Create: `code/event_ssm/scripts/stage16_eval_sweep_puressm.sh` (from `stage9_eval_sweep.sh`)
- Create: `code/event_ssm/scripts/stage16_mamba_scale_sweep_puressm.sh` (from `stage9_mamba_scale_sweep.sh`)

**Interfaces:**
- Consumes: `stage14_puressm_test_eval_local.sh` (PureSSM wrapper), ckpt `results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt`, `MAMBA_STEP_SCALE` env.
- Produces: logs `results/stage9/sweep_puressm/puressm_dt{V}.log` and `results/stage9/sweep_ss_puressm/puressm_dt{V}_ss{S}.log`.

- [ ] **Step 1: Clone the no-comp sweep** —

```bash
cp code/event_ssm/scripts/stage9_eval_sweep.sh code/event_ssm/scripts/stage16_eval_sweep_puressm.sh
```

- [ ] **Step 2: Edit `stage16_eval_sweep_puressm.sh`** — three changes (mirror the verbatim lines the research found):
  - `OURS_CKPT=` (was line ~20) → `OURS_CKPT="$REPO/results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt"`
  - `OUT=` (was line ~22) → `OUT="$REPO/results/stage9/sweep_puressm"`
  - the per-rate eval line (was line ~52) `bash "$SCRIPTS/stage7_test_eval_local.sh" "$OURS_CKPT" ...` → `bash "$SCRIPTS/stage14_puressm_test_eval_local.sh" "$OURS_CKPT" ...`
  - log name `eventssm_dt${V}.log` (was line ~47) → `puressm_dt${V}.log`
  - Delete the baseline branch (lines ~57-62) — baseline stays from the original Stage-9 run; this script is PureSSM-only.

- [ ] **Step 3: Clone + edit the compensated sweep** — `cp stage9_mamba_scale_sweep.sh stage16_mamba_scale_sweep_puressm.sh`; same three swaps (ckpt line ~23, wrapper line ~61, log `eventssm_dt${V}_ss${SS}.log` line ~52 → `puressm_...`, `OUT` line ~24 → `results/stage9/sweep_ss_puressm`). Keep `MAMBA_STEP_SCALE="$SS"` — the Mamba temporal path is identical, so compensation works unchanged.

- [ ] **Step 4: Dry-verify (no GPU) both scripts compose** —

```bash
bash -n code/event_ssm/scripts/stage16_eval_sweep_puressm.sh && bash -n code/event_ssm/scripts/stage16_mamba_scale_sweep_puressm.sh && echo "syntax OK"
```
Expected: `syntax OK`. (Full run is GPU — Step 6.)

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/scripts/stage16_eval_sweep_puressm.sh code/event_ssm/scripts/stage16_mamba_scale_sweep_puressm.sh
git commit -m "feat(stage16): PureSSM Regime-1 robustness sweeps (no-comp + Mamba-compensated)"
```

- [ ] **Step 6: USER runs the Regime-1 sweeps** (GPU; each rate is a full test eval, ~several hours total — run when the GPU is free):

```bash
bash code/event_ssm/scripts/stage16_eval_sweep_puressm.sh            # no-comp: rates 200/100/50/25/12
bash code/event_ssm/scripts/stage16_mamba_scale_sweep_puressm.sh     # Mamba Δt-compensated
```
Expected: `results/stage9/sweep_puressm/puressm_dt*.log` each ending in a `test/AP …` line. Record the 5 mAP points (compare to EventSSM 40.9/45.5/46.2/43.5/35.4).

### Task 4: PureSSM Regime-2 true-rate sweep

**Files:**
- Create: `code/event_ssm/scripts/stage16_truerate_sweep_puressm.sh` (from `stage9_truerate_sweep.sh`)

- [ ] **Step 1: Clone** — `cp code/event_ssm/scripts/stage9_truerate_sweep.sh code/event_ssm/scripts/stage16_truerate_sweep_puressm.sh`
- [ ] **Step 2: Edit** — `OURS_CKPT` (line ~23) → PureSSM ckpt; in `run_eval()` the `WRAP="$SCRIPTS/stage7_test_eval_local.sh"` for the `eventssm` key (line ~58) → `stage14_puressm_test_eval_local.sh`; the log key `eventssm` (lines ~78-79) → `puressm`; `OUT` (line ~25) → `results/stage9/sweep_tr_puressm`. **Do NOT re-render** — the script reads the existing model-agnostic `data/gen1_stage9/preproc_tr`.
- [ ] **Step 3: Dry-verify** — `bash -n code/event_ssm/scripts/stage16_truerate_sweep_puressm.sh && echo OK`
- [ ] **Step 4: Commit** — `git add ...stage16_truerate_sweep_puressm.sh && git commit -m "feat(stage16): PureSSM Regime-2 true-rate robustness sweep"`
- [ ] **Step 5: USER runs it** (GPU, hours):

```bash
bash code/event_ssm/scripts/stage16_truerate_sweep_puressm.sh        # 2x (25ms) + 10x (5ms), nc + compensated
```
Expected: `results/stage9/sweep_tr_puressm/puressm_dt*_{nc,ss*}.log`. Compute PureSSM's **true-10× retention** (mAP@10× / mAP@1×) and compare to EventSSM 63.0% / S5-RVT 62.2% / ConvLSTM 17.7%.

### Task 5: Add the PureSSM series to both robustness plots

**Files:**
- Modify: `code/event_ssm/scripts/stage9_degradation_plot.py`, `stage9_truerate_plot.py`

- [ ] **Step 1: Inspect the hardcoded model dicts** — `grep -nE "MODELS|HUE|MARKER|_dt|glob|results/stage9" code/event_ssm/scripts/stage9_degradation_plot.py code/event_ssm/scripts/stage9_truerate_plot.py`
- [ ] **Step 2: Add a `"puressm"` entry** to each script's `MODELS` (log-dir + name pattern matching Task 3/4 outputs, e.g. `sweep_puressm/puressm_dt{V}.log`), `HUE` (a distinct colour, e.g. `"#2ca02c"`), and `MARKER` dicts. Keep `eventssm`/`baseline` untouched so the existing curves still render.
- [ ] **Step 3: Regenerate the figures** (CPU-only — reads logs, no GPU):

```bash
python code/event_ssm/scripts/stage9_degradation_plot.py
python code/event_ssm/scripts/stage9_truerate_plot.py
```
Expected: `results/stage9/stage9_{degradation,truerate}_curve.{png,pdf}` now show three curves. Open the PNGs to confirm the PureSSM series renders sensibly.

- [ ] **Step 4: Commit** — `git add code/event_ssm/scripts/stage9_degradation_plot.py code/event_ssm/scripts/stage9_truerate_plot.py results/stage9/*.png results/stage9/*.pdf && git commit -m "feat(stage16): add PureSSM series to the two robustness degradation plots"` *(check results/ is not gitignored for these specific figures — the Stage-9 curves were committed before; if `*.pdf`/that path is ignored, commit only the tracked figure files.)*

---

# SLICE C — CUDA-graph deployment mode (the graph-replay speed column)

**Risk note:** this is the only genuinely-new/high-risk slice. Structure it so each task is independently valuable: the `.clone()` parity fix stands alone; the state-carrying capture has a hard correctness gate; and if real state-carry replay proves infeasible in the available time, the fallback is to report the fixed-state replay bound *with an explicit caveat* (Stage-11 already has this datapoint) — the graph number is a labeled column and is never substituted for the eager protocol.

### Task 6: The `_scan.py:62` `.clone()` fix + byte-parity test

**Files:**
- Modify: `code/event_ssm/temporal/_scan.py:62`
- Test: `code/event_ssm/tests/test_scan_clone_parity.py`

**Interfaces:**
- Consumes: `mamba2_scan_time(layer, x, state=None, step_scale=1.0) -> (out, (new_conv, last_state))` (`_scan.py`).
- Produces: identical numerical output; `new_conv` now owns its storage (survives CUDAGraph replay).

- [ ] **Step 1: Write the failing parity test** — `code/event_ssm/tests/test_scan_clone_parity.py`:

```python
"""The _scan.py:62 .clone() must not change any default-forward numbers (shared train/eval path)."""
import pytest, torch
GPU = pytest.mark.gpu

@GPU
def test_scan_output_and_state_unchanged_by_clone():
    # A two-window streaming pass must be identical before/after the .clone(); we assert the
    # invariant the fix must preserve: new_conv equals the last (d_conv-1) frames of the extended
    # input, and carrying it into a second window reproduces a single full-length scan.
    from event_ssm.temporal._scan import mamba2_scan_time
    from event_ssm.temporal.mamba_temporal import MambaTemporalBlock  # a real layer for hparams
    torch.manual_seed(0)
    dev = torch.device("cuda")
    block = MambaTemporalBlock(d_model=128).to(dev).eval()
    layer = block.layers[0]
    x = torch.randn(4, 6, 128, device=dev)               # (N, L, d_model)
    with torch.no_grad():
        full, _ = mamba2_scan_time(layer, x, None)
        a, sa = mamba2_scan_time(layer, x[:, :3], None)
        b, sb = mamba2_scan_time(layer, x[:, 3:], sa)
    torch.testing.assert_close(torch.cat([a, b], dim=1), full, rtol=2e-2, atol=2e-2)
    assert sb[0].is_contiguous()   # new_conv owns storage after .clone()
```

(Constructor signature for `MambaTemporalBlock`/`layers[0]` — confirm against `code/event_ssm/temporal/mamba_temporal.py` when implementing; adjust `d_model`/attribute names to the real API. The *behavioural* assertion — two-window == full-scan, and `new_conv` contiguous — is the fixed contract.)

- [ ] **Step 2: USER runs it on GPU, verify it fails on the contiguity assertion** (the pre-fix view is not guaranteed contiguous / aliases graph memory):

Run: `cd code/event_ssm && python -m pytest tests/test_scan_clone_parity.py -m gpu -v`
Expected: the two-window==full assertion PASSES (behaviour already correct), the `is_contiguous()` assertion may already pass on eager — the real proof is Task 7's replay test. Keep this test as the parity guard regardless.

- [ ] **Step 3: Apply the one-line fix** — `_scan.py:62`:

```python
    new_conv = xBC_ext[:, -(d_conv - 1):].detach().clone()
```

- [ ] **Step 4: USER re-runs the full temporal test suite on GPU** to prove byte-parity of the default path:

```bash
cd code/event_ssm && python -m pytest tests/test_scan_clone_parity.py tests/test_resnet_mamba.py tests/test_bimamba_spatial.py -m gpu -v
```
Expected: all PASS, including the existing `test_state_carry_changes_output` and `test_backbone_state_carry_streaming` (the `.clone()` changes storage ownership only, not values).

- [ ] **Step 5: Commit** — `git add code/event_ssm/temporal/_scan.py code/event_ssm/tests/test_scan_clone_parity.py && git commit -m "fix(stage16): clone carried conv-state in _scan (unblocks cudagraph replay; byte-parity preserved)"`

### Task 7: State-carrying CUDAGraph capture helper

**Files:**
- Create: `code/event_ssm/benchmark/graph_capture.py`
- Test: `code/event_ssm/tests/test_graph_capture.py`

**Interfaces:**
- Consumes: a `BenchModel` (`network_step(frame, state) -> (preds, new_state)`), the state-layout helpers (`resnet_mamba.py` `_state_to_bmajor`/`_state_from_bmajor`), state tensors `conv_state:(N,d_conv-1,conv_dim)` + `ssm_state:(N,nheads,headdim,d_state)` per temporal layer.
- Produces: `capture_state_carrying(model, frame_shape, device) -> replay_fn` where each `replay_fn(new_frame_tensor)` copies the frame into the static input, replays the graph, and **copies the updated state out/in** so successive calls stream correctly.

- [ ] **Step 1: Write the failing correctness test** — `code/event_ssm/tests/test_graph_capture.py`:

```python
"""Graph replay must (a) match eager output for the same input+state and (b) genuinely carry state."""
import pytest, torch
GPU = pytest.mark.gpu

@GPU
def test_state_carrying_replay_matches_eager():
    from event_ssm.benchmark.bench_models import build_model
    from event_ssm.benchmark.graph_capture import capture_state_carrying
    dev = torch.device("cuda")
    bm = build_model("puressm", device=dev, load_ckpt=True)
    frames = [torch.randn(1, 20, 256, 320, device=dev) for _ in range(4)]

    # eager reference: 4-step streaming
    st = None; eager = []
    for f in frames:
        p, st = bm.network_step(f, st); eager.append(p.float().clone())

    # graph replay: same 4 frames through the state-carrying capture
    replay = capture_state_carrying(bm, frames[0].shape, dev)
    graphed = [replay(f).float().clone() for f in frames]

    # step 0 output must match (fresh state both sides); and outputs must DIFFER across steps
    torch.testing.assert_close(graphed[0], eager[0], rtol=5e-2, atol=5e-2)
    assert not torch.allclose(graphed[0], graphed[1])   # state genuinely carried, not frozen
```

- [ ] **Step 2: USER runs it, verify it fails** — `cd code/event_ssm && python -m pytest tests/test_graph_capture.py -m gpu -v` → FAIL (`ModuleNotFoundError: graph_capture`).

- [ ] **Step 3: Implement `graph_capture.py`** — mirror `stage11_probe.py:_try_manual_cudagraph` but add static state buffers copied out/in each replay. Skeleton (fill the state-buffer wiring against the real `network_step` state structure):

```python
"""State-carrying CUDAGraph capture for streaming inference (Stage 16, spec: real state-carry
replay, not the Stage-11 fixed-state lower bound). Captures one network_step with static input
AND static recurrent-state buffers; each replay copies the new frame in, replays, and swaps the
updated state buffers so successive calls stream correctly."""
import torch

def capture_state_carrying(model, frame_shape, device):
    static_in = torch.zeros(frame_shape, device=device)
    # warm the state on a side stream (3 steps), then freeze the state tensors as static buffers
    holder = {"s": None}
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _, holder["s"] = model.network_step(static_in, holder["s"])
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()

    static_state = holder["s"]                      # static input state buffers (captured graph reads these)
    g = torch.cuda.CUDAGraph()
    out_holder = {"p": None, "s": None}
    with torch.cuda.graph(g):
        out_holder["p"], out_holder["s"] = model.network_step(static_in, static_state)

    def replay(frame):
        static_in.copy_(frame)
        g.replay()
        # carry: copy the freshly-produced state back into the static input buffers for next step
        _copy_state_(static_state, out_holder["s"])
        return out_holder["p"]
    return replay

def _copy_state_(dst, src):
    """Recursively in-place copy nested (list/tuple of) tensors src -> dst (same structure)."""
    if torch.is_tensor(dst):
        dst.copy_(src); return
    for d, s in zip(dst, src):
        _copy_state_(d, s)
```

**Implementation note for the executor:** the correctness of `_copy_state_` depends on `static_state` and `out_holder["s"]` sharing structure — verify with `network_step`'s actual return (per-layer `[(conv_state, ssm_state), ...]` for temporal stages 2-4, placeholder `zeros(B,1)` for non-temporal). If the captured graph does not update `static_state` in place (it produces a *new* `out_holder["s"]`), the copy-back closes the loop. If structural mismatches appear (e.g. non-temporal placeholders), skip non-tensor leaves. This is the crux of the slice — the test in Step 1 is the gate.

- [ ] **Step 4: USER runs the test until it passes** — `cd code/event_ssm && python -m pytest tests/test_graph_capture.py -m gpu -v` → PASS (replay matches eager at step 0; outputs differ across steps ⇒ state carried).
- [ ] **Step 5: Commit** — `git add code/event_ssm/benchmark/graph_capture.py code/event_ssm/tests/test_graph_capture.py && git commit -m "feat(stage16): state-carrying CUDAGraph capture for streaming replay"`

### Task 8: Add the graph-replay latency column to the benchmark

**Files:**
- Modify: `code/event_ssm/scripts/stage10_benchmark.py` (`measure_model`, add a `latency.graph` block)

- [ ] **Step 1: Add an opt-in graph-replay measurement** in `measure_model` (guarded by a `--graph` flag so the eager numbers are never touched): after the eager `latency` loop, if graph mode is requested and `kind in ("eventssm","puressm")`, build `capture_state_carrying`, time `replay(frames[i%n])` with `bm.time_fn`, and store under `out["latency"]["graph"] = {"p50_ms": ..., "hz": 1000/p50, "state_semantics": "carried", "protocol": "cuda-graph-replay"}`. Keep it clearly labeled and separate from `bf16`/`fp32`.
- [ ] **Step 2: Add `--graph` to the CLI** (`action="store_true"`), thread it into `measure_model(..., graph=args.graph)`.
- [ ] **Step 3: Update `validate_results`** to accept the optional `graph` sub-key (do not require it).
- [ ] **Step 4: Commit** — `git commit -am "feat(stage16): optional labeled cuda-graph-replay latency column in the benchmark"`
- [ ] **Step 5: USER runs the graph benchmark** (idle GPU):

```bash
bash code/event_ssm/scripts/stage10_run_local.sh --models all --graph
python code/event_ssm/scripts/stage10_report.py
```
Expected: JSON has `latency.graph` for eventssm+puressm; report shows the labeled eager-vs-graph column. **This is the first real state-carrying graph-replay Hz** — record it against the ~87 Hz projection.

---

# SLICE D — EventCV GT-vs-pred visuals (U6)

### Task 9: Install EventCV + smoke it

- [ ] **Step 1: Install (env-safe, numpy-only)** — `conda run -n events_signals pip install eventcv`
- [ ] **Step 2: Smoke** — load one Gen1 `.dat` and compare the event count to our converter:

```bash
conda run -n events_signals python -c "import eventcv as ecv; d = ecv.load('data/gen1_stage9/_vidprobe/17-04-04_11-00-13_cut_15_122500000_182500000_td.dat'); print(type(d))"
```
Expected: prints a numpy-backed object; no torch/cuda import errors. Confirm `python -c "import torch; print(torch.__version__)"` still prints `2.11.0+cu128` (install did not disturb the stack).
- [ ] **Step 3: Commit a note** (no code yet) — record the install in `docs/results/Stage16_results.md` (created in Task 12).

### Task 10: Per-frame prediction dump (both models)

**Files:**
- Create: `code/event_ssm/integration/pred_dump_callback.py`
- Create: `code/event_ssm/scripts/stage16_dump_predictions.py`

**Interfaces:**
- Consumes: RVT's `ObjDetOutput.PRED_PROPH` / `LABELS_PROPH` (per-timestep prophesee arrays, dtype `x/y/w/h/class_id/class_confidence`), the register path (`stage7_eval.py` runs `validation.py` unmodified via runpy, so the dump attaches via a callback, not by editing validation.py).
- Produces: `results/stage16/preds/{model}/{rec}.npy` — per-recording arrays of predicted boxes with frame index.

- [ ] **Step 1: Write a callback that saves `PRED_PROPH`** — `pred_dump_callback.py` defines a Lightning callback whose `on_test_batch_end` appends `outputs[ObjDetOutput.PRED_PROPH]` (+ frame/timestamp index) to a per-recording buffer and `np.save`s it under `results/stage16/preds/{model}/`. Model tag passed via env `STAGE16_PRED_TAG`.
- [ ] **Step 2: Write `stage16_dump_predictions.py`** — mirror `stage7_eval.py` (call `register()`, then run the RVT val path) but register the dump callback, restricted to a recording allowlist (from Task 11) so it only dumps the large-car recs, not all 470. Reads model via env (`resnet_mamba` or `puressm`) + ckpt.
- [ ] **Step 3: Unit-test the callback's buffer→file logic on CPU** with a fake `PRED_PROPH` array (`code/event_ssm/tests/test_pred_dump.py`): feed two fake batches, assert the saved `.npy` has the concatenated boxes with correct frame indices. Run `pytest tests/test_pred_dump.py -v` → PASS.
- [ ] **Step 4: Commit** — `git add code/event_ssm/integration/pred_dump_callback.py code/event_ssm/scripts/stage16_dump_predictions.py code/event_ssm/tests/test_pred_dump.py && git commit -m "feat(stage16): per-frame prediction dump callback + driver for GT-vs-pred viz"`
- [ ] **Step 5: USER runs the dump for both models** (GPU, only the allowlisted recs — minutes):

```bash
STAGE16_PRED_TAG=eventssm bash -c 'MODEL=resnet_mamba CKPT=external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt python code/event_ssm/scripts/stage16_dump_predictions.py'
STAGE16_PRED_TAG=puressm  bash -c 'MODEL=puressm      CKPT=results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt python code/event_ssm/scripts/stage16_dump_predictions.py'
```
Expected: `results/stage16/preds/{eventssm,puressm}/*.npy`. (Exact env/arg wiring to match `stage16_dump_predictions.py` as implemented in Step 2.)

### Task 11: Select ≥10 large-car test recordings

**Files:**
- Create: `code/event_ssm/scripts/stage16_select_large_car_recs.py`

- [ ] **Step 1: Write the selector** — reads the GT `*_bbox.npy` under `data/gen1_stage9/raw/detection_dataset_duration_60s_ratio_1.0/test/`, computes per-recording the count of car boxes (`class_id==0`) with area (`w*h`) in the top decile of the full-frame (304×240) area, ranks recordings, writes the top ≥10 recording names to `results/stage16/large_car_recs.txt`.
- [ ] **Step 2: Unit-test on a synthetic bbox array** (`tests/test_large_car_select.py`): a fake `.npy` with two big cars vs one small → asserts ranking. Run pytest → PASS.
- [ ] **Step 3: Run it** (CPU-only): `python code/event_ssm/scripts/stage16_select_large_car_recs.py` → `results/stage16/large_car_recs.txt` with ≥10 names. Feed this list to Task 10's allowlist and Task 12's renderer.
- [ ] **Step 4: Commit** — `git add code/event_ssm/scripts/stage16_select_large_car_recs.py code/event_ssm/tests/test_large_car_select.py results/stage16/large_car_recs.txt && git commit -m "feat(stage16): rank test recordings by large-car area for qualitative viz"`

### Task 12: GT-vs-pred overlay renderer + contact sheet + results doc

**Files:**
- Modify: `code/event_ssm/scripts/stage9_render_event_video.py` (add `--pred` overlay + `track_id` hold-smoothing)
- Create: `code/event_ssm/scripts/stage16_render_gt_vs_pred.sh`
- Create: `docs/results/Stage16_results.md`

- [ ] **Step 1: Extend the renderer** — add `--pred PATH` (loads a Task-10 `.npy` prediction array; overlays predicted boxes in a distinct style, e.g. dashed / different colour, with confidence text) alongside the existing `--boxes` GT overlay; add `--smooth` (hold each GT/pred box by `track_id` between the 4 Hz label updates so boxes track objects without flicker — the first deferred TODO). Keep existing GT-only behaviour when `--pred` absent.
- [ ] **Step 2: Test the box-hold logic on CPU** (`tests/test_render_smoothing.py`): given sparse per-`track_id` boxes, assert the hold fills intermediate frames up to the 250 ms window. Run pytest → PASS.
- [ ] **Step 3: Write the driver `stage16_render_gt_vs_pred.sh`** — for each recording in `results/stage16/large_car_recs.txt`, call the renderer with the raw `.dat.h5` + GT `_bbox.npy` + both models' pred `.npy`, `--montage`, output to `results/stage16/`. Produces a GT-vs-pred `.mp4` per rec + an 8-frame contact-sheet `.png` per rec.
- [ ] **Step 4: USER runs the renderer** (CPU/GPU-light — cv2 only, no model): `bash code/event_ssm/scripts/stage16_render_gt_vs_pred.sh` → `results/stage16/*_gt_vs_pred.mp4` + `*_montage.png` for the ≥10 large-car recs.
- [ ] **Step 5: Assemble a large-car contact sheet** — stitch the per-rec montages into one `results/stage16/large_car_contact_sheet.png` (extend the existing `--montage` hstack/vstack mechanism, or a small stitch step in the driver).
- [ ] **Step 6: Write `docs/results/Stage16_results.md`** — the efficiency table (Slice A + the graph column from Slice C), the two-regime robustness table + curves (Slice B), and links to the EventCV videos/contact sheet, each with a one-paragraph interpretation vs EventSSM/baseline. Mirror the structure of `docs/results/Stage15_results_comparison.md`.
- [ ] **Step 7: Commit** — `git add code/event_ssm/scripts/stage9_render_event_video.py code/event_ssm/scripts/stage16_render_gt_vs_pred.sh docs/results/Stage16_results.md code/event_ssm/tests/test_render_smoothing.py results/stage16/ && git commit -m "feat(stage16): GT-vs-pred large-car visuals + Stage-16 results doc"`

---

## Self-Review

**1. Spec coverage** (roadmap Stage-16 deliverables → tasks):
- "Stage-9 two-regime re-eval row" → Slice B (Tasks 3-5). ✅
- "Stage-10 bench re-run row (launcher unsets MAMBA_STEP_SCALE)" → Slice A (Tasks 1-2); the `unset` is already in `stage10_run_local.sh`. ✅
- "state-carrying CUDA-graph deployment mode … implement state copy-in/out capture, measure via Stage-10 harness" → Slice C (Tasks 6-8). ✅
- "optional torch.compile unlock = one-line `.clone()` at `temporal/_scan.py:62`, parity-test before adopting" → Task 6. ✅
- "EventCV GT-vs-pred videos + large-car contact sheet" → Slice D (Tasks 9-12). ✅
- "upgrade-lever decision: cross-scan on stages 3-4 iff headroom AND accuracy warrants" → **deferred**; this is a conditional research fork, out of scope for the pillar/visual completion. Flag for the user, do not silently drop.

**2. Placeholder scan:** the two genuinely-discovery points (Task 6 test constructor signature, Task 7 `_copy_state_` structural match) carry an explicit "confirm against the real API / this is the crux" note plus a behavioural test that is the acceptance gate — not lazy TODOs. All GPU-run steps are marked "USER runs" per the terminal policy. No `results/stage16/` path is assumed to exist — Task 8/10/11/12 create it.

**3. Type consistency:** `build_model(kind,...)` / `BenchModel` interface reused verbatim from bench_models.py; `capture_state_carrying(model, frame_shape, device) -> replay_fn` consistent between Task 7 (def) and Task 8 (use); `network_step(frame, state) -> (preds, new_state)` consistent throughout.

**Ordering / dependency notes:** Slice A is independent and highest-value (do first). Slice B is independent (heavy GPU, low code risk). Slice C Task 8 depends on Task 7 depends on Task 6; the graph column augments Slice A's table. Slice D Task 12 depends on Tasks 10+11. Slices can run in the order A → B → C → D, or A/B in parallel with C, with D last (needs nothing from A-C except the pattern).

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-07-15-stage16-pillars-visuals.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for the many small mechanical tasks here (Slices A, B, D) with tight review on the risky Slice C.

**2. Inline Execution** — I execute tasks in this session with checkpoints for your review.

**Which approach?** (Also: confirm whether to keep the "cross-scan upgrade-lever" fork in scope or defer it, and whether to start with Slice A (fastest to a real efficiency number) or a different slice.)
