# Stage-10 benchmark — run checklist (user)

**Status note:** Smoke + GPU-marked tests passed 2026-07-11; first real run completed; ONE re-run required to capture profiler-counted FLOPs (pre-fix JSON has counted=0 with † caveat).

## Preconditions
- NO Stage-9 render or eval running; GPU idle (the launcher enforces this).
- Conda env activated: `events_signals`

## Steps

### 1. Smoke test (<2 min)
```bash
bash code/event_ssm/scripts/stage10_run_local.sh --smoke
```
**PASS condition:** "[stage10] results validate clean" + `results/stage10/bench_results_smoke.json` created

Also run the GPU-marked tests:
```bash
pytest code/event_ssm/tests -m gpu -v
```

### 2. Real benchmark re-run (~20-30 min)
```bash
bash code/event_ssm/scripts/stage10_run_local.sh
```
**Note:** This run captures profiler-counted FLOPs via torch.profiler (or fvcore fallback). Expected output: `results/stage10/bench_results.json`

### 3. Generate report
```bash
python code/event_ssm/scripts/stage10_report.py
```
**Outputs:**
- `results/stage10/efficiency_table.md` and `.csv`
- `results/stage10/stage10_pareto.png` and `.pdf`
- `results/stage10/stage10_latency_breakdown.png` and `.pdf`

## Sanity Gates

All gates must pass before final validation:

1. **Parameter count:**
   - EventSSMDetector: ≈ 19.2M params
   - Baseline: ≈ 18M params

2. **FLOPs capture quality (NEW):**
   - `flops.source == "torch.profiler"` (or `"fvcore"` if fallback triggered)
   - `counted_incomplete == false` for both models
   - Expected counted GFLOPs: ≈ 12.5 (EventSSMDetector) / 10.0 (baseline)

3. **Full-pipeline throughput:**
   - Peak Hz plausible vs ~3.3 it/s observed in batched Gen1 evals

4. **Power draw:**
   - `energy.load_w` between 80 and 300 W for both models
