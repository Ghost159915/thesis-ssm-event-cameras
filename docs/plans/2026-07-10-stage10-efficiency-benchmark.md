# Stage 10 Efficiency Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage-10 efficiency-benchmark harness (spec: `docs/specs/2026-07-10-stage10-efficiency-design.md`): a one-shot GPU orchestrator that measures params/FLOPs/latency/VRAM/energy for EventSSMDetector and the S5-RVT baseline into one JSON, plus a CPU reporter that renders the efficiency table and two figures.

**Architecture:** Measure/report split. `code/event_ssm/benchmark/` holds construction (`bench_models.py`) and measurement primitives (`bench_metrics.py`); `code/event_ssm/scripts/` holds the GPU orchestrator, CPU reporter, and guarded launcher. Data flows one way: models+clip → measurements → `results/stage10/bench_results.json` → tables/figures.

**Tech Stack:** PyTorch 2.11.0+cu128 (conda `events_signals`), Hydra compose API against the vendored RVT config tree, fvcore (FLOPs counting), h5py+hdf5plugin (clip extraction), matplotlib Agg (reporting), pytest (existing `code/event_ssm/tests/` conventions).

## Global Constraints

- Conda env `events_signals` for every python/pytest command: `source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals`.
- **NEVER** plain `pip install` a torch-dependent package — `--no-deps` only (a plain install once broke the cu128 stack). After any install: `python -c "import torch; assert torch.__version__.startswith('2.11.0')"`.
- **No modifications to `external/ssms_event_cameras/`** — construct via its public APIs only.
- **The GPU is occupied by Stage-9 sweeps during implementation.** Default tests must pass on CPU. GPU-dependent tests are marked `@pytest.mark.gpu` and are NOT run by the implementer; the `--smoke` GPU pass is run later by the USER (terminal policy: hand GPU runs to the user).
- Tests live in `code/event_ssm/tests/`, run as `pytest code/event_ssm/tests/<file> -v` from the repo root (conftest injects `code/` and the RVT repo into `sys.path`).
- Commits: conventional style (`feat(stage10): ...`), **no Claude attribution, no Co-Authored-By trailers**.
- `REPO = /home/ghost/Desktop/thesis-ssm-event-cameras` (absolute paths in code via `pathlib`).
- Figures follow the validated dataviz palette: ours `#2a78d6`, baseline `#1baf7a`; ink tokens `INK=#0b0b0b, INK2=#52514e, MUTED=#898781, GRID=#e1e0d9`; matplotlib `Agg`; `facecolor="white"`.
- Checkpoints (real weights, load `strict` after prefix-strip):
  - eventssm: `external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt`
  - baseline: `checkpoints/gen1_base.ckpt`
- Stage-8 test APs for mAP/GFLOP: eventssm **0.462**, baseline **0.477**.

## File Structure

```
code/event_ssm/benchmark/__init__.py          (empty marker)
code/event_ssm/benchmark/bench_metrics.py     Task 1+2: timing, VRAM, power, state-bytes, analytic FLOPs, fvcore wrapper
code/event_ssm/benchmark/bench_models.py      Task 3: config compose, ckpt load, BenchModel wrapper
code/event_ssm/benchmark/bench_clip.py        Task 4: real-clip extraction + cache
code/event_ssm/scripts/stage10_benchmark.py   Task 5: GPU orchestrator → bench_results.json
code/event_ssm/scripts/stage10_run_local.sh   Task 6: guarded launcher
code/event_ssm/scripts/stage10_report.py      Task 7: JSON → table + 2 figures
code/event_ssm/tests/test_bench_metrics.py    Tasks 1+2 tests (CPU)
code/event_ssm/tests/test_bench_models.py     Task 3 tests (CPU; GPU parts marked)
code/event_ssm/tests/test_bench_clip.py       Task 4 tests (CPU, synthetic h5)
code/event_ssm/tests/test_bench_schema.py     Task 5 tests (CPU)
code/event_ssm/tests/test_bench_report.py     Task 7 tests (CPU, fixture JSON)
code/event_ssm/tests/fixtures/stage10_fixture.json  Task 7 fixture
```

---

### Task 1: Measurement primitives — timing, VRAM, power, state bytes

**Files:**
- Create: `code/event_ssm/benchmark/__init__.py` (empty)
- Create: `code/event_ssm/benchmark/bench_metrics.py`
- Test: `code/event_ssm/tests/test_bench_metrics.py`

**Interfaces:**
- Consumes: nothing (foundation module).
- Produces (used by Tasks 2, 3, 5):
  - `time_fn(fn, *, warmup: int = 50, iters: int = 300, device=None) -> dict` with keys `mean_ms, std_ms, p50_ms, p95_ms, hz, iters`
  - `measure_peak_vram(fn, *, device, n_calls: int = 20) -> float` (MB; returns `-1.0` on CPU)
  - `PowerSampler(interval_s: float = 0.25, _read=None)` context manager; attrs `.samples: list[float]`, `.mean_w: float`
  - `read_gpu_power_w() -> float` (pynvml if importable, else `nvidia-smi` subprocess)
  - `state_bytes(obj) -> int` (recursive tensor bytes over nested list/tuple/dict/None)

- [ ] **Step 1: Write the failing tests**

```python
# code/event_ssm/tests/test_bench_metrics.py
"""Stage-10 bench_metrics unit tests — CPU-only by default (GPU busy with Stage-9 sweeps)."""
import time
import torch
import pytest
from event_ssm.benchmark.bench_metrics import time_fn, PowerSampler, state_bytes, measure_peak_vram


def test_time_fn_measures_sleep():
    r = time_fn(lambda: time.sleep(0.001), warmup=2, iters=20, device=torch.device("cpu"))
    assert 0.8 <= r["p50_ms"] <= 10.0          # sleep(1ms) plus scheduler noise
    assert r["p50_ms"] <= r["p95_ms"]
    assert r["iters"] == 20 and r["hz"] > 0
    assert set(r) == {"mean_ms", "std_ms", "p50_ms", "p95_ms", "hz", "iters"}


def test_power_sampler_with_injected_reader():
    with PowerSampler(interval_s=0.05, _read=lambda: 42.0) as ps:
        time.sleep(0.3)
    assert len(ps.samples) >= 3
    assert abs(ps.mean_w - 42.0) < 1e-6


def test_state_bytes_recursive():
    t1 = torch.zeros(2, 3, dtype=torch.float32)      # 24 B
    t2 = torch.zeros(4, dtype=torch.float16)         # 8 B
    nested = [(t1, {"a": t2}), None, [t2]]
    assert state_bytes(nested) == 24 + 8 + 8
    assert state_bytes(None) == 0


def test_measure_peak_vram_cpu_returns_sentinel():
    assert measure_peak_vram(lambda: None, device=torch.device("cpu")) == -1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest code/event_ssm/tests/test_bench_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'event_ssm.benchmark'`

- [ ] **Step 3: Implement**

```python
# code/event_ssm/benchmark/bench_metrics.py
"""Stage-10 measurement primitives (spec §5). Device-agnostic where possible so unit tests run on CPU
while the Stage-9 sweeps own the GPU; the orchestrator passes cuda devices at benchmark time."""
import statistics
import subprocess
import threading
import time

import torch


def _sync(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def time_fn(fn, *, warmup: int = 50, iters: int = 300, device=None):
    """Steady-state latency of fn(): warmup calls, then `iters` sync-bracketed timings (spec §5.3)."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times_ms = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    ordered = sorted(times_ms)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "mean_ms": statistics.fmean(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "p50_ms": p50,
        "p95_ms": p95,
        "hz": 1000.0 / p50 if p50 > 0 else float("inf"),
        "iters": iters,
    }


def measure_peak_vram(fn, *, device, n_calls: int = 20) -> float:
    """Peak allocated MB across n_calls of fn(). -1.0 on CPU (no VRAM concept)."""
    if device is None or device.type != "cuda":
        return -1.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_calls):
        fn()
    _sync(device)
    return torch.cuda.max_memory_allocated(device) / 1024**2


def read_gpu_power_w() -> float:
    """Instantaneous board power in watts: pynvml if available, else nvidia-smi (spec §5.5 fallback)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        return float(out)


class PowerSampler:
    """Background power sampling at interval_s; `_read` is injectable for tests."""

    def __init__(self, interval_s: float = 0.25, _read=None):
        self.interval_s = interval_s
        self._read = _read or read_gpu_power_w
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._read())
            except Exception:
                pass  # a dropped sample must not kill the benchmark
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5.0)
        return False

    @property
    def mean_w(self) -> float:
        return statistics.fmean(self.samples) if self.samples else float("nan")


def state_bytes(obj) -> int:
    """Recursive tensor byte count over nested containers — the per-stream recurrent-state footprint
    (spec §5.4): call on the state structure returned by one streaming step at B=1."""
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(state_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(state_bytes(v) for v in obj)
    return 0
```

Also create the empty package marker: `code/event_ssm/benchmark/__init__.py` (empty file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest code/event_ssm/tests/test_bench_metrics.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/benchmark/__init__.py code/event_ssm/benchmark/bench_metrics.py code/event_ssm/tests/test_bench_metrics.py
git commit -m "feat(stage10): measurement primitives (timing, VRAM, power sampling, state bytes)"
```

---

### Task 2: Analytic FLOPs + fvcore wrapper

**Files:**
- Modify: `code/event_ssm/benchmark/bench_metrics.py` (append)
- Test: `code/event_ssm/tests/test_bench_metrics.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces (used by Task 5):
  - `mamba2_layer_macs_per_token(d_model, d_state=64, d_conv=4, expand=2, headdim=64) -> int`
  - `s5_block_macs_per_token(dim, state_dim, ff_mult=1.0, glu=True, include_discretize=True) -> int`
  - `fvcore_network_flops(module: torch.nn.Module, inputs: tuple) -> dict` with keys `counted_gflops: float, unsupported_ops: dict[str, int]` (FLOPs = 2 × fvcore MACs)

- [ ] **Step 1: Ensure fvcore is importable (torch-safe install)**

Run: `python -c "import fvcore.nn; print('fvcore OK')"`
If `ModuleNotFoundError`, install pure-python deps only, then re-check torch:

```bash
pip install --no-deps fvcore iopath yacs portalocker termcolor tabulate
python -c "import fvcore.nn; print('fvcore OK')"
python -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; print('torch intact')"
```

- [ ] **Step 2: Write the failing tests (append to test_bench_metrics.py)**

```python
from event_ssm.benchmark.bench_metrics import (
    mamba2_layer_macs_per_token, s5_block_macs_per_token, fvcore_network_flops,
)


def test_mamba2_macs_hand_case():
    # d_model=128, expand=2 -> d_inner=256; headdim=64 -> nheads=4; S=64; k=4; ngroups=1
    # in_proj: 128*(2*256+2*64+4)=128*644=82432 ; conv: (256+128)*4=1536
    # scan: 3*256*64=49152 ; norm/gate: 2*256=512 ; out_proj: 256*128=32768  => 166400
    assert mamba2_layer_macs_per_token(128, d_state=64, d_conv=4, expand=2, headdim=64) == 166400


def test_s5_macs_hand_case():
    # dim=c=128, P=128, ff_mult=1, glu=True (complex MAC = 4 real MACs, spec §5.2):
    # Bu: 4*128*128=65536 ; Lambda*x: 4*128=512 ; Cx: 65536 ; D: 128 ; discretize: 10*128=1280
    # ff: enc 128*128*2=32768 + dec 128*128=16384 ; norms: 4*128=512  => 182656
    assert s5_block_macs_per_token(128, 128) == 182656


def test_fvcore_counts_toy_conv():
    import torch.nn as nn
    m = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    r = fvcore_network_flops(m, (torch.randn(1, 3, 16, 16),))
    # 3*8*3*3*16*16 = 55296 MACs -> 110592 FLOPs = 1.10592e-4 GFLOPs
    assert abs(r["counted_gflops"] - 55296 * 2 / 1e9) / (55296 * 2 / 1e9) < 0.05
    assert isinstance(r["unsupported_ops"], dict)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest code/event_ssm/tests/test_bench_metrics.py -v -k "macs or fvcore"`
Expected: FAIL — `ImportError: cannot import name 'mamba2_layer_macs_per_token'`

- [ ] **Step 4: Implement (append to bench_metrics.py)**

```python
# --- Analytic FLOPs for kernels invisible to tracing counters (spec §5.2) --------------------
# Conventions: 1 MAC = 2 FLOPs; complex MAC = 4 real MACs. Constants below are documented
# approximations for the small non-matmul terms (norms, discretisation); the dominant terms
# (projections, scans) are exact.

def mamba2_layer_macs_per_token(d_model: int, d_state: int = 64, d_conv: int = 4,
                                expand: int = 2, headdim: int = 64) -> int:
    """Per-token MACs of one Mamba-2 layer as used in mamba2_scan_time (ngroups=1)."""
    d_inner = expand * d_model
    nheads = d_inner // headdim
    conv_dim = d_inner + 2 * d_state                      # xBC width
    d_in_proj = 2 * d_inner + 2 * d_state + nheads        # z + xBC + dt
    macs = d_model * d_in_proj                            # in_proj
    macs += conv_dim * d_conv                             # depthwise causal conv
    macs += 3 * d_inner * d_state                         # scan: decay + inject + readout
    macs += 2 * d_inner                                   # gated RMSNorm (approx)
    macs += d_inner * d_model                             # out_proj
    return macs


def s5_block_macs_per_token(dim: int, state_dim: int, ff_mult: float = 1.0,
                            glu: bool = True, include_discretize: bool = True) -> int:
    """Per-token MACs of one RVT S5Block (models/layers/s5): diagonal complex SSM + GEGLU FF."""
    P, c = state_dim, dim
    macs = 4 * P * c                                      # B~ u   (complex mat-vec)
    macs += 4 * P                                         # Lambda_bar * x (diagonal complex)
    macs += 4 * P * c                                     # C~ x   (complex readout)
    macs += c                                             # D skip
    if include_discretize:
        macs += 10 * P                                    # per-step bilinear discretisation (approx)
    d_ff = int(c * ff_mult)
    macs += c * d_ff * (2 if glu else 1)                  # ff_enc (+GLU gate)
    macs += d_ff * c                                      # ff_dec
    macs += 4 * c                                         # 2 LayerNorms (approx)
    return macs


def fvcore_network_flops(module, inputs) -> dict:
    """Traced FLOPs via fvcore; never raises on unsupported ops — they are returned for the JSON."""
    from fvcore.nn import FlopCountAnalysis
    fca = FlopCountAnalysis(module, inputs)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    counted_macs = fca.total()                            # fvcore reports MACs
    return {
        "counted_gflops": counted_macs * 2 / 1e9,
        "unsupported_ops": {str(k): int(v) for k, v in fca.unsupported_ops().items()},
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest code/event_ssm/tests/test_bench_metrics.py -v`
Expected: 7 PASS

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/benchmark/bench_metrics.py code/event_ssm/tests/test_bench_metrics.py
git commit -m "feat(stage10): analytic Mamba-2/S5 FLOP formulas + fvcore wrapper"
```

---

### Task 3: Model construction + BenchModel wrapper

**Files:**
- Create: `code/event_ssm/benchmark/bench_models.py`
- Test: `code/event_ssm/tests/test_bench_models.py`

**Interfaces:**
- Consumes: `state_bytes` from `bench_metrics` (Task 1).
- Produces (used by Tasks 5):
  - `build_model(kind: str, device: torch.device, load_ckpt: bool = True) -> BenchModel`, kind ∈ {"eventssm", "baseline"}
  - `strip_prefix(sd: dict, prefix: str = "mdl.") -> dict` (pure, tested)
  - `BenchModel` attrs/methods: `.name: str`, `.detector`, `.cfg`, `.num_classes: int`, `.conf: float`, `.nms: float`,
    `.fresh_state() -> None` (initial state is None; RVT contract), `.full_step(frame_b20hw, state) -> (dets, state)`,
    `.network_step(frame_b20hw, state) -> (preds, state)`, `.components(frame, state) -> dict[str, callable]`
    (keys `backbone`, `neck_head`, `postprocess` — zero-arg callables for per-component timing),
    `.param_breakdown() -> dict[str, float]` (name → M params; includes "total"),
    `.temporal_hparams() -> list[dict]` (per temporal block: kind + dims + tokens; for analytic FLOPs),
    `.state_bytes_per_stream() -> int` (one B=1 step then `state_bytes`; **GPU-only** — mamba kernels),
    `.autocast_bf16: bool` attribute toggling an autocast context inside the step methods.
- Stage token grids (fixed 256×320 input): strides (4, 8, 16, 32) → `[(64, 80), (32, 40), (16, 20), (8, 10)]`.

- [ ] **Step 1: Discovery — verify the assumptions against reality (CPU, read-only)**

```bash
ls external/ssms_event_cameras/RVT/config/           # expect val.yaml among them
grep -rn "def postprocess" external/ssms_event_cameras/RVT/models/detection/yolox/utils/boxes.py | head -2
grep -n "self.mdl" external/ssms_event_cameras/RVT/modules/detection.py | head -3
python - <<'EOF'
import torch
sd = torch.load("checkpoints/gen1_base.ckpt", map_location="cpu", weights_only=False)["state_dict"]
print("baseline keys:", list(sd)[:3])
sd2 = torch.load("external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt",
                 map_location="cpu", weights_only=False)["state_dict"]
print("eventssm keys:", list(sd2)[:3])
EOF
```
Expected: `val.yaml` exists; `postprocess` defined in `boxes.py`; keys start with `mdl.`. **If any expectation fails, adapt the constants in Step 4 accordingly (config name, prefix, import path) — this is the only sanctioned deviation.**

- [ ] **Step 2: Write the failing tests**

```python
# code/event_ssm/tests/test_bench_models.py
"""Stage-10 bench_models tests. Construction + ckpt-load run on CPU (kernels only needed at forward
time); forward-dependent pieces are @pytest.mark.gpu and run later with the --smoke pass."""
import torch
import pytest
from event_ssm.benchmark.bench_models import strip_prefix, build_model, STAGE_TOKENS

GPU = pytest.mark.gpu


def test_strip_prefix_pure():
    sd = {"mdl.backbone.w": torch.zeros(1), "mdl.head.b": torch.zeros(1), "other.k": torch.zeros(1)}
    out = strip_prefix(sd, "mdl.")
    assert set(out) == {"backbone.w", "head.b", "other.k"}


def test_stage_tokens_grid():
    assert STAGE_TOKENS == [(64, 80), (32, 40), (16, 20), (8, 10)]


def test_build_eventssm_cpu_construct_and_load():
    bm = build_model("eventssm", device=torch.device("cpu"), load_ckpt=True)
    pb = bm.param_breakdown()
    assert 15 < pb["total"] < 30                      # ~19.2 M
    assert {"backbone_spatial", "backbone_temporal", "neck", "head", "total"} <= set(pb)
    th = bm.temporal_hparams()
    assert len(th) == 3 and all(t["kind"] == "mamba2" for t in th)   # temporal on stages 2-4
    assert bm.num_classes == 2 and 0 < bm.conf < 1


def test_build_baseline_cpu_construct_and_load():
    bm = build_model("baseline", device=torch.device("cpu"), load_ckpt=True)
    pb = bm.param_breakdown()
    assert 12 < pb["total"] < 30                      # ~18 M
    th = bm.temporal_hparams()
    assert len(th) == 4 and all(t["kind"] == "s5" for t in th)       # S5 on all 4 stages


@GPU
def test_full_step_runs_on_gpu():
    dev = torch.device("cuda")
    bm = build_model("eventssm", device=dev, load_ckpt=True)
    frame = torch.zeros(1, 20, 256, 320, device=dev)
    dets, st = bm.full_step(frame, None)
    assert isinstance(dets, list) and len(dets) == 1
    assert bm.state_bytes_per_stream() > 0
```

Also register the marker — append to `code/event_ssm/pytest.ini` under `[pytest]`:

```ini
markers =
    gpu: needs an idle CUDA GPU (run via the --smoke coordination, not by default)
addopts = -p no:launch_testing -p no:launch_ros -m "not gpu"
```
(Keep the existing `-p no:...` flags; add `-m "not gpu"` so GPU tests are skipped by default and run explicitly with `-m gpu`.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest code/event_ssm/tests/test_bench_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'event_ssm.benchmark.bench_models'`

- [ ] **Step 4: Implement**

```python
# code/event_ssm/benchmark/bench_models.py
"""Stage-10 model construction (spec §3): both detectors via the RVT public API with REAL weights.
Config is composed with Hydra against the vendored RVT config tree (never hardcode dims); weights
load strict after stripping the Lightning `mdl.` prefix. BenchModel gives every metric an identical
interface for both models."""
import contextlib
import pathlib

import torch
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[3]
RVT = REPO / "external/ssms_event_cameras/RVT"

CKPTS = {
    "eventssm": RVT / "RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt",
    "baseline": REPO / "checkpoints/gen1_base.ckpt",
}
EXPERIMENT = {"eventssm": "+experiment/gen1=resnet_mamba", "baseline": "+experiment/gen1=base.yaml"}
# fixed 256x320 input, strides 4/8/16/32 -> per-stage token grids (asserted against the backbone)
STAGE_TOKENS = [(64, 80), (32, 40), (16, 20), (8, 10)]


def compose_cfg(kind: str):
    from hydra import compose, initialize_config_dir
    overrides = [
        "dataset=gen1", f"dataset.path={REPO / 'data/gen1_raw/gen1'}", "model=rnndet",
        EXPERIMENT[kind], "checkpoint=''", "use_test_set=1",
    ]
    with initialize_config_dir(config_dir=str(RVT / "config"), version_base=None):
        return compose(config_name="val", overrides=overrides)


def strip_prefix(sd: dict, prefix: str = "mdl.") -> dict:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def load_weights(detector: torch.nn.Module, ckpt_path: pathlib.Path) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = strip_prefix(sd)
    model_keys = set(detector.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    missing = model_keys - set(filtered)
    assert not missing, f"{ckpt_path.name}: ckpt covers {len(filtered)}/{len(model_keys)}; missing e.g. {sorted(missing)[:5]}"
    detector.load_state_dict(filtered, strict=True)


class BenchModel:
    def __init__(self, name: str, detector, cfg, device: torch.device):
        self.name, self.detector, self.cfg, self.device = name, detector, cfg, device
        self.num_classes = int(cfg.model.head.num_classes)
        self.conf = float(cfg.model.postprocess.confidence_threshold)
        self.nms = float(cfg.model.postprocess.nms_threshold)
        self.autocast_bf16 = True
        from models.detection.yolox.utils import postprocess  # RVT's own NMS path
        self._postprocess = postprocess

    def _ac(self):
        if self.autocast_bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def fresh_state(self):
        return None  # RVT contract: None -> zero-init states inside forward_backbone

    def network_step(self, frame, state):
        with torch.no_grad(), self._ac():
            feats, new_state = self.detector.forward_backbone(
                frame.unsqueeze(0), previous_states=state, train_step=False)
            sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
            preds, _ = self.detector.forward_detect(backbone_features=sel)
        return preds, new_state

    def full_step(self, frame, state):
        preds, new_state = self.network_step(frame, state)
        with torch.no_grad():
            dets = self._postprocess(prediction=preds.float(), num_classes=self.num_classes,
                                     conf_thre=self.conf, nms_thre=self.nms)
        return dets, new_state

    def components(self, frame, state):
        """Zero-arg callables for the per-component timing pass (spec §5.3). Uses a fixed input/state
        snapshot so each component is timed in isolation with realistic tensors."""
        with torch.no_grad(), self._ac():
            feats, st = self.detector.forward_backbone(frame.unsqueeze(0), previous_states=state, train_step=False)
            sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
            preds, _ = self.detector.forward_detect(backbone_features=sel)

        def run_backbone():
            with torch.no_grad(), self._ac():
                self.detector.forward_backbone(frame.unsqueeze(0), previous_states=state, train_step=False)

        def run_neck_head():
            with torch.no_grad(), self._ac():
                self.detector.forward_detect(backbone_features=sel)

        def run_postprocess():
            with torch.no_grad():
                self._postprocess(prediction=preds.float(), num_classes=self.num_classes,
                                  conf_thre=self.conf, nms_thre=self.nms)

        return {"backbone": run_backbone, "neck_head": run_neck_head, "postprocess": run_postprocess}

    def param_breakdown(self) -> dict:
        m = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
        out = {"total": m(self.detector), "neck": m(self.detector.fpn), "head": m(self.detector.yolox_head)}
        bb = self.detector.backbone
        if hasattr(bb, "spatial"):     # ours
            out["backbone_spatial"] = m(bb.spatial)
            out["backbone_temporal"] = m(bb.temporal)
        else:                          # baseline ViT+S5 (not split further, spec §5.1)
            out["backbone"] = m(bb)
        return out

    def temporal_hparams(self) -> list:
        """Introspect temporal blocks for the analytic FLOP add-on (spec §5.2)."""
        out = []
        bb = self.detector.backbone
        if hasattr(bb, "temporal"):    # ours: MambaTemporalBlock dict keyed by stage str
            for stage_str, block in bb.temporal.items():
                s = int(stage_str)
                h, w = STAGE_TOKENS[s - 1]
                for layer in block.layers:
                    out.append({"kind": "mamba2", "tokens": h * w, "d_model": layer.d_model,
                                "d_state": layer.d_state, "d_conv": layer.d_conv,
                                "expand": layer.expand, "headdim": layer.headdim})
        else:                          # baseline: one S5Block per RNNDetectorStage
            from models.layers.s5.s5_model import S5Block
            stage_idx = 0
            for mod in bb.modules():
                if isinstance(mod, S5Block):
                    h, w = STAGE_TOKENS[stage_idx]
                    dim = mod.attn_norm.normalized_shape[0]
                    out.append({"kind": "s5", "tokens": h * w, "dim": dim, "state_dim": dim})
                    stage_idx += 1
        return out

    def state_bytes_per_stream(self) -> int:
        from event_ssm.benchmark.bench_metrics import state_bytes
        frame = torch.zeros(1, 20, 256, 320, device=self.device)
        _, st = self.network_step(frame, None)
        return state_bytes(st)


def build_model(kind: str, device: torch.device, load_ckpt: bool = True) -> BenchModel:
    assert kind in CKPTS, kind
    if kind == "eventssm":
        from event_ssm.integration.register import register_resnet_mamba
        register_resnet_mamba()
    cfg = compose_cfg(kind)
    from models.detection.yolox_extension.models.detector import YoloXDetector
    detector = YoloXDetector(OmegaConf.create(cfg.model)) if not OmegaConf.is_config(cfg.model) \
        else YoloXDetector(cfg.model)
    if load_ckpt:
        load_weights(detector, CKPTS[kind])
    detector = detector.to(device).eval()
    return BenchModel(kind, detector, cfg, device)
```

- [ ] **Step 5: Run tests to verify they pass (CPU set)**

Run: `pytest code/event_ssm/tests/test_bench_models.py -v`
Expected: 4 PASS, 1 deselected (the `gpu` one). If `compose` fails on a MISSING key or the S5Block
introspection finds ≠4 blocks, fix against the discovery output from Step 1 — the test bounds
(param ranges, block counts) are the acceptance criteria, not the exact code above.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/benchmark/bench_models.py code/event_ssm/tests/test_bench_models.py code/event_ssm/pytest.ini
git commit -m "feat(stage10): model construction + BenchModel wrapper (real ckpts, uniform interface)"
```

---

### Task 4: Real-clip extraction

**Files:**
- Create: `code/event_ssm/benchmark/bench_clip.py`
- Test: `code/event_ssm/tests/test_bench_clip.py`

**Interfaces:**
- Consumes: nothing.
- Produces (Task 5): `load_bench_clip(n_frames: int = 64, root: Path | None = None, cache: Path | None = None) -> torch.Tensor` of shape `(n_frames, 20, 256, 320)`, dtype float32; caches `{"clip", "src"}` to `results/stage10/bench_clip.pt`.

- [ ] **Step 1: Discovery — real h5 layout (CPU, read-only)**

```bash
python - <<'EOF'
import h5py, hdf5plugin, pathlib
root = pathlib.Path("data/gen1_raw/gen1/test")
seq = sorted(p for p in root.iterdir() if p.is_dir())[0]
h5 = seq / "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
with h5py.File(h5, "r") as f:
    for k in f.keys(): print(k, f[k].shape, f[k].dtype)
EOF
```
Expected: one dataset (likely `data`) with shape `(T, 20, 240, 304)` uint8. Adapt the `KEY` constant if named differently. Also verify the float conversion point in RVT (spec §4): `grep -n "float()" external/ssms_event_cameras/RVT/modules/detection.py | head -3` — record the line in the module docstring.

- [ ] **Step 2: Write the failing test**

```python
# code/event_ssm/tests/test_bench_clip.py
import h5py, hdf5plugin
import numpy as np
import torch
from event_ssm.benchmark.bench_clip import load_bench_clip


def _make_fake_tree(tmp_path, T=16):
    seq = tmp_path / "seq_000" / "event_representations_v2" / "stacked_histogram_dt=50_nbins=10"
    seq.mkdir(parents=True)
    with h5py.File(seq / "event_representations.h5", "w") as f:
        f.create_dataset("data", data=np.random.randint(0, 10, (T, 20, 240, 304), dtype=np.uint8))
    return tmp_path


def test_load_bench_clip_shape_pad_cache(tmp_path):
    root = _make_fake_tree(tmp_path)
    cache = tmp_path / "clip.pt"
    clip = load_bench_clip(n_frames=8, root=root, cache=cache, skip=2)
    assert clip.shape == (8, 20, 256, 320) and clip.dtype == torch.float32
    assert clip[:, :, 240:, :].abs().sum() == 0 and clip[:, :, :, 304:].abs().sum() == 0  # zero pad
    assert cache.exists()
    clip2 = load_bench_clip(n_frames=8, root=root, cache=cache)   # cache hit path
    assert torch.equal(clip, clip2)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest code/event_ssm/tests/test_bench_clip.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement**

```python
# code/event_ssm/benchmark/bench_clip.py
"""Real Gen1 frames for the benchmark (spec §4): 64 consecutive dt=50 test frames, zero-padded
240x304 -> 256x320, float32 raw counts — matching the RVT eval feed (uint8 histograms are cast to
float by the module input path; verified during Task-4 discovery). Cached for reproducibility."""
import pathlib

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "data/gen1_raw/gen1/test"
DEFAULT_CACHE = REPO / "results/stage10/bench_clip.pt"
EVR = "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
KEY = "data"   # confirmed against the real file in Task-4 discovery


def load_bench_clip(n_frames: int = 64, root=None, cache=None, skip: int = 100) -> torch.Tensor:
    import h5py, hdf5plugin  # noqa: F401  (hdf5plugin registers the blosc filter)
    root = pathlib.Path(root) if root else DEFAULT_ROOT
    cache = pathlib.Path(cache) if cache else DEFAULT_CACHE
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        if blob["clip"].shape[0] == n_frames:
            return blob["clip"]
    seq = sorted(p for p in root.iterdir() if p.is_dir())[0]     # deterministic first sequence
    h5 = seq / EVR
    with h5py.File(h5, "r") as f:
        key = KEY if KEY in f else list(f.keys())[0]
        assert f[key].shape[0] >= skip + n_frames, f"sequence too short: {f[key].shape}"
        arr = f[key][skip:skip + n_frames]                        # (n, 20, 240, 304) uint8
    t = torch.from_numpy(arr).float()
    clip = torch.zeros(n_frames, t.shape[1], 256, 320)
    clip[:, :, :240, :304] = t
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clip": clip, "src": str(h5)}, cache)
    return clip
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest code/event_ssm/tests/test_bench_clip.py -v`
Expected: 1 PASS. Then one real-data spot check (CPU): `python -c "import sys; sys.path.insert(0,'code'); from event_ssm.benchmark.bench_clip import load_bench_clip; c=load_bench_clip(8); print(c.shape, c.max())"` → `(8, 20, 256, 320)` and a small positive max.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/benchmark/bench_clip.py code/event_ssm/tests/test_bench_clip.py
git commit -m "feat(stage10): real Gen1 bench clip extraction with cache"
```

---

### Task 5: GPU orchestrator

**Files:**
- Create: `code/event_ssm/scripts/stage10_benchmark.py`
- Test: `code/event_ssm/tests/test_bench_schema.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4 (exact signatures above).
- Produces: `results/stage10/bench_results.json` with schema:

```json
{"schema": 1,
 "meta": {"date": "...", "torch": "...", "cuda": "...", "gpu": "...", "driver": "...",
          "clocks_sm_mhz": 0, "git_sha": "...", "clip_src": "...", "ckpts": {"eventssm": "...", "baseline": "..."},
          "test_ap": {"eventssm": 0.462, "baseline": 0.477}, "weights": "plain mdl.* (non-EMA); compute metrics weight-independent"},
 "models": {"<kind>": {
    "params_m": {"total": 0.0},
    "flops": {"counted_gflops": 0.0, "analytic_gflops": 0.0, "total_gflops": 0.0, "unsupported_ops": {}, "counted_incomplete": false},
    "latency": {"bf16": {"full": {}, "network": {}, "components": {}, "throughput": {"b4_fps": 0.0, "b8_fps": 0.0}},
                "fp32": {"full": {}, "network": {}, "components": {}, "throughput": {"b4_fps": 0.0, "b8_fps": 0.0}}},
    "vram": {"inference_mb": 0.0, "train_mb": 0.0, "state_kb_per_stream": 0.0},
    "energy": {"idle_w": 0.0, "load_w": 0.0, "j_per_frame": 0.0, "frames": 0, "seconds": 0.0}}}}
```
  Latency leaf dicts are `time_fn` outputs. `validate_results(d) -> list[str]` returns problems (empty = valid).

- [ ] **Step 1: Write the failing schema test**

```python
# code/event_ssm/tests/test_bench_schema.py
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "stage10_benchmark",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "stage10_benchmark.py")
s10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s10)


def _minimal():
    lat = {"mean_ms": 1.0, "std_ms": 0.1, "p50_ms": 1.0, "p95_ms": 1.2, "hz": 1000.0, "iters": 10}
    prec = {"full": lat, "network": lat,
            "components": {"backbone": lat, "neck_head": lat, "postprocess": lat},
            "throughput": {"b4_fps": 1.0, "b8_fps": 1.0}}
    return {"schema": 1, "meta": {"test_ap": {"eventssm": 0.462, "baseline": 0.477}},
            "models": {"eventssm": {
                "params_m": {"total": 19.2},
                "flops": {"counted_gflops": 1.0, "analytic_gflops": 0.5, "total_gflops": 1.5,
                          "unsupported_ops": {}, "counted_incomplete": False},
                "latency": {"bf16": prec, "fp32": prec},
                "vram": {"inference_mb": 100.0, "train_mb": 1000.0, "state_kb_per_stream": 50.0},
                "energy": {"idle_w": 30.0, "load_w": 150.0, "j_per_frame": 0.5, "frames": 100, "seconds": 10.0}}}}


def test_validate_accepts_minimal():
    assert s10.validate_results(_minimal()) == []


def test_validate_flags_missing():
    d = _minimal(); del d["models"]["eventssm"]["vram"]
    problems = s10.validate_results(d)
    assert problems and "vram" in problems[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/event_ssm/tests/test_bench_schema.py -v`
Expected: FAIL — file `stage10_benchmark.py` does not exist.

- [ ] **Step 3: Implement**

```python
# code/event_ssm/scripts/stage10_benchmark.py
"""Stage-10 GPU orchestrator (spec §6): builds both models with real weights, measures the five
metrics, writes ONE results/stage10/bench_results.json. Run through stage10_run_local.sh (idle-GPU
guard). --smoke shrinks every loop for a <2-min end-to-end wiring check."""
import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
for p in (REPO / "code", REPO / "external/ssms_event_cameras/RVT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

REQUIRED_MODEL_KEYS = {"params_m", "flops", "latency", "vram", "energy"}
TEST_AP = {"eventssm": 0.462, "baseline": 0.477}   # Stage-8 one-shot test results


def validate_results(d: dict) -> list:
    problems = []
    if d.get("schema") != 1:
        problems.append("schema != 1")
    for kind, m in d.get("models", {}).items():
        missing = REQUIRED_MODEL_KEYS - set(m)
        if missing:
            problems.append(f"{kind}: missing {sorted(missing)}")
        for prec in ("bf16", "fp32"):
            lat = m.get("latency", {}).get(prec, {})
            if lat and not {"full", "network", "components", "throughput"} <= set(lat):
                problems.append(f"{kind}.latency.{prec}: incomplete")
    return problems


def _nvsmi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unavailable"


def gather_meta(clip_src: str) -> dict:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": _nvsmi("name"), "driver": _nvsmi("driver_version"),
            "clocks_sm_mhz": _nvsmi("clocks.sm"), "git_sha": sha, "clip_src": clip_src,
            "ckpts": {},  # filled by main()
            "test_ap": TEST_AP,
            "weights": "plain mdl.* (non-EMA); compute metrics weight-independent"}


def measure_model(kind: str, device, clip, smoke: bool) -> dict:
    from event_ssm.benchmark.bench_models import build_model, CKPTS
    from event_ssm.benchmark import bench_metrics as bm

    model = build_model(kind, device=device, load_ckpt=True)
    n = clip.shape[0]
    warmup, iters = (3, 10) if smoke else (50, 300)
    frames = clip.to(device)
    out = {"params_m": model.param_breakdown()}

    # ---- FLOPs: counted (fvcore on network_step via a wrapper module) + analytic add-on ----
    class NetOnly(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.network_step(x, None)[0]
    try:
        counted = bm.fvcore_network_flops(NetOnly(model), (frames[0],))
        incomplete = False
    except Exception as e:                       # tracing may fail on vmap paths (spec §8)
        counted = {"counted_gflops": 0.0, "unsupported_ops": {"trace_failed": 1, "err": str(e)[:200]}}
        incomplete = True
    analytic_macs = 0
    for t in model.temporal_hparams():
        if t["kind"] == "mamba2":
            analytic_macs += t["tokens"] * bm.mamba2_layer_macs_per_token(
                t["d_model"], d_state=t["d_state"], d_conv=t["d_conv"],
                expand=t["expand"], headdim=t["headdim"])
        else:
            analytic_macs += t["tokens"] * bm.s5_block_macs_per_token(t["dim"], t["state_dim"])
    analytic_gflops = analytic_macs * 2 / 1e9
    out["flops"] = {"counted_gflops": counted["counted_gflops"], "analytic_gflops": analytic_gflops,
                    "total_gflops": counted["counted_gflops"] + analytic_gflops,
                    "unsupported_ops": counted["unsupported_ops"], "counted_incomplete": incomplete}

    # ---- latency (bf16 + fp32): headline full, network-only, per-component, throughput ----
    out["latency"] = {}
    for prec in ("bf16", "fp32"):
        model.autocast_bf16 = (prec == "bf16")
        state = {"s": None}; idx = {"i": 0}
        def step_full():
            _, state["s"] = model.full_step(frames[idx["i"] % n], state["s"]); idx["i"] += 1
        def step_net():
            _, state["s"] = model.network_step(frames[idx["i"] % n], state["s"]); idx["i"] += 1
        lat_full = bm.time_fn(step_full, warmup=warmup, iters=iters, device=device)
        state["s"], idx["i"] = None, 0
        lat_net = bm.time_fn(step_net, warmup=warmup, iters=iters, device=device)
        comps = {name: bm.time_fn(fn, warmup=max(3, warmup // 5), iters=max(10, iters // 3), device=device)
                 for name, fn in model.components(frames[0], None).items()}
        thr = {}
        for b in (4, 8):
            batch = frames[0].repeat(b, 1, 1, 1)
            st = {"s": None}
            def step_b():
                _, st["s"] = model.network_step(batch, st["s"])
            r = bm.time_fn(step_b, warmup=max(3, warmup // 5), iters=max(10, iters // 3), device=device)
            thr[f"b{b}_fps"] = 1000.0 / r["p50_ms"] * b
        out["latency"][prec] = {"full": lat_full, "network": lat_net, "components": comps, "throughput": thr}

    # ---- VRAM ----
    model.autocast_bf16 = True
    st = {"s": None}
    def infer_step():
        _, st["s"] = model.full_step(frames[0], st["s"])
    inf_mb = bm.measure_peak_vram(infer_step, device=device, n_calls=5 if smoke else 20)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model.detector.train()
    x = frames[:5].unsqueeze(1).repeat(1, 4, 1, 1, 1)             # (L=5, B=4, 20, 256, 320)
    feats, _ = model.detector.forward_backbone(x, previous_states=None, train_step=True)
    sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
    targets = torch.zeros(4, 3, 5, device=device)
    targets[:, 0] = torch.tensor([0., 160., 128., 40., 30.], device=device)
    _, losses = model.detector.forward_detect(backbone_features=sel, targets=targets)
    loss = losses["loss"] if isinstance(losses, dict) else losses
    loss.backward()
    train_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    model.detector.zero_grad(set_to_none=True); model.detector.eval()
    out["vram"] = {"inference_mb": inf_mb, "train_mb": train_mb,
                   "state_kb_per_stream": model.state_bytes_per_stream() / 1024}

    # ---- energy (bf16 streaming; spec §5.5) ----
    idle_s, load_s = (2, 5) if smoke else (10, 60)
    with bm.PowerSampler() as ps_idle:
        time.sleep(idle_s)
    st = {"s": None}; count = {"n": 0}
    with bm.PowerSampler() as ps_load:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < load_s:
            _, st["s"] = model.full_step(frames[count["n"] % n], st["s"]); count["n"] += 1
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
    delta_w = ps_load.mean_w - ps_idle.mean_w
    out["energy"] = {"idle_w": ps_idle.mean_w, "load_w": ps_load.mean_w,
                     "j_per_frame": delta_w * elapsed / max(count["n"], 1),
                     "frames": count["n"], "seconds": elapsed}

    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="both", choices=["both", "eventssm", "baseline"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results/stage10"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "Stage-10 benchmark needs the CUDA GPU (run via stage10_run_local.sh)"
    device = torch.device("cuda")

    from event_ssm.benchmark.bench_clip import load_bench_clip, DEFAULT_CACHE
    clip = load_bench_clip(8 if args.smoke else 64)
    blob = torch.load(DEFAULT_CACHE, weights_only=False)

    from event_ssm.benchmark.bench_models import CKPTS
    results = {"schema": 1, "meta": gather_meta(blob["src"]), "models": {}}
    results["meta"]["ckpts"] = {k: str(v) for k, v in CKPTS.items()}
    kinds = ["eventssm", "baseline"] if args.models == "both" else [args.models]
    for kind in kinds:
        print(f"[stage10] measuring {kind} ({'smoke' if args.smoke else 'full'}) ...")
        results["models"][kind] = measure_model(kind, device, clip, args.smoke)

    problems = validate_results(results)
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / ("bench_results_smoke.json" if args.smoke else "bench_results.json")
    outfile.write_text(json.dumps(results, indent=2))
    print(f"[stage10] wrote {outfile}")
    if problems:
        print("[stage10] VALIDATION PROBLEMS:", problems); sys.exit(1)
    print("[stage10] results validate clean")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run schema tests to verify they pass**

Run: `pytest code/event_ssm/tests/test_bench_schema.py -v`
Expected: 2 PASS. (The GPU path is exercised later by the user-run `--smoke`.)

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/scripts/stage10_benchmark.py code/event_ssm/tests/test_bench_schema.py
git commit -m "feat(stage10): GPU orchestrator with schema validation and smoke mode"
```

---

### Task 6: Guarded launcher

**Files:**
- Create: `code/event_ssm/scripts/stage10_run_local.sh`

**Interfaces:**
- Consumes: `stage10_benchmark.py` CLI (Task 5).
- Produces: the user-facing entry point; `NVSMI` env override enables stub-testing the guard.

- [ ] **Step 1: Implement**

```bash
#!/usr/bin/env bash
# Stage-10 efficiency benchmark launcher. REFUSES to run unless the GPU is idle — benchmark numbers
# measured on a contended GPU are garbage (spec §6). Usage:
#   bash code/event_ssm/scripts/stage10_run_local.sh --smoke     # <2 min wiring check
#   bash code/event_ssm/scripts/stage10_run_local.sh             # the real ~20-30 min run
set -euo pipefail

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
NVSMI="${NVSMI:-nvidia-smi}"   # override with a stub for guard tests

read -r UTIL MEM <<<"$($NVSMI --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')"
if (( UTIL >= 10 )) || (( MEM >= 1500 )); then
  echo "[stage10] ABORT: GPU not idle (util=${UTIL}%, mem=${MEM} MiB; need <10% and <1500 MiB)." >&2
  echo "[stage10] Wait for Stage-9 sweeps/renders to finish, then re-run." >&2
  exit 1
fi
echo "[stage10] GPU idle (util=${UTIL}%, mem=${MEM} MiB) — proceeding."

set +u; source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals; set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"

OUT="$REPO/results/stage10"; mkdir -p "$OUT"
LOG="$OUT/console_$(date +%Y%m%d_%H%M%S).log"
echo "[stage10] log -> $LOG"
python "$REPO/code/event_ssm/scripts/stage10_benchmark.py" "$@" 2>&1 | tee "$LOG"
echo "[stage10] next: python $REPO/code/event_ssm/scripts/stage10_report.py"
```

- [ ] **Step 2: Syntax + guard stub tests**

```bash
bash -n code/event_ssm/scripts/stage10_run_local.sh && chmod +x code/event_ssm/scripts/stage10_run_local.sh
# stub 1: busy GPU -> must abort with exit 1
cat > /tmp/nvsmi_busy <<'EOF'
#!/usr/bin/env bash
echo "97, 15000"
EOF
chmod +x /tmp/nvsmi_busy
NVSMI=/tmp/nvsmi_busy bash code/event_ssm/scripts/stage10_run_local.sh --smoke; echo "exit=$?"
# Expected: "[stage10] ABORT: GPU not idle (util=97%, mem=15000 MiB ...)" and exit=1
# stub 2: idle GPU -> guard passes (the python then fails ONLY if run while CUDA busy; do not proceed past the guard message check — Ctrl-C after "proceeding" if Stage-9 owns the GPU)
cat > /tmp/nvsmi_idle <<'EOF'
#!/usr/bin/env bash
echo "2, 600"
EOF
chmod +x /tmp/nvsmi_idle
NVSMI=/tmp/nvsmi_idle bash -c 'source code/event_ssm/scripts/stage10_run_local.sh --help 2>&1 | head -2' || true
```
Expected: busy stub prints the ABORT line and exits 1; idle stub prints "GPU idle ... proceeding".

- [ ] **Step 3: Commit**

```bash
git add code/event_ssm/scripts/stage10_run_local.sh
git commit -m "feat(stage10): guarded launcher (refuses non-idle GPU)"
```

---

### Task 7: Reporter — table + Pareto + component bars

**Files:**
- Create: `code/event_ssm/scripts/stage10_report.py`
- Create: `code/event_ssm/tests/fixtures/stage10_fixture.json`
- Test: `code/event_ssm/tests/test_bench_report.py`

**Interfaces:**
- Consumes: `bench_results.json` (Task-5 schema, both models present).
- Produces: `results/stage10/efficiency_table.md`, `.csv`, `stage10_pareto.{png,pdf}`, `stage10_latency_breakdown.{png,pdf}`. CLI: `python stage10_report.py [--json PATH] [--out DIR]`.

- [ ] **Step 1: Create the fixture** — `code/event_ssm/tests/fixtures/stage10_fixture.json`: exactly the `_minimal()` dict from Task 5's test, extended with a `"baseline"` entry (copy the `eventssm` block, change `params_m.total` to 18.0, `flops.total_gflops` to 2.5, latency `p50_ms` values to 2.0). Write it as real JSON (no placeholders — copy the structure, fill every number).

- [ ] **Step 2: Write the failing test**

```python
# code/event_ssm/tests/test_bench_report.py
import importlib.util, json, pathlib
HERE = pathlib.Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "stage10_report", HERE.parents[1] / "scripts" / "stage10_report.py")
rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
FIXTURE = HERE.parent / "fixtures" / "stage10_fixture.json"


def test_report_renders_everything(tmp_path):
    outputs = rep.generate(json_path=FIXTURE, out_dir=tmp_path)
    for f in ("efficiency_table.md", "efficiency_table.csv",
              "stage10_pareto.png", "stage10_latency_breakdown.png"):
        assert (tmp_path / f).exists(), f
    md = (tmp_path / "efficiency_table.md").read_text()
    assert "mAP/GFLOP" in md and "eventssm" in md and "baseline" in md


def test_report_tolerates_single_model(tmp_path):
    d = json.loads(FIXTURE.read_text()); del d["models"]["baseline"]
    j = tmp_path / "partial.json"; j.write_text(json.dumps(d))
    rep.generate(json_path=j, out_dir=tmp_path)          # must not raise
    assert (tmp_path / "efficiency_table.md").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest code/event_ssm/tests/test_bench_report.py -v`
Expected: FAIL — `stage10_report.py` does not exist.

- [ ] **Step 4: Implement**

```python
# code/event_ssm/scripts/stage10_report.py
"""Stage-10 reporter (spec §6): bench_results.json -> efficiency table (md+csv), Pareto figure,
per-component latency bars. CPU-only; re-run freely — never re-measures."""
import argparse
import csv
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
HUE = {"eventssm": "#2a78d6", "baseline": "#1baf7a"}
LABEL = {"eventssm": "EventSSM (ours, Mamba)", "baseline": "S5-RVT (baseline)"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _rows(d: dict) -> list:
    rows = []
    for kind, m in d["models"].items():
        ap = d["meta"]["test_ap"].get(kind)
        gflops = m["flops"]["total_gflops"]
        bf = m["latency"]["bf16"]
        rows.append({
            "model": kind, "test_ap": ap,
            "params_m": m["params_m"]["total"],
            "gflops_total": gflops,
            "gflops_counted": m["flops"]["counted_gflops"],
            "gflops_analytic": m["flops"]["analytic_gflops"],
            "lat_full_p50_ms": bf["full"]["p50_ms"], "lat_full_p95_ms": bf["full"]["p95_ms"],
            "lat_network_p50_ms": bf["network"]["p50_ms"],
            "hz_full": bf["full"]["hz"],
            "fps_b4": bf["throughput"]["b4_fps"], "fps_b8": bf["throughput"]["b8_fps"],
            "vram_inf_mb": m["vram"]["inference_mb"], "vram_train_mb": m["vram"]["train_mb"],
            "state_kb": m["vram"]["state_kb_per_stream"],
            "j_per_frame": m["energy"]["j_per_frame"], "load_w": m["energy"]["load_w"],
            "map_per_gflop": (ap / gflops) if (ap and gflops) else None,
        })
    return rows


def _table_md(rows: list) -> str:
    cols = [("Model", "model"), ("test/AP", "test_ap"), ("Params (M)", "params_m"),
            ("GFLOPs total", "gflops_total"), ("(counted)", "gflops_counted"), ("(analytic)", "gflops_analytic"),
            ("Full p50 (ms)", "lat_full_p50_ms"), ("Full p95 (ms)", "lat_full_p95_ms"),
            ("Net p50 (ms)", "lat_network_p50_ms"), ("Hz", "hz_full"),
            ("fps@B4", "fps_b4"), ("fps@B8", "fps_b8"),
            ("VRAM inf (MB)", "vram_inf_mb"), ("VRAM train (MB)", "vram_train_mb"),
            ("State (KB/stream)", "state_kb"), ("J/frame", "j_per_frame"), ("Load (W)", "load_w"),
            ("mAP/GFLOP", "map_per_gflop")]
    fmt = lambda v: (f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v)))
    lines = ["| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for _, k in cols) + " |")
    lines.append("")
    lines.append("*Latency = bf16 streaming (B=1, L=1, state carried), full pipeline incl. postprocess/NMS "
                 "(headline) and network-only. Energy = differential J/frame vs idle, desktop-GPU proxy. "
                 "FLOPs = fvcore-counted + analytic SSM-kernel add-on (split shown).*")
    return "\n".join(lines)


def _fig_pareto(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for r in rows:
        if r["test_ap"] is None:
            continue
        ax.scatter(r["lat_full_p50_ms"], r["test_ap"] * 100, s=60 * r["params_m"],
                   color=HUE.get(r["model"], MUTED), alpha=0.85, zorder=3)
        ax.annotate(f"{LABEL.get(r['model'], r['model'])}\n{r['params_m']:.1f}M · {r['gflops_total']:.1f} GFLOPs",
                    (r["lat_full_p50_ms"], r["test_ap"] * 100), textcoords="offset points",
                    xytext=(12, -4), fontsize=8.5, color=INK2)
    ax.set_xlabel("Full-pipeline latency p50 (ms, bf16, B=1 streaming)", color=INK2)
    ax.set_ylabel("Gen1 test/AP (COCO ×100)", color=INK2)
    ax.set_title("Accuracy vs latency — bubble = parameters", color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "stage10_pareto.png", dpi=150, facecolor="white")
    fig.savefig(out_dir / "stage10_pareto.pdf", facecolor="white")
    plt.close(fig)


def _fig_components(d, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comps = ["backbone", "neck_head", "postprocess"]
    shades = {"backbone": 1.0, "neck_head": 0.65, "postprocess": 0.35}   # alpha steps of the model hue
    kinds = list(d["models"].keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = range(len(kinds))
    for xi, kind in zip(xs, kinds):
        bottom = 0.0
        for c in comps:
            v = d["models"][kind]["latency"]["bf16"]["components"][c]["p50_ms"]
            ax.bar(xi, v, bottom=bottom, width=0.5, color=HUE.get(kind, MUTED),
                   alpha=shades[c], edgecolor="white", linewidth=0.8)
            ax.annotate(f"{c} {v:.2f}", (xi, bottom + v / 2), ha="center", va="center",
                        fontsize=8, color=INK)
            bottom += v
        ax.annotate(f"Σ {bottom:.2f} ms", (xi, bottom), ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL.get(k, k) for k in kinds], color=INK2)
    ax.set_ylabel("Latency p50 (ms, bf16, B=1)", color=INK2)
    ax.set_title("Per-component latency (stacked)", color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "stage10_latency_breakdown.png", dpi=150, facecolor="white")
    fig.savefig(out_dir / "stage10_latency_breakdown.pdf", facecolor="white")
    plt.close(fig)


def generate(json_path, out_dir) -> list:
    json_path, out_dir = pathlib.Path(json_path), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(json_path.read_text())
    rows = _rows(d)
    (out_dir / "efficiency_table.md").write_text(_table_md(rows) + "\n")
    with open(out_dir / "efficiency_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _fig_pareto(rows, out_dir)
    _fig_components(d, out_dir)
    print(f"[stage10-report] wrote table + figures to {out_dir}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "results/stage10/bench_results.json"))
    ap.add_argument("--out", default=str(REPO / "results/stage10"))
    a = ap.parse_args()
    generate(a.json, a.out)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest code/event_ssm/tests/test_bench_report.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Visual check** — run the reporter on the fixture and LOOK at the figures:

Run: `python code/event_ssm/scripts/stage10_report.py --json code/event_ssm/tests/fixtures/stage10_fixture.json --out /tmp/stage10_fixture_out && ls /tmp/stage10_fixture_out`
Then view both PNGs (Read tool). Check: labels legible, no clipped text, stacked segments annotated, palette hues correct.

- [ ] **Step 7: Commit**

```bash
git add code/event_ssm/scripts/stage10_report.py code/event_ssm/tests/test_bench_report.py code/event_ssm/tests/fixtures/stage10_fixture.json
git commit -m "feat(stage10): reporter — efficiency table, Pareto, component bars"
```

---

### Task 8: Full suite + handoff checklist

**Files:**
- Modify: none (verification + docs only)
- Create: `results/stage10/RUN_CHECKLIST.md`

**Interfaces:**
- Consumes: everything above.
- Produces: green CPU suite + the user-facing run instructions.

- [ ] **Step 1: Run the complete CPU test suite**

Run: `pytest code/event_ssm/tests/ -v`
Expected: all prior Stage-3..9 tests + the new Stage-10 tests PASS; `gpu`-marked tests deselected. Zero failures.

- [ ] **Step 2: Write the run checklist**

```markdown
# Stage-10 benchmark — run checklist (user)

Preconditions: NO Stage-9 render or eval running; GPU idle (the launcher enforces this).

1. Smoke (<2 min):   bash code/event_ssm/scripts/stage10_run_local.sh --smoke
   - PASS = "[stage10] results validate clean" + results/stage10/bench_results_smoke.json
   - Also run the GPU-marked tests now (~1 min): pytest code/event_ssm/tests -m gpu -v
2. Real run (~20-30 min):   bash code/event_ssm/scripts/stage10_run_local.sh
3. Report:   python code/event_ssm/scripts/stage10_report.py
   - outputs: results/stage10/efficiency_table.md + .csv, stage10_pareto.png/pdf,
     stage10_latency_breakdown.png/pdf
Sanity gates: params ≈ 19.2M (ours) / ~18M (baseline); full-pipeline Hz plausible vs the ~3.3 it/s
seen in batched evals; energy load_w between 80 and 300 W.
```

- [ ] **Step 3: Commit**

```bash
git add results/stage10/RUN_CHECKLIST.md
git commit -m "feat(stage10): run checklist for the user-run smoke + benchmark"
```

---

## Self-Review (performed at planning time)

1. **Spec coverage:** §2 layout → File Structure; §3 construction/wrapper → Task 3; §4 clip → Task 4; §5.1 params → Task 3 (`param_breakdown`) + Task 5; §5.2 FLOPs → Tasks 2+5; §5.3 latency → Tasks 1+5; §5.4 VRAM+state → Tasks 1+3+5; §5.5 energy → Tasks 1+5; §6 orchestrator/launcher/reporter → Tasks 5/6/7; §7 tests/acceptance → every task + Task 8; §8 fallbacks → fvcore try/except (Task 5), PowerSampler injectable/fallback (Task 1), strict ckpt load (Task 3), percentiles (Task 1). No gaps found.
2. **Placeholder scan:** all steps carry complete code/commands; the only "adapt" clauses are the two sanctioned discovery steps (Task 3 Step 1, Task 4 Step 1) whose expected outputs and adaptation rules are stated.
3. **Type consistency:** `time_fn` dict keys used identically in Tasks 1/5/7 (`p50_ms` etc.); `components` keys `backbone/neck_head/postprocess` consistent between Task 3, Task 5 schema, and Task 7 reporter; `temporal_hparams` field names (`kind/tokens/d_model/...`) consistent between Tasks 3 and 5; `validate_results` requirements match the orchestrator output and the fixture.
```
