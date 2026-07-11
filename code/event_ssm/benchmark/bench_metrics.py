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


def profiler_network_flops(step_fn, *, device) -> dict:
    """Runtime FLOP count of one step_fn() call via torch.profiler (no jit tracing --
    works where fvcore cannot: Triton/complex custom kernels simply contribute 0 and
    are covered by the analytic add-on). Returns {"counted_gflops": float, "source": "torch.profiler"}."""
    from torch.profiler import ProfilerActivity, profile
    activities = [ProfilerActivity.CPU]
    if device is not None and device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    step_fn()                                              # untraced warmup (first-call noise excluded)
    _sync(device)
    with profile(activities=activities, with_flops=True) as prof:
        step_fn()
        _sync(device)
    # Unit convention, settled empirically in test_profiler_counts_toy_conv: torch.profiler's
    # `flops` field for e.g. aten::conv2d is already true FLOPs (2x MACs) -- a 55,296-MAC toy
    # conv reports flops=110592, exactly matching fvcore_network_flops's *2-MACs convention. So,
    # unlike fvcore (which reports MACs and is doubled above), no further scaling is applied here.
    # The flops annotation lives on the CPU-side aten:: event even when CUDA activity is also
    # captured (CUDA kernel events carry no flops field of their own), so summing across all
    # profiled events never double-counts a GPU op.
    total = sum(e.flops for e in prof.events() if getattr(e, "flops", None))
    return {"counted_gflops": total / 1e9, "source": "torch.profiler"}
