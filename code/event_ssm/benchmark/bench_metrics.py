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
