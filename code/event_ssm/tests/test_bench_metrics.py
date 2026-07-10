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
