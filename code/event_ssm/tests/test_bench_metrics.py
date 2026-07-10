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
